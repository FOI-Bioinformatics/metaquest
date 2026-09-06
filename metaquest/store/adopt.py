"""
Adopting a project's own downloaded FASTQ folders into the shared store.

A project that downloaded reads before it adopted the shared store (or ran a
download with the store unconfigured) ends up with real, project-owned
``fastq/<ACC>`` folders instead of links. ``adopt`` folds each of those folders
into the store: staging a copy, compressing any plain FASTQ files, writing the
dataset's sidecar and catalogue entry, and then either replacing the project's
folder with a link to the store's copy (``--move``) or leaving the project's
folder untouched and unlinked, with the store now also holding a copy
(``--copy``).

Staging always copies (never moves) the project's original folder into
``<store>/tmp/<ACC>_adopt``, regardless of ``--move``/``--copy``: the project's
own folder is never touched until the store's copy and its sidecar both exist,
so a crash mid-compression (or anywhere before that point) leaves the project's
data exactly as it was, with nothing to recover beyond removing the stale
staging copy. ``--move`` only ever removes the project's folder, and only
after that point. Peak disk use for one accession is therefore up to three
copies at once: the project's original folder, its staging copy, and (briefly,
during the final move) the store's copy.

Staging one accession runs under that accession's own lock
(``<store>/locks/<ACC>.lock``), same as a store download: two adopts of
different accessions never touch each other's staging folder, and two adopts
of the same accession serialise on this lock rather than trampling one
another's ``<ACC>_adopt`` copy.

A dataset whose files already made it into ``<store>/sra/<ACC>`` but whose
sidecar was never written (the process died between the final move and
writing the sidecar) is finished in place on the next call, without
re-touching the files.
"""

import gzip
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from metaquest.data.sra import compress_fastq, count_fastq_reads, fastq_files, fastq_stem, is_transient_folder
from metaquest.store.catalog import catalog_write
from metaquest.store.layout import StorePaths, sidecar_path, sra_dir
from metaquest.store.link import link_dataset
from metaquest.store.sidecar import Sidecar, build_sidecar, ncbi_from_metadata_xml, read_sidecar, write_sidecar

logger = logging.getLogger(__name__)

# Threads handed to compress_fastq while adopting; adoption processes one accession's files at a
# time rather than many downloads in parallel, so a modest, fixed thread count is enough.
_ADOPT_COMPRESS_THREADS = 4

# Suffix on a staging folder under paths.tmp, e.g. "SRR1_adopt".
_STAGING_SUFFIX = "_adopt"


@dataclass
class AdoptReport:
    """Outcome of one ``adopt`` call, one accession name per relevant list."""

    adopted: List[str] = field(default_factory=list)
    # --copy: the store now holds a copy too, but the project's folder is untouched and unlinked.
    copied: List[str] = field(default_factory=list)
    deduplicated: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    # Entries in project_fastq that were already store links, so nothing to adopt.
    skipped: List[str] = field(default_factory=list)
    # Populated instead of acting, when dry_run is True.
    planned: List[str] = field(default_factory=list)
    # A stale <ACC>_adopt staging folder from an earlier, interrupted run of this same
    # accession, removed (under this accession's lock) right before staging; the project's own
    # folder was untouched by that interruption, so the accession is simply re-adopted here.
    resumed: List[str] = field(default_factory=list)


def _notify(on_progress: Optional[Callable[[str, str], None]], accession: str, event: str) -> None:
    """Call ``on_progress`` in isolation so a failing callback never breaks adoption itself."""
    if on_progress is None:
        return
    try:
        on_progress(accession, event)
    except Exception as e:  # pragma: no cover - defensive, mirrors metaquest.data.sra._notify_result
        logger.warning("Progress callback failed for %s: %s", accession, e)


def _md5_file(path: Union[str, Path]) -> str:
    """MD5 hex digest of ``path``, read in 1 MiB chunks so a large file is never loaded whole."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decompressed_md5(path: Union[str, Path]) -> str:
    """MD5 of ``path``'s decompressed content: gunzips on the fly for a ``.gz`` path, else reads
    plain bytes, so a plain file and a gzipped file holding the same reads compare equal."""
    digest = hashlib.md5()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_match(project_dir: Path, store_dir: Path, sidecar: Sidecar) -> bool:
    """True when ``project_dir`` holds the same reads the sidecar's store copy records.

    Files are paired by ``fastq_stem`` (e.g. ``SRR1_1``, not ``SRR1_1.fastq.gz``), so a plain
    project copy matches a gzipped store copy of the same reads and vice versa. When both sides
    share the same compression, compares the sidecar's recorded bytes/md5 directly (cheap, no
    decompression); when they differ (a compressed size never equals a plain one), decompresses
    both sides and compares content md5 and read count instead.
    """
    project_files = {fastq_stem(p): p for p in fastq_files(project_dir)}
    sidecar_files = {fastq_stem(Path(str(entry.get("name") or ""))): entry for entry in sidecar.files}
    if set(project_files) != set(sidecar_files):
        return False
    for stem, path in project_files.items():
        record = sidecar_files[stem]
        record_name = str(record.get("name") or "")
        if str(path).endswith(".gz") == record_name.endswith(".gz"):
            if path.stat().st_size != record.get("bytes"):
                return False
            if _md5_file(path) != record.get("md5"):
                return False
            continue
        # Compression differs: sizes are never comparable, so decompress both sides and compare
        # content instead.
        store_path = store_dir / record_name
        if not store_path.is_file():
            return False
        if _decompressed_md5(path) != _decompressed_md5(store_path):
            return False
        if count_fastq_reads(path) != record.get("reads"):
            return False
    return True


def _detect_compression(acc_dir: Path) -> str:
    """``"gzip"`` when every FASTQ file in ``acc_dir`` is gzip-compressed, else ``"none"``."""
    files = fastq_files(acc_dir)
    if files and all(str(p).endswith(".gz") for p in files):
        return "gzip"
    return "none"


def _ncbi_from_folders(accession: str, metadata_folders: Sequence[Union[str, Path]]) -> Dict[str, Any]:
    """The ``ncbi`` dict for ``accession`` from the first metadata folder that has its XML."""
    for folder in metadata_folders:
        candidate = Path(folder) / f"{accession}_metadata.xml"
        if candidate.exists():
            return ncbi_from_metadata_xml(candidate) or {}
    return {}


def _finish_sidecar(
    accession: str,
    store_dir: Path,
    sc_path: Path,
    paths: StorePaths,
    metadata_folders: Sequence[Union[str, Path]],
) -> Sidecar:
    """Write the sidecar and catalogue entry for a dataset whose files already sit in the store.

    Shared by a fresh adoption (files just staged and compressed into ``store_dir``) and by
    finishing an interrupted one (files already there from an earlier, incomplete run).
    """
    ncbi = _ncbi_from_folders(accession, metadata_folders)
    compression = _detect_compression(store_dir)
    sidecar = build_sidecar(accession, store_dir, ncbi, "adopted", compression)
    write_sidecar(sc_path, sidecar)
    with catalog_write(paths) as catalog:
        catalog.upsert_dataset(sidecar)
    return sidecar


def _adopt_fresh(
    entry: Path,
    accession: str,
    paths: StorePaths,
    compress: bool,
    metadata_folders: Sequence[Union[str, Path]],
    report: AdoptReport,
) -> None:
    """Copy ``entry``'s files into ``<store>/sra/<accession>``, compressing plain files on the way.

    Always stages through a *copy* into ``paths.tmp`` first, regardless of ``--move``/``--copy``:
    the project's original folder is only ever removed (for ``--move``, by the caller, after this
    returns) once the store's copy and its sidecar both exist. A crash anywhere in this function
    leaves ``entry`` untouched; a stale ``<ACC>_adopt`` staging copy left by such a crash is
    removed here, under this accession's lock, right before staging starts again, and reported in
    ``report.resumed``.

    The whole operation (sweep, copy, compress, move, sidecar) runs under this accession's own
    lock (``<store>/locks/<ACC>.lock``, acquired with ``metaquest.data.registry._acquire_lock``,
    same as a store download): a concurrent adopt of this same accession blocks here until this
    one finishes, rather than racing it for the same staging folder.
    """
    # Imported here rather than at module level, matching the same pairing's other call site in
    # metaquest.data.sra: the store package's own modules import the data layer, so importing
    # metaquest.data.registry back at module level here risks a cycle.
    from metaquest.data.registry import _acquire_lock
    from metaquest.store.layout import lock_path

    store_dir = sra_dir(paths, accession)
    staged = paths.tmp / f"{accession}{_STAGING_SUFFIX}"
    paths.tmp.mkdir(parents=True, exist_ok=True)
    paths.locks.mkdir(parents=True, exist_ok=True)

    lock = lock_path(paths, accession)
    _acquire_lock(lock)
    try:
        if staged.exists():
            shutil.rmtree(staged)
            report.resumed.append(accession)

        shutil.copytree(entry, staged)

        if compress:
            for file_path in fastq_files(staged):
                if not str(file_path).endswith(".gz"):
                    compress_fastq(file_path, _ADOPT_COMPRESS_THREADS)

        store_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(store_dir))

        _finish_sidecar(accession, store_dir, sidecar_path(paths, accession), paths, metadata_folders)
    finally:
        lock.unlink(missing_ok=True)


def _scan_project_dir(project_dir: Path, report: AdoptReport) -> Dict[str, Path]:
    """Real (non-symlink, non-transient) accession directories directly in ``project_dir``.

    A symlink is recorded in ``report.skipped`` (already a store link, nothing to adopt) rather
    than returned.
    """
    real_dirs: Dict[str, Path] = {}
    if not project_dir.is_dir():
        return real_dirs
    for candidate in sorted(project_dir.iterdir()):
        name = candidate.name
        if is_transient_folder(name):
            continue
        if candidate.is_symlink():
            report.skipped.append(name)
            continue
        if candidate.is_dir():
            real_dirs[name] = candidate
    return real_dirs


def _scan_interrupted(paths: StorePaths, project_dir: Path, real_dirs: Dict[str, Path]) -> set:
    """Accessions whose files already sit in the store with no sidecar, and no local folder.

    Ordinarily the project's folder is only removed after the sidecar is written (see
    ``_adopt_fresh``/``_adopt_new_or_restart``), so this covers the rare case where it went
    missing some other way (e.g. removed outside MetaQuest) while the store side was left
    incomplete. Restricted to accessions not already in ``real_dirs`` (those are handled where
    they are found) and not already linked.
    """
    interrupted: set = set()
    if not paths.sra.is_dir():
        return interrupted
    for store_dir in sorted(paths.sra.iterdir()):
        if not store_dir.is_dir():
            continue
        accession = store_dir.name
        if accession in real_dirs or sidecar_path(paths, accession).is_file():
            continue
        if (project_dir / accession).is_symlink():
            continue
        interrupted.add(accession)
    return interrupted


def _dedup_or_conflict(
    accession: str,
    entry: Optional[Path],
    store_dir: Path,
    sc_path: Path,
    paths: StorePaths,
    project_dir: Path,
    dry_run: bool,
    on_progress: Optional[Callable[[str, str], None]],
    report: AdoptReport,
) -> None:
    """Handle an accession whose store folder already carries a sidecar: dedup or conflict.

    ``entry`` is None only for an accession already fully migrated (store has it, sidecar
    exists, nothing local left to reconcile); nothing to do in that case.
    """
    if entry is None:
        return
    existing = read_sidecar(sc_path)
    if existing is not None and _files_match(entry, store_dir, existing):
        if dry_run:
            report.planned.append(accession)
            return
        shutil.rmtree(entry)
        link_dataset(project_dir, accession, paths)
        report.deduplicated.append(accession)
        _notify(on_progress, accession, "deduplicated")
        return
    report.conflicts.append(accession)
    logger.warning(
        "%s: project and store copies differ; leaving both in place (store copy left untouched)",
        accession,
    )
    _notify(on_progress, accession, "conflict")


def _adopt_new_or_restart(
    accession: str,
    entry: Optional[Path],
    store_dir: Path,
    sc_path: Path,
    paths: StorePaths,
    project_dir: Path,
    move: bool,
    compress: bool,
    metadata_folders: Sequence[Union[str, Path]],
    on_progress: Optional[Callable[[str, str], None]],
    report: AdoptReport,
) -> None:
    """Get a fresh accession's files into the store (or finish one interrupted after its files
    already reached the store), then apply ``--move``/``--copy``.

    ``--move`` removes the project's folder and links it to the store's copy; ``--copy`` leaves
    the project's folder exactly as it was (a second, independent copy) and does not link it,
    since a project that asked to keep its own copy should not have it silently swapped for a
    link on the same run.
    """
    if store_dir.is_dir() and not sc_path.is_file():
        _finish_sidecar(accession, store_dir, sc_path, paths, metadata_folders)
    else:
        # A fresh accession always comes from real_dirs (interrupted ones are handled by the
        # branch above), so entry is never None here.
        assert entry is not None
        _adopt_fresh(entry, accession, paths, compress, metadata_folders, report)

    if not move:
        report.copied.append(accession)
        _notify(on_progress, accession, "copied")
        return

    if entry is not None and entry.exists():
        shutil.rmtree(entry)
    link_dataset(project_dir, accession, paths)
    report.adopted.append(accession)
    _notify(on_progress, accession, "adopted")


def adopt(
    project_fastq: Union[str, Path],
    paths: StorePaths,
    move: bool,
    dry_run: bool,
    on_progress: Optional[Callable[[str, str], None]] = None,
    compress: bool = True,
    metadata_folders: Sequence[Union[str, Path]] = (),
) -> AdoptReport:
    """Fold every real accession folder in ``project_fastq`` into the shared store.

    For each real (non-symlink, non-transient) directory ``project_fastq/<ACC>``: if the store
    already has ``<ACC>`` with a sidecar, an identical set of files is deduplicated (the project
    copy is dropped and replaced with a link) and a differing set is left as a conflict (both
    copies kept); otherwise the folder is staged into the store, compressed, sidecar'd and
    catalogued, then either linked back (``--move``, removing the project's folder) or left alone
    (``--copy``, the project keeps its own folder, unlinked). A symlink already in
    ``project_fastq`` is counted in ``skipped`` and left untouched.

    Also finishes any dataset already sitting in ``<store>/sra/<ACC>`` without a sidecar, from an
    earlier run interrupted after the files reached their final location but before the sidecar
    was written, and removes (under that accession's own lock, right before re-staging) any stale
    ``<store>/tmp/<ACC>_adopt`` copy left by a run interrupted earlier than that (the project's
    own folder is untouched in that case, so the accession is simply re-adopted; such accessions
    are reported in ``resumed``). Peak disk use for one accession adopted this way is up to three
    copies at once: see the module docstring.

    ``dry_run`` performs no filesystem changes and no database changes (including no sweep of a
    stale staging copy); every accession that would otherwise be dedup'd, adopted or restarted is
    instead listed in ``planned``. Conflicts are still detected and reported in ``conflicts`` even
    during a dry run, since detecting one never writes anything.
    """
    report = AdoptReport()
    project_dir = Path(project_fastq)

    real_dirs = _scan_project_dir(project_dir, report)
    interrupted = _scan_interrupted(paths, project_dir, real_dirs)

    for accession in sorted(set(real_dirs) | interrupted):
        entry = real_dirs.get(accession)
        store_dir = sra_dir(paths, accession)
        sc_path = sidecar_path(paths, accession)

        if store_dir.is_dir() and sc_path.is_file():
            _dedup_or_conflict(accession, entry, store_dir, sc_path, paths, project_dir, dry_run, on_progress, report)
            continue

        # Either a fresh accession (no store folder yet) or an interrupted one (store folder
        # exists, no sidecar).
        if dry_run:
            report.planned.append(accession)
            continue

        _adopt_new_or_restart(
            accession,
            entry,
            store_dir,
            sc_path,
            paths,
            project_dir,
            move,
            compress,
            metadata_folders,
            on_progress,
            report,
        )

    return report
