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
(``<store>/locks/<ACC>.lock``, see ``metaquest.store.locks``), same as a store
download: two adopts of different accessions never touch each other's staging
folder, and two adopts of the same accession serialise on this lock rather
than trampling one another's ``<ACC>_adopt`` copy.

Adoption only ever claims the project's own ``fastq/<ACC>`` folders. A dataset
whose files made it into ``<store>/sra/<ACC>`` but whose sidecar was never
written is finished in place on the next call, without re-touching the files,
but only when this project also has that accession: a sidecar-less folder for
any other accession is another project's interrupted or still running work,
and is reported rather than blessed with a sidecar and linked here.
"""

import gzip
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from metaquest.data.sra import compress_fastq, count_fastq_reads, fastq_files, fastq_stem, is_transient_folder
from metaquest.store.catalog import catalog_write
from metaquest.store.layout import StorePaths, sidecar_path, sra_dir
from metaquest.store.link import link_dataset
from metaquest.store.locks import dataset_lock, lock_holder, lock_is_held
from metaquest.store.sidecar import (
    Sidecar,
    build_sidecar,
    md5_file,
    ncbi_from_metadata_xml,
    read_sidecar,
    write_sidecar,
)

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
    # Sidecar-less store folders for accessions this project does not have: another project's
    # interrupted (or still running) work, reported and never touched.
    foreign: List[str] = field(default_factory=list)
    # Accessions whose store lock is held right now, so another run is publishing them.
    in_progress: List[str] = field(default_factory=list)
    # Accessions left alone because the store's filesystem has no room to stage them.
    refused: List[str] = field(default_factory=list)


def _notify(on_progress: Optional[Callable[[str, str], None]], accession: str, event: str) -> None:
    """Call ``on_progress`` in isolation so a failing callback never breaks adoption itself."""
    if on_progress is None:
        return
    try:
        on_progress(accession, event)
    except Exception as e:  # pragma: no cover - defensive, mirrors metaquest.data.sra._notify_result
        logger.warning("Progress callback failed for %s: %s", accession, e)


def _decompressed_md5(path: Union[str, Path]) -> str:
    """MD5 of ``path``'s decompressed content: gunzips on the fly for a ``.gz`` path, else reads
    plain bytes, so a plain file and a gzipped file holding the same reads compare equal."""
    digest = hashlib.md5()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_matches(path: Path, store_path: Path, record: Dict[str, Any]) -> bool:
    """True when ``path`` and the store copy hold the same reads, whatever their raw bytes.

    Decompresses both sides, so it costs a full pass over each file; the callers use it only
    when the cheap byte comparison cannot settle the question.
    """
    if not store_path.is_file():
        return False
    if _decompressed_md5(path) != _decompressed_md5(store_path):
        return False
    return count_fastq_reads(path) == record.get("reads")


def _files_match(project_dir: Path, store_dir: Path, sidecar: Sidecar) -> bool:
    """True when ``project_dir`` holds the same reads the sidecar's store copy records.

    Files are paired by ``fastq_stem`` (e.g. ``SRR1_1``, not ``SRR1_1.fastq.gz``), so a plain
    project copy matches a gzipped store copy of the same reads and vice versa. Compares the
    sidecar's recorded bytes and md5 first, which settles an identical copy without
    decompressing anything.

    Raw bytes are not decisive for two gzip files: the gzip header carries the compression
    time and the original file name, and a different compressor or level changes the body,
    so the same reads compressed twice differ byte for byte. A gzip pair whose recorded md5
    does not match therefore falls back to the decompressed comparison, as a pair with
    different compression on the two sides already does. Only that fallback can tell a
    duplicate from a genuine conflict here.
    """
    project_files = {fastq_stem(p): p for p in fastq_files(project_dir)}
    sidecar_files = {fastq_stem(Path(str(entry.get("name") or ""))): entry for entry in sidecar.files}
    if set(project_files) != set(sidecar_files):
        return False
    for stem, path in project_files.items():
        record = sidecar_files[stem]
        record_name = str(record.get("name") or "")
        compressed = str(path).endswith(".gz")
        if compressed == record_name.endswith(".gz"):
            if path.stat().st_size == record.get("bytes") and md5_file(path) == record.get("md5"):
                continue
            if not compressed:
                # Two plain files: the bytes are the content, so a mismatch is a real one.
                return False
        if not _content_matches(path, store_dir / record_name, record):
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


def _newest_file_time(acc_dir: Path) -> Optional[str]:
    """When the newest FASTQ file in ``acc_dir`` was last written, as an ISO timestamp.

    An adopted dataset was downloaded at some earlier point that nothing recorded, so its
    files' own age is the best available answer; ``store_gc --older-than`` then counts from
    when the reads were obtained rather than from when they were adopted.
    """
    times = []
    for file_path in fastq_files(acc_dir):
        try:
            times.append(file_path.stat().st_mtime)
        except OSError:
            continue
    if not times:
        return None
    return datetime.fromtimestamp(max(times), tz=timezone.utc).isoformat()


def _finish_sidecar(
    accession: str,
    store_dir: Path,
    sc_path: Path,
    paths: StorePaths,
    metadata_folders: Sequence[Union[str, Path]],
) -> Sidecar:
    """Write the sidecar and catalogue entry for a dataset whose files already sit in the store.

    Shared by a fresh adoption (files just staged and compressed into ``store_dir``) and by
    finishing an interrupted one (files already there from an earlier, incomplete run). The
    sidecar records ``tool="adopted"`` and the files' own age rather than this moment: nothing
    here downloaded them, and no tool version is known.
    """
    ncbi = _ncbi_from_folders(accession, metadata_folders)
    compression = _detect_compression(store_dir)
    sidecar = build_sidecar(
        accession,
        store_dir,
        ncbi,
        "",
        compression,
        tool="adopted",
        downloaded=_newest_file_time(store_dir),
    )
    write_sidecar(sc_path, sidecar)
    with catalog_write(paths) as catalog:
        catalog.upsert_dataset(sidecar)
    return sidecar


def _folder_bytes(folder: Path) -> int:
    """Total bytes of every file under ``folder``, skipping anything that cannot be stat'ed."""
    total = 0
    for sub in folder.rglob("*"):
        try:
            if sub.is_file():
                total += sub.stat().st_size
        except OSError:
            continue
    return total


def _has_room_for(accession: str, entry: Path, paths: StorePaths) -> bool:
    """True when the store's filesystem has room to stage ``entry``, with a copy to spare.

    Adoption peaks at three copies of one accession (the project's, the staging copy and,
    briefly, the store's), so it needs at least twice the folder's size free before it starts.
    A filesystem that cannot be measured is assumed to have room: refusing on an unreadable
    ``disk_usage`` would be worse than trying.
    """
    needed = _folder_bytes(entry) * 2
    try:
        free = shutil.disk_usage(paths.root).free
    except OSError as e:
        logger.warning("Could not check free space on %s: %s", paths.root, e)
        return True
    if free >= needed:
        return True
    logger.warning(
        "%s: not adopting it, the store's filesystem has %d bytes free and staging needs about %d",
        accession,
        free,
        needed,
    )
    return False


def _stage_into_store(
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
    removed here, right before staging starts again, and reported in ``report.resumed``.

    The caller holds this accession's lock (``<store>/locks/<ACC>.lock``, see ``_adopt_one``).
    """
    store_dir = sra_dir(paths, accession)
    staged = paths.tmp / f"{accession}{_STAGING_SUFFIX}"
    paths.tmp.mkdir(parents=True, exist_ok=True)

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


def _scan_foreign_incomplete(paths: StorePaths, real_dirs: Dict[str, Path]) -> List[str]:
    """Sidecar-less store folders for accessions this project is not adopting.

    Such a folder is another project's work: an interrupted run of theirs, or one still in
    flight. Adoption reports it so the store's owner can see it, and never blesses it with a
    sidecar or links it here. Only the project's own folders (``real_dirs``) are ever adopted,
    so a dataset another project is mid-download on cannot be claimed by this run.
    """
    foreign: List[str] = []
    if not paths.sra.is_dir():
        return foreign
    for store_dir in sorted(paths.sra.iterdir()):
        if not store_dir.is_dir():
            continue
        accession = store_dir.name
        if accession in real_dirs or sidecar_path(paths, accession).is_file():
            continue
        foreign.append(accession)
        logger.warning("foreign: %s, no sidecar; left untouched", accession)
    return foreign


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


def _apply_move_or_copy(
    accession: str,
    entry: Path,
    paths: StorePaths,
    project_dir: Path,
    move: bool,
    on_progress: Optional[Callable[[str, str], None]],
    report: AdoptReport,
) -> None:
    """Finish one adopted accession: ``--move`` links it, ``--copy`` leaves the project alone.

    ``--copy`` leaves the project's folder exactly as it was (a second, independent copy) and
    does not link it, since a project that asked to keep its own copy should not have it
    silently swapped for a link on the same run.
    """
    if not move:
        report.copied.append(accession)
        _notify(on_progress, accession, "copied")
        return

    if entry.exists():
        shutil.rmtree(entry)
    link_dataset(project_dir, accession, paths)
    report.adopted.append(accession)
    _notify(on_progress, accession, "adopted")


def _adopt_one(
    accession: str,
    entry: Path,
    paths: StorePaths,
    project_dir: Path,
    move: bool,
    compress: bool,
    metadata_folders: Sequence[Union[str, Path]],
    on_progress: Optional[Callable[[str, str], None]],
    report: AdoptReport,
    lock_wait: float,
) -> None:
    """Get one of this project's accessions into the store, then apply ``--move``/``--copy``.

    Everything that writes to the store happens under this accession's own lock (see
    ``metaquest.store.locks.dataset_lock``), and the store's state is read again inside it:
    a download that finished while this run waited leaves a sidecar behind, and this run must
    then compare copies rather than move its own on top of the published one.

    A sidecar-less store folder whose lock is held right now belongs to a run publishing it at
    this moment; that accession is reported ``in progress elsewhere`` and left alone.
    """
    store_dir = sra_dir(paths, accession)
    sc_path = sidecar_path(paths, accession)

    if store_dir.is_dir() and not sc_path.is_file() and lock_is_held(paths, accession):
        logger.warning(
            "%s: in progress elsewhere (%s); leaving both copies alone", accession, lock_holder(paths, accession)
        )
        report.in_progress.append(accession)
        return

    if not store_dir.is_dir() and not _has_room_for(accession, entry, paths):
        report.refused.append(accession)
        return

    paths.locks.mkdir(parents=True, exist_ok=True)
    with dataset_lock(paths, accession, wait_seconds=lock_wait):
        published_elsewhere = sc_path.is_file()
        if not published_elsewhere:
            if store_dir.is_dir():
                _finish_sidecar(accession, store_dir, sc_path, paths, metadata_folders)
            else:
                _stage_into_store(entry, accession, paths, compress, metadata_folders, report)

    if published_elsewhere:
        # Someone published this accession while we waited for the lock; the project's copy is
        # now a second copy to compare, not something to move on top of theirs.
        _dedup_or_conflict(accession, entry, store_dir, sc_path, paths, project_dir, False, on_progress, report)
        return

    _apply_move_or_copy(accession, entry, paths, project_dir, move, on_progress, report)


def adopt(
    project_fastq: Union[str, Path],
    paths: StorePaths,
    move: bool,
    dry_run: bool,
    on_progress: Optional[Callable[[str, str], None]] = None,
    compress: bool = True,
    metadata_folders: Sequence[Union[str, Path]] = (),
    lock_wait: float = 0.0,
) -> AdoptReport:
    """Fold every real accession folder in ``project_fastq`` into the shared store.

    For each real (non-symlink, non-transient) directory ``project_fastq/<ACC>``: if the store
    already has ``<ACC>`` with a sidecar, an identical set of files is deduplicated (the project
    copy is dropped and replaced with a link) and a differing set is left as a conflict (both
    copies kept); otherwise the folder is staged into the store, compressed, sidecar'd and
    catalogued, then either linked back (``--move``, removing the project's folder) or left alone
    (``--copy``, the project keeps its own folder, unlinked). A symlink already in
    ``project_fastq`` is counted in ``skipped`` and left untouched.

    Only the project's own folders are ever adopted. A sidecar-less ``<store>/sra/<ACC>`` for an
    accession this project also has is finished in place (an earlier run of this project died
    after the files reached the store but before the sidecar was written), under that accession's
    lock; a sidecar-less folder for any other accession belongs to another project, and is listed
    in ``foreign`` without being touched or linked. An accession whose lock is held right now is
    listed in ``in_progress`` and left alone, and one the store's filesystem has no room to stage
    is listed in ``refused``. Any stale ``<store>/tmp/<ACC>_adopt`` copy from a run interrupted
    mid-staging is removed (under that accession's lock, right before re-staging) and the
    accession reported in ``resumed``. Peak disk use for one accession adopted this way is up to
    three copies at once: see the module docstring.

    ``dry_run`` performs no filesystem changes and no database changes (including no sweep of a
    stale staging copy); every accession that would otherwise be dedup'd, adopted or restarted is
    instead listed in ``planned``. Conflicts are still detected and reported in ``conflicts`` even
    during a dry run, since detecting one never writes anything.
    """
    report = AdoptReport()
    project_dir = Path(project_fastq)

    real_dirs = _scan_project_dir(project_dir, report)
    report.foreign.extend(_scan_foreign_incomplete(paths, real_dirs))

    for accession in sorted(real_dirs):
        entry = real_dirs[accession]
        store_dir = sra_dir(paths, accession)
        sc_path = sidecar_path(paths, accession)

        if store_dir.is_dir() and sc_path.is_file():
            _dedup_or_conflict(accession, entry, store_dir, sc_path, paths, project_dir, dry_run, on_progress, report)
            continue

        # Either a fresh accession (no store folder yet) or one this project left incomplete
        # in the store (folder there, no sidecar).
        if dry_run:
            report.planned.append(accession)
            continue

        _adopt_one(
            accession,
            entry,
            paths,
            project_dir,
            move,
            compress,
            metadata_folders,
            on_progress,
            report,
            lock_wait,
        )

    return report
