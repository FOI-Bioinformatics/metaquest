"""Per-project registry of dataset state, the project journal.

The registry records decisions and provenance for every SRA accession a
project touches: screening results per target genome, the selection criteria,
exclusions with reasons, download outcomes with file sizes and dates, analyses,
and per-genome extraction and assembly results. Existence on disk is never
taken from the registry alone; ``status`` re-checks the filesystem and the
scanners here rebuild or reconcile the journal from what is on disk.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

from metaquest.core.constants import DEFAULT_REGISTRY_MAX_SCREENED, GENOME_FASTA_GLOBS
from metaquest.core.exceptions import DataAccessError
from metaquest.data.read_extraction import summarise_contigs
from metaquest.data.sra import accession_has_fastq, count_fastq_reads, fastq_files, is_transient_folder, verify_download

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "metaquest_registry.json"
# Version 1: version, created, updated, genomes, datasets.
# Version 2 adds project (this project's identity, recorded by `store_init`) and store
# (the shared data store this project is bound to). Both default to {} so a version-1
# file loads unchanged; the next save writes it back as version 2.
SCHEMA_VERSION = 2
STAGES = ("screened", "selected", "excluded", "downloaded", "analysed", "extracted", "assembled")
LOCK_STALE_SECONDS = 30.0
LOCK_WAIT_SECONDS = 30.0
_MATE_SUFFIXES = ("_1", "_2", "_s", "_0")
_ASSEMBLY_SUFFIX = "_assembly"
_CONTIGS_NAME = "final.contigs.fa"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class ProjectPaths:
    """Folders of one project, relative to the project root unless absolute."""

    fastq: Path = Path("fastq")
    metadata: Path = Path("metadata")
    genomes: Path = Path("genomes")
    targeted: Path = Path("targeted")
    matches: Path = Path("matches")


@dataclass
class Registry:
    version: int = SCHEMA_VERSION
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    genomes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # This project's identity ({"id", "name", "path", "created"}), recorded by `store_init`.
    project: Dict[str, Any] = field(default_factory=dict)
    # The shared data store this project is bound to ({"root", "mode", "linked"}).
    store: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None


def project_root(registry: Registry) -> Path:
    """The project root a registry's paths are recorded relative to: its file's parent folder.

    A registry not yet bound to a file (before its first save, e.g. mid-``bootstrap_from_disk``)
    falls back to the current working directory, so a path recorded at that point resolves the
    same way once the registry is later bound and saved from the same directory.
    """
    if registry.path is not None:
        return registry.path.parent.resolve()
    return Path.cwd().resolve()


def _project_relative(path: Union[str, Path], root: Path) -> str:
    """The form to record for ``path``: relative to ``root`` when inside it, else absolute.

    Keeps a project's registry movable: renaming or relocating the project directory does not
    break paths recorded under it. A path outside the project root (for example a genome FASTA
    shared from elsewhere on disk) has no project-relative form, so it is kept absolute. A path
    under a symlinked folder resolves to the real (target) location, so a project folder linked
    into an external store records that store's absolute path here; the store-relative name used
    to look the dataset up in the store is recorded separately by the store-linking feature.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def resolve_project_path(registry: Registry, value: Union[str, Path]) -> Path:
    """Resolve one path recorded in the registry against its project root.

    An absolute recorded value is returned unchanged, so entries written by a MetaQuest
    version before this change (always absolute) keep resolving correctly. A relative value
    is joined to ``project_root``. For a registry written by an older version, whose relative
    paths were relative to the working directory of that run, this still resolves correctly in
    the common case, since that working directory is normally where the registry file itself
    lived, which is exactly the project root computed here. Callers with a value that may be
    ``None`` (an optional recorded field) must check that themselves before calling.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root(registry) / path


@dataclass
class ReconcileReport:
    recorded_missing: List[str] = field(default_factory=list)
    untracked_fastq: List[str] = field(default_factory=list)
    untracked_extractions: List[Tuple[str, str]] = field(default_factory=list)
    empty_assembly_dirs: List[Tuple[str, str]] = field(default_factory=list)
    # Accessions whose project folder is a symlink with nothing at the other end, i.e. a
    # dataset the project reads from a shared store that is unmounted or has lost the copy.
    dangling_links: List[str] = field(default_factory=list)


# ----------------------------------------------------------------- persistence


def registry_path(explicit: Optional[Union[str, Path]] = None, start: Union[str, Path] = ".") -> Path:
    """The registry file to use: an explicit path, else the nearest one walking up from ``start``."""
    if explicit:
        return Path(explicit)
    current = Path(start).resolve()
    for folder in (current, *current.parents):
        candidate = folder / REGISTRY_FILENAME
        if candidate.exists():
            if folder != current:
                logger.info("Using the project registry found above the working directory: %s", candidate)
            return candidate
    return current / REGISTRY_FILENAME


def load_registry(path: Optional[Union[str, Path]] = None) -> Registry:
    """Load the registry, or an empty one bound to the path when the file does not exist."""
    target = registry_path(path)
    if not target.exists():
        return Registry(path=target)
    try:
        text = target.read_text()
    except OSError as e:
        raise DataAccessError(f"Cannot read registry {target}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DataAccessError(f"Registry is not valid JSON: {target} ({e})") from e
    return Registry(
        version=int(data.get("version", SCHEMA_VERSION)),
        created=str(data.get("created", _now())),
        updated=str(data.get("updated", _now())),
        genomes=dict(data.get("genomes", {})),
        datasets=dict(data.get("datasets", {})),
        project=dict(data.get("project", {})),
        store=dict(data.get("store", {})),
        path=target,
    )


def _acquire_lock(lock: Path) -> None:
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            # Check the wait deadline before staleness: when the two windows are equal, the
            # lock's age and the elapsed wait cross their thresholds on the same iteration,
            # and giving up (a lock truly held by another process) must win over reclaiming it.
            if time.monotonic() > deadline:
                raise DataAccessError(f"Registry is locked by another process: {lock}")
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > LOCK_STALE_SECONDS:
                logger.warning("Removing stale registry lock %s (%.0f s old)", lock, age)
                lock.unlink(missing_ok=True)
                continue
            time.sleep(0.05)


def _write_registry(registry: Registry, target: Path) -> Path:
    """Write the registry to ``target`` atomically (temp file plus rename); the caller holds the lock."""
    target.parent.mkdir(parents=True, exist_ok=True)
    registry.updated = _now()
    registry.path = target
    # A registry loaded from an older schema is upgraded to the current one on save; there
    # is no separate migration step, since every field new schema versions add already
    # defaults to {} when missing.
    registry.version = SCHEMA_VERSION
    payload = {
        "version": registry.version,
        "created": registry.created,
        "updated": registry.updated,
        "genomes": registry.genomes,
        "datasets": registry.datasets,
        "project": registry.project,
        "store": registry.store,
    }
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, target)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise DataAccessError(f"Cannot write registry {target}: {e}") from e
    return target


def save_registry(registry: Registry, path: Optional[Union[str, Path]] = None) -> Path:
    """Write an already loaded registry atomically under a lock file."""
    target = Path(path) if path else (registry.path or registry_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".lock")
    _acquire_lock(lock)
    try:
        return _write_registry(registry, target)
    finally:
        lock.unlink(missing_ok=True)


@contextmanager
def registry_transaction(path: Optional[Union[str, Path]] = None) -> Iterator[Registry]:
    """Load, mutate and save the registry under one lock, so a concurrent edit is never reverted.

    Not re-entrant: nesting a second call to this function (or to `save_registry`) for the
    same registry file inside this block's body will deadlock against the lock this call
    already holds, since the lock file is only released when this context manager exits.

    Use this instead of holding one loaded registry across a long run: the file is
    read inside the lock and written back at the end of the block. If the block
    raises, the lock is released and nothing is written.
    """
    target = registry_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".lock")
    _acquire_lock(lock)
    try:
        registry = load_registry(target)
        yield registry
        _write_registry(registry, target)
    finally:
        lock.unlink(missing_ok=True)


# --------------------------------------------------------------------- writers


def upsert_dataset(registry: Registry, accession: str) -> Dict[str, Any]:
    return registry.datasets.setdefault(accession, {})


def record_screening(
    registry: Registry,
    accession: str,
    genome_id: str,
    containment: float,
    cani: Optional[float],
    source: str,
    query_threshold: float,
    csv_path: Optional[Union[str, Path]],
) -> None:
    screening = upsert_dataset(registry, accession).setdefault("screening", {})
    screening["date"] = _now()
    screening.pop("inferred", None)
    screening.setdefault("genomes", {})[genome_id] = {
        "containment": round(float(containment), 4),
        "cani": round(float(cani), 4) if cani is not None else None,
        "csv": str(csv_path) if csv_path else None,
        "source": source,
        "query_threshold": query_threshold,
    }
    registry.genomes.setdefault(genome_id, {})


def record_genome(registry: Registry, genome_id: str, fasta: Union[str, Path], manifest: Union[str, Path]) -> None:
    """Record where a target genome's FASTA lives, and the manifest it came from."""
    root = project_root(registry)
    registry.genomes[genome_id] = {
        "fasta": _project_relative(fasta, root),
        "manifest": _project_relative(manifest, root),
        "date": _now(),
    }


def record_selection(
    registry: Registry, accessions: Sequence[str], criteria: Dict[str, Any], output: Union[str, Path]
) -> None:
    """Mark ``accessions`` selected with ``criteria``; anything selected earlier but absent now becomes unselected."""
    chosen = set(accessions)
    for accession, record in registry.datasets.items():
        selection = record.get("selection")
        if selection and selection.get("selected") and accession not in chosen:
            selection["selected"] = False
            selection["date"] = _now()
    for accession in accessions:
        upsert_dataset(registry, accession)["selection"] = {
            "selected": True,
            "date": _now(),
            "criteria": dict(criteria),
            "output": str(output),
        }


def cap_screening(registry: Registry, genome_id: str, max_screened: int = DEFAULT_REGISTRY_MAX_SCREENED) -> int:
    """Keep only the ``max_screened`` highest containments for one genome; return how many were dropped.

    A broad search can match tens of thousands of metagenomes. The CSVs remain the raw
    record; the registry keeps the best matches so it stays small enough to rewrite after
    every completed accession.
    """
    entries = [
        (acc, float(record["screening"]["genomes"][genome_id].get("containment") or 0.0))
        for acc, record in registry.datasets.items()
        if genome_id in record.get("screening", {}).get("genomes", {})
    ]
    if len(entries) <= max_screened:
        return 0
    for accession, _ in sorted(entries, key=lambda item: item[1], reverse=True)[max_screened:]:
        record = registry.datasets[accession]
        genomes = record["screening"]["genomes"]
        del genomes[genome_id]
        if not genomes:
            del record["screening"]
        if not record:
            del registry.datasets[accession]
    dropped = len(entries) - max_screened
    logger.warning(
        "Kept the %d highest containments for %s in the registry and dropped %d more "
        "(the limit is --registry-max-screened); the match CSV keeps them all",
        max_screened,
        genome_id,
        dropped,
    )
    return dropped


def record_screening_from_table(
    registry: Registry,
    table_path: Union[str, Path],
    matches_folder: Union[str, Path],
    max_screened: int = DEFAULT_REGISTRY_MAX_SCREENED,
) -> int:
    """Record a screening entry for every positive containment in a parsed containment table.

    Reads the table written by ``parse_containment_data`` and, for every column except
    ``max_containment`` and ``max_containment_annotation`` (each a genome), records one
    screening entry per row with a value greater than 0, keeping at most ``max_screened``
    accessions per genome. Cells that do not hold a number are skipped. Returns the number
    of entries recorded. If the table does not exist, logs at debug level and returns 0
    without raising.
    """
    table_path = Path(table_path)
    if not table_path.exists():
        logger.debug("Parsed containment table %s does not exist; nothing to record", table_path)
        return 0

    import pandas as pd

    table = pd.read_csv(table_path, sep="\t", index_col=0)
    genome_columns = [c for c in table.columns if c not in ("max_containment", "max_containment_annotation")]
    recorded = 0
    for accession, row in table.iterrows():
        for column in genome_columns:
            try:
                value = float(row[column])
            except (ValueError, TypeError):
                continue
            if pd.notna(value) and value > 0:
                record_screening(
                    registry,
                    str(accession),
                    column,
                    value,
                    None,
                    "matches",
                    0.0,
                    Path(matches_folder) / f"{column}.csv",
                )
                recorded += 1
    for column in genome_columns:
        cap_screening(registry, column, max_screened)
    return recorded


def record_exclusion(registry: Registry, accession: str, reason: str, source: str = "user") -> None:
    upsert_dataset(registry, accession)["exclusion"] = {
        "excluded": True,
        "reason": reason,
        "source": source,
        "date": _now(),
    }


def clear_exclusion(registry: Registry, accession: str) -> None:
    record = registry.datasets.get(accession)
    if record and "exclusion" in record:
        record["exclusion"] = {"excluded": False, "reason": "", "source": "user", "date": _now()}


def _file_entries(paths: Iterable[Path], root: Path) -> List[Dict[str, Any]]:
    entries = []
    for path in paths:
        stat = path.stat()
        entries.append(
            {
                "path": _project_relative(path, root),
                "bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            }
        )
    return entries


def record_download(
    registry: Registry,
    accession: str,
    state: str,
    fastq_dir: Union[str, Path],
    message: str = "",
    attempt: bool = True,
    complete: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    store_name: Optional[str] = None,
) -> None:
    """Record a download outcome; ``state`` is downloaded, failed, missing or skipped.

    ``attempt`` counts this call against ``attempts``; pass ``False`` when recording a
    state without an actual download attempt (e.g. a file found already present on disk).
    ``complete`` is the completeness verdict from ``metaquest.data.sra.verify_download``
    (via ``parse_verdict_message``); when omitted, any verdict already on file is left as is.
    ``source`` says where the reads came from (``"store"`` for a dataset the shared store
    holds and the project only links to) and ``store_name`` is the dataset's name inside
    that store. Both describe this outcome, so a call that names neither clears whatever
    an earlier outcome recorded rather than leaving a stale claim behind.
    """
    download = upsert_dataset(registry, accession).setdefault("download", {"attempts": 0})
    if attempt and state in ("downloaded", "failed"):
        download["attempts"] = int(download.get("attempts", 0)) + 1
    files: List[Dict[str, Any]] = []
    if state == "downloaded":
        files = _file_entries(fastq_files(Path(fastq_dir) / accession), project_root(registry))
    download.update(
        {
            "state": state,
            "date": _now(),
            "files": files,
            "bytes_total": sum(int(f["bytes"]) for f in files),
            "message": message,
        }
    )
    if complete is not None:
        download["complete"] = complete
    if source is None:
        download.pop("source", None)
        download.pop("store_name", None)
    else:
        download["source"] = source
        if store_name is not None:
            download["store_name"] = store_name
    download.pop("inferred", None)


def set_download_verdict(registry: Registry, accession: str, verdict: Dict[str, Any]) -> None:
    """Set only a downloaded accession's completeness verdict, touching nothing else.

    Unlike ``record_download``, this never resets ``date``, re-scans ``files``/``bytes_total``,
    or clears ``message``/``source``/``store_name``; use it when only the completeness verdict
    needs to change, e.g. computing one that a download recorded before verification existed
    never got.
    """
    upsert_dataset(registry, accession).setdefault("download", {"attempts": 0})["complete"] = verdict


def nan_to_none(value: Any) -> Any:
    """Return ``None`` for a pandas NaN/NA value, else ``value`` unchanged."""
    import pandas as pd

    return None if pd.isna(value) else value


def _to_int_or_none(value: Any) -> Optional[int]:
    """Convert a numeric value (int or numeric string) to ``int``; ``None`` otherwise."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def record_metadata(registry: Registry, accession: str, xml_path: Union[str, Path], fields: Dict[str, Any]) -> None:
    record: Dict[str, Any] = {"xml": _project_relative(xml_path, project_root(registry)), "date": _now()}
    for key in (
        "run_size",
        "run_md5",
        "assay_type",
        "organism",
        "collection_date",
        "library_layout",
        "platform",
        "library_strategy",
    ):
        record[key] = fields.get(key)
    for key in ("run_total_spots", "run_total_bases"):
        record[key] = _to_int_or_none(fields.get(key))
    upsert_dataset(registry, accession)["metadata"] = record


def record_analysis(
    registry: Registry, accession: str, analysis: str, output: Union[str, Path], summary: Dict[str, Any]
) -> None:
    upsert_dataset(registry, accession).setdefault("analyses", {})[analysis] = {
        "date": _now(),
        "output": _project_relative(output, project_root(registry)),
        "summary": dict(summary),
    }


def record_extraction(
    registry: Registry,
    accession: str,
    genome_id: str,
    files: Sequence[Path],
    mapped_reads: int,
    unequal_mates: bool,
    params: Dict[str, Any],
) -> None:
    root = project_root(registry)
    extractions = upsert_dataset(registry, accession).setdefault("extractions", {})
    previous = extractions.get(genome_id, {})
    genome_fasta = params.get("genome_fasta")
    extractions[genome_id] = {
        "date": _now(),
        "genome_fasta": _project_relative(genome_fasta, root) if genome_fasta is not None else None,
        "preset": params.get("preset"),
        "threshold": params.get("threshold"),
        "mapped_reads": int(mapped_reads),
        "unequal_mates": bool(unequal_mates),
        "files": [_project_relative(p, root) for p in files],
        "assembly": previous.get("assembly"),
    }
    registry.genomes.setdefault(genome_id, {})


def record_assembly(
    registry: Registry,
    accession: str,
    genome_id: str,
    assembly_dir: Union[str, Path],
    stats: Dict[str, int],
    tool_version: str,
    params: Dict[str, Any],
) -> None:
    extractions = upsert_dataset(registry, accession).setdefault("extractions", {})
    entry = extractions.setdefault(genome_id, {"files": [], "mapped_reads": None})
    entry.pop("inferred", None)
    entry["assembly"] = {
        "date": _now(),
        "dir": _project_relative(assembly_dir, project_root(registry)),
        "contigs": int(stats.get("contigs", 0)),
        "total_bp": int(stats.get("total_bp", 0)),
        "n50": int(stats.get("n50", 0)),
        "largest": int(stats.get("largest", 0)),
        "tool": "megahit",
        "version": tool_version,
        "params": dict(params),
    }


def extraction_record(registry: Registry, accession: str, genome_id: str) -> Optional[Dict[str, Any]]:
    return registry.datasets.get(accession, {}).get("extractions", {}).get(genome_id)


# --------------------------------------------------------------------- queries


def _extraction_stage(record: Dict[str, Any], stage: str, genome_id: Optional[str]) -> bool:
    """Handle the "extracted"/"assembled" stages of ``_in_stage`` (kept separate to bound complexity)."""
    extractions = record.get("extractions", {})
    if genome_id is None:
        chosen = [e for e in extractions.values() if e is not None]
    else:
        chosen = [extractions[genome_id]] if extractions.get(genome_id) is not None else []
    if stage == "extracted":
        return any((e.get("mapped_reads") or 0) > 0 for e in chosen)
    if stage == "assembled":
        return any((e.get("assembly") or {}).get("contigs", 0) > 0 for e in chosen)
    raise DataAccessError(f"Unknown stage '{stage}'. Choose one of: {', '.join(STAGES)}")


def _in_stage(record: Dict[str, Any], stage: str, genome_id: Optional[str]) -> bool:
    if stage == "screened":
        genomes = record.get("screening", {}).get("genomes", {})
        return bool(genomes) if genome_id is None else genome_id in genomes
    if stage == "selected":
        return bool(record.get("selection", {}).get("selected"))
    if stage == "excluded":
        return bool(record.get("exclusion", {}).get("excluded"))
    if stage == "downloaded":
        return record.get("download", {}).get("state") == "downloaded"
    if stage == "analysed":
        return bool(record.get("analyses"))
    return _extraction_stage(record, stage, genome_id)


def query(registry: Registry, stage: str, genome_id: Optional[str] = None) -> List[str]:
    """Accessions in ``stage`` (insertion order), optionally for one target genome."""
    if stage not in STAGES:
        raise DataAccessError(f"Unknown stage '{stage}'. Choose one of: {', '.join(STAGES)}")
    return [acc for acc, record in registry.datasets.items() if _in_stage(record, stage, genome_id)]


def stage_counts(registry: Registry) -> Dict[str, Any]:
    stages = {stage: len(query(registry, stage)) for stage in STAGES}
    genomes: Dict[str, Dict[str, Any]] = {}
    for genome_id in sorted(known_genome_ids(registry)):
        zero = [
            acc
            for acc, record in registry.datasets.items()
            if genome_id in record.get("extractions", {})
            and (record["extractions"][genome_id].get("mapped_reads") or 0) == 0
        ]
        genomes[genome_id] = {
            "extracted": len(query(registry, "extracted", genome_id)),
            "assembled": len(query(registry, "assembled", genome_id)),
            "zero_mapped": zero,
        }
    return {"stages": stages, "genomes": genomes}


def known_genome_ids(registry: Registry) -> Set[str]:
    ids: Set[str] = set(registry.genomes)
    for record in registry.datasets.values():
        ids.update(record.get("screening", {}).get("genomes", {}))
        ids.update(record.get("extractions", {}))
    return ids


# -------------------------------------------------------------------- scanners


def split_extract_filename(name: str, genome_ids: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Split ``<genome><mate>.fastq(.gz)`` into (genome_id, mate suffix); known genome ids win, longest first."""
    stem = name
    for suffix in (".fastq.gz", ".fastq", ".fq.gz", ".fq"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        return None
    for genome_id in sorted(genome_ids, key=len, reverse=True):
        if stem == genome_id:
            return genome_id, ""
        for mate in _MATE_SUFFIXES:
            if stem == genome_id + mate:
                return genome_id, mate
    for mate in _MATE_SUFFIXES:
        if stem.endswith(mate):
            return stem[: -len(mate)], mate
    return stem, ""


def scan_downloads(fastq_folder: Path) -> Dict[str, Tuple[int, int]]:
    """Accession -> (number of FASTQ files, total bytes) for every per-accession folder with reads.

    Transient folders (``<acc>_temp``, ``.sra-cache``) are skipped even when they hold a
    partial FASTQ file, so an in-progress or failed download is never mistaken for a
    downloaded accession.
    """
    found: Dict[str, Tuple[int, int]] = {}
    if not fastq_folder.is_dir():
        return found
    for folder in sorted(p for p in fastq_folder.iterdir() if p.is_dir()):
        if is_transient_folder(folder.name):
            continue
        if accession_has_fastq(folder):
            # fastq_files, not a raw glob: a zero-byte file or a .gz.tmp.<pid> leftover of an
            # interrupted compression is not a downloaded read file.
            files = fastq_files(folder)
            found[folder.name] = (len(files), sum(p.stat().st_size for p in files))
    return found


def scan_metadata(metadata_folder: Path) -> Set[str]:
    return {p.name[: -len("_metadata.xml")] for p in metadata_folder.glob("*_metadata.xml")}


def _genome_ids_on_disk(paths: ProjectPaths, registry: Optional[Registry]) -> Set[str]:
    ids: Set[str] = set(known_genome_ids(registry)) if registry else set()
    for pattern in GENOME_FASTA_GLOBS:
        for p in paths.genomes.glob(pattern):
            ids.add(p.name[: -len(pattern[1:])] if p.name.endswith(pattern[1:]) else p.stem)
    ids.update(p.stem for p in paths.matches.glob("*.csv"))
    if paths.targeted.is_dir():
        for acc_dir in paths.targeted.iterdir():
            for asm in acc_dir.glob(f"*{_ASSEMBLY_SUFFIX}"):
                ids.add(asm.name[: -len(_ASSEMBLY_SUFFIX)])
    return ids


def scan_extractions(targeted_folder: Path, genome_ids: Sequence[str]) -> Dict[str, Dict[str, List[Path]]]:
    found: Dict[str, Dict[str, List[Path]]] = {}
    if not targeted_folder.is_dir():
        return found
    for acc_dir in sorted(p for p in targeted_folder.iterdir() if p.is_dir()):
        for path in sorted(p for p in acc_dir.iterdir() if p.is_file()):
            split = split_extract_filename(path.name, genome_ids)
            if split:
                found.setdefault(acc_dir.name, {}).setdefault(split[0], []).append(path)
    return found


def scan_assemblies(targeted_folder: Path, genome_ids: Sequence[str]) -> Dict[str, Dict[str, Path]]:
    found: Dict[str, Dict[str, Path]] = {}
    if not targeted_folder.is_dir():
        return found
    for acc_dir in sorted(p for p in targeted_folder.iterdir() if p.is_dir()):
        for asm in sorted(p for p in acc_dir.glob(f"*{_ASSEMBLY_SUFFIX}") if p.is_dir()):
            found.setdefault(acc_dir.name, {})[asm.name[: -len(_ASSEMBLY_SUFFIX)]] = asm
    return found


def empty_assembly_dirs(targeted_folder: Path, genome_ids: Sequence[str]) -> List[Tuple[str, str]]:
    """(accession, genome_id) pairs whose assembly directory holds zero contigs, sorted."""
    pairs: List[Tuple[str, str]] = []
    for acc, per_genome_asm in scan_assemblies(targeted_folder, genome_ids).items():
        for genome_id, asm_dir in per_genome_asm.items():
            if summarise_contigs(asm_dir / _CONTIGS_NAME)["contigs"] == 0:
                pairs.append((acc, genome_id))
    return sorted(pairs)


def _read_accession_list(path: Optional[Union[str, Path]]) -> List[str]:
    if not path or not Path(path).exists():
        return []
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def _screening_from_matches(registry: Registry, matches_folder: Path) -> None:
    import csv

    for csv_path in sorted(matches_folder.glob("*.csv")):
        with open(csv_path, newline="") as handle:
            for row in csv.DictReader(handle):
                acc = (row.get("acc") or row.get("SRA accession") or "").strip()
                if not acc:
                    continue
                try:
                    containment = float(row.get("containment", ""))
                except ValueError:
                    continue
                cani_raw = row.get("cANI") or ""
                try:
                    cani = float(cani_raw) if cani_raw.strip() else None
                except ValueError:
                    cani = None
                record_screening(registry, acc, csv_path.stem, containment, cani, "matches", 0.0, csv_path)
                registry.datasets[acc]["screening"]["inferred"] = True


def _bootstrap_selection(
    registry: Registry, accessions_file: Optional[Union[str, Path]], parsed_containment: Optional[Union[str, Path]]
) -> None:
    wanted = _read_accession_list(accessions_file)
    if parsed_containment and Path(parsed_containment).exists():
        with open(parsed_containment) as handle:
            next(handle, None)
            wanted += [ln.split("\t", 1)[0].strip() for ln in handle if ln.strip()]
    if wanted:
        record_selection(
            registry, list(dict.fromkeys(wanted)), {"source": "bootstrap"}, accessions_file or parsed_containment or ""
        )
        for acc in wanted:
            registry.datasets[acc]["selection"]["inferred"] = True


def _bootstrap_downloads_and_metadata(registry: Registry, paths: ProjectPaths) -> None:
    for acc in scan_downloads(paths.fastq):
        record_download(registry, acc, "downloaded", paths.fastq)
        registry.datasets[acc]["download"]["inferred"] = True
        registry.datasets[acc]["download"]["attempts"] = 0
    for acc in sorted(scan_metadata(paths.metadata)):
        record_metadata(registry, acc, paths.metadata / f"{acc}_metadata.xml", {})
        registry.datasets[acc]["metadata"]["inferred"] = True


def _infer_extraction(registry: Registry, acc: str, genome_id: str, files: Sequence[Path]) -> None:
    """Record one (accession, genome) extraction found on disk, marked as inferred.

    Reads are counted in every file of the pair (both mates, singles and unpaired), so the
    inferred count approximates the number of BAM records a real extraction records.
    """
    reads = sum(count_fastq_reads(f) for f in files)
    record_extraction(registry, acc, genome_id, files, reads, False, {})
    registry.datasets[acc]["extractions"][genome_id]["inferred"] = True


def _infer_assembly(registry: Registry, acc: str, genome_id: str, asm_dir: Path) -> None:
    """Record the assembly in ``asm_dir``, marked as inferred, when it holds contigs."""
    stats = summarise_contigs(asm_dir / _CONTIGS_NAME)
    if stats["contigs"] > 0:
        record_assembly(registry, acc, genome_id, asm_dir, stats, "", {})
        registry.datasets[acc]["extractions"][genome_id]["inferred"] = True


def _bootstrap_extractions(registry: Registry, paths: ProjectPaths, genome_ids: List[str]) -> None:
    for acc, per_genome_files in scan_extractions(paths.targeted, genome_ids).items():
        for genome_id, files in per_genome_files.items():
            _infer_extraction(registry, acc, genome_id, files)
    for acc, per_genome_asm in scan_assemblies(paths.targeted, genome_ids).items():
        for genome_id, asm_dir in per_genome_asm.items():
            _infer_assembly(registry, acc, genome_id, asm_dir)


def bootstrap_from_disk(
    paths: ProjectPaths,
    accessions_file: Optional[Union[str, Path]] = None,
    parsed_containment: Optional[Union[str, Path]] = None,
    target_path: Optional[Path] = None,
) -> Registry:
    """Rebuild a registry from what is on disk; every reconstructed block carries ``"inferred": true``.

    ``target_path`` is the file the caller intends to save this registry to (``status --init``
    passes ``registry_file``); it is bound to the registry before any writer below runs, so every
    path recorded during bootstrap is already relative to the right project root, even when
    ``--registry`` points somewhere other than the working directory. Left as ``None`` (a caller
    with no file in mind, e.g. the in-memory report built when no registry exists yet), the
    writers fall back to the working directory as the project root, same as an unbound
    ``Registry()``.
    """
    registry = Registry(path=target_path)
    if paths.matches.is_dir():
        _screening_from_matches(registry, paths.matches)
    _bootstrap_selection(registry, accessions_file, parsed_containment)
    _bootstrap_downloads_and_metadata(registry, paths)
    genome_ids = sorted(_genome_ids_on_disk(paths, registry))
    for genome_id in genome_ids:
        registry.genomes.setdefault(genome_id, {})
    _bootstrap_extractions(registry, paths, genome_ids)
    return registry


def reconcile(registry: Registry, paths: ProjectPaths) -> ReconcileReport:
    """Compare the registry with the disk: mark missing downloads, register untracked work.

    Untracked FASTQ and extractions are recorded the way ``bootstrap_from_disk`` records
    them, with ``"inferred": true``, so a project worked on outside MetaQuest lands in the
    journal instead of being reported as drift on every run. They stay in the report.

    A link into a shared store whose target has gone (an unmounted store, a dataset removed
    from it) is reported as well, and is not repaired here: removing the link would lose the
    record of which datasets this project uses.
    """
    # Imported here rather than at module level: the store package imports the data layer.
    from metaquest.store.link import dangling_links

    report = ReconcileReport()
    on_disk = scan_downloads(paths.fastq)
    for acc, record in registry.datasets.items():
        download = record.get("download", {})
        if download.get("state") == "downloaded" and acc not in on_disk:
            download["state"] = "missing"
            download["date"] = _now()
            report.recorded_missing.append(acc)
    tracked = {acc for acc, r in registry.datasets.items() if r.get("download", {}).get("state") == "downloaded"}
    report.untracked_fastq = sorted(acc for acc in on_disk if acc not in tracked)
    for acc in report.untracked_fastq:
        record_download(registry, acc, "downloaded", paths.fastq, attempt=False)
        registry.datasets[acc]["download"]["inferred"] = True
        registry.datasets[acc]["download"]["attempts"] = 0
    genome_ids = sorted(_genome_ids_on_disk(paths, registry))
    assemblies = scan_assemblies(paths.targeted, genome_ids)
    for acc, per_genome_files in scan_extractions(paths.targeted, genome_ids).items():
        for genome_id, files in per_genome_files.items():
            if extraction_record(registry, acc, genome_id) is None:
                report.untracked_extractions.append((acc, genome_id))
                _infer_extraction(registry, acc, genome_id, files)
                asm_dir = assemblies.get(acc, {}).get(genome_id)
                if asm_dir is not None:
                    _infer_assembly(registry, acc, genome_id, asm_dir)
    report.empty_assembly_dirs = empty_assembly_dirs(paths.targeted, genome_ids)
    report.dangling_links = dangling_links(paths.fastq)
    _fill_missing_download_verdicts(registry, paths)
    return report


def _fill_missing_download_verdicts(registry: Registry, paths: ProjectPaths) -> None:
    """Compute a completeness verdict for a downloaded accession that never got one.

    A project downloaded before completeness verification existed (or with
    ``--no-verify-downloads``) has metadata recorded but no ``download.complete`` verdict.
    When NCBI's recorded spot count is on file, this recomputes it the same way a fresh
    download would have, against the accession's files on disk. A record whose reads came from
    the shared store is filled in from that dataset's sidecar instead: the store already
    verified it when it was downloaded, and counting the reads again through a link would
    repeat work another project has done.
    """
    for acc, record in registry.datasets.items():
        download = record.get("download") or {}
        if download.get("state") != "downloaded" or download.get("complete"):
            continue
        if download.get("source") == "store":
            complete = _store_verdict(registry, acc)
            if complete is not None:
                set_download_verdict(registry, acc, complete)
            continue
        spots = (record.get("metadata") or {}).get("run_total_spots")
        if not spots:
            continue
        acc_dir = paths.fastq / acc
        if not acc_dir.is_dir():
            continue
        verify = verify_download(acc, acc_dir, spots)
        complete = {"method": "spots", "ratio": verify["ratio"], "verdict": verify["verdict"]}
        set_download_verdict(registry, acc, complete)


def _store_verdict(registry: Registry, accession: str) -> Optional[Dict[str, Any]]:
    """The completeness verdict the store's sidecar records for ``accession``, or None.

    Reads the store root the registry itself recorded; a project whose store has moved or is
    not mounted simply gets no verdict this time round, exactly as before.
    """
    root = (registry.store or {}).get("root")
    if not root:
        return None
    # Imported here, not at module level: metaquest.store imports this module.
    from metaquest.store.layout import sidecar_path, store_paths
    from metaquest.store.sidecar import sidecar_completeness

    try:
        return sidecar_completeness(sidecar_path(store_paths(Path(root)), accession))
    except (OSError, DataAccessError) as e:
        logger.warning("Could not read the store sidecar for %s: %s", accession, e)
        return None


def to_dataframes(registry: Registry) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Flat views: one row per accession, and one row per (accession, genome) extraction."""
    import pandas as pd

    rows = []
    ext_rows = []
    for acc, record in registry.datasets.items():
        rows.append(
            {
                "accession": acc,
                "screened_genomes": ",".join(sorted(record.get("screening", {}).get("genomes", {}))),
                "selected": bool(record.get("selection", {}).get("selected", False)),
                "excluded": bool(record.get("exclusion", {}).get("excluded", False)),
                "exclusion_reason": record.get("exclusion", {}).get("reason", ""),
                "download_state": record.get("download", {}).get("state", ""),
                "download_date": record.get("download", {}).get("date", ""),
                "bytes_total": record.get("download", {}).get("bytes_total", 0),
                "metadata": "metadata" in record,
                "analyses": ",".join(sorted(record.get("analyses", {}))),
            }
        )
        for genome_id, ext in record.get("extractions", {}).items():
            asm = ext.get("assembly") or {}
            ext_rows.append(
                {
                    "accession": acc,
                    "genome_id": genome_id,
                    "mapped_reads": ext.get("mapped_reads"),
                    "extraction_date": ext.get("date"),
                    "contigs": asm.get("contigs"),
                    "total_bp": asm.get("total_bp"),
                    "n50": asm.get("n50"),
                    "assembly_date": asm.get("date"),
                }
            )
    datasets = pd.DataFrame(rows).set_index("accession") if rows else pd.DataFrame()
    extractions = pd.DataFrame(ext_rows)
    return datasets, extractions
