"""Per-project registry of dataset state, the project journal.

The registry records decisions and provenance for every SRA accession a
project touches: screening results per target genome, the selection criteria,
exclusions with reasons, download outcomes with file sizes and dates, analyses,
and per-genome extraction and assembly results. Existence on disk is never
taken from the registry alone; ``status`` re-checks the filesystem and the
scanners here rebuild or reconcile the journal from what is on disk.
"""

import gzip
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

from metaquest.core.constants import GENOME_FASTA_GLOBS
from metaquest.core.exceptions import DataAccessError
from metaquest.data.read_extraction import summarise_contigs
from metaquest.data.sra import accession_has_fastq

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "metaquest_registry.json"
SCHEMA_VERSION = 1
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
    path: Optional[Path] = None


@dataclass
class ReconcileReport:
    recorded_missing: List[str] = field(default_factory=list)
    untracked_fastq: List[str] = field(default_factory=list)
    untracked_extractions: List[Tuple[str, str]] = field(default_factory=list)
    empty_assembly_dirs: List[Tuple[str, str]] = field(default_factory=list)


# ----------------------------------------------------------------- persistence


def registry_path(explicit: Optional[Union[str, Path]] = None, start: Union[str, Path] = ".") -> Path:
    """The registry file to use: an explicit path, else the nearest one walking up from ``start``."""
    if explicit:
        return Path(explicit)
    current = Path(start).resolve()
    for folder in (current, *current.parents):
        candidate = folder / REGISTRY_FILENAME
        if candidate.exists():
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
    payload = {
        "version": registry.version,
        "created": registry.created,
        "updated": registry.updated,
        "genomes": registry.genomes,
        "datasets": registry.datasets,
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
    screening.update({"source": source, "query_threshold": query_threshold, "date": _now()})
    screening.setdefault("genomes", {})[genome_id] = {
        "containment": round(float(containment), 4),
        "cani": round(float(cani), 4) if cani is not None else None,
        "csv": str(csv_path) if csv_path else None,
    }
    registry.genomes.setdefault(genome_id, {})


def record_genome(registry: Registry, genome_id: str, fasta: Union[str, Path], manifest: Union[str, Path]) -> None:
    """Record where a target genome's FASTA lives, and the manifest it came from."""
    registry.genomes[genome_id] = {"fasta": str(fasta), "manifest": str(manifest), "date": _now()}


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


def record_screening_from_table(
    registry: Registry, table_path: Union[str, Path], matches_folder: Union[str, Path]
) -> int:
    """Record a screening entry for every positive containment in a parsed containment table.

    Reads the table written by ``parse_containment_data`` and, for every column except
    ``max_containment`` and ``max_containment_annotation`` (each a genome), records one
    screening entry per row with a value greater than 0. Returns the number of entries
    recorded. If the table does not exist, logs at debug level and returns 0 without raising.
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
            value = row[column]
            if pd.notna(value) and float(value) > 0:
                record_screening(
                    registry,
                    str(accession),
                    column,
                    float(value),
                    None,
                    "matches",
                    0.0,
                    Path(matches_folder) / f"{column}.csv",
                )
                recorded += 1
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


def _file_entries(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    entries = []
    for path in paths:
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
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
) -> None:
    """Record a download outcome; ``state`` is downloaded, failed, missing or skipped.

    ``attempt`` counts this call against ``attempts``; pass ``False`` when recording a
    state without an actual download attempt (e.g. a file found already present on disk).
    """
    download = upsert_dataset(registry, accession).setdefault("download", {"attempts": 0})
    if attempt and state in ("downloaded", "failed"):
        download["attempts"] = int(download.get("attempts", 0)) + 1
    files: List[Dict[str, Any]] = []
    if state == "downloaded":
        files = _file_entries(sorted(p for p in (Path(fastq_dir) / accession).glob("*.fastq*") if p.is_file()))
    download.update(
        {
            "state": state,
            "date": _now(),
            "files": files,
            "bytes_total": sum(int(f["bytes"]) for f in files),
            "message": message,
        }
    )
    download.pop("inferred", None)


def nan_to_none(value: Any) -> Any:
    """Return ``None`` for a pandas NaN/NA value, else ``value`` unchanged."""
    import pandas as pd

    return None if pd.isna(value) else value


def record_metadata(registry: Registry, accession: str, xml_path: Union[str, Path], fields: Dict[str, Any]) -> None:
    record: Dict[str, Any] = {"xml": str(xml_path), "date": _now()}
    for key in ("run_size", "run_md5", "assay_type", "organism", "collection_date"):
        record[key] = fields.get(key)
    upsert_dataset(registry, accession)["metadata"] = record


def record_analysis(
    registry: Registry, accession: str, analysis: str, output: Union[str, Path], summary: Dict[str, Any]
) -> None:
    upsert_dataset(registry, accession).setdefault("analyses", {})[analysis] = {
        "date": _now(),
        "output": str(output),
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
    extractions = upsert_dataset(registry, accession).setdefault("extractions", {})
    previous = extractions.get(genome_id, {})
    extractions[genome_id] = {
        "date": _now(),
        "genome_fasta": params.get("genome_fasta"),
        "preset": params.get("preset"),
        "threshold": params.get("threshold"),
        "mapped_reads": int(mapped_reads),
        "unequal_mates": bool(unequal_mates),
        "files": [str(p) for p in files],
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
    entry["assembly"] = {
        "date": _now(),
        "dir": str(assembly_dir),
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


def count_fastq_reads(path: Union[str, Path]) -> int:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return sum(1 for _ in handle) // 4
    with open(path, "rt") as handle:
        return sum(1 for _ in handle) // 4


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
    """Accession -> (number of FASTQ files, total bytes) for every per-accession folder with reads."""
    found: Dict[str, Tuple[int, int]] = {}
    if not fastq_folder.is_dir():
        return found
    for folder in sorted(p for p in fastq_folder.iterdir() if p.is_dir()):
        if accession_has_fastq(folder):
            files = [p for p in folder.glob("*.fastq*") if p.is_file()]
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


def _counts_as_read_file(name: str, genome_ids: Sequence[str]) -> bool:
    """True for the mate-1, single-end and unpaired files, so paired reads are counted once."""
    split = split_extract_filename(name, genome_ids)
    return split is not None and split[1] in ("_1", "", "_0")


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


def _infer_extraction(
    registry: Registry, acc: str, genome_id: str, files: Sequence[Path], genome_ids: Sequence[str]
) -> None:
    """Record one (accession, genome) extraction found on disk, marked as inferred."""
    reads = sum(count_fastq_reads(f) for f in files if _counts_as_read_file(f.name, genome_ids))
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
            _infer_extraction(registry, acc, genome_id, files, genome_ids)
    for acc, per_genome_asm in scan_assemblies(paths.targeted, genome_ids).items():
        for genome_id, asm_dir in per_genome_asm.items():
            _infer_assembly(registry, acc, genome_id, asm_dir)


def bootstrap_from_disk(
    paths: ProjectPaths,
    accessions_file: Optional[Union[str, Path]] = None,
    parsed_containment: Optional[Union[str, Path]] = None,
) -> Registry:
    """Rebuild a registry from what is on disk; every reconstructed block carries ``"inferred": true``."""
    registry = Registry()
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
    """
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
                _infer_extraction(registry, acc, genome_id, files, genome_ids)
                asm_dir = assemblies.get(acc, {}).get(genome_id)
                if asm_dir is not None:
                    _infer_assembly(registry, acc, genome_id, asm_dir)
    report.empty_assembly_dirs = empty_assembly_dirs(paths.targeted, genome_ids)
    return report


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
