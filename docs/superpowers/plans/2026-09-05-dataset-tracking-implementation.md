# Dataset Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every MetaQuest project a registry that records which datasets were screened, selected, excluded, downloaded, analysed, extracted and assembled (with criteria, dates, counts and parameters), make `status` show the accession x stage x genome matrix and what to run next, add a `blacklist` command, and make extraction and assembly skip work already done unless `--force`.

**Architecture:** One new module `metaquest/data/registry.py` owns the JSON journal (`metaquest_registry.json` in the project root): load, atomic locked save, `record_*` writers, `query`, filesystem scanners, `bootstrap_from_disk` and `reconcile`. Commands call one `record_*` at the end of their work (load, mutate, save). `status` reads the registry and re-checks disk, so existence is always true to the filesystem and decisions come from the journal. Four stacked branches, one PR each: `feat/registry-core` (Tasks 1-3), `feat/registry-recording` (Tasks 4-7), `feat/extraction-idempotent` (Task 8), `docs/dataset-tracking` (Task 9).

**Tech Stack:** Python 3.12, stdlib `json`/`os`/`gzip`/`datetime`, pandas only for TSV export, argparse registry (`metaquest/cli/base.py`), pytest with `tmp_path` and `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-09-05-dataset-tracking-design.md`

## Global Constraints

- black line length 120; flake8, mypy and the radon ceiling via `make check`; `make test` green after every task; `make pipeline` after Tasks 3, 8 and 9.
- CLI flags use dashes. Plain, modest language; no Unicode symbols in new code, help or docs.
- Unit tests never touch the network or external tools; patch `SecureSubprocess.run_secure` (use `tests/helpers_extraction.py::_fake_tools` for minimap2/samtools/megahit) and never the code under test.
- Registry writes happen only on the main thread, only through `save_registry`, and only after load-mutate in the same call. Presence on disk is never inferred from the registry alone by `status`.
- Every record function tolerates missing keys in older registries (use `setdefault`/`get`).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Do not `git add` the untracked `AGENTS.md`; `docs/superpowers/` files are committed in Task 9.

---

## Phase 1: registry core and status (branch `feat/registry-core`)

### Task 1: The registry module

**Files:**
- Create: `metaquest/data/registry.py`, `tests/test_data_registry.py`
- Modify: `metaquest/data/read_extraction.py` (add `summarise_contigs`)

**Interfaces:**
- Produces (all in `metaquest.data.registry`): `REGISTRY_FILENAME`, `SCHEMA_VERSION`, `STAGES`, `LOCK_STALE_SECONDS`, `ProjectPaths`, `Registry`, `ReconcileReport`, `registry_path`, `load_registry`, `save_registry`, `upsert_dataset`, `record_screening`, `record_selection`, `record_exclusion`, `clear_exclusion`, `record_download`, `record_metadata`, `record_analysis`, `record_extraction`, `record_assembly`, `extraction_record`, `query`, `stage_counts`, `known_genome_ids`, `scan_downloads`, `scan_metadata`, `scan_extractions`, `scan_assemblies`, `split_extract_filename`, `count_fastq_reads`, `bootstrap_from_disk`, `reconcile`, `to_dataframes`.
- Produces in `metaquest.data.read_extraction`: `summarise_contigs(contigs: Path) -> Dict[str, int]` with keys `contigs`, `total_bp`, `n50`, `largest`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_registry.py`:

```python
"""Tests for the per-project dataset registry (metaquest.data.registry)."""

import gzip
import json
import os
from pathlib import Path

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry as reg
from metaquest.data.read_extraction import summarise_contigs


def _fastq(path: Path, reads: int = 2, gz: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(reads))
    if gz:
        with gzip.open(path, "wt") as handle:
            handle.write(body)
    else:
        path.write_text(body)
    return path


def _project(tmp_path: Path) -> reg.ProjectPaths:
    return reg.ProjectPaths(
        fastq=tmp_path / "fastq", metadata=tmp_path / "metadata", genomes=tmp_path / "genomes",
        targeted=tmp_path / "targeted", matches=tmp_path / "matches",
    )


class TestLoadSave:
    def test_absent_registry_is_empty(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        assert r.datasets == {} and r.version == reg.SCHEMA_VERSION and r.path == tmp_path / "metaquest_registry.json"

    def test_round_trip_and_atomic_write(self, tmp_path):
        target = tmp_path / "metaquest_registry.json"
        r = reg.load_registry(target)
        reg.record_exclusion(r, "SRR1", "amplicon", source="user")
        reg.save_registry(r)
        assert target.exists()
        assert not list(tmp_path.glob("*.tmp.*")) and not list(tmp_path.glob("*.lock"))
        again = reg.load_registry(target)
        assert again.datasets["SRR1"]["exclusion"]["reason"] == "amplicon"
        assert json.loads(target.read_text())["version"] == reg.SCHEMA_VERSION

    def test_invalid_json_raises(self, tmp_path):
        target = tmp_path / "metaquest_registry.json"
        target.write_text("{not json")
        with pytest.raises(DataAccessError, match="not valid JSON"):
            reg.load_registry(target)

    def test_registry_path_walks_up(self, tmp_path, monkeypatch):
        (tmp_path / "metaquest_registry.json").write_text("{}")
        sub = tmp_path / "targeted" / "SRR1"
        sub.mkdir(parents=True)
        assert reg.registry_path(start=sub).resolve() == (tmp_path / "metaquest_registry.json").resolve()
        assert reg.registry_path(explicit=str(tmp_path / "x.json")) == tmp_path / "x.json"

    def test_stale_lock_is_removed_and_fresh_lock_times_out(self, tmp_path, monkeypatch):
        target = tmp_path / "metaquest_registry.json"
        lock = tmp_path / "metaquest_registry.json.lock"
        monkeypatch.setattr(reg, "LOCK_STALE_SECONDS", 0.3)
        lock.write_text("1")
        os.utime(lock, (0, 0))  # ancient -> stale -> removed
        reg.save_registry(reg.load_registry(target))
        assert target.exists() and not lock.exists()
        lock.write_text("1")  # fresh lock held by "another process"
        with pytest.raises(DataAccessError, match="locked"):
            reg.save_registry(reg.load_registry(target))
        lock.unlink()


class TestRecords:
    def test_screening_selection_exclusion(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_screening(r, "SRR1", "GCF_1", 0.94611, 0.9974, "branchwater", 0.1, tmp_path / "bw.csv")
        reg.record_screening(r, "SRR1", "GCF_2", 0.2, None, "branchwater", 0.1, None)
        reg.record_selection(r, ["SRR1"], {"column": "GCF_1", "threshold": 0.5}, tmp_path / "accessions.txt")
        reg.record_selection(r, ["SRR2"], {"column": "GCF_1", "threshold": 0.9}, tmp_path / "accessions.txt")
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_1"]["containment"] == 0.9461
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_2"]["cani"] is None
        assert r.datasets["SRR1"]["selection"]["selected"] is False
        assert r.datasets["SRR2"]["selection"]["selected"] is True
        assert r.datasets["SRR2"]["selection"]["criteria"]["threshold"] == 0.9
        reg.record_exclusion(r, "SRR2", "16S amplicon")
        assert reg.query(r, "excluded") == ["SRR2"]
        reg.clear_exclusion(r, "SRR2")
        assert reg.query(r, "excluded") == []

    def test_download_records_files_and_attempts(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        _fastq(tmp_path / "fastq" / "SRR1" / "SRR1_1.fastq")
        _fastq(tmp_path / "fastq" / "SRR1" / "SRR1_2.fastq")
        reg.record_download(r, "SRR1", "failed", tmp_path / "fastq", message="timeout")
        reg.record_download(r, "SRR1", "downloaded", tmp_path / "fastq")
        dl = r.datasets["SRR1"]["download"]
        assert dl["state"] == "downloaded" and dl["attempts"] == 2
        assert [Path(f["path"]).name for f in dl["files"]] == ["SRR1_1.fastq", "SRR1_2.fastq"]
        assert dl["bytes_total"] == sum(f["bytes"] for f in dl["files"]) > 0
        reg.record_download(r, "SRR3", "skipped", tmp_path / "fastq", message="--max-downloads")
        assert r.datasets["SRR3"]["download"]["attempts"] == 0

    def test_metadata_analysis_extraction_assembly(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_metadata(r, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_size": "1234", "organism": "x"})
        reg.record_analysis(r, "SRR1", "sra_stats", tmp_path / "sra_statistics.csv", {"total_reads": 10})
        reg.record_extraction(
            r, "SRR1", "GCF_1", [tmp_path / "t" / "GCF_1_1.fastq.gz"], 239464, False,
            {"genome_fasta": "genomes/GCF_1.fna", "preset": "sr", "threshold": 0.1},
        )
        reg.record_assembly(
            r, "SRR1", "GCF_1", tmp_path / "t" / "GCF_1_assembly",
            {"contigs": 188, "total_bp": 1209849, "n50": 15400, "largest": 57491}, "v1.2.9", {"threads": 1},
        )
        rec = reg.extraction_record(r, "SRR1", "GCF_1")
        assert rec["mapped_reads"] == 239464 and rec["assembly"]["n50"] == 15400 and rec["assembly"]["version"] == "v1.2.9"
        assert r.datasets["SRR1"]["metadata"]["run_size"] == "1234"
        assert r.datasets["SRR1"]["analyses"]["sra_stats"]["summary"]["total_reads"] == 10
        # a new extraction record keeps the assembly block
        reg.record_extraction(r, "SRR1", "GCF_1", [], 0, True, {"genome_fasta": "g", "preset": "sr", "threshold": 0.1})
        assert reg.extraction_record(r, "SRR1", "GCF_1")["assembly"]["contigs"] == 188

    def test_query_and_stage_counts(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        for acc, cont in (("SRR1", 0.9), ("SRR2", 0.5), ("SRR3", 0.05)):
            reg.record_screening(r, acc, "GCF_1", cont, None, "branchwater", 0.01, None)
        reg.record_selection(r, ["SRR1", "SRR2"], {"threshold": 0.1}, Path("accessions.txt"))
        reg.record_download(r, "SRR1", "downloaded", tmp_path / "fastq")
        reg.record_extraction(r, "SRR1", "GCF_1", [Path("a")], 100, False, {})
        reg.record_extraction(r, "SRR2", "GCF_1", [], 0, False, {})
        reg.record_assembly(r, "SRR1", "GCF_1", Path("d"), {"contigs": 3, "total_bp": 9, "n50": 3, "largest": 4}, "v", {})
        assert reg.query(r, "screened") == ["SRR1", "SRR2", "SRR3"]
        assert reg.query(r, "selected") == ["SRR1", "SRR2"]
        assert reg.query(r, "downloaded") == ["SRR1"]
        assert reg.query(r, "extracted", "GCF_1") == ["SRR1"]
        assert reg.query(r, "assembled", "GCF_1") == ["SRR1"]
        counts = reg.stage_counts(r)
        assert counts["stages"]["selected"] == 2 and counts["genomes"]["GCF_1"]["zero_mapped"] == ["SRR2"]
        with pytest.raises(DataAccessError, match="Unknown stage"):
            reg.query(r, "bogus")


class TestScanners:
    def test_summarise_contigs_uses_len_headers(self, tmp_path):
        fa = tmp_path / "final.contigs.fa"
        fa.write_text(">k141_1 flag=1 multi=2.0 len=10\nACGTACGTAC\n>k141_2 flag=1 multi=2.0 len=4\nACGT\n>k141_3 len=6\nACGTAC\n")
        assert summarise_contigs(fa) == {"contigs": 3, "total_bp": 20, "n50": 10, "largest": 10}
        fa.write_text(">a\nACGTACGT\n>b\nAC\n")
        assert summarise_contigs(fa) == {"contigs": 2, "total_bp": 10, "n50": 8, "largest": 8}
        assert summarise_contigs(tmp_path / "missing.fa") == {"contigs": 0, "total_bp": 0, "n50": 0, "largest": 0}

    def test_count_fastq_reads_plain_and_gz(self, tmp_path):
        assert reg.count_fastq_reads(_fastq(tmp_path / "a.fastq", reads=3)) == 3
        assert reg.count_fastq_reads(_fastq(tmp_path / "b.fastq.gz", reads=5, gz=True)) == 5

    def test_split_extract_filename(self):
        genomes = ["GCF_000008025.1", "wMel_ref_1"]
        assert reg.split_extract_filename("GCF_000008025.1_1.fastq.gz", genomes) == ("GCF_000008025.1", "_1")
        assert reg.split_extract_filename("GCF_000008025.1.fastq.gz", genomes) == ("GCF_000008025.1", "")
        assert reg.split_extract_filename("wMel_ref_1_s.fastq.gz", genomes) == ("wMel_ref_1", "_s")
        assert reg.split_extract_filename("GCF_9_1.fastq.gz", genomes) == ("GCF_9", "_1")  # suffix fallback
        assert reg.split_extract_filename("notes.txt", genomes) is None

    def test_bootstrap_and_reconcile(self, tmp_path):
        paths = _project(tmp_path)
        _fastq(paths.fastq / "SRR1" / "SRR1_1.fastq")
        _fastq(paths.fastq / "SRR2" / "SRR2_1.fastq")
        (paths.metadata).mkdir()
        (paths.metadata / "SRR1_metadata.xml").write_text("<x/>")
        paths.genomes.mkdir()
        (paths.genomes / "GCF_1.fna").write_text(">c\nACGT\n")
        _fastq(paths.targeted / "SRR1" / "GCF_1_1.fastq.gz", gz=True)
        _fastq(paths.targeted / "SRR1" / "GCF_1_2.fastq.gz", gz=True)
        asm = paths.targeted / "SRR1" / "GCF_1_assembly"
        asm.mkdir(parents=True)
        (asm / "final.contigs.fa").write_text(">k len=5\nACGTA\n")
        (paths.targeted / "SRR2" / "GCF_1_assembly").mkdir(parents=True)  # interrupted, empty
        paths.matches.mkdir()
        (paths.matches / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\nSRR2,0.3,0.9\nSRR9,0.2,0.8\n")
        acc_file = tmp_path / "accessions.txt"
        acc_file.write_text("SRR1\nSRR2\n")

        r = reg.bootstrap_from_disk(paths, accessions_file=acc_file)
        assert r.datasets["SRR9"]["screening"]["genomes"]["GCF_1"]["containment"] == 0.2
        assert r.datasets["SRR1"]["selection"]["selected"] is True and r.datasets["SRR1"]["selection"]["inferred"] is True
        assert r.datasets["SRR1"]["download"]["state"] == "downloaded" and r.datasets["SRR1"]["download"]["inferred"] is True
        assert r.datasets["SRR1"]["metadata"]["inferred"] is True and "metadata" not in r.datasets["SRR2"]
        ext = reg.extraction_record(r, "SRR1", "GCF_1")
        assert ext["mapped_reads"] == 2 and ext["assembly"]["contigs"] == 1 and ext["inferred"] is True
        assert "GCF_1" in r.genomes

        (paths.fastq / "SRR2" / "SRR2_1.fastq").unlink()
        _fastq(paths.fastq / "SRR7" / "SRR7_1.fastq")
        report = reg.reconcile(r, paths)
        assert report.recorded_missing == ["SRR2"]
        assert report.untracked_fastq == ["SRR7"]
        assert report.empty_assembly_dirs == [("SRR2", "GCF_1")]
        assert r.datasets["SRR2"]["download"]["state"] == "missing"

    def test_to_dataframes(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_selection(r, ["SRR1"], {"threshold": 0.1}, Path("a.txt"))
        reg.record_extraction(r, "SRR1", "GCF_1", [], 7, False, {"preset": "sr"})
        datasets, extractions = reg.to_dataframes(r)
        assert list(datasets.index) == ["SRR1"] and bool(datasets.loc["SRR1", "selected"]) is True
        assert extractions.loc[0, "genome_id"] == "GCF_1" and int(extractions.loc[0, "mapped_reads"]) == 7
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_data_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metaquest.data.registry'` (and `ImportError` for `summarise_contigs`).

- [ ] **Step 3: Add `summarise_contigs` to `metaquest/data/read_extraction.py`**

Append (after `assemble_extracted_reads`):

```python
def summarise_contigs(contigs: Union[str, Path]) -> Dict[str, int]:
    """Contig count, total length, N50 and largest contig of a FASTA file.

    megahit headers carry ``len=<bp>``; when present that value is used, so the
    scan reads only header lines. Otherwise sequence lengths are summed.
    """
    path = Path(contigs)
    if not path.exists():
        return {"contigs": 0, "total_bp": 0, "n50": 0, "largest": 0}
    lengths: List[int] = []
    current = 0
    have_current = False
    header_len = False
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if have_current:
                    lengths.append(current)
                have_current = True
                match = re.search(r"\blen=(\d+)", line)
                current = int(match.group(1)) if match else 0
                header_len = match is not None
            elif have_current and not header_len:
                current += len(line.strip())
    if have_current:
        lengths.append(current)
    lengths.sort(reverse=True)
    total = sum(lengths)
    n50 = 0
    running = 0
    for length in lengths:
        running += length
        if running * 2 >= total:
            n50 = length
            break
    return {"contigs": len(lengths), "total_bp": total, "n50": n50, "largest": lengths[0] if lengths else 0}
```

(Add `import re` at the top; `List`, `Dict`, `Union`, `Path` are already imported.)

- [ ] **Step 4: Create `metaquest/data/registry.py`**

```python
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.data.read_extraction import summarise_contigs
from metaquest.data.sra import accession_has_fastq

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "metaquest_registry.json"
SCHEMA_VERSION = 1
STAGES = ("screened", "selected", "excluded", "downloaded", "analysed", "extracted", "assembled")
LOCK_STALE_SECONDS = 30.0
_MATE_SUFFIXES = ("_1", "_2", "_s", "_0")
_ASSEMBLY_SUFFIX = "_assembly"
_CONTIGS_NAME = "final.contigs.fa"
_GENOME_GLOBS = ("*.fna", "*.fna.gz", "*.fasta", "*.fasta.gz", "*.fa", "*.fa.gz")


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
        data = json.loads(target.read_text())
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
    deadline = time.monotonic() + LOCK_STALE_SECONDS
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > LOCK_STALE_SECONDS:
                logger.warning("Removing stale registry lock %s (%.0f s old)", lock, age)
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise DataAccessError(f"Registry is locked by another process: {lock}")
            time.sleep(0.05)


def save_registry(registry: Registry, path: Optional[Union[str, Path]] = None) -> Path:
    """Write the registry atomically (temp file plus rename) under a lock file."""
    target = Path(path) if path else (registry.path or registry_path())
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
    lock = target.with_name(target.name + ".lock")
    _acquire_lock(lock)
    try:
        tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, target)
    finally:
        lock.unlink(missing_ok=True)
    return target


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
    registry: Registry, accession: str, state: str, fastq_dir: Union[str, Path], message: str = ""
) -> None:
    """Record a download outcome; ``state`` is downloaded, failed, missing or skipped."""
    download = upsert_dataset(registry, accession).setdefault("download", {"attempts": 0})
    if state in ("downloaded", "failed"):
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


def record_metadata(registry: Registry, accession: str, xml_path: Union[str, Path], fields: Dict[str, Any]) -> None:
    record = {"xml": str(xml_path), "date": _now()}
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
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
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
    for pattern in _GENOME_GLOBS:
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
                try:
                    containment = float(row.get("containment", ""))
                except ValueError:
                    continue
                if not acc:
                    continue
                cani_raw = row.get("cANI") or ""
                cani = float(cani_raw) if cani_raw.strip() else None
                record_screening(registry, acc, csv_path.stem, containment, cani, "matches", 0.0, csv_path)
                registry.datasets[acc]["screening"]["inferred"] = True


def bootstrap_from_disk(
    paths: ProjectPaths,
    accessions_file: Optional[Union[str, Path]] = None,
    parsed_containment: Optional[Union[str, Path]] = None,
) -> Registry:
    """Rebuild a registry from what is on disk; every reconstructed block carries ``"inferred": true``."""
    registry = Registry()
    if paths.matches.is_dir():
        _screening_from_matches(registry, paths.matches)
    wanted = _read_accession_list(accessions_file)
    if parsed_containment and Path(parsed_containment).exists():
        with open(parsed_containment) as handle:
            next(handle, None)
            wanted += [ln.split("\t", 1)[0].strip() for ln in handle if ln.strip()]
    if wanted:
        record_selection(registry, list(dict.fromkeys(wanted)), {"source": "bootstrap"}, accessions_file or parsed_containment or "")
        for acc in wanted:
            registry.datasets[acc]["selection"]["inferred"] = True
    for acc in scan_downloads(paths.fastq):
        record_download(registry, acc, "downloaded", paths.fastq)
        registry.datasets[acc]["download"]["inferred"] = True
        registry.datasets[acc]["download"]["attempts"] = 0
    for acc in sorted(scan_metadata(paths.metadata)):
        record_metadata(registry, acc, paths.metadata / f"{acc}_metadata.xml", {})
        registry.datasets[acc]["metadata"]["inferred"] = True
    genome_ids = sorted(_genome_ids_on_disk(paths, registry))
    for genome_id in genome_ids:
        registry.genomes.setdefault(genome_id, {})
    for acc, per_genome in scan_extractions(paths.targeted, genome_ids).items():
        for genome_id, files in per_genome.items():
            reads = sum(count_fastq_reads(f) for f in files if _counts_as_read_file(f.name, genome_ids))
            record_extraction(registry, acc, genome_id, files, reads, False, {})
            registry.datasets[acc]["extractions"][genome_id]["inferred"] = True
    for acc, per_genome in scan_assemblies(paths.targeted, genome_ids).items():
        for genome_id, asm_dir in per_genome.items():
            stats = summarise_contigs(asm_dir / _CONTIGS_NAME)
            if stats["contigs"] > 0:
                record_assembly(registry, acc, genome_id, asm_dir, stats, "", {})
                registry.datasets[acc]["extractions"][genome_id]["inferred"] = True
    return registry


def reconcile(registry: Registry, paths: ProjectPaths) -> ReconcileReport:
    """Compare the registry with the disk: mark missing downloads, list untracked work and empty assemblies."""
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
    genome_ids = sorted(_genome_ids_on_disk(paths, registry))
    for acc, per_genome in scan_extractions(paths.targeted, genome_ids).items():
        for genome_id in per_genome:
            if extraction_record(registry, acc, genome_id) is None:
                report.untracked_extractions.append((acc, genome_id))
    for acc, per_genome in scan_assemblies(paths.targeted, genome_ids).items():
        for genome_id, asm_dir in per_genome.items():
            if summarise_contigs(asm_dir / _CONTIGS_NAME)["contigs"] == 0:
                report.empty_assembly_dirs.append((acc, genome_id))
    return report


def to_dataframes(registry: Registry):
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_data_registry.py -v`
Expected: all PASS. Then `make check` (fix black/flake8/mypy/radon complaints: if `_in_stage` or `bootstrap_from_disk` exceed the radon ceiling, split them into `_extraction_stage` / `_bootstrap_extractions` helpers without changing behaviour).

- [ ] **Step 6: Commit**

```bash
git checkout -b feat/registry-core main
git add metaquest/data/registry.py metaquest/data/read_extraction.py tests/test_data_registry.py
git commit -m "feat: add the per-project dataset registry module

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 2: Constants, genome manifest fix, megahit version flag

**Files:**
- Modify: `metaquest/core/constants.py` (`DEFAULT_CONTAINMENT_THRESHOLD = 0.1`, add `GENOME_FASTA_GLOBS`, add `"--version"` to the megahit `safe_params`), `metaquest/cli/commands/genome.py:_create_manifest`, `metaquest/cli/commands/select.py` and `metaquest/cli/commands/read_extraction.py` (use the constant as the `--threshold` default), `metaquest/data/registry.py` (import `GENOME_FASTA_GLOBS` instead of `_GENOME_GLOBS`)
- Test: `tests/test_cli_genome.py`, `tests/test_security_comprehensive.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli_genome.py` (find the `genome_prepare` manifest tests) add:

```python
    def test_manifest_lists_plain_fna_files(self, tmp_path):
        from metaquest.cli.commands.genome import GenomePrepareCommand

        (tmp_path / "GCF_000006945.2.fna").write_text(">c\nACGT\n")
        (tmp_path / "GCF_000006945.2.faa").write_text(">p\nM\n")
        (tmp_path / "other.fasta.gz").write_bytes(b"")
        manifest = tmp_path / "manifest.csv"
        n = GenomePrepareCommand()._create_manifest(tmp_path, str(manifest))
        rows = manifest.read_text().splitlines()
        assert n == 2
        assert rows[0] == "name,genome_filename,protein_filename"
        assert rows[1].startswith("GCF_000006945.2,") and rows[1].endswith("GCF_000006945.2.faa")
        assert rows[2].startswith("other,")
```

In `tests/test_security_comprehensive.py` add:

```python
    def test_megahit_version_flag_allowed(self):
        assert SecureSubprocess._build_validated_command("megahit", ["--version"]) == ["megahit", "--version"]
```

Run both: expected FAIL (manifest count 1 and wrong name; `SecurityError` for `--version`).

- [ ] **Step 2: Implement**

`metaquest/core/constants.py`: set `DEFAULT_CONTAINMENT_THRESHOLD = 0.1`; add

```python
# FASTA file patterns accepted as genome inputs (plain and gzipped).
GENOME_FASTA_GLOBS = ("*.fna", "*.fna.gz", "*.fasta", "*.fasta.gz", "*.fa", "*.fa.gz")
```

and `"--version",` in the megahit `safe_params` set.

`metaquest/cli/commands/genome.py` `_create_manifest`:

```python
    def _create_manifest(self, output_dir: Path, manifest_file: str) -> int:
        """Create a manifest CSV from the genome FASTA files in output_dir."""
        genome_files = sorted({p for pattern in GENOME_FASTA_GLOBS for p in output_dir.glob(pattern)})
        if not genome_files:
            self.logger.warning("No genome files found in %s", output_dir)
            return 0

        manifest_path = Path(manifest_file)
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "genome_filename", "protein_filename"])
            for gf in genome_files:
                name = _genome_name(gf.name)
                protein = next((p for p in (gf.with_name(name + ".faa"), gf.with_name(name + ".faa.gz")) if p.exists()), None)
                writer.writerow([name, str(gf), str(protein) if protein else ""])

        self.logger.info("Created manifest with %d entries: %s", len(genome_files), manifest_path)
        return len(genome_files)
```

with a module-level helper:

```python
def _genome_name(filename: str) -> str:
    """Strip the FASTA suffix (and .gz) from a genome file name."""
    for suffix in (".fna.gz", ".fasta.gz", ".fa.gz", ".fna", ".fasta", ".fa"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename
```

(import `GENOME_FASTA_GLOBS` from `metaquest.core.constants`). In `metaquest/data/registry.py` replace `_GENOME_GLOBS` with the imported `GENOME_FASTA_GLOBS`. In `select.py` and `read_extraction.py` use `default=DEFAULT_CONTAINMENT_THRESHOLD` for `--threshold` (help text unchanged; the rendered default stays 0.1).

- [ ] **Step 3: Verify and commit**

Run: `pytest tests/test_cli_genome.py tests/test_security_comprehensive.py tests/test_data_registry.py tests/test_cli_select.py tests/test_cli_read_extraction.py -v && make check && make test`

```bash
git add metaquest/core/constants.py metaquest/cli/commands/genome.py metaquest/cli/commands/select.py metaquest/cli/commands/read_extraction.py metaquest/data/registry.py tests/test_cli_genome.py tests/test_security_comprehensive.py
git commit -m "fix: list plain FASTA files in the genome manifest; share the containment default

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 3: `status` on the registry

**Files:**
- Modify: `metaquest/cli/commands/status.py` (rewrite), `tests/test_cli_status.py`

**Interfaces:**
- Consumes: everything in `metaquest.data.registry`.
- Produces: flags `--targeted-folder` (default `targeted`), `--matches-folder` (default `matches`), `--registry`, `--stage {screened,...}`, `--genome` (append), `--init`, `--reconcile`, `--export-tsv PREFIX`, `--next`; JSON keys `registry`, `on_disk`, `wanted` (unchanged shape), `stages`, `genomes`, `drift`, `next`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_status.py` (it already builds `Namespace` objects; add the new attributes with defaults through a helper if the file lacks one):

```python
def _status_args(root, **overrides):
    base = dict(
        fastq_folder=str(root / "fastq"), metadata_folder=str(root / "metadata"), genomes_folder=str(root / "genomes"),
        targeted_folder=str(root / "targeted"), matches_folder=str(root / "matches"), registry=str(root / "metaquest_registry.json"),
        accessions_file=None, parsed_containment=None, stage=None, genome=None, init=False, reconcile=False,
        export_tsv=None, next=False, list_missing=False, json=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _project_tree(root):
    for acc in ("SRR1", "SRR2"):
        d = root / "fastq" / acc
        d.mkdir(parents=True)
        (d / f"{acc}_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
    (root / "matches").mkdir()
    (root / "matches" / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\nSRR2,0.4,0.9\nSRR3,0.2,0.8\n")
    (root / "accessions.txt").write_text("SRR1\nSRR2\nSRR3\n")


class TestStatusWithRegistry:
    def test_without_registry_bootstraps_in_memory_and_hints(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(_status_args(tmp_path, accessions_file=str(tmp_path / "accessions.txt")))
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["registry"]["exists"] is False
        assert out["stages"]["screened"]["count"] == 3 and out["stages"]["downloaded"]["count"] == 2
        assert out["wanted"]["total"] == 3 and out["on_disk"]["fastq_accessions"] == 2
        assert not (tmp_path / "metaquest_registry.json").exists()

    def test_init_persists_bootstrap(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        assert rc == 0 and (tmp_path / "metaquest_registry.json").exists()
        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR1"]["download"]["inferred"] is True

    def test_stage_and_genome_filters_list_accessions(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        rc = StatusCommand().execute(_status_args(tmp_path, stage="screened", genome=["GCF_1"], json=False))
        out = capsys.readouterr().out
        assert rc == 0 and "SRR1" in out and "SRR3" in out

    def test_next_suggests_download_and_extraction(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        capsys.readouterr()
        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        commands = [n["command"] for n in out["next"]]
        assert any(c.startswith("metaquest download_sra") for c in commands)          # SRR3 selected, not downloaded
        assert any("extract_target_reads" in c and "GCF_1" in c for c in commands)    # SRR1/SRR2 downloaded, not extracted

    def test_reconcile_marks_missing_and_untracked(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        (tmp_path / "fastq" / "SRR2" / "SRR2_1.fastq").unlink()
        (tmp_path / "fastq" / "SRR8").mkdir()
        (tmp_path / "fastq" / "SRR8" / "SRR8_1.fastq").write_text("@r\nA\n+\nI\n")
        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        out = json.loads(capsys.readouterr().out)
        assert out["drift"]["recorded_missing"] == ["SRR2"] and out["drift"]["untracked_fastq"] == ["SRR8"]
        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR2"]["download"]["state"] == "missing"

    def test_export_tsv(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, export_tsv=str(tmp_path / "registry")))
        assert (tmp_path / "registry_datasets.tsv").exists() and (tmp_path / "registry_extractions.tsv").exists()

    def test_text_report_shows_stage_matrix(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "Local inventory" in out and "Stages" in out and "screened" in out and "GCF_1" in out
```

Run: `pytest tests/test_cli_status.py -v` -> the new tests FAIL (`AttributeError` on the new flags / missing keys).

- [ ] **Step 2: Rewrite `metaquest/cli/commands/status.py`**

Keep `name`, `help`, `group`, the existing four folder flags, `--accessions-file`, `--parsed-containment`, `--list-missing`, `--json`, and the existing `on_disk`/`wanted` computation (lines 73-104 today, moved into `_inventory_report(args) -> dict`). Add:

```python
        parser.add_argument("--targeted-folder", default="targeted", help="Root folder of extracted reads and assemblies")
        parser.add_argument("--matches-folder", default="matches", help="Folder of Branchwater match CSVs")
        parser.add_argument("--registry", default=None, help="Registry file (default: metaquest_registry.json found upwards from here)")
        parser.add_argument("--stage", choices=list(STAGES), default=None, help="List the accessions in one stage")
        parser.add_argument("--genome", action="append", default=None, help="Restrict extraction and assembly stages to a genome id (repeatable)")
        parser.add_argument("--init", action="store_true", help="Create the registry from what is on disk")
        parser.add_argument("--reconcile", action="store_true", help="Compare the registry with the disk and record missing downloads")
        parser.add_argument("--export-tsv", default=None, help="Write <PREFIX>_datasets.tsv and <PREFIX>_extractions.tsv")
        parser.add_argument("--next", action="store_true", help="Suggest the commands that advance the most accessions")
```

`execute` flow:

```python
    def execute(self, args: argparse.Namespace) -> int:
        try:
            paths = ProjectPaths(Path(args.fastq_folder), Path(args.metadata_folder), Path(args.genomes_folder), Path(args.targeted_folder), Path(args.matches_folder))
            registry_file = registry_path(args.registry)
            existed = registry_file.exists()
            if existed and not args.init:
                registry = load_registry(registry_file)
            else:
                registry = bootstrap_from_disk(paths, args.accessions_file, args.parsed_containment)
                registry.path = registry_file
                if args.init:
                    save_registry(registry)
                    self.logger.info("Registry written to %s", registry_file)
            drift = None
            if args.reconcile:
                drift = reconcile(registry, paths)
                save_registry(registry)
            report = self._inventory_report(args)
            report["registry"] = {"path": str(registry_file), "exists": existed or args.init, "updated": registry.updated}
            counts = stage_counts(registry)
            report["stages"] = {s: {"count": counts["stages"][s], "accessions": query(registry, s)} for s in STAGES}
            report["genomes"] = self._genome_report(registry, paths, args.genome, counts)
            report["drift"] = self._drift_report(drift) if drift else {}
            if args.next:
                report["next"] = self._next_steps(registry, paths)
            if args.export_tsv:
                self._export_tsv(registry, args.export_tsv)
            if not existed and not args.init:
                self.logger.info("No registry yet; the stages above were reconstructed from disk. Run: metaquest status --init")
            self._emit(args, report, registry)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error building status report: %s", e)
            return 1
```

Helpers (each small, under the radon ceiling): `_genome_report` adds per genome `extracted`, `assembled`, `zero_mapped` plus, from disk, `empty_assembly_dirs` (via `scan_assemblies` + `summarise_contigs`) restricted to `args.genome` when given; `_drift_report` turns `ReconcileReport` into a dict; `_next_steps` returns a list of `{"command": ..., "accessions": [...]}` for: selected and not excluded and not downloaded -> `metaquest download_sra --accessions-file <selection output>`; downloaded and not extracted for genome G -> `metaquest extract_target_reads --genome-id G --genome-fasta genomes/G.fna`; extracted with reads and not assembled -> `metaquest extract_target_reads --genome-id G --genome-fasta genomes/G.fna --assemble`; `_export_tsv` uses `to_dataframes` and `write_csv(..., sep="\t")`; `_emit` prints JSON, or the text report: the existing inventory block, then a "Stages" table (`stage  count  detail` where detail shows the selection criteria and date for `selected`, the number of excluded with reasons, and per-genome lines `GCF_1: extracted 3 (2 with 0 mapped reads), assembled 1, 1 empty assembly dir`), then `--stage` listing, `--list-missing` gaps (selected but not downloaded; downloaded but not extracted per genome), drift, and next steps. Reuse `read_records`-free plain string building; keep each render function focused.

- [ ] **Step 3: Verify, pipeline, commit**

Run: `pytest tests/test_cli_status.py -v && make check && make test && make pipeline`
Expected: green; the walkthrough's existing `status` step still passes.

```bash
git add metaquest/cli/commands/status.py tests/test_cli_status.py
git commit -m "feat: status reports the accession by stage matrix from the registry

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `feat/registry-core` -> `main`.

---

## Phase 2: recording in every stage command (branch `feat/registry-recording`, based on `feat/registry-core`)

### Task 4: `blacklist` command

**Files:**
- Create: `metaquest/cli/commands/blacklist.py`, `tests/test_cli_blacklist.py`
- Modify: `metaquest/cli/main.py` (import; register after `SelectDatasetsCommand()`), `tests/test_cli_main.py` (`expected_commands`), `metaquest/data/sra.py:_read_blacklist_files` (strip inline `# reason` comments and skip comment lines), `tests/test_data_sra.py`

**Interfaces:**
- Command `blacklist` (group `Reads`): `--add ACC [ACC ...]`, `--remove ACC [ACC ...]`, `--from-file PATH`, `--reason TEXT` (required with `--add`/`--from-file`), `--list`, `--blacklist-file` (default `blacklist.txt`), `--registry`. Exactly one of `--add`, `--remove`, `--from-file`, `--list`.
- Writes `record_exclusion`/`clear_exclusion` and keeps `blacklist.txt` in sync (one accession per line, `# reason` comment after it; removal deletes the line).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the blacklist command."""

import argparse
import json

from metaquest.cli.commands.blacklist import BlacklistCommand


def _args(tmp_path, **kw):
    base = dict(add=None, remove=None, from_file=None, reason=None, list=False,
                blacklist_file=str(tmp_path / "blacklist.txt"), registry=str(tmp_path / "metaquest_registry.json"))
    base.update(kw)
    return argparse.Namespace(**base)


def test_add_records_reason_and_writes_file(tmp_path, capsys):
    rc = BlacklistCommand().execute(_args(tmp_path, add=["SRR2517418"], reason="16S amplicon mislabelled as WGS"))
    assert rc == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    exclusion = data["datasets"]["SRR2517418"]["exclusion"]
    assert exclusion["excluded"] is True and exclusion["reason"] == "16S amplicon mislabelled as WGS"
    assert (tmp_path / "blacklist.txt").read_text().splitlines() == ["SRR2517418  # 16S amplicon mislabelled as WGS"]


def test_add_requires_reason(tmp_path):
    assert BlacklistCommand().execute(_args(tmp_path, add=["SRR1"])) == 1


def test_remove_clears_and_rewrites_file(tmp_path):
    BlacklistCommand().execute(_args(tmp_path, add=["SRR1", "SRR2"], reason="test"))
    assert BlacklistCommand().execute(_args(tmp_path, remove=["SRR1"])) == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    assert data["datasets"]["SRR1"]["exclusion"]["excluded"] is False
    assert (tmp_path / "blacklist.txt").read_text().splitlines() == ["SRR2  # test"]


def test_from_file_and_list(tmp_path, capsys):
    (tmp_path / "bad.txt").write_text("SRR5\n# comment\nSRR6\n")
    assert BlacklistCommand().execute(_args(tmp_path, from_file=str(tmp_path / "bad.txt"), reason="host-dominated")) == 0
    assert BlacklistCommand().execute(_args(tmp_path, list=True)) == 0
    out = capsys.readouterr().out
    assert "SRR5" in out and "host-dominated" in out


def test_registered():
    from metaquest.cli.main import create_parser

    args = create_parser().parse_args(["blacklist", "--list"])
    assert args.list is True and args.blacklist_file == "blacklist.txt"
```

And in `tests/test_data_sra.py`:

```python
    def test_blacklist_reader_ignores_inline_reasons_and_comment_lines(self, tmp_path):
        from metaquest.data.sra import _read_blacklist_files

        bl = tmp_path / "blacklist.txt"
        bl.write_text("# written by metaquest blacklist\nSRR1  # amplicon\nSRR2\n\n")
        assert _read_blacklist_files([bl]) == {"SRR1", "SRR2"}
```

- [ ] **Step 2: Implement**

```python
"""CLI command that records datasets excluded on purpose, with a reason."""

import argparse
from pathlib import Path
from typing import Dict, List

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.registry import clear_exclusion, load_registry, query, record_exclusion, save_registry


def read_blacklist_file(path: Path) -> Dict[str, str]:
    """Accession -> reason from a blacklist file (``ACC  # reason`` lines; comments and blanks skipped)."""
    entries: Dict[str, str] = {}
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        accession, _, reason = text.partition("#")
        entries[accession.strip()] = reason.strip()
    return entries


def write_blacklist_file(path: Path, entries: Dict[str, str]) -> None:
    path.write_text("".join(f"{acc}  # {reason}\n" if reason else f"{acc}\n" for acc, reason in sorted(entries.items())))


class BlacklistCommand(BaseCommand):
    """Exclude datasets from downloads and extraction, recording why."""

    @property
    def name(self) -> str:
        return "blacklist"

    @property
    def help(self) -> str:
        return "Record datasets to exclude, with a reason; keeps blacklist.txt and the registry in step"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument("--add", nargs="+", metavar="ACCESSION", help="Accessions to exclude")
        action.add_argument("--remove", nargs="+", metavar="ACCESSION", help="Accessions to allow again")
        action.add_argument("--from-file", help="File of accessions to exclude, one per line")
        action.add_argument("--list", action="store_true", help="Show the excluded accessions and reasons")
        parser.add_argument("--reason", default=None, help="Why the accessions are excluded (required with --add and --from-file)")
        parser.add_argument("--blacklist-file", default="blacklist.txt", help="Plain list kept for download_sra --blacklist")
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            blacklist_path = Path(args.blacklist_file)
            entries = read_blacklist_file(blacklist_path)
            if args.list:
                for acc in query(registry, "excluded"):
                    print(f"{acc}\t{registry.datasets[acc]['exclusion'].get('reason', '')}")
                return 0
            if args.remove:
                for acc in args.remove:
                    clear_exclusion(registry, acc)
                    entries.pop(acc, None)
                self.logger.info("Removed %d accession(s) from the blacklist", len(args.remove))
            else:
                if not args.reason:
                    raise MetaQuestError("--reason is required when adding to the blacklist")
                accessions: List[str] = args.add or _read_plain_list(Path(args.from_file))
                for acc in accessions:
                    record_exclusion(registry, acc, args.reason)
                    entries[acc] = args.reason
                self.logger.info("Excluded %d accession(s): %s", len(accessions), args.reason)
            write_blacklist_file(blacklist_path, entries)
            save_registry(registry)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error updating the blacklist: %s", e)
            return 1


def _read_plain_list(path: Path) -> List[str]:
    if not path.exists():
        raise MetaQuestError(f"Accession file not found: {path}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
```

Register in `metaquest/cli/main.py` (`from metaquest.cli.commands.blacklist import BlacklistCommand`; instance after `SelectDatasetsCommand()`), add `"blacklist"` to `expected_commands`.

In `metaquest/data/sra.py:_read_blacklist_files`, replace the line loop body so that `accession = line.split("#", 1)[0].strip()` and blank results are skipped; `download_sra --blacklist blacklist.txt` then reads the file this command writes.

- [ ] **Step 3: Verify and commit**

Run: `pytest tests/test_cli_blacklist.py tests/test_cli_main.py -v && make check && make test`

```bash
git checkout -b feat/registry-recording feat/registry-core
git add metaquest/cli/commands/blacklist.py metaquest/cli/main.py metaquest/data/sra.py tests/test_cli_blacklist.py tests/test_cli_main.py tests/test_data_sra.py
git commit -m "feat: add blacklist command recording exclusions with reasons

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 5: Screening and selection are recorded

**Files:**
- Modify: `metaquest/cli/commands/branchwater_search.py`, `metaquest/cli/commands/containment.py` (`ParseContainmentCommand.execute`), `metaquest/cli/commands/select.py`
- Test: `tests/test_cli_branchwater_search.py`, `tests/test_cli_commands.py` (parse_containment), `tests/test_cli_select.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli_select.py`:

```python
def test_selection_is_recorded_in_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    rc = SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A", threshold=0.5, registry=None))
    assert rc == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    sel = data["datasets"]["SRR1"]["selection"]
    assert sel["selected"] is True and sel["criteria"]["column"] == "GCF_A" and sel["criteria"]["threshold"] == 0.5
    assert "SRR2" not in data["datasets"] or not data["datasets"]["SRR2"]["selection"]["selected"]
```

(`_args` gains `registry=None`.) `tests/test_cli_branchwater_search.py`: in `test_fasta_default_output_path` assert afterwards that `metaquest_registry.json` in `tmp_path` holds `SRR1` with `screening.genomes["GCF_000008025.1"]["containment"] == 0.9` and `source == "branchwater"`. `tests/test_cli_commands.py` parse_containment test: after executing on a matches folder with one CSV, the registry holds every accession under `screening.genomes[<csv stem>]` with `source == "matches"`.

- [ ] **Step 2: Implement**

Each command gets `parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")` and, after its main work:

`branchwater_search.py`:

```python
            registry = load_registry(args.registry)
            for accession, containment in matches:
                record_screening(registry, accession, source.stem, containment, containment ** (1 / KSIZE) if containment > 0 else 0.0, "branchwater", args.threshold, output)
            save_registry(registry)
```

(`KSIZE` from `metaquest.data.branchwater_search`; skip recording when `matches` is empty but still create the registry so `status` finds it.)

`containment.py` `ParseContainmentCommand.execute` after `parse_containment_data(...)`: read `parsed_containment_file` with `pd.read_csv(sep="\t", index_col=0)`, and for every genome column (all columns except `max_containment`, `max_containment_annotation`) and every row with a numeric value `> 0`, `record_screening(registry, acc, column, value, None, "matches", 0.0, Path(args.matches_folder) / f"{column}.csv")`; then `save_registry`. Put this in a helper `_record_screening_from_table(registry, table_path, matches_folder)` in `metaquest/data/registry.py` (test it there too: `test_record_screening_from_table`).

`select.py` after writing the file:

```python
            registry = load_registry(args.registry)
            record_selection(
                registry,
                accessions,
                {
                    "column": args.genome_id or "max_containment",
                    "threshold": args.threshold,
                    "metadata_column": args.metadata_column,
                    "metadata_value": args.metadata_value,
                    "table": str(args.parsed_containment),
                },
                output,
            )
            save_registry(registry)
```

- [ ] **Step 3: Verify and commit**

Run: `pytest tests/test_cli_select.py tests/test_cli_branchwater_search.py tests/test_cli_commands.py tests/test_data_registry.py -v && make check && make test`

```bash
git add metaquest tests
git commit -m "feat: record screening and selection in the registry

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 6: Downloads are recorded as they finish

**Files:**
- Modify: `metaquest/data/sra.py` (`_execute_parallel_downloads`, `_retry_failed_downloads`, `download_sra`, `_handle_download_failure`), `metaquest/cli/commands/sra.py`
- Test: `tests/test_data_sra.py`, `tests/test_cli_commands.py`

**Interfaces:**
- `download_sra(..., blacklist_accessions: Optional[Set[str]] = None, on_result: Optional[Callable[[str, bool, str], None]] = None)`; the returned dict gains `skipped_accessions` (cut by `--max-downloads`).
- `_execute_parallel_downloads(..., on_result=None)` calls `on_result(accession, success, message)` on the main thread inside the `as_completed` loop; the retry pass calls it again with the final outcome.
- `failed_accessions.txt` is written only by `metaquest/data/sra.py:_handle_download_failure`, using `FAILED_ACCESSIONS_FILE`; the CLI's `_report_failed_downloads` only logs.

- [ ] **Step 1: Write the failing tests**

`tests/test_data_sra.py`:

```python
    def test_on_result_called_on_main_thread_per_accession(self, tmp_path, monkeypatch):
        import threading

        seen = []
        main = threading.get_ident()

        def on_result(acc, ok, message):
            seen.append((acc, ok, threading.get_ident() == main))

        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        with patch("metaquest.data.sra.download_accession", side_effect=[(True, "Downloaded 2 files"), (False, "Download failed: x")]):
            stats = download_sra(tmp_path / "fastq", acc, max_workers=2, max_retries=0, on_result=on_result)
        assert sorted(a for a, _, _ in seen) == ["SRR1", "SRR2"] and all(on_main for _, _, on_main in seen)
        assert stats["failed_accessions"] == ["SRR2"] or stats["failed_accessions"] == ["SRR1"]

    def test_max_downloads_cutoffs_are_returned(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\nSRR3\n")
        with patch("metaquest.data.sra.download_accession", return_value=(True, "Downloaded 2 files")):
            stats = download_sra(tmp_path / "fastq", acc, max_downloads=1)
        assert len(stats["skipped_accessions"]) == 2

    def test_registry_exclusions_join_the_blacklist(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        stats = download_sra(tmp_path / "fastq", acc, dry_run=True, blacklist_accessions={"SRR2"})
        assert stats["blacklisted_accessions"] == ["SRR2"]
```

`tests/test_cli_commands.py` (`TestDownloadSraCommand`): a test that patches `metaquest.cli.commands.sra.shutil.which` to a path and `metaquest.cli.commands.sra.download_sra` with a fake that calls `on_result("SRR1", True, "Downloaded 2 files")` and returns a stats dict with `already_downloaded_accessions=["SRR3"]`, `blacklisted_accessions=["SRR4"]`, `skipped_accessions=["SRR5"]`, `failed_accessions=["SRR2"]`, `results={"SRR1": "...", "SRR2": "Download failed: t"}`; assert the registry file next to the FASTQ folder holds states downloaded (SRR1, with files listed if you create `fastq/SRR1/SRR1_1.fastq`), failed (SRR2, attempts 1), and skipped (SRR5, message "--max-downloads"); `already_present` accessions are recorded as `downloaded` with `inferred` absent. Also assert `failed_accessions.txt` exists exactly once (written by the data layer) and the CLI did not write it (patch `Path.write_text`? simpler: the fake `download_sra` writes it and the test checks the CLI leaves the content unchanged).

- [ ] **Step 2: Implement**

`metaquest/data/sra.py`: thread `on_result` from `download_sra` into `_download_with_retries` -> `_execute_parallel_downloads` and `_retry_failed_downloads`; call `on_result(acc, success, message)` right after each future result is appended (and in the retry pass); compute `skipped_accessions = accessions_to_download[max_downloads:]` before truncation and return it (empty list when no limit); union `blacklist_accessions` into the set read from files before `_check_existing_downloads`; keep the `failed_accessions.txt` writer but use `FAILED_ACCESSIONS_FILE`.

`metaquest/cli/commands/sra.py`: add `--registry`; in `execute`, before calling `download_sra`:

```python
            registry = load_registry(args.registry)
            excluded = set(query(registry, "excluded"))
            fastq_dir = Path(args.fastq_folder)

            def on_result(accession: str, success: bool, message: str) -> None:
                record_download(registry, accession, "downloaded" if success else "failed", fastq_dir, message)
                save_registry(registry)
```

pass `blacklist_accessions=excluded, on_result=on_result`; after the run record `already_downloaded_accessions` as `downloaded` (only if not already recorded as downloaded), `blacklisted_accessions` as `skipped` with message `"blacklisted"`, `skipped_accessions` as `skipped` with message `"--max-downloads"`, then `save_registry` once. In `_report_failed_downloads` delete the file write; keep the log lines.

- [ ] **Step 3: Verify and commit**

Run: `pytest tests/test_data_sra.py tests/test_cli_commands.py -v && make check && make test`

```bash
git add metaquest/data/sra.py metaquest/cli/commands/sra.py tests/test_data_sra.py tests/test_cli_commands.py
git commit -m "feat: record download outcomes in the registry as they complete

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 7: Metadata and analyses are recorded; dashboards read saved profiles

**Files:**
- Modify: `metaquest/cli/commands/metadata.py` (`DownloadMetadataCommand`, `ParseMetadataCommand`), `metaquest/data/metadata.py` (`download_metadata` gains `accessions_file`), `metaquest/cli/commands/sra_enhanced.py` (`SRAStatsCommand`, `SRAValidateCommand`), `metaquest/cli/commands/sra_intelligent.py` (`SRAQualityProfileCommand`, `SRAInteractiveDashboardCommand`)
- Test: `tests/test_cli_commands.py`, `tests/test_data_metadata*.py`, `tests/test_cli_commands_sra_enhanced.py`, `tests/test_cli_sra_intelligent.py`

- [ ] **Step 1: Write the failing tests**

- `download_metadata --accessions-file`: with the file given, the wanted set is the file's accessions (the matches folder is not read); test with `_download_single_metadata` patched, assert it is called for exactly those accessions and that each downloaded XML is recorded (`registry.datasets[acc]["metadata"]["xml"]`).
- `parse_metadata`: after parsing a folder with one XML that yields `Run_Size`/`Run_MD5` (use the existing XML fixture in the metadata tests), the registry holds `run_size` and `run_md5` for that accession.
- `sra_stats`: after a run with `generate_statistics_report` patched to write a two-row CSV, each accession has `analyses.sra_stats.output` pointing at the report and `summary.total_reads` from the CSV row.
- `sra_validate`: after a run over a tmp FASTQ tree, each validated accession has `analyses.validate.summary == {"passed": True|False, "files": n}`.
- `sra_profile_quality`: each profiled accession has `analyses.quality.summary` with `grade`, `total_reads`, `gc_content`, and `output` = the summary JSON (or the per-accession JSON when `--detailed-reports`).
- `sra_dashboard --quality-profiles DIR`: with `<DIR>/<acc>_quality_profile.json` present, `profile_dataset_quality` is not called and the dashboard is built from the saved profile (patch `SRAReportGenerator.generate_quality_dashboard` and assert the loader path via a new `load_quality_profiles(dir) -> Dict[str, QualityProfile]` in `metaquest/sra/analytics.py` that the command calls).

- [ ] **Step 2: Implement**

Add `--registry` to the five commands. `download_metadata(..., accessions_file=None)`: when given, `all_accessions` comes from the file (reuse `_read_accession_list` pattern) instead of `_get_unique_accessions`. Record after each successful `_download_single_metadata` by returning the list of downloaded accessions from `download_metadata` (add `downloaded: List[str]` to its return, keeping existing keys) and calling `record_metadata(registry, acc, metadata_folder / f"{acc}_metadata.xml", {})` in the CLI. `parse_metadata` returns its DataFrame already; the CLI iterates rows and calls `record_metadata` with `run_size`, `run_md5`, `assay_type` (from `Experiment_Library_Strategy` if present), `organism` (`Sample_Scientific_Name`), `collection_date` when those columns exist. `sra_stats`: after `generate_statistics_report`, read the CSV with `read_records` and `record_analysis(registry, row["accession"], "sra_stats", report_path, {"total_reads": ..., "gc_content": ..., "avg_read_length": ...})`. `sra_validate`: `record_analysis(..., "validate", "", {"passed": bool, "files": int})` per accession. `sra_profile_quality`: `record_analysis(..., "quality", <json path>, {"grade": profile.quality_grade, "total_reads": profile.total_reads, "gc_content": profile.gc_content})`. `sra_dashboard`: implement `load_quality_profiles` (reads each `*_quality_profile.json` into a `QualityProfile`) and, when `--quality-profiles` is given and a profile exists, pass the loaded profiles to the reporter (add an optional `profiles` argument to `generate_quality_dashboard` that bypasses `profile_dataset_quality`).

- [ ] **Step 3: Verify and commit**

Run the listed test files, `make check && make test`.

```bash
git add metaquest tests
git commit -m "feat: record metadata and analyses in the registry; dashboards reuse saved profiles

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `feat/registry-recording` -> `main`.

---

## Phase 3: extraction and assembly (branch `feat/extraction-idempotent`, based on `feat/registry-recording`)

### Task 8: Extraction and assembly records, skip and `--force`

**Files:**
- Modify: `metaquest/data/read_extraction.py` (`extract_target_reads` returns `Dict[str, ExtractionResult]`, new `force` and `skip` parameters, `assemble_extracted_reads` handles an existing folder), `metaquest/cli/commands/read_extraction.py`, `tests/helpers_extraction.py`
- Test: `tests/test_read_extraction.py`, `tests/test_cli_read_extraction.py`

**Interfaces:**
- `extract_target_reads(..., force: bool = False, already_done: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, ExtractionResult]`: `already_done` maps accession -> the registry's extraction record for this genome; a sample is skipped (returned as `ExtractionResult(files=[Path(p) for p in record["files"]], mapped_records=record["mapped_reads"], unequal_mates=record["unequal_mates"], skipped=True)`) when `force` is False, the record's `genome_fasta`, `preset` and `threshold` equal the current call, and every recorded file exists (or `mapped_reads == 0`).
- `ExtractionResult` gains `skipped: bool = False`.
- `assemble_extracted_reads(reads, output_dir, threads=4, min_contig_len=None, force=False) -> Path`: if `output_dir/final.contigs.fa` exists and not `force`, log "already assembled" and return without running megahit; if the folder exists without contigs and not `force`, raise `ProcessingError("Assembly folder exists but holds no contigs (interrupted run?); rerun with --force")`; with `force`, `shutil.rmtree` first.
- `megahit_version() -> str` in `read_extraction.py`: `run_secure("megahit", ["--version"]).stdout.strip()` or `""` on any error.
- CLI: `--force`, `--registry`; records `record_extraction` for every processed sample (including zero-mapped) and `record_assembly` after each assembly with `summarise_contigs` stats, the megahit version, and `{"threads": asm_threads, "min_contig_len": ...}`; uses the registry's records as `already_done`.

- [ ] **Step 1: Write the failing tests**

`tests/test_read_extraction.py`:

```python
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_recorded_extraction_is_skipped_unless_forced(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            first = extract_target_reads(parsed_containment=table, genome_id="GCF_1", genome_fasta=genome,
                                         fastq_folder=root / "fastq", output_folder=root / "targeted", threshold=0.5)
            record = {"SRR1": {"genome_fasta": str(genome), "preset": "sr", "threshold": 0.5,
                               "mapped_reads": first["SRR1"].mapped_records, "unequal_mates": False,
                               "files": [str(p) for p in first["SRR1"].files]}}
            calls_before = len(state["calls"])
            again = extract_target_reads(parsed_containment=table, genome_id="GCF_1", genome_fasta=genome,
                                         fastq_folder=root / "fastq", output_folder=root / "targeted", threshold=0.5,
                                         already_done=record)
            assert again["SRR1"].skipped is True and len(state["calls"]) == calls_before
            forced = extract_target_reads(parsed_containment=table, genome_id="GCF_1", genome_fasta=genome,
                                          fastq_folder=root / "fastq", output_folder=root / "targeted", threshold=0.5,
                                          already_done=record, force=True)
            assert forced["SRR1"].skipped is False and len(state["calls"]) > calls_before

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_assembly_skips_existing_contigs_and_refuses_empty_dir(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asm"
            out.mkdir()
            (out / "final.contigs.fa").write_text(">c len=4\nACGT\n")
            assemble_extracted_reads([Path(tmp) / "r1.fq.gz", Path(tmp) / "r2.fq.gz"], out)
            assert not mock_run.called
            (out / "final.contigs.fa").unlink()
            with pytest.raises(ProcessingError, match="rerun with --force"):
                assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out)
            assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out, force=True)
            assert mock_run.called
```

`tests/test_cli_read_extraction.py`: `test_execute_records_extraction_and_assembly` (registry file next to the output folder holds `extractions["GCF_1"]["mapped_reads"]` and, with `assemble=True` and a fake megahit that writes `final.contigs.fa` (extend `_fake_tools` so a megahit call creates `<-o dir>/final.contigs.fa` with two `len=` headers), `assembly["contigs"] == 2`); `test_second_run_skips_and_returns_0` (second execute makes no `run_secure` calls for minimap2/samtools; `--force` makes them again).

- [ ] **Step 2: Implement**

In `read_extraction.py`: add `skipped: bool = False` to `ExtractionResult`; add `force` and `already_done` parameters; before `_map_and_extract`, `record = (already_done or {}).get(accession)`; if `record and not force and _record_matches(record, genome_path, preset, threshold)` then build the skipped result and `logger.info("%s already extracted against %s (%d mapped reads); use --force to redo", ...)`. `_record_matches` compares `str(genome_path)`, preset, float threshold and checks `all(Path(p).exists() for p in record["files"])` or `record["mapped_reads"] == 0`. Change `results[accession] = outcome` (whole result) and the dry-run entry to `ExtractionResult([], 0)`. In `assemble_extracted_reads` add the existing-folder handling described above (`import shutil`). Add `megahit_version()`.

In the CLI: `--force`, `--registry`; load the registry, build `already_done = {acc: rec for acc in selected if (rec := extraction_record(registry, acc, args.genome_id))}`; pass `force` and `already_done`; `with_reads = {acc: r.files for acc, r in results.items() if r.files}`; after extraction, `record_extraction` for every accession in `results` that was not skipped; after each assembly, `record_assembly(registry, acc, genome_id, out_dir, summarise_contigs(out_dir / "final.contigs.fa"), version, params)`; save once at the end (and once after extraction so a crash in assembly keeps the extraction records). Update `tests/helpers_extraction.py` so the megahit stub writes `final.contigs.fa` under the `-o` folder.

- [ ] **Step 3: Verify, pipeline, commit**

Run: `pytest tests/test_read_extraction.py tests/test_cli_read_extraction.py tests/test_data_registry.py -v && make check && make test && make pipeline`

```bash
git checkout -b feat/extraction-idempotent feat/registry-recording
git add metaquest tests
git commit -m "feat: record extraction and assembly results; skip work already done unless --force

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `feat/extraction-idempotent` -> `main`.

---

## Phase 4: docs and walkthrough (branch `docs/dataset-tracking`, based on `feat/extraction-idempotent`)

### Task 9: Documentation, walkthrough, gitignore, planning docs

**Files:**
- Modify: `README.md`, `docs/ARCHITECTURE.md`, `docs/branchwater_workflow.md`, `CLAUDE.md`, `AGENTS.md` (untracked, edit only), `local_test.sh`, `.gitignore`

- [ ] **Step 1: README**

Add a section "Project state" after "Checking What Is Already Available Locally":

```markdown
### Project state

MetaQuest keeps a journal of every dataset a project touches in `metaquest_registry.json` in the
project root: which genomes it was screened against and with what containment, whether it was
selected (and by which threshold and filter), whether it is excluded and why, the download outcome
with file sizes and dates, which analyses ran, and for each target genome the number of mapped reads
and the assembly statistics. Commands update it as they finish; `status` reads it and always
re-checks the disk, so a deleted folder shows up as missing rather than done.

```bash
metaquest status --init                      # create the registry from an existing project
metaquest status                             # accession by stage matrix, per genome
metaquest status --stage extracted --genome GCF_000008025.1
metaquest status --next                      # which commands would advance the most datasets
metaquest status --reconcile                 # record files removed by hand, list untracked work
metaquest status --export-tsv registry       # registry_datasets.tsv and registry_extractions.tsv
metaquest blacklist --add SRR2517418 --reason "16S amplicon mislabelled as WGS"
```

`extract_target_reads` skips samples already extracted or assembled with the same genome, preset
and threshold; pass `--force` to redo them. Commit `metaquest_registry.json` with your project if
you want the decisions to travel with the results.
```

Update the `status` section to mention the new flags, the `download_sra` section to mention that
registry exclusions are honoured, and the `extract_target_reads` section to mention `--force`.

- [ ] **Step 2: Other docs and the walkthrough**

`docs/ARCHITECTURE.md`: a "Project registry" paragraph under the data layer describing `metaquest/data/registry.py` as the journal, the record functions, and the rule that presence is re-checked on disk. `docs/branchwater_workflow.md`: one line per step naming what the registry records. `CLAUDE.md`/`AGENTS.md`: add `blacklist` to the command groups and a line on the registry under Data Processing Pipeline. `.gitignore`: add the comment `# metaquest_registry.json is meant to be committed with a project` (no ignore line) and ignore `metaquest_registry.json.lock` and `metaquest_registry.json.tmp.*`. `local_test.sh`: after `select_datasets` add `metaquest status --init`, `metaquest blacklist --add SRR31320538 --reason "walkthrough example"`, `metaquest status --stage excluded`, and after the dry-run extraction a second `metaquest status --json` whose `stages.selected.count` is asserted to be greater than 0 with a small Python one-liner.

- [ ] **Step 3: Verify and commit**

Run: `make check && make test && make pipeline` and `grep -n "metaquest_registry" README.md docs/ARCHITECTURE.md docs/branchwater_workflow.md CLAUDE.md .gitignore local_test.sh` (every file hits).

```bash
git checkout -b docs/dataset-tracking feat/extraction-idempotent
git add README.md docs/ARCHITECTURE.md docs/branchwater_workflow.md CLAUDE.md .gitignore local_test.sh docs/superpowers/specs/2026-09-05-dataset-tracking-design.md docs/superpowers/plans/2026-09-05-dataset-tracking-implementation.md
git commit -m "docs: describe the project registry, status stages and the blacklist command

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `docs/dataset-tracking` -> `main`.

---

## Verification after all four branches merge

- `make check`, `make test`, `make pipeline` green on `main`; `make test-network` passes.
- Real data (scratch e2e folder from the audit, 10 Wolbachia runs): `metaquest status --init` then `status` shows 3 extracted vs GCF_000008025.1 (2 with 0 mapped reads), 5 without metadata, `SRR2517620` listed under drift as untracked; `blacklist --add SRR2517418 --reason "16S amplicon mislabelled as WGS"` removes it from `status --next`; a second `extract_target_reads --assemble` against wMel skips all three samples; `--force` rebuilds them and the registry's assembly block carries the new date and the same contig count.
