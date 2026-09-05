# Follow-up Implementation Plan: Branchwater search, single download command, taxonomy tables, CLI polish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the pipeline start from a genome FASTA (`branchwater_search`), leave one download command, make `taxonomic_summary` accept both taxonomy tables, and tidy the CLI (grouped help, hidden aliases, one threshold rule, no stubs or dead code).

**Architecture:** Four stacked branches off local `main` (ee0238e), each one PR: `feat/branchwater-search` (Tasks 1-2), `refactor/single-download-command` (Tasks 3-6), `feat/taxonomy-table-unification` (Task 7), `refactor/cli-polish` (Tasks 8-11). New code lives in small modules with one responsibility (`metaquest/data/branchwater_search.py`, one CLI module per command); removals delete whole units and their tests rather than leaving stubs. Every task is red-green with `requests`, `subprocess.run` or `SecureSubprocess.run_secure` patched.

**Tech Stack:** Python 3.12, argparse command registry (`metaquest/cli/base.py`), pandas, Biopython `SeqIO`, sourmash Python API (optional extra `metaquest[sourmash]`), `requests`, pytest with `unittest.mock`; black, flake8, mypy, radon via `make check`.

**Spec:** `docs/superpowers/specs/2026-09-05-followup-design.md`

## Global Constraints

- Python >= 3.12; black line length 120; `make check` (black, flake8, mypy, radon ceiling, CDN guard) and `make test` must pass after every task; `make pipeline` after Tasks 2, 6, 7 and 11.
- CLI flags use dashes. Plain, modest language in help, logs and docs; no Unicode symbols in new text.
- Unit tests never touch the network or external tools: patch `metaquest.data.branchwater_search.requests.post`, `metaquest.utils.security.subprocess.run` or `SecureSubprocess.run_secure`. sourmash-dependent tests call `pytest.importorskip("sourmash")`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Do not `git add` the untracked `AGENTS.md`; `docs/superpowers/` may be committed in Task 11 (Step 6) only.

---

## Phase A: `branchwater_search` (branch `feat/branchwater-search`)

### Task 1: Branchwater search module

**Files:**
- Create: `metaquest/data/branchwater_search.py`
- Create: `tests/test_branchwater_search.py`

**Interfaces:**
- Produces: `DEFAULT_SERVER`, `KSIZE`, `SCALED`, `BRANCHWATER_COLUMNS`, `sketch_fasta(fasta_path) -> Dict[str, Any]`, `load_signature(sig_path) -> Dict[str, Any]`, `search_index(signature, threshold, server=DEFAULT_SERVER, timeout=600) -> List[Tuple[str, float]]`, `write_branchwater_csv(matches, output_path, ksize=KSIZE) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_branchwater_search.py`:

```python
"""Tests for metaquest.data.branchwater_search (no network; sourmash optional)."""

import json
import random
from unittest.mock import Mock, patch

import pytest
import requests

from metaquest.core.exceptions import DataAccessError
from metaquest.data.branchwater_search import (
    BRANCHWATER_COLUMNS,
    DEFAULT_SERVER,
    load_signature,
    search_index,
    sketch_fasta,
    write_branchwater_csv,
)

SIG_OBJECT = {
    "class": "sourmash_signature",
    "name": "wmel",
    "signatures": [{"num": 0, "ksize": 21, "seed": 42, "max_hash": 18446744073709552, "mins": [1, 2, 3], "molecule": "DNA"}],
}


def _write_fasta(path, seed=7, length=5000):
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGT") for _ in range(length))
    path.write_text(f">contig1 test\n{seq[:2500]}\n>contig2\n{seq[2500:]}\n")


class TestSketchFasta:
    def test_builds_k21_scaled1000_signature(self, tmp_path):
        pytest.importorskip("sourmash")
        fasta = tmp_path / "GCF_000000001.1.fna"
        _write_fasta(fasta)
        signature = sketch_fasta(fasta)
        sketch = signature["signatures"][0]
        assert signature["name"] == "GCF_000000001.1"
        assert sketch["ksize"] == 21
        assert round(2**64 / sketch["max_hash"]) == 1000
        assert len(sketch["mins"]) > 0

    def test_missing_fasta_raises(self, tmp_path):
        pytest.importorskip("sourmash")
        with pytest.raises(DataAccessError, match="not found"):
            sketch_fasta(tmp_path / "missing.fna")

    def test_empty_fasta_raises(self, tmp_path):
        pytest.importorskip("sourmash")
        fasta = tmp_path / "empty.fna"
        fasta.write_text("")
        with pytest.raises(DataAccessError, match="No sequences"):
            sketch_fasta(fasta)

    def test_missing_sourmash_gives_install_hint(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sourmash"):
                raise ImportError("no sourmash")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        fasta = tmp_path / "x.fna"
        _write_fasta(fasta)
        with pytest.raises(DataAccessError, match=r"metaquest\[sourmash\]"):
            sketch_fasta(fasta)


class TestLoadSignature:
    def test_list_form(self, tmp_path):
        path = tmp_path / "wmel.sig"
        path.write_text(json.dumps([SIG_OBJECT]))
        assert load_signature(path)["name"] == "wmel"

    def test_object_form(self, tmp_path):
        path = tmp_path / "wmel.sig"
        path.write_text(json.dumps(SIG_OBJECT))
        assert load_signature(path)["signatures"][0]["ksize"] == 21

    def test_wrong_parameters_raise(self, tmp_path):
        bad = json.loads(json.dumps(SIG_OBJECT))
        bad["signatures"][0]["ksize"] = 31
        path = tmp_path / "k31.sig"
        path.write_text(json.dumps([bad]))
        with pytest.raises(DataAccessError, match="k=21, scaled=1000"):
            load_signature(path)

    def test_missing_or_invalid(self, tmp_path):
        with pytest.raises(DataAccessError, match="not found"):
            load_signature(tmp_path / "none.sig")
        path = tmp_path / "bad.sig"
        path.write_text("{not json")
        with pytest.raises(DataAccessError, match="not valid JSON"):
            load_signature(path)


class TestSearchIndex:
    @patch("metaquest.data.branchwater_search.requests.post")
    def test_posts_signature_and_parses_csv(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text="SRA accession,containment\nSRR1,0.5\nSRR2,0.9\n")
        matches = search_index(SIG_OBJECT, 0.1)
        assert matches == [("SRR2", 0.9), ("SRR1", 0.5)]
        args, kwargs = mock_post.call_args
        assert args[0] == f"{DEFAULT_SERVER}/search"
        assert kwargs["json"] == {"threshold": 0.1, "signature": SIG_OBJECT}
        assert kwargs["timeout"] == 600

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_http_error_raises(self, mock_post):
        mock_post.return_value = Mock(status_code=500, text="boom")
        with pytest.raises(DataAccessError, match="HTTP 500"):
            search_index(SIG_OBJECT, 0.1)

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_connection_error_raises(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("down")
        with pytest.raises(DataAccessError, match="search failed"):
            search_index(SIG_OBJECT, 0.1, server="https://example.org/")

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_unexpected_header_raises(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text="<html>redirect</html>")
        with pytest.raises(DataAccessError, match="Unexpected Branchwater response"):
            search_index(SIG_OBJECT, 0.1)

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_empty_result(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text="SRA accession,containment\n")
        assert search_index(SIG_OBJECT, 0.1) == []


class TestWriteBranchwaterCsv:
    def test_header_and_cani(self, tmp_path):
        out = write_branchwater_csv([("SRR11011981", 0.9461), ("SRR2", 0.0)], tmp_path / "bw" / "wmel.csv")
        lines = out.read_text().splitlines()
        assert lines[0] == ",".join(BRANCHWATER_COLUMNS)
        assert lines[1] == "SRR11011981,0.9461,0.9974,,,,,,,"
        assert lines[2] == "SRR2,0.0000,0.0000,,,,,,,"

    def test_empty_matches_write_header_only(self, tmp_path):
        out = write_branchwater_csv([], tmp_path / "empty.csv")
        assert out.read_text().splitlines() == [",".join(BRANCHWATER_COLUMNS)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_branchwater_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metaquest.data.branchwater_search'`.

- [ ] **Step 3: Implement the module**

Create `metaquest/data/branchwater_search.py`:

```python
"""Query the Branchwater index of SRA metagenomes with a genome sketch.

The public Branchwater service (https://branchwater.sourmash.bio) indexes about
1.1 million SRA metagenomes as FracMinHash sketches (k=21, scaled=1000). Its
search API takes one sourmash signature and a minimum containment and returns
the matching SRA accessions with their containment. This module builds the
sketch, runs the search and writes the result in the CSV layout the rest of
MetaQuest reads (see ``use_branchwater``). The metadata columns are left empty;
``download_metadata`` fills them from NCBI.
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import requests

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "https://api.branchwater.sourmash.bio"
KSIZE = 21
SCALED = 1000
BRANCHWATER_COLUMNS = [
    "acc",
    "containment",
    "cANI",
    "biosample",
    "bioproject",
    "assay_type",
    "collection_date_sam",
    "geo_loc_name_country_calc",
    "organism",
    "lat_lon",
]
SOURMASH_HINT = "sourmash is required to sketch a genome. Install it with: pip install 'metaquest[sourmash]'"


def sketch_fasta(fasta_path: Union[str, Path]) -> Dict[str, Any]:
    """Build the k=21, scaled=1000 sourmash signature Branchwater expects for one FASTA file.

    Returns the signature as the JSON object sourmash writes (one element of a
    ``.sig`` file), ready to be posted to the search API.

    Raises:
        DataAccessError: If sourmash is not installed, the file is missing, or it holds no sequences.
    """
    try:
        from sourmash import MinHash, SourmashSignature
        from sourmash.signature import save_signatures_to_json
    except ImportError as e:
        raise DataAccessError(SOURMASH_HINT) from e
    from Bio import SeqIO

    path = Path(fasta_path)
    if not path.exists():
        raise DataAccessError(f"Genome FASTA not found: {path}")

    minhash = MinHash(n=0, ksize=KSIZE, scaled=SCALED)
    n_records = 0
    for record in SeqIO.parse(str(path), "fasta"):
        minhash.add_sequence(str(record.seq).upper(), force=True)
        n_records += 1
    if n_records == 0:
        raise DataAccessError(f"No sequences found in {path}")

    signature = SourmashSignature(minhash, name=path.stem, filename=path.name)
    buffer = io.BytesIO()
    save_signatures_to_json([signature], buffer)
    payload = json.loads(buffer.getvalue().decode("utf-8"))
    logger.info(
        "Sketched %s: %d sequence(s), %d hashes (k=%d, scaled=%d)", path.name, n_records, len(minhash.hashes), KSIZE, SCALED
    )
    return payload[0]


def load_signature(sig_path: Union[str, Path]) -> Dict[str, Any]:
    """Read one sourmash JSON signature (list or single object) and check its sketch parameters."""
    path = Path(sig_path)
    if not path.exists():
        raise DataAccessError(f"Signature file not found: {path}")
    try:
        with open(path) as handle:
            data = json.load(handle)
    except json.JSONDecodeError as e:
        raise DataAccessError(f"Signature file is not valid JSON: {path} ({e})") from e

    signature = data[0] if isinstance(data, list) and data else data
    if not isinstance(signature, dict) or not signature.get("signatures"):
        raise DataAccessError(f"Not a sourmash signature file: {path}")
    _check_sketch_parameters(signature, path)
    return signature


def _check_sketch_parameters(signature: Dict[str, Any], path: Path) -> None:
    sketch = signature["signatures"][0]
    ksize = sketch.get("ksize")
    max_hash = sketch.get("max_hash") or 0
    scaled = round(2**64 / max_hash) if max_hash else None
    if ksize != KSIZE or scaled != SCALED:
        raise DataAccessError(f"Branchwater needs k={KSIZE}, scaled={SCALED}; {path.name} has k={ksize}, scaled={scaled}")


def search_index(
    signature: Dict[str, Any], threshold: float, server: str = DEFAULT_SERVER, timeout: int = 600
) -> List[Tuple[str, float]]:
    """Post a signature to the Branchwater search API and return (accession, containment) pairs, best first."""
    url = f"{server.rstrip('/')}/search"
    try:
        response = requests.post(url, json={"threshold": threshold, "signature": signature}, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise DataAccessError(f"Branchwater search failed: {e}") from e
    if response.status_code != 200:
        raise DataAccessError(f"Branchwater search returned HTTP {response.status_code}: {response.text[:200]}")
    return _parse_search_csv(response.text)


def _parse_search_csv(text: str) -> List[Tuple[str, float]]:
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if "SRA accession" not in fields or "containment" not in fields:
        raise DataAccessError(f"Unexpected Branchwater response header: {fields}")
    matches: List[Tuple[str, float]] = []
    for row in reader:
        try:
            matches.append((row["SRA accession"].strip(), float(row["containment"])))
        except (TypeError, ValueError, AttributeError):
            logger.warning("Skipping malformed Branchwater row: %s", row)
    matches.sort(key=lambda match: match[1], reverse=True)
    return matches


def write_branchwater_csv(
    matches: List[Tuple[str, float]], output_path: Union[str, Path], ksize: int = KSIZE
) -> Path:
    """Write matches in Branchwater CSV layout; cANI is derived from containment, metadata columns stay empty."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    empty_metadata = [""] * (len(BRANCHWATER_COLUMNS) - 3)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BRANCHWATER_COLUMNS)
        for accession, containment in matches:
            cani = round(containment ** (1 / ksize), 4) if containment > 0 else 0.0
            writer.writerow([accession, f"{containment:.4f}", f"{cani:.4f}"] + empty_metadata)
    logger.info("Wrote %d match(es) to %s", len(matches), path)
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_branchwater_search.py -v`
Expected: all PASS (the sketch tests run because sourmash 4.9.4 is installed here).

- [ ] **Step 5: Gates and commit**

Run: `make check && make test`

```bash
git checkout -b feat/branchwater-search main
git add metaquest/data/branchwater_search.py tests/test_branchwater_search.py
git commit -m "feat: add Branchwater search module (sketch, query, Branchwater CSV)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 2: `branchwater_search` command, registration and docs

**Files:**
- Create: `metaquest/cli/commands/branchwater_search.py`, `tests/test_cli_branchwater_search.py`
- Modify: `metaquest/cli/main.py` (import and registration), `tests/test_cli_main.py:34-58` (expected list), `README.md:62-64`, `docs/branchwater_workflow.md` (new block after "Option B")

**Interfaces:**
- Consumes: Task 1 functions.
- Produces: command `branchwater_search`; class `BranchwaterSearchCommand`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_branchwater_search.py`:

```python
"""Tests for the branchwater_search CLI command."""

import argparse
from pathlib import Path
from unittest.mock import patch

from metaquest.cli.commands.branchwater_search import BranchwaterSearchCommand
from metaquest.core.exceptions import DataAccessError


def _args(**kwargs):
    base = dict(genome_fasta=None, signature=None, threshold=0.1, branchwater_folder="branchwater", output=None, server="https://s")
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestBranchwaterSearchCommand:
    def test_properties(self):
        cmd = BranchwaterSearchCommand()
        assert cmd.name == "branchwater_search"
        assert "Branchwater" in cmd.help

    def test_parser_requires_one_input(self):
        parser = argparse.ArgumentParser()
        BranchwaterSearchCommand().configure_parser(parser)
        args = parser.parse_args(["--genome-fasta", "g.fna"])
        assert args.threshold == 0.1 and args.branchwater_folder == "branchwater" and args.output is None
        try:
            parser.parse_args([])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("one of --genome-fasta/--signature must be required")

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index")
    @patch("metaquest.cli.commands.branchwater_search.sketch_fasta")
    def test_fasta_default_output_path(self, mock_sketch, mock_search, mock_write, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_sketch.return_value = {"signatures": []}
        mock_search.return_value = [("SRR1", 0.9)]
        rc = BranchwaterSearchCommand().execute(_args(genome_fasta="genomes/GCF_000008025.1.fna"))
        assert rc == 0
        mock_search.assert_called_once_with({"signatures": []}, 0.1, server="https://s")
        mock_write.assert_called_once_with([("SRR1", 0.9)], Path("branchwater") / "GCF_000008025.1.csv")

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index")
    @patch("metaquest.cli.commands.branchwater_search.load_signature")
    def test_signature_and_explicit_output(self, mock_load, mock_search, mock_write, tmp_path):
        mock_load.return_value = {"signatures": []}
        mock_search.return_value = []
        rc = BranchwaterSearchCommand().execute(_args(signature="wmel.sig", output=str(tmp_path / "out.csv")))
        assert rc == 0
        mock_write.assert_called_once_with([], Path(tmp_path / "out.csv"))

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index", return_value=[])
    @patch("metaquest.cli.commands.branchwater_search.load_signature", return_value={"signatures": []})
    def test_zero_matches_warns(self, _load, _search, _write, caplog):
        with caplog.at_level("WARNING"):
            assert BranchwaterSearchCommand().execute(_args(signature="wmel.sig")) == 0
        assert "control genome" in caplog.text

    @patch("metaquest.cli.commands.branchwater_search.load_signature", side_effect=DataAccessError("bad sig"))
    def test_error_returns_1(self, _load):
        assert BranchwaterSearchCommand().execute(_args(signature="wmel.sig")) == 1

    def test_registered(self):
        from metaquest.cli.main import create_parser, register_all_commands

        register_all_commands()
        parser = create_parser()
        action = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None))
        assert "branchwater_search" in action.choices
```

Add `"branchwater_search",` to `expected_commands` in `tests/test_cli_main.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cli_branchwater_search.py tests/test_cli_main.py -v`
Expected: FAIL with `ModuleNotFoundError` and the missing expected command.

- [ ] **Step 3: Implement the command**

Create `metaquest/cli/commands/branchwater_search.py`:

```python
"""CLI command that searches the Branchwater index with a genome and writes a Branchwater CSV."""

import argparse
from pathlib import Path

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.branchwater_search import (
    DEFAULT_SERVER,
    load_signature,
    search_index,
    sketch_fasta,
    write_branchwater_csv,
)


class BranchwaterSearchCommand(BaseCommand):
    """Search Branchwater with a genome FASTA or a sourmash signature."""

    @property
    def name(self) -> str:
        return "branchwater_search"

    @property
    def help(self) -> str:
        return "Search the Branchwater index of SRA metagenomes with a genome and write a Branchwater CSV"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--genome-fasta", help="Genome FASTA to sketch (needs the sourmash extra)")
        source.add_argument("--signature", help="Existing sourmash signature file (k=21, scaled=1000)")
        parser.add_argument("--threshold", type=float, default=0.1, help="Minimum containment reported by Branchwater")
        parser.add_argument("--branchwater-folder", default="branchwater", help="Folder for the Branchwater CSV")
        parser.add_argument("--output", default=None, help="Output CSV (default: <branchwater-folder>/<input stem>.csv)")
        parser.add_argument("--server", default=DEFAULT_SERVER, help="Branchwater search API base URL")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if args.genome_fasta:
                signature = sketch_fasta(args.genome_fasta)
                source = Path(args.genome_fasta)
            else:
                signature = load_signature(args.signature)
                source = Path(args.signature)

            output = Path(args.output) if args.output else Path(args.branchwater_folder) / f"{source.stem}.csv"
            matches = search_index(signature, args.threshold, server=args.server)
            write_branchwater_csv(matches, output)

            if matches:
                self.logger.info(
                    "%d metagenome(s) contain %s at >= %.2f; best containment %.4f (%s)",
                    len(matches),
                    source.stem,
                    args.threshold,
                    matches[0][1],
                    matches[0][0],
                )
            else:
                self.logger.warning(
                    "No metagenome reached containment %.2f. The public index has returned nothing for known "
                    "positives before; try a control genome (for example Salmonella LT2, GCF_000006945.2) to "
                    "check that the index is answering.",
                    args.threshold,
                )
            self.logger.info("Next: metaquest use_branchwater --branchwater-folder %s", output.parent)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error searching Branchwater: %s", e)
            return 1
```

In `metaquest/cli/main.py` add `from metaquest.cli.commands.branchwater_search import BranchwaterSearchCommand` next to the `select` import and `BranchwaterSearchCommand(),` as the first entry after `DownloadTestGenomeCommand(),` in `register_all_commands()`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cli_branchwater_search.py tests/test_cli_main.py -v`
Expected: all PASS.

- [ ] **Step 5: Documentation**

Replace README section 1 (the paragraph under `### 1. Getting Containment Files from Branchwater`) with:

```markdown
Search the Branchwater index directly from a genome. The command sketches the FASTA with sourmash
(install the extra with `pip install 'metaquest[sourmash]'`, or use `environment.yml`), queries the
public search API and writes `branchwater/<genome>.csv` in the layout the next steps read:

```bash
metaquest branchwater_search --genome-fasta genomes/GCF_000008025.1.fna --threshold 0.1
```

The CSV carries the accession, containment and cANI; the metadata columns are empty until step 5
(`download_metadata`) fills them from NCBI. If you already have a k=21, scaled=1000 sourmash signature,
pass `--signature file.sig` instead of the FASTA.

Alternatively, search at [https://branchwater.sourmash.bio/](https://branchwater.sourmash.bio/) in a
browser, download the CSV, and save it to the same folder.
```

In `docs/branchwater_workflow.md`, after the "Option B" block add:

```markdown
## Searching from a FASTA

`branchwater_search` replaces the manual download. It needs the sourmash extra and network access:

```bash
metaquest branchwater_search --genome-fasta genomes/GCF_000008025.1.fna --threshold 0.1 --branchwater-folder branchwater
metaquest use_branchwater --branchwater-folder branchwater --matches-folder matches
```

A run that returns zero matches is reported as a warning, not an error; verify the index with a genome
known to be abundant in metagenomes before trusting an empty result.
```

Run: `make check && make test && make pipeline`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add metaquest/cli/commands/branchwater_search.py metaquest/cli/main.py tests/test_cli_branchwater_search.py tests/test_cli_main.py README.md docs/branchwater_workflow.md
git commit -m "feat: add branchwater_search command to start the pipeline from a genome

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase B: single download command (branch `refactor/single-download-command`, based on `feat/branchwater-search`)

### Task 3: `download_sra` gains a tool check and `--report-file`

**Files:**
- Modify: `metaquest/data/sra.py` (`download_sra`, both return dictionaries)
- Modify: `metaquest/cli/commands/sra.py` (`DownloadSraCommand`)
- Test: `tests/test_data_sra.py`, `tests/test_cli_commands.py`

**Interfaces:**
- Produces: `download_sra(...)` returned dict gains `already_downloaded_accessions: List[str]` and `blacklisted_accessions: List[str]` (sorted); CLI flag `--report-file`; `DownloadSraCommand._write_report(report_file, stats)` writing `accession,status,message` with statuses `downloaded`, `failed`, `already_present`, `blacklisted`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli_commands.py`, inside the `DownloadSraCommand` test class (find `class TestDownloadSraCommand`), add:

```python
    @patch("metaquest.cli.commands.sra.shutil.which", return_value=None)
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_missing_fasterq_dump_exits_1(self, mock_download, _which, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            fastq_folder=str(tmp_path / "fastq"), accessions_file=str(acc), max_downloads=None, num_threads=4,
            max_workers=4, dry_run=False, force=False, max_retries=1, temp_folder=None, blacklist=None, report_file=None,
        )
        assert DownloadSraCommand().execute(args) == 1
        mock_download.assert_not_called()

    @patch("metaquest.cli.commands.sra.shutil.which", return_value=None)
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_dry_run_skips_tool_check(self, mock_download, _which, tmp_path):
        mock_download.return_value = {"total": 1, "already_downloaded": 0, "blacklisted": 0, "to_download": 1, "successful": 0, "failed": 0}
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            fastq_folder=str(tmp_path / "fastq"), accessions_file=str(acc), max_downloads=None, num_threads=4,
            max_workers=4, dry_run=True, force=False, max_retries=1, temp_folder=None, blacklist=None, report_file=None,
        )
        assert DownloadSraCommand().execute(args) == 0

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_report_file_lists_every_status(self, mock_download, _which, tmp_path):
        mock_download.return_value = {
            "total": 4, "already_downloaded": 1, "blacklisted": 1, "successful": 1, "failed": 1,
            "failed_accessions": ["SRR2"],
            "results": {"SRR1": "Downloaded 2 files", "SRR2": "Download failed: timeout"},
            "already_downloaded_accessions": ["SRR3"], "blacklisted_accessions": ["SRR4"],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\nSRR3\nSRR4\n")
        report = tmp_path / "reports" / "download_report.csv"
        args = argparse.Namespace(
            fastq_folder=str(tmp_path / "fastq"), accessions_file=str(acc), max_downloads=None, num_threads=4,
            max_workers=4, dry_run=False, force=False, max_retries=1, temp_folder=None, blacklist=None,
            report_file=str(report),
        )
        DownloadSraCommand().execute(args)
        assert report.read_text().splitlines() == [
            "accession,status,message",
            "SRR1,downloaded,Downloaded 2 files",
            "SRR2,failed,Download failed: timeout",
            "SRR3,already_present,",
            "SRR4,blacklisted,",
        ]
```

In `tests/test_data_sra.py`, in the class that tests `download_sra` (find `def test_download_sra_dry_run` or similar), add:

```python
    def test_download_sra_returns_accession_lists(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        (tmp_path / "fastq" / "SRR2").mkdir(parents=True)
        (tmp_path / "fastq" / "SRR2" / "SRR2_1.fastq").write_text("@r\nA\n+\nI\n")
        black = tmp_path / "black.txt"
        black.write_text("SRR1\n")
        stats = download_sra(tmp_path / "fastq", acc, dry_run=True, blacklist=[black])
        assert stats["already_downloaded_accessions"] == ["SRR2"]
        assert stats["blacklisted_accessions"] == ["SRR1"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cli_commands.py -k "fasterq_dump or report_file or tool_check" tests/test_data_sra.py -k accession_lists -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'shutil'`, `KeyError`).

- [ ] **Step 3: Implement**

`metaquest/data/sra.py`: in `download_sra`, add to the dry-run return dictionary and to the final `download_stats` dictionary:

```python
            "already_downloaded_accessions": sorted(str(a) for a in already_downloaded),
            "blacklisted_accessions": sorted(str(a) for a in blacklisted),
```

`metaquest/cli/commands/sra.py`: add `import csv` and `import shutil` to the imports. In `DownloadSraCommand.configure_parser` add:

```python
        parser.add_argument(
            "--report-file",
            default=None,
            help="Write a CSV of accession,status,message after the run (statuses: downloaded, failed, already_present, blacklisted)",
        )
```

At the top of `execute` (inside the `try`, before `download_sra` is called):

```python
            if not args.dry_run and shutil.which("fasterq-dump") is None:
                self.logger.error(
                    "fasterq-dump not found on PATH. Install sra-tools, for example: conda install -c bioconda sra-tools"
                )
                return 1
```

After the summary logging of a real run (where `_log_download_summary` is called), add:

```python
            if args.report_file:
                self._write_report(args.report_file, stats)
                self.logger.info("Download report written to %s", args.report_file)
```

and the method:

```python
    @staticmethod
    def _write_report(report_file: str, stats: dict) -> None:
        """Write one row per accession with its outcome."""
        failed = set(stats.get("failed_accessions", []))
        rows = []
        for accession, message in stats.get("results", {}).items():
            rows.append((accession, "failed" if accession in failed else "downloaded", message))
        rows.extend((acc, "already_present", "") for acc in stats.get("already_downloaded_accessions", []))
        rows.extend((acc, "blacklisted", "") for acc in stats.get("blacklisted_accessions", []))
        path = Path(report_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["accession", "status", "message"])
            writer.writerows(sorted(rows))
```

Update every existing `argparse.Namespace(...)` for `DownloadSraCommand` in `tests/test_cli_commands.py` to include `report_file=None`, and patch `shutil.which` to a path in the tests that expect a real run to proceed (or set `dry_run=True` where they already do).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cli_commands.py tests/test_data_sra.py -v`
Expected: all PASS.

- [ ] **Step 5: Gates and commit**

Run: `make check && make test`

```bash
git checkout -b refactor/single-download-command feat/branchwater-search
git add metaquest/data/sra.py metaquest/cli/commands/sra.py tests/test_data_sra.py tests/test_cli_commands.py
git commit -m "feat: check for fasterq-dump and add --report-file to download_sra

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 4: Remove `sra_download` and the enhanced downloader module

**Files:**
- Delete: `metaquest/data/sra_enhanced.py`, `tests/test_sra_enhanced_extended.py`
- Modify: `metaquest/data/sra_metadata.py` (append `estimate_download_time`), `metaquest/cli/commands/sra_enhanced.py` (imports, `SRAInfoCommand.execute`, delete `SRADownloadEnhancedCommand`), `metaquest/cli/main.py`, `tests/test_sra_enhanced.py`, `tests/test_sra_metadata_extended.py`, `tests/test_cli_commands_sra_enhanced.py`, `tests/test_cli_main.py`

**Interfaces:**
- Produces: `metaquest.data.sra_metadata.estimate_download_time(total_size_gb, bandwidth_mbps=100, num_parallel=4) -> float` (moved verbatim).
- Removes: command `sra_download`, `EnhancedSRADownloader`, `create_download_report`, `verify_sra_tools`, `_fastq_suffix`.

- [ ] **Step 1: Move `estimate_download_time` and write its test in the new home**

Append to `metaquest/data/sra_metadata.py` the `estimate_download_time` function exactly as it exists in `metaquest/data/sra_enhanced.py:421-443` (docstring included). Add to `tests/test_sra_metadata_extended.py`:

```python
class TestEstimateDownloadTime:
    def test_scales_with_size_and_parallelism(self):
        from metaquest.data.sra_metadata import estimate_download_time

        one = estimate_download_time(1.0, 100.0, 4)
        assert one == pytest.approx(1.0 * 1024 * 8 / (100.0 * 4 * 0.8) / 3600)
        assert estimate_download_time(2.0, 100.0, 4) == pytest.approx(2 * one)
```

Run: `pytest tests/test_sra_metadata_extended.py -k EstimateDownloadTime -v` -> PASS (the function now exists in both places).

- [ ] **Step 2: Rewire `sra_info` and delete the download command**

In `metaquest/cli/commands/sra_enhanced.py` replace the import block (lines 12-21) with:

```python
from metaquest.data.sra_metadata import (
    SRAMetadataClient,
    create_download_preview,
    estimate_download_time,
    save_metadata_report,
    generate_statistics_report,
)
```

In `SRAInfoCommand.execute` replace the two lines

```python
            downloader = EnhancedSRADownloader(args.email, args.api_key)
            metadata, tech_counts, total_size_gb = downloader.preview_downloads(accessions)
```

with

```python
            client = SRAMetadataClient(args.email, args.api_key)
            metadata, tech_counts, total_size_gb = create_download_preview(accessions, client)
```

Delete the whole `class SRADownloadEnhancedCommand` (from its `class` line through the line before `class SRAStatsCommand`). Update the module docstring to name the three remaining commands. In `metaquest/cli/main.py` remove `SRADownloadEnhancedCommand` from the import and from `register_all_commands()`.

- [ ] **Step 3: Delete the module and trim the tests**

```bash
git rm metaquest/data/sra_enhanced.py tests/test_sra_enhanced_extended.py
```

`tests/test_sra_enhanced.py`: remove the `from metaquest.data.sra_enhanced import (...)` block; delete every test class or function that references `EnhancedSRADownloader`, `verify_sra_tools`, `estimate_download_time` or `create_download_report` (expected: `TestEnhancedSRADownloader`, `TestSRAUtilities`, and any `TestSRAIntegration` cases using the downloader); keep `TestSRAMetadataClient`, `TestTechnologyDetection`, `TestReadStatistics`. Rename the file to `tests/test_sra_metadata_client.py` with `git mv` and update its module docstring.

`tests/test_cli_commands_sra_enhanced.py`: remove `SRADownloadEnhancedCommand` from the import and delete `class TestSRADownloadEnhancedCommand`. Update the `SRAInfoCommand` tests that patched `EnhancedSRADownloader` to patch `metaquest.cli.commands.sra_enhanced.create_download_preview` and `metaquest.cli.commands.sra_enhanced.SRAMetadataClient` instead; assert `create_download_preview` is called with the accession list and the client instance.

`tests/test_cli_main.py`: remove `"sra_download",` from `expected_commands`.

- [ ] **Step 4: Verify**

Run: `grep -rn "sra_enhanced\b\|EnhancedSRADownloader\|verify_sra_tools\|create_download_report" metaquest tests README.md docs CLAUDE.md | grep -v "cli/commands/sra_enhanced\|superpowers"`
Expected: only README/docs mentions remain (Task 6 removes them).

Run: `pytest tests/test_sra_metadata_client.py tests/test_sra_metadata_extended.py tests/test_cli_commands_sra_enhanced.py tests/test_cli_main.py -v && make check && make test`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A metaquest tests
git commit -m "refactor: remove the sra_download command and the enhanced downloader module

download_sra is the single download path. sra_info keeps its NCBI preview through
create_download_preview; estimate_download_time moves to sra_metadata.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 5: Remove `sra-download-intelligent`, the download manager and its report code

**Files:**
- Delete: `metaquest/sra/download_manager.py`, `tests/test_sra_intelligent_download.py`
- Modify: `metaquest/sra/__init__.py`, `metaquest/sra/reporting.py`, `metaquest/cli/commands/sra_intelligent.py`, `metaquest/cli/main.py`, `tests/test_cli_sra_intelligent.py`, `tests/test_sra_reporting_starter.py`, `tests/test_sra_reporting_extended.py`, `tests/test_cli_main.py`

**Interfaces:**
- Removes: command `sra-download-intelligent` (and alias `sra_download_intelligent`), `IntelligentDownloadManager`, `BandwidthManager`, `DownloadOptimizer`, `CheckpointManager`, `NetworkConditions`, `DownloadCheckpoint`, `DownloadProgress`, `DownloadSession`, `SRAReportGenerator.create_download_summary`, `_create_download_plots`, `_generate_download_html`, `_generate_simple_download_html`, `export_metadata_enriched`, dashboard flag `--download-session`.

- [ ] **Step 1: Delete the manager and its tests**

```bash
git rm metaquest/sra/download_manager.py tests/test_sra_intelligent_download.py
```

`metaquest/sra/__init__.py`: delete the `from .download_manager import (...)` block, the eight "Download Management" entries in `__all__`, and rewrite the module docstring to describe analytics and reporting only (three lines, no example code that references the manager).

- [ ] **Step 2: Trim reporting**

In `metaquest/sra/reporting.py` delete `from metaquest.sra.download_manager import DownloadSession`, the methods `create_download_summary`, `_create_download_plots`, `_generate_download_html`, `_generate_simple_download_html` and `export_metadata_enriched`, and any private helper that only they called (run `flake8` to find now-unused imports and remove them). Keep `_simple_shell` (used by the quality and comparative HTML).

- [ ] **Step 3: Trim the CLI**

In `metaquest/cli/commands/sra_intelligent.py`: delete `class SRAIntelligentDownloadCommand` (from its `class` line to the line before `class SRAQualityProfileCommand`); trim the `from metaquest.sra import (...)` block to the names still used; in `SRAInteractiveDashboardCommand` delete the `--download-session` argument and the whole `if args.download_session:` block (keep the `if args.dashboard_type in ["download", "full"]:` branch only if anything else is inside it; otherwise delete it and remove `"download"` from `--dashboard-type` choices, defaulting the help to name `quality`, `comparative`, `full`). Delete `import json` if unused. In `metaquest/cli/main.py` remove `SRAIntelligentDownloadCommand` from the import and registration.

- [ ] **Step 4: Trim the tests**

`tests/test_cli_sra_intelligent.py`: delete `class TestSRAIntelligentDownloadCommand`, the `from metaquest.sra.download_manager import (...)` block, every `download_session=...` entry in `argparse.Namespace(...)` calls, and any dashboard test whose name or body mentions `download_session`. `tests/test_sra_reporting_starter.py` and `tests/test_sra_reporting_extended.py`: delete `MockDownloadSession`, the `mock_download_session` fixture, and every test referencing `create_download_summary`, `_create_download_plots`, `_generate_download_html`, `_generate_simple_download_html` or `export_metadata_enriched`. `tests/test_cli_main.py`: remove the `("sra-download-intelligent", "sra_download_intelligent")` pair from `test_snake_case_aliases_resolve`.

- [ ] **Step 5: Verify and commit**

Run: `grep -rn "download_manager\|IntelligentDownloadManager\|DownloadSession\|download_session\|sra-download-intelligent\|sra_download_intelligent" metaquest tests | grep -v superpowers`
Expected: nothing.

Run: `make check && make test`
Expected: green.

```bash
git add -A metaquest tests
git commit -m "refactor: remove sra-download-intelligent and the download manager

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 6: Documentation for the single download command

**Files:**
- Modify: `README.md` (external-tools table row; "Choosing which datasets to download" paragraph; replace the sections "Which SRA download command should I use?", "Intelligent SRA Downloading" and "Enhanced SRA Features"), `docs/SRA_ENHANCED_FEATURES.md` (rewrite), `CLAUDE.md` and `AGENTS.md` (lines 72-78, 92, 309, 331)

- [ ] **Step 1: README**

1. External-tools table: the `fasterq-dump` row becomes `| \`fasterq-dump\` (sra-tools) | \`download_sra\` |`.
2. Under "Choosing which datasets to download", replace the two sentences starting "`accessions.txt` is the input for" with:

```markdown
`accessions.txt` is the input for `download_sra`, which writes `fastq/<accession>/<accession>_1.fastq`
(and `_2` for paired runs), the layout `status`, `sra_stats`, `sra_profile_quality`, `sra_dashboard`
and `extract_target_reads` read.
```

3. Replace the sections "Which SRA download command should I use?" and "Intelligent SRA Downloading" (from the first heading to the line before "### SRA Quality Profiling") with:

```markdown
### Downloading reads

`download_sra` runs `fasterq-dump` in parallel, skips accessions whose FASTQ files already exist
(unless `--force`), retries failures, and writes `fastq/failed_accessions.txt` for reruns:

```bash
metaquest download_sra --accessions-file accessions.txt --fastq-folder fastq --max-workers 4 --num-threads 4
metaquest download_sra --accessions-file accessions.txt --dry-run
metaquest download_sra --accessions-file accessions.txt --report-file download_report.csv
```

`--report-file` writes one row per accession with the status `downloaded`, `failed`, `already_present`
or `blacklisted`. To see sizes and sequencing technology before downloading, use `sra_info` (needs an
email for NCBI); see `docs/SRA_ENHANCED_FEATURES.md`.
```

4. Delete the "Enhanced SRA Features" section (heading through the closing code fence before "## Visualizing Results").

- [ ] **Step 2: `docs/SRA_ENHANCED_FEATURES.md`**

Replace the file with (plain text, no emoji):

```markdown
# SRA information, statistics and validation

Three commands support the `download_sra` step. All read an accession list (one per line) and none
of them download reads.

## sra_info

Fetches NCBI metadata for the accessions and prints sizes, layouts, technologies and an estimated
download time. Requires `--email` (NCBI asks for it); `--api-key` raises the rate limit.

```bash
metaquest sra_info --accessions-file accessions.txt --email you@example.org --output-report sra_info_report.csv
```

## sra_stats

Reads the downloaded FASTQ files under `--fastq-folder` (default `fastq`) and reports read counts,
bases, read-length statistics, GC content and mean quality per accession.

```bash
metaquest sra_stats --fastq-folder fastq --output-report sra_stats.csv
```

## sra_validate

Checks that each accession folder holds readable FASTQ files; `--check-pairs` also verifies that
paired files have matching read counts.

```bash
metaquest sra_validate --fastq-folder fastq --check-pairs
```

Downloads themselves are documented in the README under "Downloading reads".
```

- [ ] **Step 3: CLAUDE.md and AGENTS.md**

In both files: change the sentence "The intelligent SRA package provides four main CLI commands:" to "The SRA package provides three analysis commands:", delete the `sra-download-intelligent` bullet, rename the three remaining bullets to `sra_profile_quality`, `sra_dashboard`, `sra_compare`; in the pipeline list item 5 replace the parenthesised commands with `(download_sra, sra_profile_quality, sra_dashboard)`; delete the line `- [x] **IntelligentDownloadManager** - Resume capability, bandwidth optimization` and the line `- **download_manager.py** - Intelligent downloading with checkpoints and bandwidth management`. Leave the rest of both files as they are.

- [ ] **Step 4: Verify and commit**

Run: `grep -n "sra_download\b\|sra-download-intelligent\|download_manager\|IntelligentDownloadManager\|Enhanced SRA Features" README.md docs/SRA_ENHANCED_FEATURES.md CLAUDE.md AGENTS.md`
Expected: nothing. Then `make pipeline`.

```bash
git add README.md docs/SRA_ENHANCED_FEATURES.md CLAUDE.md
git commit -m "docs: describe download_sra as the single download command

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(`AGENTS.md` is untracked; edit it but do not add it.)

---

## Phase C: taxonomy tables (branch `feat/taxonomy-table-unification`, based on `refactor/single-download-command`)

### Task 7: `taxonomic_summary` accepts the `enrich_taxonomy` map

**Files:**
- Modify: `metaquest/data/defaults.py` (add `read_records`), `metaquest/data/taxonomy.py` (`_build_taxonomy_lineage_map`, `create_taxonomic_summary` error handling), `metaquest/cli/commands/advanced_analysis.py` (`TaxonomicSummaryCommand`), `README.md` ("Taxonomic Summary Analysis")
- Test: `tests/test_data_defaults.py`, `tests/test_taxonomy.py`, `tests/test_cli_commands_advanced_analysis.py`

**Interfaces:**
- Produces: `read_records(path) -> pd.DataFrame`; `_build_taxonomy_lineage_map` accepts either table; `MAP_RANK_COLUMNS` in `metaquest/data/taxonomy.py`.

- [ ] **Step 1: Write the failing tests**

`tests/test_data_defaults.py`:

```python
class TestReadRecords:
    def test_tsv_keeps_first_column(self, tmp_path):
        from metaquest.data.defaults import read_records

        f = tmp_path / "taxonomy.tsv"
        f.write_text("genome_id\tgenus\nGCF_1\tWolbachia\n")
        df = read_records(f)
        assert list(df.columns) == ["genome_id", "genus"]
        assert df.loc[0, "genome_id"] == "GCF_1"

    def test_csv(self, tmp_path):
        from metaquest.data.defaults import read_records

        f = tmp_path / "v.csv"
        f.write_text("original_name,is_valid\nE. coli,True\n")
        assert list(read_records(f).columns) == ["original_name", "is_valid"]
```

`tests/test_taxonomy.py` (append; `pd`, `pytest`, `create_taxonomic_summary` and `ProcessingError` are already imported there, add what is missing):

```python
class TestTaxonomyMapInput:
    def _abundance(self):
        return pd.DataFrame({"GCF_1": [0.9, 0.0], "GCF_2": [0.1, 0.7], "GCF_9": [0.0, 0.2]}, index=["S1", "S2"])

    def _taxonomy_map(self):
        return pd.DataFrame(
            {
                "genome_id": ["GCF_1", "GCF_2"],
                "species": ["Wolbachia pipientis", "Wolbachia sp947251865"],
                "genus": ["Wolbachia", "Wolbachia"],
                "family": ["Anaplasmataceae", "Anaplasmataceae"],
                "order": ["Rickettsiales", "Rickettsiales"],
                "class_name": ["Alphaproteobacteria", ""],
                "phylum": ["Pseudomonadota", "Pseudomonadota"],
                "organism": ["", ""],
                "tax_id": ["", ""],
            }
        )

    def test_genus_summary_from_map(self):
        result = create_taxonomic_summary(self._abundance(), self._taxonomy_map(), level="genus", min_abundance=0.0)
        assert result.loc["S1", "Wolbachia"] == pytest.approx(1.0)
        assert result.loc["S2", "Unclassified_genus"] == pytest.approx(0.2)

    def test_class_name_maps_to_class_rank(self):
        result = create_taxonomic_summary(self._abundance(), self._taxonomy_map(), level="class", min_abundance=0.0)
        assert result.loc["S1", "Alphaproteobacteria"] == pytest.approx(0.9)
        assert result.loc["S1", "Unclassified_class"] == pytest.approx(0.1)

    def test_unknown_table_shape_raises(self):
        with pytest.raises(ProcessingError, match="validate_taxonomy .* or enrich_taxonomy"):
            create_taxonomic_summary(self._abundance(), pd.DataFrame({"foo": [1]}), level="genus")
```

`tests/test_cli_commands_advanced_analysis.py` (append; real files, nothing patched):

```python
class TestTaxonomicSummaryWithEnrichMap:
    def test_parsed_containment_plus_taxonomy_tsv(self, tmp_path):
        from metaquest.cli.commands.advanced_analysis import TaxonomicSummaryCommand

        cont = tmp_path / "parsed_containment.txt"
        cont.write_text("\tGCF_1\tGCF_2\tmax_containment\tmax_containment_annotation\nS1\t0.9\t0.1\t0.9\tGCF_1\nS2\t0.0\t0.7\t0.7\tGCF_2\n")
        tax = tmp_path / "taxonomy.tsv"
        tax.write_text(
            "genome_id\tspecies\tgenus\tfamily\torder\tclass_name\tphylum\torganism\ttax_id\n"
            "GCF_1\tWolbachia pipientis\tWolbachia\tAnaplasmataceae\tRickettsiales\tAlphaproteobacteria\tPseudomonadota\t\t\n"
            "GCF_2\tWolbachia sp947251865\tWolbachia\tAnaplasmataceae\tRickettsiales\tAlphaproteobacteria\tPseudomonadota\t\t\n"
        )
        args = argparse.Namespace(
            abundance_file=str(cont), taxonomy_file=str(tax), output_dir=str(tmp_path / "out"), levels=["genus"], min_abundance=0.0
        )
        assert TaxonomicSummaryCommand().execute(args) == 0
        out = (tmp_path / "out" / "taxonomy_summary_genus.csv").read_text().splitlines()
        assert out[0].startswith(",Wolbachia")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_data_defaults.py -k ReadRecords tests/test_taxonomy.py -k MapInput tests/test_cli_commands_advanced_analysis.py -k EnrichMap -v`
Expected: FAIL (`ImportError` for `read_records`; `KeyError: 'is_valid'`).

- [ ] **Step 3: Implement**

`metaquest/data/defaults.py`, after `read_table`:

```python
def read_records(path: Union[str, Path]) -> pd.DataFrame:
    """Read a record table (one row per entity, no index column); TSV or CSV by suffix."""
    path = Path(path)
    return pd.read_csv(path, sep=_separator_for(path))
```

`metaquest/data/taxonomy.py`: add near the top

```python
# Columns of the enrich_taxonomy map and the lineage rank each one holds.
MAP_RANK_COLUMNS = {
    "species": "species",
    "genus": "genus",
    "family": "family",
    "order": "order",
    "class_name": "class",
    "phylum": "phylum",
}
```

Replace `_build_taxonomy_lineage_map` with:

```python
def _build_taxonomy_lineage_map(taxonomy_data: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Map a species name or genome id to {rank: name}, from either taxonomy table MetaQuest writes."""
    columns = set(taxonomy_data.columns)
    if "genome_id" in columns:
        return _lineages_from_taxonomy_map(taxonomy_data)
    if {"original_name", "is_valid", "lineage"}.issubset(columns):
        return _lineages_from_validation_table(taxonomy_data)
    raise ProcessingError(
        "Taxonomy file must come from validate_taxonomy (columns original_name, is_valid, lineage) "
        "or enrich_taxonomy (column genome_id with rank columns)"
    )


def _lineages_from_validation_table(taxonomy_data: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    taxonomy_dict: Dict[str, Dict[str, str]] = {}
    for _, row in taxonomy_data.iterrows():
        if not (row["is_valid"] and row["lineage"]):
            continue
        lineage_dict = {}
        for part in str(row["lineage"]).split(";"):
            if ":" in part:
                rank, name = part.split(":", 1)
                lineage_dict[rank.lower()] = name
        taxonomy_dict[row["original_name"]] = lineage_dict
    return taxonomy_dict


def _lineages_from_taxonomy_map(taxonomy_data: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    lineages: Dict[str, Dict[str, str]] = {}
    for _, row in taxonomy_data.iterrows():
        ranks = {}
        for column, rank in MAP_RANK_COLUMNS.items():
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                ranks[rank] = value.strip()
        lineages[str(row["genome_id"])] = ranks
    return lineages
```

In `create_taxonomic_summary` and `analyze_taxonomic_composition`, change the `except Exception as e:` handlers to re-raise `ProcessingError` unchanged:

```python
    except ProcessingError:
        raise
    except Exception as e:
        raise ProcessingError(f"Failed to create taxonomic summary: {e}")
```

`metaquest/cli/commands/advanced_analysis.py`: import `read_records` from `metaquest.data.defaults`; in `TaxonomicSummaryCommand.execute` replace `taxonomy_df = pd.read_csv(args.taxonomy_file)` with `taxonomy_df = read_records(args.taxonomy_file)` (drop the local `import pandas as pd` if it becomes unused); set the `--taxonomy-file` help to `"Taxonomy table from enrich_taxonomy (TSV, genome_id and ranks) or validate_taxonomy (CSV)"`. Existing tests that patch `pandas.read_csv` for the taxonomy file must patch `metaquest.cli.commands.advanced_analysis.read_records` instead.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_data_defaults.py tests/test_taxonomy.py tests/test_taxonomy_extended.py tests/test_cli_commands_advanced_analysis.py -v`
Expected: all PASS.

- [ ] **Step 5: README and commit**

Replace the "Taxonomic Summary Analysis" section body with:

```markdown
Summarise containment per taxonomic rank. The taxonomy table can be the map written by
`enrich_taxonomy` (genome ids and ranks, the usual route after `parse_containment`) or the validation
CSV written by `validate_taxonomy`:

```bash
metaquest enrich_taxonomy --parsed-containment parsed_containment.txt --output taxonomy.tsv
metaquest taxonomic_summary --abundance-file parsed_containment.txt --taxonomy-file taxonomy.tsv \
    --levels phylum class order family genus --output-dir taxonomic_summaries

metaquest taxonomic_summary --abundance-file abundance_matrix.csv --taxonomy-file validation_results.csv
```
```

Run: `make check && make test && make pipeline`

```bash
git checkout -b feat/taxonomy-table-unification refactor/single-download-command
git add metaquest/data/defaults.py metaquest/data/taxonomy.py metaquest/cli/commands/advanced_analysis.py tests/test_data_defaults.py tests/test_taxonomy.py tests/test_cli_commands_advanced_analysis.py README.md
git commit -m "feat: accept the enrich_taxonomy map in taxonomic_summary

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase D: CLI polish (branch `refactor/cli-polish`, based on `feat/taxonomy-table-unification`)

### Task 8: Grouped help, hidden aliases, snake_case names

**Files:**
- Modify: `metaquest/cli/base.py` (`BaseCommand.group`, `CommandRegistry.setup_parsers`), `metaquest/cli/main.py` (`create_parser`, `_HelpFormatter`, `_commands_epilog`, `GROUP_ORDER`), every command class (add `group`), `metaquest/cli/commands/sra_intelligent.py` (names and aliases), `tests/test_cli_main.py`, `tests/test_cli_sra_intelligent.py` (name assertions), `README.md`, `CLAUDE.md`, `AGENTS.md` (kebab names)

**Interfaces:**
- Produces: `BaseCommand.group -> str` (default `"Other"`); main help epilog "commands by pipeline step"; canonical `sra_profile_quality`, `sra_dashboard`, `sra_compare` with hidden kebab aliases.

- [ ] **Step 1: Write the failing tests**

Replace `test_snake_case_aliases_resolve` in `tests/test_cli_main.py` with:

```python
    def test_kebab_aliases_parse_but_are_hidden(self):
        parser = create_parser()
        choices = self._subcommand_choices(parser)
        for snake, kebab in (
            ("sra_profile_quality", "sra-profile-quality"),
            ("sra_dashboard", "sra-dashboard"),
            ("sra_compare", "sra-compare"),
        ):
            assert snake in choices and kebab in choices
        help_text = parser.format_help()
        assert "sra_dashboard" in help_text
        assert "sra-dashboard" not in help_text
        assert "{" not in help_text.split("commands by pipeline step")[0].split("usage:")[1]

    def test_help_groups_commands_by_step(self):
        help_text = create_parser().format_help()
        for group in ("Containment:", "Metadata:", "Genomes:", "Reads:", "Analysis:"):
            assert group in help_text
        assert help_text.index("Containment:") < help_text.index("Reads:")

    def test_every_command_declares_a_group(self):
        register_all_commands()
        ungrouped = [c.name for c in command_registry.get_all_commands().values() if c.group == "Other"]
        assert ungrouped == []

    def test_subcommand_help_shows_defaults(self):
        parser = create_parser()
        sub = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None)).choices["parse_containment"]
        assert "(default: matches)" in sub.format_help()
```

Update `tests/test_cli_sra_intelligent.py` assertions from `cmd.name == "sra-profile-quality"` (and the other two) to the snake_case names, and assert `"sra-profile-quality" in cmd.aliases` (same pattern for the other two).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cli_main.py tests/test_cli_sra_intelligent.py -v`
Expected: the four new tests FAIL; name assertions FAIL.

- [ ] **Step 3: Implement the registry and parser**

`metaquest/cli/base.py`: add to `BaseCommand` after `aliases`:

```python
    @property
    def group(self) -> str:
        """Pipeline step the command belongs to, used to group the main help listing."""
        return "Other"
```

Replace `CommandRegistry.setup_parsers` with:

```python
    def setup_parsers(self, main_parser: argparse.ArgumentParser) -> None:
        """Add one subparser per command plus a hidden subparser per alias."""
        subparsers = main_parser.add_subparsers(
            title="commands",
            dest="command",
            metavar="COMMAND",
            help="Run 'metaquest COMMAND --help' for the options of one command",
        )
        subparsers.required = True

        for command in self._commands.values():
            self._add_parser(subparsers, command, command.name, command.help)
            for alias in command.aliases:
                self._add_parser(subparsers, command, alias, argparse.SUPPRESS)

    @staticmethod
    def _add_parser(subparsers, command: BaseCommand, name: str, help_text: str) -> None:
        subparser = subparsers.add_parser(
            name,
            help=help_text,
            description=command.help,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        command.configure_parser(subparser)
        subparser.set_defaults(func=command.execute)
```

`metaquest/cli/main.py`: add before `create_parser`:

```python
GROUP_ORDER = ["Containment", "Metadata", "Genomes", "Reads", "Analysis", "Other"]


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show option defaults and keep the epilog's line breaks."""


def _commands_epilog(commands: Dict[str, BaseCommand]) -> str:
    """List commands under their pipeline step for the main --help."""
    by_group: Dict[str, List[BaseCommand]] = {}
    for command in commands.values():
        by_group.setdefault(command.group, []).append(command)
    lines = ["commands by pipeline step:"]
    for group in GROUP_ORDER + sorted(set(by_group) - set(GROUP_ORDER)):
        if group not in by_group:
            continue
        lines.append(f"  {group}:")
        for command in by_group[group]:
            lines.append(f"    {command.name:<28} {command.help}")
    return "\n".join(lines)
```

(add `Dict`, `List` to the typing import and `from metaquest.cli.base import BaseCommand, command_registry`). In `create_parser`, call `register_all_commands()` first, then build the parser with `epilog=_commands_epilog(command_registry.get_all_commands())` and `formatter_class=_HelpFormatter`, then `command_registry.setup_parsers(parser)`.

- [ ] **Step 4: Declare groups and rename the kebab commands**

Add a `group` property to every command class:

| Group | Commands |
|---|---|
| `Containment` | branchwater_search, use_branchwater, parse_containment, plot_containment, explore_containment, enrich_taxonomy, find_by_taxonomy |
| `Metadata` | extract_branchwater_metadata, download_metadata, parse_metadata, check_metadata_attributes, count_metadata, single_sample, plot_metadata_counts |
| `Genomes` | genome_search, genome_download, genome_prepare, download_test_genome |
| `Reads` | select_datasets, download_sra, status, sra_info, sra_stats, sra_validate, sra_profile_quality, sra_dashboard, sra_compare, extract_target_reads |
| `Analysis` | diversity_analysis, interactive_plot, validate_taxonomy, taxonomic_summary |

(`assemble_datasets` is removed in Task 10; give it no group.) In `metaquest/cli/commands/sra_intelligent.py` swap each `name`/`aliases` pair so `name` returns the snake_case form and `aliases` returns `[<kebab form>]`.

Replace `sra-profile-quality`, `sra-dashboard`, `sra-compare` with the snake_case names in `README.md` (lines 217, 256, 263, 275, 282, 293), `CLAUDE.md` and `AGENTS.md` (the command bullets and pipeline item 5, if Task 6 left any kebab form).

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_cli_main.py tests/test_cli_sra_intelligent.py tests/test_cli_commands.py -v && make check && make test && metaquest --help | head -40`
Expected: green; the help output shows `COMMAND` in the usage line and the grouped listing.

```bash
git checkout -b refactor/cli-polish feat/taxonomy-table-unification
git add -A metaquest tests README.md CLAUDE.md
git commit -m "refactor: group the command listing, hide aliases, show defaults in every help

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 9: One threshold rule (greater-or-equal)

**Files:**
- Modify: `metaquest/processing/counts.py:55,140`, `metaquest/processing/containment.py:116,185,188,227`, `metaquest/data/branchwater.py:328`, `metaquest/data/metadata.py:66`, `metaquest/visualization/plots.py:399`; help strings in `metaquest/cli/commands/metadata.py:42,160`, `metaquest/cli/commands/samples.py:49`
- Test: `tests/test_processing_counts.py`, `tests/test_processing_containment.py`, `tests/test_data_branchwater.py`, `tests/test_data_metadata.py`, `tests/test_visualization_plots.py`

- [ ] **Step 1: Write the failing boundary tests**

`tests/test_processing_counts.py`:

```python
def test_count_metadata_includes_samples_at_the_threshold(tmp_path):
    from metaquest.processing.counts import count_metadata

    summary = tmp_path / "parsed_containment.txt"
    summary.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.5\n")
    meta = tmp_path / "meta.txt"
    meta.write_text("Run_ID\torganism\nSRR1\tA\nSRR2\tB\n")
    out = tmp_path / "counts.txt"
    result = count_metadata(summary, meta, "organism", 0.9, out)
    assert list(result.index) == ["A"]
```

`tests/test_data_branchwater.py`:

```python
def test_summary_counts_samples_at_exactly_one(tmp_path):
    from metaquest.data.branchwater import parse_containment_data

    matches = tmp_path / "matches"
    matches.mkdir()
    (matches / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,1.0,1.0\nSRR2,0.5,0.9\n")
    parse_containment_data(matches, tmp_path / "parsed.txt", tmp_path / "summary.txt", 0.5)
    rows = dict(line.split("\t") for line in (tmp_path / "summary.txt").read_text().splitlines()[1:])
    assert rows["1.0"] == "1" and rows["0.5"] == "2"
```

(Check the exact `count_metadata` signature in `metaquest/processing/counts.py` and adjust the positional arguments to its parameter order.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_processing_counts.py -k threshold tests/test_data_branchwater.py -k exactly_one -v`
Expected: FAIL (strict comparison excludes the boundary rows).

- [ ] **Step 3: Implement**

Change `> threshold` to `>= threshold` at each listed site (`plots.py:399` becomes `x if x >= threshold else 0`), and the log texts in `containment.py` from `"... > {threshold}"` to `"... >= {threshold}"`. Update the three help strings to end with `" (inclusive)"`. Run the five affected test files; where an existing test encoded the strict rule (a sample at the boundary expected to be excluded), change the expectation and say so in the report.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_processing_counts.py tests/test_processing_containment.py tests/test_data_branchwater.py tests/test_data_metadata.py tests/test_visualization_plots.py -v && make check && make test`

```bash
git add metaquest tests
git commit -m "fix: apply containment thresholds inclusively in every command

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 10: Remove the stubs, wire t-SNE

**Files:**
- Modify: `metaquest/cli/commands/sra.py` (delete `AssembleDatasetsCommand`), `metaquest/cli/commands/__init__.py`, `metaquest/cli/main.py`, `metaquest/data/sra.py` (delete `assemble_datasets`), `metaquest/cli/commands/containment.py` (delete `--file-format`), `metaquest/cli/commands/advanced_analysis.py` (t-SNE branch), `README.md` (drop `--file-format branchwater` from the section 4 example)
- Test: `tests/test_cli_commands.py`, `tests/test_data_sra.py`, `tests/test_cli_main.py`, `tests/test_cli_commands_advanced_analysis.py`

- [ ] **Step 1: Write the failing t-SNE test**

In `tests/test_cli_commands_advanced_analysis.py` replace the test that expects `plot_type="tsne"` to fail (line ~403) with:

```python
    @patch("metaquest.cli.commands.advanced_analysis.create_interactive_tsne")
    @patch("metaquest.cli.commands.advanced_analysis.read_matrix")
    def test_execute_tsne_plot(self, mock_read, mock_tsne, tmp_path):
        import pandas as pd

        mock_read.return_value = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}, index=["s1", "s2", "s3"])
        args = argparse.Namespace(
            data_file="m.csv", metadata_file=None, plot_type="tsne", color_by=None, size_by=None,
            output_file=str(tmp_path / "t.html"), title=None, no_show=True,
        )
        assert InteractivePlotCommand().execute(args) == 0
        assert mock_tsne.call_args.kwargs["output_file"] == str(tmp_path / "t.html")
        assert mock_tsne.call_args.kwargs["show_plot"] is False
```

- [ ] **Step 2: Implement**

`advanced_analysis.py`: import `create_interactive_tsne` alongside the other interactive functions; replace the final `else: raise MetaQuestError(...)` branch with:

```python
            elif args.plot_type == "tsne":
                create_interactive_tsne(
                    data_df,
                    metadata_df,
                    color_by=args.color_by,
                    title=args.title or "Interactive t-SNE Plot",
                    output_file=args.output_file,
                    show_plot=show_plot,
                )
```

(use the same local variable names the `pca` branch uses for the data frame, metadata frame and show flag).

Remove `assemble_datasets`: delete `class AssembleDatasetsCommand` from `metaquest/cli/commands/sra.py` and its import of `assemble_datasets`; delete `def assemble_datasets` from `metaquest/data/sra.py`; remove `AssembleDatasetsCommand` from `metaquest/cli/commands/__init__.py` (import and `__all__`) and from `metaquest/cli/main.py`; delete `class TestAssembleDatasetsCommand` in `tests/test_cli_commands.py` and the `assemble_datasets` class in `tests/test_data_sra.py` (plus their imports); remove `"assemble_datasets"` from `expected_commands`.

Remove `--file-format`: delete the argument in `metaquest/cli/commands/containment.py`; in `tests/test_cli_commands.py` delete the assertion `args.file_format is None` and the `"--file-format"` parse case; in `README.md` section 4 drop ` --file-format branchwater` from the command.

- [ ] **Step 3: Verify and commit**

Run: `grep -rn "assemble_datasets\|AssembleDatasetsCommand\|file_format\|file-format\|not yet implemented" metaquest tests README.md | grep -v "detect_file_format\|validate_csv_file"`
Expected: nothing.

Run: `pytest tests/test_cli_commands.py tests/test_data_sra.py tests/test_cli_main.py tests/test_cli_commands_advanced_analysis.py -v && make check && make test`

```bash
git add -A metaquest tests README.md
git commit -m "refactor: remove the assemble_datasets stub and --file-format, wire interactive t-SNE

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 11: Delete dead functions, trim the allowlist, validate samtools output paths

**Files:**
- Modify (delete functions and their tests): `metaquest/processing/containment.py` (`filter_samples_by_containment`, `find_co_occurring_genomes`; tests in `tests/test_processing_containment.py`), `metaquest/processing/counts.py` (`count_metadata_by_category`, `summarize_metadata_column`; `tests/test_processing_counts.py`), `metaquest/data/file_io.py` (`process_files_in_directory`, `read_json`, `write_json`; `tests/test_data_file_io.py`), `metaquest/core/validation.py` (`validate_csv_file`; `tests/test_basic.py`), `metaquest/visualization/plots.py` (`plot_heatmap`; `tests/test_visualization_plots.py`), `metaquest/visualization/interactive.py` (`create_beta_diversity_plot`; `tests/test_interactive_plots_extended.py`), `metaquest/processing/diversity.py` (`calculate_dispersion`; `tests/test_diversity.py`), `metaquest/plugins/visualizers/heatmap.py` (`create_presence_heatmap`), `metaquest/plugins/visualizers/bar.py` (`create_grouped_bar_chart`; `tests/test_visualizers_bar_extended.py`), `metaquest/plugins/visualizers/map.py` (`create_choropleth`) (tests for the three plugin functions in `tests/test_plugins_comprehensive.py`), `metaquest/data/taxonomy.py` (`suggest_species_corrections`; `tests/test_taxonomy_extended.py`), `metaquest/data/genome_download.py` (`download_from_file`, `create_genome_manifest`, `check_datasets_available`; `tests/test_genome_download.py`), `metaquest/plugins/base.py` (`discover_plugins`, `register_discovered_plugins`; `tests/test_plugins_comprehensive.py`), `metaquest/utils/security.py` (`validate_file_path`; `tests/test_security_comprehensive.py`)
- Modify: `metaquest/core/constants.py` (drop `"--gzip"` from the fasterq-dump `safe_params`), `metaquest/utils/security.py` (drop `"--split-3"` from `FASTERQ_DUMP_BOOLEAN_FLAGS` and `"-e"` from `FASTERQ_DUMP_INTEGER_FLAGS`; add `"-0"` and `"-s"` to `PATH_VALUE_FLAGS`)
- Test: `tests/test_security_comprehensive.py`

- [ ] **Step 1: Write the failing validator test**

```python
class TestSamtoolsOutputPaths:
    def test_fastq_output_flags_are_path_validated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = ["fastq", "-1", "out/r1.fq.gz", "-2", "out/r2.fq.gz", "-s", "out/s.fq.gz", "-0", "out/o.fq.gz", "in.bam"]
        cmd = SecureSubprocess._build_validated_command("samtools", args)
        resolved = str((tmp_path / "out" / "o.fq.gz").resolve())
        assert cmd[cmd.index("-0") + 1] == resolved
        assert cmd[cmd.index("-s") + 1] == str((tmp_path / "out" / "s.fq.gz").resolve())

    def test_dead_fasterq_dump_flags_are_rejected(self):
        with pytest.raises(SecurityError):
            SecureSubprocess._build_validated_command("fasterq-dump", ["--gzip", "SRR000001"])
```

Run: `pytest tests/test_security_comprehensive.py -k "OutputPaths or dead_fasterq" -v` -> FAIL.

- [ ] **Step 2: Delete the dead functions**

For each function in the file list: delete the `def` (and any private helper only it used), then delete every test function or class in the named test file that references it (`grep -n "<name>" tests/<file>`), and remove now-unused imports until `flake8` is clean. `validate_file_path` in `metaquest/utils/security.py` is the module-level function at the bottom, not the class method. In `metaquest/plugins/base.py`, removing the two discovery functions also removes their `importlib`/`pkgutil` imports if nothing else uses them. Confirm nothing else references any removed name: `grep -rn "<name>" metaquest tests` must be empty for each.

- [ ] **Step 3: Trim the allowlist and validate the samtools output paths**

Apply the three edits in `metaquest/utils/security.py` and the one in `metaquest/core/constants.py`. Any test that used `--gzip` as an example of an allowed flag switches to `--split-files`.

- [ ] **Step 4: Verify, pipeline, commit**

Run: `make check && make test && make pipeline` and `metaquest --help`
Expected: all green; the help lists no removed command.

```bash
git add -A metaquest tests
git commit -m "refactor: delete unused functions and dead allowlist entries; validate samtools output paths

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 5: Real-data check**

Run (network, tools from the conda envs; not part of the suite):

```bash
PATH=$HOME/miniforge3/envs/sra-tools/bin:$PATH make test-network
cd /private/tmp/claude-501/-Users-andreassjodin-Code-metaquest/1de29173-6332-4d93-9042-6d9d3e6f9a0f/scratchpad/e2e \
  && metaquest branchwater_search --genome-fasta genomes/GCF_000008025.1.fna --threshold 0.1 --output branchwater_live/GCF_000008025.1.csv
```

Expected: the smoke test passes; the live search either lists matches or prints the zero-match warning and writes the header-only CSV (the index was empty on 2026-09-04). Record the outcome in the task report.

- [ ] **Step 6: Commit the planning documents**

```bash
git add docs/superpowers/specs/2026-09-04-metaquest-audit.md docs/superpowers/plans/2026-09-04-audit-repairs.md docs/superpowers/specs/2026-09-05-followup-design.md docs/superpowers/plans/2026-09-05-followup-implementation.md
git commit -m "docs: add the audit spec and the two implementation plans

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Verification after all four branches merge

- `make check`, `make test`, `make pipeline` green on `main`; `make test-network` passes with sra-tools on PATH.
- `metaquest --help` shows five groups, `COMMAND` in the usage line, no kebab names, no `sra_download`, `sra-download-intelligent` or `assemble_datasets`.
- `metaquest branchwater_search --help`, `download_sra --help` (with `--report-file`) and `taxonomic_summary --help` show defaults.
