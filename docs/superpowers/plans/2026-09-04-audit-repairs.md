# Audit Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SRA download layer work again, stop commands from reporting success with empty results, and repair the broken hand-offs between pipeline steps found by the 2026-09-04 audit.

**Architecture:** Four small branches, each a PR off `main` in order: `fix/download-layer` (validator, path allowlist, network smoke test), `fix/honest-results` (extraction counts, analytics FASTQ locator, plot reader, gz suffix), `feat/workflow-handoffs` (GTDB parsing, metadata default resolution, TSV/CSV matrix loader, new `select_datasets` command), `docs/audit-followup` (cartopy noise, executed docs, environment file). Every task is red-green-refactor with `run_secure` or `subprocess.run` mocked; one opt-in network test covers the real fasterq-dump path.

**Tech Stack:** Python 3.12, argparse command registry (`metaquest/cli/base.py`), pandas, pytest with `unittest.mock`, black/flake8/mypy via `make check`.

**Spec:** `docs/superpowers/specs/2026-09-04-metaquest-audit.md`

## Global Constraints

- Python >= 3.12; black line length 120; `make check` (black, flake8, mypy, radon) and `make test` must pass after every task.
- CLI arguments use dashes (`--fastq-dir`), never underscores.
- Unit tests never run external tools; patch `metaquest.utils.security.subprocess.run` or `SecureSubprocess.run_secure`.
- Plain, modest language in help text, log messages and docs. No Unicode symbols in new log or help strings.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## Phase 1: download layer (branch `fix/download-layer`)

### Task 1: Validate fasterq-dump arguments by flag semantics, not position

**Files:**
- Modify: `metaquest/utils/security.py:146-186` (`_build_validated_command`)
- Modify: `tests/test_security_comprehensive.py:471-476` (delete the wrong-contract test)
- Test: `tests/test_security_comprehensive.py`, `tests/test_data_sra.py`

**Interfaces:**
- Produces: `SecureSubprocess._build_validated_command(executable: str, args: List[str]) -> List[str]` unchanged in signature; new module constants `FASTERQ_DUMP_BOOLEAN_FLAGS`, `FASTERQ_DUMP_INTEGER_FLAGS`, `PATH_VALUE_FLAGS` in `metaquest/utils/security.py`.

- [ ] **Step 1: Write the failing contract tests**

Append to `tests/test_security_comprehensive.py` (before the `# SUCCESS METRICS` banner), and delete `test_fasterq_dump_flag_value_at_index_one_validated_as_accession` (lines 471-476):

```python
class TestFasterqDumpCommandContract:
    """The exact argument list metaquest.data.sra builds must pass validation unchanged."""

    def test_real_download_argument_order_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = ["--threads", "4", "--progress", "SRR2517620", "-O", "out/SRR2517620_temp"]
        cmd = SecureSubprocess._build_validated_command("fasterq-dump", args)
        expected_out = str((tmp_path / "out" / "SRR2517620_temp").resolve())
        assert cmd == ["fasterq-dump", "--threads", "4", "--progress", "SRR2517620", "-O", expected_out]

    def test_non_numeric_thread_count_rejected(self):
        with pytest.raises(SecurityError, match="Invalid integer value for --threads"):
            SecureSubprocess._build_validated_command("fasterq-dump", ["--threads", "four", "SRR000001"])

    def test_bad_accession_after_boolean_flag_rejected(self):
        with pytest.raises(SecurityError, match="Invalid SRA accession format"):
            SecureSubprocess._build_validated_command("fasterq-dump", ["--progress", "not-an-accession"])

    def test_split_files_does_not_swallow_accession(self):
        cmd = SecureSubprocess._build_validated_command("fasterq-dump", ["--split-files", "SRR000001"])
        assert cmd == ["fasterq-dump", "--split-files", "SRR000001"]
```

Append to `tests/test_data_sra.py` inside the class that holds `test_download_accession_success`:

```python
    def test_download_accession_command_passes_validation(self, tmp_path, monkeypatch):
        """The command download_accession builds must survive SecureSubprocess validation unmocked."""
        monkeypatch.chdir(tmp_path)
        with patch("metaquest.utils.security.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            with patch("metaquest.data.sra._handle_download_output", return_value=(True, "Downloaded 2 files")):
                success, message = download_accession(
                    "SRR2517620", tmp_path / "fastq", num_threads=4, temp_folder=tmp_path / "tmp"
                )
        assert success is True, message
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "fasterq-dump"
        assert cmd[1:4] == ["--threads", "4", "--progress"]
        assert cmd[4] == "SRR2517620"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_security_comprehensive.py::TestFasterqDumpCommandContract tests/test_data_sra.py -k "passes_validation" -v`
Expected: `test_real_download_argument_order_is_accepted` and `test_download_accession_command_passes_validation` FAIL with `SecurityError: Invalid SRA accession format: 4`; `test_non_numeric_thread_count_rejected` FAIL (no error raised or wrong message).

- [ ] **Step 3: Rewrite `_build_validated_command`**

In `metaquest/utils/security.py`, add after `logger = logging.getLogger(__name__)`:

```python
# fasterq-dump flags that never take a value; any other allowlisted fasterq-dump
# flag consumes the following token as its value.
FASTERQ_DUMP_BOOLEAN_FLAGS = frozenset(
    {"--progress", "--split-files", "--split-3", "--skip-technical", "--include-technical", "--force", "--gzip"}
)
# fasterq-dump flags whose value must be a non-negative integer.
FASTERQ_DUMP_INTEGER_FLAGS = frozenset({"--threads", "-e"})
# Flags (any tool) whose value is a filesystem path and must pass validate_path.
PATH_VALUE_FLAGS = frozenset({"-O", "-o", "--out-dir", "--temp", "-1", "-2"})
```

Replace the body of `_build_validated_command` (keep the docstring) with:

```python
        cmd = [cls.validate_executable(executable)]

        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith("-"):
                cls.validate_parameter(executable, arg)
                cmd.append(arg)

                takes_value = not (executable == "fasterq-dump" and arg in FASTERQ_DUMP_BOOLEAN_FLAGS)
                if takes_value and i + 1 < len(args) and not args[i + 1].startswith("-"):
                    i += 1
                    value = args[i]
                    if arg in PATH_VALUE_FLAGS:
                        value = str(cls.validate_path(value))
                    elif executable == "fasterq-dump" and arg in FASTERQ_DUMP_INTEGER_FLAGS:
                        if not value.isdigit():
                            raise SecurityError(f"Invalid integer value for {arg}: {value}")
                    cmd.append(value)
            else:
                # Positional argument: for fasterq-dump this is the SRA accession.
                if executable == "fasterq-dump":
                    arg = cls.validate_accession_for_subprocess(arg)
                cmd.append(arg)

            i += 1

        return cmd
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_security_comprehensive.py tests/test_data_sra.py -v`
Expected: all PASS. If `test_parse_standalone_argument` or other older tests fail because they relied on the old index rule, update their expected command list to the new behaviour (a flag value is never validated as an accession).

- [ ] **Step 5: Run the full gates and commit**

Run: `make check && make test`
Expected: both green.

```bash
git checkout -b fix/download-layer main
git add metaquest/utils/security.py tests/test_security_comprehensive.py tests/test_data_sra.py
git commit -m "fix: validate fasterq-dump accession by flag semantics, not argument index

Every download command passed --threads N before the accession, so the thread
count was validated as an accession and all downloads failed before starting.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 2: Allow system temp dirs and user-chosen output roots in `validate_path`

**Files:**
- Modify: `metaquest/utils/security.py` (imports, `validate_path`, new `allowed_roots`, `add_allowed_root`)
- Modify: `metaquest/data/sra.py:200-236` (`download_accession`), `metaquest/data/sra.py:73-108` (`_prepare_temp_folder`)
- Modify: `metaquest/data/sra_enhanced.py:85` (`download_accession_enhanced`)
- Modify: `metaquest/data/read_extraction.py` (`extract_target_reads`, `assemble_extracted_reads`)
- Test: `tests/test_security_comprehensive.py`, `tests/test_data_sra.py`

**Interfaces:**
- Produces: `SecureSubprocess.allowed_roots() -> List[Path]`, `SecureSubprocess.add_allowed_root(path: Union[str, Path]) -> Path`, `SecureSubprocess._extra_roots: List[Path]` (class attribute, cleared by tests).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_security_comprehensive.py`:

```python
class TestAllowedRoots:
    """validate_path accepts system temp dirs and roots registered from user-supplied folders."""

    @pytest.fixture(autouse=True)
    def _reset_roots(self):
        SecureSubprocess._extra_roots.clear()
        yield
        SecureSubprocess._extra_roots.clear()

    def test_system_tempdir_is_allowed(self):
        import shutil
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        try:
            target = temp_dir / "reads"
            assert SecureSubprocess.validate_path(target) == target.resolve()
        finally:
            shutil.rmtree(temp_dir)

    def test_registered_root_permits_external_folder(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        monkeypatch.setattr(
            SecureSubprocess, "allowed_roots", classmethod(lambda cls: [work.resolve()] + list(cls._extra_roots))
        )
        with pytest.raises(SecurityError, match="outside allowed directories"):
            SecureSubprocess.validate_path(external / "out.sam")

        SecureSubprocess.add_allowed_root(external)
        assert SecureSubprocess.validate_path(external / "out.sam") == (external / "out.sam").resolve()

    def test_root_prefix_collision_rejected(self, tmp_path, monkeypatch):
        root = tmp_path / "data"
        root.mkdir()
        sibling = tmp_path / "data2"
        sibling.mkdir()
        monkeypatch.setattr(SecureSubprocess, "allowed_roots", classmethod(lambda cls: [root.resolve()]))
        with pytest.raises(SecurityError, match="outside allowed directories"):
            SecureSubprocess.validate_path(sibling / "x.txt")

    def test_add_allowed_root_is_idempotent(self, tmp_path):
        SecureSubprocess.add_allowed_root(tmp_path)
        SecureSubprocess.add_allowed_root(tmp_path)
        assert SecureSubprocess._extra_roots.count(tmp_path.resolve()) == 1
```

Append to `tests/test_data_sra.py` (same class as Task 1):

```python
    def test_download_accession_registers_output_and_temp_roots(self, tmp_path, monkeypatch):
        from metaquest.utils.security import SecureSubprocess

        SecureSubprocess._extra_roots.clear()
        monkeypatch.chdir(tmp_path)
        with patch("metaquest.utils.security.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            with patch("metaquest.data.sra._handle_download_output", return_value=(True, "Downloaded 2 files")):
                download_accession("SRR2517620", tmp_path / "fastq", temp_folder=tmp_path / "scratch")
        assert (tmp_path / "fastq").resolve() in SecureSubprocess._extra_roots
        assert (tmp_path / "scratch").resolve() in SecureSubprocess._extra_roots
        SecureSubprocess._extra_roots.clear()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_security_comprehensive.py::TestAllowedRoots tests/test_data_sra.py -k "registers_output" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'allowed_roots'` / `'_extra_roots'`.

- [ ] **Step 3: Implement roots in `SecureSubprocess`**

In `metaquest/utils/security.py` add `import tempfile` to the imports. Inside `class SecureSubprocess`, after `SAFE_PARAMETERS = ...`, add:

```python
    # Directories registered at runtime from user-supplied output or temp folders.
    _extra_roots: List[Path] = []

    @classmethod
    def allowed_roots(cls) -> List[Path]:
        """Directories under which validated paths may fall."""
        roots = [Path.cwd(), Path.home(), Path("/tmp"), Path(tempfile.gettempdir())]
        roots.extend(cls._extra_roots)
        return [root.resolve() for root in roots]

    @classmethod
    def add_allowed_root(cls, path: Union[str, Path]) -> Path:
        """Permit paths under a directory the user chose on the command line."""
        resolved = Path(path).resolve()
        if resolved not in cls._extra_roots:
            cls._extra_roots.append(resolved)
        return resolved
```

Replace `validate_path` (make it a classmethod; external callers keep using `SecureSubprocess.validate_path(...)`):

```python
    @classmethod
    def validate_path(cls, path: Union[str, Path], allow_creation: bool = True) -> Path:
        """
        Validate and sanitize a file or directory path.

        A path is accepted when it lies under the working directory, the home
        directory, the system temp directory, or a root registered with
        ``add_allowed_root``. Parent-directory segments are rejected when the
        caller does not create the path.

        Raises:
            SecurityError: If the path is unsafe
        """
        path_obj = Path(path).resolve()

        if not any(path_obj == root or root in path_obj.parents for root in cls.allowed_roots()):
            raise SecurityError(f"Path outside allowed directories: {path}")

        if not allow_creation and ".." in Path(path).parts:
            raise SecurityError(f"Unsafe path component in: {path}")

        return path_obj
```

- [ ] **Step 4: Register roots at the four call sites**

`metaquest/data/sra.py`, in `download_accession` immediately after `output_path = Path(output_folder) / accession`:

```python
    SecureSubprocess.add_allowed_root(Path(output_folder))
```

`metaquest/data/sra.py`, in `_prepare_temp_folder` inside the `else` branch that logs "Using temp folder", before the `return temp_path_obj`:

```python
            SecureSubprocess.add_allowed_root(temp_path_obj)
```

(`SecureSubprocess` is already imported in `metaquest/data/sra.py`; confirm with `grep -n "SecureSubprocess" metaquest/data/sra.py`.)

`metaquest/data/sra_enhanced.py`, in `download_accession_enhanced` immediately after line 85 `output_path = Path(output_folder) / accession`:

```python
        SecureSubprocess.add_allowed_root(Path(output_folder))
```

`metaquest/data/read_extraction.py`, in `extract_target_reads` after `output_root = Path(output_folder)`:

```python
    SecureSubprocess.add_allowed_root(output_root)
```

and in `assemble_extracted_reads` after `out_dir = Path(output_dir)`:

```python
    SecureSubprocess.add_allowed_root(out_dir.parent)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_security_comprehensive.py tests/test_data_sra.py tests/test_read_extraction.py tests/test_sra_enhanced.py tests/test_sra_enhanced_extended.py -v`
Expected: all PASS. `test_validate_path_outside_allowed_dirs` still passes (system paths are not under any root).

- [ ] **Step 6: Gates and commit**

Run: `make check && make test`

```bash
git add metaquest/utils/security.py metaquest/data/sra.py metaquest/data/sra_enhanced.py metaquest/data/read_extraction.py tests/test_security_comprehensive.py tests/test_data_sra.py
git commit -m "fix: accept system temp dirs and user-chosen output roots in path validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 3: Opt-in network smoke test for the real fasterq-dump path

**Files:**
- Modify: `pyproject.toml:150-153` (`[tool.pytest.ini_options]`)
- Modify: `Makefile:21-22`
- Create: `tests/test_network_smoke.py`
- Modify: `CONTRIBUTING.md` (one paragraph)

**Interfaces:**
- Consumes: `metaquest.data.sra.download_accession(accession, output_folder, num_threads=4, force=False, temp_folder=None) -> Tuple[bool, str]`.

- [ ] **Step 1: Register the marker and exclude it by default**

In `pyproject.toml` replace the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers -m 'not network'"
testpaths = ["tests"]
markers = [
    "network: calls NCBI or SRA over the internet; run explicitly with 'pytest -m network'",
]
```

In `Makefile` add after the `test:` target:

```make
test-network:
	pytest tests/test_network_smoke.py -m network -x -v
```

- [ ] **Step 2: Write the smoke test**

Create `tests/test_network_smoke.py`:

```python
"""Opt-in smoke tests that talk to NCBI SRA (run with: make test-network).

SRR2517620 is a 425-spot MiSeq mosquito metagenome; fasterq-dump fetches it in
a few seconds, which is enough to prove the download command line and the
output layout end to end.
"""

import shutil
import subprocess

import pytest

from metaquest.data.sra import download_accession

pytestmark = pytest.mark.network

TINY_RUN = "SRR2517620"
TINY_RUN_SPOTS = 425

needs_fasterq_dump = pytest.mark.skipif(shutil.which("fasterq-dump") is None, reason="fasterq-dump not on PATH")
needs_cli = pytest.mark.skipif(shutil.which("metaquest") is None, reason="metaquest CLI not on PATH")


def _read_count(path):
    with open(path) as handle:
        return sum(1 for _ in handle) // 4


@needs_fasterq_dump
def test_download_accession_writes_paired_fastq(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok, message = download_accession(TINY_RUN, tmp_path / "fastq", num_threads=2)
    assert ok, message

    files = sorted(p.name for p in (tmp_path / "fastq" / TINY_RUN).glob("*.fastq"))
    assert files == [f"{TINY_RUN}_1.fastq", f"{TINY_RUN}_2.fastq"]
    assert _read_count(tmp_path / "fastq" / TINY_RUN / f"{TINY_RUN}_1.fastq") == TINY_RUN_SPOTS


@needs_fasterq_dump
@needs_cli
def test_download_sra_cli_round_trip(tmp_path):
    (tmp_path / "accessions.txt").write_text(f"{TINY_RUN}\n")
    result = subprocess.run(
        ["metaquest", "download_sra", "--accessions-file", "accessions.txt", "--fastq-folder", "fastq"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "fastq" / TINY_RUN / f"{TINY_RUN}_1.fastq").exists()
    assert "Successfully downloaded: 1 datasets" in result.stderr + result.stdout
```

- [ ] **Step 3: Run it both ways**

Run: `pytest tests/test_network_smoke.py -v`
Expected: `2 deselected` (marker excluded by default).

Run (needs the `sra-tools` env on PATH, e.g. `PATH=$HOME/miniforge3/envs/sra-tools/bin:$PATH make test-network`):
Expected: 2 PASS in under 60 s. If this fails with `Invalid SRA accession format` or `Path outside allowed directories`, Tasks 1 or 2 are incomplete.

- [ ] **Step 4: Document and commit**

Add to `CONTRIBUTING.md` under the testing section:

```markdown
### Network smoke test

`make test-network` downloads one 425-spot SRA run (SRR2517620) with fasterq-dump and
checks the files MetaQuest writes. It needs `fasterq-dump` on the PATH and internet
access, so it is excluded from `make test`. Run it before releasing any change to
`metaquest/data/sra.py` or `metaquest/utils/security.py`.
```

```bash
git add pyproject.toml Makefile tests/test_network_smoke.py CONTRIBUTING.md
git commit -m "test: add opt-in network smoke test for the fasterq-dump download path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `fix/download-layer` -> `main`.

---

## Phase 2: honest results (branch `fix/honest-results`, based on `fix/download-layer`)

### Task 4: Count mapped reads in `extract_target_reads` and never report success for zero

**Files:**
- Modify: `metaquest/core/constants.py:137-150` (samtools `safe_params`: add `"-c"`)
- Modify: `metaquest/data/read_extraction.py:60-172`
- Modify: `metaquest/cli/commands/read_extraction.py:60-95`
- Test: `tests/test_read_extraction.py`, `tests/test_cli_read_extraction.py`

**Interfaces:**
- Produces: `ExtractionResult(files: List[Path], mapped_records: int, unequal_mates: bool)` dataclass; `_map_and_extract(...) -> ExtractionResult`; `extract_target_reads(...) -> Dict[str, List[Path]]` unchanged (a sample with zero mapped reads maps to `[]`); `UNEQUAL_MATES_MARKER = "different number of records"`.

- [ ] **Step 1: Add a tool simulator to the tests and write the failing tests**

At the top of `tests/test_read_extraction.py` (after the imports) add:

```python
import gzip


UNEQUAL_WARNING = "[W::mm_bseq_read_frag2] query files have different number of records; extra records skipped."


def _fake_tools(state):
    """Stand in for minimap2/samtools: record calls and create the FASTQ files samtools would write.

    state keys: mapped (int, default 10), nonempty (flags whose output file gets a read,
    default ("-1", "-2")), unequal (bool, emit minimap2's mate-count warning).
    """

    def run(executable, args, **kwargs):
        state.setdefault("calls", []).append((executable, list(args)))
        result = MagicMock(returncode=0, stdout="", stderr="")
        if executable == "minimap2" and state.get("unequal"):
            result.stderr = UNEQUAL_WARNING
        if executable == "samtools" and args[:2] == ["view", "-c"]:
            result.stdout = f"{state.get('mapped', 10)}\n"
        if executable == "samtools" and args[0] == "fastq":
            for flag in ("-1", "-2", "-0", "-s"):
                if flag in args and flag in state.get("nonempty", ("-1", "-2")):
                    path = Path(args[args.index(flag) + 1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(path, "wt") as handle:
                        handle.write("@r1\nACGT\n+\nIIII\n")
        return result

    return run
```

In `TestExtractTargetReads`, change the two existing tests so the mock creates files, and extend their assertions:

```python
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_paired_extraction_command_construction(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
            )
            assert list(results) == ["SRR1"]
            assert [p.name for p in results["SRR1"]] == ["GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"]
            # The orphan file (-0) is removed when empty.
            assert not (root / "targeted" / "SRR1" / "GCF_1_0.fastq.gz").exists()

        tools = [c[0] for c in state["calls"]]
        assert tools == ["minimap2", "samtools", "samtools", "samtools"]
        assert state["calls"][2][1][:2] == ["view", "-c"]
        fastq_args = state["calls"][3][1]
        assert fastq_args[0] == "fastq"
        assert all(flag in fastq_args for flag in ("-1", "-2", "-s", "-0"))

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_single_end_uses_flag_0(self, mock_run):
        state = {"nonempty": ("-0",)}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=False)
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
            )
            assert [p.name for p in results["SRR1"]] == ["GCF_1.fastq.gz"]
        fastq_args = state["calls"][3][1]
        assert "-0" in fastq_args
```

Add three new tests to the same class:

```python
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_zero_mapped_records_writes_nothing(self, mock_run, caplog):
        state = {"mapped": 0}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            with caplog.at_level("WARNING"):
                results = extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                )
            assert results == {"SRR1": []}
            assert list((root / "targeted" / "SRR1").glob("*.fastq.gz")) == []
        assert "No reads from SRR1 mapped to GCF_1" in caplog.text
        # samtools fastq is never run when nothing mapped.
        assert [c[0] for c in state["calls"]] == ["minimap2", "samtools", "samtools"]

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_unequal_mates_fall_back_to_orphan_file(self, mock_run, caplog):
        state = {"unequal": True, "nonempty": ("-0",)}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            with caplog.at_level("WARNING"):
                results = extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                )
            assert [p.name for p in results["SRR1"]] == ["GCF_1_0.fastq.gz"]
        assert "different read counts" in caplog.text

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_mapped_count_is_logged(self, mock_run, caplog):
        state = {"mapped": 1234}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            with caplog.at_level("INFO"):
                extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                )
        assert "1234 mapped records for SRR1" in caplog.text
```

In `tests/test_cli_read_extraction.py`, add a test and update the mocks of `test_execute_extracts` and `test_execute_with_assembly` to use the same simulator (copy `_fake_tools` and `UNEQUAL_WARNING` into this file, or move both into a new `tests/helpers_extraction.py` and import from there):

```python
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_returns_1_when_no_sample_yields_reads(self, mock_run):
        mock_run.side_effect = _fake_tools({"mapped": 0})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                )
            )
        assert rc == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_read_extraction.py tests/test_cli_read_extraction.py -v`
Expected: the new tests FAIL (`view -c` never called; orphan file logic missing; rc == 0).

- [ ] **Step 3: Allow `samtools view -c`**

In `metaquest/core/constants.py`, inside the `"samtools"` `safe_params` set, add `"-c",` after `"-f",`.

- [ ] **Step 4: Implement counting and orphan handling**

In `metaquest/data/read_extraction.py` add imports `import gzip` and `from dataclasses import dataclass`. After `MINIMAP2_PRESETS`, add:

```python
# minimap2 prints this when the two mate files differ in length; it then maps single-end.
UNEQUAL_MATES_MARKER = "different number of records"


@dataclass
class ExtractionResult:
    """Outcome of mapping one sample against the target genome."""

    files: List[Path]
    mapped_records: int
    unequal_mates: bool = False


def _count_bam_records(bam_path: Path) -> int:
    """Number of records in a BAM file, via ``samtools view -c``."""
    result = SecureSubprocess.run_secure("samtools", ["view", "-c", str(bam_path)])
    text = (result.stdout or "").strip()
    return int(text) if text.isdigit() else 0


def _fastq_is_empty(path: Path) -> bool:
    """True when the file is missing or has no records (gzip or plain)."""
    if not path.exists():
        return True
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return handle.readline() == ""
```

Replace `_map_and_extract` with:

```python
def _map_and_extract(
    accession: str,
    reads: List[Path],
    genome_fasta: Path,
    out_dir: Path,
    genome_id: str,
    preset: str,
    threads: int,
) -> ExtractionResult:
    """Map one sample's reads to the target genome and write the mapped reads.

    Returns the FASTQ files written (two for paired input, one otherwise, none when
    nothing mapped) together with the mapped-record count. Intermediate SAM/BAM
    files are removed.
    """
    ensure_directory(out_dir)
    sam_path = out_dir / f"{genome_id}.sam"
    bam_path = out_dir / f"{genome_id}.mapped.bam"

    # 1. Align reads to the reference (SAM output).
    minimap_args = ["-a", "-x", preset, "-t", str(threads), "-o", str(sam_path), str(genome_fasta)]
    minimap_args.extend(str(r) for r in reads)
    aligned = SecureSubprocess.run_secure("minimap2", minimap_args)
    unequal = UNEQUAL_MATES_MARKER in (aligned.stderr or "")
    if unequal:
        logger.warning(
            "%s: the mate files have different read counts, so minimap2 mapped them as single-end reads; "
            "re-download with fasterq-dump (which keeps mates in step) for paired extraction",
            accession,
        )

    # 2. Keep only mapped records (-F 4 drops the unmapped flag) and count them.
    SecureSubprocess.run_secure(
        "samtools", ["view", "-b", "-F", "4", "-@", str(threads), "-o", str(bam_path), str(sam_path)]
    )
    mapped = _count_bam_records(bam_path)
    if mapped == 0:
        logger.warning("No reads from %s mapped to %s; nothing written", accession, genome_id)
        for tmp in (sam_path, bam_path):
            tmp.unlink(missing_ok=True)
        return ExtractionResult([], 0, unequal)

    # 3. Export mapped reads back to FASTQ. Reads without a mate flag (single-end
    #    mapping, or a fallback after unequal mates) go to the -0 file.
    if len(reads) >= 2:
        out1 = out_dir / f"{genome_id}_1.fastq.gz"
        out2 = out_dir / f"{genome_id}_2.fastq.gz"
        singles = out_dir / f"{genome_id}_s.fastq.gz"
        orphans = out_dir / f"{genome_id}_0.fastq.gz"
        SecureSubprocess.run_secure(
            "samtools",
            ["fastq", "-1", str(out1), "-2", str(out2), "-s", str(singles), "-0", str(orphans), str(bam_path)],
        )
        for path in (out1, out2, singles, orphans):
            if _fastq_is_empty(path):
                path.unlink(missing_ok=True)
        if out1.exists() and out2.exists():
            written = [out1, out2]
        elif orphans.exists():
            written = [orphans]
        else:
            written = [singles] if singles.exists() else []
    else:
        out0 = out_dir / f"{genome_id}.fastq.gz"
        SecureSubprocess.run_secure("samtools", ["fastq", "-0", str(out0), str(bam_path)])
        written = [] if _fastq_is_empty(out0) else [out0]

    for tmp in (sam_path, bam_path):
        tmp.unlink(missing_ok=True)

    return ExtractionResult(written, mapped, unequal)
```

In `extract_target_reads`, replace the two lines that call `_map_and_extract` and log with:

```python
        outcome = _map_and_extract(accession, reads, genome_path, output_root / accession, genome_id, preset, threads)
        results[accession] = outcome.files
        if outcome.files:
            logger.info(
                "Extracted %d mapped records for %s -> %s",
                outcome.mapped_records,
                accession,
                ", ".join(str(p) for p in outcome.files),
            )
```

- [ ] **Step 5: Make the CLI exit 1 when no sample produced reads**

In `metaquest/cli/commands/read_extraction.py` `execute`, replace `self.logger.info("Extracted reads for %d sample(s)", len(results))` and the assembly loop with:

```python
            with_reads = {acc: files for acc, files in results.items() if files}
            self.logger.info("Extracted reads for %d of %d sample(s)", len(with_reads), len(results))
            if results and not with_reads:
                self.logger.error(
                    "No reads mapped to %s in any sample; check the FASTQ files and --preset", args.genome_id
                )
                return 1

            if args.assemble:
                asm_threads = resolve_assembly_threads(args.assembly_threads, args.threads)
                if args.assembly_threads is None and asm_threads < args.threads:
                    self.logger.info(
                        "Running megahit single-threaded on macOS (its parallel sort is unstable here); "
                        "override with --assembly-threads"
                    )
                for accession, reads in with_reads.items():
                    out_dir = Path(args.output_folder) / accession / f"{args.genome_id}_assembly"
                    assemble_extracted_reads(reads, out_dir, threads=asm_threads, min_contig_len=args.min_contig_len)
                self.logger.info("Assembled %d sample(s)", len(with_reads))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_read_extraction.py tests/test_cli_read_extraction.py tests/test_security_comprehensive.py -v`
Expected: all PASS.

- [ ] **Step 7: Gates and commit**

Run: `make check && make test`

```bash
git checkout -b fix/honest-results fix/download-layer
git add metaquest/core/constants.py metaquest/data/read_extraction.py metaquest/cli/commands/read_extraction.py tests/test_read_extraction.py tests/test_cli_read_extraction.py
git commit -m "fix: count mapped reads in extract_target_reads and fail on empty output

Unequal mate files made minimap2 map single-end and samtools fastq -1/-2 wrote
nothing while the command reported success. Reads are now counted, unflagged
reads go to a -0 file, and the command exits 1 when no sample yields reads.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 5: Find FASTQ in the real download layout; dashboard and compare exit 1 without data

**Files:**
- Modify: `metaquest/sra/analytics.py:336-341` (`SRADatasetAnalyzer.__init__`), `:360` (call site), `:644-658` (locator)
- Modify: `metaquest/sra/reporting.py:16,55-58`
- Modify: `metaquest/cli/commands/sra_intelligent.py:67,254,596` (defaults), `:506-571` (dashboard execute), `:678-717` (compare execute), compare/dashboard `configure_parser`
- Test: `tests/test_sra_analytics.py:192-211,302`, `tests/test_cli_sra_intelligent.py`

**Interfaces:**
- Produces: `SRADatasetAnalyzer(fastq_dir: Optional[Union[str, Path]] = None)`; `SRADatasetAnalyzer.find_fastq(accession: str) -> Optional[Path]` (replaces `_locate_fastq_file`); `SRAReportGenerator(output_dir, fastq_dir=None)`.

- [ ] **Step 1: Write the failing tests**

Replace `test_locate_fastq_file` and `test_locate_fastq_file_not_found` in `tests/test_sra_analytics.py` with:

```python
    def test_find_fastq_in_per_accession_folder(self, tmp_path):
        acc_dir = tmp_path / "fastq" / "SRR123456"
        acc_dir.mkdir(parents=True)
        (acc_dir / "SRR123456_2.fastq").write_text("")
        r1 = acc_dir / "SRR123456_1.fastq"
        r1.write_text("")
        analyzer = SRADatasetAnalyzer(fastq_dir=tmp_path / "fastq")
        assert analyzer.find_fastq("SRR123456") == r1

    def test_find_fastq_flat_layout(self, tmp_path):
        flat = tmp_path / "fastq" / "SRR123456.fastq.gz"
        flat.parent.mkdir(parents=True)
        flat.write_text("")
        analyzer = SRADatasetAnalyzer(fastq_dir=tmp_path / "fastq")
        assert analyzer.find_fastq("SRR123456") == flat

    def test_find_fastq_defaults_to_fastq_folder_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r1 = tmp_path / "fastq" / "SRR1" / "SRR1_1.fastq"
        r1.parent.mkdir(parents=True)
        r1.write_text("")
        assert SRADatasetAnalyzer().find_fastq("SRR1") == Path("fastq/SRR1/SRR1_1.fastq")

    def test_find_fastq_missing_returns_none(self, tmp_path):
        assert SRADatasetAnalyzer(fastq_dir=tmp_path).find_fastq("NONE") is None
```

Change line 302 `patch.object(self.analyzer, "_locate_fastq_file", ...)` to `patch.object(self.analyzer, "find_fastq", ...)`.

In `tests/test_cli_sra_intelligent.py` add (it already imports `argparse`; add the import if not):

```python
class TestHonestExits:
    def test_compare_returns_1_without_fastq(self, tmp_path):
        from metaquest.cli.commands.sra_intelligent import SRAComparativeAnalysisCommand

        groups = tmp_path / "groups.json"
        groups.write_text('{"a": ["SRR000001"], "b": ["SRR000002"]}')
        args = argparse.Namespace(
            groups_file=str(groups),
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "out"),
            statistical_tests=False,
            generate_report=False,
        )
        assert SRAComparativeAnalysisCommand().execute(args) == 1

    def test_dashboard_returns_1_without_fastq(self, tmp_path):
        from metaquest.cli.commands.sra_intelligent import SRAInteractiveDashboardCommand

        acc_file = tmp_path / "acc.txt"
        acc_file.write_text("SRR000001\n")
        args = argparse.Namespace(
            accessions_file=str(acc_file),
            download_session=None,
            quality_profiles=None,
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "dash"),
            title="t",
            dashboard_type="quality",
            no_open=True,
        )
        assert SRAInteractiveDashboardCommand().execute(args) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_sra_analytics.py -k find_fastq -v` and the two CLI tests.
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'fastq_dir'`; CLI returns 0).

- [ ] **Step 3: Implement the locator**

In `metaquest/sra/analytics.py` replace `SRADatasetAnalyzer.__init__` with:

```python
    def __init__(self, fastq_dir: Optional[Union[str, Path]] = None):
        self.quality_analyzer = SequenceQualityAnalyzer()
        self.fastq_dir = Path(fastq_dir) if fastq_dir else None
```

Replace `_locate_fastq_file` with:

```python
    def find_fastq(self, accession: str) -> Optional[Path]:
        """Locate one FASTQ file for an accession under the configured download folder.

        Downloads live in ``<folder>/<accession>/<accession>_1.fastq`` (fasterq-dump
        layout); flat ``<folder>/<accession>.fastq.gz`` files are accepted too. When no
        folder was given, ``fastq`` and then ``sra_downloads`` in the working directory
        are searched. R1 sorts before R2.
        """
        roots = [self.fastq_dir] if self.fastq_dir else [Path("fastq"), Path("sra_downloads")]
        for root in roots:
            for pattern in (f"{accession}/{accession}*.fastq*", f"{accession}*.fastq*"):
                matches = sorted(p for p in root.glob(pattern) if p.is_file())
                if matches:
                    return matches[0]
        return None
```

Change line 360 to `fastq_path = self.find_fastq(accession)`.

In `metaquest/sra/reporting.py` change line 16 to `from typing import Dict, List, Optional, Union, Any` and replace the constructor at lines 55-58 with:

```python
    def __init__(self, output_dir: Union[str, Path], fastq_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = SRADatasetAnalyzer(fastq_dir=fastq_dir)
```

- [ ] **Step 4: Wire the CLI**

In `metaquest/cli/commands/sra_intelligent.py`:

1. Change the three `default="sra_downloads"` (lines 67, 254, 596) to `default="fastq"`.
2. In `SRAInteractiveDashboardCommand.configure_parser` add:

```python
        parser.add_argument("--fastq-dir", default="fastq", help="Directory containing downloaded FASTQ files")
```

3. In `SRAInteractiveDashboardCommand.execute`, replace `reporter = SRAReportGenerator(output_dir=str(output_dir))` with:

```python
            reporter = SRAReportGenerator(output_dir=str(output_dir), fastq_dir=args.fastq_dir)
            missing = [acc for acc in accessions if reporter.analyzer.find_fastq(acc) is None]
            if len(missing) == len(accessions):
                logger.error(
                    "No FASTQ files found for any of %d accession(s) under %s", len(accessions), args.fastq_dir
                )
                return 1
            if missing:
                logger.warning("FASTQ missing for %d accession(s), e.g. %s", len(missing), ", ".join(missing[:5]))
```

4. In `SRAComparativeAnalysisCommand.execute`, replace `comparison = SRADatasetAnalyzer().compare_datasets(groups)` with:

```python
            comparison = SRADatasetAnalyzer(fastq_dir=args.fastq_dir).compare_datasets(groups)
            if not comparison.summary_statistics:
                logger.error("No FASTQ data was read for any accession under %s; nothing to compare", args.fastq_dir)
                return 1
```

and `SRAReportGenerator(output_dir=str(output_dir))` in the same method with `SRAReportGenerator(output_dir=str(output_dir), fastq_dir=args.fastq_dir)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_sra_analytics.py tests/test_sra_reporting_starter.py tests/test_sra_reporting_extended.py tests/test_cli_sra_intelligent.py tests/test_cli_commands.py -v`
Expected: all PASS. Any test that constructs `argparse.Namespace` for the dashboard command must now include `fastq_dir`.

- [ ] **Step 6: Gates and commit**

Run: `make check && make test`

```bash
git add metaquest/sra/analytics.py metaquest/sra/reporting.py metaquest/cli/commands/sra_intelligent.py tests/
git commit -m "fix: locate FASTQ in the per-accession layout and fail dashboards without data

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 6: Read `count_metadata` output in `plot_metadata_counts` and save bar charts

**Files:**
- Modify: `metaquest/visualization/plots.py:215-226` (`_load_counts_df`), `:305-315` (bar branch)
- Test: `tests/test_visualization_plots.py`

**Interfaces:**
- Produces: `_load_counts_df(file_path, limit) -> pd.DataFrame` with columns `category`, `count`; bar output file named `<input stem>_bar.<format>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_visualization_plots.py`:

```python
class TestCountsLoader:
    def test_header_row_and_genome_columns_are_summed(self, tmp_path):
        from metaquest.visualization.plots import _load_counts_df

        f = tmp_path / "metadata_counts.txt"
        f.write_text("Sample_Scientific_Name\tGCF_A\tGCF_B\nmetagenome\t3\t0\nfood metagenome\t0\t2\n")
        df = _load_counts_df(f, limit=20)
        assert list(df["category"]) == ["metagenome", "food metagenome"]
        assert list(df["count"]) == [3, 2]

    def test_headerless_two_columns(self, tmp_path):
        from metaquest.visualization.plots import _load_counts_df

        f = tmp_path / "stats.txt"
        f.write_text("GCF_B\t7\nGCF_A\t9\n")
        df = _load_counts_df(f, limit=20)
        assert list(df["category"]) == ["GCF_A", "GCF_B"]
        assert list(df["count"]) == [9, 7]

    def test_bar_plot_saves_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "counts.txt"
        f.write_text("a\t3\nb\t1\n")
        plot_metadata_counts(file_path=str(f), plot_type="bar", save_format="png")
        assert (tmp_path / "counts_bar.png").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_visualization_plots.py::TestCountsLoader -v`
Expected: first test FAIL (`TypeError` sorting strings vs numbers or wrong categories), third FAIL (file missing).

- [ ] **Step 3: Implement**

In `metaquest/visualization/plots.py` replace `_load_counts_df` with:

```python
def _looks_numeric(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def _load_counts_df(file_path: Union[str, Path, pd.DataFrame], limit: int) -> pd.DataFrame:
    """Load a counts table as [category, count] and keep the top `limit` rows.

    Two file shapes are accepted: the header-less ``<category>\\t<count>`` statistics
    file, and the ``count_metadata`` table whose first row is a header and whose
    remaining columns hold per-genome counts, which are summed per category.
    """
    if isinstance(file_path, pd.DataFrame):
        df = file_path.copy()
        if df.shape[1] < 2:
            raise VisualizationError("File must have at least two columns (category and count)")
        df.columns = ["category", "count"] + [f"col{i + 3}" for i in range(df.shape[1] - 2)]
    else:
        raw = pd.read_csv(file_path, sep="\t", header=None, dtype=str)
        if raw.shape[1] < 2:
            raise VisualizationError("File must have at least two columns (category and count)")
        if not _looks_numeric(raw.iat[0, 1]):
            raw = raw.iloc[1:].reset_index(drop=True)
        counts = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        df = pd.DataFrame({"category": raw.iloc[:, 0].astype(str), "count": counts})

    return df.sort_values(by="count", ascending=False).head(limit)
```

Replace the bar branch in `plot_metadata_counts` with:

```python
        if plot_type == "bar":
            plugin = visualizer_registry.get("bar")
            bar_output = None
            if save_format and not isinstance(file_path, pd.DataFrame):
                bar_output = f"{Path(file_path).stem}_bar.{save_format}"
            fig = plugin.create_plot(  # type: ignore[attr-defined]
                data=df,
                x_column="category",
                y_column="count",
                title=title,
                colors=colors,
                horizontal=True,
                output_file=bar_output,
                output_format=save_format or "png",
            )
```

Update the `--file-path` help in `metaquest/cli/commands/metadata.py:196` to `"Counts table from count_metadata (or its _stats file)"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_visualization_plots.py tests/test_visualizers_bar_extended.py tests/test_cli_commands.py -v`
Expected: all PASS.

- [ ] **Step 5: Gates and commit**

Run: `make check && make test`

```bash
git add metaquest/visualization/plots.py metaquest/cli/commands/metadata.py tests/test_visualization_plots.py
git commit -m "fix: read count_metadata tables in plot_metadata_counts and save bar charts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 7: Keep the real suffix when `sra_download` renames files

**Files:**
- Modify: `metaquest/data/sra_enhanced.py:236-262`
- Test: `tests/test_sra_enhanced.py:290-296`, `tests/test_sra_enhanced_extended.py:275-345`

**Interfaces:**
- Produces: `_fastq_suffix(path: Path) -> str` module function in `metaquest/data/sra_enhanced.py`.

- [ ] **Step 1: Write the failing test**

In `tests/test_sra_enhanced.py` add next to the existing rename test (reuse its downloader fixture and paths):

```python
    def test_illumina_rename_keeps_uncompressed_suffix(self, tmp_path):
        downloader = EnhancedSRADownloader("test@example.com")
        output_path = tmp_path / "SRR001"
        file1 = tmp_path / "SRR001_1.fastq"
        file2 = tmp_path / "SRR001_2.fastq"
        renamed = downloader._rename_files_by_technology([file1, file2], "illumina", output_path)
        assert renamed[file1] == output_path / "SRR001_R1.fastq"
        assert renamed[file2] == output_path / "SRR001_R2.fastq"

    def test_illumina_rename_keeps_gz_suffix(self, tmp_path):
        downloader = EnhancedSRADownloader("test@example.com")
        output_path = tmp_path / "SRR001"
        file1 = tmp_path / "SRR001_1.fastq.gz"
        renamed = downloader._rename_files_by_technology([file1], "illumina", output_path)
        assert renamed[file1] == output_path / "SRR001_R1.fastq.gz"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_sra_enhanced.py -k "keeps_uncompressed_suffix" -v`
Expected: FAIL, `_R1.fastq.gz != _R1.fastq`.

- [ ] **Step 3: Implement**

In `metaquest/data/sra_enhanced.py` add a module-level helper above the class:

```python
def _fastq_suffix(path: Path) -> str:
    """Suffix to keep on a renamed read file; fasterq-dump writes plain FASTQ."""
    return ".fastq.gz" if path.name.endswith(".gz") else ".fastq"
```

In `_rename_files_by_technology`, replace the three illumina branches:

```python
            if len(fastq_files) == 2:
                renamed_files[fastq_files[0]] = output_path / f"{accession}_R1{_fastq_suffix(fastq_files[0])}"
                renamed_files[fastq_files[1]] = output_path / f"{accession}_R2{_fastq_suffix(fastq_files[1])}"
            elif len(fastq_files) == 1:
                renamed_files[fastq_files[0]] = output_path / f"{accession}_R1{_fastq_suffix(fastq_files[0])}"
            else:
                for i, file in enumerate(fastq_files, 1):
                    renamed_files[file] = output_path / f"{accession}_R{i}{_fastq_suffix(file)}"
```

Update the existing assertions at `tests/test_sra_enhanced.py:295-296` and `tests/test_sra_enhanced_extended.py:278-279,332,345` to expect `.fastq` when the input files are plain `.fastq` (they are: `SRR001_1.fastq`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_sra_enhanced.py tests/test_sra_enhanced_extended.py -v`
Expected: all PASS.

- [ ] **Step 5: Gates and commit**

Run: `make check && make test`

```bash
git add metaquest/data/sra_enhanced.py tests/test_sra_enhanced.py tests/test_sra_enhanced_extended.py
git commit -m "fix: do not label uncompressed fasterq-dump output as .fastq.gz

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `fix/honest-results` -> `main` (base retargets automatically after PR 1 merges).

---

## Phase 3: workflow hand-offs (branch `feat/workflow-handoffs`, based on `fix/honest-results`)

### Task 8: Parse GTDB taxon matches and fan out to species

**Files:**
- Modify: `metaquest/data/gtdb.py:41-63,71-73,104-127`
- Test: `tests/test_gtdb.py`

**Interfaces:**
- Produces: `search_taxon(taxon_name, limit=100) -> List[Dict]` now returns `[{"name": "s__Wolbachia pipientis"}, ...]` for the `{"matches": [...]}` response; `_is_representative` also honours `gtdb_species_rep`; `get_accessions_for_genus` fans out over `s__` matches via `get_accessions_for_species`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gtdb.py`:

```python
TAXON_MATCHES_RESPONSE = {
    "matches": ["g__Wolbachia", "s__Wolbachia massiliensis", "s__Wolbachia pipientis", "s__Wolbachia pipientis_A"]
}

SPECIES_GENOMES_RESPONSE = {
    "name": "Wolbachia pipientis",
    "genomes": [
        {"accession": "GCA_000174095.1", "gtdb_species_rep": False, "ncbi_org_name": "Wolbachia sp."},
        {"accession": "GCA_021378375.1", "gtdb_species_rep": True, "ncbi_org_name": "Wolbachia pipientis"},
    ],
}


class TestLiveApiShapes:
    @patch("metaquest.data.gtdb.requests.get")
    def test_search_taxon_matches_shape(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = TAXON_MATCHES_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_taxon("Wolbachia")

        assert [r["name"] for r in result] == TAXON_MATCHES_RESPONSE["matches"]

    @patch("metaquest.data.gtdb.search_species")
    def test_species_rep_key_recognised(self, mock_search):
        mock_search.return_value = SPECIES_GENOMES_RESPONSE["genomes"]

        assert get_accessions_for_species("Wolbachia pipientis") == ["GCA_021378375.1"]

    @patch("metaquest.data.gtdb.search_species")
    @patch("metaquest.data.gtdb.search_taxon")
    def test_genus_fans_out_to_species(self, mock_taxon, mock_species):
        mock_taxon.return_value = [{"name": n} for n in TAXON_MATCHES_RESPONSE["matches"]]
        mock_species.side_effect = lambda name: [
            {"accession": f"GCA_{abs(hash(name)) % 10**9:09d}.1", "gtdb_species_rep": True}
        ]

        accessions = get_accessions_for_genus("Wolbachia")

        assert len(accessions) == 3
        assert mock_species.call_args_list[0][0][0] == "Wolbachia massiliensis"
        assert all(acc.startswith("GCA_") for acc in accessions)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_gtdb.py::TestLiveApiShapes -v`
Expected: all three FAIL.

- [ ] **Step 3: Implement**

In `metaquest/data/gtdb.py`, in `search_taxon` replace the dict branch:

```python
    if isinstance(data, dict):
        if "matches" in data:
            return [{"name": str(name)} for name in data["matches"]]
        return data.get("results", [data])
```

Replace `_is_representative`:

```python
def _is_representative(record: Dict) -> bool:
    """Whether a GTDB record is flagged as a representative genome (any key spelling)."""
    return bool(record.get("isRep") or record.get("is_representative") or record.get("gtdb_species_rep"))
```

Replace `get_accessions_for_genus`:

```python
def get_accessions_for_genus(genus_name: str, representative_only: bool = True) -> List[str]:
    """Get representative accessions for all species in a genus.

    The live taxon endpoint returns taxon names only; each ``s__`` species name is
    resolved through the species endpoint. Older record-shaped responses (with an
    accession per record) are still handled.
    """
    taxon_results = search_taxon(genus_name)
    if not taxon_results:
        return []

    accessions: List[str] = []
    seen_species = set()
    for record in taxon_results:
        name = str(record.get("species") or record.get("name", ""))
        if name in seen_species:
            continue
        seen_species.add(name)

        if name.startswith("s__") and _record_accession(record) is None:
            species_name = name[3:]
            logger.debug("Resolving species %s for genus %s", species_name, genus_name)
            accessions.extend(get_accessions_for_species(species_name, representative_only))
            continue

        accession = _keep_accession(record, representative_only)
        if accession:
            accessions.append(accession)

    if representative_only and not accessions:
        for record in taxon_results:
            accession = _record_accession(record)
            if accession and accession not in accessions:
                accessions.append(accession)

    return accessions
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_gtdb.py tests/test_genome_download.py -v`
Expected: all PASS (existing record-shaped fixtures still pass through the `_keep_accession` branch).

Manual check (network): `metaquest genome_search --genus Wolbachia --format tsv` prints three accessions.

- [ ] **Step 5: Gates and commit**

Run: `make check && make test`

```bash
git checkout -b feat/workflow-handoffs fix/honest-results
git add metaquest/data/gtdb.py tests/test_gtdb.py
git commit -m "fix: parse GTDB taxon matches and resolve genus searches through species

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 9: Resolve the metadata table default and accept TSV or CSV matrices

**Files:**
- Create: `metaquest/data/defaults.py`
- Modify: `metaquest/cli/commands/metadata.py:108-111,145-148`, `metaquest/cli/commands/samples.py:29-33`
- Modify: `metaquest/cli/commands/advanced_analysis.py:74,79,159,164,354`
- Test: `tests/test_data_defaults.py` (new), `tests/test_cli_commands.py`, `tests/test_cli_commands_advanced_analysis.py`

**Interfaces:**
- Produces: `resolve_metadata_table(explicit: Optional[str]) -> Path`, `read_matrix(path: Union[str, Path]) -> pd.DataFrame`, `METADATA_TABLE_CANDIDATES` in `metaquest/data/defaults.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_defaults.py`:

```python
"""Tests for default-file resolution shared by CLI commands."""

from pathlib import Path

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.data.defaults import read_matrix, resolve_metadata_table


class TestResolveMetadataTable:
    def test_explicit_path_wins(self, tmp_path):
        table = tmp_path / "mine.txt"
        table.write_text("Run_ID\tx\n")
        assert resolve_metadata_table(str(table)) == table

    def test_ncbi_table_preferred_over_branchwater(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "metadata_table.txt").write_text("Run_ID\tx\n")
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "branchwater_metadata.txt").write_text("Run_ID\tx\n")
        assert resolve_metadata_table(None) == Path("metadata_table.txt")

    def test_falls_back_to_branchwater_metadata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "branchwater_metadata.txt").write_text("Run_ID\tx\n")
        assert resolve_metadata_table(None) == Path("metadata/branchwater_metadata.txt")

    def test_missing_lists_candidates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(DataAccessError, match="metadata_table.txt.*branchwater_metadata.txt"):
            resolve_metadata_table(None)

    def test_explicit_missing_raises(self, tmp_path):
        with pytest.raises(DataAccessError, match="nope.txt"):
            resolve_metadata_table(str(tmp_path / "nope.txt"))


class TestReadMatrix:
    def test_tsv_from_parse_containment_drops_text_columns(self, tmp_path):
        f = tmp_path / "parsed_containment.txt"
        f.write_text("\tGCF_A\tGCF_B\tmax_containment\tmax_containment_annotation\nSRR1\t0.9\t0.1\t0.9\tGCF_A\n")
        df = read_matrix(f)
        assert list(df.columns) == ["GCF_A", "GCF_B", "max_containment"]
        assert list(df.index) == ["SRR1"]

    def test_csv(self, tmp_path):
        f = tmp_path / "abundance.csv"
        f.write_text(",sp1,sp2\nS1,1,2\n")
        df = read_matrix(f)
        assert df.loc["S1", "sp2"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_data_defaults.py -v`
Expected: FAIL with `ModuleNotFoundError: metaquest.data.defaults`.

- [ ] **Step 3: Implement the module**

Create `metaquest/data/defaults.py`:

```python
"""Resolution of default input files so pipeline steps chain without extra flags."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)

# Order matters: the NCBI table (parse_metadata) is richer than the Branchwater one.
METADATA_TABLE_CANDIDATES: List[Path] = [
    Path("metadata_table.txt"),
    Path("metadata/metadata_table.txt"),
    Path("metadata/branchwater_metadata.txt"),
]


def resolve_metadata_table(explicit: Optional[str]) -> Path:
    """Return the metadata table to use.

    An explicit path must exist. Otherwise the first existing candidate in
    ``METADATA_TABLE_CANDIDATES`` is used, so a project that only ran
    ``extract_branchwater_metadata`` still works with the defaults.

    Raises:
        DataAccessError: If nothing usable exists.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise DataAccessError(f"Metadata table not found: {explicit}")
        return path

    for candidate in METADATA_TABLE_CANDIDATES:
        if candidate.exists():
            logger.info("Using metadata table %s", candidate)
            return candidate

    tried = ", ".join(str(c) for c in METADATA_TABLE_CANDIDATES)
    raise DataAccessError(
        f"No metadata table found (tried {tried}). Run parse_metadata or extract_branchwater_metadata, "
        "or pass --metadata-file."
    )


def read_matrix(path: Union[str, Path]) -> pd.DataFrame:
    """Read a samples x features table as numeric data.

    Tab-separated ``.txt``/``.tsv`` files (as written by parse_containment) and
    comma-separated files are both accepted. Non-numeric columns such as
    ``max_containment_annotation`` are dropped with a log line.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    numeric = df.select_dtypes(include="number")
    dropped = [str(c) for c in df.columns if c not in numeric.columns]
    if dropped:
        logger.info("Ignoring non-numeric column(s) in %s: %s", path.name, ", ".join(dropped))
    return numeric
```

- [ ] **Step 4: Use it in the CLI**

`metaquest/cli/commands/metadata.py`:
- `CheckMetadataAttributesCommand`: `--file-path` `default=None`, help `"Parsed metadata table (default: metadata_table.txt, else metadata/branchwater_metadata.txt)"`; in `execute` call `check_metadata_attributes(str(resolve_metadata_table(args.file_path)), args.output_file)`.
- `CountMetadataCommand`: `--metadata-file` `default=None`, same help; in `execute` pass `metadata_file=str(resolve_metadata_table(args.metadata_file))`.
- Add `from metaquest.data.defaults import resolve_metadata_table` to the imports. `DataAccessError` is a `MetaQuestError`, so the existing `except MetaQuestError` branches report it.

`metaquest/cli/commands/samples.py`: same change for `--metadata-file` and `count_single_sample(metadata_file=str(resolve_metadata_table(args.metadata_file)), ...)`.

`metaquest/cli/commands/advanced_analysis.py`: import `read_matrix` and replace the five `pd.read_csv(<file>, index_col=0)` calls at lines 74, 79, 159, 164 and 354 with `read_matrix(<file>)`. Leave line 257 (`species_file`) and 357 (`taxonomy_file`) as they are.

Update tests that construct a `Namespace` with `metadata_file="metadata_table.txt"` for these commands: since the resolver now requires the file to exist, create it in `tmp_path` or patch `metaquest.cli.commands.metadata.resolve_metadata_table`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_data_defaults.py tests/test_cli_commands.py tests/test_cli_commands_advanced_analysis.py -v`
Expected: all PASS.

Manual check: in a folder containing only `parsed_containment.txt` and `metadata/branchwater_metadata.txt`, `metaquest count_metadata --metadata-column Sample_Scientific_Name` succeeds; `metaquest interactive_plot --data-file parsed_containment.txt --plot-type heatmap --no-show --output-file h.html` writes the file.

- [ ] **Step 6: Gates and commit**

Run: `make check && make test`

```bash
git add metaquest/data/defaults.py metaquest/cli/commands/metadata.py metaquest/cli/commands/samples.py metaquest/cli/commands/advanced_analysis.py tests/
git commit -m "feat: resolve metadata table defaults and accept TSV or CSV matrices

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 10: `select_datasets` command writes `accessions.txt` from containment and metadata

**Files:**
- Create: `metaquest/processing/selection.py`, `metaquest/cli/commands/select.py`
- Modify: `metaquest/cli/main.py:62-103` (import and register), `tests/test_cli_main.py:34-58` (expected list)
- Test: `tests/test_processing_selection.py`, `tests/test_cli_select.py`

**Interfaces:**
- Produces: `select_accessions(parsed_containment, genome_id=None, threshold=0.1, metadata_file=None, metadata_column=None, metadata_value=None) -> List[str]` (sorted by containment, descending); CLI command `select_datasets` with `--parsed-containment`, `--genome-id`, `--threshold`, `--metadata-file`, `--metadata-column`, `--metadata-value`, `--output`.
- Consumes: `resolve_metadata_table` from Task 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_processing_selection.py`:

```python
"""Tests for accession selection from containment and metadata tables."""

import pytest

from metaquest.core.exceptions import ProcessingError
from metaquest.processing.selection import select_accessions


@pytest.fixture
def tables(tmp_path):
    cont = tmp_path / "parsed_containment.txt"
    cont.write_text(
        "\tGCF_A\tGCF_B\tmax_containment\tmax_containment_annotation\n"
        "SRR1\t0.95\t0.10\t0.95\tGCF_A\n"
        "SRR2\t0.20\t0.80\t0.80\tGCF_B\n"
        "SRR3\t0.05\t0.02\t0.05\tGCF_A\n"
    )
    meta = tmp_path / "branchwater_metadata.txt"
    meta.write_text(
        "Run_ID\tSample_Scientific_Name\tgeo_loc_name_country_calc\n"
        "SRR1\tWolbachia pipientis\tAustria\n"
        "SRR2\tmosquito metagenome\tFrance\n"
        "SRR3\tmosquito metagenome\tFrance\n"
    )
    return cont, meta


def test_default_uses_max_containment(tables):
    cont, _ = tables
    assert select_accessions(cont, threshold=0.5) == ["SRR1", "SRR2"]


def test_genome_column_and_threshold(tables):
    cont, _ = tables
    assert select_accessions(cont, genome_id="GCF_B", threshold=0.5) == ["SRR2"]


def test_metadata_filter_is_case_insensitive(tables):
    cont, meta = tables
    result = select_accessions(
        cont, threshold=0.0, metadata_file=meta, metadata_column="geo_loc_name_country_calc", metadata_value="france"
    )
    assert result == ["SRR2", "SRR3"]


def test_unknown_genome_lists_columns(tables):
    cont, _ = tables
    with pytest.raises(ProcessingError, match="GCF_A"):
        select_accessions(cont, genome_id="GCF_Z")


def test_metadata_column_without_value_raises(tables):
    cont, meta = tables
    with pytest.raises(ProcessingError, match="metadata_value"):
        select_accessions(cont, metadata_file=meta, metadata_column="Sample_Scientific_Name")
```

Create `tests/test_cli_select.py`:

```python
"""Tests for the select_datasets CLI command."""

import argparse

from metaquest.cli.commands.select import SelectDatasetsCommand


def _args(tmp_path, **kwargs):
    base = dict(
        parsed_containment=str(tmp_path / "parsed_containment.txt"),
        genome_id=None,
        threshold=0.5,
        metadata_file=None,
        metadata_column=None,
        metadata_value=None,
        output=str(tmp_path / "accessions.txt"),
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_writes_one_accession_per_line(tmp_path):
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    rc = SelectDatasetsCommand().execute(_args(tmp_path))
    assert rc == 0
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\n"


def test_missing_table_returns_1(tmp_path):
    rc = SelectDatasetsCommand().execute(_args(tmp_path))
    assert rc == 1


def test_command_is_registered():
    from metaquest.cli.main import create_parser, register_all_commands

    register_all_commands()
    parser = create_parser()
    action = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None))
    assert "select_datasets" in action.choices
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_processing_selection.py tests/test_cli_select.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the processing function**

Create `metaquest/processing/selection.py`:

```python
"""Select SRA accessions from a parsed containment table, optionally filtered by metadata."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError, ProcessingError

logger = logging.getLogger(__name__)

DEFAULT_COLUMN = "max_containment"


def select_accessions(
    parsed_containment: Union[str, Path],
    genome_id: Optional[str] = None,
    threshold: float = 0.1,
    metadata_file: Optional[Union[str, Path]] = None,
    metadata_column: Optional[str] = None,
    metadata_value: Optional[str] = None,
) -> List[str]:
    """Return accessions whose containment meets the threshold, best first.

    Args:
        parsed_containment: Table from parse_containment (samples x genomes, tab-separated).
        genome_id: Genome column to rank on; defaults to ``max_containment``.
        threshold: Minimum containment (inclusive).
        metadata_file: Optional metadata table keyed by Run_ID in its first column.
        metadata_column: Column in the metadata table to filter on.
        metadata_value: Required value for that column (case-insensitive match).

    Raises:
        DataAccessError: If a table is missing.
        ProcessingError: If a column is unknown or the metadata filter is incomplete.
    """
    table_path = Path(parsed_containment)
    if not table_path.exists():
        raise DataAccessError(f"Parsed containment table not found: {table_path}")
    if bool(metadata_column) != bool(metadata_value):
        raise ProcessingError("metadata_column and metadata_value must be given together")

    containment = pd.read_csv(table_path, sep="\t", index_col=0)
    column = genome_id or DEFAULT_COLUMN
    if column not in containment.columns:
        raise ProcessingError(
            f"Column '{column}' not found in {table_path.name}. Available columns: "
            f"{', '.join(str(c) for c in containment.columns)}"
        )

    values = pd.to_numeric(containment[column], errors="coerce").fillna(0.0)
    selected = values[values >= threshold].sort_values(ascending=False)
    accessions = [str(acc) for acc in selected.index]
    logger.info("%d accession(s) meet %s >= %.3f", len(accessions), column, threshold)

    if metadata_column and metadata_value:
        if metadata_file is None:
            raise ProcessingError("metadata_file is required when filtering on metadata")
        meta_path = Path(metadata_file)
        if not meta_path.exists():
            raise DataAccessError(f"Metadata table not found: {meta_path}")
        metadata = pd.read_csv(meta_path, sep="\t", index_col=0, dtype=str)
        if metadata_column not in metadata.columns:
            raise ProcessingError(
                f"Column '{metadata_column}' not found in {meta_path.name}. Available columns: "
                f"{', '.join(str(c) for c in metadata.columns)}"
            )
        wanted = metadata_value.strip().lower()
        matching = {str(idx) for idx, val in metadata[metadata_column].items() if str(val).strip().lower() == wanted}
        accessions = [acc for acc in accessions if acc in matching]
        logger.info("%d accession(s) remain after %s == %r", len(accessions), metadata_column, metadata_value)

    return accessions
```

- [ ] **Step 4: Implement the command and register it**

Create `metaquest/cli/commands/select.py`:

```python
"""CLI command that turns containment results into an accession list for downloads."""

import argparse
from pathlib import Path

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.defaults import resolve_metadata_table
from metaquest.processing.selection import select_accessions


class SelectDatasetsCommand(BaseCommand):
    """Write the accessions that meet a containment threshold (and optional metadata filter)."""

    @property
    def name(self) -> str:
        return "select_datasets"

    @property
    def help(self) -> str:
        return "Write an accessions file from parsed containment, for the download commands"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--parsed-containment", default="parsed_containment.txt", help="Table from parse_containment"
        )
        parser.add_argument("--genome-id", default=None, help="Genome column to rank on (default: max_containment)")
        parser.add_argument("--threshold", type=float, default=0.1, help="Minimum containment, inclusive")
        parser.add_argument(
            "--metadata-file",
            default=None,
            help="Metadata table for filtering (default: metadata_table.txt, else metadata/branchwater_metadata.txt)",
        )
        parser.add_argument("--metadata-column", default=None, help="Metadata column to filter on")
        parser.add_argument("--metadata-value", default=None, help="Required value in that column")
        parser.add_argument("--output", default="accessions.txt", help="Output file, one accession per line")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            metadata_file = None
            if args.metadata_column:
                metadata_file = resolve_metadata_table(args.metadata_file)

            accessions = select_accessions(
                parsed_containment=args.parsed_containment,
                genome_id=args.genome_id,
                threshold=args.threshold,
                metadata_file=metadata_file,
                metadata_column=args.metadata_column,
                metadata_value=args.metadata_value,
            )

            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(f"{acc}\n" for acc in accessions))
            self.logger.info("Wrote %d accession(s) to %s", len(accessions), output)
            if not accessions:
                self.logger.warning("No accessions met the criteria; %s is empty", output)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error selecting datasets: %s", e)
            return 1
```

In `metaquest/cli/main.py` import `SelectDatasetsCommand` from `metaquest.cli.commands.select` and add `SelectDatasetsCommand(),` after `ExtractTargetReadsCommand(),` in `register_all_commands`. Add `"select_datasets"` to `expected_commands` in `tests/test_cli_main.py`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_processing_selection.py tests/test_cli_select.py tests/test_cli_main.py -v`
Expected: all PASS.

- [ ] **Step 6: Gates and commit**

Run: `make check && make test`

```bash
git add metaquest/processing/selection.py metaquest/cli/commands/select.py metaquest/cli/main.py tests/test_processing_selection.py tests/test_cli_select.py tests/test_cli_main.py
git commit -m "feat: add select_datasets to write accessions.txt from containment results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `feat/workflow-handoffs` -> `main`.

---

## Phase 4: docs and noise (branch `docs/audit-followup`, based on `feat/workflow-handoffs`)

### Task 11: Silence the cartopy import warning and point at the extra

**Files:**
- Modify: `metaquest/plugins/visualizers/map.py:25-42`
- Test: `tests/test_plugins_comprehensive.py` (the cartopy tests live there; `VisualizationError` and `patch` are already imported)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plugins_comprehensive.py`:

```python
def test_missing_cartopy_message_names_the_extra():
    from metaquest.plugins.visualizers import map as map_module

    with patch.object(map_module, "CARTOPY_AVAILABLE", False):
        with pytest.raises(VisualizationError, match=r"pip install 'metaquest\[maps\]'"):
            map_module._validate_cartopy_availability()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests -k "names_the_extra" -v`
Expected: FAIL (message says `pip install cartopy`).

- [ ] **Step 3: Implement**

In `map.py` change the `except ImportError` branch to log at debug level:

```python
except ImportError:
    CARTOPY_AVAILABLE = False
    GeoAxes = None
    logger.debug("Cartopy not available; map visualization is disabled")
```

and the error text in `_validate_cartopy_availability` to:

```python
        raise VisualizationError(
            "Cartopy is required for map visualization. Install it with: pip install 'metaquest[maps]'"
        )
```

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_plugins_comprehensive.py -v && metaquest --help 2>&1 | grep -c Cartopy`
Expected: tests PASS; grep prints `0`.

```bash
git checkout -b docs/audit-followup feat/workflow-handoffs
git add metaquest/plugins/visualizers/map.py tests/
git commit -m "fix: log the missing cartopy notice at debug level and name the maps extra

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 12: Executed walkthrough, corrected docs, environment file

**Files:**
- Modify: `local_test.sh`
- Modify: `README.md:70-80,100-110,296-305`, README "Downloading SRA" and "Which SRA download command" sections, intelligent-command examples (`sra_downloads` -> `fastq`)
- Modify: `docs/branchwater_workflow.md:48-92`
- Create: `environment.yml`
- Modify: `.gitignore` (add `build/local_test/`)

- [ ] **Step 1: Make `local_test.sh` run in a scratch folder and cover the repaired chain**

Replace `local_test.sh` with:

```bash
#!/bin/bash
# local_test.sh: end-to-end CLI walkthrough on the bundled Branchwater sample.
# Runs in build/local_test so tracked files under test_data/ are never rewritten.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$ROOT/build/local_test"
rm -rf "$WORK"
mkdir -p "$WORK/branchwater" "$WORK/genomes"
cp "$ROOT/test_data/branchwater/salmonella_subset.csv" "$WORK/branchwater/"
if [ -f "$ROOT/test_data/GCF_000008985.1.fasta" ]; then
    cp "$ROOT/test_data/GCF_000008985.1.fasta" "$WORK/genomes/GCF_000008985.1.fna"
fi
cd "$WORK"

check() { if [ -e "$1" ]; then echo "ok   $1"; else echo "FAIL $1 missing"; exit 1; fi; }

echo "use_branchwater"
metaquest use_branchwater --branchwater-folder branchwater --matches-folder matches
check matches/salmonella_subset.csv

echo "parse_containment"
metaquest parse_containment --matches-folder matches --parsed-containment-file parsed_containment.txt \
    --summary-containment-file summary_containment.txt --step-size 0.1
check parsed_containment.txt
check summary_containment.txt

echo "extract_branchwater_metadata"
metaquest extract_branchwater_metadata --branchwater-folder branchwater --metadata-folder metadata
check metadata/branchwater_metadata.txt

echo "count_metadata (metadata table resolved from metadata/branchwater_metadata.txt)"
metaquest count_metadata --metadata-column Sample_Scientific_Name --threshold 0.9 --output-file metadata_counts.txt
check metadata_counts.txt

echo "plot_metadata_counts"
metaquest plot_metadata_counts --file-path metadata_counts.txt --plot-type bar --save-format png
check metadata_counts_bar.png

echo "plot_containment"
metaquest plot_containment --file-path parsed_containment.txt --column max_containment --plot-type rank --save-format png
check parsed_containment.txt_rank_max_containment.png

echo "select_datasets"
metaquest select_datasets --threshold 0.95 --output accessions.txt
check accessions.txt
test "$(wc -l < accessions.txt)" -gt 0

echo "status"
metaquest status --parsed-containment parsed_containment.txt --list-missing

if [ -f genomes/GCF_000008985.1.fna ]; then
    echo "extract_target_reads --dry-run"
    metaquest extract_target_reads --parsed-containment parsed_containment.txt --genome-id salmonella_subset \
        --genome-fasta genomes/GCF_000008985.1.fna --threshold 0.95 --dry-run
fi

echo "All steps passed (outputs in $WORK)"
```

Add `build/local_test/` to `.gitignore`. Run `make pipeline`; expected: "All steps passed" and `git status --short` shows no tracked changes.

- [ ] **Step 2: Correct the README**

Apply these edits to `README.md`:

1. In section 4 replace `*Example output:* summary.txt and containment.txt` with `*Example output:* parsed_containment.txt (samples x genomes) and summary_containment.txt (counts per containment step).`
2. Rename section 5 heading to `### 5. Downloading Metadata from NCBI (richer alternative to step 3)`; after section 7 add:

```markdown
### 8. Counting metadata values

`count_metadata`, `single_sample` and `check_metadata_attributes` use `metadata_table.txt` from
`parse_metadata` when it exists and otherwise fall back to `metadata/branchwater_metadata.txt` from
step 3, so either metadata route works with the defaults:

```bash
metaquest count_metadata --metadata-column Sample_Scientific_Name --threshold 0.9 --output-file metadata_counts.txt
```

This writes `metadata_counts.txt` (one row per value, one column per genome) and `metadata_counts_stats.txt`.
```

3. In "Plotting Metadata Counts" replace the example with `metaquest plot_metadata_counts --file-path metadata_counts.txt --plot-type bar --save-format png` and add the sentence `The bar chart is written next to the input as metadata_counts_bar.png.`
4. Replace every `--summary-column GCF_000008985.1` with `--summary-column <genome column from parsed_containment.txt>` and note that the column name is the Branchwater file stem.
5. At the top of the SRA download section add:

```markdown
### Choosing which datasets to download

```bash
metaquest select_datasets --threshold 0.9 --output accessions.txt
metaquest select_datasets --genome-id GCF_000008025.1 --threshold 0.5 \
    --metadata-column geo_loc_name_country_calc --metadata-value France --output accessions.txt
```

`accessions.txt` is the input for `download_sra`, `sra_download` and `sra-download-intelligent`.
All three write `fastq/<accession>/<accession>_1.fastq` (and `_2` for paired runs), which is the layout
`status`, `sra_stats`, `sra-profile-quality`, `sra-dashboard` and `extract_target_reads` read.
```

6. Replace `sra_downloads` with `fastq` in the intelligent-command examples (lines 196, 215, 222, 252).
7. Add an "External tools" subsection under Installation:

```markdown
### External tools

Download and assembly steps call command-line tools that are not Python packages:

| Tool | Used by |
|---|---|
| `fasterq-dump` (sra-tools) | `download_sra`, `sra_download`, `sra-download-intelligent` |
| `datasets` (ncbi-datasets-cli) | `genome_download`, `genome_prepare`, `download_test_genome` |
| `minimap2`, `samtools` | `extract_target_reads` |
| `megahit` | `extract_target_reads --assemble` |

`environment.yml` installs all of them together with MetaQuest:

```bash
conda env create -f environment.yml
conda activate metaquest
```

Map plots need the optional extra: `pip install 'metaquest[maps]'`.
```

- [ ] **Step 3: Correct `docs/branchwater_workflow.md`**

Delete the "Containment Threshold Optimization" and "Large Dataset Optimization" and streaming code blocks (lines 48-70 and 86-92). Replace with:

```markdown
### Containment steps

`parse_containment` does not filter; it records every sample and summarizes how many samples exceed
each containment step. Choose the step size for the summary and apply thresholds downstream:

```bash
metaquest parse_containment --matches-folder matches --step-size 0.05
metaquest select_datasets --threshold 0.9 --output accessions.txt
metaquest count_metadata --metadata-column Sample_Scientific_Name --threshold 0.9
```
```

- [ ] **Step 4: Create `environment.yml`**

```yaml
name: metaquest
channels:
  - conda-forge
  - bioconda
dependencies:
  - python>=3.12
  - pip
  - sra-tools>=3.0
  - ncbi-datasets-cli
  - minimap2
  - samtools
  - megahit
  - sourmash
  - pip:
      - -e .
```

- [ ] **Step 5: Verify docs against the CLI and commit**

Run:

```bash
make pipeline
grep -n "batch-size\|--streaming\|--chunk-size\|use_branchwater .*--max-workers\|parse_containment .*--threshold" README.md docs/*.md
git status --short
```

Expected: pipeline passes; grep prints nothing; only the intended files are modified.

```bash
git add local_test.sh .gitignore README.md docs/branchwater_workflow.md environment.yml
git commit -m "docs: run the walkthrough in a scratch folder and match the README to the CLI

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Open PR `docs/audit-followup` -> `main`.

---

## Verification after all four PRs merge

- `make check && make test` green on `main`.
- `PATH=$HOME/miniforge3/envs/sra-tools/bin:$PATH make test-network` passes.
- Real-data check with the audit fixture (see memory note `wolbachia-e2e-dataset`): `extract_target_reads` on SRR11011979 against wMel either extracts about 295k reads into the `_0` orphan file with the unequal-mates warning, or exits 1 with a clear message; it never reports success with empty files.
- `metaquest genome_search --genus Wolbachia` lists three accessions.

## Follow-up plans (not part of this plan)

1. **Branchwater search command.** `branchwater_search --genome x.fna --threshold 0.1 --output branchwater/<stem>.csv`: sketch with sourmash (k=21, scaled=1000), POST `{"threshold": t, "signature": <sig>}` to `https://api.branchwater.sourmash.bio/search`, join BioSample metadata from NCBI/EBI, write the Branchwater CSV. Include a positive-control check (Salmonella LT2 must return matches) so an empty index is reported, not silently accepted.
2. **Download consolidation.** Keep `download_sra` as the single engine, make `sra_download` and `sra-download-intelligent` thin aliases, delete the inert bandwidth/checkpoint/resume flags and the fabricated size estimate, and drop the duplicate accession-file and blacklist readers.
3. **Taxonomy table formats.** `taxonomic_summary` expects the `is_valid` table from `validate_taxonomy`, while `enrich_taxonomy`, `explore_containment` and `find_by_taxonomy` share the `genome_id` map; accept both in `taxonomic_summary` or document the two routes.
4. **CLI polish.** `ArgumentDefaultsHelpFormatter` on every subparser, a grouped command listing with aliases hidden, one threshold semantics (`>=`) everywhere, hide `assemble_datasets`, remove the `--file-format` flag and the tsne stub, and delete the dead functions listed in the audit.
