# Lactobacillus Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Critical and Important findings of the 2026-09-23 Lactobacillus end-to-end audit so that a store on an ExFAT or SMB volume under macOS works, `store_gc` can never remove data a project links, and the remaining broken commands (`sra_info`, `sra_compare`, `plot_containment`, extraction dry run, assembly errors) behave as documented.

**Architecture:** One helper (`visible_files` in `metaquest/data/file_io.py`) becomes the single way to list files or folders, and every listing site routes through it. The store gains a small append-only journal (`<store>/journal/projects.jsonl`, `usage.jsonl`) so `store_reindex` can rebuild the project and usage tables, and `store_gc` refuses to act when it cannot see any project. Everything else is a local fix in the function the audit named, each with a regression test that reproduces the observed behaviour first.

**Tech Stack:** Python 3.12, pytest, pandas, sqlite3 (store catalogue), argparse CLI, black/flake8/mypy via `make check`.

**Spec:** `docs/superpowers/specs/2026-09-23-lactobacillus-e2e-audit.md` (finding ids S<stage>-<n> below refer to it). Evidence: `/Volumes/sekvens2/metaquest-e2e/audit_log.md`.

## Global Constraints

- Line length 120, black formatting, flake8 clean, mypy clean: `make check` must pass after every task.
- No Unicode in code or docs. Modest scientific language in messages and docstrings.
- CLI arguments use dashes (`--temp-folder`), never underscores.
- Tests never touch a real store: every store test uses a `tmp_path` store; `tests/conftest.py` already sets `HOME`, `XDG_CONFIG_HOME` and unsets `METAQUEST_DATA` (autouse `isolate_store_discovery`).
- Tests must not depend on tools on PATH: patch `metaquest.data.sra.shutil.which` (or `metaquest.utils.security.shutil.which`) explicitly.
- The `fastq/` folder names, sidecar schema version (`schema: 1`) and registry version (2) do not change; new sidecar fields are optional with defaults.
- Commit after every task with a conventional prefix (`fix:`, `feat:`, `test:`, `docs:`).

## Review Focus

1. A dataset folder holding only `._*` files and no real FASTQ (macOS left the AppleDouble files after the data file was deleted): `fastq_files` must return `[]`, `accession_has_fastq` must be False, `store_adopt` must skip it, `status` must not count it. (Task 1, Task 5)
2. A store whose catalogue file was deleted and whose journal is also missing (a store created before this plan): `store_reindex` must rebuild datasets, warn that no project records exist, and `store_gc` must refuse until a project runs `store_init`/links again. (Task 4)
3. Ctrl-C while two workers run prefetch: no second prefetch may start, the two children must be terminated, the lock files released, and the exit code non-zero. (Task 7)
4. `sra_info` on a run whose efetch XML has several RUN elements (a multi-run experiment): one row per RUN accession with that RUN's own size and spots. (Task 9)
5. `extract_target_reads --dry-run` on a project with 16 000 screened samples and 6 downloaded: at most one summary line for the non-downloaded ones, and the six listed. (Task 13)

---

## File structure

- `metaquest/data/file_io.py`: add `is_hidden_name(name)` and `visible_files(directory, *patterns, dirs=False)`; `list_files` gains `include_hidden=False`.
- `metaquest/data/sra.py`: `fastq_files` uses `visible_files`; `_store_fetch` defaults the fasterq-dump scratch to `<store>/tmp/<ACC>_fqtmp`; `_execute_parallel_downloads` handles `KeyboardInterrupt`; `_process_download_results` logs a final count; `_safe_rmtree` tolerates ENOENT.
- `metaquest/utils/security.py`: `run_secure` tracks children so `terminate_children()` can stop them; `CalledProcessError` carries stderr into a `ProcessingError`-style message via `format_called_process_error(e)`.
- `metaquest/store/adopt.py`: hidden-name filters, empty-folder refusal, verify before `rmtree`, `--copy` honoured in dedup, `copytree(ignore=...)`.
- `metaquest/store/journal.py` (new): append and replay of project and usage records.
- `metaquest/store/catalog.py`, `metaquest/store/usage.py`, `metaquest/cli/commands/store.py`: journal writes, reindex replay, gc refusal, verify rescan and spots from metadata.
- `metaquest/data/branchwater.py`, `metaquest/cli/commands/branchwater.py`, `metaquest/cli/commands/containment.py`: error counts returned and exit 1; plot default and naming.
- `metaquest/cli/commands/genome.py`, `metaquest/cli/commands/status.py`, `metaquest/data/registry.py`, `metaquest/sra/analytics.py`: hidden-name filters; `--next` picks a skip-excluded selection; export prints paths and drops the index.
- `metaquest/data/sra_metadata.py`, `metaquest/data/metadata.py`: RUN attributes and run accessions; `LIBRARY_*` XPath.
- `metaquest/sra/analytics.py`, `metaquest/cli/commands/sra_intelligent.py`, `metaquest/sra/reporting.py`: JSON-safe statistics, optional `--accessions-file`, profiles passed to the comparative dashboard.
- `metaquest/data/read_extraction.py`, `metaquest/cli/commands/read_extraction.py`: dry-run intersection, megahit stderr and `--tmp-dir`, parameter-change message, MAPQ wording.
- `tests/test_data_sra.py`: three tests patch `shutil.which`.
- `README.md`, `docs/pipeline_overview.md`: the corrections in Appendix A of the spec.

---

### Task 1: One helper for visible files, applied to every FASTQ, FASTA, CSV, XML and profile listing

**Files:**
- Modify: `metaquest/data/file_io.py:40-56`
- Modify: `metaquest/data/sra.py:150-166` (`fastq_files`)
- Modify: `metaquest/cli/commands/genome.py:312`
- Modify: `metaquest/cli/commands/status.py:141-147`
- Modify: `metaquest/data/registry.py:729, 741, 745-753, 798`
- Modify: `metaquest/data/branchwater.py:226`
- Modify: `metaquest/sra/analytics.py:163`
- Test: `tests/test_data_file_io.py`, `tests/test_data_sra.py`, `tests/test_cli_genome.py`, `tests/test_cli_status.py`

**Interfaces:**
- Produces: `is_hidden_name(name: str) -> bool` (True for names starting with "." including "._"), `visible_files(directory, *patterns: str, dirs: bool = False) -> List[Path]` (sorted, deduplicated, hidden names removed, `dirs=True` returns directories instead of files; a missing directory returns `[]`), and `list_files(directory, pattern="*", include_hidden=False)`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests for the helper**

Append to `tests/test_data_file_io.py`:

```python
from metaquest.data.file_io import is_hidden_name, visible_files


class TestVisibleFiles:
    def test_hidden_names(self):
        assert is_hidden_name("._SRR1_1.fastq.gz")
        assert is_hidden_name(".DS_Store")
        assert not is_hidden_name("SRR1_1.fastq.gz")

    def test_visible_files_drops_appledouble_and_sorts(self, tmp_path):
        (tmp_path / "b.fna").write_text(">b\nA\n")
        (tmp_path / "a.fna").write_text(">a\nA\n")
        (tmp_path / "._a.fna").write_bytes(b"\x00\x05\x16\x07")
        (tmp_path / ".DS_Store").write_bytes(b"\x00")
        (tmp_path / "sub").mkdir()
        assert [p.name for p in visible_files(tmp_path, "*.fna")] == ["a.fna", "b.fna"]
        assert [p.name for p in visible_files(tmp_path, "*", dirs=True)] == ["sub"]
        assert visible_files(tmp_path / "missing", "*") == []

    def test_visible_files_two_patterns_no_duplicates(self, tmp_path):
        (tmp_path / "x.fastq.gz").write_text("@r\nA\n+\nI\n")
        assert [p.name for p in visible_files(tmp_path, "*.fastq.gz", "*.gz")] == ["x.fastq.gz"]

    def test_list_files_hides_dotfiles_by_default(self, tmp_path):
        (tmp_path / "g.csv").write_text("a\n")
        (tmp_path / "._g.csv").write_bytes(b"\x00")
        from metaquest.data.file_io import list_files
        assert [p.name for p in list_files(tmp_path, "*.csv")] == ["g.csv"]
        assert len(list_files(tmp_path, "*.csv", include_hidden=True)) == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_data_file_io.py::TestVisibleFiles -v`
Expected: FAIL with `ImportError: cannot import name 'is_hidden_name'`

- [ ] **Step 3: Implement the helper**

In `metaquest/data/file_io.py`, replace `list_files` (lines 40-56) with:

```python
def is_hidden_name(name: str) -> bool:
    """True for a dotfile name, including the ``._<name>`` AppleDouble files macOS writes next to
    every file on a volume without native extended attributes (ExFAT, SMB, some NAS shares)."""
    return name.startswith(".")


def visible_files(directory: Union[str, Path], *patterns: str, dirs: bool = False) -> List[Path]:
    """Entries directly in ``directory`` matching any of ``patterns``, hidden names removed.

    Returns files (or directories when ``dirs`` is True), sorted by name, without duplicates.
    A directory that does not exist yields an empty list. Every folder listing in metaquest
    goes through here so that ``._*`` and ``.DS_Store`` never count as data.
    """
    base = Path(directory)
    if not base.is_dir():
        return []
    found = set()
    for pattern in patterns or ("*",):
        for candidate in base.glob(pattern):
            if is_hidden_name(candidate.name):
                continue
            try:
                keep = candidate.is_dir() if dirs else candidate.is_file()
            except OSError:
                continue
            if keep:
                found.add(candidate)
    return sorted(found)


def list_files(directory: Union[str, Path], pattern: str = "*", include_hidden: bool = False) -> List[Path]:
    """List the files in ``directory`` matching ``pattern``; hidden names are dropped unless asked for."""
    try:
        directory = Path(directory)
        if include_hidden:
            return list(directory.glob(pattern))
        return visible_files(directory, pattern)
    except Exception as e:
        logger.warning(f"Error listing files in {directory} with pattern {pattern}: {e}")
        return []
```

- [ ] **Step 4: Run the helper tests**

Run: `pytest tests/test_data_file_io.py -v`
Expected: PASS (the existing `TestListFiles` tests keep passing; if one of them lists a dotfile on purpose, pass `include_hidden=True` in that test).

- [ ] **Step 5: Write the failing regression tests for the callers**

Append to `tests/test_data_sra.py`:

```python
class TestFastqFilesIgnoresAppleDouble:
    def test_appledouble_not_listed(self, tmp_path):
        from metaquest.data.sra import fastq_files, accession_has_fastq
        acc = tmp_path / "SRR1"
        acc.mkdir()
        (acc / "SRR1_1.fastq.gz").write_bytes(b"x" * 10)
        (acc / "._SRR1_1.fastq.gz").write_bytes(b"\x00\x05\x16\x07")
        assert [p.name for p in fastq_files(acc)] == ["SRR1_1.fastq.gz"]

    def test_folder_with_only_appledouble_has_no_fastq(self, tmp_path):
        from metaquest.data.sra import fastq_files, accession_has_fastq
        acc = tmp_path / "SRR2"
        acc.mkdir()
        (acc / "._SRR2_1.fastq.gz").write_bytes(b"\x00\x05\x16\x07")
        assert fastq_files(acc) == []
        assert accession_has_fastq(acc) is False
```

Append to `tests/test_cli_genome.py` (inside or beside `TestGenomeDownloadCommand`, using its existing manifest test as the template for constructing the command and args):

```python
def test_manifest_ignores_appledouble_files(tmp_path, monkeypatch):
    from metaquest.cli.commands.genome import GenomePrepareCommand
    monkeypatch.chdir(tmp_path)
    genomes = tmp_path / "genomes"
    genomes.mkdir()
    (genomes / "GCF_1.fna").write_text(">c\nACGT\n")
    (genomes / "._GCF_1.fna").write_bytes(b"\x00\x05\x16\x07")
    count = GenomePrepareCommand()._create_manifest(genomes, "genome_manifest.csv", str(tmp_path / "r.json"))
    assert count == 1
    assert "._GCF_1" not in (tmp_path / "genome_manifest.csv").read_text()
```

Append to `tests/test_cli_status.py` (reuse `_make_tree` and `_status_args` at lines 45 and 171):

```python
def test_inventory_ignores_appledouble(tmp_path):
    root = _make_tree(tmp_path)
    (root / "metadata" / "._SRR1_metadata.xml").write_bytes(b"\x00\x05")
    (root / "genomes" / "._g.fna").write_bytes(b"\x00\x05")
    (root / "fastq" / "._SRR1").mkdir()
    report = StatusCommand()._inventory_report(root / "fastq", root / "metadata", root / "genomes")
    assert report["metadata_xml"] == 1
    assert report["genome_fasta"] == 1
    assert report["fastq_accessions"] == 1
```

(Adjust the `_inventory_report` call to its actual signature at `status.py:135`; the test intent is the three counts.)

- [ ] **Step 6: Run them to verify they fail**

Run: `pytest tests/test_data_sra.py::TestFastqFilesIgnoresAppleDouble tests/test_cli_genome.py::test_manifest_ignores_appledouble_files tests/test_cli_status.py::test_inventory_ignores_appledouble -v`
Expected: FAIL (two files listed; count 2; metadata_xml 2).

- [ ] **Step 7: Route the callers through the helper**

`metaquest/data/sra.py` `fastq_files`:

```python
from metaquest.data.file_io import visible_files

def fastq_files(acc_dir: Union[str, Path]) -> List[Path]:
    """Non-empty, visible FASTQ files directly in ``acc_dir``, sorted by name (see ``visible_files``)."""
    found = []
    for candidate in visible_files(acc_dir, *FASTQ_GLOBS):
        try:
            if candidate.stat().st_size > 0:
                found.append(candidate)
        except OSError:
            continue
    return found
```

`metaquest/cli/commands/genome.py:312`:

```python
genome_files = visible_files(output_dir, *GENOME_FASTA_GLOBS)
```

`metaquest/cli/commands/status.py:141-147`:

```python
on_disk_fastq = sorted(
    d.name for d in visible_files(fastq_dir, dirs=True) if not is_transient_folder(d.name) and accession_has_fastq(d)
)
on_disk_meta = sorted(p.name[: -len("_metadata.xml")] for p in visible_files(meta_dir, "*_metadata.xml"))
on_disk_genomes = sorted(p.name for p in visible_files(genomes_dir, *GENOME_FASTA_GLOBS))
```

`metaquest/data/registry.py`: at 729 use `visible_files(fastq_folder, dirs=True)`; at 741 `visible_files(metadata_folder, "*_metadata.xml")`; at 745-753 `visible_files(genomes, *GENOME_FASTA_GLOBS)`, `visible_files(paths.matches, "*.csv")`, `visible_files(targeted, dirs=True)`; at 798 `visible_files(matches_folder, "*.csv")`.

`metaquest/data/branchwater.py:226`: `csv_files = visible_files(source_path, "*.csv")`.

`metaquest/sra/analytics.py:163`: `for path in visible_files(directory, f"*{_QUALITY_PROFILE_SUFFIX}"):`.

Check the imports (`from metaquest.data.file_io import visible_files`) do not create an import cycle: `file_io.py` imports only `metaquest.core.exceptions`, so `data/sra.py`, `data/registry.py` and `cli/commands/*` can import it.

- [ ] **Step 8: Run the whole suite and the quality gates**

Run: `make test && make check`
Expected: all pass. If a status test asserted a dotfile count, update the assertion (the new behaviour is the documented one).

- [ ] **Step 9: Commit**

```bash
git add metaquest/data/file_io.py metaquest/data/sra.py metaquest/cli/commands/genome.py metaquest/cli/commands/status.py metaquest/data/registry.py metaquest/data/branchwater.py metaquest/sra/analytics.py tests/test_data_file_io.py tests/test_data_sra.py tests/test_cli_genome.py tests/test_cli_status.py
git commit -m "fix: ignore AppleDouble and dotfiles in every folder listing (S1-3, S2-3, S4-3, S5-2, S6-6)"
```

---

### Task 2: Store folder scans, copies and removals ignore hidden names

**Files:**
- Modify: `metaquest/store/adopt.py:230-239, 293, 306-324, 327-346`
- Modify: `metaquest/cli/commands/store.py:385-404, 639-644, 1413-1442`
- Modify: `metaquest/data/sra.py` `_safe_rmtree`
- Test: `tests/test_store_adopt.py`, `tests/test_cli_store.py`

**Interfaces:**
- Consumes: `visible_files`, `is_hidden_name` from Task 1.
- Produces: `ADOPT_COPY_IGNORE = shutil.ignore_patterns("._*", ".DS_Store")` in `adopt.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store_adopt.py` (helpers `_write_fastq_gz` at line 21, `init_store`, `adopt` are already imported there):

```python
class TestAdoptIgnoresAppleDouble:
    def test_staged_copy_has_no_appledouble_and_sidecar_is_complete(self, tmp_path):
        paths = init_store(tmp_path / "store")
        project = tmp_path / "proj" / "fastq"
        acc = project / "SRR1"
        acc.mkdir(parents=True)
        _write_fastq_gz(acc / "SRR1_1.fastq.gz")
        (acc / "._SRR1_1.fastq.gz").write_bytes(b"\x00\x05\x16\x07")
        (project / "._SRR1").write_bytes(b"\x00\x05")
        (project / ".DS_Store").write_bytes(b"\x00")
        report = adopt(project, paths, move=True, dry_run=False, compress=True, metadata_folders=[], lock_wait=0)
        assert report.adopted == ["SRR1"]
        names = sorted(p.name for p in (paths.sra / "SRR1").iterdir())
        assert "._SRR1_1.fastq.gz" not in names
        sc = read_sidecar(sidecar_path(paths, "SRR1"))
        assert sc.state == "complete"
        assert [f["name"] for f in sc.files] == ["SRR1_1.fastq.gz"]
```

Append to `tests/test_cli_store.py` (builders `_reindex_args`, `_verify_args`, `_sidecar` exist at lines 49-129):

```python
def test_reindex_and_verify_ignore_hidden_entries(tmp_path, monkeypatch):
    paths = init_store(tmp_path / "store")
    monkeypatch.chdir(tmp_path)
    write_sidecar(sidecar_path(paths, "SRR1"), _sidecar("SRR1"))
    (paths.sra / "._SRR1").write_bytes(b"\x00\x05")
    (paths.sra / ".sra-cache").mkdir()
    (paths.sra / ".hidden_dir").mkdir()
    assert StoreReindexCommand().execute(_reindex_args(data_root=str(paths.root))) == 0
    rc = StoreVerifyCommand().execute(_verify_args(data_root=str(paths.root)))
    assert rc == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_store_adopt.py::TestAdoptIgnoresAppleDouble tests/test_cli_store.py::test_reindex_and_verify_ignore_hidden_entries -v`
Expected: FAIL (`._SRR1_1.fastq.gz` copied, sidecar failed; reindex reports `.hidden_dir` unreadable and returns 1).

- [ ] **Step 3: Implement**

`metaquest/store/adopt.py`:

```python
from metaquest.data.file_io import is_hidden_name, visible_files

ADOPT_COPY_IGNORE = shutil.ignore_patterns("._*", ".DS_Store")

def _folder_bytes(folder: Path) -> int:
    total = 0
    for sub in folder.rglob("*"):
        if is_hidden_name(sub.name):
            continue
        try:
            if sub.is_file():
                total += sub.stat().st_size
        except OSError:
            continue
    return total
```

In `_scan_project_dir`, iterate `for candidate in sorted(project_dir.iterdir()):` and add `if is_hidden_name(name): continue` right after `name = candidate.name`. In `_scan_foreign_incomplete`, replace the loop header with `for store_dir in visible_files(paths.sra, dirs=True):`. At line 293 use `shutil.copytree(entry, staged, ignore=ADOPT_COPY_IGNORE)`.

`metaquest/cli/commands/store.py`: in `_read_all_sidecars` iterate `for acc_dir in visible_files(paths.sra, dirs=True):` (drop the `.sra-cache` special case, it is hidden); in `_accessions_to_check` return `[p.name for p in visible_files(paths.sra, dirs=True)]`; in `_leftover_candidates` skip entries whose `is_hidden_name(name)` is True except the explicit `.sra-cache` handling that already exists.

`metaquest/data/sra.py` `_safe_rmtree`: pass `onerror` (or `onexc` on 3.12) that ignores `FileNotFoundError`, so macOS deleting `._X` together with `X` no longer produces "Could not remove directory ... No such file or directory: '._X'":

```python
def _safe_rmtree(path: Path) -> None:
    def _ignore_missing(func, target, exc_info):
        exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
        if isinstance(exc, FileNotFoundError):
            return
        raise exc

    try:
        shutil.rmtree(path, onerror=_ignore_missing)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not remove directory %s: %s", path, e)
```

(Keep the existing behaviour for other errors; read the current `_safe_rmtree` body first and preserve its logging text.)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_store_adopt.py tests/test_cli_store.py tests/test_data_sra.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/store/adopt.py metaquest/cli/commands/store.py metaquest/data/sra.py tests/test_store_adopt.py tests/test_cli_store.py
git commit -m "fix: store scans, copies and removals ignore AppleDouble files (S5-2, S5-7)"
```

---

### Task 3: Match-file errors fail the command

**Files:**
- Modify: `metaquest/data/branchwater.py:88-141, 200-256, 429-490`
- Modify: `metaquest/cli/commands/branchwater.py:44-50, 80-90`, `metaquest/cli/commands/containment.py` (parse_containment execute)
- Test: `tests/test_data_branchwater.py`, `tests/test_cli_commands.py`

**Interfaces:**
- Produces: `process_branchwater_files(...) -> Dict[str, Path]` unchanged, plus a new module-level `class FileErrors(Exception)` is NOT introduced; instead each of the three functions gains a keyword `errors: Optional[List[str]] = None` list that the caller passes and inspects (the failing file names are appended). The CLI commands return 1 when the list is non-empty.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_branchwater.py` in `TestProcessBranchwaterFiles`:

```python
    def test_unreadable_csv_is_reported_to_caller(self, tmp_path):
        source = tmp_path / "bw"
        target = tmp_path / "matches"
        source.mkdir()
        (source / "good.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\n")
        (source / "bad.csv").write_bytes(b"\x00\x05\x16\x07\xb0")
        errors: list = []
        result = process_branchwater_files(source, target, errors=errors)
        assert "good" in result
        assert errors == ["bad.csv"]
```

And a CLI test in `tests/test_cli_commands.py` (find `TestUseBranchwaterCommand` and add):

```python
    def test_use_branchwater_returns_1_on_file_error(self, tmp_path):
        source = tmp_path / "bw"
        source.mkdir()
        (source / "bad.csv").write_bytes(b"\x00\x05\x16\x07\xb0")
        args = argparse.Namespace(branchwater_folder=str(source), matches_folder=str(tmp_path / "m"))
        assert UseBranchwaterCommand().execute(args) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data_branchwater.py -k unreadable tests/test_cli_commands.py -k returns_1_on_file_error -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'errors'`; exit 0).

- [ ] **Step 3: Implement**

In each of `process_branchwater_files`, `extract_metadata_from_branchwater` and `parse_containment_data`, add the parameter `errors: Optional[List[str]] = None` and, in the per-file `except Exception as e:` block, add `if errors is not None: errors.append(csv_file.name)`. Change the summary line to `logger.warning(...)` when `error_count` is non-zero:

```python
    level = logger.warning if error_count else logger.info
    level(f"Processed {processed_count} files with {error_count} errors")
```

In `metaquest/cli/commands/branchwater.py` (both commands) and the parse_containment command in `containment.py`:

```python
            errors: List[str] = []
            process_branchwater_files(args.branchwater_folder, args.matches_folder, errors=errors)
            if errors:
                self.logger.error("%d match file(s) could not be read: %s", len(errors), ", ".join(errors))
                return 1
            return 0
```

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_data_branchwater.py tests/test_cli_commands.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/data/branchwater.py metaquest/cli/commands/branchwater.py metaquest/cli/commands/containment.py tests/test_data_branchwater.py tests/test_cli_commands.py
git commit -m "fix: unreadable match files fail use_branchwater, parse_containment and extract_branchwater_metadata (S2-6)"
```

---

### Task 4: Project and usage journal; reindex replays it; gc refuses without project records

**Files:**
- Create: `metaquest/store/journal.py`
- Modify: `metaquest/store/layout.py` (`StorePaths.journal`, created by `init_store`), `metaquest/store/catalog.py:293-340` (call the journal), `metaquest/cli/commands/store.py:358-440` (reindex replay), `metaquest/cli/commands/store.py:1359-1402, 1466-1520` (gc refusal)
- Test: `tests/test_store_journal.py` (new), `tests/test_cli_store.py`, `tests/test_store_gc.py`

**Interfaces:**
- Produces: `journal.append_project(paths, project_id, name, path, registry)`, `journal.append_usage(paths, accession, project_id, genome_id, stage, detail)`, `journal.replay(paths, catalog) -> Tuple[int, int]` (projects, usage rows restored). `StorePaths.journal: Path` = `<root>/journal`. Files: `projects.jsonl`, `usage.jsonl`, one JSON object per line, appended with `open(..., "a")` under the existing store-wide catalogue write lock (`catalog_write` holds it, so append inside `Catalog.upsert_project`/`record_usage`, which run within it).
- Consumes: `Catalog.upsert_project`, `Catalog.record_usage` (they call the journal after their SQL).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_store_journal.py`:

```python
import json

from metaquest.store.catalog import catalog_write
from metaquest.store.layout import init_store
from metaquest.store import journal


def test_upsert_project_and_usage_are_journaled(tmp_path):
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        c.upsert_project("pid1", "proj", str(tmp_path / "proj"), str(tmp_path / "proj" / "r.json"))
        c.record_usage("SRR1", "pid1", "GCF_1", "linked", "test")
    projects = [json.loads(line) for line in (paths.journal / "projects.jsonl").read_text().splitlines()]
    usage = [json.loads(line) for line in (paths.journal / "usage.jsonl").read_text().splitlines()]
    assert projects[0]["project_id"] == "pid1" and projects[0]["name"] == "proj"
    assert usage[0]["accession"] == "SRR1" and usage[0]["stage"] == "linked"


def test_replay_restores_projects_and_usage_into_a_fresh_catalog(tmp_path):
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        c.upsert_project("pid1", "proj", str(tmp_path / "proj"), "r.json")
        c.record_usage("SRR1", "pid1", "", "downloaded", "")
    (paths.root / "catalog.sqlite").unlink()
    with catalog_write(paths) as c:
        restored = journal.replay(paths, c)
        assert restored == (1, 1)
        assert c.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
        assert c.conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0] == 1


def test_replay_without_journal_returns_zero(tmp_path):
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        assert journal.replay(paths, c) == (0, 0)
```

Append to `tests/test_cli_store.py` `TestStoreReindexNeverLosesHistory` (line 1320):

```python
    def test_reindex_after_catalog_loss_restores_projects_and_usage(self, tmp_path, monkeypatch):
        paths = init_store(tmp_path / "store")
        monkeypatch.chdir(tmp_path)
        write_sidecar(sidecar_path(paths, "SRR1"), _sidecar("SRR1"))
        with catalog_write(paths) as c:
            c.upsert_dataset(_sidecar("SRR1"))
            c.upsert_project("pid1", "proj", str(tmp_path), "r.json")
            c.record_usage("SRR1", "pid1", "", "linked", "")
        (paths.root / "catalog.sqlite").unlink()
        assert StoreReindexCommand().execute(_reindex_args(data_root=str(paths.root))) == 0
        with catalog_write(paths) as c:
            assert c.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
            assert c.conn.execute("SELECT COUNT(*) FROM usage WHERE accession='SRR1'").fetchone()[0] == 1
```

Append to `tests/test_store_gc.py` `TestStoreGcCommand` (helpers `_gc_args`, `_sidecar`, `_write_dataset_dir` at lines 28-60):

```python
    def test_gc_refuses_when_catalog_has_no_projects(self, tmp_path, capsys):
        paths = init_store(tmp_path / "store")
        _write_dataset_dir(paths, "SRR1")
        write_sidecar(sidecar_path(paths, "SRR1"), _sidecar("SRR1"))
        with catalog_write(paths) as c:
            c.upsert_dataset(_sidecar("SRR1"))
        rc = StoreGcCommand().execute(_gc_args(data_root=str(paths.root), yes=True))
        assert rc == 1
        assert (paths.sra / "SRR1" / "SRR1.fastq.gz").exists()
        err = capsys.readouterr().err
        assert "no project" in err.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_store_journal.py tests/test_cli_store.py -k restores_projects tests/test_store_gc.py -k no_projects -v`
Expected: FAIL (`ModuleNotFoundError: metaquest.store.journal`; gc returns 0 and removes the dataset).

- [ ] **Step 3: Implement the journal**

`metaquest/store/layout.py`: add `journal: Path` to `StorePaths` (value `root / "journal"`) and create it in `init_store` alongside `tmp`, `locks`, `metadata`, `sra`. Read `layout.py:24-92` first and follow how `tmp` is declared and created.

Create `metaquest/store/journal.py`:

```python
"""Append-only journal of the store's project and usage records.

The SQLite catalogue is the working copy of "which project used which dataset", but it is the
only copy: sidecars know nothing about projects. ``store_reindex`` rebuilds the catalogue from
sidecars and would otherwise come back with no projects and no usage, after which ``store_gc``
sees every dataset as unused. Every ``upsert_project`` and ``record_usage`` therefore also
appends one JSON line here, and ``replay`` feeds those lines back into a rebuilt catalogue.
Appends happen inside ``catalog_write``, which holds the store-wide write lock.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from metaquest.store.layout import StorePaths

logger = logging.getLogger(__name__)

PROJECTS_FILE = "projects.jsonl"
USAGE_FILE = "usage.jsonl"


def _append(paths: StorePaths, name: str, record: Dict[str, Any]) -> None:
    paths.journal.mkdir(parents=True, exist_ok=True)
    record = dict(record, at=datetime.now(timezone.utc).isoformat())
    with open(paths.journal / name, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_project(paths: StorePaths, project_id: str, name: str, path: str, registry: str) -> None:
    _append(paths, PROJECTS_FILE, {"project_id": project_id, "name": name, "path": path, "registry": registry})


def append_usage(paths: StorePaths, accession: str, project_id: str, genome_id: str, stage: str, detail: str) -> None:
    _append(
        paths,
        USAGE_FILE,
        {"accession": accession, "project_id": project_id, "genome_id": genome_id, "stage": stage, "detail": detail},
    )


def _lines(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable journal line %s:%d", path, number)


def replay(paths: StorePaths, catalog: Any) -> Tuple[int, int]:
    """Feed every journaled project and usage record into ``catalog``; returns the two counts.

    The catalogue calls back into ``append_*`` when it writes, so replay runs with journaling
    suspended (``catalog.journal_enabled = False``) to avoid duplicating the file.
    """
    projects = 0
    usage = 0
    previous = getattr(catalog, "journal_enabled", True)
    catalog.journal_enabled = False
    try:
        for record in _lines(paths.journal / PROJECTS_FILE):
            catalog.upsert_project(record["project_id"], record.get("name", ""), record.get("path", ""), record.get("registry", ""))
            projects += 1
        for record in _lines(paths.journal / USAGE_FILE):
            catalog.record_usage(
                record["accession"], record["project_id"], record.get("genome_id", ""), record.get("stage", ""), record.get("detail", "")
            )
            usage += 1
    finally:
        catalog.journal_enabled = previous
    return projects, usage
```

`metaquest/store/catalog.py`: give `Catalog` an attribute `journal_enabled: bool = True` (set in `__init__`), and at the end of `upsert_project` add:

```python
        if self.journal_enabled:
            from metaquest.store import journal
            journal.append_project(self.paths, project_id, name, path, registry)
```

and at the end of `record_usage` (after the INSERT):

```python
        if self.journal_enabled:
            from metaquest.store import journal
            journal.append_usage(self.paths, accession, project_id, genome_id, stage, detail)
```

(The `Catalog` holds `self.paths`; confirm at `catalog.py` `__init__`.)

- [ ] **Step 4: Reindex replays, gc refuses**

`StoreReindexCommand.execute`: after `count = catalog.reindex(sidecars)` add:

```python
                projects, usage = journal.replay(paths, catalog)
                if projects == 0:
                    self.logger.warning(
                        "No project records could be restored (no journal under %s); every dataset will look unused "
                        "until each project runs store_init or store_link again",
                        paths.journal,
                    )
            print(f"Reindexed {count} dataset(s); restored {projects} project(s) and {usage} usage record(s)")
```

`StoreGcCommand.execute`: before building candidates, inside the `catalog` context:

```python
            project_count = catalog.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            if project_count == 0 and any(True for _ in catalog.conn.execute("SELECT 1 FROM datasets LIMIT 1")):
                self.logger.error(
                    "The catalogue records no project at all, so nothing can be told apart from unused data. "
                    "Run store_reindex (which replays the journal) or store_init from each project first."
                )
                return 1
```

- [ ] **Step 5: Run tests and gates**

Run: `pytest tests/test_store_journal.py tests/test_cli_store.py tests/test_store_gc.py tests/test_store_catalog.py tests/test_store_usage.py -q && make check`
Expected: PASS. Existing gc tests that build a catalogue with datasets but no projects and expect removal must add `c.upsert_project(...)` first; the test names will tell you which (they now get exit 1).

- [ ] **Step 6: Commit**

```bash
git add metaquest/store/journal.py metaquest/store/layout.py metaquest/store/catalog.py metaquest/cli/commands/store.py tests/test_store_journal.py tests/test_cli_store.py tests/test_store_gc.py
git commit -m "feat: journal project and usage records; reindex replays them; gc refuses without projects (S5-14, S5-15)"
```

---

### Task 5: store_adopt refuses empty folders, verifies before removing, honours --copy on dedup

**Files:**
- Modify: `metaquest/store/adopt.py:267-303 (_stage_into_store), 306-324, 349-382, 385-409, 412-463`
- Test: `tests/test_store_adopt.py`

**Interfaces:**
- Produces: `AdoptReport.empty: List[str]` (new field, accessions skipped because the folder held no FASTQ), `AdoptReport.failed: List[str]` (staged copy came back `failed`/`partial`; project copy kept).
- Consumes: `fastq_files` (Task 1), `read_sidecar`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store_adopt.py`:

```python
class TestAdoptSafety:
    def test_empty_folder_is_not_adopted(self, tmp_path):
        paths = init_store(tmp_path / "store")
        project = tmp_path / "proj" / "fastq"
        (project / "SRR9").mkdir(parents=True)
        report = adopt(project, paths, move=True, dry_run=False, compress=True, metadata_folders=[], lock_wait=0)
        assert report.adopted == []
        assert report.empty == ["SRR9"]
        assert not (paths.sra / "SRR9").exists()
        assert (project / "SRR9").is_dir() and not (project / "SRR9").is_symlink()

    def test_failed_staged_copy_keeps_project_folder(self, tmp_path, monkeypatch):
        paths = init_store(tmp_path / "store")
        project = tmp_path / "proj" / "fastq"
        acc = project / "SRR1"
        acc.mkdir(parents=True)
        _write_fastq_gz(acc / "SRR1_1.fastq.gz")
        import metaquest.store.adopt as adopt_mod
        real_build = adopt_mod.build_sidecar

        def failing_build(*a, **kw):
            sc = real_build(*a, **kw)
            sc.state = "failed"
            sc.error = "simulated gzip error"
            return sc

        monkeypatch.setattr(adopt_mod, "build_sidecar", failing_build)
        report = adopt(project, paths, move=True, dry_run=False, compress=True, metadata_folders=[], lock_wait=0)
        assert report.failed == ["SRR1"]
        assert (project / "SRR1" / "SRR1_1.fastq.gz").exists()
        assert not (project / "SRR1").is_symlink()

    def test_copy_dedup_leaves_project_folder(self, tmp_path):
        paths = init_store(tmp_path / "store")
        project = tmp_path / "proj" / "fastq"
        acc = project / "SRR1"
        acc.mkdir(parents=True)
        _write_fastq_gz(acc / "SRR1_1.fastq.gz")
        adopt(project, paths, move=False, dry_run=False, compress=True, metadata_folders=[], lock_wait=0)
        assert (project / "SRR1").is_dir() and not (project / "SRR1").is_symlink()
        report = adopt(project, paths, move=False, dry_run=False, compress=True, metadata_folders=[], lock_wait=0)
        assert report.deduplicated == ["SRR1"]
        assert not (project / "SRR1").is_symlink()
        assert (project / "SRR1" / "SRR1_1.fastq.gz").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_store_adopt.py::TestAdoptSafety -v`
Expected: FAIL (`AttributeError: 'AdoptReport' object has no attribute 'empty'`; symlink created in the other two).

- [ ] **Step 3: Implement**

In `AdoptReport` (find the dataclass near the top of `adopt.py`) add `empty: List[str] = field(default_factory=list)` and `failed: List[str] = field(default_factory=list)`.

In `adopt(...)` (466-536), before dispatching each accession from `real_dirs`:

```python
        if not fastq_files(entry):
            logger.warning("%s: no FASTQ files in %s; not adopted", accession, entry)
            report.empty.append(accession)
            continue
```

Thread `move` into `_dedup_or_conflict` (add parameter `move: bool` after `dry_run`) and change the dedup branch:

```python
        if not move:
            report.deduplicated.append(accession)
            logger.info("%s: identical copy already in the store; project copy kept (--copy)", accession)
            _notify(on_progress, accession, "deduplicated")
            return
        shutil.rmtree(entry)
        link_dataset(project_dir, accession, paths)
```

Update both call sites (`_adopt_one` passes `move`; the `adopt` loop passes `move`).

In `_adopt_one`, after `_stage_into_store`/`_finish_sidecar` and before `_apply_move_or_copy`:

```python
    published = read_sidecar(sc_path)
    if published is None or published.state not in ("complete", "unverified"):
        state = published.state if published else "missing sidecar"
        logger.error("%s: store copy is %s (%s); project copy kept", accession, state, (published.error if published else ""))
        report.failed.append(accession)
        return
```

(`"unverified"` is not a sidecar state today; keep the tuple to `("complete",)` if `build_sidecar` never emits it. Check `STORE_READY_STATES` in `data/sra.py:824` and use the same constant.)

Print the two new buckets in `StoreAdoptCommand` (`store.py:443-605`) next to `conflicts` and `skipped`: `f"..., empty {len(report.empty)}, failed {len(report.failed)}"`.

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_store_adopt.py tests/test_cli_store.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/store/adopt.py metaquest/cli/commands/store.py tests/test_store_adopt.py
git commit -m "fix: store_adopt skips empty folders, keeps the project copy on a failed stage and honours --copy (S5-7, S5-8, S5-12)"
```

---

### Task 6: Download scratch stays in the store; final download count

**Files:**
- Modify: `metaquest/data/sra.py:914-969 (_store_fetch), 1096-1135 (_process_download_results)`
- Modify: `metaquest/cli/commands/sra.py:214-222`
- Test: `tests/test_data_sra.py`, `tests/test_cli_commands.py`

**Interfaces:**
- Produces: `_store_fetch` passes `temp_folder=store.tmp / f"{accession}_fqtmp"` when the caller gave none and removes it afterwards; the summary logs "Downloaded N of M (k failed)" once at the end; the CLI summary adds "Linked from store: k datasets".

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data_sra.py` in `TestDownloadSraStore` (helpers `_store`, `_accessions`, `_fake_download` at 1890-1927):

```python
    def test_store_download_uses_store_tmp_for_fasterq_scratch(self, tmp_path):
        paths = _store(tmp_path)
        calls = []
        with patch("metaquest.data.sra.download_accession", side_effect=_fake_download(calls)):
            download_sra(_accessions(tmp_path, "SRR1"), str(tmp_path / "fastq"), data_root=str(paths.root))
        kwargs = calls[0][1]
        assert Path(kwargs["temp_folder"]).parent == paths.tmp
        assert Path(kwargs["temp_folder"]).name == "SRR1_fqtmp"
        assert not Path(kwargs["temp_folder"]).exists()
```

(Match how `_fake_download` records `calls`; if it stores `(args, kwargs)`, index accordingly. Read `test_data_sra.py:1914-1927`.)

Append to `TestProcessDownloadResults` (line 1376):

```python
    def test_final_count_logged_once(self, caplog):
        results = [("SRR1", (True, "ok")), ("SRR2", (False, "boom")), ("SRR3", (True, "linked from store, 2 files"))]
        with caplog.at_level(logging.INFO, logger="metaquest.data.sra"):
            ok, failed = _process_download_results(results, ["SRR1", "SRR2", "SRR3"], {}, [])
        assert (ok, failed) == (2, 1)
        assert "Downloaded 2 of 3 (1 failed)" in caplog.text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data_sra.py -k "fasterq_scratch or final_count_logged" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`_store_fetch` (after the `sra_cache` default at 942-944):

```python
    own_scratch: Optional[Path] = None
    if download_kwargs.get("temp_folder") is None:
        own_scratch = store.tmp / f"{accession}_fqtmp"
        download_kwargs["temp_folder"] = own_scratch
    ...
    try:
        success, message = download_accession(accession, store.tmp, staging_folder=store.tmp, **download_kwargs)
    finally:
        if own_scratch is not None:
            _safe_rmtree(own_scratch)
```

`is_transient_folder`: also return True for names ending in `_fqtmp`, so status and gc treat the scratch as transient.

`_process_download_results`: remove the `% 5` block and after the loop add:

```python
    logger.info("Downloaded %d of %d (%d failed)", successful_count, len(accessions_to_download), failed_count)
```

`cli/commands/sra.py` `_log_download_summary`: count messages starting with `STORE_LINKED_PREFIX` in `stats` (add `stats["linked"]` where `stats` is built in the command; read `_result_recorder` at 346-379) and print `f"  Linked from store: {stats['linked']} datasets"`.

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_data_sra.py tests/test_cli_commands.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/data/sra.py metaquest/cli/commands/sra.py tests/test_data_sra.py tests/test_cli_commands.py
git commit -m "fix: fasterq-dump scratch under the store tmp folder; one final download count (S5-1, S5-3)"
```

---

### Task 7: Ctrl-C stops download_sra and its child processes

**Files:**
- Modify: `metaquest/utils/security.py:229-291`
- Modify: `metaquest/data/sra.py:1292-1350, 1138-1264`
- Test: `tests/test_utils_security.py` (or the existing security test file), `tests/test_data_sra.py`

**Interfaces:**
- Produces: `SecureSubprocess.terminate_children(grace: float = 5.0) -> int` (terminates then kills every tracked child, returns the count), module-level `SecureSubprocess._children: Set[subprocess.Popen]` guarded by `threading.Lock`; `metaquest.data.sra.STOP = threading.Event()` checked by `download_accession` before each tool call; `_execute_parallel_downloads` re-raises `KeyboardInterrupt` after cancelling.

- [ ] **Step 1: Write the failing tests**

Add to the security test file:

```python
def test_run_secure_tracks_children_and_terminate_children_kills_them():
    import threading, time
    from metaquest.utils.security import SecureSubprocess
    started = threading.Event()

    def run_sleep():
        started.set()
        try:
            SecureSubprocess.run_secure("sleep", ["30"])
        except Exception:
            pass

    t = threading.Thread(target=run_sleep)
    t.start()
    started.wait(2)
    time.sleep(0.3)
    assert SecureSubprocess.terminate_children(grace=1.0) == 1
    t.join(5)
    assert not t.is_alive()
```

(`sleep` must be an allowed executable; if `run_secure` validates the name against an allow list, add `"sleep"` for the test via the existing allow mechanism or patch `SecureSubprocess.ALLOWED_EXECUTABLES`; read `security.py:1-60`.)

Append to `tests/test_data_sra.py`:

```python
class TestDownloadInterrupt:
    def test_keyboard_interrupt_cancels_pending_and_terminates_children(self, tmp_path):
        from metaquest.data import sra as sra_mod
        calls = []

        def worker(acc, *a, **kw):
            calls.append(acc)
            if acc == "SRR1":
                raise KeyboardInterrupt
            return True, "ok"

        with patch.object(sra_mod.SecureSubprocess, "terminate_children", return_value=0) as term:
            with pytest.raises(KeyboardInterrupt):
                sra_mod._execute_parallel_downloads(
                    ["SRR1", "SRR2", "SRR3", "SRR4"], tmp_path, 1, 1, False, None, {}, [], downloader=worker
                )
        term.assert_called_once()
        assert sra_mod.STOP.is_set()
        sra_mod.STOP.clear()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests -k "terminate_children or TestDownloadInterrupt" -v`
Expected: FAIL (`AttributeError: terminate_children`; `STOP` missing).

- [ ] **Step 3: Implement**

`metaquest/utils/security.py`: replace the `subprocess.run(cmd, **secure_kwargs)` call with a tracked `Popen`:

```python
    _children: Set[subprocess.Popen] = set()
    _children_lock = threading.Lock()

    @classmethod
    def terminate_children(cls, grace: float = 5.0) -> int:
        """Terminate, then kill, every child started by ``run_secure`` that is still running."""
        with cls._children_lock:
            children = list(cls._children)
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + grace
        for child in children:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.kill()
        return len(children)
```

and inside `run_secure`:

```python
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=safe_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen_kwargs
            )
            with cls._children_lock:
                cls._children.add(proc)
            try:
                out, err = proc.communicate(timeout=timeout or MAX_SUBPROCESS_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise SecurityError(f"Command timed out: {' '.join(cmd)}")
            finally:
                with cls._children_lock:
                    cls._children.discard(proc)
            completed = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd, output=out, stderr=err)
            return completed
```

Keep `**kwargs` support: the callers pass nothing else today (grep `run_secure(` to confirm); if any passes `check=False` or `capture_output`, honour `check` explicitly.

`metaquest/data/sra.py`:

```python
STOP = threading.Event()
```

In `download_accession`, before prefetch and before fasterq-dump: `if STOP.is_set(): return False, "interrupted"`. In `_execute_parallel_downloads`:

```python
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {...}
        try:
            for future in as_completed(futures):
                ...
        except KeyboardInterrupt:
            STOP.set()
            logger.warning("Interrupted; cancelling pending downloads and stopping running tools")
            executor.shutdown(wait=False, cancel_futures=True)
            SecureSubprocess.terminate_children()
            raise
```

In `_retry_failed_downloads`, return early when `STOP.is_set()`. In the CLI `download_sra` command, catch `KeyboardInterrupt` around the call, log "Download interrupted by the user" and return 130.

- [ ] **Step 4: Run tests and gates**

Run: `make test && make check`
Expected: PASS. Every existing test that patches `subprocess.run` inside `metaquest.utils.security` (for example `test_download_accession_command_passes_validation` at `test_data_sra.py:791`) must now patch `metaquest.utils.security.subprocess.Popen` instead; update those tests, keeping their assertions on the command line.

- [ ] **Step 5: Commit**

```bash
git add metaquest/utils/security.py metaquest/data/sra.py metaquest/cli/commands/sra.py tests/
git commit -m "fix: Ctrl-C cancels pending downloads and terminates prefetch and fasterq-dump (S5-10)"
```

---

### Task 8: store_verify reads spot counts from metadata, promotes readable datasets, and can rescan

**Files:**
- Modify: `metaquest/cli/commands/store.py:606-776` (`StoreVerifyCommand`)
- Modify: `metaquest/store/sidecar.py` (reuse `ncbi_from_metadata_xml`, `build_sidecar`)
- Test: `tests/test_cli_store.py` `TestStoreVerifyCommand` (line 625)

**Interfaces:**
- Produces: `--rescan` flag (rebuild the sidecar file list from disk with `build_sidecar`, keeping `ncbi`, `stats`, `downloaded`, `tool`); `--spots` falls back to `<store>/metadata/<ACC>_metadata.xml`, then `<project>/metadata/<ACC>_metadata.xml`, via `ncbi_from_metadata_xml`, and with `--fix-state` stores the found `ncbi` block in the sidecar; `--fix-state` sets `state="complete"` and clears `error` when bytes (and md5 if checked) pass and every file is readable, even when the verdict is `unverified`.

- [ ] **Step 1: Write the failing tests**

Append to `TestStoreVerifyCommand`:

```python
    def test_spots_from_store_metadata_xml(self, tmp_path, monkeypatch):
        paths = init_store(tmp_path / "store")
        monkeypatch.chdir(tmp_path)
        acc_dir = paths.sra / "SRR1"
        acc_dir.mkdir(parents=True)
        _write_fastq_gz(acc_dir / "SRR1_1.fastq.gz", "@r1\nACGT\n+\nIIII\n@r2\nACGT\n+\nIIII\n")
        sc = _sidecar_matching_disk(acc_dir, "SRR1", state="failed", reads=2)
        sc.ncbi = {}
        sc.error = "._SRR1_1.fastq.gz: missing"
        write_sidecar(sidecar_path(paths, "SRR1"), sc)
        (paths.metadata / "SRR1_metadata.xml").write_text(
            '<EXPERIMENT_PACKAGE_SET><EXPERIMENT_PACKAGE><RUN_SET><RUN accession="SRR1" total_spots="2" total_bases="8" size="100">'
            '<SRAFiles><SRAFile filename="SRR1_1.fastq.gz" md5="00"/></SRAFiles></RUN></RUN_SET>'
            "</EXPERIMENT_PACKAGE></EXPERIMENT_PACKAGE_SET>"
        )
        rc = StoreVerifyCommand().execute(_verify_args(data_root=str(paths.root), spots=True, fix_state=True))
        assert rc == 0
        fixed = read_sidecar(sidecar_path(paths, "SRR1"))
        assert fixed.state == "complete"
        assert fixed.completeness["verdict"] == "complete"
        assert fixed.ncbi.get("spots") == 2
        assert fixed.error is None

    def test_fix_state_promotes_readable_dataset_without_spots(self, tmp_path, monkeypatch):
        paths = init_store(tmp_path / "store")
        monkeypatch.chdir(tmp_path)
        acc_dir = paths.sra / "SRR1"
        acc_dir.mkdir(parents=True)
        _write_fastq_gz(acc_dir / "SRR1_1.fastq.gz")
        sc = _sidecar_matching_disk(acc_dir, "SRR1", state="failed", reads=1)
        sc.ncbi = {}
        sc.error = "old error"
        write_sidecar(sidecar_path(paths, "SRR1"), sc)
        rc = StoreVerifyCommand().execute(_verify_args(data_root=str(paths.root), spots=True, fix_state=True))
        assert rc == 0
        fixed = read_sidecar(sidecar_path(paths, "SRR1"))
        assert fixed.state == "complete" and fixed.error is None
        assert fixed.completeness["verdict"] == "unverified"

    def test_rescan_rebuilds_file_list_from_disk(self, tmp_path, monkeypatch):
        paths = init_store(tmp_path / "store")
        monkeypatch.chdir(tmp_path)
        acc_dir = paths.sra / "SRR1"
        acc_dir.mkdir(parents=True)
        _write_fastq_gz(acc_dir / "SRR1_1.fastq.gz")
        sc = _sidecar_matching_disk(acc_dir, "SRR1", state="failed", reads=1)
        sc.files.insert(0, {"name": "._SRR1_1.fastq.gz", "bytes": 4096, "md5": "x", "reads": None})
        write_sidecar(sidecar_path(paths, "SRR1"), sc)
        rc = StoreVerifyCommand().execute(_verify_args(data_root=str(paths.root), rescan=True, fix_state=True))
        assert rc == 0
        fixed = read_sidecar(sidecar_path(paths, "SRR1"))
        assert [f["name"] for f in fixed.files] == ["SRR1_1.fastq.gz"]
        assert fixed.state == "complete"
```

Add `rescan=False` to `_verify_args` (line 100-110 area).

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cli_store.py::TestStoreVerifyCommand -k "store_metadata_xml or promotes_readable or rescan" -v`
Expected: FAIL (state stays failed; unknown attribute `rescan`).

- [ ] **Step 3: Implement**

In `configure_parser` add `--rescan` (`store_true`, help "Rebuild each dataset's file list from the files on disk before checking (use after files were added or removed by hand)").

In `_verify_one` (680-731), when `args.spots` and `sidecar.ncbi.get("spots")` is falsy, try in order `paths.metadata / f"{acc}_metadata.xml"` and `Path(registry.paths.metadata or "metadata") / f"{acc}_metadata.xml"` (read how the command resolves the project registry; if none, skip the second) with `ncbi_from_metadata_xml(path)`; when it yields spots, use them for the verdict and store `result["ncbi_found"] = ncbi`.

When `args.rescan`, before verifying, rebuild:

```python
            rebuilt = build_sidecar(acc, sra_dir(paths, acc), sidecar.ncbi, sidecar.tool_version, sidecar.compression, tool=sidecar.tool, downloaded=sidecar.downloaded)
            rebuilt.stats, rebuilt.stats_computed = sidecar.stats, sidecar.stats_computed
            sidecar = rebuilt
            result["rescanned"] = True
```

and treat `sidecar.files` as the new truth (bytes/md5 are then trivially consistent; the spots check still runs).

In `_fix_state`, replace the early return:

```python
        spots_verdict = result.get("spots_verdict")
        if result.get("ncbi_found") and not sidecar.ncbi.get("spots"):
            sidecar.ncbi = dict(result["ncbi_found"])
        state_for_verdict = {"complete": "complete", "truncated": "partial"}
        if spots_verdict in state_for_verdict:
            new_state = state_for_verdict[spots_verdict]
            completeness = {"method": "spots", "ratio": result.get("spots_ratio"), "verdict": spots_verdict}
        elif spots_verdict == "corrupt":
            return
        else:
            # No spot count anywhere, but every byte and checksum matched: the files are what was
            # written, so the dataset is complete with an unverified read count.
            new_state = "complete"
            completeness = {"method": "unverified", "ratio": None, "verdict": "unverified"}
        changed = result.get("rescanned") or new_state != sidecar.state or sidecar.completeness != completeness or sidecar.error
        if not changed:
            return
        sidecar.state = new_state
        sidecar.completeness = completeness
        sidecar.error = None
        write_sidecar(sidecar_path(paths, result["accession"]), sidecar)
        with catalog_write(paths) as catalog:
            catalog.upsert_dataset(sidecar)
        result["state"] = new_state
```

Also print, in the verify table, the source of the spot count when it came from a metadata file (`verdict` column unchanged; add "spots: metadata" to a `note` column only if the table code makes that cheap; otherwise a log line).

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_cli_store.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/cli/commands/store.py tests/test_cli_store.py
git commit -m "fix: store_verify reads spots from metadata, promotes readable datasets and can rescan file lists (S5-5, S5-9, S5-13)"
```

---

### Task 9: sra_info reports run accessions, sizes and dates from the RUN element

**Files:**
- Modify: `metaquest/data/sra_metadata.py:199-271`
- Test: `tests/test_sra_metadata_extended.py`

**Interfaces:**
- Produces: `_parse_sra_xml` returns one `SRADatasetInfo` per RUN (keyed by the RUN accession); `SRADatasetInfo.size_mb` from `RUN/@size` bytes, `spots` from `RUN/@total_spots`, `bases` from `RUN/@total_bases`, `release_date` from `RUN/@published`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sra_metadata_extended.py`:

```python
REAL_EFETCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<EXPERIMENT_PACKAGE_SET><EXPERIMENT_PACKAGE>
<EXPERIMENT accession="SRX100" alias="e"><TITLE>gut sample</TITLE>
<DESIGN><LIBRARY_DESCRIPTOR><LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY><LIBRARY_SOURCE>METAGENOMIC</LIBRARY_SOURCE>
<LIBRARY_SELECTION>RANDOM</LIBRARY_SELECTION><LIBRARY_LAYOUT><PAIRED/></LIBRARY_LAYOUT></LIBRARY_DESCRIPTOR></DESIGN>
<PLATFORM><ILLUMINA><INSTRUMENT_MODEL>Illumina NovaSeq 6000</INSTRUMENT_MODEL></ILLUMINA></PLATFORM></EXPERIMENT>
<SUBMISSION accession="SRA100" received="2023-03-01"/>
<STUDY accession="SRP1"><IDENTIFIERS><EXTERNAL_ID namespace="BioProject">PRJNA1</EXTERNAL_ID></IDENTIFIERS></STUDY>
<SAMPLE accession="SRS1"><SAMPLE_NAME><SCIENTIFIC_NAME>gut metagenome</SCIENTIFIC_NAME></SAMPLE_NAME></SAMPLE>
<RUN_SET><RUN accession="SRR100" total_spots="4866463" total_bases="1459938900" size="482592813" published="2023-03-23"/>
<RUN accession="SRR101" total_spots="10" total_bases="1500" size="1048576" published="2023-03-24"/></RUN_SET>
</EXPERIMENT_PACKAGE></EXPERIMENT_PACKAGE_SET>"""


def test_parse_real_efetch_shape_reports_runs():
    client = SRAMetadataClient(email="a@b.c")
    results = client._parse_sra_xml(REAL_EFETCH_XML)
    assert set(results) == {"SRR100", "SRR101"}
    run = results["SRR100"]
    assert run.spots == 4866463 and run.bases == 1459938900
    assert abs(run.size_mb - 482592813 / (1024 * 1024)) < 0.01
    assert run.release_date == "2023-03-23"
    assert run.strategy == "WGS" and run.layout == "PAIRED" and run.organism == "gut metagenome"
    assert run.bioproject == "PRJNA1"
    assert abs(run.avg_length - 300.0) < 0.01
```

(Use the client class name and constructor as `tests/test_sra_metadata_extended.py:189` does.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sra_metadata_extended.py::test_parse_real_efetch_shape_reports_runs -v`
Expected: FAIL (`set(results) == {"SRX100"}`).

- [ ] **Step 3: Implement**

Replace `_aggregate_runs` and the run handling in `_extract_dataset_info`:

```python
    @staticmethod
    def _run_numbers(run) -> Tuple[int, int, float, str]:
        """spots, bases, size in MB and published date of one ``<RUN>`` element."""

        def _int(name: str) -> int:
            value = run.get(name)
            return int(value) if value and value.isdigit() else 0

        size_bytes = _int("size")
        return _int("total_spots"), _int("total_bases"), size_bytes / (1024 * 1024), run.get("published", "") or ""
```

`_extract_dataset_info(self, package) -> List[SRADatasetInfo]`: keep the experiment-level fields, then:

```python
            runs = package.findall(".//RUN_SET/RUN")
            infos = []
            for run in runs or [None]:
                if run is None:
                    accession, spots, bases, size_mb, published = experiment.get("accession", ""), 0, 0, 0.0, ""
                else:
                    accession = run.get("accession", "") or experiment.get("accession", "")
                    spots, bases, size_mb, published = self._run_numbers(run)
                infos.append(SRADatasetInfo(accession=accession, ..., spots=spots, bases=bases,
                                            avg_length=bases / spots if spots else 0.0, size_mb=size_mb,
                                            release_date=published or submission_received, ...))
            return infos
```

Fix element truthiness: use `is not None` for `library_descriptor`, `layout_elem`, `sample`, `study`, `submission` (`submission_received = submission.get("received", "") if submission is not None else ""`). Update `_parse_sra_xml` (line ~178) to extend `results` with every info: `for info in infos: results[info.accession] = info`. Update the existing `MOCK_SRA_XML` tests: they may now key by the RUN accession if the fixture has one; adjust assertions.

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_sra_metadata_extended.py tests/test_sra_metadata_client.py tests/test_cli_commands.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/data/sra_metadata.py tests/test_sra_metadata_extended.py
git commit -m "fix: sra_info reports run accessions, sizes, bases and dates from the RUN element (S3-2)"
```

---

### Task 10: parse_metadata fills the library strategy, source and selection

**Files:**
- Modify: `metaquest/data/metadata.py:499-502`
- Test: `tests/test_data_metadata.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_metadata.py` `TestParseMetadataXml`:

```python
    def test_library_fields_under_design(self, tmp_path):
        xml = (
            '<EXPERIMENT_PACKAGE_SET><EXPERIMENT_PACKAGE><EXPERIMENT accession="SRX1"><DESIGN>'
            "<LIBRARY_DESCRIPTOR><LIBRARY_NAME>lib</LIBRARY_NAME><LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY>"
            "<LIBRARY_SOURCE>METAGENOMIC</LIBRARY_SOURCE><LIBRARY_SELECTION>RANDOM</LIBRARY_SELECTION>"
            "<LIBRARY_LAYOUT><PAIRED/></LIBRARY_LAYOUT></LIBRARY_DESCRIPTOR></DESIGN>"
            "<PLATFORM><ILLUMINA/></PLATFORM></EXPERIMENT>"
            '<RUN_SET><RUN accession="SRR1" total_spots="5"/></RUN_SET></EXPERIMENT_PACKAGE></EXPERIMENT_PACKAGE_SET>'
        )
        path = tmp_path / "SRR1_metadata.xml"
        path.write_text(xml)
        fields = parse_metadata_xml(path)
        assert fields["Experiment_Library_Strategy"] == "WGS"
        assert fields["Experiment_Library_Source"] == "METAGENOMIC"
        assert fields["Experiment_Library_Selection"] == "RANDOM"
```

(Use the function the class already tests; `parse_metadata_xml` is at `metadata.py:573`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data_metadata.py -k library_fields_under_design -v`
Expected: FAIL (None).

- [ ] **Step 3: Implement**

`metaquest/data/metadata.py:499-502`:

```python
        experiment_library_name = tree.findtext(".//EXPERIMENT//LIBRARY_DESCRIPTOR/LIBRARY_NAME")
        experiment_library_strategy = tree.findtext(".//EXPERIMENT//LIBRARY_DESCRIPTOR/LIBRARY_STRATEGY")
        experiment_library_source = tree.findtext(".//EXPERIMENT//LIBRARY_DESCRIPTOR/LIBRARY_SOURCE")
        experiment_library_selection = tree.findtext(".//EXPERIMENT//LIBRARY_DESCRIPTOR/LIBRARY_SELECTION")
```

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_data_metadata.py -q && make check`
Expected: PASS (the old fixtures with LIBRARY_DESCRIPTOR directly under EXPERIMENT still match `//`).

- [ ] **Step 5: Commit**

```bash
git add metaquest/data/metadata.py tests/test_data_metadata.py
git commit -m "fix: read library strategy, source and selection under EXPERIMENT/DESIGN (S3-3)"
```

---

### Task 11: status --next never suggests a list with excluded runs; plot_containment always produces a file

**Files:**
- Modify: `metaquest/cli/commands/status.py:223-240`
- Modify: `metaquest/cli/commands/containment.py:109-113`, `metaquest/visualization/plots.py:137-143`
- Test: `tests/test_cli_status.py`, `tests/test_visualization_plots.py`, `tests/test_cli_commands.py`

**Interfaces:**
- Produces: `_download_next_steps` groups by the latest selection output whose criteria have `skip_excluded` true; for accessions whose only selection output was written with `skip_excluded` false it suggests `metaquest select_datasets ... --skip-excluded` first. `plot_containment --save-format` defaults to `"png"`; output name `f"{Path(file_path).stem}_{plot_type}_{column}.{save_format}"`; `_save_plot_if_needed` returns the output path and the command logs it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_status.py` (pattern from the `--next` tests at 243-308):

```python
def test_next_does_not_suggest_a_no_skip_excluded_list(tmp_path, monkeypatch):
    root = _project_tree(tmp_path)
    monkeypatch.chdir(root)
    StatusCommand().execute(_status_args(root, init=True))
    from metaquest.data.registry import load_registry, registry_transaction, record_selection, record_exclusion
    with registry_transaction(str(root / "metaquest_registry.json")) as reg:
        record_exclusion(reg, ["SRR2"], "isolate")
        record_selection(reg, ["SRR1", "SRR2"], {"skip_excluded": False, "column": "GCF_A", "threshold": 0.5}, "sel_noskip.txt")
    steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
    commands = [s["command"] for s in steps]
    assert not any("sel_noskip.txt" in c for c in commands)
    assert any("select_datasets" in c and "--skip-excluded" in c for c in commands)
```

(Use the real names of the exclusion helper in `data/registry.py`; grep `def record_exclusion` or the blacklist command's helper.)

Replace `test_save_plot_if_needed_no_format` in `tests/test_visualization_plots.py:268` with:

```python
    def test_save_plot_default_png_and_stem_name(self, tmp_path):
        import matplotlib.pyplot as plt
        plt.figure()
        out = _save_plot_if_needed(tmp_path / "parsed_containment.txt", "rank", "max_containment", None)
        assert out == tmp_path / "parsed_containment_rank_max_containment.png"
        assert out.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cli_status.py -k no_skip_excluded_list tests/test_visualization_plots.py -k default_png -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`status.py` `_download_next_steps`:

```python
        by_output: Dict[str, List[str]] = {}
        needs_reselect: List[str] = []
        for acc in to_download:
            selection = registry.datasets[acc].get("selection", {})
            criteria = selection.get("criteria") or {}
            if criteria.get("skip_excluded") is False:
                needs_reselect.append(acc)
                continue
            output = selection.get("output") or "accessions.txt"
            by_output.setdefault(output, []).append(acc)
        steps = [
            {"command": f"metaquest download_sra --accessions-file {output}", "accessions": accs}
            for output, accs in by_output.items()
        ]
        if needs_reselect:
            steps.append(
                {
                    "command": "metaquest select_datasets ... --skip-excluded  (the last selection kept excluded runs)",
                    "accessions": needs_reselect,
                }
            )
        return steps
```

`plots.py`:

```python
def _save_plot_if_needed(file_path, plot_type, column, save_format) -> Path:
    fmt = save_format or "png"
    output_file = Path(file_path).parent / f"{Path(file_path).stem}_{plot_type}_{column}.{fmt}"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")
    return output_file
```

`containment.py`: `--save-format` default `"png"`, help "Image format of the saved plot (default: png)"; log the returned path at INFO in `execute` and add the "Next:" hint line used by the other containment commands.

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_cli_status.py tests/test_visualization_plots.py tests/test_cli_commands.py -q && make check`
Expected: PASS (update any test asserting that no file is written without a format).

- [ ] **Step 5: Commit**

```bash
git add metaquest/cli/commands/status.py metaquest/cli/commands/containment.py metaquest/visualization/plots.py tests/
git commit -m "fix: status --next skips no-skip-excluded lists; plot_containment saves png by default with a stem name (S4-2, S2-5, S2-7)"
```

---

### Task 12: sra_compare serialises numpy values; --accessions-file optional; dashboard reuses profiles

**Files:**
- Modify: `metaquest/sra/analytics.py:999-1010`, `metaquest/cli/commands/sra_intelligent.py:77-81, 153-162, 399-410, 476-481, 598-615`
- Test: `tests/test_sra_analytics.py`, `tests/test_cli_sra_intelligent.py`

**Interfaces:**
- Produces: `metaquest/sra/analytics.py::json_safe(value)` (recursively converts numpy scalars to Python, NaN/inf to None, sets to lists); `_save_comparison_results` uses `json.dump(json_safe(payload), ...)`; `--accessions-file` is optional for `sra_profile_quality` (required unless `--accession`) and `sra_dashboard` (required unless `--quality-profiles` names a directory with profiles); comparative dashboards receive `profiles=profiles or None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sra_analytics.py`:

```python
def test_statistical_tests_are_json_serialisable():
    import json
    import numpy as np
    from metaquest.sra.analytics import json_safe
    payload = {"significant": np.bool_(True), "p": np.float64(0.01), "nan": float("nan"), "s": {1, 2}}
    text = json.dumps(json_safe(payload))
    assert json.loads(text) == {"significant": True, "p": 0.01, "nan": None, "s": [1, 2]}


def test_compare_datasets_result_dumps(tmp_path):
    from metaquest.sra.analytics import SRADatasetAnalyzer
    from tests.test_cli_sra_intelligent import make_profile
    analyzer = SRADatasetAnalyzer(tmp_path)
    groups = {"a": ["A1", "A2", "A3"], "b": ["B1", "B2"]}
    profiles = {acc: make_profile(acc, gc_content=0.40 + i * 0.01) for i, acc in enumerate(groups["a"] + groups["b"])}
    with patch.object(analyzer, "profile_dataset_quality", side_effect=lambda acc, **kw: profiles[acc]):
        result = analyzer.compare_datasets(groups, statistical_tests=True)
    import json
    json.dumps(json_safe(result["statistical_tests"]))
```

(Adapt `compare_datasets`'s real name and signature from `analytics.py` around line 600-700, and `make_profile`'s keyword names from `test_cli_sra_intelligent.py:30`.)

Append to `tests/test_cli_sra_intelligent.py`:

```python
def test_profile_quality_accepts_single_accession_without_file(tmp_path):
    parser = argparse.ArgumentParser()
    SRAQualityProfileCommand().configure_parser(parser)
    args = parser.parse_args(["--accession", "SRR1"])
    assert args.accession == "SRR1" and args.accessions_file is None


def test_dashboard_accepts_profiles_dir_without_file(tmp_path):
    parser = argparse.ArgumentParser()
    SRAInteractiveDashboardCommand().configure_parser(parser)
    args = parser.parse_args(["--quality-profiles", str(tmp_path)])
    assert args.accessions_file is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sra_analytics.py -k "json_serialisable or result_dumps" tests/test_cli_sra_intelligent.py -k "without_file" -v`
Expected: FAIL (`ImportError json_safe`; `SystemExit: 2`).

- [ ] **Step 3: Implement**

`analytics.py`:

```python
def json_safe(value: Any) -> Any:
    """Return ``value`` with numpy scalars, sets and non-finite floats replaced by JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value
```

Line 1009: `tests[col]["significant"] = bool(p_value < 0.05)`.

`sra_intelligent.py`: `_save_comparison_results` dumps `json_safe(payload)`. `--accessions-file` `required=False` in both commands; in `_resolve_accessions` raise a `ValidationError("Give --accessions-file or --accession")` when both are missing (and for the dashboard, when neither a file nor a profiles directory with profiles is given, after loading profiles use `sorted(profiles)` as the accession list). At 476-481 pass `profiles=profiles or None` to `create_comparative_analysis`. When `--quality-profiles` is given but is not a directory, log a warning naming the path.

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_sra_analytics.py tests/test_cli_sra_intelligent.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/sra/analytics.py metaquest/cli/commands/sra_intelligent.py tests/test_sra_analytics.py tests/test_cli_sra_intelligent.py
git commit -m "fix: sra_compare writes JSON-safe results; single accession and profiles dir work without --accessions-file (S6-2, S6-3, S6-8)"
```

---

### Task 13: Extraction dry run summarises, megahit errors surface, tmp dir and parameter changes are explicit

**Files:**
- Modify: `metaquest/data/read_extraction.py:231-249, 453-460, 504-527, 561-684, 709-792`
- Modify: `metaquest/cli/commands/read_extraction.py:335-357, 359-420, 437-495`
- Modify: `metaquest/cli/commands/status.py:292-295`
- Test: `tests/test_read_extraction.py`, `tests/test_cli_read_extraction.py`, `tests/test_cli_status.py`

**Interfaces:**
- Produces: `extract_target_reads(..., available: Optional[Set[str]] = None)`: when given, samples not in it are counted and reported once ("N of the M selected sample(s) have no FASTQ under <folder>; skipped") instead of one warning each; `assemble_extracted_reads(..., tmp_dir: Optional[Path] = None)` passes `--tmp-dir`; a megahit failure raises `ProcessingError("megahit failed (exit N): <last 5 stderr lines>")`; `_record_matches` compares `min_mapq` too and the redo logs "redoing X: <reason>"; the MAPQ clause is only printed when `min_mapq > 0`; `_export_tsv` writes with `index=False` for the extractions frame and logs both paths.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_read_extraction.py`:

```python
def test_dry_run_summarises_missing_fastq(tmp_path, caplog):
    tmp = _make_tree(tmp_path)
    (tmp / "fastq" / "SRR2").rename(tmp / "fastq" / "SRR2_gone")
    with caplog.at_level(logging.INFO):
        extract_target_reads(str(tmp / "parsed.tsv"), "GCF_1", str(tmp / "GCF_1.fna"), str(tmp / "fastq"), str(tmp / "targeted"),
                             threshold=0.1, dry_run=True, available={"SRR1", "SRR3"})
    assert caplog.text.count("No FASTQ files found for") == 0
    assert "1 of the 3 selected sample(s) have no FASTQ" in caplog.text


def test_megahit_failure_reports_stderr(tmp_path):
    import subprocess
    reads = [tmp_path / "r.fastq.gz"]
    reads[0].write_bytes(b"x")
    err = subprocess.CalledProcessError(1, ["megahit"], output="", stderr="line1\nOSError: [Errno 45] Operation not supported\n")
    with patch("metaquest.data.read_extraction.SecureSubprocess.run_secure", side_effect=err):
        with pytest.raises(ProcessingError) as excinfo:
            assemble_extracted_reads(reads, tmp_path / "asm", threads=1, tmp_dir=tmp_path / "scratch")
    assert "Operation not supported" in str(excinfo.value)


def test_megahit_receives_tmp_dir(tmp_path):
    state = {"calls": []}
    reads = [tmp_path / "r.fastq.gz"]
    reads[0].write_bytes(b"x")
    with patch("metaquest.data.read_extraction.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
        assemble_extracted_reads(reads, tmp_path / "asm", threads=1, tmp_dir=tmp_path / "scratch")
    megahit_call = next(c for c in state["calls"] if c[0] == "megahit")
    assert "--tmp-dir" in megahit_call[1]


def test_record_matches_compares_min_mapq(tmp_path):
    record = {"genome_fasta": str(tmp_path / "g.fna"), "preset": "sr", "threshold": 0.1, "min_mapq": 20, "mapped_reads": 0}
    (tmp_path / "g.fna").write_text(">a\nA\n")
    assert _record_matches(record, tmp_path / "g.fna", "sr", 0.1, min_mapq=20)
    assert not _record_matches(record, tmp_path / "g.fna", "sr", 0.1, min_mapq=0)
```

(Adapt the `extract_target_reads` positional signature to the real one at `read_extraction.py:561`, and how `_fake_tools` records calls, `tests/helpers_extraction.py:39`.)

Append to `tests/test_cli_status.py`, extending `test_export_tsv` (399-402):

```python
    ext = (root / "out_extractions.tsv").read_text().splitlines()[0]
    assert ext.startswith("accession\t")
    assert "out_datasets.tsv" in caplog.text and "out_extractions.tsv" in caplog.text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_read_extraction.py -k "summarises_missing or megahit or min_mapq" tests/test_cli_status.py -k export_tsv -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`read_extraction.py`:

- `_record_matches(record, genome_path, preset, threshold, min_mapq: int = 0)`: add `recorded_mapq = record.get("min_mapq"); if recorded_mapq is not None and int(recorded_mapq) != int(min_mapq): return False`. Pass `min_mapq` at the call in `_extract_one_sample`; when `record` exists and `not done and not force`, log `logger.info("%s: redoing extraction, recorded parameters differ from this call", accession)`.
- `extract_target_reads(..., available: Optional[Set[str]] = None)`: after `samples = select_samples_for_genome(...)`, when `available is not None`: `missing = [a for a in samples if a not in available]; samples = [a for a in samples if a in available]; if missing: logger.info("%d of the %d selected sample(s) have no FASTQ under %s; skipped", len(missing), len(missing) + len(samples), fastq_root)`.
- Line 453-460: build the clause `f" and MAPQ below {min_mapq}" if min_mapq > 0 else ""` and log `"%s: kept %d of %d mapped records (secondary/supplementary%s removed: %d)"`.
- `assemble_extracted_reads(..., tmp_dir: Optional[Path] = None)`: `if tmp_dir is not None: tmp_dir.mkdir(parents=True, exist_ok=True); SecureSubprocess.add_allowed_root(tmp_dir); args += ["--tmp-dir", str(tmp_dir)]`; wrap the call:

```python
    try:
        SecureSubprocess.run_secure("megahit", args)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stderr or "").strip().splitlines()[-5:])
        raise ProcessingError(f"megahit failed (exit {exc.returncode}) for {out_dir}:\n{tail}") from exc
```

`cli/commands/read_extraction.py`: compute `available = {acc for acc, rec in registry.datasets.items() if rec.get("download", {}).get("state") == "downloaded"} | set(scan_downloads(args.fastq_folder))` (use `metaquest.data.registry.scan_downloads`, line 719) and pass it; pass `tmp_dir=Path(args.temp_folder) if args.temp_folder else Path(args.output_folder) / ".megahit-tmp"` to `assemble_extracted_reads` in `_assemble` and remove that default folder afterwards with `shutil.rmtree(..., ignore_errors=True)`.

`status.py` `_export_tsv`:

```python
        datasets_path = f"{prefix}_datasets.tsv"
        extractions_path = f"{prefix}_extractions.tsv"
        write_csv(datasets_df, datasets_path, sep="\t")
        write_csv(extractions_df, extractions_path, sep="\t", index=False)
        self.logger.info("Wrote %s and %s", datasets_path, extractions_path)
```

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_read_extraction.py tests/test_cli_read_extraction.py tests/test_cli_status.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metaquest/data/read_extraction.py metaquest/cli/commands/read_extraction.py metaquest/cli/commands/status.py tests/
git commit -m "fix: extraction dry run summarises, megahit errors and tmp dir surface, parameter changes are logged (S7-1, S7-3, S7-5, S7-6, S7-8, S8-1)"
```

---

### Task 14: Download tests independent of tools on PATH

**Files:**
- Modify: `tests/test_data_sra.py:766-790, 791-840, 841-870`

- [ ] **Step 1: Reproduce the failure**

Run: `PATH=/Users/andreassjodin/miniforge3/envs/sra-tools/bin:$PATH pytest tests/test_data_sra.py -k "TestDownloadAccession" -q`
Expected: 3 failed (`Expected 'run_secure' to have been called once. Called 2 times.`). On a machine without sra-tools, simulate with `patch("metaquest.data.sra.shutil.which", side_effect=lambda t: f"/usr/bin/{t}")` in one of those tests to see the same failure.

- [ ] **Step 2: Fix the three tests**

Wrap each of `test_download_accession_success`, `test_download_accession_command_passes_validation` and `test_download_accession_redownload_truncated_forces_fresh_download` in `with patch("metaquest.data.sra.shutil.which", return_value=None):` (or add it as a decorator with the corresponding mock argument), matching the style of tests 987-1066 in the same file.

- [ ] **Step 3: Verify under both PATHs**

Run: `pytest tests/test_data_sra.py -q && PATH=/Users/andreassjodin/miniforge3/envs/sra-tools/bin:$PATH pytest tests/test_data_sra.py -q`
Expected: PASS both times.

- [ ] **Step 4: Commit**

```bash
git add tests/test_data_sra.py
git commit -m "test: download tests no longer depend on sra-tools being on PATH (S0-1)"
```

---

### Task 15: Documentation pass

**Files:**
- Modify: `README.md`, `docs/pipeline_overview.md`, `CLAUDE.md` and `AGENTS.md` store section (AGENTS.md is untracked: edit, never `git add`)

- [ ] **Step 1: Apply Appendix A of the spec**

Work through items (1)-(4) of `docs/superpowers/specs/2026-09-23-lactobacillus-e2e-audit.md` Appendix A, in order. For each item change the quoted sentence; the new facts are the ones this plan implements: `plot_containment` saves png by default; `--temp-folder` documented for `download_sra` and `extract_target_reads`, with the store default; `download_metadata` must run before `download_sra` for the spot-count verdict (and `store_verify --spots` now also reads the store's metadata folder); `sra_info` sizes; `store_adopt --copy` semantics; `store_reindex` replays the journal and `store_gc` refuses without projects; the ExFAT note (AppleDouble files ignored; assembly needs a filesystem with FIFOs, use `--temp-folder` on a local disk).

- [ ] **Step 2: Check for non-ASCII**

Run: `python3 -c "import sys; [print(f, sorted({c for c in open(f).read() if ord(c)>127})) for f in ['README.md','docs/pipeline_overview.md']]"`
Expected: two empty lists.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/pipeline_overview.md CLAUDE.md
git commit -m "docs: correct the pipeline docs after the Lactobacillus audit"
```

---

## Verification after the last task

- `make check` and `make test` pass; `make test-network` passes with the sra-tools bin on PATH.
- On the reference data in `/Volumes/sekvens2/metaquest-e2e` (ExFAT): in project `crispatus`, `metaquest store_verify --data-root ../store --spots --rescan --fix-state` reports every dataset `complete`; `metaquest store_reindex` followed by `store_status` shows the same project count as before; `store_gc --dry-run` lists no dataset as unused while the three projects link them; `download_sra --dry-run` on `accessions.txt` says 7 already downloaded; `genome_prepare --accession-file genome_accessions.txt` writes two manifest rows; `plot_containment --file-path parsed_containment.txt --plot-type rank` writes `parsed_containment_rank_max_containment.png`; `sra_info --accessions-file accessions.txt --email ...` prints non-zero sizes keyed by SRR; `extract_target_reads ... --dry-run` prints one summary line instead of 16 597 warnings.

## Deferred (Minor or Usability findings not in this plan)

S1-1 (old genus name gives a raw GTDB 400), S1-4 (genome_download writes no registry entry), S2-2 (tie order of the
best hit), S2-4 ("Next:" hints on parse/explore), S3-1 (extract_branchwater_metadata warning), S4-1 (a named target list
instead of "last selection"), S5-4 (status wording for a failed store dataset), S6-1/S6-4/S6-7 (unit labels, sample size
printed as total reads, JSON key names), S6-5 (sra_validate message), S7-2 (dangling links unreported), S7-7 (relink
drops reads_r1), S8-2 (duplicate tool errors: not reproduced in code). Each is a one-line change with a one-test
regression; schedule them after this plan lands.
