# Pipeline Audit and Shared Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every pipeline stage faster, more robust and more informative, and add a shared data store so several organism projects reuse one copy of each downloaded metagenome with central tracking of which project and genome used it.

**Architecture:** Track A hardens downloads and metadata (completeness verdicts, prefetch, compression). Track B adds the `metaquest/store/` package (root discovery, layout, sidecars, SQLite catalogue, symlinked projects, usage rows) and nine flat `store_*` commands. Track C improves screening, selection, extraction, assembly and the analysis commands. Track D documents everything. Commands keep calling the data layer through `SecureSubprocess.run_secure`; every new tool flag is allowlisted in `metaquest/core/constants.py` and faked in tests.

**Tech Stack:** Python 3.12, stdlib (`sqlite3`, `tomllib`, `gzip`, `hashlib`, `json`), pandas, Biopython, requests with `urllib3.util.retry.Retry`; external tools sra-tools (`prefetch`, `fasterq-dump`), `pigz` (optional), minimap2, samtools, megahit, seqkit (optional).

**Spec:** `docs/superpowers/specs/2026-09-06-pipeline-audit-and-store-design.md`

## Global Constraints

- black line length 120; flake8, mypy and the radon ceiling (grade D fails) via `make check`; `make test` green after every task; `make pipeline` after the last task of each bundle.
- CLI flags use dashes; plain, modest language; no Unicode in code, help or docs.
- Unit tests never touch the network or external tools: patch `SecureSubprocess.run_secure` (extend `tests/helpers_extraction.py::_fake_tools` for new tools) or `requests`; never write into the repository root or the user's home or config directory (monkeypatch `HOME`, `XDG_CONFIG_HOME`, `METAQUEST_DATA`; every registry, store and config path lives under `tmp_path`).
- Every new external-tool flag: add it to `ALLOWED_BIOINFORMATICS_TOOLS` in `metaquest/core/constants.py`, to `PATH_VALUE_FLAGS` in `metaquest/utils/security.py` when it takes a path, and a case in `tests/test_security_comprehensive.py`.
- Registry writes only through `registry_transaction` or `save_registry` after load-mutate; catalogue writes only through `catalog.py` under its lock; a catalogue or registry failure never fails the science step (isolate like `_notify_result` in `metaquest/data/sra.py:32-45`).
- A project that never runs `store_init` behaves exactly as today; registry schema 1 files keep loading; existing `status --json` keys are unchanged.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JUCv6TG8f99GTkypHbQzJV`; never `git add` the untracked `AGENTS.md`.

Branches (stacked, each bundle one branch, PR per bundle): `feat/downloads-you-can-trust` (A1), `feat/download-tools` (A2), `feat/store-core` (B1), `feat/store-link` (B2), `feat/store-usage` (B3), `feat/screen-select` (C1), `feat/extract-assemble` (C2), `feat/analysis-cache` (C3), `docs/audit-round` (D). C1 and C2 branch from A1's tip; C3 from B1's tip; the merge order is A1, A2, B1, B2, B3, C1, C2, C3, D.

---

## Bundle A1: downloads you can trust (branch `feat/downloads-you-can-trust` from `main`)

### Task 1: NCBI XML attributes and spots in the registry

**Files:**
- Modify: `metaquest/data/metadata.py:252-256` (`_extract_metadata_fields`), `metaquest/data/registry.py:396-401` (`record_metadata`), `metaquest/cli/commands/metadata.py` (`_record_row`)
- Test: `tests/test_data_metadata.py` (or the existing metadata test module), `tests/test_data_registry.py`, `tests/test_cli_commands.py`

**Interfaces:**
- `_extract_metadata_fields` returns `Run_Total_Spots`, `Run_Total_Bases`, `Run_Size` from the `<RUN>` attributes (`total_spots`, `total_bases`, `size`), `Run_MD5` and `Run_Filename` from the first `<SRAFile>` whose `semantic_name` is `run` (else the first `SRAFile`), `Run_Spot_Length` from `Statistics/Read/@average` summed over reads when present; the old child-element reads stay as fallbacks. Adds `Experiment_Library_Layout` (`PAIRED`/`SINGLE`) and `Platform` (the tag name under `<PLATFORM>`).
- `record_metadata(..., fields)` also keeps `run_total_spots`, `run_total_bases`, `library_layout`, `platform`, `library_strategy` (ints where numeric, else None).

- [ ] **Step 1: Failing tests.** A minimal XML fixture string with `<RUN accession="SRR1" total_spots="47964651" total_bases="14389395300" size="4744553813">`, `<SRAFiles><SRAFile filename="SRR1" md5="abc" semantic_name="run"/></SRAFiles>`, `<PLATFORM><ILLUMINA>...</ILLUMINA></PLATFORM>`, `<LIBRARY_LAYOUT><PAIRED/></LIBRARY_LAYOUT>`; assert the parsed dict holds `"47964651"`, `"abc"`, `"PAIRED"`, `"ILLUMINA"`. Registry: `record_metadata` keeps `run_total_spots == 47964651` as int. CLI: `parse_metadata` on a folder with that XML records the spots.
- [ ] **Step 2: Implement** with `run = tree.find(".//RUN")`, `run.get("total_spots") if run is not None else tree.findtext(".//RUN/Total_spots")` and the same shape for the others; helper `_first_srafile(tree)`.
- [ ] **Step 3: Verify** `pytest tests/test_data_metadata*.py tests/test_data_registry.py tests/test_cli_commands.py -q && make check && make test`.
- [ ] **Step 4: Commit** `fix: read NCBI run size, spots and md5 from the XML attributes`.

### Task 2: completeness verdict for downloads

**Files:**
- Modify: `metaquest/data/registry.py:534-541` (`count_fastq_reads`), `metaquest/data/sra.py:48-56, 128-160, 163-196` (`accession_has_fastq`, `_check_existing_download`, `_handle_download_output`), `metaquest/data/sra.py` (new `verify_download`), `metaquest/cli/commands/sra.py` (`--verify-downloads`, default on; `--redownload-truncated`), `metaquest/cli/commands/status.py` (verdict column)
- Test: `tests/test_data_registry.py`, `tests/test_data_sra.py`, `tests/test_cli_commands.py`, `tests/test_cli_status.py`

**Interfaces:**
- `count_fastq_reads(path) -> int`: binary chunked count, `total_newlines // 4`, gzip via `gzip.open(path, "rb")`, 1 MiB blocks.
- `FASTQ_GLOBS = ("*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz")` in `metaquest/core/constants.py`; `fastq_files(acc_dir) -> List[Path]` in `sra.py` returns non-empty files matching them, sorted.
- `accession_has_fastq(acc_dir) -> bool`: `acc_dir` is a directory (a symlink to one counts), `fastq_files` is non-empty; when `acc_dir / "<acc>.json"` exists (a store sidecar, Task 8) and its `state` is `partial`, `failed` or `downloading`, return False.
- `verify_download(accession, acc_dir, expected_spots: Optional[int], expected_bytes: Optional[int]) -> Dict[str, Any]` returns `{"reads_r1": n, "expected_spots": s, "ratio": r, "verdict": "complete"|"truncated"|"unverified", "bytes_total": b}`; `complete` when `ratio >= 0.99`, `truncated` when below, `unverified` when spots unknown.
- `download_accession(..., expected_spots=None)` runs `verify_download` inside `_handle_download_output` and returns `(True, message)` with the verdict in the message (`"Downloaded 2 files, complete (300000 of 300000 spots)"`); `download_sra(..., expected_spots: Optional[Dict[str, int]] = None)` passes per-accession spots; the CLI reads spots from `registry.datasets[acc]["metadata"]["run_total_spots"]` when present.
- `record_download` stores `download["complete"] = verdict dict` when supplied (`complete: Optional[Dict] = None` keyword); the skip path (`_check_existing_download`) treats a registry verdict `truncated` as not downloaded when `redownload_truncated` is set, and always unlinks a dangling symlink instead of `rmdir()`.

- [ ] **Step 1: Failing tests.** Chunked count equals the old count on plain and gz files (reuse `_fastq` helper); `accession_has_fastq` is False for a zero-byte file and True for `x.fq.gz`; `verify_download` verdicts for 300000/48000000 (`truncated`), 999/1000 (`complete`), unknown (`unverified`); `download_sra` with `download_accession` patched to create files records the verdict through `on_result`'s message and the CLI test asserts `download.complete.verdict` in the registry; `status --json` shows `stages.downloaded` unchanged and a new `report["downloads"]["truncated"]` list; `_check_existing_download` on a dangling symlink returns False without raising.
- [ ] **Step 2: Implement.** Keep `accession_has_fastq` the single truth; `_handle_download_output` calls `verify_download` and logs a warning for `truncated`.
- [ ] **Step 3: Verify** the four test files, `make check`, `make test`.
- [ ] **Step 4: Commit** `feat: verify download completeness against NCBI spots and reject empty FASTQ`.

### Task 3: resume, retry and error classes

**Files:**
- Modify: `metaquest/data/sra.py:228-277, 360-427` (`download_accession`, `_retry_failed_downloads`), `metaquest/cli/commands/sra.py` (default workers)
- Test: `tests/test_data_sra.py`, `tests/test_cli_commands.py`

**Interfaces:**
- On failure `download_accession` keeps `<acc>_temp` (no `_safe_rmtree`) and returns `(False, "<class>: <message>")` where class is `network` (stderr matches `timeout|connection|resolve|network|curl`), `disk-full` (`No space left|ENOSPC`), `not-found` (`not found|invalid accession|403|404`), else `unknown`; `classify_download_error(text) -> str` is a pure helper.
- `_retry_failed_downloads` retries only `network` and `unknown`, with `force=False`, sleeping `2 ** attempt` seconds (patchable `time.sleep`), and stops the whole run with `DataAccessError` on `disk-full`.
- `download_sra` default `max_workers=None` resolved as `min(MAX_CONCURRENT_DOWNLOADS, max(1, (os.cpu_count() or 4) // num_threads), 4)` in the CLI when `--max-workers` is not given; log a warning when `workers * threads > cpu_count`.

- [ ] **Step 1: Failing tests.** `classify_download_error` table test; a failed download leaves `<acc>_temp` in place; retry skips `not-found`, retries `network` with `force=False` (assert the call kwargs) and sleeps; `disk-full` raises `DataAccessError`; CLI default worker count with `os.cpu_count` patched to 8 and threads 4 gives 2.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: keep partial downloads for resume, classify errors, size the worker pool`. Open PR A1.

---

## Bundle A2: download tools (branch `feat/download-tools` from A1)

### Task 4: per-tool boolean flags and `.sra` positional validation

**Files:**
- Modify: `metaquest/utils/security.py:26-35, 176-200`, `metaquest/core/constants.py:61-72`
- Test: `tests/test_security_comprehensive.py`

**Interfaces:**
- `BOOLEAN_FLAGS: Dict[str, FrozenSet[str]] = {"fasterq-dump": {...existing..., "--split-3"}, "prefetch": {"--progress", "--resume", "--version"}, "pigz": {"-f", "-k", "--version"}, "minimap2": {"-a", "--version"}, "samtools": {"-b", "-c", "--version"}, "megahit": {"--no-mercy", "--version"}}` replaces `FASTERQ_DUMP_BOOLEAN_FLAGS`; `takes_value = arg not in BOOLEAN_FLAGS.get(executable, frozenset())`.
- Positional validation: for `fasterq-dump` and `prefetch`, a positional that matches `SRA_ACCESSION_PATTERN` passes `validate_accession_for_subprocess`; one ending in `.sra` passes `validate_path`; anything else raises `SecurityError`.
- Allowlist: `prefetch` with `{"-O", "--max-size", "--progress", "--resume", "--version"}`; fasterq-dump gains `"--split-3"`; `pigz` with `{"-p", "-f", "-k", "--version"}`. `-O` is already in `PATH_VALUE_FLAGS`.

- [ ] **Step 1: Failing tests.** `prefetch --progress SRR1` keeps the accession as a validated positional; `fasterq-dump --split-3 --threads 4 <tmp>/SRR1/SRR1.sra` validates the path; `pigz -p 4 -f <file>` allowed; `minimap2 -a -x sr` does not swallow `-x`; an unknown positional for prefetch raises.
- [ ] **Step 2-4: Implement, verify, commit** `refactor: per-tool boolean flags and .sra positionals in the subprocess allowlist`.

### Task 5: prefetch, split-3 and compression in `download_accession`

**Files:**
- Modify: `metaquest/data/sra.py:196-277` (`download_accession`), `163-196` (`_handle_download_output`), `metaquest/cli/commands/sra.py` (`--sra-cache`, `--keep-sra`, `--no-prefetch`, `--compress/--no-compress`, `--compress-level`), `environment.yml` (add `pigz`), `tests/helpers_extraction.py` (fakes for `prefetch`, `pigz`)
- Test: `tests/test_data_sra.py`, `tests/test_cli_commands.py`

**Interfaces:**
- `download_accession(accession, output_folder, num_threads=4, force=False, temp_folder=None, expected_spots=None, sra_cache: Optional[Path] = None, use_prefetch: bool = True, keep_sra: bool = False, compress: bool = True)`.
- Sequence: `prefetch -O <cache> --max-size 100G --progress <acc>` (cache default `<output_folder>/.sra-cache`, registered with `add_allowed_root`), then `fasterq-dump --split-3 --skip-technical --threads N -O <temp> [--temp <t>] <cache>/<acc>/<acc>.sra`; when `use_prefetch` is False or `shutil.which("prefetch")` is None, fall back to today's direct call. After a `complete` or `unverified` verdict delete `<cache>/<acc>` unless `keep_sra`.
- `compress_fastq(path, threads) -> Path`: `pigz -p N -f <path>` when `shutil.which("pigz")`, else Python `gzip.open(..., compresslevel=6)` copy and unlink; called for every file in `_handle_download_output` when `compress`; the verdict is computed before compression (counting plain files is cheaper).
- Single-end runs now produce `<acc>.fastq(.gz)` (from `--split-3`); `fastq_files` and `_find_paired_reads` (`sra.py:684-704`) accept it.

- [ ] **Step 1: Failing tests.** With `run_secure` faked (recording calls, creating `<temp>/<acc>_1.fastq` etc. on the fasterq-dump call and `<cache>/<acc>/<acc>.sra` on prefetch): the call order and args; fallback without prefetch; `.sra` removed after success and kept with `keep_sra`; compression through the Python path produces `.fastq.gz` files whose read count matches; pigz path builds `pigz -p 4 -f <file>`; `accession_has_fastq` true afterwards.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: prefetch then fasterq-dump --split-3, gzip the reads`. Open PR A2.

---

## Bundle B1: store core (branch `feat/store-core` from A2)

### Task 6: store root resolution and layout

**Files:**
- Create: `metaquest/store/__init__.py`, `metaquest/store/resolve.py`, `metaquest/store/layout.py`, `tests/test_store_resolve.py`, `tests/test_store_layout.py`
- Modify: `metaquest/core/constants.py` (`STORE_MARKER = "metaquest_store.json"`, `STORE_ENV = "METAQUEST_DATA"`, `CONFIG_DIRNAME = "metaquest"`, `CONFIG_FILENAME = "config.toml"`)

**Interfaces:**
```python
# resolve.py
def config_path() -> Path            # $XDG_CONFIG_HOME/metaquest/config.toml or ~/.config/metaquest/config.toml
def read_config() -> Dict[str, Any]  # {} when absent; tomllib
def write_config_data_root(root: Path) -> Path   # writes "[store]\ndata_root = \"...\"\n" preserving other keys textually
def resolve_store_root(explicit: Optional[str], registry_root: Optional[str], require_marker: bool = True) -> Optional[Path]
# precedence: explicit, METAQUEST_DATA, registry_root, config; returns None when nothing set; raises DataAccessError
# when the chosen root lacks the marker and require_marker; logs which rule won; registers the root with
# SecureSubprocess.add_allowed_root
# layout.py
@dataclass class StorePaths: root, marker, catalog, catalog_lock, locks, tmp, sra, metadata
def store_paths(root: Path) -> StorePaths
def init_store(root: Path) -> StorePaths      # creates folders and marker {"version": 1, "id": uuid4, "created": iso}
def sra_dir(paths, acc) -> Path; def sidecar_path(paths, acc) -> Path; def lock_path(paths, acc) -> Path
```
- [ ] **Step 1: Failing tests.** Precedence matrix (flag, env via `monkeypatch.setenv`, registry, config via `monkeypatch.setenv("XDG_CONFIG_HOME", tmp)`); unknown root without marker raises naming the rule; `write_config_data_root` round-trips through `read_config`; `init_store` is idempotent and the marker keeps its id; `add_allowed_root` called with the resolved root (assert via `SecureSubprocess.allowed_roots()`).
- [ ] **Step 2-4: Implement, verify, commit** `feat: store root discovery and layout`.

### Task 7: sidecars

**Files:**
- Create: `metaquest/store/sidecar.py`, `tests/test_store_sidecar.py`

**Interfaces:**
```python
SIDECAR_SCHEMA = 1
@dataclass class Sidecar: accession, state, layout, downloaded, tool, tool_version, compression,
    files: List[Dict], reads_per_mate, bases_total, ncbi: Dict, completeness: Dict, stats: Dict, stats_computed
def read_sidecar(path) -> Optional[Sidecar]; def write_sidecar(path, sidecar) -> Path   # atomic, sorted keys
def build_sidecar(accession, acc_dir, ncbi: Dict, tool_version: str, compression: str) -> Sidecar
# stats every file (bytes, md5 via hashlib in 1 MiB chunks, reads via count_fastq_reads), layout from file names,
# completeness via verify_download semantics (reuse metaquest.data.sra.verify_download)
def ncbi_from_metadata_xml(xml_path) -> Dict     # spots, bases, size, files[{name, md5}] via the Task 1 parser
```
- [ ] **Step 1: Failing tests.** Round trip; `build_sidecar` on a two-mate tmp folder gives `PAIRED`, per-file md5 equal to `hashlib.md5` of the file, verdicts complete/partial/unverified; invalid JSON returns None with a warning.
- [ ] **Step 2-4: Implement, verify, commit** `feat: per-dataset sidecars`.

### Task 8: SQLite catalogue

**Files:**
- Create: `metaquest/store/catalog.py`, `tests/test_store_catalog.py`

**Interfaces:**
```python
class Catalog:
    def __init__(self, paths: StorePaths)
    def __enter__/__exit__            # opens sqlite3 with foreign_keys on, tries journal_mode=WAL, else delete
    def migrate(self) -> None         # creates the schema from the spec (store_meta, datasets, files, projects, usage, view unused_datasets)
    def upsert_dataset(self, sidecar: Sidecar) -> None
    def get_dataset(self, accession) -> Optional[Dict]
    def upsert_project(self, project_id, name, path, registry) -> None
    def record_usage(self, accession, project_id, genome_id, stage, detail="") -> None   # first_used kept, last_used updated
    def projects_for(self, accession) -> List[Dict]
    def datasets_for_project(self, project_id) -> List[Dict]
    def unused(self) -> List[str]
    def bytes_by_genome(self) -> Dict[str, int]
    def datasets_for_genome(self, genome_id) -> List[Tuple[str, str]]   # (accession, project_id)
    def reindex(self, sidecars: Iterable[Sidecar]) -> int
def catalog_write(paths) -> ContextManager[Catalog]   # acquires paths.catalog_lock with registry._acquire_lock, yields, commits
```
- [ ] **Step 1: Failing tests.** Schema created and `migrate` idempotent; upsert then get; usage upsert keeps `first_used`; the four queries; `reindex` rebuilds `datasets`/`files` from sidecars and keeps `usage`; WAL fallback when the pragma returns `delete` (monkeypatch `sqlite3.Connection.execute` for that one statement); two writers through `catalog_write` serialise (second waits on the lock; test with a thread and a short `LOCK_WAIT_SECONDS`).
- [ ] **Step 2-4: Implement, verify, commit** `feat: SQLite catalogue with locked writes`.

### Task 9: `store_init`, `store_status`, `store_reindex` and `--data-root` plumbing

**Files:**
- Create: `metaquest/cli/commands/store.py`, `tests/test_cli_store.py`
- Modify: `metaquest/cli/main.py` (register the three, group `Store`), `metaquest/cli/commands/sra.py` and `status.py` (`--data-root`, resolved through `resolve_store_root(args.data_root, registry.store.get("root"))`; without a store, behaviour unchanged), `metaquest/data/registry.py` (schema 2: `Registry.project: Dict`, `Registry.store: Dict`, read with defaults, written; `SCHEMA_VERSION = 2`), `tests/test_cli_main.py` (`expected_commands`), `tests/test_data_registry.py`

**Interfaces:**
- `store_init --data-root PATH [--project-name NAME] [--set-default] [--registry]`: `init_store`, `Catalog.migrate`, writes `store.root` and `project{id (uuid4 unless present), name (default: cwd name), path (absolute), created}` into the registry, `--set-default` writes the config; `.gitignore` guard: when `.git` exists, append `fastq/` if absent and warn if `git ls-files fastq` is non-empty (run via `subprocess.run(["git", "ls-files", "fastq"])` only when `.git` exists; patch in tests).
- `store_status [--data-root] [--json] [--verbose]`: root, dataset counts by state, bytes total, project count, stale projects (registry path missing); `--verbose` lists datasets.
- `store_reindex [--data-root]`: rebuilds from sidecars, prints the count.
- Registry v2 keys default to `{}`; a v1 file loads with `version` 1 and is written as 2 on the next save.

- [ ] **Step 1: Failing tests.** Each command with tmp roots; a v1 registry loads and round-trips with `project`/`store` empty; `download_sra --dry-run --data-root <tmp store>` resolves and logs the root; `status` prints a `Store` block when a root resolves and nothing when it does not.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: store_init, store_status, store_reindex and --data-root plumbing`. Open PR B1.

---

## Bundle B2: link, dedup, registry paths (branch `feat/store-link` from B1)

### Task 10: project-relative registry paths

**Files:**
- Modify: `metaquest/data/registry.py` (`project_root(registry) -> Path` = registry file's parent; `_project_relative(path, root) -> str`; used by `_file_entries`, `record_genome`, `record_extraction`, `record_assembly`, `record_analysis`, `record_metadata`; readers resolve relative entries against the project root: `_record_matches` in `read_extraction.py`, `extraction_record` consumers, `status`)
- Test: `tests/test_data_registry.py`, `tests/test_read_extraction.py`, `tests/test_cli_status.py`

- [ ] **Step 1: Failing tests.** Recording a file under the project root stores a relative path; one outside stores absolute; moving the project directory (rename `tmp_path/a` to `tmp_path/b`) keeps the skip check working when run from the new root; existing absolute entries still resolve.
- [ ] **Step 2-4: Implement, verify, commit** `fix: store registry paths relative to the project root`.

### Task 11: symlinks and the store branch of `download_sra`

**Files:**
- Create: `metaquest/store/link.py` (`link_dataset(project_fastq, acc, paths, mode) -> Path`, `unlink_dataset`, `is_store_link`, `dangling_links(project_fastq) -> List[str]`), `tests/test_store_link.py`
- Modify: `metaquest/data/sra.py` (`download_sra(..., store: Optional[StorePaths] = None, link_mode="auto", accept_partial=False, resume_partial=True)`, `download_accession` writes into `store.tmp / f"{acc}_temp"` and moves to `sra_dir`, takes `lock_path(paths, acc)` with `registry._acquire_lock`, writes the sidecar, upserts the catalogue, links), `metaquest/data/registry.py` (`ReconcileReport.dangling_links`, `reconcile` reports them), `metaquest/cli/commands/sra.py` (`--link-mode`, `--accept-partial`, `--no-resume-partial`), `tests/helpers_extraction.py`
- Test: `tests/test_data_sra.py`, `tests/test_cli_commands.py`, `tests/test_data_registry.py`

**Interfaces:**
- Store branch per accession: sidecar `complete`/`unverified` -> link, return `(True, "linked from store")`, the CLI records `downloaded` with `attempt=False`, `source: "store"`; `partial`/`failed` -> redownload when `resume_partial`, else `(False, "partial in store; rerun with --resume-partial")`; absent -> lock, download, verify with `ncbi_from_metadata_xml(store.metadata / f"{acc}_metadata.xml")` when present, compress, sidecar, catalogue, link.
- `link_dataset` creates a relative symlink when the store root and the project share a parent, else absolute; `mode="copy"` copies.
- `accession_has_fastq` follows symlinks (already) and honours the sidecar state (Task 2).

- [ ] **Step 1: Failing tests.** Link created relative/absolute/copy; `scan_downloads` and `status` see the linked dataset; dangling link reported by `reconcile` and skipped by `accession_has_fastq`; `download_sra` with a fake `download_accession`: complete in store -> no download call and a link; partial -> download; absent -> download into the store tmp, sidecar written, catalogue row present, link created; two `download_sra` calls for the same absent accession from two project folders (sequential in the test) download once.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`).
- [ ] **Step 4: Commit** `feat: download into the shared store and link projects to it`.

### Task 12: `store_link`, `store_unlink`, `store_adopt`, `store_verify`

**Files:**
- Modify: `metaquest/cli/commands/store.py`, `metaquest/cli/main.py`, `tests/test_cli_main.py`
- Create: `metaquest/store/adopt.py` (`adopt(project_fastq, paths, move: bool, dry_run: bool, on_progress) -> AdoptReport`), `tests/test_store_adopt.py`, extend `tests/test_cli_store.py`

**Interfaces:**
- `store_adopt [--fastq-folder fastq] [--move|--copy] [--dry-run] [--data-root] [--registry]`: for each `fastq/<ACC>` directory (not a link): if the store has the accession, compare bytes and md5 of same-named files; equal -> drop the project copy and link; different -> keep both and report a conflict; else move (same filesystem) or copy, compress when not compressed, write the sidecar, catalogue upsert, link, `record_download(..., attempt=False)` refresh, usage `linked` (Task 14 adds the usage call; here just link). Restartable: a half-moved accession (files in both places) is completed on rerun. Refuses `--move` across filesystems (`os.stat().st_dev` differs) unless `--copy`.
- `store_verify [ACC...] [--md5] [--spots] [--fix-state]`: recomputes bytes (always), md5 (`--md5`), read counts against `ncbi.spots` (`--spots`), prints a table, `--fix-state` rewrites sidecar state and the catalogue.
- `store_link ACC... [--fastq-folder] [--link-mode]`, `store_unlink ACC...` (removes only symlinks, never directories).

- [ ] **Step 1: Failing tests.** Adopt on a tmp project with two accessions moves them, writes sidecars, links; dedup path drops an identical copy; conflict path keeps both; dry run writes nothing; restart completes a half-moved accession; verify flags a truncated dataset and `--fix-state` updates it; unlink refuses a real directory.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: store_adopt, store_verify, store_link and store_unlink`. Open PR B2.

---

## Bundle B3: usage tracking (branch `feat/store-usage` from B2)

### Task 13: usage rows from every stage

**Files:**
- Create: `metaquest/store/usage.py` (`record_usage_safe(paths, registry, accession, genome_id, stage, detail="")`: resolves `project.id`, upserts the project row with the current path and registry, records usage; any exception logged at warning, never raised), `tests/test_store_usage.py`
- Modify: `metaquest/cli/commands/sra.py` (`linked`/`downloaded`), `sra_enhanced.py` and `sra_intelligent.py` (`analysed`), `read_extraction.py` (`extracted`, `assembled` per genome), `store.py` (`store_adopt` records `linked`)
- Test: extend the CLI tests of each command with a tmp store and assert the catalogue rows

- [ ] **Step 1: Failing tests.** Each stage writes its row with the right `genome_id`; a catalogue failure (patch `catalog_write` to raise) leaves the command's exit code and outputs unchanged; a moved project (change `project.path`) refreshes the projects row.
- [ ] **Step 2-4: Implement, verify, commit** `feat: record dataset usage per project and genome in the catalogue`.

### Task 14: `store_usage` and `store_gc`

**Files:**
- Modify: `metaquest/cli/commands/store.py`, `metaquest/cli/main.py`, `tests/test_cli_main.py`, `tests/test_cli_store.py`
- Create: `tests/test_store_gc.py`

**Interfaces:**
- `store_usage [--accession ACC] [--project NAME|ID] [--organism GENOME_ID] [--unused] [--bytes-by-organism] [--json]`.
- `store_gc [--dry-run (default)] [--yes] [--older-than DAYS] [--keep-partial]`: candidates are `unused` datasets (no usage rows or only rows from stale projects); stale = registry path missing or its `project.id` differs; prints candidates and bytes; deletes only with `--yes`, removing `sra/<ACC>` and the catalogue rows.

- [ ] **Step 1: Failing tests.** Queries on a seeded catalogue; stale detection; gc dry run deletes nothing; `--yes` removes the folder and rows; `--keep-partial` spares partial datasets.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: store_usage and store_gc`. Open PR B3.

---

## Bundle C1: screening and selection (branch `feat/screen-select` from A1)

### Task 15: Branchwater retry, cache and streaming; cANI details table

**Files:**
- Modify: `metaquest/data/branchwater_search.py:126-158` (`search_index(signature, threshold, server, timeout, cache_dir: Optional[Path] = None, refresh=False, max_cache_age_days: Optional[int] = None)`), `metaquest/cli/commands/branchwater_search.py` (`--no-cache`, `--refresh`, `--max-cache-age-days`), `metaquest/data/branchwater.py:242-362` (`_process_genome_containments` keeps `additional_data`; `_generate_containment_summary` writes `parsed_containment_details.tsv`), `metaquest/cli/commands/containment.py` (`--details-file`)
- Test: `tests/test_data_branchwater_search.py`, `tests/test_cli_branchwater_search.py`, `tests/test_data_branchwater.py`, `tests/test_cli_commands.py`

**Interfaces:**
- `search_index` uses a `requests.Session` mounted with `HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"]))`; other 4xx raise at once; the response is parsed from `iter_lines()`.
- Cache key `sha256(json.dumps({"mins": sorted(mins), "ksize", "scaled", "threshold", "server"}, sort_keys=True))`; files `<cache_dir>/<key>.csv` and `<key>.json` (`{"fetched": iso, "server", "threshold", "rows"}`); hit logs `using cached Branchwater result from <date>`; default `cache_dir = output.parent / ".branchwater-cache"`.
- Details table columns: `accession, genome_id, containment, cANI, biosample, bioproject, assay_type, organism, geo_loc_name, lat_lon` (missing values empty), written next to the parsed table as `<stem>_details.tsv`.

- [ ] **Step 1: Failing tests.** Retry adapter present on the session (inspect `session.adapters["https://"].max_retries.total == 4`); 404 raises without retry (mock `requests.Session.post`); cache write then hit without a request; `--refresh` bypasses; stale by age refetches; details table rows for a matches folder with cANI columns; `parsed_containment.txt` byte-identical to before.
- [ ] **Step 2-4: Implement, verify, commit** `feat: Branchwater retries and response cache; keep cANI and sample metadata`.

### Task 16: selection flags

**Files:**
- Modify: `metaquest/processing/selection.py:16-76` (`select_accessions(..., top_n: Optional[int] = None, exclude: Optional[Set[str]] = None, genome_ids: Optional[List[str]] = None, require: str = "any")`), `metaquest/cli/commands/select.py` (`--top-n`, `--skip-excluded/--no-skip-excluded` default on, `--skip-downloaded`, `--genome-ids`, `--require {any,all}`), `metaquest/data/registry.py` (`record_selection` per-accession `{rank, column, value}` under `selection.ranked` capped at the selected list)
- Test: `tests/test_processing_selection.py` (or existing), `tests/test_cli_select.py`, `tests/test_data_registry.py`

- [ ] **Step 1: Failing tests.** top-N after threshold and exclusions; excluded accessions removed by default and the log line counts; downloaded removed with the flag; `any`/`all` across two columns; argparse rejects `--genome-id` with `--genome-ids`; registry keeps ranks.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: select_datasets --top-n, registry-aware skips and multi-genome rules`. Open PR C1.

---

## Bundle C2: extraction and assembly (branch `feat/extract-assemble` from C1)

### Task 17: index reuse, alignment filters, threads, SAM lifetime, mate pre-count

**Files:**
- Modify: `metaquest/data/read_extraction.py:161-232, 280-320` (`build_index(genome_fasta, preset, index_dir) -> Path`; `_map_and_extract(..., reference: Path, min_mapq: int, sam_dir: Optional[Path])`; `extract_target_reads(..., min_mapq=0, temp_folder=None, allow_truncated=False, mate_counts: Optional[Dict[str, Tuple[int, int]]] = None)`), `metaquest/core/constants.py` (minimap2 `-d`; samtools `-q`), `metaquest/utils/security.py` (`-d` in `PATH_VALUE_FLAGS`), `metaquest/cli/commands/read_extraction.py` (`--min-mapq`, `--temp-folder`, `--allow-truncated`, `--debug-keep-sam`), `tests/helpers_extraction.py` (index call creates the `.mmi`; view honours `-F 0x904`/`-q`)
- Test: `tests/test_read_extraction.py`, `tests/test_cli_read_extraction.py`, `tests/test_security_comprehensive.py`

**Interfaces:**
- Index at `<output_folder>/.index/<genome_stem>.<preset>.mmi`, rebuilt when the FASTA mtime is newer; on a minimap2 failure with the index, retry once with the FASTA and log.
- `samtools view -b -F 0x904 [-q N] -@ T -o <bam> <sam>`; log `kept X of Y records (secondary/supplementary removed: Z, below MAPQ: W)` using two `samtools view -c` calls (`-c -F 4` and `-c -F 0x904`); `samtools fastq -@ T ...`; the SAM is unlinked right after the BAM exists (kept with `--debug-keep-sam`); `params` recorded: `filter_flags: "0x904"`, `min_mapq`, `index`.
- Unequal mates: when `mate_counts` (from the registry download record or a fresh `count_fastq_reads` of both mates, cached into the download record) differ, log and map single-end deliberately; the stderr substring stays as a second signal.
- `partial`/`truncated` download verdict: skip the sample with a warning unless `allow_truncated`.

- [ ] **Step 1: Failing tests.** Index built once for three samples and reused (call list has one `-d` call); rebuilt when the FASTA is touched newer; filters and counts in the samtools calls; `-@` on fastq; SAM absent after the run; truncated sample skipped; mate count mismatch handled without the stderr marker.
- [ ] **Step 2-4: Implement, verify, commit** `feat: reuse the minimap2 index, drop secondary alignments, thread samtools fastq`.

### Task 18: assembly presets, cleanup, coverage, richer stats

**Files:**
- Modify: `metaquest/data/read_extraction.py:359-455` (`assemble_extracted_reads(..., preset: Optional[str] = "meta-sensitive", keep_intermediate=False)`; `assembly_coverage(contigs, reads, preset, threads, tmp_dir) -> Dict`; `summarise_contigs` adds `n90`, `gc`, `contigs_ge_1kb`), `metaquest/core/constants.py` (megahit `--presets`; samtools `sort`, `coverage`, `-T`, `-m`), `metaquest/cli/commands/read_extraction.py` (`--assembly-preset {default,meta-sensitive,meta-large}`, `--keep-intermediate`, `--no-coverage`), `metaquest/data/registry.py` (`record_assembly` passes the whole stats dict through), `tests/helpers_extraction.py`
- Test: `tests/test_read_extraction.py`, `tests/test_cli_read_extraction.py`, `tests/test_data_registry.py`, `tests/test_security_comprehensive.py`

**Interfaces:**
- megahit args gain `--presets <preset>` unless `default`; explicit `--k-min/--k-max/--k-step` with a preset raises `ProcessingError`.
- After a successful assembly `intermediate_contigs/` is removed unless `keep_intermediate`.
- `assembly_coverage`: `minimap2 -a -x <preset> -t T -o <tmp.sam> final.contigs.fa <reads>`, `samtools view -b -F 0x904 -@ T -o <tmp.bam>`, `samtools view -c <tmp.bam>`; returns `{reads_mapped, mapping_rate (mapped / extracted mapped_reads), mean_depth_estimate (mapped * avg_read_len / total_bp)}`; `samtools sort` + `samtools coverage` produce `per_contig` breadth when `--coverage-detail` is given (P2).
- Stats recorded: `contigs, total_bp, n50, n90, largest, gc, contigs_ge_1kb, genome_fraction_estimate (total_bp / genome_length), reads_mapped, mapping_rate, mean_depth_estimate`.

- [ ] **Step 1: Failing tests.** Preset in the megahit call and rejection with k flags; `intermediate_contigs/` removed; coverage numbers from faked counts; N90 and GC from a small contig file; registry assembly block has the new keys.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: megahit presets, cleanup, and assembly coverage from mapped reads`. Open PR C2.

---

## Bundle C3: analysis cache (branch `feat/analysis-cache` from B1)

### Task 19: shared statistics record

**Files:**
- Create: `metaquest/store/stats.py`, `tests/test_store_stats.py`
- Modify: `metaquest/data/sra_metadata.py:330-419, 515-543` (`calculate_read_statistics(..., max_reads=100000, cached: Optional[Dict] = None)`, `_dataset_stats_row` prefers the cache), `metaquest/sra/analytics.py:141-224` (`analyze_fastq_quality(fastq_path, sample_size=10000, sampler="uniform")` reservoir sampling; `gc_distribution` becomes `gc_histogram` bucket dict), `metaquest/sra/reporting.py:95-132` (`load_quality_profiles` accepts both keys), `metaquest/cli/commands/sra_intelligent.py` (`--sample-size`, JSON writes `gc_histogram`), `metaquest/sra/analytics.py:512-583` (`compare_datasets(..., profiles=None)`, `detect_dataset_anomalies(..., profiles=None)`), `metaquest/cli/commands/sra_intelligent.py` (`sra_compare` loads `--quality-profiles`)

**Interfaces:**
```python
def compute_dataset_stats(files: List[Path], sample_size: int = 10000, use_seqkit: bool = True) -> Dict[str, Any]
# reads_per_file, reads_total, bases_total, avg/min/max_read_length, n50, gc_content, length_histogram,
# quality_summary {mean, median, q25, q75}, duplication_rate, n_content, sample_size, sampled, computed (iso),
# signature {name: [bytes, mtime]} for invalidation
def cached_stats(acc_dir: Path, sidecar_path: Optional[Path]) -> Optional[Dict]   # sidecar stats when signature matches
def store_stats(sidecar_path, stats) -> None
```
- Counting: streaming block count; `seqkit stats -T -j N` when installed (allowlist `seqkit` with `stats -T -j --version`).
- Reservoir sampling over the whole file for the quality/GC/complexity part; `sampled=True` and `sample_size` reported.

- [ ] **Step 1: Failing tests.** Counts on plain and gz; uniform sample includes reads from the tail of a 50 000-read synthetic file; cache hit and invalidation by size change; `analyze_fastq_quality` output holds `gc_histogram` and no per-read list; `load_quality_profiles` reads old and new JSON; `compare_datasets(profiles=...)` does not call `profile_dataset_quality`; `calculate_read_statistics` uses the cache when given.
- [ ] **Step 2-4: Implement, verify, commit** `feat: one cached statistics record per dataset, uniform sampling, profile reuse`.

### Task 20: `sra_validate` rewrite and pre-flight tool checks

**Files:**
- Modify: `metaquest/cli/commands/sra_enhanced.py:225-372` (`_fastq_format_issues` checks the first record's four lines of every file; `_mate_count_issues` from cached stats or `count_fastq_reads`; `_completeness_issues` from the sidecar or the registry verdict; `--md5`), `metaquest/cli/commands/read_extraction.py` and `sra_intelligent.py` (pre-flight `shutil.which` for minimap2, samtools, megahit with `conda install -c bioconda <tool>` hints; `--dry-run` skips it)
- Test: `tests/test_cli_commands_sra_enhanced.py`, `tests/test_cli_read_extraction.py`

- [ ] **Step 1: Failing tests.** Validate fails a partial sidecar with the "N reads on disk vs M spots at NCBI" message; mate mismatch reported; first-record shape check catches a broken header without reading the whole file (assert via a large file and a patched `open` call count, or a time bound); `--md5` mismatch reported; missing minimap2 returns 1 before any `run_secure` call with the hint in the log.
- [ ] **Step 2-3: Implement, verify** (`make check`, `make test`, `make pipeline`).
- [ ] **Step 4: Commit** `feat: sra_validate checks completeness and mates; pre-flight tool checks`. Open PR C3.

---

## Bundle D: docs and walkthrough (branch `docs/audit-round` from C3 after all merges)

### Task 21: documentation, walkthrough, memory

**Files:**
- Modify: `README.md` (new "Shared data store" section after "Project state": init, adopt, usage, gc, `METAQUEST_DATA`, config file; per-stage flag mentions: `--verify-downloads`, `--sra-cache`, `--no-compress`, `--top-n`, `--skip-downloaded`, `--min-mapq`, `--assembly-preset`, `--sample-size`, Branchwater cache flags), `docs/pipeline_overview.md` (store paragraph per stage, external tools table adds prefetch, pigz, seqkit), `docs/ARCHITECTURE.md` (store package), `docs/branchwater_workflow.md` (cache and details table), `CLAUDE.md` and `AGENTS.md` (Store group, store rules), `environment.yml` (pigz, seqkit optional comment), `local_test.sh` (`store_init --data-root store`, `download_sra --dry-run --data-root store`, `store_status --json` count assertion, `select_datasets --top-n 5`), `.gitignore` (nothing new; document the project template in README)
- [ ] **Step 1:** Write; verify every flag with `metaquest <cmd> --help`; `make check`, `make test`, `make pipeline`; grep that README, pipeline_overview, ARCHITECTURE and CLAUDE.md all mention `store_init`.
- [ ] **Step 2: Commit** `docs: shared data store, stage flags and the walkthrough`. Open PR D.

---

## Verification after all bundles merge

- `make check`, `make test`, `make pipeline`, `make test-network` green on `main`.
- Real data (Wolbachia scratch directory; tools on PATH from the `mt`, `mq-megahit` and `sra-tools` conda envs): `parse_metadata` fills spots and size; `status` lists the 300 000-read runs as truncated; `store_init --data-root <scratch>/store`; `store_adopt` in the wMel project then in the wPip project (the second dedups); `store_usage --accession SRR11011981` shows both projects; `extract_target_reads` rerun shows the index reuse and the secondary-alignment filter in the log; `sra_stats`, `sra_validate`, `sra_profile_quality` agree on read counts; `download_sra --dry-run` shows the link plan.
