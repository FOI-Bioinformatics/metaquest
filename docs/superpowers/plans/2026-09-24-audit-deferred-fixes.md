# Audit Deferred Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Minor and Usability findings deferred by the Lactobacillus audit fix plan, in priority order: the store and extraction behaviours that can still lose or mislabel data first, then the messages a first-time user reads, then the test gaps the reviewers named.

**Architecture:** Every change is local to the function the finding names; no new subsystem. The store journal gains a hostname field and replays before the catalogue rebuild; adoption records a `copied` usage row so `store_gc` keeps copied datasets; megahit scratch becomes a per-run temporary folder; messages and JSON keys are aligned with what the console prints. Tests reproduce each finding first.

**Tech Stack:** Python 3.12, pytest, pandas, sqlite3, argparse; black/flake8/mypy via `make check`.

**Spec:** `docs/superpowers/specs/2026-09-23-lactobacillus-e2e-audit.md` (finding ids S<stage>-<n>) and the deferred lists of `docs/superpowers/plans/2026-09-23-lactobacillus-audit-fixes.md` and its final review (items named "final review" below). Merged state: `main` at 3f7f2e4.

## Global Constraints

- Line length 120, black formatting, flake8 clean, mypy clean, cyclomatic complexity ceiling: `make check` must pass after every task.
- No Unicode in code or docs. Modest scientific language in messages and docstrings.
- CLI arguments use dashes. Existing flag names and output file names do not change.
- Tests never touch a real store (tmp_path only; the autouse fixtures in `tests/conftest.py` isolate discovery) and never depend on tools on PATH or the network.
- Sidecar schema version (1) and registry version (2) unchanged; journal records stay one JSON object per line and older lines without new keys must still replay.
- Commit after every task with a conventional prefix and the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer.

## Review Focus

1. A store whose journal holds usage lines for a dataset `store_gc` removed: after `store_reindex` no placeholder row for it exists and `store_usage` does not list it. (Task 1)
2. Two `extract_target_reads --assemble` runs started at the same time on one output folder for two genomes: neither removes the other's megahit scratch. (Task 2)
3. A project that adopted a dataset with `--copy` and then a second project links the same accession: `store_gc --dry-run` keeps the dataset and names the copying project as a user. (Task 1)
4. `select_datasets --no-record` followed by `status --next`: the registry's selected stage and the suggested download list are unchanged from before the exploratory run. (Task 4)
5. Ctrl-C in the main thread while two real child processes run inside `_execute_parallel_downloads`: the interrupt is re-raised within a few seconds, no lock file remains, no child stays tracked. (Task 7)

---

## File structure

- `metaquest/store/journal.py`, `metaquest/store/catalog.py`, `metaquest/cli/commands/store.py`: replay order, hostname in journal lines, copied usage rows, gc `--json` refusal object, stale flag on an empty reindex, `_wrap_sqlite_errors` on backfill.
- `metaquest/cli/commands/read_extraction.py`, `metaquest/data/read_extraction.py`: per-run megahit scratch, dangling-link report, redo reason wording, docstrings.
- `metaquest/cli/commands/sra.py`, `metaquest/data/sra.py`: relink carries the read count; summary distinguishes links.
- `metaquest/cli/commands/genome.py`, `metaquest/data/gtdb.py`: old-name hint; `genome_download` records genomes.
- `metaquest/cli/commands/branchwater_search.py`, `containment.py`, `explore.py`, `branchwater.py`: stable best-hit ordering, "Next:" hints, empty-metadata warning.
- `metaquest/cli/commands/select.py`, `status.py`: `--no-record`, runnable reselect suggestion, store-state wording.
- `metaquest/cli/commands/sra_enhanced.py`, `sra_intelligent.py`, `metaquest/sra/analytics.py`, `metaquest/data/sra_metadata.py`: unit labels, sample size wording, JSON keys, validate message, `sra_info` filtered to the requested runs, `json_safe` pandas branch, "not a directory" wording.
- Tests: the files named in each task, plus a new `tests/test_interrupt_e2e.py`.

---

### Task 1: Store journal and gc: replay order, copied datasets, hostname, `--json` refusal, empty reindex, backfill errors

**Files:**
- Modify: `metaquest/cli/commands/store.py` (StoreReindexCommand.execute around 440-450; StoreAdoptCommand.execute around 595-625; StoreGcCommand.execute and `_refuse_before_candidates`), `metaquest/store/journal.py`, `metaquest/store/catalog.py` (`upsert_project`, `backfill` call site)
- Test: `tests/test_cli_store.py`, `tests/test_store_gc.py`, `tests/test_store_journal.py`

**Interfaces:**
- Produces: journal project lines carry `hostname`; `replay` passes it to `upsert_project(..., hostname=...)` (new optional keyword, default `socket.gethostname()`), and usage lines' `at` sets `first_used`/`last_used` on replay (`record_usage(..., at=None)` optional keyword); `store_reindex` replays the journal BEFORE `catalog.reindex(...)`; `store_adopt` records a usage row with stage `copied` for `report.copied` and for a dedup under `--copy`; `store_gc --json` prints `{"error": "<message>"}` on either refusal; a reindex that restores no projects and finds no datasets clears a stale `rebuilt_without_projects` flag; `journal.backfill_from_catalog` is wrapped with `_wrap_sqlite_errors` semantics (a `sqlite3.Error` becomes `DataAccessError`).

- [ ] **Step 1: Write the failing tests**

`tests/test_cli_store.py` (TestStoreReindexNeverLosesHistory):

```python
    def test_reindex_does_not_resurrect_a_removed_dataset(self, tmp_path, monkeypatch):
        paths = init_store(tmp_path / "store")
        monkeypatch.chdir(tmp_path)
        write_sidecar(sidecar_path(paths, "SRR1"), _sidecar("SRR1"))
        with catalog_write(paths) as c:
            c.upsert_dataset(_sidecar("SRR1"))
            c.upsert_project("pid1", "proj", str(tmp_path), "r.json")
            c.record_usage("SRR1", "pid1", "", "linked", "")
            c.record_usage("SRR9", "pid1", "", "linked", "")  # SRR9 was removed by gc later
        assert StoreReindexCommand().execute(_reindex_args(data_root=str(paths.root))) == 0
        with catalog_write(paths) as c:
            rows = {r["accession"] for r in c.conn.execute("SELECT accession FROM datasets").fetchall()}
        assert rows == {"SRR1"}

    def test_replay_restores_hostname_and_first_used(self, tmp_path, monkeypatch):
        paths = init_store(tmp_path / "store")
        monkeypatch.chdir(tmp_path)
        write_sidecar(sidecar_path(paths, "SRR1"), _sidecar("SRR1"))
        with catalog_write(paths) as c:
            c.upsert_dataset(_sidecar("SRR1"))
            c.upsert_project("pid1", "proj", str(tmp_path), "r.json", hostname="other-host")
            c.record_usage("SRR1", "pid1", "", "linked", "", at="2026-01-01T00:00:00+00:00")
        (paths.root / "catalog.sqlite").unlink()
        assert StoreReindexCommand().execute(_reindex_args(data_root=str(paths.root))) == 0
        with catalog_write(paths) as c:
            row = c.conn.execute("SELECT hostname FROM projects WHERE project_id='pid1'").fetchone()
            used = c.conn.execute("SELECT first_used FROM usage WHERE accession='SRR1'").fetchone()
        assert row["hostname"] == "other-host"
        assert used["first_used"].startswith("2026-01-01")
```

`tests/test_cli_store.py` (TestStoreAdoptCommand):

```python
    def test_copy_adoption_records_copied_usage_so_gc_keeps_it(self, tmp_path, monkeypatch):
        # adopt with --copy (see test_copy_mode_dedup_does_not_record_link for the fixture), then
        # store_gc --dry-run must list the dataset under kept with reason naming the project
```

`tests/test_store_gc.py`:

```python
    def test_json_refusal_prints_an_error_object(self, tmp_path, capsys):
        # datasets present, no projects: execute(_gc_args(..., json=True)) == 1 and json.loads(stdout)["error"]
    def test_empty_reindex_clears_a_stale_rebuilt_flag(self, tmp_path):
        # set the flag, remove all datasets, reindex: flag gone
```

`tests/test_store_journal.py`:

```python
def test_backfill_sqlite_error_becomes_data_access_error(tmp_path, monkeypatch):
    # monkeypatch catalog.conn.execute to raise sqlite3.OperationalError inside backfill; expect DataAccessError
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest tests/test_cli_store.py tests/test_store_gc.py tests/test_store_journal.py -k "resurrect or hostname or copied_usage or json_refusal or stale_rebuilt or sqlite_error" -v`
Expected: FAIL on each (SRR9 placeholder present; hostname is the local host; gc lists the copied dataset as unused; no JSON; flag stays; raw sqlite3 error).

- [ ] **Step 3: Implement**

- `journal.append_project` adds `"hostname": socket.gethostname()`; `replay` passes `record.get("hostname")` and `record.get("at")` through; `Catalog.upsert_project(..., hostname: Optional[str] = None)` uses it (default the local host); `Catalog.record_usage(..., at: Optional[str] = None)` uses `at` for `first_used`/`last_used` when given.
- `StoreReindexCommand.execute`: call `journal.replay(paths, catalog)` first, then `catalog.reindex(sidecars)`; when projects == 0 and no sidecars, `catalog.delete_meta(REBUILT_WITHOUT_PROJECTS)`.
- `StoreAdoptCommand.execute`: after `_record_linked`, `record_usage_many(paths, registry, [(acc, genome_id="", stage="copied", detail="store_adopt --copy") for acc in report.copied + (report.deduplicated if not args.move else [])])`.
- `StoreGcCommand`: on either refusal, if `args.json`, `print(json.dumps({"error": message}))`.
- `journal.backfill_from_catalog`: wrap the two `conn.execute` calls in `try/except sqlite3.Error as e: raise DataAccessError(str(e)) from e`.

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_cli_store.py tests/test_store_gc.py tests/test_store_journal.py tests/test_store_catalog.py tests/test_store_usage.py -q && make check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "fix: journal replays before the rebuild, keeps hostnames and dates, copied datasets count as used, gc --json refusal (audit deferred)"
```

---

### Task 2: Extraction: per-run megahit scratch, dangling links reported, redo reason, docstrings

**Files:**
- Modify: `metaquest/cli/commands/read_extraction.py` (`_assemble` around 390-412; `execute`), `metaquest/data/read_extraction.py` (`_extract_one_sample` redo log; `_megahit_args` and `assemble_extracted_reads` docstrings)
- Test: `tests/test_cli_read_extraction.py`, `tests/test_read_extraction.py`

**Interfaces:**
- Produces: the default megahit scratch is `tempfile.mkdtemp(dir=args.output_folder, prefix=".megahit-tmp-")` per run, removed in the same `finally`; before extraction the CLI logs one WARNING listing dangling `fastq/<ACC>` links (`metaquest.store.link.dangling_links(fastq_folder)`) when any exist; the redo log distinguishes "recorded parameters differ" from "recorded output file missing".

- [ ] **Step 1: Write the failing tests**

```python
def test_default_megahit_scratch_is_unique_per_run(tmp_path, monkeypatch):
    # run _assemble twice with the fake tool runner recording --tmp-dir; assert the two values differ
    # and both start with "<output>/.megahit-tmp-", and neither exists afterwards

def test_dangling_links_are_reported_once(tmp_path, caplog):
    # fastq/SRR9 -> missing target; execute(--dry-run); assert one WARNING naming SRR9 and "dangling"

def test_redo_reason_names_missing_output(tmp_path, caplog):
    # record with matching params but files pointing to a deleted path; assert "recorded output file missing" logged
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest tests/test_cli_read_extraction.py tests/test_read_extraction.py -k "unique_per_run or dangling or redo_reason" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** the three changes above and add the `ProcessingError` for a nested `tmp_dir` to both docstrings' `Raises:` sections.

- [ ] **Step 4: Run tests and gates**

Run: `pytest tests/test_cli_read_extraction.py tests/test_read_extraction.py -q && make check`

- [ ] **Step 5: Commit**

```bash
git commit -am "fix: per-run megahit scratch, dangling links reported, redo reason named (audit deferred S7-2)"
```

---

### Task 3: Download relink and summary: read count carried, links counted separately

**Files:**
- Modify: `metaquest/cli/commands/sra.py` (`_result_recorder._record_result` around 346-379; `_log_download_summary`), `metaquest/data/sra.py` (`_link_result`, data-layer summary around 1600-1605)
- Test: `tests/test_cli_commands.py`, `tests/test_data_sra.py`

**Interfaces:**
- Produces: `_link_result` message stays "linked from store, N files"; `_record_result` for a linked result takes `reads_r1` from `sidecar.reads_per_mate` (via `sidecar_completeness`) and never overwrites an existing `complete` block with a None-valued one (when the sidecar has no count, keep the registry's previous `reads_r1`); the data-layer summary prints "Linked from store: k" and "Newly downloaded" excludes links.

- [ ] **Step 1: Write the failing tests**

```python
def test_relink_keeps_previous_read_count(tmp_path):
    # registry has download.complete.reads_r1 = 5 and verdict truncated; store sidecar complete with reads_per_mate None;
    # download_sra with --redownload-truncated links; assert registry reads_r1 == 5 and verdict complete (from sidecar)
def test_summary_counts_links_separately(caplog):
    # results: one "linked from store, 2 files", one "Downloaded 2 files, complete ...; stored":
    # assert "Newly downloaded: 1" and "Linked from store: 1" in the data-layer summary
```

- [ ] **Step 2: Run and watch them fail**. Run: `pytest tests/test_data_sra.py tests/test_cli_commands.py -k "relink_keeps or counts_links" -v`

- [ ] **Step 3: Implement**; keep `STORE_LINKED_PREFIX` as the single detection.

- [ ] **Step 4: Run tests and gates**. Run: `pytest tests/test_data_sra.py tests/test_cli_commands.py -q && make check`

- [ ] **Step 5: Commit**

```bash
git commit -am "fix: a relink keeps the recorded read count and the download summary counts links separately (audit deferred S7-7, S5-3)"
```

---

### Task 4: Selection and status: `select_datasets --no-record`, runnable reselect suggestion, store-state wording

**Files:**
- Modify: `metaquest/cli/commands/select.py` (add `--no-record`, skip `record_selection` when set), `metaquest/cli/commands/status.py` (`_download_next_steps` reselect command string; `_inventory_report` wording when a linked accession's store state is not ready)
- Test: `tests/test_cli_select.py`, `tests/test_cli_status.py`

**Interfaces:**
- Produces: `select_datasets --no-record` writes the output file and logs the counts but leaves the registry untouched; the reselect suggestion is a runnable command built from the recorded criteria: `metaquest select_datasets --genome-id <column> --threshold <threshold> --skip-excluded --output <output>` (with `--genome-ids ... --require ...` when the criteria hold `genome_ids`); `status` prints "FASTQ: N present, M missing, K linked to a store dataset that is not complete" when links resolve to datasets whose sidecar state is outside `STORE_READY_STATES`.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_record_leaves_registry_untouched(tmp_path, monkeypatch):
    # run select_datasets twice: normal, then --no-record with a different threshold; registry selected set unchanged
def test_reselect_suggestion_is_runnable(tmp_path, monkeypatch):
    # after a --no-skip-excluded selection, _download_next_steps returns a command starting with
    # "metaquest select_datasets --genome-id GCF_A --threshold 0.5 --skip-excluded --output sel_noskip.txt"
def test_status_names_links_to_incomplete_store_datasets(tmp_path, monkeypatch):
    # store dataset with state failed linked from fastq/SRR1; status text contains "linked to a store dataset that is not complete"
```

- [ ] **Step 2: Run and watch them fail**. Run: `pytest tests/test_cli_select.py tests/test_cli_status.py -k "no_record or runnable or not_complete" -v`

- [ ] **Step 3: Implement**.

- [ ] **Step 4: Run tests and gates**. Run: `pytest tests/test_cli_select.py tests/test_cli_status.py -q && make check`

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: select_datasets --no-record; runnable reselect suggestion; status names links to incomplete store datasets (audit deferred S4-1, S5-4)"
```

---

### Task 5: Screening and genome messages: GTDB old-name hint, `genome_download` records genomes, stable best hit, "Next:" hints, empty-metadata warning

**Files:**
- Modify: `metaquest/cli/commands/genome.py` (search error message around 104; `GenomeDownloadCommand.execute` records genomes with `record_genome`), `metaquest/data/gtdb.py` (26, 52: include the HTTP status and a hint for 400), `metaquest/cli/commands/branchwater_search.py` (86: choose the best hit by `(-containment, accession)`), `metaquest/cli/commands/containment.py` and `explore.py` and `branchwater.py` (add one `Next:` INFO line after success: use_branchwater -> parse_containment; parse_containment -> plot_containment/select_datasets; explore_containment -> open the HTML), `metaquest/data/branchwater.py` (`extract_metadata_from_branchwater`: WARNING when no metadata column beyond Run_ID and cANI is populated, naming `download_metadata`)
- Test: `tests/test_cli_genome.py`, `tests/test_data_gtdb.py` (or the existing gtdb test file), `tests/test_cli_commands.py`, `tests/test_data_branchwater.py`

- [ ] **Step 1: Write the failing tests** (one per item: a 400 from GTDB yields a message containing "not a current GTDB name" and the hint to try the current genus; `genome_download` leaves `registry.genomes` with the two accessions; a tie at the top picks the alphabetically first accession deterministically; each command logs a line starting with "Next:"; the API-shaped CSV yields the warning).

- [ ] **Step 2: Run and watch them fail**.

- [ ] **Step 3: Implement**.

- [ ] **Step 4: Run tests and gates**. Run: `pytest tests/test_cli_genome.py tests/test_cli_commands.py tests/test_data_branchwater.py tests/test_data_gtdb*.py -q && make check`

- [ ] **Step 5: Commit**

```bash
git commit -am "fix: clearer GTDB name errors, genome_download records genomes, stable best hit, Next hints, empty-metadata warning (audit deferred S1-1, S1-4, S2-2, S2-4, S3-1)"
```

---

### Task 6: Analysis labels and `sra_info` scope

**Files:**
- Modify: `metaquest/cli/commands/sra_intelligent.py` (138: "Total reads" -> "Total reads (mates counted)"; 302 likewise; 611: "Mean reads in sample" instead of "Mean total reads"; profile JSON keys: add `sequence_complexity` alongside `complexity_score`, keep the old key), `metaquest/cli/commands/sra_enhanced.py` (`sra_stats` summary "Total reads (mates counted)"; 424: the validate message becomes "store state failed: <sidecar error or 'see store_verify'>"), `metaquest/data/sra_metadata.py` (`get_sra_metadata`/`_parse_sra_xml`: keep only runs whose accession was requested when a request list is known; fall back to `Statistics/@nspots` when the RUN attributes are missing), `metaquest/sra/analytics.py` (`json_safe`: convert `pd.Series`/`pd.DataFrame` via `.to_dict()`), `sra_intelligent.py` 193 wording "is not a directory".
- Test: `tests/test_cli_sra_intelligent.py`, `tests/test_cli_commands.py`, `tests/test_sra_metadata_extended.py`, `tests/test_sra_analytics.py`

- [ ] **Step 1: Write the failing tests** (labels present in console output; JSON has both keys; validate message; `_parse_sra_xml(xml, requested={"SRR100"})` drops SRR101; `json_safe(pd.Series([1]))` gives a list/dict; wording).

- [ ] **Step 2: Run and watch them fail**.

- [ ] **Step 3: Implement**.

- [ ] **Step 4: Run tests and gates**. Run: `pytest tests/test_cli_sra_intelligent.py tests/test_cli_commands.py tests/test_sra_metadata_extended.py tests/test_sra_analytics.py -q && make check`

- [ ] **Step 5: Commit**

```bash
git commit -am "fix: read-count units labelled, profile JSON keys aligned, validate message, sra_info limited to requested runs (audit deferred S6-1, S6-4, S6-5, S6-7)"
```

---

### Task 7: Test gaps named by the reviewers

**Files:**
- Create: `tests/test_interrupt_e2e.py` (from `.claude/scratch/interrupt_e2e_probe.py` in the main checkout: two workers, real `sleep` children through `run_secure` with `ALLOWED_EXECUTABLES` patched, a `dataset_lock` per worker, SIGINT from a timer; assert `KeyboardInterrupt` re-raised within 5 s, no lock file left, no child tracked, no `sleep` process alive)
- Modify: `tests/test_data_branchwater.py` (real unreadable-file tests for `extract_metadata_from_branchwater` and `parse_containment_data`; caplog WARNING level of the summary line), `tests/test_store_adopt.py` (`_folder_bytes` ignores files inside hidden directories; a hidden-files-only folder reaches `report.empty`), `tests/test_store_journal.py` (replay twice is idempotent), `tests/test_store_gc.py` (`--dry-run` and `--json` under the no-project refusal), `tests/test_cli_status.py` (mixed skip/no-skip selection), `tests/test_sra_metadata_extended.py` (EXPERIMENT package without RUN_SET yields one record; RUN without accession falls back), `tests/test_data_metadata.py` (`Experiment_Library_Name` asserted), `tests/test_cli_sra_intelligent.py` (execute-level exit 1 when no accession source)
- Modify: `metaquest/store/adopt.py` `_folder_bytes` (skip hidden directories, not only hidden names: `if any(is_hidden_name(p) for p in sub.relative_to(folder).parts): continue`)

- [ ] **Step 1: Write the tests** (each must fail or be shown to exercise the branch: for the pure-coverage ones, run with `--cov` on the target module and name the newly covered lines in the report).

- [ ] **Step 2: Run** `pytest tests/test_interrupt_e2e.py -q` and the modified files; `make test && make check`.

- [ ] **Step 3: Commit**

```bash
git commit -am "test: end-to-end interrupt, unreadable match files, hidden folders in adopt, idempotent replay, gc refusal outputs, mixed selections (audit deferred)"
```

---

### Task 8: Documentation for the new flags and behaviours

**Files:**
- Modify: `README.md`, `docs/pipeline_overview.md`, `CLAUDE.md` store section: `select_datasets --no-record`; `store_gc --accept-rebuilt` and the rebuilt flag; `store_adopt --copy` datasets count as used; per-run megahit scratch folder name; read-count units; `sra_info` rows limited to the requested runs.

- [ ] **Step 1: Apply**, keep ASCII only (`python3 -c "..."` check from the previous plan), `make check`.

- [ ] **Step 2: Commit**

```bash
git commit -am "docs: deferred audit fixes (no-record, accept-rebuilt, copied datasets, scratch folders, units)"
```

---

## Verification after the last task

- `make check` and `make test` pass.
- On the sekvens2 reference data: `store_reindex` from `crispatus` restores the projects and no placeholder rows appear in `store_usage --unused`; `select_datasets --no-record --threshold 0.95` leaves `status` unchanged; `status --next` after a `--no-skip-excluded` selection prints a runnable `select_datasets ... --skip-excluded` line; `extract_target_reads --assemble` on one run on the APFS copy leaves no `.megahit-tmp-*` folder behind.
