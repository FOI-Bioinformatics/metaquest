# Audit Leftovers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last two spec findings no fix plan picked up (S5-6 download summary, S2-1 sourmash hint) and the minor notes the reviewers of the two previous fix plans parked, in that order.

**Architecture:** Small, local changes in the CLI layer (`metaquest/cli/commands/`), the store journal and catalogue (`metaquest/store/`), and tests. No new modules, no new commands, no schema change beyond one optional JSON key on usage journal lines.

**Tech Stack:** Python 3.12, argparse command registry, SQLite catalogue, append-only JSONL journal, pytest with tmp_path stores.

**Spec:** `docs/superpowers/specs/2026-09-23-lactobacillus-e2e-audit.md` (finding ids S<stage>-<n>) plus the review notes recorded in `docs/superpowers/plans/2026-09-24-audit-deferred-fixes.md` (items named "review note" below). Merged state: `main` at 8df4e1b.

## Global Constraints

- Line length 120; `make check` (black, flake8, mypy, complexity) and `make test` pass before every commit.
- ASCII only in code, messages and docs; plain scientific wording.
- CLI flags use dashes. Tests use `tmp_path` stores only and monkeypatch `METAQUEST_DATA`, `XDG_CONFIG_HOME` and `HOME`; no test depends on a tool on PATH.
- Every behavioural change is written RED first. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add` AGENTS.md.

## Review Focus

1. A failed download followed by a successful re-download: the sidecar carries no error and the run's summary is printed once, with links counted separately from new downloads. (Task 1)
2. Journal usage lines replayed out of order: `last_used` ends as the latest date, `first_used` as the earliest. (Task 3)
3. `store_gc --json`, `store_status --json`, `store_usage --json` with no store: valid JSON on stdout, exit 1. (Task 4)
4. `status --next` on a registry whose recorded threshold is not a number: one WARNING and a usable command with the default threshold. (Task 4)
5. `sra_info` for a sample accession (SRS): the runs of that sample's package are listed. (Task 5)

---

### Task 1: One download summary, links counted separately (S5-6)

**Files:**
- Modify: `metaquest/cli/commands/sra.py` (`_log_download_summary` ~line 230, `_count_linked` ~line 222, the call ~line 552)
- Test: `tests/test_cli_commands.py` (~lines 2295-2360, the two `test_summary_*` tests), `tests/test_cli_store.py` (fix-state)

**Interfaces:**
- Consumes: `download_sra()` in `metaquest/data/sra.py` already logs the whole summary via `_log_download_run_summary` (lines ~1530-1563: "Download summary:", "Total accessions", "Already downloaded", "Blacklisted", "Newly downloaded", "Linked from store", "Failed downloads").
- Produces: the CLI logs no second "Download summary:" block; its "Successfully downloaded: N" line, which counted links as downloads, disappears.

- [ ] **Step 1: Write the failing test**

Replace the assertion in `test_summary_reports_how_many_were_linked_from_store` and `test_summary_omits_linked_line_when_nothing_was_linked` (tests/test_cli_commands.py ~2295-2360) with one test:

```python
    def test_cli_logs_no_second_download_summary(self, mock_download, _which, tmp_path, caplog):
        """The data layer prints the run summary; the CLI must not print a second block whose
        "Successfully downloaded" line counted store links as downloads (audit S5-6)."""
        mock_download.return_value = {
            "total": 2, "already_downloaded": 0, "blacklisted": 0, "successful": 2, "failed": 0,
            "failed_accessions": [], "results": {"SRR1": "linked from store, 1 files", "SRR2": "downloaded"},
            "already_downloaded_accessions": [], "blacklisted_accessions": [], "skipped_accessions": [],
            "aborted": None,
        }
        args = self._args(tmp_path)  # reuse the existing helper/fixture the two old tests used
        with caplog.at_level("INFO"):
            assert DownloadSraCommand().execute(args) == 0
        assert "Download summary:" not in caplog.text
        assert "Successfully downloaded" not in caplog.text
```

Keep whatever fixture setup the two old tests used (mock_download, _which, the args builder); only the assertions change.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_cli_commands.py -k "no_second_download_summary" -v`
Expected: FAIL on `"Download summary:" not in caplog.text`.

- [ ] **Step 3: Remove the CLI summary block**

In `metaquest/cli/commands/sra.py` delete `_log_download_summary` and `_count_linked` (if `_count_linked` has no other caller) and the call at ~line 552. Keep `_report_failed_downloads` (it prints the retry command the data layer does not print) and `_write_report`. Remove the now-unused `STORE_LINKED_PREFIX` import if nothing else uses it.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cli_commands.py -q`
Expected: PASS.

- [ ] **Step 5: Sidecar error cleared (the second half of S5-6)**

`store_verify --fix-state` already sets `sidecar.error = None` (metaquest/cli/commands/store.py ~line 1002). Check `tests/test_cli_store.py` for a test that a failed sidecar with an error string ends with `error is None` after `--fix-state`; if none exists, add one next to the other fix-state tests (build a sidecar with `state="failed", error="fasterq-dump exit 3"` and full files on disk, run `store_verify --fix-state`, reload the sidecar, assert `state == "complete"` and `error is None`).

- [ ] **Step 6: Commit**

```bash
git add metaquest/cli/commands/sra.py tests/test_cli_commands.py tests/test_cli_store.py
git commit -m "fix: download_sra prints one summary; the CLI block that counted store links as downloads is gone (audit S5-6)"
```

---

### Task 2: sourmash install hint names the interpreter (S2-1)

**Files:**
- Modify: `metaquest/data/branchwater_search.py` (`SOURMASH_HINT` line 47, raise at line 63)
- Test: `tests/test_data_branchwater_search.py` (or wherever `sketch_fasta`'s missing-sourmash path is tested; grep `SOURMASH_HINT` and `sourmash is required`)

**Interfaces:**
- Produces: `sourmash_hint() -> str` returning `f"sourmash is required to sketch a genome. Install it into this interpreter with: {sys.executable} -m pip install 'metaquest[sourmash]'"`. Keep `SOURMASH_HINT` as a module constant equal to `sourmash_hint()` evaluated at import for any importer, or remove it if only line 63 uses it.

- [ ] **Step 1: Write the failing test**

```python
def test_missing_sourmash_hint_names_the_interpreter(monkeypatch):
    import sys
    from metaquest.data import branchwater_search as bs
    monkeypatch.setitem(sys.modules, "sourmash", None)  # import raises ImportError
    with pytest.raises(DataAccessError) as exc:
        bs.sketch_fasta(Path("genome.fna"))
    assert sys.executable in str(exc.value)
    assert "-m pip install 'metaquest[sourmash]'" in str(exc.value)
```

If `sketch_fasta` imports sourmash lazily inside the function (line ~60), `monkeypatch.setitem(sys.modules, "sourmash", None)` makes the import raise; if it imports at module level, patch the import the way the existing missing-sourmash test does.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_data_branchwater_search.py -k interpreter -v`
Expected: FAIL (hint lacks `sys.executable`).

- [ ] **Step 3: Implement**

```python
import sys

def sourmash_hint() -> str:
    """Install hint naming the interpreter this process runs under, so a user with several
    Python environments installs sourmash into the one metaquest actually uses."""
    return (
        "sourmash is required to sketch a genome. Install it into this interpreter with: "
        f"{sys.executable} -m pip install 'metaquest[sourmash]'"
    )
```

and `raise DataAccessError(sourmash_hint()) from e` at line 63. Update README.md if it quotes the old hint text (grep `metaquest[sourmash]`).

- [ ] **Step 4: Run tests, commit**

Run: `python -m pytest tests/test_data_branchwater_search.py -q`

```bash
git add metaquest/data/branchwater_search.py tests/test_data_branchwater_search.py README.md
git commit -m "fix: sourmash install hint names the running interpreter (audit S2-1)"
```

---

### Task 3: Journal keeps the true last-used date (review notes)

**Files:**
- Modify: `metaquest/store/catalog.py` (`record_usage` ~lines 355-410), `metaquest/store/journal.py` (`append_usage` ~60-80, `replay` ~150-165, `backfill_from_catalog` ~185-240)
- Test: `tests/test_store_journal.py`, `tests/test_store_catalog.py`

**Interfaces:**
- Produces: `Catalog.record_usage(..., at: Optional[str] = None, last_used: Optional[str] = None)`; `journal.append_usage(..., at=None, last_used=None)` writes a `last_used` key only when given; `replay` passes `record.get("last_used")`; `backfill_from_catalog` passes `at=row["first_used"]`, `last_used=row["last_used"]`.

- [ ] **Step 1: Write the failing tests** (tests/test_store_journal.py)

```python
def test_replay_out_of_order_keeps_latest_last_used_and_earliest_first_used(store_paths):
    with catalog_write(store_paths) as c:
        c.upsert_project("pid1", "proj", "/p", "/p/metaquest_registry.json")
    journal.append_usage(store_paths, "SRR1", "pid1", "", "linked", "", at="2026-09-02T00:00:00+00:00")
    journal.append_usage(store_paths, "SRR1", "pid1", "", "linked", "", at="2026-09-01T00:00:00+00:00")
    with catalog_write(store_paths) as c:
        journal.replay(store_paths, c)
        row = c.conn.execute("SELECT first_used, last_used FROM usage WHERE accession='SRR1'").fetchone()
    assert row["first_used"] == "2026-09-01T00:00:00+00:00"
    assert row["last_used"] == "2026-09-02T00:00:00+00:00"


def test_backfill_line_carries_first_and_last_used(store_paths):
    with catalog_write(store_paths, journal_enabled=False) as c:  # use whatever the existing backfill test does
        c.upsert_project("pid1", "proj", "/p", "/p/metaquest_registry.json")
        c.record_usage("SRR1", "pid1", "", "linked", "", at="2026-09-01T00:00:00+00:00")
        c.record_usage("SRR1", "pid1", "", "linked", "", at="2026-09-03T00:00:00+00:00")
        journal.backfill_from_catalog(store_paths, c)
    lines = [json.loads(line) for line in (store_paths.journal / "usage.jsonl").read_text().splitlines()]
    assert lines[-1]["at"] == "2026-09-01T00:00:00+00:00"
    assert lines[-1]["last_used"] == "2026-09-03T00:00:00+00:00"
```

Adapt fixture names to the ones tests/test_store_journal.py already uses (read its existing backfill and replay tests first; `first_used` must be set once, so the existing `record_usage` behaviour for `first_used` is unchanged).

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_store_journal.py -k "out_of_order or first_and_last" -v`
Expected: both FAIL (`last_used` is the later line's `at`, i.e. 09-01, and the backfill line has no `last_used`).

- [ ] **Step 3: Implement**

In `record_usage`:

```python
        now = at or _now()
        latest = last_used or now
        ...
            INSERT INTO usage (accession, project_id, genome_id, stage, first_used, last_used, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(accession, project_id, genome_id, stage) DO UPDATE SET
                first_used=min(first_used, excluded.first_used),
                last_used=max(last_used, excluded.last_used),
                detail=CASE WHEN excluded.last_used >= last_used THEN excluded.detail ELSE detail END
            """,
            (accession, project_id, genome_id, stage, now, latest, detail),
```

(ISO-8601 UTC strings compare correctly as text; `_now()` already emits them.) Pass `last_used=latest` only when `last_used` was given to `journal.append_usage`, so ordinary calls keep writing one `at`. In `append_usage`, add the key only when not None. In `replay`, pass `last_used=record.get("last_used")`. In `backfill_from_catalog`, pass `at=row["first_used"] or row["last_used"]`, `last_used=row["last_used"]`. Update the docstrings that describe `at`.

- [ ] **Step 4: Run tests, commit**

Run: `python -m pytest tests/test_store_journal.py tests/test_store_catalog.py tests/test_store_gc.py -q`

```bash
git add metaquest/store/catalog.py metaquest/store/journal.py tests/test_store_journal.py tests/test_store_catalog.py
git commit -m "fix: usage records keep the earliest first_used and latest last_used across replay and backfill (review note)"
```

---

### Task 4: CLI polish from the review notes

**Files:**
- Modify: `metaquest/cli/commands/store.py` (`_no_store_hint` line 64 and its callers in `store_status`, `store_usage`, `store_gc`), `metaquest/cli/commands/status.py` (`_reselect_command` ~269-315), `metaquest/cli/commands/select.py` (criteria dict ~lines 167-180), `metaquest/cli/commands/containment.py` (line 91), `metaquest/cli/commands/genome.py` (~line 342)
- Test: `tests/test_cli_store.py`, `tests/test_store_gc.py`, `tests/test_cli_status.py`, `tests/test_cli_select.py`, `tests/test_cli_containment.py` (or wherever `parse_containment`'s hint is tested), `tests/test_cli_genome.py`

- [ ] **Step 1: Failing tests, one per item**

(a) No store, JSON requested:
```python
def test_gc_json_without_store_prints_json_error(tmp_path, monkeypatch, capsys):
    # same isolation as test_no_store_configured_returns_1 in tests/test_store_gc.py
    args = ... json=True ...
    assert StoreGcCommand().execute(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"].startswith("No store configured")
```
Same shape for `store_status --json` and `store_usage --json` in tests/test_cli_store.py.

(b) Malformed threshold warns:
```python
def test_reselect_hint_warns_when_recorded_threshold_is_not_a_number(project, caplog):
    # record a selection, then edit the registry's criteria threshold to "abc"
    ...
    with caplog.at_level("WARNING"):
        run status --next
    assert "threshold" in caplog.text and "not a number" in caplog.text
    assert "--threshold 0.5" in out  # the default, DEFAULT_CONTAINMENT_THRESHOLD
```

(c) Metadata file recorded and reproduced:
```python
def test_selection_records_resolved_metadata_file_and_reselect_hint_repeats_it(project, tmp_path):
    # select_datasets --metadata-file meta.csv --metadata-column x --metadata-value y ...
    criteria = registry.datasets[acc]["selection"]["criteria"]
    assert criteria["metadata_file"] == str((project / "meta.csv").resolve())
    # status --next output contains "--metadata-file " + shlex.quote(that path)
```

(d) Next hint uses the alphabetically first genome column:
```python
def test_parse_containment_hint_names_the_first_genome_column_sorted(...):
    # summary.genome_to_samples = {"GCF_B": [...], "GCF_A": [...]}
    assert "--genome-id GCF_A" in caplog.text
```

(e) genome_prepare with no genome files opens no transaction:
```python
def test_prepare_with_no_genomes_leaves_registry_untouched(tmp_path):
    # empty genome folder; registry file absent before; after execute it must still be absent
    # (or, if the command always creates one, its 'updated' field unchanged)
```

- [ ] **Step 2: Run them, expect five FAILs**

- [ ] **Step 3: Implement**

(a) `_no_store_hint(as_json: bool = False)`: `print(json.dumps({"error": msg}))` when `as_json`, else the current text; callers pass `getattr(args, "json", False)`.
(b) In `_reselect_command`, when `_as_float`/`_as_positive_int` return None for a value that was present and not None, `logger.warning("Recorded %s %r is not a number; the suggested command uses the default", name, value)`; same for an unknown `require`.
(c) In select.py add `"metadata_file": str(Path(args.metadata_file).resolve()) if args.metadata_file else None` to the criteria dict; in `_reselect_command` emit `--metadata-file {shlex.quote(path)}` when the criteria hold one (next to the existing metadata column/value handling).
(d) `genome_id = next(iter(sorted(getattr(summary, "genome_to_samples", None) or {})), None)`.
(e) `if rows:` around the `registry_transaction` block in genome.py ~342.

- [ ] **Step 4: Run the touched test files, `make check`, commit**

```bash
git add metaquest/cli/commands/store.py metaquest/cli/commands/status.py metaquest/cli/commands/select.py metaquest/cli/commands/containment.py metaquest/cli/commands/genome.py tests/
git commit -m "fix: JSON error without a store, reselect hint warns on malformed values and repeats --metadata-file, sorted Next hint, no empty genome transaction (review notes)"
```

Update README.md / docs/pipeline_overview.md only if they list the `--json` error shape or the reselect command's flags.

---

### Task 5: Remaining test gaps and one parser tidy-up

**Files:**
- Modify: `metaquest/data/sra_metadata.py` (~lines 196-212)
- Test: `tests/test_store_gc.py`, `tests/test_store_adopt.py`, `tests/test_sra_metadata_extended.py`, `tests/test_security_comprehensive.py`

- [ ] **Step 1: Parser tidy-up (RED first)**

Make `_package_matches_requested` failures non-fatal per package: compute the match inside the per-package `try`, e.g.

```python
            matched = []
            for package in packages:
                try:
                    if requested_upper is None or self._package_matches_requested(package, requested_upper):
                        matched.append(package)
                except Exception as e:
                    logger.warning(f"Failed to inspect dataset package: {e}")
            if requested_upper is not None and packages and not matched:
                logger.warning("requested accessions matched no package in the reply; listing every run returned")
                matched = packages
            for package in matched:
                ...existing per-package extraction try...
```

RED test: monkeypatch `_package_matches_requested` to raise for one of two packages; the other package's runs are still returned and a WARNING is logged.

- [ ] **Step 2: Test gaps**

- tests/test_sra_metadata_extended.py: `test_parse_sra_xml_requested_sample_accession_keeps_whole_package` (request `SRS1`, the SAMPLE accession of the shared-package fixture) and `test_parse_sra_xml_requested_filter_with_package_without_run_set` (a package with EXPERIMENT SRX9 and no RUN_SET plus a normal package; request `SRX9`: one record keyed by SRX9, the other package dropped).
- tests/test_store_gc.py: extend `test_copy_adoption_records_copied_usage_so_gc_keeps_it` (or add a sibling) with a second project that links the same accession; assert `--dry-run --json` lists the accession under `in_use` (or the branch's equivalent key) with both project names, the copying one included. Add `test_gc_json_refuses_when_rebuilt_without_projects` asserting the JSON object's `error` mentions `--accept-rebuilt`.
- tests/test_store_adopt.py `test_matches_the_bytes_the_staging_copy_actually_produces`: add `src / "._cache" / "x.bin"` (an ignored directory) and keep the equality assertion.
- tests/test_security_comprehensive.py: delete the order-dependent leak test (the autouse fixture at line 22 is the guard); leave the minimap2/samtools tests as they are (accepted as documentation by the final reviewer).

- [ ] **Step 3: Run the four test files, `make test`, `make check`, commit**

```bash
git add metaquest/data/sra_metadata.py tests/test_sra_metadata_extended.py tests/test_store_gc.py tests/test_store_adopt.py tests/test_security_comprehensive.py
git commit -m "test: sample-level and no-RUN_SET filter cases, copied usage with a second project, gc --json refusal, ignored directory in byte count; package match failures are per package (review notes)"
```

---

## Verification after the last task

- `make check` and `make test` pass.
- On the sekvens2 reference data (`source /Volumes/sekvens2/metaquest-e2e/env.sh`, `PYTHONPATH=<worktree>`, run from `/Volumes/sekvens2/metaquest-e2e/crispatus` with `--data-root /Volumes/sekvens2/metaquest-e2e/store`): `store_reindex` then `store_usage --project crispatus --json` shows `last_used` dates from 2026-09-23/24, not the reindex time; `store_gc --json` with `METAQUEST_DATA` unset and no `--data-root` from a folder without a registry prints a JSON error; `status --next` still prints the runnable reselect line; `download_sra --accessions-file <one already-linked run>` prints one "Download summary:" block.
