# Storage, Resume and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make stored datasets, assemblies and resumed downloads trustworthy (an assembly is redone when its reads or parameters changed, a partial store copy is never used as complete, one spot count decides completeness, reconcile re-verifies, gc and reindex recover from kills), then give the project a run history and a project-level report (release 0.8.0 for the correctness half, 0.9.0 for reporting).

**Architecture:** Optional fields on the existing registry blocks and store sidecar (schema versions unchanged); new writer and helper modules beside the frozen `registry.py` and `read_extraction.py` (`registry_assembly.py`, `data/sra/spots.py`, `metadata_fields.py`, `assembly_identity.py`, `cli/commands/extraction_assembly.py`, `cli/commands/sra_verdicts.py`); megahit output staged under a hidden name and published by rename with an inputs marker; an append-only run log under `<project>/.metaquest/runs/` with a `runs` command; a `project_report` command rendering Markdown always and HTML with the interactive extra; the dead report module removed.

**Tech Stack:** Python 3.12, dataclasses, `O_EXCL` lock files and the existing lockfile module, atomic writes from `data/file_io.py`, lxml and stdlib XML for spot counts, Jinja2 and Plotly for HTML (optional extra), pytest with fake tools on PATH.

**Spec:** the storage/assembly/resume audit and the report audit of 2026-10-01 against main 80d5891 (in-session; findings restated in the Context with file:line references verified against the code by the design pass).

## Review Focus

1. An assembly folder must never be reported as current when its reads or parameters differ from what built it, and an interrupted megahit must leave nothing a later run mistakes for finished output.
2. A `partial` or `failed` store copy must never be linked, extracted or assembled without an explicit `--accept-partial`/`--allow-truncated`, whatever the registry verdict says.
3. A `truncated` verdict must never be replaced by `unverified` by a relink, a reconcile or a redownload.
4. Old registries and sidecars must round-trip byte-identical; every new field is optional and omitted when unset.
5. The run log and `project_report` must never change a command's exit code or read FASTQ files, and must do nothing when no registry exists.

---

# Storage, resume and reporting plan (2026-10-01)

## Context

Two read-only audits of main at 80d5891 (v0.7.0) looked at how datasets and assemblies are stored, how an
interrupted or truncated download resumes, and what report outputs exist. The storage audit found that the data
layer is largely sound (atomic publishes under per-accession locks, verified sidecars in the store, a registry record
per download) but that several paths still mistake stale or partial data for finished work:

- An assembly is reused whenever `final.contigs.fa` exists, even after its reads were re-extracted with other
  parameters, and `record_extraction` carries the old assembly block forward (`data/assembly.py:167-172`,
  `registry.py:617`). An assembly folder without contigs (interrupted megahit) raises and ends the whole assembly
  loop; recovery needs a global `--force` or an undocumented `rm`, contrary to docs/hpc.md.
- The store publishes and links a `partial` copy; extraction skips only registry verdicts of `truncated`, so a partial
  store copy whose registry verdict is `unverified` is extracted and assembled (`store_handoff.py:182-204`).
- Expected spot counts come from two sources (registry metadata vs the XML the sidecar reads), so a store relink can
  overwrite a `truncated` verdict with `unverified`; reconcile never re-checks `unverified`; `status --init` records
  metadata with empty fields.
- `store_gc` never removes a bare `tmp/<ACC>` staging folder left by a kill; a partial copy whose NCBI count can never
  be met is refetched on every run; `--keep-sra` defeats `--force` and `--redownload-truncated`; after a SIGKILL a
  present accession is re-recorded with no verdict; the fasterq-dump scratch lives in the system temp dir.
- Smaller: copy-mode relink fails on a real folder; adopt publishes before writing the sidecar and a sidecar-less folder
  blocks reindex; no download-verdict column in results; `prefetch --max-size 100G` hard-coded.

The report audit found no single report: `sra_report` (HTML quality and group comparison), `sra_profile` (CSV and
JSON), `results_table` (TSV), `status` (text, JSON, TSV), `download_sra --report-file`, `doctor`, the store commands
and the containment outputs each cover one angle. Gaps: analyses and exports overwrite so runs cannot be compared;
`results_table` ignores `sra_report`'s numbers; `store_verify` has no `--json`; a finished PDF/HTML generator in
`visualization/reporting.py` has no command; no per-run download summary with reasons; no cross-stage funnel; scipy
is not pre-flighted for `sra_report --groups-file`; the coverage TSV is only discoverable by naming convention.

The user chose: two releases, storage first (0.8.0 correctness, 0.9.0 reporting); the append-only run log with a
`runs` command; remove the dead report module and add `project_report`; append new columns at the end. Execution:
subagent-driven in a worktree as before (plan committed first, ledger, implementer and reviewer per task, two
implementers at a time on disjoint files, whole-branch review per phase, one fix wave, gates, real-data checks on
sekvens2, finishing menu; merge, push and tag are the user's decisions).

## Global constraints

Python >= 3.12 only; no new runtime dependency (Markdown written by hand; HTML only with the `interactive` extra);
ASCII only in code, tests, scripts, config and Markdown; 120-column lines; module size guard (`registry.py` frozen at
1001 lines, `data/read_extraction.py` at 964 lines and MI 29.6, `cli/commands/sra.py` has 34 spare lines,
`cli/commands/read_extraction.py` 79: new code goes into new modules; D101-D103 and B902 enforced on every new
module, listed in setup.cfg's comment); atomic writes through `data/file_io.py` helpers (the gate enforces it);
tests on tmp_path with fake tools and fake data, never a real tool, store or network; CLI flags with dashes; registry
`SCHEMA_VERSION` 2 and sidecar schema 1 unchanged (optional fields only; old files round-trip byte-identical, pinned
by the existing round-trip tests); existing output column order kept (additions at the end); every behaviour change
named in the CHANGELOG under its release; plain scientific wording; commit trailer
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (check trailers before each branch review); run the suite
once with `PATH=/usr/bin:/bin` before merging.

Reuse, do not reinvent: `registry_blocks.py` dataclasses with `_optional()`, new writer modules beside
`registry_timing.py`, `registry_update`/`RegistryBatch`, `held_lock`/`dataset_lock`/`sample_extraction_lock`,
`extraction_staging.py` (hidden staging and per-folder record), `unique_temp_path`/`write_text_atomic`/`open_atomic`,
`verify_download` and `COMPLETE_RATIO_THRESHOLD` (`data/sra/fastq.py`), `STORE_READY_STATES` and `_store_precheck`
(`store_handoff.py`), `is_transient_folder` (`data/sra/cleanup.py`), `status_report.build_report` and
`stage_members`, `BaseCommand.emit/emit_json/fail`, `ExitCode`, `require()`, `utils/html.py` (`REPORT_CSS`,
`plotly_js_script`), the autoescaping Jinja pattern in `sra/reporting.py`, `settings.SETTINGS` (`_spec`) and
`RuntimeSettings`, `tests/helpers_extraction._fake_tools`, `tests/helpers_tools.fake_tool`, `tests/fake_http.py`.

Decisions taken (do not reopen): `STORE_READY_STATES` and sidecar state names unchanged (no `partial-stable`);
`COMPLETE_RATIO_THRESHOLD` and the result message format unchanged (`parse_verdict_message`, `STORE_SAVED_SUFFIX`
parse it); `is_transient_folder` does not match a bare `<ACC>` (in `fastq/` that is a dataset); `--force` keeps its
meaning for extraction; no `--force-assembly`, `--allow-partial`, `--refetch-archive` or `--reverify` flags; the
megahit version does not trigger a redo; an interrupted compression still publishes (the data is verified complete);
reconcile does not remove dangling links; no sweeping of the system temp folder; `record_analysis`/`record_export`
keep overwrite semantics (history lives in the run log); no `runs` list inside the registry; `classify_download_error`
unchanged (exit code 4 and the retry pass depend on it); `failed_accessions.txt` format unchanged; no web dashboard,
no PDF, no matplotlib in `project_report`; `project_report` reads no FASTQ.

---

# Phase 1: storage, assembly and resume (release 0.8.0)

Waves (disjoint files inside a wave): wave 0 = T1, T2, T3, T4, T11; wave 1 = T5 then T6 (same file), T7, T8, T9,
T10; wave 2 = T12 then T13 (same CLI file), T14, T15; wave 3 = T16. Each task lists its changelog lines in its
report; T16 merges them, so parallel tasks never edit CHANGELOG.md.

## Task 1: Registry: assembly inputs field; re-extraction drops the assembly

Files: `data/registry_blocks.py` (`AssemblyBlock.inputs: Optional[Dict[str, Any]] = _optional()`), `data/registry.py`
(`record_extraction` sets `assembly=None` instead of carrying the previous block; net lines <= 0; one docstring
sentence), new `data/registry_assembly.py`, tests (`test_registry_blocks.py`, `test_data_registry.py:631-633` changed
to assert `assembly is None`, new `test_registry_assembly.py`).
```python
def set_assembly_inputs(registry, accession, genome_id, inputs: Optional[Dict[str, Any]]) -> None  # None drops the key
def assembly_predates_extraction(registry, accession, genome_id) -> bool   # asm.date < ext.date; False if either missing
def legacy_assembly_current(registry, accession, genome_id, preset, min_contig_len) -> bool  # missing value = wildcard
```
Tests: old registry round-trips byte-identical; `inputs` omitted while None; legacy rule truth table; reconcile's
inferred extraction followed by `record_assembly` still records.

## Task 2: Spot count helpers; optional sidecar `refetch` field

Files: new `data/sra/spots.py`, `data/sra/fastq.py` (`verify_download` keeps its signature, its verdict branch calls
`verdict_for_count`), `store/sidecar.py` (`refetch: Optional[Dict[str, Any]] = None`, popped while None; schema 1),
`data/sra/__init__.py` exports, tests.
```python
def verdict_for_count(reads_r1: Optional[int], expected_spots: Optional[int]) -> Dict[str, Any]
def spots_from_xml(accession: str, folders: Iterable[Union[str, Path]]) -> Optional[int]   # parse_metadata_xml; OSError/ValueError/ParseError -> None
def expected_spots(registry, accession, store=None, xml_folders=()) -> Optional[int]
    # order: registry metadata.run_total_spots; previous verdict's expected_spots; sidecar ncbi.spots; XML in the folders
def merged_verdict(previous, new, reads_r1, expected) -> Optional[Dict[str, Any]]           # never truncated -> unverified
```
Keep the store import local to avoid an import cycle. Tests: threshold edges; each lookup step with XML on tmp_path;
`merged_verdict` never downgrades; old sidecar JSON byte-identical; `refetch` round-trips.

## Task 3: Metadata field mapping in the data layer

Files: new `data/metadata_fields.py` (`FIELD_COLUMNS`, `metadata_fields(row)`, `metadata_fields_from_xml(path)` ->
`{}` with a warning on a parse error, `fill_metadata_from_xml(registry, metadata_folder) -> List[str]` filling only
blocks whose `run_total_spots` is None and whose XML exists, keeping `inferred`), `cli/commands/metadata.py` imports
and aliases, tests with a minimal XML fixture.

## Task 4: Assembly staging and identity (data layer)

Files: `data/assembly.py`, new `data/assembly_identity.py`, `tests/test_read_extraction.py:734` rewritten, new
`tests/test_assembly_identity.py`.
```python
MARKER_NAME = ".metaquest-assembly.json"
@dataclass(frozen=True)
class AssemblyInputs: reads: Tuple[Tuple[str, int], ...]; extraction_date: Optional[str]; preset: Optional[str]
                      min_contig_len: Optional[int]; k_flags: Tuple[Tuple[str, int], ...] = ()
    # to_dict, from_dict -> Optional[AssemblyInputs], differences(other) -> List[str]
def assembly_inputs(reads, extraction_date, preset, min_contig_len, k_flags=None) -> AssemblyInputs  # names and sizes, not mtime
def read_marker(out_dir) -> Optional[Dict[str, Any]]; def write_marker(out_dir, inputs, version, params) -> Path
def assembly_state(out_dir, expected: Optional[AssemblyInputs], accept_unmarked: bool) -> Tuple[str, str]
    # "absent" | "current" | "stale" | "incomplete", reason
def sweep_staging(out_dir) -> List[Path]      # removes .<name>.*.tmp beside out_dir; caller holds the sample lock
def publish_assembly(staging, out_dir) -> None # aside via unique_temp_path, rename in, restore on failure, drop aside
```
`assemble_extracted_reads(..., *, expected=None, accept_unmarked=True, version="", params=None)`: megahit writes into
`unique_temp_path(out_dir)` (hidden, ignored by `scan_assemblies`/`visible_files`) and the result is published by
rename; a folder without contigs is a WARNING and a redo, not an error; `force` no longer removes the folder up front;
the marker is written into staging (and in place for an accepted unmarked folder); `expected=None` keeps the old
skip-on-contigs behaviour for existing callers; return value unchanged. Tests with the fake megahit: stale marker
redone, matching marker skipped, unmarked folder accepted or redone per `accept_unmarked`, a megahit failure leaves the
old folder intact and no staging, `scan_assemblies` ignores staging, a `tmp_dir` inside `out_dir` still refused.

## Task 5: Store handoff: completeness-aware publish and link, one spot source

Files: `data/sra/store_handoff.py`, `tests/test_data_sra.py` (store class around 2355-2480).
`_store_state(store, accession, expected_spots=None)`; `_store_precheck(..., expected_spots=None)` treats a `ready`
copy whose verdict is `unverified` but short against `expected_spots` (via `reads_per_mate`, no file read) as
incomplete and rewrites the sidecar under the lock (`_reverify_sidecar`); `_publish_decision(previous, new)` ->
"publish" | "keep_previous" (a refetch that comes back worse never replaces a better copy); `_store_fetch(...,
accept_partial=False)` fills `ncbi.spots` from the registry value when the XML has none, publishes a non-ready result
for resume but does not link it without `accept_partial`, unlinks an existing link, and returns
`(False, "incomplete: <ACC> store copy holds R of S spots; kept for --resume-partial; rerun with --accept-partial to
use it")` (classifies `unknown`, exit 1; check `_LOCK_MESSAGE_RE`/`_NETWORK_ERROR_RE` do not match). Tests: partial not
linked by default, linked with `--accept-partial`; complete previous copy kept; sidecar takes registry spots; short
`ready`/`unverified` copy refetched with the sidecar rewritten. Changelog: "Fixed: a partial store download is no
longer linked into the project without `--accept-partial`."

## Task 6: Store refetch limit (after T5)

`STORE_PARTIAL_REFETCH_LIMIT = 2`; `_refetch_record(previous, new)` and `_refetch_exhausted(sidecar)` using the
sidecar `refetch` field (`unchanged`, `reads_per_mate`, `expected_spots`, `last`); an exhausted copy without `force`
links with `--accept-partial` (note " (partial)") or returns `(False, "incomplete: N refetches of <ACC> gained no reads
(R of S spots); NCBI's count may not be reachable; use --accept-partial, or --force to fetch again")`; on
`keep_previous` the published sidecar's counter is updated under the lock. Tests: third identical partial fetch does
not download; `force` does; a gain resets; a complete result clears `refetch`.

## Task 7: Downloads: archive cache, scratch location, prefetch size

Files: `data/sra/accession.py`, `core/settings.py` (spec plus `RuntimeSettings` field), `core/constants.py`,
`docs/configuration.md`, tests. A redownload always wipes the cached archive (`accession.py:411`);
`_discard_cached_archive(cache_path, accession, message, keep=False)` keeps only complete or unverified archives under
`--keep-sra`, never a truncated one; `_project_download` uses `fastq/.metaquest-tmp/<ACC>_fqtmp` as fasterq-dump
scratch when no `--temp-folder` is given (wiped under the lock before the download and in `finally`; `tempfile.mkdtemp`
no longer called); `DEFAULT_PREFETCH_MAX_SIZE = "100G"` and the setting `prefetch_max_size`
(`METAQUEST_PREFETCH_MAX_SIZE`, `^\d+[KMGTkmgt]?$`) reaches the prefetch arguments. Changelog: three Changed/Added lines
(keep-sra semantics, scratch location with the advice to set `--temp-folder`/`METAQUEST_TEMP_FOLDER` for local
scratch, the new setting).

## Task 8: Reconcile: reverify, metadata from XML, unmounted store (needs T2, T3)

Files: `data/registry_reconcile.py`, tests. `ReconcilePlan.unavailable_links` (dangling link whose resolved target's
parent `sra/` does not exist); `StoreReconcileReport(ReconcileReport)` with `store_unavailable` (built here because
registry.py is frozen); such accessions are skipped by the missing loop with a warning; an untracked accession whose
previous block has `source == "store"` is re-recorded with `source` and `store_name` and `update_linked(add=True)`;
`_fill_metadata` (from `fill_metadata_from_xml`) runs before `_fill_verdicts`; `_fill_verdicts` also re-checks
`unverified` verdicts once a spot count is known: plain projects count mate 1 plus the orphan file once per download,
store links compare the sidecar's `reads_per_mate` (no file read); the full verdict dict is written. Tests: unverified
plain re-checked; store link re-checked without reading (patch `count_fastq_reads` to raise); XML fills metadata and
then the verdict in one run; unmounted store not marked missing; dataset gone from a mounted store marked missing;
re-record keeps `source`; scan/apply split holds. Document the one-time cost of the first reconcile.

## Task 9: gc removes staged dataset folders

`cli/commands/store/gc.py` `_leftover_candidates`: folders under `tmp/` matching the anchored
`SRA_ACCESSION_PATTERN` whose lock is not held, reason "leftover (staged download)"; no age filter (created under the
dataset lock, so the lock check suffices). Tests: `tmp/SRR1` listed and removed with `--yes`; kept while its lock is
live; other names in `tmp/` untouched.

## Task 10: Store small fixes: copy relink, adopt sidecar, reindex

`store/link.py` `_clear_existing(link, replace_store_copy=False) -> Optional[Path]`: in copy mode a real folder holding
`<ACC>.json` is moved aside after the staged copy completes and swapped in; a folder without the sidecar is still
refused. `store/adopt.py` `_write_staged_sidecar(accession, staged, metadata_folders) -> Sidecar` before the move;
`_finish_sidecar` stays for resuming an interrupted adoption. `cli/commands/store/reindex.py`: `_read_all_sidecars`
returns `(sidecars, unreadable, missing)`; a missing sidecar is rebuilt under a non-blocking `dataset_lock` by
`rebuild_missing_sidecar(accession, store_dir, paths)` (new in adopt.py, writes without cataloguing) before
`catalog_write`; a held lock means a warning and a skip; an unreadable sidecar is still refused. Tests for each.

## Task 11: results_table gains `download_verdict`

`processing/results.py`: append `download_verdict` at the end of `RESULTS_COLUMNS` (ruling: end, not after
`download_state`), filled from `rb.raw(registry, acc, "download", "complete")["verdict"]` when present. Update the
pinned column list and header tests.

## Task 12: CLI assembly: identity, per-sample failure, lock (needs T1, T4)

Files: new `cli/commands/extraction_assembly.py` taking `_assemble` and `_assembly_memory` out of
`cli/commands/read_extraction.py`; tests. `assemble_samples(command, args, with_reads, results, store) ->
AssemblyOutcome(assembled, reused, failed: Dict[str, str], busy)`: per sample load the registry once for the extraction
date; take `sample_extraction_lock` non-blocking (held -> `busy`, INFO); `sweep_staging`; build `expected`; set
`accept_unmarked = legacy_assembly_current(...) and not assembly_predates_extraction(...)`; clear the registry assembly
when stale or `--force`; call `assemble_extracted_reads` catching `ProcessingError` into `failed`; when megahit ran,
`record_assembly` + `set_assembly_inputs` + timing; when it did not and no record exists, record from the marker's
version and params. `execute` returns 1 when `failed` is non-empty after an ERROR line listing them. Tests: re-extraction
with a new `--min-mapq` then `--assemble` redoes; legacy record predating the extraction redone; contig-less folder
redone; sample 1 fails, sample 2 assembled, exit 1; busy lock skipped; stale staging swept; unchanged inputs do not run
megahit. Changelog: three lines (stale redo, interrupted megahit no longer blocks, per-sample failure).

## Task 13: CLI extraction check on download completeness (after T12)

`cli/commands/read_extraction.py`: `_truncated_downloads` becomes `_unusable_downloads(registry, fastq_folder) ->
Dict[str, Dict[str, Any]]` adding every in-folder sidecar in a non-ready state with a `reason` ("store copy partial (R
of S spots)", "store copy failed verification: <error>"); `--allow-truncated` covers them (no new flag);
`data/read_extraction.py` `_skip_if_truncated` rewritten in three lines so the message carries the reason (net -3
lines; MI stays >= 29.6); help text updated. Tests: partial store link with `unverified` registry verdict skipped;
`--allow-truncated` extracts it; failed-sidecar message; plain `unverified` not skipped.

## Task 14: download_sra CLI: spot fallback, no downgrade, SIGKILL verify (needs T2, T5)

New `cli/commands/sra_verdicts.py` (sra.py is near its ceiling): `registry_inputs(args, project_registry, store)`
moved here with the XML fallback (project then store metadata folder via `spots_from_xml`);
`present_verdicts(fastq_dir, accessions, registry, expected) -> Dict[str, Optional[dict]]` runs `verify_download`
(mate 1 only) for present non-store accessions without a `downloaded` record, outside the registry lock;
`linked_verdict(previous, sidecar, expected)` wraps `merged_verdict`. `sra.py`: `_record_run_outcomes` uses
`present_verdicts`; the recorder mutation calls `linked_verdict` inside the lock; `expected_spots` passed to
`_result_recorder`. Tests: SIGKILL-style present accession gets a `truncated` verdict; a store relink never turns
`truncated` into `unverified`; XML-only spots verify a plain download.

## Task 15: status CLI (needs T3, T8)

`cli/commands/status/command.py`: after `bootstrap_from_disk`, `fill_metadata_from_xml(registry, paths.metadata)`;
`render_text.py` prints `store_unavailable` as a WARNING line and `--json` carries the key. Tests: `--init` records
`run_total_spots` from the XML; reconcile with an unmounted store warns and changes no records.

## Task 16: Docs, changelog and release 0.8.0

README (store completeness paragraph incl. reverification on reconcile; `--keep-sra`; `--redownload-truncated`;
prefetch size; extraction and assembly identity, resume, per-sample failure, `--allow-truncated` scope; results
column), docs/hpc.md (assembly resume and staging; scratch under `.metaquest-tmp`; first-reconcile cost),
docs/configuration.md (`prefetch_max_size`), docs/ARCHITECTURE.md (assembly marker and staging; spot lookup order),
CHANGELOG `## [0.8.0]` with one line per behaviour change from T1-T15, version 0.8.0 in pyproject.toml and
`metaquest/__init__.py`; `make check`, `make test`, `make pipeline`, `python -m build`, no tag.

---

# Phase 2: reporting (release 0.9.0)

Waves: wave 1 = T17, T19 (with T22 folded in), T21, T23, T26; wave 2 = T18 (needs T17), then T20 and T24 one after
the other (both touch `status_report.py`); wave 3 = T25 (needs T23's `failure_reason`, T24's `funnel`, T17's
`read_runs`); wave 4 = T27, then T28. Each task adds its own bullet under `## [Unreleased]`; T28 consolidates.

## Task 17: Run log core

Files: new `data/run_log.py`; `cli/main.py` (+25: `_record_run_if_wanted(parsed_args, argv, started, clock,
exit_code)` in a `finally` around `parsed_args.func(...)`, never changing the exit code; catches OSError,
DataAccessError, ValueError, TypeError with a warning), `cli/base.py` (`BaseCommand.records_run(self, args) -> bool`,
False by default), `core/settings.py` (`run_log` bool, `METAQUEST_RUN_LOG`, default true), `tests/conftest.py` (autouse
sets `METAQUEST_RUN_LOG=false`; run-log tests turn it on).
Layout under `<project>/.metaquest/runs/` (project = the registry file's parent; written only when the registry file
exists): `runs.jsonl` appended under `held_lock(runs.jsonl.lock, LockPolicy("Run log", stale_seconds=60,
wait_seconds=10))`, a partial last line skipped with one warning; `<run_id>.json` detail files written with
`write_text_atomic`, the last `DETAILS_KEPT_PER_COMMAND = 10` per command kept (pruned runs keep their line with
`"detail": null`).
```python
RUN_LOG_SCHEMA = 1
@dataclass class RunRecord: run_id, command, started, finished, seconds, exit_code, argv, args, version, host, pid,
                            summary: Dict[str, Any], detail: Optional[str], schema: int = 1   # to_dict / from_dict
def runs_dir(project) -> Path; def new_run_id(command, started) -> str      # "20261001T120501Z-sra_profile-3fa2"
def json_safe_args(args) -> Dict[str, Any]                                   # drops "_*" and "func"; masks secrets
def note_run(args, summary=None, detail=None) -> None                        # merges into args._run_summary / _run_detail
def record_run(project, command, argv, args, started, seconds, exit_code) -> Optional[RunRecord]
def read_runs(project, command=None) -> List[RunRecord]; def read_detail(project, record) -> Optional[Dict[str, Any]]
def resolve_run(records, selector) -> RunRecord                              # id, unique prefix, "latest", "previous"
```
`argv` masked with the existing argv masking from `main`. Tests: round trip; malformed last line; pruning; selectors;
API key masked; no registry means no folder; a non-opting command writes nothing; an unwritable log keeps the exit code;
the setting off writes nothing. Docs: README "Project state", docs/configuration.md row.

## Task 18: `runs` command (needs T17)

New `cli/commands/runs.py` and `processing/run_diff.py`; registered in the Environment group. `metaquest runs
[--command NAME] [--limit 20] [--show RUN] [--diff RUN_A RUN_B] [--accession ACC] [--registry] [--json]`.
```python
def diff_summaries(a, b) -> List[Dict[str, Any]]            # key, before, after, delta for numbers
def diff_details(a, b) -> Dict[str, Any]                     # added, removed, changed{acc:{field:[old,new]}}
def accession_history(project, records, accession) -> List[Dict[str, Any]]
```
Exit 0; 1 with no run log; `ValidationError` through `fail`. Tests: two fake `sra_profile` runs with one changed
`gc_percent`; `--accession`; `--json` one document; pruned detail reported as not kept. README "Run history".

## Task 19: Quality columns fall back to the "report" analysis (T22 folded in)

`data/registry_blocks.py`: `quality_summary(registry, accession) -> Tuple[Dict[str, Any], Optional[str]]` (source
"profile" | "report" | "legacy" | None): whichever of "profile" and "report" is newer by `datetime.fromisoformat`
wins (unparsable counts as oldest; equal dates favour "profile"), None fields filled from the other, legacy analyses
only when neither exists; `profile_summary` returns `quality_summary(...)[0]`. `cli/commands/sra_report.py`: the
"report" summary adds `total_reads` and `total_bases`; with `--groups-file`, `require("scipy.stats", "analysis",
"Comparing dataset groups (--groups-file)")` before `load_groups`, whether or not `--no-report` is given (T22).
Tests: only "report" recorded fills gc and grade; newer "report" wins and fills None from "profile"; equal dates
favour "profile"; legacy unchanged; fixture round trip byte-identical; without scipy `--groups-file` exits 3 with no
output and the profiler never called. Changelog: behaviour change (runs that used to succeed with every test skipped
now fail at once without scipy).

## Task 20: results_table and `--export-tsv` columns (after T19)

`processing/results.py`: append after `download_verdict` (from Phase 1): `quality_source`, `assembly_largest`,
`assembly_n90`, `assembly_gc_percent` (gc x 100, two decimals), `assembly_contigs_ge_1kb`,
`assembly_mean_depth_estimate`, `assembly_dir`, `coverage_tsv` (project-relative); `_dataset_fields` uses
`quality_summary`. `processing/status_report.py` `to_dataframes`: extractions TSV appends `coverage_tsv`, `n90`,
`largest`, `assembly_dir`. Tests: pinned column list; a fake assembly `extra` (gc 0.4123 -> 41.23); missing assembly
gives empty cells; project-relative path; header check.

## Task 21: `store_verify --json`

`cli/commands/store/verify.py`: `_verify_one` keeps `state_before`; `_mark_failed`, `_fix_state` and
`_fix_state_locked` set `result["fix"] = {"action": "updated" | "unchanged" | "skipped-in-use" | "skipped-changed",
"state_before", "state_after", "error"}`; `_json_row(result)` drops the `Sidecar` and `ncbi_found` objects; document
`{"root", "checks": {md5, spots, rescan, fix_state}, "counts": {verdict: n}, "datasets": [...], "fixed": [...]}` via
`emit_json`; no store -> `_no_store_hint(args.json)`; text mode with `--fix-state` prints one line per change. Tests
on a fake store: size mismatch row "corrupt" and one JSON document; `--fix-state --json` before/after states; lock held
elsewhere -> "skipped-in-use"; no store with `--json` -> error document, exit 1.

## Task 23: Per-run download summary

New `data/sra/run_report.py`; `cli/commands/sra.py` moves `_write_report` out (file ends under 790 lines) and its
`--report-file` help changes.
```python
REASONS = ("network", "not-found", "disk-full", "insufficient-space", "locked", "interrupted", "unknown")
REPORT_HEADER = ("accession", "status", "message", "seconds", "reason", "attempts"); DOWNLOAD_RUN_FILE = "download_run.json"
def failure_reason(message) -> str     # interrupted / insufficient-space / locked first, then classify_download_error
class RunOutcomes: wrap(on_result) -> on_result; observe(accession, success, message); attempts: Dict[str, int]; last
def report_rows(stats, timings, attempts) -> List[Tuple[str, ...]]; def stats_from_outcomes(outcomes) -> Dict[str, Any]
def write_report_csv(path, rows) -> None; def run_document(*, stats, outcomes, started, finished, exit_code, aborted,
                                                           settings, paths) -> Dict[str, Any]
def write_run_document(fastq_dir, document) -> Path     # <fastq>/download_run.json, always except dry runs
```
Attempts count outcomes that started a download (not "not attempted", interrupted, insufficient-space, already
exists, store links). The document holds totals, `failures_by_reason`, `failed` rows (accession, reason, attempts,
message), `aborted`, exit code, host, settings (workers, threads, retries, max_downloads, force, verify, redownload,
prefetch, compress, min_free_gb, link_mode) and paths. `execute` writes both files after a normal finish, an abort, a
flush failure, and on `KeyboardInterrupt` (`aborted="interrupted"`, exit 130); dry runs write neither. Tests: one per
reason; a retried network failure counts 2 attempts; disk-full "not attempted" rows count 0; interrupt mid-run writes
both files and exits 130; flush failure still writes; first four CSV columns unchanged; dry run writes nothing.
Changelog: new CSV columns at the end, `download_run.json`, report written on interrupt too.

## Task 24: Stage funnel in status (after T20)

New `processing/project_funnel.py` `funnel(registry, members=None) -> Dict[str, Any]`: ordered stages screened,
selected (with excluded), downloaded (accessions, bytes = sum of `bytes_total`, seconds, failed), analysed, extracted
(accessions, pairs, seconds), assembled (accessions, pairs, total_bp, seconds), counts from `stage_members`, one pass
over the datasets; `status_report.build_report` adds `funnel`; `render_text.py` prints one line (`funnel: 1200 screened,
300 selected, 280 downloaded (1.2 TB, 41 h), 250 extracted, 90 assembled`; commas, not arrows). Tests: counts equal
`stage_members`; bytes summed; untimed datasets stay None; JSON key present; existing text lines unchanged.

## Task 25: `project_report` command (needs T17, T23, T24)

New `processing/project_report.py` (`build_project_report(registry, *, max_rows=200, include_environment=True,
runs_limit=10) -> Dict[str, Any]`), `processing/project_report_markdown.py` (`render_markdown(report) -> str`, cells
escape `|` and newlines), `visualization/project_report.py` (`render_html(report) -> str` through `require()` for
plotly and jinja2, autoescape, `REPORT_CSS`, `plotly_js_script`), `cli/commands/project_report.py` registered in the
Environment group. Sections: project (registry path, updated, version), funnel, genomes (extracted, assembled,
zero_mapped, median breadth and depth), extractions (rows sorted by genome then mapped_reads, cut to `max_rows` with
`rows_total` and a pointer to `results_table`), downloads (verdict counts, truncated and unverified lists), failures
(state failed with `failure_reason`, attempts, date), timing (`timing_summary` plus min, p25, median, p75, p90, max per
kind), environment (`run_checks(network=False)` status, counts, non-ok checks), outputs (`project.exports` and each
analysis name's outputs), runs (last N run-log lines). CLI: `--output-dir project_report`, `--html {auto,always,never}`
(auto: a missing extra logs one INFO line naming `metaquest[interactive]`; always: exit 3), `--max-rows`,
`--no-environment`, `--no-record`, `--registry`, `--json`. Always writes `project_report.md` and `project_report.json`,
HTML when available; records export `project_report`; no registry -> exit 1 pointing to `status --init`. Tests: fake
registry gives the expected sections; truncation note at `max_rows`; without jinja2 auto writes md and json only and
always exits 3 writing nothing; with plotly the HTML has every section id; `--no-environment` skips `run_checks`; no
FASTQ opened.

## Task 26: Remove the dead report code

Delete `metaquest/visualization/reporting.py`, `metaquest/visualization/templates/report_template.html`, the three
`tests/test_visualization_reporting_*.py`, `test_html_report_without_jinja2_is_an_error` in
`tests/test_optional_imports.py`, its `KNOWN_EXCEPTIONS` entry, its line in `scripts/atomic_writes_allowlist.txt` and
`setup.cfg:92`; check coverage `omit` and package-data first. Changelog Removed line.

## Task 27: Wire the run log into commands (needs T17-T25)

`records_run` and `note_run` in: download_sra (summary from `run_document`), sra_profile (detail
`{"analyses": {"profile": {...}}}`), sra_report ("report" detail), results_table (export summary; respects
`--no-record`), extract_target_reads (detail rows keyed `acc/genome`: mapped_reads, breadth, mean_depth), store_verify
(counts and fixed), project_report, select_datasets, blacklist, status (only with `--init` or `--reconcile`). Read-only
invocations (plain status, doctor, runs) record nothing. One CLI test per wired command through `main([...])` in a tmp
project with the run log on.

## Task 28: Docs, changelog and release 0.9.0

README mentions `runs` and `project_report` (docs-command check), the extras table says `project_report` HTML needs
`interactive`, "Run history" and "Project report" sections, results columns, `store_verify --json`, download run
summary, funnel line; docs/configuration.md `run_log`; docs/ARCHITECTURE.md run log and report modules; CHANGELOG
`## [0.9.0]` consolidated from T17-T27; version 0.9.0; `make check`, `make test`, `make pipeline`, `python -m build`,
no tag.

---

## Verification

Per task: `conda run -n metaquest make check` and the affected tests; per phase: `make check`, `make test`,
`make pipeline`, `python -m build`, the suite once with `PATH=/usr/bin:/bin` (CI has no tools), CI matrix green on
push, nightly smoke green.

Real data (sekvens2 crispatus project, env `metaquest`, PYTHONPATH to the worktree, scratch registry copies only):
- Phase 1: `status --reconcile` on a scratch copy re-verifies the `unverified` downloads that have a spot count and
  reports the count re-checked; `status --json`, `results_table` (minus the new column) and `parse_containment`
  byte-identical to 0.7.0; `store_gc --json` on the real store unchanged (no new candidates besides any genuine bare
  `tmp/<ACC>` leftover, which is reported, not removed); on the APFS copy, `extract_target_reads --force` then
  `--assemble` on SRR23946447 writes the assembly marker, a rerun skips megahit, a rerun after re-extraction with a
  different `--min-mapq` redoes it, and a folder emptied of contigs is redone with a warning; `download_sra` of one
  present accession with `--keep-sra --force` wipes and refetches the archive on a scratch registry (network; skip if
  offline and say so).
- Phase 2: `project_report --output-dir <scratch>` renders Markdown, JSON and HTML in the env; `runs` lists the runs the
  checks produced and `runs --diff latest previous` shows the expected change; `results_table` adds only the new
  trailing columns; `store_verify --json` on the real store parses as one document; `status` prints the funnel line
  with the project's known counts.

## Not in scope

Changes to `STORE_READY_STATES`, sidecar or registry schema versions, `COMPLETE_RATIO_THRESHOLD`, result message
formats, `classify_download_error`, `failed_accessions.txt`; new flags beyond those named; megahit-version-triggered
redo; sweeping the system temp folder; removing dangling links in reconcile; a `runs` list inside the registry; a web
dashboard, PDF output or matplotlib in `project_report`; `--json` on every command; combining coverage TSVs; reading
FASTQ in `project_report`.
