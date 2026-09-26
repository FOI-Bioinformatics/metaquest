# Performance and Memory Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the eight speed and memory problems the 2026-09-26 code-quality audit measured on the real crispatus project (14,146 registry datasets, 16k-row containment table, 2,201 metadata XML files, one 11.3M-read run): per-accession registry transactions, a per-record Python FASTQ sampler, a full SAM of every read, row loops over the containment table, repeated FASTQ passes, double XML parsing, two quadratic status patterns, and serial uncached taxonomy lookups. Add performance tests with bounds so they cannot return.

**Architecture:** Every change keeps outputs identical (same files, same JSON keys, same numbers) unless a task says otherwise; the registry file becomes compact JSON (same keys, no indentation). Hot loops move to numpy or set operations; FASTQ work becomes single streaming passes over compressed bytes; subprocess intermediates shrink at the source (minimap2 emits only hits). One new module `metaquest/data/sra/sampling.py` holds the shared record sampler; one new test module holds bounded performance tests.

**Tech Stack:** Python 3.12, pandas and numpy, gzip/zlib streaming, samtools/minimap2 through `SecureSubprocess`, requests with `urllib3.util.Retry`, pytest with synthetic fixtures.

**Spec:** the code-quality audit of 2026-09-26 (in-session; measurements: `download_sra` about 1 s of registry load and dump per skipped accession; `sra_profile` 25 s vs a 3.5 s decompression floor; minimap2 `-a` without `--sam-hit-only`; `iterrows` in containment summary, screening, metadata, taxonomy; four reads of mate 1 after a store download; two lxml parses per XML; O(n^2) in `download_next_steps` and `stage_filter_accessions`; GTDB/NCBI without Session, retry or incremental cache). Merged state: `main` at 486cd39 (v0.5.0).

## Global Constraints

- Python >= 3.12; line length 120; `make check` (black, flake8 incl. B902 and D1xx, mypy, no-print, ASCII, module size and MI, docs-command gates) and `make test` green per commit; ASCII only; plain scientific wording; CLI flags use dashes.
- Tests on tmp_path; no tool on PATH except `sleep`; no network except `network`-marked tests; store tests use tmp_path stores.
- Every behavioural change has a RED test first; every speed change has a before/after timing in the task report on the synthetic fixture named in Task 9, and where possible on the sekvens2 crispatus project.
- Outputs must not change unless the task says so; where a task promises identical output, the test compares old and new results on a fixture.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add` AGENTS.md. All Python through the `metaquest` conda env (3.12).

## Review Focus

1. `download_sra` interrupted with Ctrl-C after some results: every result recorded before the interrupt is in the registry (batched writes flush on interrupt and at the end). (Task 1)
2. The index-based sampler returns exactly `min(sample_size, total)` records, deterministic for a seed, uniform over both mates, and identical to a brute-force selection on a small file; a file whose sidecar count is wrong (fewer records than claimed) still terminates with the records that exist. (Task 2)
3. With `--sam-hit-only`, `mapped_total`, the kept count after the 0x904 filter, and the coverage mapping rate are unchanged for a fixture where the fake minimap2 emits unmapped records. (Task 3)
4. A registry written compact (no indentation) is read by `load_registry` and by the 0.4.0 fixture round-trip test, and `git diff` of two saves differs only in changed values (keys still sorted). (Task 1)
5. `parse_metadata` after the single pass writes the same columns in the same order and the same values as before on the test_data XML fixtures. (Task 6)

---

### Task 1: One registry transaction per run in `download_sra`; compact registry JSON

**Files:**
- Modify: `metaquest/cli/commands/sra.py` (`_record_run_outcomes` ~L268-306, `_result_recorder` ~L335-360, `execute`), `metaquest/data/registry.py` (`_write_registry` ~L219-244, `registry_transaction`), `metaquest/cli/commands/read_extraction.py` (`_has_assembly_record` ~L362: pass the loaded registry instead of `load_registry` per sample)
- Test: `tests/test_cli_commands.py`, `tests/test_data_registry.py`, `tests/test_registry_blocks.py` (fixture round trip), `tests/test_performance_regressions.py` (Task 9 creates it; this task adds its first two tests)

**Interfaces:**
- Produces: `RegistryBatch` context in `metaquest/data/registry.py`: `with registry_batch(path, flush_every=50, flush_seconds=30.0) as batch: batch.apply(fn)` where `fn(registry)` mutates; the batch holds the lock only during a flush, flushes when `flush_every` mutations or `flush_seconds` elapsed, on exit, and in a `finally` on exception or `KeyboardInterrupt`. `_result_recorder` and `_record_run_outcomes` use it. `_write_registry` writes `json.dumps(payload, sort_keys=True, separators=(",", ":"))` plus a trailing newline.

- [ ] **Step 1: Failing tests**

```python
def test_run_outcomes_write_the_registry_once(tmp_path, monkeypatch):
    writes = []
    monkeypatch.setattr(registry_mod, "_write_registry", lambda reg, *a, **k: writes.append(1) or _real_write(reg, *a, **k))
    stats = {"already_downloaded_accessions": [f"SRR{i}" for i in range(300)], "blacklisted_accessions": ["SRR9"], "skipped_accessions": ["SRR8"]}
    DownloadSraCommand()._record_run_outcomes(args, stats, fastq_dir, store=None)
    assert len(writes) == 1


def test_result_recorder_flushes_on_interrupt(tmp_path):
    rec = DownloadSraCommand()._result_recorder(args, fastq_dir, None)
    with pytest.raises(KeyboardInterrupt):
        with registry_batch_of(rec) as batch:   # however the recorder exposes its batch
            rec("SRR1", True, "downloaded"); raise KeyboardInterrupt
    assert rb.download_block(load_registry(args.registry), "SRR1").state == "downloaded"


def test_registry_is_written_compact_and_reloads(tmp_path):
    save_registry(registry, path)
    text = path.read_text()
    assert "\n  " not in text and text.endswith("\n")
    assert load_registry(path).datasets == registry.datasets
```

- [ ] **Step 2: Run to verify failure** (`pytest tests/test_cli_commands.py -k "write_the_registry_once or flushes_on_interrupt" tests/test_data_registry.py -k compact -v`; expected FAIL: 302 writes; indentation present)
- [ ] **Step 3: Implement** `registry_batch` (one `registry_transaction` per flush applying the queued mutations), rewrite `_record_run_outcomes` as one batch (the `record_usage_many` call stays after the loop), make `_result_recorder` queue into a batch created in `execute` around the download call with `try/finally` flush (interrupt path included), pass the registry into `_has_assembly_record`, and switch `_write_registry` to compact JSON. Keep `sort_keys=True`.
- [ ] **Step 4: Measure** on a synthetic 20k-dataset registry: `_record_run_outcomes` with 5,000 skipped accessions before (seconds) and after (under 2 s); `_write_registry` time before and after. Record in the report; on sekvens2 run `download_sra --accessions-file candidates.txt --dry-run` and a real run on an all-present list and record wall times.
- [ ] **Step 5: `make check`, `make test`; commit** `perf: one registry transaction per download run, batched result writes, compact registry JSON`.

---

### Task 2: Index-based FASTQ sampler and numpy quality histograms

**Files:**
- Create: `metaquest/data/sra/sampling.py`
- Modify: `metaquest/sra/quality.py` (`_sample_uniform` ~L178-212, `_get_quality_distribution` ~L255-265, quality score accumulation), `metaquest/store/stats.py` (`_reservoir_sample` ~L60-80, `compute_dataset_stats` ~L260-290), `metaquest/data/sra/__init__.py` (re-export)
- Test: `tests/test_sra_quality.py` (or the existing quality tests), `tests/test_store_stats.py`, `tests/test_performance_regressions.py`

**Interfaces:**
- Produces: `sample_records(paths: Sequence[Path], sample_size: int, total_records: Optional[int] = None, seed: int = 0) -> List[Tuple[bytes, bytes]]` in `sampling.py`: if `total_records` is None, count with `count_fastq_reads` per file first (one binary pass); draw `k = min(sample_size, total)` sorted distinct indices with `random.Random(seed).sample(range(total), k)`; stream each file with `gzip.open(path, "rb")` (or plain open) reading 4 lines per record, advancing a record counter, decoding only selected records (`(seq, qual)` as bytes), stopping after the last index; if a file ends early, return what exists. `quality_histogram(quals: Iterable[bytes]) -> np.ndarray` (length 94, `np.bincount(np.frombuffer(q, np.uint8) - 33, minlength=94)` summed). `distribution_from_histogram(hist) -> Dict[str, float]` gives mean, median, percentiles, the q20/q30 fractions the current `_get_quality_distribution` reports.
- Consumes: `count_fastq_reads` (fastq.py), sidecar `reads_per_mate` via `store.stats.cached_stats` when present (pass as `total_records`).

- [ ] **Step 1: Failing tests**

```python
def test_sample_records_matches_brute_force_selection(tmp_path):
    path = write_fastq_gz(tmp_path / "r.fastq.gz", n=1000)          # helper writing records "r<i>" with seq f"A{i}"
    got = sample_records([path], 50, seed=3)
    idx = sorted(random.Random(3).sample(range(1000), 50))
    assert [s.decode() for s, _ in got] == [f"A{i}" for i in idx]


def test_sample_records_returns_all_when_fewer_than_sample_size(tmp_path): ...
def test_sample_records_spans_both_mates(tmp_path): ...            # indices past file 1 land in file 2
def test_sample_records_tolerates_overstated_total(tmp_path): ... # total_records=2000 for a 1000-record file returns <= 1000 without hanging
def test_quality_histogram_matches_list_based_distribution(): ...  # mean/median/q30 equal to the old implementation on a fixture within 1e-9
```

- [ ] **Step 2: Run to verify failure** (module missing)
- [ ] **Step 3: Implement**, then make `quality.py` `_sample_uniform` and `stats.py` `_reservoir_sample` call `sample_records` (stats passes the count it just computed; quality passes the cached count when `load_dataset_stats` has it) and keep their return shapes; accumulate qualities into a histogram instead of `quality_scores.extend(...)` and derive the distribution from it. Delete the reservoir loops.
- [ ] **Step 4: Measure** `sra_profile --sample-size 10000` on the sekvens2 run before (25 s) and after (target under 9 s, floor 3.5 s); peak RSS via `/usr/bin/time -l`. Performance test: a 100k-record gz fixture sampled to 10k in under 2.0 s (bound 3x the local time).
- [ ] **Step 5: `make check`, `make test`; commit** `perf: index-based FASTQ sampler shared by profiling and store statistics; quality scores as histograms`.

---

### Task 3: minimap2 emits only hits

**Files:**
- Modify: `metaquest/data/read_extraction.py` (`_minimap2_map_args` ~L856-861, `_count_records` ~L85, `_run_minimap2` ~L150-185, `assembly_coverage` ~L1055), `metaquest/core/constants.py` (`SAFE_PARAMETERS["minimap2"]`), `metaquest/utils/security.py` (`BOOLEAN_FLAGS["minimap2"]` gains `--sam-hit-only`)
- Test: `tests/test_read_extraction.py`, `tests/test_security_comprehensive.py`, `tests/helpers_extraction.py`

- [ ] **Step 1: Failing tests**: the builder emits `--sam-hit-only`; the validator accepts it (real `_build_validated_command`); with the fake minimap2 emitting two unmapped and three mapped records (one secondary), `mapped_total` and the kept count after the 0x904 filter equal the pre-change values (compute them once on the current code and pin them); `assembly_coverage` mapping rate unchanged for the same fixture.
- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement**: add the flag to the builder (both call sites share it) and to the allow-lists; audit how `mapped_total` is counted (`samtools view -c -F4` on the SAM: with hit-only output `-F4` is a no-op, keep it for safety) and that nothing derives the total read count from SAM record counts; update the fake tool to honour the flag.
- [ ] **Step 4: Measure** on the sekvens2 APFS copy (`hdiutil attach /Volumes/sekvens2/metaquest-e2e/mqapfs.sparsebundle`, project `/Volumes/mqapfs/crispatus`): `extract_target_reads --force` on SRR23946447 before and after; record SAM size and wall time; detach afterwards.
- [ ] **Step 5: `make check`, `make test`; commit** `perf: minimap2 writes only mapped records (--sam-hit-only)`.

---

### Task 4: Vectorised containment summary and screening

**Files:**
- Modify: `metaquest/data/branchwater.py` (`_generate_containment_summary` ~L405-460, `_process_genome_containments` ~L340-400: drop the per-row `Containment` objects and `details_rows` duplication where unused, or build them lazily), `metaquest/data/registry.py` (`record_screening_from_table` ~L400-445 takes a DataFrame, builds one screening dict per accession with one timestamp, caps per genome with `nlargest` before writing; `cap_screening` becomes a no-op check), `metaquest/cli/commands/containment.py` (~L79-85 passes the in-memory table)
- Test: `tests/test_data_branchwater.py`, `tests/test_data_registry.py`, `tests/test_cli_containment.py`, `tests/test_performance_regressions.py`

- [ ] **Step 1: Failing tests**: `_generate_containment_summary` output (`genome_to_samples`, `sample_to_genomes`, threshold counts, per-genome counts) equal to the current implementation on a 200x4 fixture with ties and zeros (pin the current output as expected); `record_screening_from_table` produces a registry equal to the current one on the same fixture apart from the `date` values, which must all be equal to each other within one run; perf: 20,000 x 5 summary plus screening under 1.5 s.
- [ ] **Step 2: Run to verify failure** (perf test and the equal-dates test fail today)
- [ ] **Step 3: Implement**: `values = df[genomes].to_numpy(float)`; `rows, cols = np.nonzero(values > 0)`; build the mappings from them; threshold counts with a sorted copy of `max_containment` and `np.searchsorted`; screening dicts from the same `rows, cols` grouped by row; `_now()` once per run.
- [ ] **Step 4: Measure** `parse_containment` on sekvens2 before (2.1 s) and after.
- [ ] **Step 5: `make check`, `make test`; commit** `perf: containment summary and screening built with numpy in one pass`.

---

### Task 5: One fused FASTQ pass after download and adoption

**Files:**
- Modify: `metaquest/data/sra/fastq.py` (new `fastq_digest(path) -> FastqDigest(records: int, md5: str, size: int)`: read compressed bytes once, feed md5 and a `zlib.decompressobj(16 + zlib.MAX_WBITS)` newline counter; plain files count directly), `metaquest/store/sidecar.py` (`build_sidecar` ~L131-155 uses digests, passes counts to `verify_download`), `metaquest/data/sra/fastq.py` `verify_download(..., reads_r1: Optional[int] = None, reads_orphan: Optional[int] = None)` skips the recount when given, `metaquest/data/sra/store_handoff.py` (~L180), `metaquest/data/sra/accession.py` (~L195), `metaquest/store/adopt.py` (`_content_matches` ~L138-159)
- Test: `tests/test_data_sra.py`, `tests/test_store_sidecar.py`, `tests/test_store_adopt.py`

- [ ] **Step 1: Failing tests**: `fastq_digest` equals `(count_fastq_reads(p), md5_file(p), p.stat().st_size)` for gz and plain fixtures; a spy on `gzip.open`/`open` shows `build_sidecar` reads each FASTQ exactly once; `verify_download` with precomputed counts opens no file.
- [ ] **Step 2-3: Run, implement.**
- [ ] **Step 4: Measure** `store_adopt --copy` of one crispatus run into a tmp store on sekvens2 before and after (wall time).
- [ ] **Step 5: `make check`, `make test`; commit** `perf: one streaming pass computes md5 and read counts after download and adoption`.

---

### Task 6: Single-pass metadata parsing

**Files:**
- Modify: `metaquest/data/metadata.py` (`parse_metadata` ~L601-674, `get_unique_sample_attributes` ~L689-699 folded into the main pass, `_extract_sample_attributes` uses a `set`, `_extract_metadata_fields` uses precompiled XPath or a single tree walk, `download_metadata` ~L265-268 one `os.listdir`), `metaquest/cli/commands/metadata.py` (`ParseMetadataCommand` ~L209-212 `itertuples`/records, `project_root` once; `DownloadMetadataCommand` ~L152-163 reuses the parsed fields instead of a third parse)
- Test: `tests/test_data_metadata.py`, `tests/test_cli_metadata.py`, `tests/test_performance_regressions.py`

- [ ] **Step 1: Failing tests**: the metadata table from the test_data XML fixtures is identical (columns, order, values) to the current output (pin it); a spy counts one `lxml.etree.parse` per file; perf: 300 synthetic XML with 40 attributes each parsed under 1.5 s.
- [ ] **Step 2-3: Run, implement** (collect fixed fields and attributes in one pass; build the DataFrame from sparse dicts with `columns=` fixed order: the 30 fixed fields then attributes sorted as before).
- [ ] **Step 4: Measure** `parse_metadata` on sekvens2 before (3.3 s) and after.
- [ ] **Step 5: `make check`, `make test`; commit** `perf: metadata XML parsed once per file; sparse attribute table`.

---

### Task 7: Status and results without quadratic passes

**Files:**
- Modify: `metaquest/cli/commands/status/suggest.py` (`download_next_steps` ~L153-156: compute `wanted = selected - excluded - downloaded` once, iterate `registry.datasets` and test membership), `metaquest/processing/status_report.py` (`stage_filter_accessions` ~L140-147 `dict.fromkeys`; `inventory_report` ~L102-105 membership against the listings already built instead of `accession_has_fastq` per accession; `_wanted_accessions` ~L56 `usecols=[0]`), `metaquest/processing/results.py` (`_row` ~L96-129: build the per-accession blocks once and reuse across genomes)
- Test: `tests/test_cli_status.py`, `tests/test_processing_results.py`, `tests/test_performance_regressions.py`

- [ ] **Step 1: Failing tests**: outputs equal to the current ones on a fixture (status report dict, results rows); perf: `build_report` on the 20k synthetic registry under 1.0 s and `results_rows` on 20k x 3 under 1.5 s (measure current first; the bounds must fail today on at least one).
- [ ] **Step 2-3: Run, implement.**
- [ ] **Step 4: Measure** `status --json` and `results_table` on sekvens2 before (0.7 s, 1.2 s) and after; also `/usr/bin/time -l` RSS.
- [ ] **Step 5: `make check`, `make test`; commit** `perf: status and results without quadratic passes or per-accession filesystem probes`.

---

### Task 8: Taxonomy lookups with a session, retries and incremental caches

**Files:**
- Modify: `metaquest/data/gtdb.py`, `metaquest/data/genome_taxonomy.py` (~L94-123 lookups, ~L208 cache write, `annotate_containment_with_taxonomy` ~L228-243), `metaquest/data/taxonomy.py` (`TaxonomyClient` ~L146-205, cache write ~L300-304), `metaquest/cli/commands/explore.py` (`enrich_taxonomy` ~L60 `nrows=0`, ~L65-68 single cache write)
- Test: `tests/test_gtdb.py`, `tests/test_genome_taxonomy.py`, `tests/test_taxonomy_extended.py`, `tests/test_cli_explore.py`

**Interfaces:**
- Produces: one `requests.Session` per client with `HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]))`; the TSV/CSV caches are appended after every successful lookup (open in append mode, header once) so an interrupted run keeps what it fetched; `annotate_containment_with_taxonomy` uses `df[genomes].melt(ignore_index=False)` filtered to `> 0` and maps taxonomy columns from a dict.

- [ ] **Step 1: Failing tests**: a mocked adapter shows one Session reused across lookups and a 503 retried; a lookup that raises after two successes leaves two cache rows on disk; the melt-based annotation equals the current output on a fixture with zeros (zeros excluded, as documented in the docstring; if the current output includes zero rows, pin the new behaviour and note it in the changelog).
- [ ] **Step 2-3: Run, implement.**
- [ ] **Step 4: `make check`, `make test`; commit** `perf: taxonomy clients reuse a session with retries and write their caches incrementally`.

---

### Task 9: Performance tests with bounds

**Files:**
- Create: `tests/test_performance_regressions.py`, `tests/perf_fixtures.py` (synthetic 20k-dataset registry builder with screening, selection, downloads, extractions; 100k-record FASTQ gz writer; 300 synthetic XML writer)
- Modify: `tests/test_performance_simple.py` (delete the benchmarks without thresholds and the memory tests that assert nothing; keep the edge-case tests), `pyproject.toml` (a `perf` marker, run by default), `CLAUDE.md` (testing section)

- [ ] **Step 1:** Fixtures first (Tasks 1, 2, 4, 6, 7 reference them; if this task runs after them, move their tests here). Each bound is 3x the local time measured after the fix and written in the test docstring with the date and machine; tests use `time.perf_counter()` and fail with the measured time in the message. Also one memory bound: `tracemalloc` peak of `load_registry` on the 20k fixture under 250 MB.
- [ ] **Step 2:** `make check`, `make test` (runtime of the new module under 20 s); commit `test: performance regression tests with bounds for registry, sampler, containment, metadata and status`.

---

### Task 10: Release 0.5.1

**Files:**
- Modify: `pyproject.toml`, `metaquest/__init__.py` (0.5.1), `CHANGELOG.md` (a "Performance" section with the before/after numbers from the task reports; a note that the registry file is now compact JSON with the same keys; a note if Task 8 changes the zero-row behaviour of `find_by_taxonomy`), `README.md`/`docs/pipeline_overview.md` where they mention sampling or intermediates

- [ ] **Step 1:** Write the changelog from the task reports; `make check`, `make test`, `make pipeline`; `python -m build` and check the wheel version; remove dist/ and build/. Commit `build: version 0.5.1 and changelog`. Tagging is the user's step at the finishing menu.

---

## Verification after the last task

- `make check`, `make test`, `make pipeline` green in the `metaquest` env; `tests/test_performance_regressions.py` runs in under 20 s.
- On sekvens2 crispatus (env `metaquest`, `PYTHONPATH` to the worktree): the before/after table for `status --json`, `results_table`, `select_datasets`, `parse_containment`, `parse_metadata`, `sra_profile --sample-size 10000` (target under 9 s), and `download_sra --accessions-file candidates.txt` on an all-present list (target under 10 s for 300 present accessions); the registry after one `status` differs from before only in `updated` and formatting; on the APFS copy, `extract_target_reads --force` on SRR23946447 writes a SAM smaller than 20 percent of the previous size and records the same `mapped_reads`.
