# Foundation plan from the gap analysis (2026-09-24)

## Context

Three fix rounds this month closed the audit findings; main is at ecbc16b, 2103 tests, CI green. The gap
analysis (session of 2026-09-24) found that the remaining weaknesses are foundational rather than defects:
the code still carries Python 3.11 fallbacks while `pyproject.toml` already says `requires-python >= 3.12`;
the developer's editable install sits in a 3.11 base interpreter, so local runs never exercise the 3.12 path
CI uses (today's CI failure came from that gap); dependency floors predate 3.12; CI tests one interpreter on
Linux only; the pipeline has no consolidated per-genome, per-run results table and no breadth of coverage;
selection cannot budget by run size or spot count; and there has never been a tagged release.

The user chose: Python floor 3.12 with a refreshed conda environment ("use modern python"), and
"foundation first" scope. Larger items (multi-genome extraction pass, dependency trimming, SRA command
merge) go to a follow-up plan.

Execution: subagent-driven in a worktree branched from main, one task per dispatch, reviews as in the
previous rounds, whole-branch review, then the finishing menu. Global constraints as before: line length
120, ASCII only, dashes in flags, `make check` and `make test` green per commit, tests on tmp_path with no
tools on PATH, plain scientific wording, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

### Task 1: Python 3.12 floor, modern dependency floors, refreshed conda environment

Files: `pyproject.toml`, `requirements.txt`, `environment.yml`, `metaquest/data/sra.py`, `Makefile`,
`README.md`, `CONTRIBUTING.md`.

- `metaquest/data/sra.py` L154-176 (`_safe_rmtree`): drop the `sys.version_info` branch and the
  `onerror` path; keep `shutil.rmtree(path, onexc=_ignore_missing)` and simplify `_ignore_missing` to take
  the exception object only (the tuple shape existed for 3.11). Remove the now-unused `sys` import if
  nothing else uses it. Update the docstring. Tests in `tests/test_data_sra.py::TestSafeRmtreeIgnoresMissingFiles`
  keep passing; add one asserting that `_ignore_missing` re-raises a `PermissionError` given directly.
- `pyproject.toml`: keep `requires-python = ">=3.12"`; raise runtime floors to the first releases with 3.12
  wheels: pandas>=2.1, numpy>=1.26, scipy>=1.11, scikit-learn>=1.3, matplotlib>=3.8, seaborn>=0.13,
  biopython>=1.82, lxml>=4.9.3, statsmodels>=0.14, networkx>=3.2, umap-learn>=0.5.5, plotly>=5.18,
  jinja2>=3.1, requests>=2.31, upsetplot>=0.9; extras cartopy>=0.22, sourmash>=4.8. Dev floors:
  black>=24.0, mypy>=1.8, flake8>=7.0, pytest>=8.0, pytest-cov>=5.0, radon>=6.0. Keep black
  `target-version = ['py312']` and mypy `python_version = "3.12"`; add classifier `3.13` (already there).
  Mirror the same floors in `requirements.txt`.
- `environment.yml`: pin `python=3.12` (exact minor, not `>=`, so `conda env create` cannot pick 3.14
  where umap/numba wheels lag), keep the tool list, add `pip` extras `-e ".[dev]"` variant? No: keep
  runtime `-e .` and document `pip install -e ".[dev]"` after activation. Add `seqkit` uncommented as
  optional? Keep as is.
- `Makefile`: add `env` target (`conda env create -f environment.yml || conda env update -f environment.yml --prune`)
  and `env-dev` (`conda run -n metaquest pip install -e ".[dev]"`); make `test`, `lint`, `check`, `pipeline`
  use `python -m pytest` / `python -m flake8` / `python -m black` so they run under whichever interpreter is
  active rather than a stale console script; print the interpreter version at the start of `check`.
- Developer environment (documented in README "Development" section, done by the user, not by the plan):
  `conda env update -n metaquest -f environment.yml --prune` (the existing `metaquest` env is 3.13.2; the
  plan pins 3.12, so `conda env remove -n metaquest && make env`), then `make env-dev`, `conda activate
  metaquest`, and `pip uninstall metaquest` from the 3.11 base so `metaquest` on PATH is the env's.
- README/CONTRIBUTING: "Python 3.12 or newer"; replace the stale test count lines (README L13, L56, L786:
  "1280 tests, 92% coverage") with wording that carries no number; CLAUDE.md status lines (L165, L192,
  "995 tests", "October 2025") likewise.

Verification: `make check`, `make test` under the 3.12 env; `python -c "import metaquest"` under 3.13 in
the `claude` env as a smoke check.

### Task 2: CI matrix across interpreters and macOS

Files: `.github/workflows/ci.yml`.

- `test` job: `matrix: os: [ubuntu-latest, macos-latest], python-version: ["3.12", "3.13"]`,
  `runs-on: ${{ matrix.os }}`, `fail-fast: false`, name "Test ${{ matrix.os }} / Python ${{ matrix.python-version }}".
  On macOS the interrupt end-to-end test and the store tests run for real; `sleep` exists on both.
- `lint` job: re-enable the mypy step (`mypy metaquest`, the local `make check` already passes it).
- Keep `build` and `integration` on ubuntu 3.12; add a `python-version: "3.13"` install smoke to
  `build` ("Test installation" step under both interpreters) only if cheap; otherwise leave.
- Add a workflow-level `concurrency` group cancelling superseded runs on the same ref.

Verification: push the branch, all matrix legs green.

### Task 3: Breadth of coverage against the reference genome during extraction

Files: `metaquest/data/read_extraction.py`, `metaquest/data/registry.py`, `metaquest/cli/commands/read_extraction.py`,
`metaquest/core/constants.py`, `tests/helpers_extraction.py`, `tests/test_read_extraction.py`,
`tests/test_cli_read_extraction.py`, `tests/test_data_registry.py`, `tests/test_security_comprehensive.py`, docs.

samtools facts: `samtools coverage` needs coordinate-sorted input (no index without `-r`); columns
`#rname startpos endpos numreads covbases coverage meandepth ...`; no minimum-depth option, so record breadth
at >= 1x and mean depth only. `samtools sort -o X` needs no `-T`.

- `read_extraction.py`: `ExtractionResult` gains `coverage: Optional[Dict[str, Any]] = None` (keys `breadth`,
  `mean_depth`, `covered_bases`, `reference_bp`, `coverage_tsv`). New builders beside the existing ones:
  `_samtools_sort_args(threads, out_path, in_path)` -> `["sort", "-@", threads, "-o", out, in]` and
  `_samtools_coverage_args(out_path, bam_path)` -> `["coverage", "-o", out, bam]`. New
  `summarise_coverage_table(path)` (pandas; `length = endpos - startpos + 1`; breadth = sum covbases / sum length;
  mean_depth = length-weighted meandepth; 4 dp; None when reference_bp is 0). New
  `reference_coverage(accession, genome_id, bam_path, out_dir, sam_root, threads)`: sorted BAM at
  `sam_root/<genome>.mapped.sorted.bam` (same placement as the SAMs, so `--temp-folder` applies and the root is
  already allow-listed), TSV at `out_dir/<genome>_coverage.tsv`; sort then coverage through `run_secure`;
  on `CalledProcessError`, `SecurityError`, `DataAccessError`, `OSError`, `ValueError` log
  `"%s: reference coverage skipped (%s)"`, remove a partial TSV, return None; `finally` unlink the sorted BAM.
  Call it in `_map_and_extract` between the FASTQ export (~L470) and the BAM deletion (~L473); both zero-mapped
  early returns precede it, so it never runs on zero reads. Keep helpers out of `_map_and_extract` (complexity 15).
- `registry.py`: `record_extraction(..., coverage=None)` writes `breadth`, `mean_depth` (None when absent) and a
  project-relative `coverage_tsv`; `to_dataframes` gains `breadth` and `mean_depth` columns.
- `cli/commands/read_extraction.py` `_record_result`: pass `coverage=outcome.coverage`; usage detail
  `"<n> mapped reads, breadth <b:.3f>"` when known.
- `constants.py` samtools `safe_params`: add `"sort"`, `"coverage"` (subcommands; `-o`, `-@` already allowed and
  `-o` is path-validated).
- Tests: extend `_fake_tools` in `tests/helpers_extraction.py` to serve `sort` (write the `-o` file) and `coverage`
  (write `state["coverage_rows"]` or a two-contig default; `state["coverage_fail"]` raises). New
  `TestReferenceCoverage`: aggregation (100 bp/50 covered/depth 2 + 300 bp/300 covered/depth 10 -> breadth 0.875,
  mean depth 8.0); TSV written and record carries breadth, no `*.sorted.bam` left, sort precedes coverage in
  `state["calls"]`; zero-mapped sample runs neither; failure is a WARNING and the FASTQ still lands; sorted BAM
  lands under `--temp-folder`. CLI test: registry record has breadth, mean_depth and a relative `coverage_tsv`.
  Registry tests for the new keys and their defaults; `to_dataframes` asserts the breadth column. Security test:
  add the two builders' outputs to the samtools validator test.
- Docs: one sentence in README "Project state"/extraction section and docs/pipeline_overview.md step 5.

### Task 4: `results_table` command, the study's consolidated output

Files: new `metaquest/processing/results.py`, new `metaquest/cli/commands/results.py`,
`metaquest/data/registry.py`, `metaquest/cli/commands/__init__.py`, `metaquest/cli/main.py`, new
`tests/test_processing_results.py`, new `tests/test_cli_results.py`, `tests/test_data_registry.py`, docs.

A dedicated command rather than extending `status --export-tsv` (that is a registry dump; this joins registry,
parsed containment table and metadata with its own filter and ordering).

- `processing/results.py`: `RESULTS_COLUMNS` in this order: accession, genome_id, containment, selected, excluded,
  exclusion_reason, download_state, run_total_spots, run_size, mapped_reads, mapping_rate_to_reference, breadth,
  mean_depth, contigs, total_bp, n50, genome_fraction_estimate, assembly_mapping_rate.
  `screened_pairs(registry, parsed_table)`: every (accession, genome) from `screening.genomes` (value
  `containment`), every genome column of the parsed table except `max_containment*` with a value > 0 (table wins,
  it is unrounded and not capped by `cap_screening`), plus pairs with an extraction record but no screening
  (containment None). `results_rows(registry, parsed_table=None, genome_id=None, min_containment=0.0)`: one dict
  per pair; `run_total_spots`/`run_size` via `to_int_or_none` (public alias of registry's `_to_int_or_none`);
  `mapping_rate_to_reference = mapped_reads / run_total_spots` when both > 0 (docstring: BAM records over spots,
  paired data can exceed 1; no silent halving); breadth/mean_depth from Task 3 keys, empty when absent; assembly
  fields from `extraction["assembly"]`. Sort key `(containment is None, -(containment or 0), accession, genome_id)`.
  `results_dataframe(rows)` with `columns=RESULTS_COLUMNS` (header-only when empty).
- `cli/commands/results.py` `ResultsTableCommand(BaseCommand)`, name `results_table`, group "Reads". Flags:
  `--output` (default `results.tsv`), `--genome-id`, `--parsed-containment` (default
  `DEFAULT_PARSED_CONTAINMENT_FILE`; missing -> INFO and registry only), `--min-containment` (default 0.0),
  `--registry`, `--no-record`. Writes with `write_csv(df, output, sep="\t", index=False)`; logs
  `"Wrote %d row(s) covering %d accession(s) and %d genome(s) to %s"` (WARNING on zero rows); records via new
  `registry.record_export(registry, name, output, summary)` under `registry.project["exports"][name]`
  (`date`, project-relative `output`, `summary`); `Next:` hint. Verify `store_init` (cli/commands/store.py
  ~L208-215) copies the project block so the key survives.
- Register in `cli/commands/__init__.py` and `cli/main.py` after `StatusCommand()`.
- Tests: processing tests build a registry in memory with `record_screening`, `record_selection`,
  `record_exclusion`, `record_metadata`, `record_extraction(..., coverage=...)`, `record_assembly`: one row per
  pair in the stated order; `min_containment` drops low pairs; `genome_id` restricts; mapping rate is
  mapped/spots else None; parsed table adds pairs the registry capped; extraction without screening appears with
  empty containment; exact column order. CLI tests (Namespace helper like tests/test_cli_select.py `_args`):
  header written; missing parsed table falls back with INFO; export recorded; `--no-record` leaves the registry
  untouched; zero rows warns and writes a header-only file; command registered. Registry test for `record_export`.
- Docs: README and pipeline_overview "Project state" gain `metaquest results_table --output results.tsv`.

### Task 5: Selection filters on run size, spots and platform

Files: `metaquest/processing/selection.py`, `metaquest/cli/commands/select.py`, `metaquest/cli/commands/status.py`,
`tests/test_processing_selection.py`, `tests/test_cli_select.py`, `tests/test_cli_status.py`, docs.

- `selection.py`: constants `RUN_SIZE_COLUMN = "Run_Size"`, `SPOTS_COLUMN = "Run_Total_Spots"`,
  `PLATFORM_COLUMN = "Platform"` (names from metadata.py `_extract_metadata_fields`). `parse_size(value) -> int`
  accepting `1024`, `500M`, `2G`, `1.5G`, `2GB` with decimal multipliers (SRA sizes are bytes), `ProcessingError`
  otherwise. `@dataclass RunFilters(max_run_size, min_spots, max_spots, platform)` with `active()`.
  `_load_metadata(metadata_file)` lifted out of `_filter_by_metadata` so the table is read once.
  `_filter_by_run(ranked, metadata, filters, table_name)`: `pd.to_numeric(..., errors="coerce")`; a run absent
  from the table or without a value is dropped and counted in the log
  (`"%d accession(s) dropped by --max-run-size > %d bytes (%d with no Run_Size value)"`): the user asked for a
  ceiling, an unknown size may be far above it, and the count keeps the loss visible. Platform: strip/lower
  equality, missing dropped and counted. A missing column: WARNING naming the table and the flag ("Branchwater-
  derived tables carry none") and that filter is skipped. `_log_selected_volume` after top_n:
  `"Selected volume: %.2f GB across %d run(s) (%d with unknown size)"`. `select_accessions_ranked(...,
  run_filters=None)`: metadata file required when filters are active; order threshold -> exclude -> metadata
  equality -> run filters -> top_n -> volume log.
- `select.py`: argparse types `_size_bytes` (wraps `parse_size` into `ArgumentTypeError`; reject 0) and
  `_non_negative_int`; flags `--max-run-size BYTES`, `--min-spots N`, `--max-spots N`, `--platform NAME`; resolve
  the metadata table when a column or an active filter needs it; criteria gain `max_run_size` (bytes),
  `min_spots`, `max_spots`, `platform`.
- `status.py` `_reselect_command`: emit the three numbers via `_as_positive_int` + `_warn_malformed` like top_n
  (`--max-run-size 500000000`), `--platform` shlex-quoted.
- Tests: selection fixture with `Run_Size`, `Run_Total_Spots`, `Platform` and one row with an empty size:
  `parse_size` accepts suffixes and rejects garbage; max size drops larger and unknown runs and logs counts; min
  and max spots bound the selection; platform is case-insensitive; filters require a metadata file; missing
  column warns and leaves the ranking unchanged (Branchwater fixture); filters apply before top_n; volume logged.
  CLI: `_args` gains the four fields; criteria recorded; argparse parses `500M` and rejects `abc`; filters
  autodetect the metadata table. Status: reselect reproduces the four flags (bytes as integer, platform quoted
  when it has a space); malformed max_run_size warns.
- Docs: README and pipeline_overview step 2 gain the flags and the unknown-value rule.

Tasks 3, 4 and 5 touch disjoint functions; Task 4 reads the keys Task 3 writes and tolerates their absence, so
they can be reviewed and merged in any order.

### Task 6: Release 0.4.0

Files: `pyproject.toml` (version), `CHANGELOG.md` (new), `README.md`, `.github/workflows/release.yml` (new).

- Bump `version = "0.4.0"`; write `CHANGELOG.md` with sections for 0.4.0 summarising the three audit
  rounds (store journal, AppleDouble handling, download layer, extraction scratch, results table, breadth,
  selection filters, Python 3.12 floor) in plain language, one line per user-visible change.
- `release.yml`: on tag `v*`, build sdist/wheel (reuse the `build` job steps), `twine check`, create a
  GitHub release with the artifacts attached. Publishing to PyPI is a separate decision (needs a token);
  the workflow leaves a commented `pypa/gh-action-pypi-publish` step.
- After merge: `git tag -a v0.4.0 -m "MetaQuest 0.4.0"` and push the tag (user action, confirmed at the
  finishing step).

## Verification after the last task

- `make check`, `make test`, `make pipeline` under the 3.12 conda env.
- CI matrix green on the branch.
- On the sekvens2 reference data (`source /Volumes/sekvens2/metaquest-e2e/env.sh` with the new env's
  bin first on PATH): `extract_target_reads` on one already-extracted crispatus run with `--force` on
  the APFS copy records `breadth` and `mean_depth`; `results_table --output results.tsv` in the crispatus
  project yields one row per screened pair with containment and the extraction columns filled for the
  six extracted runs; `select_datasets --genome-id GCF_000091765.1 --threshold 0.89 --max-run-size 600M
  --no-record --output /tmp/sel.txt` logs the excluded count and the total volume; `status --next` after a
  recorded filtered selection prints the new flags in the reselect command.
