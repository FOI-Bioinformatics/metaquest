# Changelog

All notable changes to MetaQuest are documented in this file. Dates are in YYYY-MM-DD format.

## [0.5.1] - 2026-09-26

### Performance

Measurements below are from a 14,146-dataset project (the crispatus Lactobacillus study) on an
external USB volume on an Apple silicon laptop, unless marked synthetic.

- `download_sra` now records its outcome in one registry transaction per run instead of one write
  per accession: 549 accessions recorded in 184.7 s before, 0.8 s after. The registry file is also
  written as compact JSON instead of indented JSON, with the same keys and values, so the file
  shrank from 16.0 MB to 9.7 MB; a tool that reads the registry as JSON is unaffected.
- `sra_profile --sample-size 10000` on one 11.3-million-spot paired run: 25.1 s and 279 MB peak RSS
  before, 20.9 s and 219 MB after. The sampler now indexes records and reads the gzip stream in 1
  MiB blocks, and keeps quality scores as histograms rather than per-read lists; the remaining time
  is reading the compressed files from disk.
- `extract_target_reads` now runs minimap2 with `--sam-hit-only`, so only mapped records reach the
  SAM file: on SRR23946447 the SAM shrank from 4.09 GB to 742 MB and wall time fell from 79 s to 30
  s, with the same `mapped_reads` count as before.
- `parse_containment`: 2.29 s and 283 MB before, 0.92 s and 210 MB after on the same project; a
  synthetic 20,000-by-5 table went from 4.34 s to 0.39 s. The summary and screening tables are now
  built with numpy in one pass, and a screening run records one timestamp per run instead of one
  per row; output tables are byte-identical to before.
- Download and store adoption now compute md5 and read counts in one streaming pass
  (`fastq_digest`) instead of reading each mate file several times; the store path now reads mate 1
  twice instead of four times.
- `parse_metadata`: each XML file is now parsed once into a sparse attribute table; 2,201 files went
  from 0.74 s to 0.41 s with a warm cache, and RSS from 185 MB to 136 MB. The TSV output is
  byte-identical to before.
- `status` and `results_table` no longer make per-accession filesystem probes or quadratic passes
  over the registry: `status --json` went from 1.27 s to 0.63 s, and `results_table` (29,190 rows)
  from 1.15 s to 0.75 s. Both outputs are byte-identical to before.
- The NCBI and GTDB taxonomy clients now reuse one HTTP session with retries, and write their caches
  incrementally instead of once at the end of a run.

### Changed

- `find_by_taxonomy`'s containment/taxonomy annotation is now built with a `melt` instead of a
  per-row loop, and drops rows whose containment is zero (or missing); the previous behaviour kept
  those as zero-valued rows. A search with no zero-containment cells is unaffected.

### Testing

- Added `tests/test_performance_regressions.py` with timing and memory bounds covering the registry,
  the FASTQ sampler, containment processing, metadata parsing, and `status`/`results_table`.

## [0.5.0] - 2026-09-25

### Breaking changes

- Optional packages moved behind extras: `analysis` (scikit-learn, scipy), `interactive` (plotly,
  jinja2), `maps` (cartopy), `sourmash`, and `all` (every extra). A plain `pip install .` no longer
  installs any of them; a command that needs a missing extra stops with a `ConfigurationError`
  naming the extra and the interpreter to install it into, for example
  `/path/to/python -m pip install 'metaquest[analysis]'`. `statsmodels`, `umap-learn`, `networkx`,
  `upsetplot`, and `seaborn` are no longer dependencies at all. A command that previously produced a
  degraded plain-text or plain-HTML output when a package was absent now stops with that same error
  instead; the correlation heatmap in `interactive_plot` is drawn with matplotlib, which is a core
  dependency, so it no longer needs an extra.
- `sra_stats`, `sra_profile_quality`, `sra_dashboard`, and `sra_compare` are replaced by two
  commands: `sra_profile` (statistics table and a quality profile JSON per accession) and
  `sra_report` (one HTML report, with a group comparison and statistical tests when `--groups-file`
  is given). The four old names still parse; each prints the new command name and exits with status
  2 rather than running. They are hidden from `metaquest --help`. See "Renamed SRA commands" in
  README.md for the full flag mapping.
- `sra_validate --accessions` is replaced by `--accessions-file` and repeatable `--accession`,
  matching `sra_profile` and `sra_report`. `--include-contamination` is removed; the adapter figures
  are always computed.
- GC content is reported in percent (`gc_percent`, 0-100) everywhere: the `sra_profile` CSV, the
  per-accession quality profile JSON (`complexity_score` also replaces the old ad hoc key), and the
  `results_table` output. The previous `gc_content` key (a 0-1 fraction in the JSON, already percent
  in the CSV) is gone. `avg_quality` in the CSV is now a per-read mean over a sample of reads from
  every mate file, rather than mate-1 records only.
- `results_table` gains `total_reads`, `gc_percent`, and `quality_grade` columns, inserted after
  `run_size` and before `mapped_reads`. Every later column shifts position accordingly; a script
  that reads `results_table`'s output by column index rather than by header name must be updated.
- `import metaquest.data.sra` now imports a package (`metaquest/data/sra/`) rather than a single
  module. The public functions re-exported from `metaquest.data.sra` are unchanged; private helper
  names moved to their new submodules (`fastq`, `cleanup`, `accession`, `store_handoff`, `retry`,
  `download`) and are no longer importable from the old single-file path.
- Public Python API changes outside `metaquest.data.sra`:
  `metaquest.data.sra_metadata.calculate_read_statistics` is removed (statistics come from
  `metaquest.sra.dataset_stats` and `metaquest.sra.profiles`), and
  `metaquest.data.sra_metadata.generate_statistics_report` now takes a sequence of statistics rows
  and an output path instead of a FASTQ folder and a sample size. `SequenceQualityAnalyzer` moved to
  `metaquest/sra/quality.py`; it is still importable from `metaquest.sra` and
  `metaquest.sra.analytics`. `metaquest.data.branchwater_search.sourmash_hint` and
  `SOURMASH_HINT` are removed; a missing sourmash is reported through
  `metaquest.core.optional.require` like every other extra.
- Error lines that end a command with exit status 1 (for example "No accessions found in file" and
  "FASTQ folder ... does not exist") are now written to the log on stderr instead of stdout. A
  script that parsed those lines from stdout must read stderr or check the exit status instead. The
  `--json` error document for a missing store stays on stdout.
- Registry analysis keys changed: `sra_profile` records its outcome under `profile` and `sra_report`
  under `report`, replacing the `sra_stats` and `quality` keys. Only `results_table` and `status`
  still fall back to the old keys when reading a registry written by an earlier version.

### Exception handling

- Added a `flake8-blind-except` lint gate (`B902`, bare `except Exception:`) with a per-module
  exemption list in `setup.cfg` that only shrinks as modules are narrowed.
- Narrowed broad exception handling in six modules (`metaquest/data/sra.py` and its later split,
  `sra_metadata.py`, `metadata.py`, `branchwater.py`, `visualization/reporting.py`, and
  `sra_intelligent.py`): a caught exception now names the specific error types the surrounding code
  can actually raise, so a programming error (for example a `TypeError` or `AttributeError`) is no
  longer swallowed and reported as a data or network failure. Corrupt gzip streams (`zlib.error`)
  and truncated NCBI replies are now handled explicitly rather than falling through a catch-all.

### Output

- Every command now writes to stdout through exactly one channel: `BaseCommand.emit`,
  `emit_raw`, and `emit_json` (plus the module-level `emit_error_json` for callers with no command
  instance). Library modules that used to print now log or return their lines instead, and the
  calling command writes them. The four `--json` commands (`status`, `store_status`, `store_usage`,
  `store_gc`) each emit exactly one JSON document.
- Added a `make check` gate (`scripts/check_no_print.sh`) that fails on any `print(` or
  `sys.stdout.write` outside `metaquest/cli/base.py`.

### Dependency audit

- Added `pip-audit --strict --desc .` to the CI lint job, and a new weekly `audit.yml` workflow that
  runs the same check on a schedule.
- The weekly Branchwater workflow's verify step now makes real assertions about the pipeline's
  output instead of only echoing that it ran.

### Documentation

- Added a `flake8-docstrings` gate (`D101`/`D102`/`D103`: a docstring on every public class, method,
  and function), enforced on a defined set of modules: the registry, the `metaquest/data/sra/` and
  `metaquest/store/` packages, the `metaquest/cli/commands/store/` and
  `metaquest/cli/commands/status/` packages, `processing/selection.py`, `processing/results.py`,
  `processing/status_report.py`, `data/registry_blocks.py`, `core/optional.py`, `sra/quality.py`,
  and the SRA analysis command and profiling modules. Every other
  module has the check switched off explicitly in `setup.cfg` rather than by a wildcard, so the
  enforced set stays visible and can grow one file at a time.

### Module layout

- Split `metaquest/data/sra.py` into the package `metaquest/data/sra/` (`fastq`, `cleanup`,
  `accession`, `store_handoff`, `retry`, `download`), each module under 600 lines.
- Split `metaquest/cli/commands/store.py` into `metaquest/cli/commands/store/` (one module per
  command: `init`, `status`, `reindex`, `adopt`, `verify`, `link`, `usage`, `gc`, plus a shared
  `_shared.py`), and `metaquest/cli/commands/status.py` into
  `metaquest/processing/status_report.py` (report building) and `metaquest/cli/commands/status/`
  (`command.py`, `suggest.py`, `render_text.py`).
- Added typed registry blocks (`metaquest/data/registry_blocks.py`): every block in
  `metaquest_registry.json` (screening, selection, exclusion, download, metadata, analysis,
  extraction, assembly, project, store) is now a dataclass with `from_dict`/`to_dict`, round-trips
  any key it does not itself declare, and keeps the on-disk JSON layout unchanged. `update_linked`
  moved from the store commands into `metaquest/data/registry.py`, next to the other registry
  writers.
- Added a module size and maintainability gate: 800 lines and a maintainability index of 20 or
  higher per module under `metaquest/`, with a shrinking `KNOWN_EXCEPTIONS` list in
  `tests/test_module_sizes.py` for modules that were already over a ceiling when the gate was
  added.
- Added an ASCII-only gate for `metaquest/`, `tests/`, `scripts/`, `Makefile`, `setup.cfg`, and
  `pyproject.toml`. A single line can be exempted with an `# ascii-ok` marker and a reason; the
  package author's name and the test fixtures that check Unicode handling use it.
- Added a documented-commands gate: every command visible in `metaquest --help` must be mentioned
  in README.md, and README.md must not invoke a command the CLI registry does not have.

### Continuous integration

- Added a nightly smoke workflow (`.github/workflows/smoke.yml`) that runs
  `scripts/smoke_chain.sh` against the real SRR2517620 run on NCBI/SRA, using conda-installed
  tools (fasterq-dump, prefetch, minimap2, samtools); `make test-network` runs the same chain
  locally.

## [0.4.0] - 2026-09-24

### Breaking changes

- Python 3.11 is no longer supported; MetaQuest now requires Python 3.12 or later.
- `download_sra` no longer prints its own download summary from the CLI layer; the summary line
  now comes from the store layer once, so a run that only links existing store datasets is no
  longer counted as a download.
- `select_datasets --no-record` now refuses to overwrite a recorded selection's output file; give
  it a different `--output` path or run without `--no-record`.

### Python floor and packaging

- Raised the minimum supported Python version to 3.12 and the floor for MetaQuest's own
  dependencies to versions tested on 3.12 and 3.13.
- Documented the Python 3.12 floor in the README and dropped stale test-count claims.

### Continuous integration

- The test job now runs on a matrix of Ubuntu and macOS with Python 3.12 and 3.13, with a mypy
  type-check step enabled.
- `make clean` no longer deletes `test_data`, so a local pipeline run's sample data survives a
  clean.
- Tests that exercise the SRA download path no longer depend on `sra-tools` being on `PATH`.

### Store

- The shared store now keeps an append-only journal of project and usage records; `store_reindex`
  replays it after rebuilding the catalogue from sidecar files, so a reindex no longer loses which
  projects used which datasets.
- `store_gc` refuses to run against a catalogue that has datasets but no project records, and
  reports its `--json` refusal the same way it reports other refusals.
- `store_adopt` refuses an empty accession folder, verifies the store's copy before removing
  anything from the project, and `--copy` no longer records a link when the accession is already a
  duplicate in the store.
- `store_verify --fix-state` only promotes a dataset to `complete` once its files have actually
  been read through, never on size or md5 alone; `--rescan` reports a folder with no FASTQ files as
  missing instead of silently skipping it; spot counts are also read from a project's own
  `metadata/` folder when the store does not have them.
- Folder scans, copies, and removals throughout the store and CLI now ignore macOS AppleDouble
  sidecar files (`._*`) and dotfiles, so they never look like real FASTQ, FASTA, CSV, XML, or
  profile files.
- Journal replay and catalogue backfill now keep the earliest recorded `first_used` and the latest
  `last_used` date for a dataset, and preserve the hostname and time already on a catalogue row
  instead of overwriting them.

### Downloads

- Ctrl-C during `download_sra` now cancels pending downloads and terminates any running `prefetch`
  or `fasterq-dump` process immediately, without waiting on a store lock or on a tool started after
  the interrupt.
- `fasterq-dump` scratch files are written under the store's own tmp folder and cleaned up
  afterwards.
- A relink of an already-downloaded accession keeps its recorded read count, and the download
  summary counts relinks separately from new downloads.
- The hint to install `sourmash` now names the Python interpreter that is actually running, not a
  generic `pip install` line.

### Selection

- `select_datasets` gained `--max-run-size`, `--min-spots`, and `--max-spots` (with a K/M/G/T
  suffix on run size) and `--platform` filters, and logs the excluded count and total volume of
  what a filtered selection kept. A requested filter whose column is missing from the metadata table
  (as in a Branchwater-derived table) is an error, and nothing is written or recorded.
- `select_datasets --no-record` skips writing the selection into the project registry, and
  `status --next` prints a runnable reselect command that reproduces the metadata filter, top-N,
  source table, and the new size/spot/platform filters, shell-quoted where needed.
- `select_datasets` now records the metadata table it actually used, whether autodetected or
  passed explicitly. The `status --next` reselect hint warns once on a malformed recorded filter
  value and leaves that flag out of the suggested command (or uses its default) instead of failing
  silently.

### Extraction and assembly

- `extract_target_reads` now records breadth of coverage (fraction of the reference genome covered
  at 1x or more) and length-weighted mean depth from `samtools coverage`, alongside the existing
  read counts.
- Each extraction and assembly run gets its own megahit scratch folder under the output directory,
  named with a random suffix, so concurrent runs sharing one output folder no longer collide; a
  dangling `fastq/<ACC>` symlink is reported as one warning naming every broken link.
- `--force` on read extraction or assembly now clears a stale registry record even when its output
  folder is already gone, and keeps an earlier recorded genome-download manifest instead of
  discarding it.
- Extraction and assembly dry runs summarise what would run; megahit errors surface the tool's own
  message, and tmp-dir and parameter changes are logged explicitly.
- A forced `extract_target_reads` rerun that keeps no reads now removes the earlier
  `<genome>_coverage.tsv`.

### Results

- Added the `results_table` command, which writes one row per screened (accession, genome) pair
  from the registry and the parsed containment table, joined with the extraction and coverage
  columns for that pair.

### SRA info and metadata

- `sra_info` now reports run accessions, sizes, base counts, and dates from the `RUN` element, and
  its accession filter also matches BioProject and BioSample identifiers, at the package level
  rather than the run level.
- Metadata parsing now fills in library strategy, source, and selection from the
  `EXPERIMENT`/`DESIGN` section when present.
- `sra_compare` writes JSON-safe results (no raw numpy types) and works from a single accession or
  a quality-profiles directory without requiring `--accessions-file`.
- A package whose inspection fails no longer makes `sra_info` fall back to listing every run
  returned; that fallback now applies only when every package was inspected without error, and a
  separate warning names how many packages could not be inspected.

### Documentation

- Reworded the README's testing summary from "All modules now thoroughly covered" to "Critical
  modules now thoroughly covered", matching the modules the README and CLAUDE.md elsewhere still
  list as uncovered.
- Reflowed a wrapped paragraph in the README's read-extraction section.
- Dropped stale per-module coverage percentages from CLAUDE.md's list of well-tested reference
  files, keeping the file list itself.
- Added a Releases section to the README describing the tag-triggered release workflow.

## [0.3.1] and earlier

See the git history for changes before 0.4.0; this file starts tracking releases from 0.4.0.
