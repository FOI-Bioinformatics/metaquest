# Changelog

All notable changes to MetaQuest are documented in this file. Dates are in YYYY-MM-DD format.

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
  what a filtered selection kept.
- `select_datasets --no-record` skips writing the selection into the project registry, and
  `status --next` prints a runnable reselect command that reproduces the metadata filter, top-N,
  source table, and the new size/spot/platform filters, shell-quoted where needed.
- `select_datasets` now records the metadata table it actually used, whether autodetected or
  passed explicitly, and warns once on a malformed filter value instead of failing silently.

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
- `results_table --output` no longer leaves a stale coverage column behind when a run has zero
  kept reads.

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
