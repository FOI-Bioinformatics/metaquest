# Dataset tracking in MetaQuest: SWOT, gap analysis, and improvement plan

## Context

MetaQuest screens SRA metagenomes against a genome (Branchwater), selects datasets, downloads reads,
profiles them, extracts target reads and assembles them. A run therefore has a population of accessions
that move through stages: screened -> selected -> downloaded -> analysed (stats, quality, validation)
-> extracted (per target genome) -> assembled. The maintainer asked how well MetaQuest keeps track of
which datasets are in which stage, with a SWOT and a gap analysis, and a plan to close the gaps.

This plan was written in read-only mode against `main` at 2d2c9ce (2026-09-05), using the code, the
docs, and the real Wolbachia working directory from the 2026-09-04 audit (10 runs, two target genomes)
as evidence.

## Evidence: how state is tracked today

### Observed on the real working directory (10 Wolbachia runs)

Per accession, the stages are recognisable only by whether files exist in stage-specific folders:

| Accession | fastq | NCBI XML | quality JSON | extracted (wMel) | assembled (wMel) |
|---|---|---|---|---|---|
| SRR11011981 | y | y | y | y | y |
| SRR11011979 | y | y | y | y | y (empty, 0 reads) |
| SRR11011978 | y | y | y | y | y (empty, 0 reads) |
| ERR16650665 | y | y | - | - (extracted vs wPip in another folder) | - |
| ERR15315137 | y | y | - | - | - |
| SRR1661114 .. SRR6485752 (5) | y | - | - | - | - |

`metaquest status --json` on that directory reports only `fastq_accessions: 11, metadata_xml: 5,
genome_fasta: 2` and `wanted: 10 / fastq_present 10 / metadata_missing [5]`. It cannot say that three
runs were extracted against wMel, that two of those yielded no reads, that two other runs were
extracted against wPip into a different folder, that one accession (SRR2517620) is a leftover test
download that was never wanted, that one run (SRR2517418) is a mislabelled amplicon library that
should be blacklisted, or which threshold produced the wanted list. The 2 GB of reads have no
recorded size, checksum or date. The only decision records are `fastq/failed_accessions.txt` and the
optional `download_report.csv`, both per run and overwritten on rerun.

### Code paths (from exploration, main at 2d2c9ce)

- Wanted set: `status` derives it from `--accessions-file` and/or the index of `--parsed-containment`
  (`metaquest/cli/commands/status.py:48-64`); `download_metadata` derives it from `matches/*.csv` plus
  a threshold (`metaquest/data/metadata.py:27-74`); `select_datasets` derives it from the containment
  table plus optional metadata filter (`metaquest/processing/selection.py:16-76`). Three different
  derivations, no shared source of truth, three different default thresholds (`constants.py` 0.5,
  `select.py` 0.1, `metadata.py` 0.0).
- Presence = file existence only: FASTQ via `accession_has_fastq` (`metaquest/data/sra.py:31`, any
  `*.fastq*` under `fastq/<acc>/`), metadata via `metadata/<acc>_metadata.xml`, genomes via a FASTA glob.
  No size, checksum or date is recorded or checked, although `parse_metadata` extracts `Run_Size` and
  `Run_MD5` into `metadata_table.txt` (`metaquest/data/metadata.py:244-246`).
- Selection provenance is discarded: `select_datasets` writes the bare accession list
  (`metaquest/cli/commands/select.py:57-63`); threshold, genome id, metadata filter and date are lost.
- Download outcome: `failed_accessions.txt` written by both `data/sra.py:419` and
  `cli/commands/sra.py:117` (duplicated, hard-coded name although `FAILED_ACCESSIONS_FILE` exists in
  constants); `--report-file` CSV with statuses downloaded/failed/already_present/blacklisted omits
  accessions cut by `--max-downloads`. Partial downloads leave no state (`<acc>_temp` is removed).
  Blacklists are read (`_read_blacklist_files`, `data/sra.py:42-70`) but never written by the tool.
- Models: `Containment`, `GenomeInfo`, `SRAMetadata`, `TaxonomyInfo`, `ContainmentSummary` in
  `metaquest/core/models.py` are in-memory only; nothing serialises a dataset record.
- `status` is blind to extraction and assembly outputs, quality profiles, Branchwater metadata,
  blacklists and failed downloads.

### Analysis, extraction and assembly stages

- Extraction (`metaquest/data/read_extraction.py:111-252`): outputs `targeted/<acc>/<genome>_1/_2/_s/_0.fastq.gz`
  (or `<genome>.fastq.gz` single-end); empty files are deleted. Mapped-record counts, threshold, preset
  and which samples were selected exist only in the log and in memory (`ExtractionResult`); nothing is
  written. No skip logic: a rerun re-maps every selected sample and overwrites outputs.
- Assembly (`read_extraction.py:277-316`): MetaQuest writes nothing of its own beside megahit's `log`,
  `options.json`, `checkpoints.txt`, `final.contigs.fa`; no contig count, N50, coverage or version is
  recorded. A second `--assemble` run fails inside megahit because the output folder already exists
  (no `exists()` check).
- `sra_stats` writes a per-accession CSV (`sra_statistics.csv` by default) that no command reads back;
  `sra_info` writes `sra_info_report.csv`; `sra_validate` persists nothing (results only printed).
- `sra_profile_quality` writes `<out>/quality_summary.json` (with `total_analyzed`, `failed_accessions`,
  the only place any command records success vs failure per accession) and, only with
  `--detailed-reports`, `<out>/<acc>_quality_profile.json`. `sra_dashboard --quality-profiles` is accepted
  but never used (`sra_intelligent.py:286-289` vs `317-374`); dashboards and `sra_compare` recompute from
  FASTQ every time and record nothing about skips.
- Genomes: `genome_download` is idempotent on `<out>/<acc>.fna`; `genome_prepare --manifest-file` globs
  `*.fna.gz`/`*.fasta.gz` (`metaquest/cli/commands/genome.py:298-316`) although the downloader writes
  plain `.fna`, so the manifest is empty after a real run. No download date, tool version or filter is
  recorded.
- `status` is unaware of `targeted/`, quality JSON, statistics CSV and assemblies.

### Documentation and reusable pieces

- The README documents the order select_datasets -> download_sra -> (status at any point) -> sra_info,
  sra_stats, sra_validate, sra_profile_quality, sra_dashboard, sra_compare -> extract_target_reads, and
  names `fastq/failed_accessions.txt` and `--report-file` as the retry and outcome records. "Blacklisted"
  appears only as a report status; no blacklist file or format is documented. None of
  `docs/ARCHITECTURE.md`, `docs/branchwater_workflow.md`, `docs/SRA_ENHANCED_FEATURES.md` mention state,
  resume, manifests or provenance; ARCHITECTURE.md lists "Database integration" under future extensions.
- Reusable helpers: `metaquest/data/file_io.py` (`ensure_directory`, `list_files`, `read_csv`,
  `write_csv`), `metaquest/data/defaults.py` (`resolve_metadata_table`, `read_matrix`, `read_table`,
  `read_records`). No JSON, hashing or timestamp helpers. Dependencies offer pandas plus the standard
  library (`json`, `csv`, `sqlite3`); nothing like pydantic or duckdb.
- Project layout assumed by tests (`tests/test_cli_status.py:28-33`): sibling `fastq/`, `metadata/`,
  `genomes/` under a project root plus top-level tables (`parsed_containment.txt`, `accessions.txt`).
  `.gitignore` ignores `genomes/` and `examples/` but not `fastq/`, `matches/`, `branchwater/` or the
  output tables, an inconsistency a registry location must not repeat.

## SWOT: dataset tracking

**Strengths**
- The filesystem layout is regular and per accession (`fastq/<acc>/`, `metadata/<acc>_metadata.xml`,
  `targeted/<acc>/<genome>_*`), so presence can be re-derived from disk; nothing can drift out of
  step with reality because nothing is cached.
- Every download-side command is idempotent on presence (`download_sra`, `download_metadata`,
  `genome_download` skip what exists; `--force` overrides), and `download_sra` writes a retry list.
- `status` exists, has `--json`, and reconciles a wanted list against disk; `select_datasets` gives the
  wanted list a first-class producer; `--report-file` gives a per-run outcome record.
- Quality profiling already records success and failure per accession in `quality_summary.json`;
  `sra_stats` writes a per-accession CSV; `parse_metadata` carries NCBI's `Run_Size` and `Run_MD5`.

**Weaknesses**
- No record of decisions: which threshold, genome and filter produced `accessions.txt`; why an accession
  was excluded; which runs are blacklisted and why. The tool reads blacklists but never writes one.
- No record of provenance: no dates, sizes, checksums, tool versions or parameters for downloads,
  extractions or assemblies; mapped-read counts and assembly statistics exist only in logs.
- `status` sees three of six stages; extraction, assembly and analyses are invisible, and multiple
  target genomes (the normal case) have no representation at all.
- Three definitions of "wanted" (status, download_metadata, select_datasets) with three default
  thresholds; a leftover download shows up as an extra accession nobody asked for.
- Presence means "some file exists": a truncated FASTQ, an empty extraction folder or an empty
  assembly folder all count as done. Rerunning extraction redoes everything; rerunning assembly
  crashes on the existing folder.
- Analysis outputs are write-only: nothing reads `sra_statistics.csv` or the quality JSON back
  (`sra_dashboard --quality-profiles` is dead), and `sra_validate` keeps no record.
- Two known bugs in this area: duplicated `failed_accessions.txt` writers, and an always-empty
  `genome_prepare` manifest.

**Opportunities**
- All the raw facts already exist at the moment each command finishes (containment values, selection
  criteria, download results dictionary, `ExtractionResult.mapped_records`, megahit's `final.contigs.fa`
  and `options.json`, NCBI `Run_Size`/`Run_MD5`); recording them is a matter of writing one line per event.
- A single per-project state file plus a richer `status` would give the scientist the one table they
  actually want: accession x stage, per genome, with dates and counts, and a list of what to run next.
- Idempotent skip with `--force` for extraction and assembly follows directly from recorded outputs.
- An explicit blacklist with reasons turns the mislabelled-amplicon case into a recorded decision
  instead of tribal knowledge.
- Machine-readable state enables `status --next` (what to run), `status --list-missing --stage extracted
  --genome X`, and later a Snakemake or Nextflow wrapper.

**Threats**
- A state file that is not reconciled against disk drifts (files deleted by hand, runs on another
  machine); the design must treat disk as authoritative for presence and the registry as authoritative
  for decisions and provenance.
- Parallel download workers and a user running two commands at once can race on one file; atomic
  replace and single-writer-per-command are needed.
- Multi-genome extraction gives the state a two-dimensional shape (accession x genome); a flat table
  will not do.
- More recorded state means more places for stale information after manual edits; every writer must be
  cheap and every reader tolerant of missing fields.
- Scope creep towards a workflow engine; the goal is bookkeeping, not scheduling.

## Gap analysis

| Need | Today | Gap |
|---|---|---|
| Which accessions were screened, against which genome, with what containment | `matches/*.csv`, `parsed_containment.txt` (recomputable) | Adequate as data; not linked to later stages |
| Which were selected and why | `accessions.txt` (bare list) | Criteria, date and excluded set not recorded |
| Which are excluded on purpose | user-maintained blacklist files, never written by the tool | No command, no reasons, not shown by `status` |
| Which are downloaded, complete, when, how big | folder existence; `Run_Size`/`Run_MD5` sit unused in the metadata table | No completeness check, no date, no size, no checksum |
| Which failed and why | `failed_accessions.txt` (overwritten per run), `--report-file` | No history, no attempt count, `--max-downloads` cut-offs missing |
| Which have metadata | `metadata/<acc>_metadata.xml` existence | Branchwater metadata not counted |
| Which are analysed (stats, quality, validated) | `sra_statistics.csv`, `quality_summary.json`, nothing for validate | Not visible in `status`; validate unrecorded |
| Which are extracted, per genome, with how many reads | `targeted/<acc>/<genome>_*.fastq.gz` files; counts in log only | Not visible in `status`; zero-read runs look done; no skip on rerun |
| Which are assembled, per genome, with what result | `final.contigs.fa` plus megahit's own files | Not visible; no contig/N50 record; rerun crashes |
| What to run next | nothing | No command answers it |
| One consistent wanted set and threshold | three derivations, three defaults | Needs one source and one default |

## Decisions taken with the maintainer (2026-09-05)

- Model: registry plus reconcile. A per-project state file records decisions and provenance; presence
  is always re-checked against disk, so the filesystem stays the truth for "exists" and the registry
  the truth for "why, when, how much".
- Scope of the first increment: registry module and recording in every stage command, richer `status`
  with the accession x stage x genome matrix and a what-to-run-next list, a `blacklist` command with
  reasons, idempotent skip with `--force` for extraction and assembly, plus the two bug fixes (empty
  genome manifest, duplicated `failed_accessions.txt` writer).
- Format: one JSON document, `metaquest_registry.json`, in the project root; `status --export-tsv`
  writes flat tables for spreadsheets.

## Recommended approach

### The registry file

- Path: `metaquest_registry.json` in the project root (the folder holding `fastq/`, `matches/`,
  `targeted/`). Every writing command accepts `--registry PATH`; the default is found by walking up
  from the working directory (like git), so `status` works from a subfolder.
- Shape: pretty-printed JSON, sorted keys, one field per line (grep and `git diff` friendly).
  `{"version": 1, "created", "updated", "genomes": {<genome_id>: {...}}, "datasets": {<acc>: <record>}}`.
- Per-accession record (ISO dates, all keys optional so older files still load):
  `screening{source, server, query_threshold, date, genomes{<genome>: {containment, cani, csv}}}`,
  `selection{selected, date, criteria{column, threshold, metadata_column, metadata_value, table}, output}`,
  `exclusion{excluded, reason, source, date}`,
  `download{state: downloaded|failed|missing|skipped, date, attempts, files[{path, bytes, mtime}], bytes_total, message}`,
  `metadata{xml, date, run_size, run_md5, assay_type, organism, collection_date}`,
  `analyses{sra_stats|quality|validate: {date, output, summary{}}}`,
  `extractions{<genome>: {date, genome_fasta, preset, threshold, mapped_reads, unequal_mates, files[],
  assembly{date, dir, contigs, total_bp, n50, largest, tool, version, params}}}`.
  NCBI's `Run_MD5` is the archive checksum, not the FASTQ's, so it is stored for reference only; no
  false "verified" claim.
- Writes: `save_registry` writes `metaquest_registry.json.tmp.<pid>` and `os.replace`s it (atomic);
  an `O_CREAT|O_EXCL` lock file with a 30 s stale timeout guards two terminals; every writer does
  load, mutate, save in one call. Parallel download workers never touch the registry: the main thread
  records results as they complete (see `download_sra` below).
- Size guard: screening entries are stored only for matches at or above the search threshold and
  capped per genome (`--registry-max-screened`, default 5000, warn and keep the best); the CSVs remain
  the raw record.

### New module `metaquest/data/registry.py` (no pandas except in the TSV export)

```
REGISTRY_FILENAME = "metaquest_registry.json"; SCHEMA_VERSION = 1
@dataclass Registry(version, created, updated, genomes: Dict[str, dict], datasets: Dict[str, dict])
@dataclass ProjectPaths(fastq, metadata, genomes, targeted, matches)
registry_path(explicit=None, start=".") -> Path
load_registry(path=None) -> Registry              # empty Registry when absent
save_registry(registry, path=None) -> Path        # atomic, locked
upsert_dataset(registry, accession) -> dict
record_screening(registry, accession, genome_id, containment, cani, source, query_threshold, csv_path)
record_selection(registry, accessions, criteria, output)   # others previously selected become selected=false
record_exclusion(registry, accession, reason, source="user"); clear_exclusion(registry, accession)
record_download(registry, accession, state, fastq_dir, message="")   # stats the files itself
record_metadata(registry, accession, xml_path, fields)
record_analysis(registry, accession, analysis, output, summary)
record_extraction(registry, accession, genome_id, files, mapped_reads, unequal_mates, params)
record_assembly(registry, accession, genome_id, assembly_dir, stats, tool_version, params)
extraction_record(registry, accession, genome_id) -> Optional[dict]
query(registry, stage, genome_id=None) -> List[str]; stage_counts(registry) -> dict
reconcile(registry, paths) -> ReconcileReport      # recorded-but-missing, untracked, empty assembly dirs
bootstrap_from_disk(paths) -> Registry             # marks every inferred field "inferred": true
to_dataframes(registry) -> (datasets_df, extractions_df)
```
Reuse: `accession_has_fastq` (`metaquest/data/sra.py:31`), `ensure_directory`/`list_files`/`write_csv`
(`metaquest/data/file_io.py`), `read_table`/`read_records` (`metaquest/data/defaults.py`). The
filesystem scanners from the "filesystem only" design (`scan_downloads`, `scan_extractions`,
`scan_assemblies`, `contig_stats` parsing megahit's `len=` headers, `count_fastq_reads`) live in the
same module and serve both `bootstrap_from_disk` and `reconcile`.

### Who records what

| Command | Where | Records |
|---|---|---|
| `branchwater_search`, `use_branchwater`/`parse_containment` | `cli/commands/branchwater_search.py`, `cli/commands/containment.py` (`ParseContainmentCommand.execute`) | `record_screening` per accession and genome (source, threshold, csv) |
| `select_datasets` | `cli/commands/select.py` after writing the list | `record_selection` with full criteria |
| `download_sra` | `cli/commands/sra.py`; `metaquest/data/sra.py:_execute_parallel_downloads` gains `on_result: Optional[Callable[[str, bool, str], None]]` called on the main thread in the `as_completed` loop | `record_download` per accession as it finishes (checkpoint), then already-present, blacklisted and `--max-downloads` skips; registry exclusions are unioned into the blacklist via a new `blacklist_accessions` argument of `download_sra` |
| `download_metadata`, `parse_metadata` | `metaquest/data/metadata.py` (`_download_single_metadata`, `parse_metadata` where `Run_Size`/`Run_MD5` are read) | `record_metadata` |
| `sra_stats`, `sra_validate`, `sra_profile_quality` | `cli/commands/sra_enhanced.py`, `cli/commands/sra_intelligent.py` | `record_analysis` with output path and headline numbers (validate finally gets a record) |
| `extract_target_reads` | `cli/commands/read_extraction.py` | `record_extraction` then `record_assembly`; `extract_target_reads` returns `Dict[str, ExtractionResult]` so counts reach the CLI; new `summarise_contigs(path) -> {contigs, total_bp, n50, largest}` in `data/read_extraction.py`; megahit version via `run_secure("megahit", ["--version"])` (add `--version` to the megahit allowlist in `core/constants.py`) |
| `blacklist` (new) | `cli/commands/blacklist.py` | `record_exclusion`/`clear_exclusion`; also appends to `blacklist.txt` so `download_sra --blacklist` keeps working |
| `genome_prepare` | `cli/commands/genome.py:_create_manifest` | `registry.genomes[<name>] = {fasta, manifest, date}` |

### `status` (rewrite of `metaquest/cli/commands/status.py`)

- Keeps `--fastq-folder`, `--metadata-folder`, `--genomes-folder`, `--accessions-file`,
  `--parsed-containment`, `--list-missing`, `--json`; adds `--targeted-folder`, `--matches-folder`,
  `--registry`, `--stage {screened,selected,excluded,downloaded,analysed,extracted,assembled}`,
  `--genome` (repeatable), `--init`, `--reconcile`, `--export-tsv`, `--next`.
- Wanted set: the registry's selected accessions when a registry exists; otherwise the existing
  file-based sources. This retires the third derivation; `download_metadata` gains `--accessions-file`
  so it can use the same list.
- Output: the existing on-disk block (unchanged keys) followed by a stage matrix: one row per stage with
  count and the recorded criteria and date, then per-genome rows for extraction and assembly including
  "n with 0 mapped reads" and "n empty assembly dirs". `--stage X --genome Y` lists accessions;
  `--list-missing` lists the gaps at each stage; `--next` prints the commands that would advance the
  most accessions (for example `download_sra` for the selected-but-missing, `extract_target_reads` for
  downloaded-but-unextracted per genome).
- `--json` schema: `{"registry": {path, version, updated}, "on_disk": {...as today...},
  "wanted": {...as today...}, "stages": {<stage>: {count, accessions}}, "genomes": {<g>: {extracted,
  assembled, zero_mapped, empty_assembly_dirs}}, "drift": {recorded_missing, untracked}}`.
- No registry: `status` bootstraps in memory, prints the reconstructed matrix and says
  `no registry yet; run: metaquest status --init`. `--init` persists the bootstrap with
  `"inferred": true` on reconstructed fields. `--reconcile` repairs drift (marks missing downloads,
  registers untracked FASTQ and extractions, flags empty assembly dirs).
- Rendering split into `_render_stage_matrix`, `_render_genomes`, `_render_drift`, `_render_next` to
  stay under the radon ceiling.

### Idempotent extraction and assembly

- `extract_target_reads(..., force=False)`: skip an (accession, genome) whose registry record exists
  with the same `genome_fasta`, `preset` and `threshold` and whose files still exist (log "already
  extracted"); zero-mapped samples are recorded with `mapped_reads: 0` and skipped on rerun too, which
  the filesystem alone could not distinguish from "not run".
- Assembly: skip when `<genome>_assembly/final.contigs.fa` exists; an assembly dir without contigs is
  reported as interrupted; `--force` removes the megahit dir first (fixes the crash on rerun).
- `--force` added to `extract_target_reads` and passed to both steps.

### Bug fixes folded in

- Genome manifest: move the FASTA glob tuple from `status.py` into `core/constants.py` as
  `GENOME_FASTA_GLOBS`, use it in `genome_prepare._create_manifest` (`cli/commands/genome.py:298-316`)
  and `status`; derive the name by stripping the matched suffix; look for `.faa` and `.faa.gz`.
- `failed_accessions.txt`: written once, in `metaquest/data/sra.py`, using `FAILED_ACCESSIONS_FILE`
  from constants; the CLI only logs the retry hint.
- `sra_dashboard --quality-profiles` becomes real: read the saved profile JSON (through the registry's
  analysis records) instead of recomputing.
- One default containment threshold: `DEFAULT_CONTAINMENT_THRESHOLD = 0.1` in constants, used by
  `select_datasets`, `extract_target_reads` and `status`.

### Files

- Create: `metaquest/data/registry.py`, `metaquest/cli/commands/blacklist.py`,
  `tests/test_data_registry.py`, `tests/test_cli_blacklist.py`.
- Modify: `metaquest/cli/commands/status.py` (rewrite), `metaquest/cli/commands/{select,sra,
  read_extraction,branchwater_search,containment,genome,sra_enhanced,sra_intelligent}.py`,
  `metaquest/data/{sra,read_extraction,metadata}.py`, `metaquest/core/constants.py`,
  `metaquest/cli/main.py` (register `blacklist`, group `Reads`), `tests/helpers_extraction.py` and the
  affected test files, `local_test.sh` (add `status --init`, `blacklist`, a rerun of
  `extract_target_reads --dry-run` showing skips), README (new "Project state" section, `status`,
  `blacklist`), `docs/ARCHITECTURE.md` (registry as the project journal), `docs/branchwater_workflow.md`,
  `CLAUDE.md`/`AGENTS.md` command lists, `.gitignore` (do not ignore `metaquest_registry.json`).

### Implementation order (each step is one reviewable commit; PR-sized groups marked)

PR 1: registry core and status
1. `metaquest/data/registry.py` with load/save/upsert/record_*/query/scanners/bootstrap/reconcile and
   `tests/test_data_registry.py` (tmp_path trees: absent file, round trip, atomic replace leaves no
   temp file, lock timeout, upsert merge, bootstrap of a fake project, reconcile drift, TSV export,
   megahit header parsing, gz and plain read counting).
2. Constants: `GENOME_FASTA_GLOBS`, `DEFAULT_CONTAINMENT_THRESHOLD`, megahit `--version`; genome
   manifest fix with a test using a plain `.fna`.
3. `status` rewrite with `tests/test_cli_status.py` extended (bootstrap message, `--init`, matrix text,
   `--stage`/`--genome`, `--json` schema keeps `on_disk`/`wanted`, drift, `--next`, `--export-tsv`).

PR 2: recording in every stage command
4. `blacklist` command (add/remove/list, `--reason`, `--from-file`; writes registry and `blacklist.txt`).
5. `select_datasets`, `branchwater_search`, `parse_containment` record screening and selection.
6. `download_sra`: `on_result` callback, `record_download`, registry exclusions unioned, single
   `failed_accessions.txt` writer, `--max-downloads` skips recorded as `skipped`.
7. `download_metadata`/`parse_metadata` (with `--accessions-file`), `sra_stats`, `sra_validate`,
   `sra_profile_quality` record; `sra_dashboard --quality-profiles` reads saved profiles.

PR 3: extraction and assembly
8. `extract_target_reads` returns `ExtractionResult` per accession; `summarise_contigs`; records
   extraction and assembly; skip logic and `--force`; tests assert `run_secure` is not called on rerun.

PR 4: docs and walkthrough
9. README, ARCHITECTURE, workflow doc, CLAUDE.md/AGENTS.md, `local_test.sh`, `.gitignore`.

## Verification

- Unit: `make check` and `make test` green after every step; new tests never touch the network or
  external tools (patch `SecureSubprocess.run_secure`; simulate samtools/megahit outputs with the
  existing `tests/helpers_extraction.py` pattern).
- Walkthrough: `make pipeline` extended to run `status --init`, `blacklist add`, `select_datasets`,
  `status --stage selected`, a second `extract_target_reads --dry-run` that reports the skip, and
  `status --json` whose `stages` counts are asserted by the script.
- Real data: rerun `status --init` in the Wolbachia scratch directory (10 runs, two genomes) and check
  the matrix shows 3 extracted vs wMel (2 with zero reads), 2 vs wPip, 5 without metadata, the leftover
  SRR2517620 as untracked, and `blacklist add SRR2517418 --reason "16S amplicon mislabelled as WGS"`
  removes it from `--next`; then `extract_target_reads --assemble` a second time must skip all three
  wMel samples, and with `--force` must rebuild them. `make test-network` still passes.
- Compatibility: the existing `status --json` consumers (`tests/test_cli_status.py` keys `on_disk`,
  `wanted`) keep passing; a project without a registry behaves as today plus the bootstrap hint.
