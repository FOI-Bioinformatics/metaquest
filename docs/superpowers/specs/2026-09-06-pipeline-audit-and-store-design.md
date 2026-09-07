# Pipeline audit (six stages) and a shared data store across organism projects

## Context

MetaQuest's pipeline runs screen, select, download, analyse, extract, assemble, and since 2026-09-06
each project keeps a registry (`metaquest_registry.json`) of what happened to each dataset. The
maintainer asked for three things: (1) an audit of each stage for speed, robustness and quality of
output with concrete improvements; (2) a common download folder so a second organism project reuses
reads already fetched by the first, with no second download and no second copy (the same metagenome
holds many organisms); (3) central tracking of which datasets exist, are complete, and which project
or organism used them. Decisions taken with the maintainer on 2026-09-06: store found through an
environment variable plus a user config file; SQLite catalogue in the store; per-accession symlinks
from projects into the store; implement everything the audit lists, not only the must-fix items.

Written in read-only plan mode against `main` at 44619cd, from the code, the docs, and the real
Wolbachia working directory (11 runs, two target genomes).

## Audit: what is weak today (evidence, file:line)

### 1. Screen
- One `requests.post`, 600 s timeout, no retry or backoff, 4xx and 5xx alike, whole response
  buffered, no response cache (`metaquest/data/branchwater_search.py:126-142`). The public index has
  been flaky before, so one transient error aborts the whole run.
- `parse_containment` keeps containment only; cANI and the sample metadata columns are read and then
  dropped (`metaquest/data/branchwater.py:266-274`, `plugins/formats/branchwater.py:71-79`).
- Merging is one dict then one DataFrame (no repeated concat): scale is fine (`branchwater.py:243-303`).

### 2. Select
- One ranking column, no top-N, no rule across several genomes, and no exclusion of accessions the
  registry already knows as downloaded or excluded (`processing/selection.py:16-76`,
  `cli/commands/select.py:29-45`).

### 3. Download
- fasterq-dump is called directly (`--threads N --progress <acc> -O <temp>`), no `prefetch`,
  `--split-3`/`--skip-technical` allowlisted but unused, output left as plain FASTQ
  (`metaquest/data/sra.py:240-254, 163-196`). The Wolbachia project holds 2.0 GB plain FASTQ for 11 runs.
- Completeness: "some `*.fastq*` file exists" is the only check (`sra.py:174-180`, `accession_has_fastq`
  `sra.py:48-56`); zero-byte or truncated files count as downloaded and are skipped forever. Real data:
  every run in the audit project holds exactly 300 000 read pairs while NCBI reports 48 M spots for
  ERR15315137; nothing notices. `Run_MD5`/`Run_Size` reach the registry and are never compared.
- No resume: an interruption wipes `<acc>_temp` and fasterq-dump's cache (`sra.py:228-277`); one retry
  round with `force=True` (deleting the partial work), no backoff, no error classes (`sra.py:360-427`);
  4 workers x 4 threads with no core awareness; `MAX_CONCURRENT_DOWNLOADS` unused (`constants.py:193`).
- Folders are relative to the working directory; no config, env var or data root anywhere. Download and
  extraction register their output folders as allowed roots, so a folder on another volume works today
  (`sra.py:220, 121`; `utils/security.py:50-63, 118-139`). Registry paths are stored as given, so the
  default relative `fastq` breaks when read from another directory (`registry.py:344-386`).
- Security-layer trap for new tools: the "flag takes no value" rule is hard-wired to fasterq-dump
  (`security.py:28-30, 182-190`), so `prefetch --progress <acc>` would swallow the accession; and the
  skip path calls `rmdir()` on an empty entry (`sra.py:156`), which fails on a dangling symlink.

### 4. Metadata and analysis
- `parse_metadata` reads `.//RUN/Total_spots`, `.//RUN/size`, `.//RUN/md5` as child elements
  (`metadata.py:252-256`); NCBI puts `total_spots`, `total_bases`, `size` as attributes of `<RUN>` and
  `md5` on `<SRAFile>`. Confirmed on the real XML: those columns are empty in every row, which is why no
  completeness check could work anyway.
- One efetch per accession with a fixed 0.5 s sleep, no API key (`metadata.py:82-129`); a batching
  client with API-key support already exists (`sra_metadata.py:60-162`) but has no disk cache.
- Three commands read the same FASTQ three ways: `sra_stats` parses every read in Python
  (`sra_metadata.py:330-409`); `sra_profile_quality` takes the first 10 000 reads, not configurable
  (`sra/analytics.py:141-224, 387-446`) and stores the per-read GC list in every profile JSON
  (`analytics.py:207, 437`); `sra_validate` parses the whole first file just to see it is non-empty
  (`cli/commands/sra_enhanced.py:266-292`). `sra_compare` and anomaly detection always reprofile
  (`analytics.py:448-457, 527, 583`). Nothing compares read counts with NCBI spots or checks an MD5.
- Only fasterq-dump has a pre-flight check; a missing minimap2, samtools or megahit surfaces mid-run as
  `SecurityError("Subprocess execution failed: [Errno 2] ...")` (`security.py:262-263`).

### 5. Extract
- `minimap2 -a -x <preset> -t N -o <sam> <genome.fna> <reads>` re-indexes the genome for every sample
  (`data/read_extraction.py:182-184`); the SAM sits uncompressed on disk; `samtools view -b -F 4` is the
  only filter, so secondary and supplementary alignments are counted and exported (`194-196`);
  `--secondary=no` is allowlisted but never passed; no MAPQ option; `samtools fastq` runs without
  threads (`211-226`); unequal mates are detected by a substring of minimap2's stderr (`39, 185`).

### 6. Assemble
- megahit runs with defaults (no `--presets`, k range or memory although allowlisted,
  `constants.py:76-92`); `intermediate_contigs/` is left behind (`read_extraction.py:359-417`);
  `summarise_contigs` reports count, total length, N50, largest only, no coverage or completeness
  (`420-455`).

### Cross-cutting
- Caches (`taxonomy_cache.csv`, quality profiles, report CSVs) are bare names in the working directory;
  nothing is reused across projects. The repository ignores `genomes/` but not `fastq/`, `targeted/`,
  `matches/`, `metadata/`, and no ignore template ships for user projects.

## Design: the shared store

### Location and discovery (`metaquest/store/resolve.py`)
Precedence: `--data-root` flag, then `METAQUEST_DATA`, then `store.root` in the project registry, then
`~/.config/metaquest/config.toml` (`$XDG_CONFIG_HOME` honoured; read with stdlib `tomllib`, written by a
small writer for the one key `[store] data_root`), then none (today's behaviour: `--fastq-folder` is a
plain folder). A root is accepted only if it holds `metaquest_store.json` or is being created by
`store_init`. The resolved root is registered once with `SecureSubprocess.add_allowed_root` in the CLI
layer so a store on `/Volumes/...` or a NAS passes `validate_path`. Decided defaults: `store.root` is
also written into the project registry (self-describing project; a stale root falls back to env/config
with a warning).

### Layout
```
<root>/metaquest_store.json           {"version": 1, "id": <uuid4>, "created": ...}
<root>/catalog.sqlite (+ .lock)        catalogue (section below)
<root>/locks/<ACC>.lock                per-accession download lock (two projects cooperate)
<root>/tmp/                            fasterq-dump temp, prefetch cache, <ACC>_temp staging
<root>/sra/<ACC>/<ACC>_1.fastq.gz, <ACC>_2.fastq.gz, <ACC>.json   (sidecar)
<root>/metadata/<ACC>_metadata.xml     shared from day one (small, one per accession)
```
Genomes stay per project in the first release (curated per organism). Sidecar `<ACC>.json`: schema,
accession, state (`complete|partial|failed|downloading`), layout, downloaded date, tool and version,
compression, `files[{name, bytes, md5, reads}]`, `reads_per_mate`, `bases_total`, `ncbi{spots, bases,
size, files[{name, md5}]}`, `completeness{method, ratio, verdict}`, `stats{...}` (analysis cache). The
store is self-describing: the catalogue is rebuilt from sidecars by `store_reindex`.

Complete means: exit 0, every mate has the same read count, `reads_per_mate == ncbi.spots` (ratio at or
above 0.99 accepted). Unknown spots gives `unverified` (usable by default). Fewer reads gives `partial`:
not used without `--accept-partial`; the 300 000-read runs land here loudly.

Compression: gzip in the store (pigz `-p N` when installed, else Python gzip level 6 in a thread);
`--no-compress` keeps plain files. minimap2 and megahit read `.gz` natively.

### Catalogue (`metaquest/store/catalog.py`, SQLite, stdlib)
`PRAGMA journal_mode=WAL` when the filesystem reports it, else rollback journal; every write also under
the registry's lock protocol on `catalog.sqlite.lock`, so two projects on one machine or a NAS serialise.
Tables: `store_meta(key, value)`; `datasets(accession PK, state, layout, compression, reads_per_mate,
bases_total, bytes_total, ncbi_spots, ncbi_bases, completeness, downloaded, tool_version, updated)`;
`files(accession, name, bytes, md5, reads)`; `projects(project_id PK, name, path, registry, created,
last_seen)`; `usage(accession, project_id, genome_id default '', stage in linked|downloaded|analysed|
extracted|assembled, first_used, last_used, detail; PK on the four)`; view `unused_datasets`. Queries:
which projects used SRR1; which datasets are unused; bytes per organism; which datasets organism X was
analysed in across projects.

### Project link: per-accession symlinks
`fastq/<ACC>` -> `<root>/sra/<ACC>` (relative when the store shares an ancestor, absolute otherwise;
`--link-mode absolute|relative|copy`). `scan_downloads`, `status` and `reconcile` follow symlinks
already; `accession_has_fastq` is hardened (below). `store_init`/`store_adopt` append `fastq/` to the
project's `.gitignore` when a `.git` directory exists and warn if `fastq` is tracked. `reconcile`
reports dangling links (new `ReconcileReport.dangling_links`), `store_link` repairs them.

### Identity and usage
Registry schema 2 (v1 files load unchanged; absent keys default to `{}`): top-level `project{id, name,
path, created, organisms[], genome_ids[]}` and `store{root, mode, linked[]}`. Every path written into
a project registry becomes relative to the project root when inside it, absolute otherwise (one helper
used by `_file_entries`, `record_genome`, `record_extraction`, `record_analysis`), which makes a project
directory movable. Usage rows are written at link or download (`genome_id=''`), at each analysis, at
extraction and assembly per genome; the catalogue call is isolated like `_notify_result` so a
catalogue failure never fails the science step. Moved or deleted projects: usage rows keep the last
known path; `store_gc` and `store_status` report a project as stale when its registry is gone or its
id differs; nothing is deleted automatically.

### Dedup in `download_sra`
Per accession, before any network call: sidecar `complete` (or `unverified`) -> link, usage `linked`,
`record_download(state="downloaded", attempt=False)`; `partial`/`failed` or files without sidecar ->
re-download with `--resume-partial` (default) else report; absent -> take `locks/<ACC>.lock`, download
into `<root>/tmp/<ACC>_temp`, count reads per mate, compare with `ncbi.spots` from the store's metadata
XML, compress, write the sidecar, move into `sra/<ACC>`, upsert the catalogue, link, record usage.
`accession_has_fastq` becomes: directory or symlink to one, holding a non-empty `*.fastq`, `*.fastq.gz`,
`*.fq` or `*.fq.gz`, and no sidecar in state `partial|failed|downloading`; the skip path unlinks a
dangling symlink instead of `rmdir()`.

### Commands (flat, new group "Store", registered in `cli/main.py`)
`store_init` (`--data-root`, `--project-name`, `--set-default`), `store_status` (`--json`,
`--verbose`), `store_adopt` (`--fastq-folder`, `--move|--copy`, `--dry-run`), `store_verify`
(`[ACC...]`, `--md5`, `--spots`, `--fix-state`), `store_usage` (`--accession`, `--project`,
`--organism`, `--unused`, `--bytes-by-organism`, `--json`), `store_gc` (`--dry-run` default,
`--yes`, `--older-than`, `--keep-partial`), `store_link`, `store_unlink`, `store_reindex`.
`--data-root` is added to `download_sra`, `download_metadata`, `status`, `sra_stats`, `sra_validate`,
`sra_profile_quality`, `sra_compare`, `sra_dashboard`, `extract_target_reads`; it never replaces
`--fastq-folder`. `status` gains a store section (root, store-backed vs local, partial, dangling).

### Migration for the Wolbachia project
`store_init --data-root <volume>/store --set-default`; in each project `store_adopt --dry-run` then
`store_adopt` (moves `fastq/<ACC>` into the store, dedups an accession already there after a byte
comparison, compresses, writes sidecars, replaces entries with symlinks, mints the project id, records
`linked` usage); `download_metadata` then `store_verify --spots` (the 300 000-vs-48 M runs become
`partial`); `download_sra --resume-partial` refetches only those. `store_adopt` is restartable and never
deletes a project file before the store copy verifies.

## Improvements per stage (all to be implemented; P1 first)

### Screen
- P1 robustness: `requests.Session` with `Retry(total=4, backoff_factor=2, status_forcelist=[429,500,502,503,504])`; other 4xx fail at once (`branchwater_search.py:126-142`).
- P1 speed: response cache keyed by sha256 of `{mins, ksize, scaled, threshold, server}` at `<output-parent>/.branchwater-cache/<key>.csv` with a timestamp sidecar; `--no-cache`, `--refresh`, `--max-cache-age-days`.
- P1 output: keep cANI and sample metadata in a long-format `parsed_containment_details.tsv` (accession, genome_id, containment, cANI, biosample, bioproject, assay_type, organism, geo, lat_lon), written by `_generate_containment_summary` (`branchwater.py:280-362`); `parsed_containment.txt` unchanged.
- P2 speed: stream the response with `iter_lines()` (`_parse_search_csv`, `145-158`).

### Select
- P1: `--top-n` (after threshold, filter and exclusions); `--skip-excluded` on by default and `--skip-downloaded` opt-in, both from `query(registry, ...)`; log "N already downloaded, M excluded, K selected".
- P2: `--genome-ids A B --require any|all` (mutually exclusive with `--genome-id`); `record_selection` gains rank, column and value per accession.

### Download
- P1 correctness (before the store): XML attribute fix; `record_metadata` keeps `run_total_spots`, `run_total_bases`, `platform`, `library_strategy`; chunked `count_fastq_reads` (`buf.count(b"\n") // 4`, `registry.py:534-540`); `verify_download` writes `download.complete{reads_r1, expected_spots, ratio, verdict}`; `accession_has_fastq` rejects empty files and honours verdicts; resume instead of wipe (`sra.py:262-276`), retries with `force=False` and `2**attempt` sleep (`sra.py:400-406`).
- P1 tools: generalise `FASTERQ_DUMP_BOOLEAN_FLAGS` to a per-tool map (`security.py:28-30, 182-190`); allowlist `prefetch` (`-O --max-size --progress --resume --version`), `--split-3`, `pigz` (`-p -f -k --version`); positional validation accepts an accession or a `.sra` path under an allowed root; `download_accession` runs `prefetch -O <cache> --max-size 100G <acc>` then `fasterq-dump --split-3 --skip-technical --threads N -O <temp> --temp <t> <cache>/<acc>/<acc>.sra`; the `.sra` is deleted after a verified conversion unless `--keep-sra`; compression as above.
- P2: error classes (network, disk-full, not-found, unknown; never retry not-found, abort on disk-full); core-aware default workers `min(MAX_CONCURRENT_DOWNLOADS, cpu_count // num_threads, 4)`.
- P3: `seqkit stats -T` fast path when installed.

### Metadata and analysis
- P1: batched efetch through `SRAMetadataClient._fetch_batch_metadata` (200 per call) with a splitter that writes one `<ACC>_metadata.xml` per package keyed on `.//RUN/@accession` (`metadata.py:189-219`); `--api-key` / `NCBI_API_KEY`; drop the flat sleep.
- P1: shared per-dataset statistics in the sidecar `stats` block (`metaquest/store/stats.py`: streaming line counts, uniform reservoir sample of `--sample-size` reads, default 10 000, seqkit when installed; invalidated by size or mtime), consumed by `calculate_read_statistics`, `_dataset_stats_row`, `analyze_fastq_quality`; per-read GC lists replaced by a histogram (`gc_histogram`, loader accepts both keys); `compare_datasets` and `detect_dataset_anomalies` accept `profiles` and reuse saved ones.
- P1: `sra_validate` checks the first record's shape instead of parsing a whole file, compares mate read counts, reports the sidecar verdict (`partial` fails with "N reads on disk vs M spots at NCBI"), `--md5` against `ncbi.files[].md5`; pre-flight `shutil.which` for minimap2, samtools, megahit with conda hints.
- P2: `calculate_read_statistics(max_reads=100000)` with a raw four-line reader; HTTP 400/404 not retried.

### Extract
- P1 speed: build `<output-folder>/.index/<genome>.<preset>.mmi` once per genome (`minimap2 -x <preset> -d <index> <fna>`; add `-d` to the allowlist and to `PATH_VALUE_FLAGS`), rebuild when the FASTA is newer, fall back to the FASTA on error.
- P1 output: `samtools view -b -F 0x904 -@ N` (drop unmapped, secondary, supplementary) and `-q <min-mapq>` when `--min-mapq > 0` (default 0, documented 20 with the divergent-strain caveat; add `-q` to the allowlist); record `filter_flags` and `min_mapq` in the extraction params; log how many records each filter removed.
- P1 speed: `samtools fastq -@ N`.
- P2: delete the SAM right after the BAM is written and place it under `--temp-folder`; replace the stderr substring with a pre-flight mate count from the cached stats; skip or require `--allow-truncated` for `partial` downloads.

### Assemble
- P1: remove `intermediate_contigs/` unless `--keep-intermediate`; `--assembly-preset {default,meta-sensitive,meta-large}` with `meta-sensitive` the default for these small targeted read sets (add `--presets` to the allowlist; reject mixing with explicit k flags); record in `record_assembly` params.
- P1 output: `assembly_coverage`: map the extracted reads back to `final.contigs.fa` (minimap2 then `samtools view -b -F 0x904`, `samtools view -c`), report `reads_mapped`, `mapping_rate`, `mean_depth` (labelled estimate) in the assembly stats.
- P2: `summarise_contigs` adds N90, GC, contigs at or above 1 kb, `total_bp / genome_length`; per-contig breadth via `samtools sort` + `samtools coverage` (allowlist `sort`, `coverage`, `-T`, `-m`).
- P3: probe multi-thread megahit on macOS and fall back to one thread on a crash, memoised per run.

## Files

- Create: `metaquest/store/{__init__,resolve,layout,sidecar,catalog,usage,stats}.py`,
  `metaquest/cli/commands/store.py` (nine command classes), tests `tests/test_store_*.py`.
- Modify: `metaquest/data/{sra,metadata,sra_metadata,registry,read_extraction,branchwater,branchwater_search}.py`,
  `metaquest/processing/selection.py`, `metaquest/sra/{analytics,reporting}.py`,
  `metaquest/core/constants.py` (allowlists, `DEFAULT_TOP_N` reuse, store constants),
  `metaquest/utils/security.py` (per-tool boolean flags, `.sra` positional, `-d`),
  `metaquest/cli/commands/{sra,metadata,select,status,read_extraction,branchwater_search,containment,sra_enhanced,sra_intelligent}.py`,
  `metaquest/cli/main.py`, `environment.yml` (add `pigz`, optional `seqkit`), README, `docs/pipeline_overview.md`,
  `docs/ARCHITECTURE.md`, `docs/branchwater_workflow.md`, CLAUDE.md/AGENTS.md, `local_test.sh` (a store walkthrough on the sample data), `.gitignore` guidance.
- Reuse: `registry_transaction` and `_acquire_lock` (`registry.py:115-191`) for the catalogue lock; `_notify_result` isolation pattern (`sra.py:32-45`); `SRAMetadataClient` batching (`sra_metadata.py:130-162`); `load_quality_profiles` (`sra/reporting.py:95-132`); `BranchWaterFormatPlugin.extract_metadata` (`plugins/formats/branchwater.py:120-140`); `tests/helpers_extraction.py::_fake_tools` for every new tool call.

## Implementation order (two tracks; each bundle is one reviewable PR stack)

Track A, data you can trust (must land first because the store's completeness verdict depends on it):
1. A1 Downloads you can trust: XML attribute fix, spots in the registry, chunked read count,
   `verify_download` and the `download.complete` verdict, hardened `accession_has_fastq`, resume and
   retry fixes, `status` shows the verdict. No new tools.
2. A2 Tool plumbing: per-tool boolean flags, `prefetch` + `--split-3` + pigz compression, `.sra`
   positional validation, core-aware workers, error classes.

Track B, the store:
3. B1 Store core: `resolve`, `layout`, `sidecar`, `catalog` (schema, lock, WAL fallback,
   `store_reindex`), `store_init`/`store_status`, `--data-root` plumbing on `download_sra` and `status`.
4. B2 Link, dedup, registry v2: store branch of `download_sra` with the per-accession lock and
   compression, symlinks, project-relative registry paths, `store_link`/`unlink`/`adopt`/`verify`,
   dangling-link reconcile, `.gitignore` guard, store section in `status`.
5. B3 Usage tracking: `usage.py`, hooks at link/download/analyse/extract/assemble, `store_usage`,
   `store_gc`, stale and moved projects.

Track C, stage quality (independent of B after A1):
6. C1 Screening and selection: retry and cache, details table, `--top-n`, skip flags, multi-genome rule.
7. C2 Extraction and assembly: `.mmi` reuse, `-F 0x904` and `--min-mapq`, `samtools fastq -@`, SAM
   lifetime, mate pre-count, `intermediate_contigs/` cleanup, presets, coverage, richer contig stats.
8. C3 Analysis stage: sidecar statistics cache, uniform sampling and `--sample-size`, GC histogram,
   profile reuse in compare and anomalies, `sra_validate` rewrite, pre-flight tool checks.
9. D Docs and walkthrough: README (store section, per-stage flags), `docs/pipeline_overview.md`,
   ARCHITECTURE (store package), workflow doc, CLAUDE.md/AGENTS.md, `local_test.sh` store walkthrough
   on the sample data, memory note.

Dependencies: A1 before everything; A2 before B2 (compression and prefetch live in the store download
path); B1 before B2 before B3; C1 and C2 need only A1; C3 needs B1 (sidecar module). Execution as in
the previous rounds: subagent-driven, one implementer and one reviewer per task, stacked branches per
bundle, final whole-branch review and one fix wave per track.

## Verification

- Unit: `make check` and `make test` green after every task; store tests use `tmp_path` stores and
  monkeypatched `METAQUEST_DATA`; no network, no external tools (every new tool call has a fake in
  `_fake_tools`); no test writes into the repository root or the user's config directory (monkeypatch
  `XDG_CONFIG_HOME`/`HOME`).
- Walkthrough: `make pipeline` extended with `store_init --data-root build/local_test/store`, a
  `download_sra --dry-run` that shows the link plan, `store_status --json` counts asserted.
- Real data (Wolbachia scratch directory, tools in the `mt` and `mq-megahit` conda envs): after A1,
  `parse_metadata` fills spots and size and `status` flags the runs as partial; after B2, `store_adopt`
  moves the 11 runs into a store once, the wMel and wPip projects both link to them, and
  `store_usage --accession SRR11011981` lists both projects with their genome ids; after C2, a rerun
  of extraction shows the secondary-alignment filter lowering mapped counts and the `.mmi` index reused;
  after C3, `sra_stats`, `sra_validate` and `sra_profile_quality` agree on read counts and finish in
  seconds instead of minutes on 200 MB files; `make test-network` still passes.
- Compatibility: a project that never runs `store_init` behaves as today; v1 registries load; the
  existing `status --json` keys are unchanged.

## Decisions taken in this plan (change if you disagree)

Compression on by default (`--no-compress` available); config file is TOML at
`~/.config/metaquest/config.toml`; the store root is also recorded in the project registry;
`unverified` datasets are usable; `store_gc` only reports unless `--yes`; `metadata/` shared from day
one, `genomes/` per project; flat `store_*` commands rather than nested subcommands; registry schema 2
with v1 still readable; `--skip-excluded` on by default; `meta-sensitive` the default megahit preset;
`--min-mapq` default 0.
