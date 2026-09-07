# Pipeline Overview

MetaQuest moves a population of public SRA datasets through six stages: screen, select, download,
analyse, extract, assemble. Each stage has one or two commands. Every command records what it decided
in the project registry (`metaquest_registry.json` in the project root), so `status` can show where each
dataset stands at any time. This page lists the stages in order, the commands that belong to each, the
files they produce and what the registry keeps. The [README](../README.md) documents every flag; the
[Branchwater workflow](branchwater_workflow.md) covers the screening step in more depth.

## Stage summary

| Stage | Commands | Main output | Recorded in the registry |
|---|---|---|---|
| 1. Screen | `branchwater_search`, `use_branchwater`, `parse_containment` | `matches/*.csv`, `parsed_containment.txt` | containment and cANI per accession and genome |
| 2. Select | `select_datasets`, `blacklist` | `accessions.txt`, `blacklist.txt` | selection criteria and date; exclusions with reasons |
| 3. Download | `download_sra` | `fastq/<accession>/` | outcome per accession: downloaded, failed, skipped |
| 4. Analyse | `sra_stats`, `sra_validate`, `sra_profile_quality`, `sra_dashboard`, `sra_compare` | CSV, JSON and HTML reports | one analysis entry per accession |
| 5. Extract | `extract_target_reads` | `targeted/<accession>/<genome>_*.fastq.gz` | mapped read count, parameters, files |
| 6. Assemble | `extract_target_reads --assemble` | `targeted/<accession>/<genome>_assembly/final.contigs.fa` | contig count, total length, N50, megahit version |

Metadata commands run alongside stages 1 and 2 and are listed after the stages below.

## 1. Screen

Find the metagenomes that contain the target genome.

```bash
# Query the Branchwater index with a genome FASTA (writes branchwater/<name>.csv)
metaquest branchwater_search --genome-fasta genomes/GCF_000008025.1.fna

# Or place CSVs downloaded from the Branchwater web site in matches/ and convert them
metaquest use_branchwater --branchwater-folder branchwater --matches-folder matches

# Build the sample by genome containment table and its summary
metaquest parse_containment --matches-folder matches
```

`parse_containment` writes `parsed_containment.txt` (one row per SRA accession, one column per genome,
plus `max_containment`) and `summary_containment.txt` (how many samples pass each containment step), as
well as `parsed_containment_details.tsv` with cANI and the sample metadata columns kept alongside the
containment values. `plot_containment` draws the distribution. The registry keeps every screened
accession with its containment per genome, capped per genome by `--registry-max-screened` (default
5000).

`branchwater_search` retries a failing request automatically and caches a response under
`.branchwater-cache/` next to its output, keyed by the genome, thresholds and server; `--no-cache`,
`--refresh` and `--max-cache-age-days` control the cache.

## 2. Select

Decide which datasets are worth downloading, and which ones to leave out on purpose.

```bash
# Accessions whose containment for the genome is at or above the threshold
metaquest select_datasets --genome-id GCF_000008025.1 --threshold 0.5 --output accessions.txt

# The same, restricted to a metadata value
metaquest select_datasets --genome-id GCF_000008025.1 --threshold 0.5 \
    --metadata-column Sample_Scientific_Name --metadata-value "Drosophila melanogaster"

# Exclude a dataset and say why
metaquest blacklist --add SRR2517418 --reason "16S amplicon mislabelled as WGS"
```

The registry records the column, threshold, metadata filter, date and output file of the latest
selection; running `select_datasets` again replaces the previous selection. Exclusions are kept with
their reason in the registry and in `blacklist.txt`, and `download_sra` honours them without further
flags. `--top-n` keeps only the highest-containment accessions after every other filter; excluded
accessions are dropped by default (`--skip-excluded`) and `--skip-downloaded` also drops accessions the
registry already records as downloaded.

## 3. Download

```bash
metaquest download_sra --accessions-file accessions.txt --fastq-folder fastq --max-workers 4
```

Reads land in `fastq/<accession>/` by default: `prefetch` fetches the archive, then `fasterq-dump`
converts it (`--no-prefetch` calls fasterq-dump directly), and the resulting FASTQ files are
gzip-compressed unless `--no-compress` is given. As each download finishes the registry records it as
downloaded (with file sizes and a completeness verdict from comparing the read count against NCBI's
spot count), failed (with the attempt count and message) or skipped (blacklisted or cut by
`--max-downloads`). Failed accessions are also written to `fastq/failed_accessions.txt` for a retry,
and `--report-file` writes one row per accession. A rerun skips accessions already on disk; `--force`
downloads them again. `--dry-run` reports the plan and touches neither the disk nor the registry.

When a shared data store is configured (`--data-root`, `METAQUEST_DATA`, or the project's recorded
store), `download_sra` looks up each accession in the store first. A complete or unverified dataset is
linked into `fastq/<accession>` without a new download; a partial dataset is re-downloaded unless
`--accept-partial`; an accession absent from the store is downloaded once, compressed, and added to the
store for every project to reuse. A per-accession lock (`--lock-wait`) lets two projects on the same
store cooperate on the same accession without downloading it twice. See the store commands
(`store_init`, `store_status`, `store_adopt`, `store_usage`, `store_gc`, and related commands) in the
[README](../README.md#shared-data-store) and the store package section of
[ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## 4. Analyse

```bash
metaquest sra_stats --fastq-folder fastq --output-report sra_statistics.csv
metaquest sra_validate --fastq-folder fastq
metaquest sra_profile_quality --fastq-dir fastq --accessions-file accessions.txt --detailed-reports
metaquest sra_dashboard --fastq-dir fastq --accessions-file accessions.txt --quality-profiles sra_quality_profiles
```

`sra_stats` writes read counts, GC content and read lengths per accession; `sra_validate` checks the
FASTQ files, and with `--check-pairs` compares mate read counts, and with `--md5` compares each file's
checksum against the value recorded for it; `sra_profile_quality` grades each dataset and writes a
summary JSON (and a JSON per accession with `--detailed-reports`). Each of them records one analysis
entry per accession in the registry. `sra_dashboard` and `sra_compare` build interactive reports; with
`--quality-profiles` the dashboard reuses saved profiles instead of profiling the reads again.

Read counts are always exact; per-read metrics (GC content, quality, length) come from a sample of
`--sample-size` reads per dataset (default 10000, uniform by default, or `--sampler head`). For a
dataset held in the shared data store, this sample and the exact counts are cached in the store sidecar
(the `stats` block, invalidated when the FASTQ file's size or modification time changes), so
`sra_stats`, `sra_validate` and `sra_profile_quality` compute it once and reuse it rather than
re-reading the file each time.

## 5. Extract

Map each selected sample against the target genome and keep only the reads that map.

```bash
metaquest extract_target_reads --parsed-containment parsed_containment.txt \
    --genome-id GCF_000008025.1 --genome-fasta genomes/GCF_000008025.1.fna --threshold 0.1
```

Samples at or above the threshold in the containment table are mapped with minimap2 (preset `sr` by
default) and the mapped reads are written with samtools to `targeted/<accession>/`. The registry records
the number of mapped reads, the genome FASTA, preset, threshold and the files written, including samples
that mapped no reads. A rerun with the same genome, preset and threshold skips samples already recorded;
`--dry-run` lists what would be extracted and what would be skipped; `--force` redoes them.

The genome is indexed once per preset under `<output-folder>/.index/` and the index is reused across
samples rather than rebuilt each time. Unmapped, secondary and supplementary alignments are always
dropped; `--min-mapq` (default 0) additionally drops mapped records below a mapping-quality threshold.
`--fastq-folder` accepts a folder made of store symlinks the same way it accepts plain per-accession
folders, so extraction reads directly from the shared store when a project links into one.

## 6. Assemble

```bash
metaquest extract_target_reads --parsed-containment parsed_containment.txt \
    --genome-id GCF_000008025.1 --genome-fasta genomes/GCF_000008025.1.fna --assemble
```

With `--assemble` the extracted reads of each sample go through megahit into
`targeted/<accession>/<genome>_assembly/`. The registry records contig count, total length, N50, largest
contig, the megahit version and the parameters. An existing `final.contigs.fa` is kept unless `--force`;
an assembly folder without contigs is reported as interrupted, and `--force` removes it and runs again.
On macOS megahit runs single-threaded by default (`--assembly-threads` overrides).

`--assembly-preset` sets megahit's `--presets` value: `meta-sensitive` (the default), `meta-large`, or
`default` for no preset. `intermediate_contigs/` is removed after a successful assembly unless
`--keep-intermediate`. Unless `--no-coverage`, the extracted reads are mapped back onto the assembled
contigs to report `reads_mapped`, `mapping_rate` and an estimated `mean_depth` in the assembly stats.

## Metadata (alongside stages 1 and 2)

```bash
# Fast: the metadata embedded in the Branchwater CSVs
metaquest extract_branchwater_metadata --branchwater-folder branchwater --metadata-folder metadata

# Richer: NCBI metadata per accession, then a consolidated table
metaquest download_metadata --email you@example.org --matches-folder matches --metadata-folder metadata
metaquest parse_metadata --metadata-folder metadata --metadata-table-file metadata_table.txt

# Summaries
metaquest count_metadata --metadata-column Sample_Scientific_Name
metaquest plot_metadata_counts --file-path counts_Sample_Scientific_Name.txt
```

`download_metadata --accessions-file accessions.txt` fetches metadata for the selected datasets only.
The registry keeps the metadata file, run size, run checksum, library strategy and organism per
accession. Accessions are fetched from NCBI in batches (`--batch-size`, default 200, one XML per
accession written from each batch response) rather than one request per accession; `--api-key` (or the
`NCBI_API_KEY` environment variable) raises the request rate limit. With `--data-root`, each fetched
XML is also copied into `<data-root>/metadata/` so every project sharing the store has it.

## Project state at any point

```bash
metaquest status                              # inventory and the accession by stage matrix, per genome
metaquest status --stage extracted --genome GCF_000008025.1
metaquest status --next                       # commands that would advance the most datasets
metaquest status --init                       # build the registry for a project created before it existed
metaquest status --reconcile                  # record files removed by hand, register untracked work
metaquest status --export-tsv registry        # registry_datasets.tsv and registry_extractions.tsv
metaquest status --json                       # the same report for scripts
```

Presence is always re-checked on disk, so a folder deleted by hand shows up as missing rather than done.
Decisions and provenance (why a dataset was selected or excluded, when it was downloaded, how many
reads mapped) come from the registry. Commit `metaquest_registry.json` with the project if the decisions
should travel with the results.

`status --data-root` reports whether the project uses a shared data store, and if so the store's root,
whether each dataset is store-backed or local, and any dataset that is partial or whose store symlink is
dangling. `status --reconcile` records a dangling link the same way it records other files removed by
hand.

## The walkthrough

`make pipeline` runs `local_test.sh` on the bundled sample data: `use_branchwater`, `parse_containment`,
`extract_branchwater_metadata`, `count_metadata`, `plot_metadata_counts`, `plot_containment`,
`select_datasets`, `status --init`, `blacklist`, `status --stage excluded`, `status`, a dry-run
`extract_target_reads`, and a `status --json` check of the registry counts. It also walks through the
shared data store on the same sample data: `store_init`, a `download_sra --dry-run` against the store
(so no network access is needed), `store_status --json`, and `select_datasets --top-n`. It needs no
network access and no external tools; the store steps redirect `HOME` and unset `METAQUEST_DATA` so
they never touch a real store or the user's configuration file.

## External tools by stage

| Stage | Tool | Purpose |
|---|---|---|
| Screen | sourmash (Python package) | sketching the genome for `branchwater_search` |
| Download | sra-tools (`prefetch`, `fasterq-dump`) | fetching reads |
| Download | pigz (optional) | parallel gzip compression of downloaded FASTQ files |
| Extract | minimap2, samtools | mapping and filtering reads |
| Assemble | megahit | assembling the extracted reads |
| Analyse | seqkit (optional) | faster read statistics for `sra_stats` and `sra_profile_quality` |

The README's installation section lists them; `environment.yml` installs sra-tools, pigz,
ncbi-datasets-cli, minimap2, samtools, megahit and sourmash by default (pigz is a real dependency, not
optional there), and comments out `seqkit` as one additional, optional line.
