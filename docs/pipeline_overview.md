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
plus `max_containment`) and `summary_containment.txt` (how many samples pass each containment step).
`plot_containment` draws the distribution. The registry keeps every screened accession with its
containment per genome, capped per genome by `--registry-max-screened` (default 5000).

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
flags.

## 3. Download

```bash
metaquest download_sra --accessions-file accessions.txt --fastq-folder fastq --max-workers 4
```

Reads land in `fastq/<accession>/` through fasterq-dump. As each download finishes the registry records
it as downloaded (with file sizes), failed (with the attempt count and message) or skipped (blacklisted
or cut by `--max-downloads`). Failed accessions are also written to `fastq/failed_accessions.txt` for a
retry, and `--report-file` writes one row per accession. A rerun skips accessions already on disk;
`--force` downloads them again. `--dry-run` reports the plan and touches neither the disk nor the
registry.

## 4. Analyse

```bash
metaquest sra_stats --fastq-folder fastq --output-report sra_statistics.csv
metaquest sra_validate --fastq-folder fastq
metaquest sra_profile_quality --fastq-dir fastq --accessions-file accessions.txt --detailed-reports
metaquest sra_dashboard --fastq-dir fastq --accessions-file accessions.txt --quality-profiles sra_quality_profiles
```

`sra_stats` writes read counts, GC content and read lengths per accession; `sra_validate` checks the
FASTQ files; `sra_profile_quality` grades each dataset and writes a summary JSON (and a JSON per
accession with `--detailed-reports`). Each of them records one analysis entry per accession in the
registry. `sra_dashboard` and `sra_compare` build interactive reports; with `--quality-profiles` the
dashboard reuses saved profiles instead of profiling the reads again.

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
accession.

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

## The walkthrough

`make pipeline` runs `local_test.sh` on the bundled sample data: `use_branchwater`, `parse_containment`,
`extract_branchwater_metadata`, `count_metadata`, `plot_metadata_counts`, `plot_containment`,
`select_datasets`, `status --init`, `blacklist`, `status --stage excluded`, `status`, a dry-run
`extract_target_reads`, and a `status --json` check of the registry counts. It needs no network access
and no external tools.

## External tools by stage

| Stage | Tool | Purpose |
|---|---|---|
| Screen | sourmash (Python package) | sketching the genome for `branchwater_search` |
| Download | sra-tools (`fasterq-dump`) | fetching reads |
| Extract | minimap2, samtools | mapping and filtering reads |
| Assemble | megahit | assembling the extracted reads |

The README's installation section lists them; `environment.yml` installs all of them together with MetaQuest.
