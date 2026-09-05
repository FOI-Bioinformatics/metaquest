# SRA information, statistics and validation

Three commands support the `download_sra` step. All read an accession list (one per line) and none
of them download reads.

## sra_info

Fetches NCBI metadata for the accessions and prints sizes, layouts, technologies and an estimated
download time. Requires `--email` (NCBI asks for it); `--api-key` raises the rate limit.

```bash
metaquest sra_info --accessions-file accessions.txt --email you@example.org --output-report sra_info_report.csv
```

## sra_stats

Reads the downloaded FASTQ files under `--fastq-folder` (default `fastq`) and reports read counts,
bases, read-length statistics, GC content and mean quality per accession.

```bash
metaquest sra_stats --fastq-folder fastq --output-report sra_stats.csv
```

## sra_validate

Checks that each accession folder holds readable FASTQ files; `--check-pairs` also verifies that
paired files have matching read counts.

```bash
metaquest sra_validate --fastq-folder fastq --check-pairs
```

Downloads themselves are documented in the README under "Downloading reads".
