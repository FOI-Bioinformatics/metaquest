# MetaQuest

**MetaQuest** is a comprehensive command-line bioinformatics toolkit for analyzing metagenomic datasets based on genome containment. The software processes Branchwater CSV files, downloads SRA metadata from NCBI, and provides advanced visualization and analysis capabilities including diversity analysis, interactive plotting, and taxonomic validation.

## Features

- **Branchwater Integration**: Process and analyze containment data from JGI Branchwater
- **Intelligent SRA Management**: Advanced downloading with resume capability, quality profiling, and statistical reporting
- **Diversity Analysis**: Calculate alpha/beta diversity metrics with statistical testing
- **Interactive Visualizations**: Create dynamic plots (PCA, heatmaps, diversity comparisons)
- **Taxonomic Validation**: Validate species names against NCBI taxonomy database
- **Plugin Architecture**: Extensible format handlers and visualization plugins
- **Robust Implementation**: Type hints, comprehensive test coverage, numerical stability

## Installation

Requires Python 3.12 or newer.

### Quick Start (Recommended)
```bash
git clone https://github.com/FOI-Bioinformatics/MetaQuest.git
cd MetaQuest
make dev-install  # Installs with all development dependencies
```

### Alternative Installation
```bash
# Traditional approach (still supported)
pip install -r requirements.txt
pip install .
```

### Development environment

The conda environment pins Python 3.12, matching `requires-python` in `pyproject.toml`:

```bash
make env            # conda env create -f environment.yml (or update --prune if it exists)
make env-dev        # pip install -e ".[dev]" into the "metaquest" conda environment
conda activate metaquest
```

`make env-dev` runs `conda run -n metaquest pip install -e ".[dev]"`, so it always installs into
the environment named `metaquest` regardless of what is currently active; `conda activate
metaquest` is only needed afterward, to use the `metaquest` console script and the interpreter
day to day. The package should not stay installed (editable or otherwise) in another
interpreter's site-packages at the same time; `pip uninstall metaquest` from any other
environment before relying on the `metaquest` console script, so it resolves to the 3.12
environment's copy rather than a stale one.

### External tools

Download and assembly steps call command-line tools that are not Python packages:

| Tool | Used by |
|---|---|
| `fasterq-dump`, `prefetch` (sra-tools) | `download_sra` (prefetch first, then fasterq-dump; `--no-prefetch` skips prefetch) |
| `pigz` (optional) | `download_sra`, `store_adopt` (parallel gzip; falls back to Python's gzip module when absent) |
| `datasets` (ncbi-datasets-cli) | `genome_download`, `genome_prepare`, `download_test_genome` |
| `minimap2`, `samtools` | `extract_target_reads` |
| `megahit` | `extract_target_reads --assemble` |
| `seqkit` (optional) | `sra_stats`, `sra_profile_quality` (faster read statistics; falls back to a plain Python reader when absent) |

`environment.yml` installs all of them together with MetaQuest:

```bash
conda env create -f environment.yml
conda activate metaquest
```

Map plots need the optional extra: `pip install 'metaquest[maps]'`.

### Development Commands
```bash
make help           # Show all available commands
make test          # Run tests with coverage
make lint          # Run code quality checks
make check         # Full quality validation
make clean         # Clean build artifacts
```

## Usage with Branchwater

### 0. Getting a Target Genome FASTA

`branchwater_search` (step 1 below) sketches a genome FASTA, so a genome is needed first. Use
`genome_prepare` to search GTDB by species or genus, download the matching assemblies, and write a
manifest CSV (and a registry entry for each genome); `genome_download` also fetches assemblies by
accession, species, or genus, but leaves the downloaded `genomes/ncbi_dataset.zip` where it lands and
does not record anything in the registry:

```bash
metaquest genome_prepare --species "Lactobacillus crispatus" --output-dir genomes
metaquest genome_download --accessions GCF_000006945.2 --output-dir genomes
```

`genome_search` looks up GTDB accessions without downloading anything (`--species`/`--genus`, `--all`
for every genome instead of just representatives, `--format list`/`tsv`).

### 1. Getting Containment Files from Branchwater

Search the Branchwater index directly from a genome. The command sketches the FASTA with sourmash
(install the extra with `pip install 'metaquest[sourmash]'`, or use `environment.yml`), queries the
public search API and writes `branchwater/<genome>.csv` in the layout the next steps read:

```bash
metaquest branchwater_search --genome-fasta genomes/GCF_000008025.1.fna --threshold 0.1
```

The CSV carries the accession, containment and cANI, nothing else; the server is sent `--threshold` but
is not always relied on to apply it, so the client drops any returned match below the threshold as well.
If you already have a k=21, scaled=1000 sourmash signature, pass `--signature file.sig` instead of the
FASTA.

Alternatively, search at [https://branchwater.sourmash.bio/](https://branchwater.sourmash.bio/) in a
browser, download the CSV, and save it to the same folder. A CSV from the web site carries extra sample
columns (organism, biosample, collection date, and similar) that the API search does not return; step 3
below only has something to extract when the CSV came from the web site.

A search that fails with a transient network or server error retries automatically (4 attempts with
backoff). A repeated search with the same genome, thresholds and server reads a cached response from
`.branchwater-cache/` next to the output CSV instead of querying again; `--no-cache` disables the cache
for one run, `--refresh` forces a new query and updates the cache, and `--max-cache-age-days N` expires
a cached entry older than N days.

### 2. Process Branchwater Files

Process the downloaded files to prepare them for the MetaQuest pipeline:

```bash
metaquest use_branchwater --branchwater-folder /path/to/branchwater/files --matches-folder matches
```

* `branchwater-folder`: The directory where Branchwater CSV files are located.
* `matches-folder`: The directory where the processed files will be saved.

A CSV that cannot be read (missing, unreadable, or the wrong format) makes `use_branchwater` exit with
status 1 rather than skipping it silently; `parse_containment` and `extract_branchwater_metadata` do the
same for a match file they cannot read.

### 3. Extract Basic Metadata from Branchwater Files (Optional)

You can extract basic metadata directly from Branchwater CSV files without downloading from NCBI, but
only when the CSVs carry sample columns to begin with. A CSV from `branchwater_search` (the API path)
has only the accession, containment, and cANI, so this step's output is just those two columns; a CSV
downloaded from the Branchwater web site carries an `organism` column and similar sample fields, which
this step does extract:

```bash
metaquest extract_branchwater_metadata --branchwater-folder /path/to/branchwater/files --metadata-folder metadata
```

> Note: a few Branchwater columns are renamed to canonical names in the output; in
> particular `organism` becomes `Sample_Scientific_Name`. Use the output column names
> (e.g. `--metadata-column Sample_Scientific_Name`) in later `count_metadata` /
> `single_sample` steps. On the API path, `Sample_Scientific_Name` does not exist, since
> there is no `organism` column to rename; use `download_metadata` (step 5) instead.

### 4. Summarizing Results

After processing the Branchwater files, you can summarize the results:

```bash
metaquest parse_containment --matches-folder matches --parsed-containment-file parsed_containment.txt --summary-containment-file summary_containment.txt --step-size 0.05
```

*Example output:* parsed_containment.txt (samples x genomes) and summary_containment.txt (counts per
containment step, named here explicitly; the default summary file name is `top_containments.txt`, and
the default `--step-size` is 0.1).

### 5. Downloading Metadata from NCBI (richer alternative to step 3)

For more comprehensive metadata, you can download it from NCBI. This writes one XML per accession into
`metadata_folder`; it does not add columns to the Branchwater CSV or to `parsed_containment.txt`, and
step 6 (`parse_metadata`) is what turns those XML files into a table:

```bash
metaquest download_metadata --matches-folder matches --metadata-folder metadata --threshold 0.95 --email [EMAIL]
```

* `matches_folder`: Directory containing match files.
* `metadata_folder`: Directory where the metadata files will be saved.
* `threshold`: Only consider matches with containment at or above this threshold.
* `--accessions-file`: Fetch metadata for exactly these accessions instead of scanning the matches
  folder; the natural choice after `select_datasets` has already narrowed the list down.
* `--batch-size`: Accessions per NCBI request, 1-500 (default 200); `--api-key` (or the `NCBI_API_KEY`
  environment variable) raises the request rate limit.

If you plan to run `download_sra` afterwards, run `download_metadata` first: `download_sra` can only
compare a download's read count against NCBI's recorded spot count (the `complete`/`truncated`/
`unverified` verdict) when that count is already on disk, either from this step or from a store's own
metadata folder. Running `download_metadata` after the fact does not retroactively add a verdict to
downloads that already finished; `store_verify --spots` (see "Shared data store" below) can compute one
later, but only if a metadata XML for the accession exists somewhere it looks by then.

### 6. Parsing Metadata

Once the metadata is downloaded, you can parse it to generate a more concise and readable format. The
default output name, `metadata_table.txt`, matters: `count_metadata`, `single_sample`, and
`check_metadata_attributes` all look for a file by that exact name when no explicit metadata file is
given (see step 8), so naming it something else here means passing that name explicitly to every later
command too.

```bash
metaquest parse_metadata --metadata-folder metadata --metadata-table-file metadata_table.txt
```

*Example output:* metadata_table.txt

### 7. Check Metadata Attributes

This step helps in understanding the distribution of metadata attributes:

```bash
metaquest check_metadata_attributes --file-path metadata_table.txt --output-file metadata_table_overview.txt
```

*Example output:* metadata_table_overview.txt

### 8. Counting metadata values

`count_metadata`, `single_sample` and `check_metadata_attributes` use `metadata_table.txt` from
`parse_metadata` when it exists and otherwise fall back to `metadata/branchwater_metadata.txt` from
step 3, so either metadata route works with the defaults:

```bash
metaquest count_metadata --metadata-column Sample_Scientific_Name --threshold 0.9 --output-file metadata_counts.txt
```

This writes `metadata_counts.txt` (default output name, one row per value, one column per genome) and
`metadata_counts_stats.txt`. `--threshold` defaults to 0.5 when omitted.

To instead see the distribution of genomes across datasets grouped by a metadata attribute, pass
`--summary-file` and `--metadata-file` explicitly:

```bash
metaquest count_metadata --summary-file parsed_containment.txt --metadata-file metadata_table.txt --metadata-column Sample_Scientific_Name --threshold 0.95 --output-file genome_counts.txt
```

*Example output:* genome_counts.txt

### 9. Single Sample Analysis

To analyze a single sample from the summary, you can use the `single_sample` command:

```bash
metaquest single_sample --summary-file parsed_containment.txt --metadata-file metadata_table.txt --summary-column <genome column from parsed_containment.txt> --metadata-column Sample_Scientific_Name --threshold 0.95
```

### 10. Checking What Is Already Available Locally

Before downloading, use `status` to see which reads, metadata, and genomes are already present
so nothing is fetched twice. Given a wanted list it also reports what is still missing:

```bash
metaquest status --accessions-file accessions.txt --list-missing
metaquest status --parsed-containment parsed_containment.txt --json
```

The command reconciles against the standard layout (`fastq/<ACC>/`, `metadata/<ACC>_metadata.xml`,
`genomes/<GCF>.fna`); override the locations with `--fastq-folder`, `--metadata-folder`, and
`--genomes-folder`. Genome downloads skip accessions whose FASTA already exists; pass `--force` to
re-download or `--dry-run` to report present-vs-missing without downloading:

```bash
metaquest genome_download --accessions GCF_000006945.2 --dry-run
```

`status` also reads the project registry (see "Project state" below): `--stage` lists the accessions
in one pipeline stage, `--genome` restricts the extraction and assembly stages to one target genome,
`--reconcile` records files that were removed by hand and registers work found on disk that the
registry did not know about, `--export-tsv` writes the registry as two TSV tables, and `--next`
suggests which command would advance the most datasets.

### Project state

MetaQuest keeps a journal of every dataset a project touches in `metaquest_registry.json` in the
project root: which genomes it was screened against and with what containment, whether it was
selected (and by which threshold and filter), whether it is excluded and why, the download outcome
with file sizes and dates, which analyses ran, and for each target genome the number of mapped reads
and the assembly statistics. Commands update it as they finish; `status` reads it and always
re-checks the disk, so a deleted folder shows up as missing rather than done.

```bash
metaquest status --init                      # create the registry from an existing project
metaquest status                             # accession by stage matrix, per genome
metaquest status --stage extracted --genome GCF_000008025.1
metaquest status --next                      # which commands would advance the most datasets
metaquest status --reconcile                 # record files removed by hand, register untracked work
metaquest status --export-tsv registry       # registry_datasets.tsv and registry_extractions.tsv
metaquest blacklist --add SRR2517418 --reason "16S amplicon mislabelled as WGS"
metaquest results_table --output results.tsv
```

`results_table` writes one row per screened accession and genome, joining containment, selection,
exclusion, download, run size, mapped reads, reference coverage and assembly statistics.

`extract_target_reads` skips samples already extracted or assembled with the same genome, preset
and threshold; pass `--force` to redo them.

Each `select_datasets` run replaces the previous selection: only the accessions of the latest run
count as selected, and `status --stage selected` shows which those are. To keep the registry small
on a broad search, `branchwater_search` and `parse_containment` record at most 5000 accessions per
genome, the ones with the highest containment; change that with `--registry-max-screened`. The
match CSVs always keep every hit.

`status --next` lists a runnable `download_sra` command for accessions ready to download; for
accessions still selected under a `select_datasets --no-skip-excluded` run, it instead prints a
`select_datasets ... --skip-excluded` command (reproducing that run's genome, threshold, metadata and
top-N and run filter criteria) so rerunning it drops the excluded accessions from the selection file first.

Commit `metaquest_registry.json` with your project if you want the decisions to travel with the
results.

### Shared data store

A second organism project studying the same metagenomes does not need its own copy of the reads.
`metaquest/store` keeps one gzip-compressed copy of each SRA accession, tracked in a small SQLite
catalogue, and links it into every project that uses it.

A store must exist before `download_sra` links a dataset or `store_adopt` runs; without one, every
command behaves as it does today, with plain per-project folders. MetaQuest finds the store root in
this order: the `--data-root` flag, then the `METAQUEST_DATA` environment variable, then `store.root`
recorded in the project registry, then `[store] data_root` in `~/.config/metaquest/config.toml`
(honouring `$XDG_CONFIG_HOME`). If the configured root is unreachable, `status` and the analysis
commands warn and continue without the store; `download_sra` and the `store_*` commands stop.

```bash
metaquest store_init --data-root /data/metaquest_store --set-default   # create a store, remember it
metaquest store_adopt --fastq-folder fastq --dry-run                   # preview folding this project in
metaquest store_adopt --fastq-folder fastq --move                      # move reads into the store, link back
metaquest store_adopt --fastq-folder fastq --copy                      # copy into the store, keep the project folder
metaquest store_status --json                                          # dataset and byte counts
metaquest store_status --verbose                                       # also list every dataset in the store
metaquest store_usage --accession SRR11011981                          # which projects used this run
metaquest store_usage --project my_project                             # every dataset that project has used
metaquest store_usage --organism GCF_000008025.1                       # every dataset used for that target genome
metaquest store_usage --unused                                         # datasets in the store with no recorded usage
metaquest store_gc --dry-run                                           # candidates for removal, nothing deleted
```

`store_adopt --move` (the default) removes each accession's project folder and replaces it with a link
to the store's copy. `store_adopt --copy` leaves the project's folder exactly as it was, real and
unlinked, whether the accession is freshly adopted or turns out to duplicate what the store already
holds; a folder with no FASTQ files at all (an empty or interrupted download) is never adopted either
way and is reported separately, not silently skipped. A `--copy` adoption still records the copying
project as a user of the dataset (the same as a linked one), so `store_usage` and `store_gc` do not
treat a copied dataset as unused.

Inside a project, each linked accession appears as `fastq/<ACCESSION>`, a symlink to
`<data-root>/sra/<ACCESSION>` (relative when the store and project share a parent folder, absolute
otherwise; override with `--link-mode`). `store_init` and `store_adopt` add `fastq/` to the project's
`.gitignore` when the project is a git repository. `store_link` and `store_unlink` manage one link at a
time; `store_verify` checks a dataset's files against its recorded size, md5 (`--md5`) or NCBI spot
count (`--spots`). A `--spots` check reads the expected count from the dataset's own sidecar if it has
one, and otherwise falls back to `<data-root>/metadata/<ACCESSION>_metadata.xml` and then to
`metadata/<ACCESSION>_metadata.xml` in the calling project, so a count found by `download_metadata`
either at download time or afterwards is picked up. `--fix-state` rewrites the sidecar and catalogue
entry when a check finds a mismatch, but only promotes a dataset to `complete` after actually reading
through its files (via the `--spots` check, or, if `--spots` was not requested, a read-through
`--fix-state` performs itself); matching size and md5 alone is not enough, since a file can still be a
truncated or corrupt gzip stream underneath. `--rescan` rebuilds a dataset's recorded file list from
what is actually on disk before checking, for files added or removed by hand; `store_reindex` rebuilds
the SQLite catalogue from the sidecar files if it is ever lost, replaying an append-only journal under
the store's `journal/` folder to restore which projects used which datasets. If that replay restores no
project at all while the rebuilt catalogue still holds datasets, `store_reindex` sets a
`rebuilt_without_projects` catalogue flag (every dataset would otherwise look unused); `store_gc` then
refuses to run until each project using the store has run `store_init` or `store_link` again and
`store_gc` is passed `--accept-rebuilt`, or until a later `store_reindex` restores at least one project
on its own. `store_gc --json` prints the report (or, on a refusal, `{"error": ...}`) as one JSON object;
with no store configured at all, `store_gc`, `store_status` and `store_usage` all print that same
`{"error": ...}` shape under `--json` instead of the plain-text hint.
`store_gc` never removes a dataset a project still links or another run is working on; `--older-than
DAYS` restricts it to datasets downloaded at least that many days ago, `--keep-partial` never removes a
`partial` dataset, and it also reports (and, with `--yes`, removes) leftover temp artifacts under the
store's `tmp/` folder left by an interrupted download or adoption.

A per-accession lock (`locks/<ACCESSION>.lock`, with a heartbeat) stops two projects from downloading
the same accession into the store at once; a lock with no heartbeat for 10 minutes is treated as
abandoned and taken over. `--lock-wait SECONDS` bounds how long `download_sra` and `store_adopt` wait
for another project's lock on the same accession before giving up; the default, 0, means wait without a
time limit, for as long as the other project's heartbeat keeps showing it is still working (an abandoned
lock is still taken over after 10 minutes either way). A positive value bounds the wait to that many
seconds instead.

On a store shared over a network filesystem or between machines, each project records the hostname it
last ran on; a project not seen from the current machine looks stale here even when it is still active
on another one. `store_gc` leaves a stale project's datasets alone unless `--include-stale` is given,
since staleness is only ever judged from the machine running the command. The catalogue uses a rollback
journal rather than WAL on such filesystems, trading some write throughput for correctness when several
hosts write to it at once.

Every stored dataset carries a completeness verdict: `complete` (the read count per mate matches NCBI's
recorded spot count, at or above a 0.99 ratio), `partial` (fewer reads than expected; not used by
`download_sra` unless `--accept-partial` is given, and re-downloaded by default unless
`--no-resume-partial`), or `unverified` (no expected spot count was found: no metadata XML was present
at download or adoption time, from `download_metadata` or `store_adopt --metadata-folder`; usable by
default). `store_verify --spots` and `status --reconcile` compute a verdict for a dataset that lacks
one, now also checking `<data-root>/metadata/` and the calling project's own `metadata/` folder for an
XML `download_metadata` wrote after the fact (see "Downloading reads" above and `store_verify` below).

A download into the store runs `prefetch` (fixed at `--max-size 100G`) before `fasterq-dump`; if the
kept `.sra` archive or a temporary build folder grow past 1 GB combined, the download summary warns and
names the folder, and `store_gc --dry-run` separately lists such leftovers as removal candidates. Without
`--temp-folder`, `fasterq-dump`'s own scratch files default to `<data-root>/tmp/<ACCESSION>_fqtmp`
(inside the store, not the system temp directory) when a store is configured.

`--data-root` is accepted by `download_sra`, `download_metadata`, `status`, `sra_stats`, `sra_validate`,
`sra_profile_quality`, and `extract_target_reads`; it never replaces `--fastq-folder`, which still names
where the project expects its reads (as a folder or as the store's symlink). `sra_compare` and
`sra_dashboard` do not take `--data-root` themselves; instead they reuse quality profiles a store-aware
`sra_profile_quality` run already saved, via `--quality-profiles`. With a store configured, `status`
flags a wanted accession whose `fastq/<ACC>` is a link into the store but whose store dataset is not yet
`complete` or `unverified` (still `downloading`, `failed`, or `partial`) as linked to a store dataset
that is not complete, rather than simply listing it as missing.

### 11. Targeted Read Extraction Before Assembly

To assemble only the reads relevant to a target genome (a small, targeted assembly rather than a
whole-metagenome assembly), use `extract_target_reads`. For every sample whose containment for the
target genome meets the threshold, it maps the reads with minimap2 and keeps the mapped reads with
samtools, writing them to `targeted/<ACC>/`:

```bash
metaquest extract_target_reads \
  --parsed-containment parsed_containment.txt \
  --genome-id GCF_000006945.2 \
  --genome-fasta genomes/GCF_000006945.2.fna \
  --fastq-folder fastq --output-folder targeted \
  --threshold 0.5 --preset sr
```

Use `--dry-run` to list the qualifying samples without running any tool, `--preset` to match the read
type (`sr` for Illumina, `map-ont`/`map-pb`/`map-hifi` for long reads), `--threads` for minimap2 and
samtools (default 4), and `--assemble` to run megahit on each sample's extracted reads. `--data-root`
names a shared data store (see "Shared data store" above) to resolve `--fastq-folder` against, the same
way other store-aware commands do. This step requires `minimap2`, `samtools`, and (for `--assemble`)
`megahit` to be installed and on the PATH; `--dry-run` never checks for them, since it runs no tool. A
sample selected by containment but not yet downloaded is common on a broad search, so `--dry-run` prints
one summary line for however many such samples there are, rather than one warning line per sample. A
`fastq/<ACC>` symlink whose target is missing (typically an unmounted shared store) is logged as one
WARNING naming every dangling link found, rather than failing silently for each one.

On macOS the assembly defaults to a single thread, because megahit 1.2.9's parallel k-mer sorting step
is unstable on recent macOS releases (mapping with minimap2/samtools still uses `--threads`). Override
the assembly thread count explicitly with `--assembly-threads` if your megahit build handles more. A
megahit failure is reported with the tool's own error message (the last few lines of its stderr), not
just the exit code. megahit needs FIFOs for its scratch files, which some filesystems do not provide
(ExFAT, some network shares); `--temp-folder DIR` points megahit's scratch elsewhere, at a local POSIX
filesystem, when the default location does not support them. Without `--temp-folder`, each run creates
its own scratch folder directly under `--output-folder`, named `.megahit-tmp-<random suffix>` (a sibling
of the per-accession assembly directories, never inside one, since megahit refuses to run when its `-o`
directory already exists), and removes it afterwards, even on failure; the random suffix lets two
concurrent runs sharing one output folder keep separate scratch space. A macOS ExFAT or SMB volume's
stray `._*` AppleDouble sidecar files are ignored wherever MetaQuest lists a folder's contents, so they
never look like real FASTQ or genome files.

Mapped reads always drop unmapped, secondary and supplementary alignments; `--min-mapq` additionally
discards records below a mapping-quality threshold (default 0, keep every mapped record). A value of
20 is reasonable for a close relative of the target genome, but a divergent strain can genuinely map
with a low MAPQ, so raising the threshold can discard real matches. For each sample with kept reads,
`samtools coverage` on the kept alignments (it also skips duplicate and QC-fail reads by default) writes
`targeted/<ACC>/<genome>_coverage.tsv`, and the
registry records the breadth of the reference covered at 1x or more and the length-weighted mean depth
(a failure of this step is logged as a warning and leaves the extracted reads in place). `--assembly-preset` selects
megahit's `--presets` value: `meta-sensitive` (the default, suited to these small targeted read sets),
`meta-large`, or `default` (no `--presets` flag). Unless `--no-coverage`, the extracted reads are mapped
back onto the assembled contigs to report a mapping rate and estimated mean depth alongside the other
assembly statistics.

A sample already extracted or assembled with the same genome FASTA, preset, and threshold is skipped
on a rerun, including samples that mapped zero reads; an assembly folder with no contigs is reported
as interrupted with a hint to rerun. Pass `--force` to redo extraction and assembly regardless.

## Advanced SRA Operations

### Choosing which datasets to download

```bash
metaquest select_datasets --threshold 0.9 --output accessions.txt
metaquest select_datasets --genome-id GCF_000008025.1 --threshold 0.5 \
    --metadata-column geo_loc_name_country_calc --metadata-value France --output accessions.txt
metaquest select_datasets --threshold 0.9 --top-n 20 --output accessions.txt
metaquest select_datasets --genome-ids GCF_000008025.1 GCF_000006945.2 --require all --threshold 0.5 \
    --output accessions.txt
metaquest select_datasets --threshold 0.5 --max-run-size 2G --min-spots 1000000 --platform ILLUMINA \
    --output accessions.txt
```

`--parsed-containment` names the input table (default `parsed_containment.txt`). `--genome-id` ranks on
one genome column (default `max_containment` when omitted); `--genome-ids` ranks on several columns
together instead, with `--require any`/`all` deciding whether one or every listed column must meet the
threshold. `--top-n N` keeps only the N accessions with the highest containment after every other filter
is applied; excluded accessions are skipped by default (`--skip-excluded`, on unless `--no-skip-excluded`
is given), and `--skip-downloaded` additionally drops accessions the registry already records as
downloaded, useful when re-running selection on an expanded search. `--no-record`
still writes the output file and logs the counts, but does not record the selection in the registry, so
`status` is left unchanged; use it for an exploratory run that should not redefine the target list.
Because `status --next` points `download_sra` at the recorded selection's file, `--no-record` refuses to
overwrite a file that a recorded selection names (including the default `accessions.txt`); give it an
`--output` that names a scratch file instead.

`--max-run-size BYTES` (a byte count; suffixes K, M, G and T are powers of 10, so `500M` is 500000000
bytes), `--min-spots N`, `--max-spots N` and `--platform NAME` (case-insensitive, e.g. `ILLUMINA`) filter
on the `Run_Size`, `Run_Total_Spots` and `Platform` columns of the NCBI metadata table (`metadata_table.txt`
from `parse_metadata`, found automatically, or `--metadata-file`). They apply after the threshold,
exclusions and the metadata filter and before `--top-n`. A run absent from the table or without a value in
the filtered column is dropped, since the bound cannot be checked for it, and the log counts such runs.
Branchwater-derived tables carry none of these columns; the filter is then skipped with a warning. The
log also reports the summed size of the selected runs.

`accessions.txt` is the input for `download_sra`, which writes
`fastq/<accession>/<accession>_1.fastq.gz` (and `_2` for paired runs; gzip-compressed by default, see
below), the layout `status`, `sra_stats`, `sra_profile_quality`, `sra_dashboard` and
`extract_target_reads` read.

### Downloading reads

`download_sra` runs `fasterq-dump` in parallel, skips accessions whose FASTQ files already exist
(unless `--force`), retries failures, and writes `fastq/failed_accessions.txt` for reruns. Ctrl-C
cancels downloads that have not started yet, stops the running `prefetch`/`fasterq-dump` processes for
the ones in progress, and lets the run exit rather than leaving orphaned tool processes behind:

```bash
metaquest download_sra --accessions-file accessions.txt --fastq-folder fastq --max-workers 4 --num-threads 4
metaquest download_sra --accessions-file accessions.txt --dry-run
metaquest download_sra --accessions-file accessions.txt --report-file download_report.csv
```

`--report-file` writes one row per accession with the status `downloaded`, `failed`, `already_present`,
`blacklisted`, or `skipped` (accessions skipped by `--max-downloads`). To see sizes and sequencing
technology before downloading, use `sra_info` (needs an email for NCBI); see
`docs/SRA_ENHANCED_FEATURES.md`. `sra_info` filters per experiment package, not per run: it lists every
run of each experiment package that a requested run, experiment, sample, study, BioProject or BioSample
accession matches (including sibling lanes or replicates of the same experiment as a requested run), and
drops runs of packages that match none of the requested accessions. If a reply would be filtered down to
nothing, it lists every run returned and logs a warning instead.

By default, `download_sra` runs `prefetch` before `fasterq-dump` (`--no-prefetch` reverts to calling
`fasterq-dump` directly) and gzip-compresses the resulting FASTQ files (`--no-compress` leaves them
plain; pigz is used for compression when installed, otherwise Python's gzip module). `--sra-cache DIR`
sets where prefetch keeps its downloaded `.sra` archives (default `<fastq-folder>/.sra-cache`); pass
`--keep-sra` to retain a verified archive instead of deleting it after conversion.
`--verify-downloads` (on by default; `--no-verify-downloads` turns it off) compares each download's
read count against NCBI's recorded spot count and records a verdict in the registry: `complete` (ratio
at or above 0.99), `truncated` (fewer reads than expected), or `unverified` (the expected spot count is
not known, e.g. `download_metadata` was never run for this accession). This check runs even for a
project with no store configured. `--redownload-truncated` re-fetches an accession whose registry
verdict is `truncated` instead of skipping it on a rerun; a dataset held in the shared store instead
carries the store's own verdict (`complete`, `partial`, or `unverified`, described in "Shared data
store" above).

`--temp-folder DIR` sets where `fasterq-dump` writes its scratch files while converting each accession;
without it, a plain per-project download uses the system's temp directory, and a download into a shared
store (see "Shared data store" above) defaults to `<store>/tmp/<ACCESSION>_fqtmp` instead.
`extract_target_reads` (see "Targeted Read Extraction Before Assembly" above) accepts the same flag for
megahit's scratch files.

On a real run, `download_sra` also honours the project registry: accessions excluded with
`blacklist` are skipped automatically, without needing `--blacklist blacklist.txt` on every call
(though that flag still works). `--dry-run` neither reads nor writes the registry, so it does not
apply blacklist exclusions and does not record anything. Use `blacklist` to record an exclusion
with a reason, keeping `blacklist.txt` and the registry in step:

```bash
metaquest blacklist --add SRR2517418 --reason "16S amplicon mislabelled as WGS"
metaquest blacklist --list
metaquest blacklist --remove SRR2517418
```

### SRA Quality Profiling

`sra_profile_quality`, `sra_compare` and `sra_dashboard` name the FASTQ folder with `--fastq-dir`,
while the rest of the pipeline (`download_sra`, `sra_stats`, `sra_validate`, `extract_target_reads`,
`status`) uses `--fastq-folder`; both flags point at the same kind of folder, one per-accession
directory of downloaded reads, the naming just differs by command.

Generate comprehensive quality profiles for downloaded SRA datasets:

```bash
# Profile multiple datasets with detailed reports
metaquest sra_profile_quality \
    --accessions-file accessions.txt \
    --fastq-dir fastq \
    --output-dir quality_profiles \
    --detailed-reports

# Profile single dataset
metaquest sra_profile_quality \
    --accession SRR123456 \
    --fastq-dir fastq \
    --include-contamination
```

Read totals are always exact; per-read metrics such as GC content, quality and length are computed
from a sample of the reads, `--sample-size` per dataset (default 10000), drawn uniformly across the
file by default or from just the start with `--sampler head`. `sra_stats` takes the same
`--sample-size` flag for the same reason. The sample is cached in the store sidecar and reused by
`sra_stats`, `sra_validate` and `sra_profile_quality` alike until the underlying file's size or
modification time changes.

`sra_stats` and `sra_profile_quality` label every printed read total "(mates counted)": a paired-end
run's two mate files are counted separately, so the figure is twice the spot count NCBI reports for
that run. The per-accession quality profile JSON (`--detailed-reports`) writes the complexity score
under both `complexity_score` and the newer `sequence_complexity` key, so either name can be read back.

### Interactive SRA Dashboards

Generate interactive HTML dashboards for SRA analysis:

```bash
# Comprehensive dashboard
metaquest sra_dashboard \
    --accessions-file accessions.txt \
    --output-dir dashboards \
    --title "Project SRA Analysis" \
    --dashboard-type full

# Quality analysis dashboard only
metaquest sra_dashboard \
    --accessions-file accessions.txt \
    --dashboard-type quality
```

Pass `--quality-profiles DIR` to reuse quality profiles already saved by `sra_profile_quality`
instead of recomputing them. `--accessions-file` is required unless `--quality-profiles` names a
directory with saved profiles, in which case every accession found there is dashboarded and
`--accessions-file` can be left out entirely.

### Comparative SRA Analysis

Perform statistical comparisons between SRA dataset groups:

```bash
# Compare treatment vs control groups
metaquest sra_compare \
    --groups-file comparison_groups.json \
    --fastq-dir fastq \
    --statistical-tests \
    --generate-report
```

Example groups file format:
```json
{
  "Treatment_Group": ["SRR123456", "SRR123457"],
  "Control_Group": ["SRR789012", "SRR789013"]
}
```

The per-group summary prints "Mean reads in sample", the average of each dataset's `total_reads` value
(mates counted, see "SRA Quality Profiling" above), not a per-mate or per-sample-size figure.

## Visualizing Results

### Plotting Containment Data

Plot the distribution of containment scores. `plot_containment` always writes an image file next to the
input table (png by default, or the format `--save-format` names), rather than just displaying it, named
`<input-stem>_<plot-type>_<column>.<format>`:

```bash
metaquest plot_containment --file-path parsed_containment.txt --column max_containment --plot-type rank --save-format png --threshold 0.05
```

This writes `parsed_containment_rank_max_containment.png`; `sourmash scripts metaquest_plot` (registered
by the `sourmash` extra) calls the same `plot_containment` function and writes files with the same
naming.

Available plot types: rank, histogram, box, violin

### Plotting Metadata Counts

Visualize the distribution of metadata attributes. Unlike `plot_containment`, `plot_metadata_counts`
only writes a file when `--save-format` is given; without it, the plot is built but never saved anywhere:

```bash
metaquest plot_metadata_counts --file-path metadata_counts.txt --plot-type bar --save-format png
```

The bar chart is written in the current working directory as metadata_counts_bar.png.

Available plot types: bar, pie, radar

## Advanced Analysis Features

### Diversity Analysis
Calculate comprehensive diversity metrics for your metagenomic datasets:

```bash
# Calculate alpha and beta diversity with PERMANOVA
metaquest diversity_analysis \
    --abundance-file abundance_matrix.csv \
    --metadata-file sample_metadata.csv \
    --alpha-metrics shannon simpson chao1 \
    --beta-metric bray_curtis \
    --permanova-formula "treatment + site"
```

### Interactive Visualizations
Create dynamic, browser-based plots for data exploration:

```bash
# Interactive PCA plot
metaquest interactive_plot \
    --data-file abundance_matrix.csv \
    --metadata-file metadata.csv \
    --plot-type pca \
    --color-by treatment \
    --output-file pca_plot.html

# Interactive heatmap
metaquest interactive_plot \
    --data-file abundance_matrix.csv \
    --plot-type heatmap \
    --title "Species Abundance Heatmap"

# Diversity comparison plots  
metaquest interactive_plot \
    --data-file abundance_matrix.csv \
    --metadata-file metadata.csv \
    --plot-type diversity \
    --color-by treatment_group
```

### Taxonomic Validation
Validate species names against NCBI taxonomy database:

```bash
# Validate species from text file
metaquest validate_taxonomy \
    --species-file species_list.txt \
    --email your.email@domain.com \
    --output-file validation_results.csv

# Validate from CSV with specific column
metaquest validate_taxonomy \
    --species-file data.csv \
    --species-column organism_name \
    --email your.email@domain.com
```

### Taxonomic Summary Analysis
Summarise containment per taxonomic rank. The taxonomy table can be the map written by
`enrich_taxonomy` (genome ids and ranks, the usual route after `parse_containment`) or the validation
CSV written by `validate_taxonomy`. `enrich_taxonomy --cache` names the GTDB lookup cache file it reads
and appends to (default `taxonomy_cache.tsv`):

```bash
metaquest enrich_taxonomy --parsed-containment parsed_containment.txt --output taxonomy.tsv
metaquest taxonomic_summary --abundance-file parsed_containment.txt --taxonomy-file taxonomy.tsv \
    --levels phylum class order family genus --output-dir taxonomic_summaries

metaquest taxonomic_summary --abundance-file abundance_matrix.csv --taxonomy-file validation_results.csv
```

### Genome Search and Containment Exploration
A few smaller commands round out genome lookup and containment browsing:

- `genome_search --species "..."` or `--genus "..."` looks up GTDB accessions without downloading
  anything; `--all` includes every genome instead of just GTDB representatives, and `--format tsv`
  pairs each accession with the queried name.
- `explore_containment --parsed-containment parsed_containment.txt --min-containment 0.1` writes an
  interactive HTML browser of the containment table (default `containment_explorer.html`) and a GTDB
  taxonomy cache (default `taxonomy_cache.tsv`) alongside it, computing taxonomy on the fly unless
  `--taxonomy-map` is given.
- `find_by_taxonomy --parsed-containment parsed_containment.txt --taxonomy-map taxonomy.tsv --genus ...`
  (or `--family`/`--species`) filters the containment table to one taxonomic group; `--format summary`
  prints a per-rank count instead of the full table.

## Documentation

For comprehensive documentation including advanced features and technical details, see the [docs/](docs/) directory:

- **[Pipeline Overview](docs/pipeline_overview.md)** - The six stages (screen, select, download, analyse, extract, assemble), the commands of each, and what the project registry records
- **[SRA Information, Statistics and Validation](docs/SRA_ENHANCED_FEATURES.md)** - Dataset information, read statistics and validation commands supporting `download_sra`
- **[Branchwater Workflow](docs/branchwater_workflow.md)** - Detailed workflow guide for branchwater functionality
- **[Architecture](docs/ARCHITECTURE.md)** - Technical architecture and design decisions
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines, testing strategies, and architectural patterns for contributors

## Development & Testing

MetaQuest follows modern Python development practices with comprehensive testing and quality assurance.

### Current Status
- **Test Coverage**: Comprehensive coverage across core functionality
- **CLI Commands**: Fully covered, including the intelligent SRA commands
- **Data Layer**: Thoroughly covered across core modules (sra_metadata, taxonomy)
- **Core Processing**: Comprehensive coverage with edge case testing
- **SRA Advanced Features**: Well covered for reporting, quality profiling, and analytics
- **Visualization Plugins**: Bar chart plugin thoroughly covered
- **Integration Tests**: End-to-end workflow tests
- **Performance Benchmarks**: Benchmarked tests with pytest-benchmark for regression detection
- **Code Quality**: All linting checks passing

### Recent Enhancements
Significant improvements have been implemented across the codebase:

- **Intelligent SRA Package**: Complete implementation of next-generation SRA capabilities including intelligent downloads with resume functionality, comprehensive quality profiling, and interactive dashboard generation
- **Major Test Coverage Achievement**: Substantial coverage improvement with a large batch of comprehensive tests added across multiple files
  - Extended test suites for critical modules (sra_reporting, sra_intelligent, sra_metadata, bar visualizer, taxonomy)
  - Integration test suite with end-to-end workflow tests
  - Performance benchmarks using pytest-benchmark
  - All modules now thoroughly covered
- **Architecture Refinement**: Orphan code removal and clean separation of concerns between data layer and advanced SRA features
- **Quality Assurance**: All linting violations resolved and formatting standards enforced
- **Testing Best Practices**: Comprehensive mocking patterns, edge case coverage, and realistic test data established

### Development Workflow
```bash
# Set up development environment
make dev-install

# Run quality checks before committing
make check          # Format, lint, and type check
make test          # Run tests with coverage report
make pipeline      # Full integration test

# View available commands
make help
```

### Testing Structure
- **Comprehensive Test Suite**: covers CLI, data processing, visualization, and advanced SRA features
  - Unit tests: 170+ tests per critical module with extended test files
  - Integration tests: 12 end-to-end workflow tests (`tests/test_integration_simple.py`)
  - Performance tests: 25 benchmarked tests (`tests/test_performance_simple.py`)
- **Numerical Stability**: Edge cases in statistical computing properly handled (zero values, inf, nan, boundary conditions)
- **Integration Testing**: End-to-end workflow validation via `local_test.sh` and comprehensive integration suite
- **Mock Architecture**: NCBI APIs (Entrez, Taxonomy), Plotly/Jinja2 templates, file operations, and subprocess calls systematically mocked
- **Quality Assurance**: Automated formatting, linting, and type checking with zero warnings
- **Coverage**: Run `make test` for current coverage metrics

### Architecture Highlights
- **Layered Architecture**: CLI, Core, Data, Processing, Visualization layers
- **Advanced SRA Package**: Specialized module for intelligent SRA operations (`metaquest.sra`)
- **Plugin System**: Extensible format handlers and visualizers  
- **Command Registry**: Modular CLI command architecture
- **Modern Packaging**: Uses `pyproject.toml` with backward compatibility

## Contributing

We welcome contributions to MetaQuest. Whether you want to report a bug, suggest a feature, or contribute code, your input is valuable.

### Quick Start for Contributors
```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/MetaQuest.git
cd MetaQuest

# 2. Set up development environment  
make dev-install

# 3. Create a feature branch
git checkout -b feature/my-new-feature

# 4. Make your changes and test
make check          # Ensure code quality
make test          # Run test suite
make pipeline      # Integration tests

# 5. Commit and push
git commit -m "Add new feature"
git push origin feature/my-new-feature

# 6. Create a Pull Request on GitHub
```

### Contribution Guidelines
- **Code Quality**: All contributions must pass `make check` (formatting, linting, type checking)
- **Testing**: New features should include tests with proper edge case handling
- **Documentation**: Update relevant documentation for new features
- **Architecture**: Follow existing patterns (see `CLAUDE.md` for detailed guidance)

### Current Priority Areas
Help us improve MetaQuest by contributing to these areas:
- **Remaining Visualization Modules**: Add tests for `interactive.py`, `reporting.py`, and `plots.py` (currently 0% coverage)
- **Processing Layer Enhancement**: Optimize containment analysis algorithms and extend diversity analysis features
- **Plugin Development**: Add new format handlers or visualization plugins beyond bar charts
- **Performance Optimization**: Large-scale dataset handling and memory efficiency improvements
- **Documentation**: Expand workflow examples, API documentation, and testing guides

**Note**: Critical SRA modules, data layer, and CLI commands now have excellent coverage (86-99%). The focus has shifted to remaining visualization and processing modules.

For detailed development guidelines, architectural patterns, and comprehensive testing strategies, see [CLAUDE.md](CLAUDE.md).