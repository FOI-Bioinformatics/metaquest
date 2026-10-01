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
make dev-install  # Installs every extra plus the development tools
```

### Core install and optional extras

A plain install brings only the core packages (pandas, numpy, matplotlib, biopython, lxml,
requests). That is enough to load the CLI and run the download, containment, metadata and
matplotlib plotting steps. Other features need an extra:

| Extra | Packages | Needed for |
|---|---|---|
| `analysis` | scikit-learn, scipy | `diversity_analysis`, PCA, t-SNE and clustered heatmaps in `interactive_plot`, group statistics in `sra_report --groups-file` |
| `interactive` | plotly, jinja2 | `interactive_plot`, `explore_containment`, the `sra_report` HTML report (not needed with `--no-report`) |
| `maps` | cartopy | geographic sample maps |
| `sourmash` | sourmash | sketching a genome in `branchwater_search`, and the `sourmash scripts metaquest_*` plugin |
| `all` | all of the above | everything |

```bash
pip install .                      # core only
pip install '.[analysis,interactive]'
pip install '.[all]'               # every extra; environment.yml does this
```

A command that needs a missing extra stops with an error naming the extra and the interpreter to
install it into, for example:

```
Diversity analysis needs the 'scikit-learn' package. Install it into this interpreter with:
/path/to/python -m pip install 'metaquest[analysis]'
```

`requirements.txt` lists the core packages and `requirements-all.txt` the core plus every extra.
Since 0.5.0 the extras are no longer installed by default, and seaborn, statsmodels, umap-learn,
networkx and upsetplot are no longer dependencies.

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

| Tool (conda package) | Oldest version | Used by |
|---|---|---|
| `fasterq-dump`, `prefetch` (sra-tools) | 3.0 | `download_sra` (prefetch, then fasterq-dump; or `--no-prefetch`) |
| `pigz` (optional) | 2.4 | `download_sra`, `store_adopt` (parallel gzip; else Python's gzip module) |
| `datasets` (ncbi-datasets-cli) | 16 | `genome_download`, `genome_prepare` |
| `minimap2` | 2.17 (first with `--sam-hit-only`) | `extract_target_reads` |
| `samtools` | 1.10 (first with `samtools coverage`) | `extract_target_reads` |
| `megahit` | 1.2.9 | `extract_target_reads --assemble` |
| `seqkit` (optional) | any | `sra_profile`, `sra_report` (faster read statistics; else a Python reader) |

`download_test_genome` fetches its genome over HTTPS and needs none of these tools. A command checks the
tools it needs before it starts any work (`download_sra` checks `fasterq-dump`; `extract_target_reads`
checks minimap2 and samtools, and megahit with `--assemble`; `genome_download` and `genome_prepare`
check `datasets`). A missing tool, one older than the version above, or one that cannot run (it exits
with an error when asked for its version) stops the command with exit code 3 and a message that lists
every such tool with the `conda install` command that provides it. The optional tools (`prefetch`,
`pigz`, `seqkit`) are not checked by the commands, which work without them; `metaquest doctor` (below)
reports every tool in the table, and a tool older than its oldest supported version is a failed check
there. The table lives in `metaquest/utils/tools.py`, and `environment.yml` pins the same versions.

`environment.yml` installs all of them together with MetaQuest:

```bash
conda env create -f environment.yml
conda activate metaquest
```

Map plots need the `maps` extra (see the extras table above); `environment.yml` installs every extra.

### Checking the environment

`doctor` (listed under Environment in `metaquest --help`) reports whether this environment can run
MetaQuest, without changing anything:

```bash
metaquest doctor                              # every check, one line each
metaquest doctor --for extract_target_reads   # a tool this command needs is a failure, not a warning
metaquest doctor --json                       # the same report as one JSON document
metaquest doctor --network                    # also check that NCBI and Branchwater answer (10 s each)
```

It checks the Python and MetaQuest versions; every external tool, with its path, version and the
oldest supported version; that the config file parses, with each runtime setting and where its value
came from; the shared data store, if one is configured (`--data-root`, `METAQUEST_DATA`, the registry
or the config file), for its marker, write access and free space; free space at the project folder
(`--project`, default the working directory), the temporary folder and the `.sra` cache against
`min_free_gb`; the CPUs available against the node's total, the memory limit and any SLURM job
variables; and that the nearest project registry loads. A missing tool is a warning unless `--for`
names a command that needs it; a tool below its oldest supported version, a config file or variable
that does not parse, a store without its marker, an unreadable registry and, with `--network`, an
unreachable service are failures. `doctor` exits with 0 when nothing failed (warnings allowed) and
with 3 otherwise. It still runs when the config file is malformed, which stops every other command,
and reports the parse error as a failed check. `make doctor` runs it from a checkout.

### Configuration

Run-time settings (tool timeout, free-space floor, lock limits, NCBI email and API key, logging,
megahit memory, the download worker cap) are taken from a flag, then a `METAQUEST_<NAME>` environment
variable, then the `[runtime]` table of `~/.config/metaquest/config.toml` (or
`$XDG_CONFIG_HOME/metaquest/config.toml`), then a built-in default. The shared data store root has its
own order: `--data-root`, `METAQUEST_DATA`, the project registry, then `[store] data_root` in the same
file. [docs/configuration.md](docs/configuration.md) lists every setting with its type, default,
variable and flag. With an email address set there (`METAQUEST_NCBI_EMAIL` or `ncbi_email`), `--email`
can be left out of `download_metadata`, `sra_info` and `validate_taxonomy`.

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
exclusion, download, run size, the dataset profile of `sra_profile` (total reads, GC in percent,
quality grade), mapped reads, reference coverage and assembly statistics. A registry written before
0.5.0 still fills the profile columns from its `sra_stats` and `quality` analyses. It records the
export in the project registry when one exists, and does not create a registry when there is none.
The last three columns, `download_seconds`, `extraction_seconds` and `assembly_seconds`, are the
recorded run times, empty where a step was not timed (data recorded before 0.7.0, or a dataset linked
from the store). The extraction time of a sample is measured from the end of the previous sample, so
for the first sample it includes reading the containment table and building the minimap2 index; the
assembly time is the megahit run alone, without the contig summary and the coverage mapping.
`status --json` summarises the same times under `timing` (counts, totals and medians), the text report
adds one timing line when anything was timed, and `status --export-tsv` adds the same columns.
The last column, `download_verdict`, is the accession's recorded download completeness verdict
(`complete`, `truncated` or `unverified`), empty where none was recorded.

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

`--data-root` is accepted by `download_sra`, `download_metadata`, `status`, `sra_profile`, `sra_report`,
`sra_validate` and `extract_target_reads`; it never replaces `--fastq-folder`, which still names
where the project expects its reads (as a folder or as the store's symlink). `sra_report` can also
reuse the quality profiles an earlier `sra_profile` run saved, via `--quality-profiles`. With a store configured, `status`
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
the assembly thread count explicitly with `--assembly-threads` if your megahit build handles more.
`--assembly-memory` sets megahit's `--memory`: `auto` (the default; also `METAQUEST_ASSEMBLY_MEMORY` or
`assembly_memory` in `[runtime]`) gives 90% of the memory limit detected for the job (a cgroup limit, or
`SLURM_MEM_PER_NODE` or `SLURM_MEM_PER_CPU`) and leaves megahit's own default when no limit is found, as
on macOS; a size such as `32G` or `32000M` is passed in bytes; a fraction such as `0.5` is passed as it
is, and megahit applies it to the whole node's memory, so under a scheduler give a size instead. A
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

Mapped reads always drop unmapped, secondary and supplementary alignments; `--min-mapq` additionally discards records
below a mapping-quality threshold (default 0, keep every mapped record). A value of 20 is reasonable for a close
relative of the target genome, but a divergent strain can genuinely map with a low MAPQ, so raising the threshold can
discard real matches. For each sample with kept reads, `samtools coverage` on the kept alignments (it also skips
duplicate and QC-fail reads by default) writes `targeted/<ACC>/<genome>_coverage.tsv`, and the registry records the
breadth of the reference covered at 1x or more and the length-weighted mean depth (a failure of this step is logged as
a warning and leaves the extracted reads in place). `--assembly-preset` selects megahit's `--presets` value:
`meta-sensitive` (the default, suited to these small targeted read sets), `meta-large`, or `default` (no `--presets`
flag). Unless `--no-coverage`, the extracted reads are mapped back onto the assembled contigs to report a mapping rate
and estimated mean depth alongside the other assembly statistics.

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
A requested filter whose column is missing from the table is an error: `select_datasets` exits with
status 1 without writing its output or recording a selection. Branchwater-derived tables carry none of
these columns, so run `download_metadata` and `parse_metadata` for the candidate list first. When the column
exists but a filter drops every remaining candidate because none has a value, a warning gives the same
hint. The log also reports the summed size of the selected runs.

`accessions.txt` is the input for `download_sra`, which writes
`fastq/<accession>/<accession>_1.fastq.gz` (and `_2` for paired runs; gzip-compressed by default, see
below), the layout `status`, `sra_profile`, `sra_report`, `sra_validate` and
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
`blacklisted`, or `skipped` (accessions skipped by `--max-downloads`). The last column, `seconds`, is how
long that accession's download took in this run; it is empty for a dataset linked from the store and
for accessions that were not downloaded. To see sizes and sequencing
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
A project without a shared store also takes a per-accession lock (`<fastq-folder>/.locks/<ACCESSION>.lock`,
same heartbeat and 10-minute takeover as the store's, bounded by `--lock-wait`), so two runs on one
project download each accession once. Each download is built, verified and compressed under
`<fastq-folder>/.metaquest-tmp/` and then moved into `<fastq-folder>/<ACCESSION>` with one rename, so
another run never sees a partly written folder. Both hidden folders are ignored by `status` and the
download inventory. A `--sra-cache` folder shared by two projects without a store is not locked.
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

`--timeout SECONDS` bounds how long `prefetch` or `fasterq-dump` (and, for `extract_target_reads`,
minimap2, samtools or megahit) is allowed to run before it is stopped; the default, 0, means no limit
(also settable with the `METAQUEST_TIMEOUT` environment variable or `timeout` in `config.toml`'s
`[runtime]` table). A stopped tool is reported the same way as a network failure, so `download_sra`
retries it and, if every accession that failed did so for that reason, exits with status 4 (see "Exit
codes" below). A `--version` probe used to record a tool's version always uses a fixed 30-second
timeout, regardless of this setting.

`--max-workers N` sets how many accessions download at once. Without it the number is the CPUs
available to this job (the CPU affinity mask, which reflects a SLURM allocation, else
`SLURM_CPUS_PER_TASK`, else the machine's CPU count) divided by `--num-threads`, at least 1 and at most
4, since downloads are limited by the network more than by CPUs; `METAQUEST_MAX_WORKERS_CAP` (or
`max_workers_cap` in `[runtime]`) changes that cap.

Before each accession starts, `download_sra` checks that the filesystems it writes to have room for it:
the FASTQ folder (or the store's `tmp` folder), the `fasterq-dump` temporary folder and, with prefetch,
the `.sra` cache. An accession whose run size is in the registry (from `download_metadata`) needs about
8 times that size for the temporary files, 10 times for the FASTQ folder (the uncompressed files and
the gzip files written from them), plus the size itself for the cache; locations on one filesystem add
up, and downloads already running are counted.
An accession of unknown size needs `--min-free-gb` (default 10; also `METAQUEST_MIN_FREE_GB` or
`min_free_gb` in `[runtime]`) on each filesystem. A download that does not fit while others are running
waits until they finish and release their space. One that would not fit even with nothing else running
is not started and fails alone, with an `insufficient-space: not enough free space on ...` message;
the other accessions continue. `--min-free-gb 0` turns the check off. A filesystem whose free space
cannot be read is assumed to have room.

A disk that fills up during a download (the tool itself reports that it is out of space) stops the
pass: the downloads not yet started are recorded as `disk-full: not attempted`, the running ones
finish, no retry pass runs, and the command exits with status 1.

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

### Running several instances at once

Two or more `metaquest` processes can work on the same project, or the same shared store, at once, on
one machine or on several machines sharing a network filesystem. Every long-running command installs a
graceful signal handler: the first `SIGINT`, `SIGTERM`, or `SIGHUP` it receives stops cleanly, letting
whatever unit of work was already in progress (an accession, a sample, a registry write) finish and be
recorded rather than being cut off mid-write; a second signal is logged rather than acted on, so it
cannot interrupt that final write; a third abandons the run at once, stops any running external tool,
and exits with status 130.

Downloads are locked per accession, both into a shared store (`<data-root>/locks/<ACCESSION>.lock`) and
into a plain project with no store (`<fastq-folder>/.locks/<ACCESSION>.lock`), so two runs racing to
fetch the same accession cooperate instead of corrupting each other's output: the second run waits for
the first, then finds the finished files and reports them already present. `--lock-wait SECONDS` bounds
how long a run waits for another run's lock on the same accession before giving up, for both a store and
a plain project; the default, 0, waits without a time limit for as long as the lock's heartbeat keeps
showing its holder is still working. A lock whose heartbeat stops for 10 minutes (a crashed or killed
process) is taken over by the next waiter; a holder that has already died on the same machine is taken
over at once, without waiting out that window. The project registry is locked the same way, with a
120-second stale window, so a command that is slow to write (a large batch, a contended filesystem) is
not mistaken for dead by another command waiting to record its own results.

`store_gc` keeps a dataset for one day after it was last linked or downloaded even when nothing
currently holds its lock, on top of never removing a dataset a live project still uses or links.

See "Shared data store" above for the store's own locking and `store_gc`/`store_verify` behaviour, and
"Downloading reads" above for `--lock-wait` and `--sra-cache`.

### Exit codes

Every command exits with one of these codes, so a batch script (a SLURM job, for example) can decide
whether to resubmit:

| Code | Meaning | What to do |
|---|---|---|
| 0 | Success | |
| 1 | Failure: bad input, missing file, a dataset not found, a tool error | Fix the cause; a rerun alone will not help |
| 2 | Usage error: an unknown flag or command, or a renamed command | Correct the command line |
| 3 | Configuration: the environment lacks something the command needs | Fix the environment or configuration |
| 4 | Retryable: a network failure, or a wait for a lock that reached its limit (see below) | Rerun later |
| 130 | Interrupted by `SIGINT`, `SIGTERM` or `SIGHUP` | Rerun; finished work is kept |

Code 3 covers a missing optional package; an external tool that is missing, cannot run, or is older than
the version in "External tools" above; a malformed `config.toml` or a setting that does not parse; a missing NCBI
email address; and a `--log-file` that cannot be opened.
Code 4 covers an NCBI request of `sra_info` that could not connect, timed out or got HTTP 429 or 5xx
after its retries, a `download_metadata` run that could not reach NCBI for any accession, and a wait
for another run's lock (the project registry, the store catalogue, a store dataset) that gave up. The
per-accession `--lock-wait` of `download_sra` is the exception: an accession given up there counts as
locked, which gives 1 (see below).

Two commands decide from the outcome of each accession:

- `download_sra` exits with 4 only when every accession that failed did so for a network reason
  (including a tool stopped by `--timeout`). If any failed for another reason (not found, disk full,
  not enough free space, locked by another run, interrupted) it exits with 1. An accession given up
  after `--lock-wait` counts as locked, so it gives 1, not 4.
- `download_metadata` logs each accession NCBI did not return and exits with 0. When NCBI could not be
  reached for any accession (every request failed with a connection error, a timeout or HTTP 429 or
  5xx after its retries) it exits with 4. Run it again to fetch only the missing ones.

See [docs/hpc.md](docs/hpc.md) for a resubmission loop on code 4 under SLURM.

### Logging

Log lines go to stderr; a command's result (tables, JSON) goes to stdout. These options work on every
command, before or after the command name (`metaquest --quiet download_sra ...` and
`metaquest download_sra ... --quiet` are the same):

| Option | Effect |
|---|---|
| `--log-level LEVEL` | Console level: DEBUG, INFO (the default), WARNING, ERROR or CRITICAL |
| `-q`, `--quiet` | Only warnings and errors on the console (same as `--log-level WARNING`) |
| `-v`, `--verbose` | Debug lines and full tracebacks on the console (same as `--log-level DEBUG`) |
| `--log-file PATH` | Also append every line to `PATH`; its folder is created |
| `--progress-every N` | Log a progress summary every N items of a long run (default 50) |

`-q` and `-v` cannot be given together, and either one overrides `--log-level`. `store_status` keeps its
own `--verbose` (list every dataset), so its log level is raised with `metaquest -v store_status`.
Each option can also be set with an environment variable (`METAQUEST_LOG_LEVEL`, `METAQUEST_LOG_FILE`,
`METAQUEST_PROGRESS_EVERY`) or in the `[runtime]` table of `~/.config/metaquest/config.toml`
(`log_level`, `log_file`, `progress_every`); a flag takes precedence over both.

The log file is opened for appending, so several runs on one machine can share one file on a local
disk. Give each process its own file on a network filesystem (for example one per SLURM array task, as
in [docs/hpc.md](docs/hpc.md)): appends from several hosts to one NFS file can interleave or be lost.
It receives every line at INFO or above, also when the console is quiet, and DEBUG lines when the
console is at DEBUG. Each line names the host and process ID that wrote it:

```
2026-09-30 14:02:11 node17[48213] WARNING metaquest.data.sra.retry: Failed to download SRR1234567: ...
```

When a command fails, the console shows the error on one line and the log file keeps the full
traceback; without a log file, rerun with `--log-level DEBUG` (or `-v`) to see it. At DEBUG a run also
logs the MetaQuest version, its command line (with the `--api-key` value hidden), the host, the process
ID and, under SLURM, `SLURM_JOB_ID` and `SLURM_ARRAY_TASK_ID`. `METAQUEST_LOG_HOST=true` (or
`log_host = true` in `[runtime]`) puts the host and process ID on console lines too.

`download_sra`, `download_metadata` and `extract_target_reads` log a progress summary every
`--progress-every` items (50 by default) and at least every 5 minutes while items are finishing, then
one closing line with the totals and the time taken (for `download_sra`, after the retry pass):

```
download_sra: 150/2000 done (148 ok, 2 failed), 3.1/min, about 9 h 57 min left
download_sra: finished 2000/2000 (1990 ok, 10 failed) in 10 h 45 min
```

The lines about one accession or sample (a download starting, skipped or finished, its temporary
folder, a store link, a retry, an NCBI metadata request, an extracted sample) are logged at DEBUG;
`--progress-every 0` turns the summaries off and logs them at INFO instead. Warnings and errors about an
accession are logged at their own level either way. Some per-accession lines stay at INFO: the first
"prefetch not found on PATH" of a run, and the lines about an unusual event (a download interrupted or
redone with `--force`, a truncated archive removed for the next attempt, a wait for running downloads to
free space, a wait for a lock another run holds).

### Running on a cluster

[docs/hpc.md](docs/hpc.md) describes running `download_sra` and `extract_target_reads` as SLURM jobs:
checking a node with `metaquest doctor --for download_sra`, an array job template that splits the
accession list, what is written when the walltime signal arrives and how a rerun resumes, chaining jobs
with `--dependency=afterok` and resubmitting on exit code 4, a shared data store on a group
filesystem, the caveats of NFS (locks, clock skew, temporary files, log files), and megahit's memory
under a job limit.

### SRA Quality Profiling

`sra_profile` computes statistics and a quality profile for each downloaded dataset. It reads the
same `--fastq-folder` as the rest of the pipeline and, by default, profiles every accession folder
in it; `--accessions-file` and repeated `--accession` flags restrict the run to the accessions named:

```bash
# Every accession folder under fastq/
metaquest sra_profile --fastq-folder fastq

# Selected accessions, printing the path of each profile JSON as it is written
metaquest sra_profile \
    --accessions-file accessions.txt \
    --accession SRR123456 \
    --output-dir quality_profiles \
    --detailed-reports
```

Each dataset is profiled once, by one path. Read and base totals, mean read length and GC content
come from the dataset's statistics record: exact read counts (from `seqkit stats` when it is
installed, otherwise a streaming count) and GC from a uniform sample of the first mate file's reads.
The record is cached in the store sidecar and reused by `sra_profile`, `sra_report` and
`sra_validate --check-pairs` until a file's size or modification time changes. The per-read quality,
complexity, duplication and adapter figures come from a sample of `--sample-size` reads per dataset
(default 10000) drawn from every mate file, uniformly across each file by default or from the start
of each file with `--sampler head`.

The command writes three things: the table `--output-report` (default `sra_statistics.csv`, one
row per accession), one `<accession>_quality_profile.json` per accession in `--output-dir` (default
`sra_quality_profiles`) together with `quality_summary.json`, and a `profile` analysis per accession in
the project registry. GC is given in percent (0 to 100) in all three, under `gc_percent`; the profile
JSON carries the complexity score under `complexity_score`. Printed read totals are labelled
"(mates counted)": a paired-end run's two mate files are counted separately, so the figure is twice
the spot count NCBI reports for that run. The command exits with status 1 when any accession has no
readable FASTQ files; the others are still profiled and written.

### SRA Reports

`sra_report` writes one HTML report, `sra_report.html` in `--output-dir` (default `sra_reports`),
with the figures behind it in `sra_report.json`. Without groups the report covers the quality of
the datasets named by `--accessions-file`; with `--groups-file` it adds a comparison of the groups
and their statistical tests (a t-test for two groups, one-way ANOVA for more):

```bash
# Quality of a set of datasets
metaquest sra_report --accessions-file accessions.txt --title "Project SRA Analysis"

# Treatment against control, reusing the profiles an earlier sra_profile run saved
metaquest sra_report \
    --groups-file comparison_groups.json \
    --quality-profiles sra_quality_profiles
```

Example groups file format:
```json
{
  "Treatment_Group": ["SRR123456", "SRR123457"],
  "Control_Group": ["SRR789012", "SRR789013"]
}
```

Each accession is profiled once, by the same path as `sra_profile`, and that one profile is used
for both the quality section and the comparison. An accession found in `--quality-profiles` is not
profiled again; with no `--accessions-file` or `--groups-file`, every profile in that folder is
reported on. An accession without readable FASTQ files is left out with a warning. The report opens
in a browser unless `--no-open` is given; `--no-report` writes only `sra_report.json` and prints the
results, and needs neither plotly nor jinja2. A `report` analysis is recorded per accession.

### Renamed SRA commands (0.5.0)

Four commands were merged into two in 0.5.0. The old names still parse, print where the command
went and exit with status 2:

| Before 0.5.0 | Since 0.5.0 |
|---|---|
| `sra_stats` | `sra_profile` |
| `sra_profile_quality` (`sra-profile-quality`) | `sra_profile` |
| `sra_dashboard` (`sra-dashboard`) | `sra_report` |
| `sra_compare` (`sra-compare`) | `sra_report --groups-file` |
| `--fastq-dir` | `--fastq-folder` |
| `sra_stats --accessions A B`, `sra_validate --accessions A B` | `--accession A --accession B` or `--accessions-file` |
| `sra_dashboard --dashboard-type` | removed: one report, with the comparison when `--groups-file` is given |
| `sra_compare --statistical-tests`, `--generate-report` | removed: tests come with `--groups-file`; `--no-report` skips the HTML |
| `--include-contamination` | removed: the adapter figures are always computed |
| `gc_content` (0-1 fraction in the profile JSON, percent in the CSV) | `gc_percent` (percent everywhere) |
| registry analyses `sra_stats`, `quality`, and none for the dashboard | `profile`, `report` |
| `avg_quality` in `sra_statistics.csv`: mean of per-read mean quality over mate-1 records | same column name, now the mean base quality over a sample of reads from all mates |

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
- **[Running on a cluster](docs/hpc.md)** - SLURM array jobs, walltime signals, exit codes and
  resubmission, a shared store and NFS
- **[Configuration](docs/configuration.md)** - Every runtime setting with its type, default,
  environment variable and flag
- **[Architecture](docs/ARCHITECTURE.md)** - Technical architecture and design decisions
- **[Packaging](docs/packaging.md)** - PyPI and bioconda distribution, and the checklist before turning on
  PyPI publishing
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
- **Major Test Coverage Achievement**: Substantial coverage improvement with a large batch of comprehensive
  tests added across multiple files
  - Extended test suites for critical modules (sra_reporting, the SRA analysis commands, sra_metadata, bar visualizer, taxonomy)
  - Integration test suite with end-to-end workflow tests
  - Performance benchmarks using pytest-benchmark
  - Critical modules now thoroughly covered
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

### Nightly Smoke Test

A scheduled GitHub Actions workflow (`.github/workflows/smoke.yml`) runs `scripts/smoke_chain.sh`
against real NCBI/SRA services every night: it downloads the tiny SRR2517620 run, validates and
profiles it, extracts reads against the bundled test genome, and writes the results table. It is
the only CI job that touches the network, using the tools the `metaquest` conda environment
installs (fasterq-dump, prefetch, minimap2, samtools). Run the same chain locally with
`make test-network` (needs the `metaquest` conda environment; see `make env`).

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

## Releases

Pushing a tag of the form `vX.Y.Z` (for example `v0.5.0`) triggers the release workflow
(`.github/workflows/release.yml`), which builds the sdist and wheel, checks them with `twine` and
`check-wheel-contents`, and publishes a GitHub release with the built packages attached. See
[CHANGELOG.md](CHANGELOG.md) for the changes in each release.

A second job in that workflow, `pypi`, publishes the built distribution to PyPI; it is skipped
until the repository variable `PYPI_PUBLISH` is set to `true`. See
[docs/packaging.md](docs/packaging.md) for the checklist before turning that on, and for a
bioconda recipe template (submitted, once PyPI publishing works, to the separate
`bioconda-recipes` repository; this repository does not carry a recipe folder itself).

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