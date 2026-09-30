# Changelog

All notable changes to MetaQuest are documented in this file. Dates are in YYYY-MM-DD format.

## [0.7.0] - unreleased

### Added

- Logging options on every command, before or after the command name: `--log-file PATH` appends every
  line at INFO or above to a file, each stamped with host name and process ID and with the full traceback
  of a failure; `-q/--quiet` (warnings and errors only on the console) and `-v/--verbose` (DEBUG), which
  cannot be combined and override `--log-level`. At DEBUG a run logs the version, its command line (the
  `--api-key` value hidden), host, process ID and SLURM job and array task IDs. See "Logging" in the
  README.
- Progress summaries for `download_sra` and `download_metadata`: one line every `--progress-every N`
  items (default 50; also `METAQUEST_PROGRESS_EVERY` or `progress_every` in `[runtime]`) and at least every
  5 minutes, such as `download_sra: 150/2000 done (148 ok, 2 failed), 3.1/min, about 9 h 57 min left`,
  and a closing line with the totals and the time taken.
- Free-space check for `download_sra`: before each accession starts, the filesystems it writes to (FASTQ
  or store `tmp` folder, `fasterq-dump` temporary folder, `.sra` cache) must have room for it, counting
  the downloads already running. An accession with a registry run size needs 8 times that size for its
  FASTQ files and again for the temporary files, plus the size for the cache; the factor comes from
  seven runs of the crispatus test store, whose uncompressed FASTQ was 6.98 to 7.73 times the `.sra`
  size (gzip-compressed: 1.49 to 1.66 times). An accession of unknown size needs `--min-free-gb` (default
  10; `METAQUEST_MIN_FREE_GB`, `[runtime] min_free_gb`); 0 turns the check off. A download that does not
  fit is not started and fails with `disk-full: insufficient free space on <mount>: ...`.
- `--assembly-memory` on `extract_target_reads` (also `METAQUEST_ASSEMBLY_MEMORY`, `[runtime]
  assembly_memory`): `auto`, the default, passes megahit `--memory` as 90% of the memory limit detected
  for the job (cgroup v2 or v1, else `SLURM_MEM_PER_NODE`) and omits it when none is found; a size such as
  `32G` is passed in bytes; a fraction is passed as it is and applies to the node's whole memory.
- The default number of parallel downloads is the CPUs available to the job (the affinity mask, which a
  SLURM cpuset limits, else `SLURM_CPUS_PER_TASK`, else the CPU count) divided by `--num-threads`, at
  most 4; `METAQUEST_MAX_WORKERS_CAP` (or `[runtime] max_workers_cap`) changes the cap, which the
  `--max-workers` help now names. `seqkit stats` in the store statistics uses at most as many threads.
### Changed

- A command failure is one line on the console unless it is at DEBUG; the traceback now always goes to
  the log file when there is one. The hint "Use --log-level DEBUG for full traceback" is shown only when
  there is no log file.
- The per-accession INFO lines of `download_sra` (a download that succeeded, a retry that succeeded) and
  of `download_metadata` (each NCBI request) are now logged at DEBUG, replaced at INFO by the progress
  summaries. `--progress-every 0` logs them at INFO again.
- Exit codes: 3 for a configuration problem (`ConfigurationError`: a missing optional package or NCBI
  email address, a malformed `config.toml`), 4 for a retryable failure (an NCBI request that could not
  connect, timed out or got HTTP 429 or 5xx; a wait for the registry or catalogue lock that reached its
  limit; a `download_sra` run whose every failure was a network one), 130 for an interrupt, 1 for any
  other failure. See "Exit codes" in the README.
- External tool timeout is now configurable and, by default, unlimited: `run_secure` used to give every
  external tool (`fasterq-dump`, `prefetch`, `minimap2`, `samtools`, `megahit`) a fixed one-hour limit
  (`MAX_SUBPROCESS_TIMEOUT`) even when a caller passed no timeout at all, and `timeout=0` was silently
  treated as "use the one-hour limit" instead of "no limit". `--timeout SECONDS` on `download_sra` and
  `extract_target_reads` (also `METAQUEST_TIMEOUT` or `[runtime] timeout` in `config.toml`) now sets it;
  0, the default, means no limit. A `--version` probe always uses a fixed 30-second timeout regardless of
  this setting, so a hung tool cannot stall version detection.
- One table of external tools (`metaquest/utils/tools.py`) with the oldest supported versions: sra-tools
  3.0, minimap2 2.17, samtools 1.10 (for `samtools coverage`), megahit 1.2.9. `download_sra`,
  `extract_target_reads`, `genome_download` and `genome_prepare` check the tools they need before any
  work starts and exit with 3 (was 1 for `download_sra` and `extract_target_reads`; `genome_download`
  and `genome_prepare` had no check and failed inside the first `datasets` call) when one is missing or
  older than its floor, listing every problem with its `conda install` command.

### Fixed

- megahit under cgroups (a SLURM job, a container) sized its memory from the node's total, not from the
  job's limit, and could be killed for exceeding it; `--assembly-memory auto` now passes the job's limit.
- A disk that filled up during the first download pass did not stop it: every remaining accession was
  still started and failed in turn, and the retry pass tried again. The first disk-full result now
  cancels the downloads not yet started (recorded as `disk-full: not attempted`, as the retry pass
  already did), lets the running ones finish, and skips the retry pass.

### Upgrade notes

- A script that searched the INFO log of `download_sra` or `download_metadata` for one line per
  accession no longer finds those lines: run with `--progress-every 0` to keep them at INFO, or with
  `--log-level DEBUG`. Warnings and errors naming an accession are unchanged.
- A script that treated every non-zero exit as the same failure keeps working. A script that tested for
  exactly 1 should also accept 3 and 4; 4 means the same command may succeed if run again later.
- `fasterq-dump`/`prefetch` older than 3.0 (sra-tools 2.x) are now refused by `download_sra` with exit
  code 3; install sra-tools 3.0 or later (`environment.yml` already does).

## [0.6.0] - 2026-09-30

### Added

- `registry_update(path, mutation)`: loads the registry, applies a mutation to it and writes it back, all
  under one lock, returning the mutation's result. Several commands were migrated onto it; see Changed.
- Process-level concurrency tests (`tests/test_concurrency_processes.py`, `tests/helpers_processes.py`):
  spawn the real CLI as a subprocess with fake tools on `PATH` and exercise same-accession contention for
  a plain project and for a store, registry contention during a download, two `SIGTERM`s during a
  download, a tool that ignores `SIGTERM`, a `SIGKILL`ed lock holder, and `store_gc` against a dataset
  another run is downloading.
- An atomic-writes gate in `make check` (`scripts/check_atomic_writes.sh`): fails on a direct
  `.write_text`, `.write_bytes`, `.to_csv`, `copy2`, `copyfile`, `write_html` or `savefig` call, or an
  `open(path, "w...")`/`"x..."`, `path.open("w...")` or `open(path, mode)` call, outside the small allowlist
  in `scripts/atomic_writes_allowlist.txt`. Figure writers (matplotlib `savefig`, plotly `write_html`) are
  on the allowlist: a figure is regenerable and overwritten whole, and no later step reads it.
- A `multiprocess` pytest marker for tests that start real subprocesses; it runs as part of the default
  suite.
- Locks for a shared minimap2 index build and for per-sample read extraction
  (`<index>.lock`; `<output-folder>/.locks/<ACCESSION>.<GENOME_ID>.lock`), so two overlapping processes
  building the same index or extracting the same sample no longer duplicate the work; a sample already
  being extracted elsewhere is reported skipped rather than mapped twice.

### Changed

- One lock mechanism (`metaquest/utils/lockfile.py`) now backs the registry lock, the store catalogue
  lock, every per-accession dataset lock, the new plain-project download lock, and the new index-build
  and per-sample-extraction locks. Every lock file carries a holder record (pid, host, start time, a
  random token) refreshed by a heartbeat thread. A lock judged stale is reclaimed only under a second,
  short-lived guard file, so two waiters judging one lock stale at the same instant still produce exactly
  one winner, and a holder that stalls past its stale window (a slow write, a sleeping laptop) is not
  destroyed as long as its heartbeat keeps running. A holder whose process has since died on the same
  host is taken over at once instead of waiting out the stale window.
- The registry lock's stale window is now 120 seconds (was 30), refreshed by the heartbeat every 5
  seconds; the wait before a writer gives up is unchanged at 30 seconds. The store catalogue lock waits
  up to 60 seconds and is itself judged stale after 120 seconds. Every lock's "gave up waiting" message is
  now one format across every lock kind, naming what is locked, the holder's pid and host, and since when.
- A plain project's downloads (no shared store) now take a per-accession lock the same way a store does,
  so two runs on one project no longer both write into `fastq/<ACCESSION>_temp` and delete each other's
  in-flight output. Each download is built, verified and compressed under
  `fastq/.metaquest-tmp/<ACCESSION>` and published into `fastq/<ACCESSION>` with one rename.
  `--lock-wait` now bounds this wait too, the same as it already bounded the store's lock. The new
  in-lock "already exists" check this adds (so a second process finding a finished download does not
  re-fetch it) honours `--redownload-truncated` the same way the existing pre-lock check already did: an
  accession the registry's last verification marked `truncated` is still re-downloaded once the lock is
  held when that flag was given, rather than being reported already present.
- `store_gc` treats a dataset used (linked or downloaded) within the last day as in use even when nothing
  currently holds its lock, and re-checks each candidate's lock, usage, and link state again immediately
  before removing it, not only at classification time; a dataset kept for either reason is listed under
  `in_use` with the specific reason. `store_verify --fix-state` re-reads the sidecar under the dataset
  lock immediately before writing it back, and skips the write if another process changed it meanwhile or
  if the dataset is locked.
- A store catalogue write that fails after a dataset has already been published, but before it is
  linked into the project, no longer fails the download: the result is reported as a success with
  "; catalogue pending; stored" in its message, the dataset is still linked, and `store_reindex` repairs
  the missing catalogue row from the sidecar already on disk.
- Every command now handles `SIGINT`, `SIGTERM`, and `SIGHUP`. The first turns into a `KeyboardInterrupt`
  so a `finally` block or a registry batch can flush what has been done; a second signal is logged rather
  than raised, so it cannot cut a final write short; a third abandons the write, terminates any running
  tool, and exits with status 130. A `store_adopt --dry-run` scan stopped this way also exits 130, the
  same as a real run, rather than completing its report.
- `import metaquest` no longer calls `setup_logging()`; a library host now configures its own logging, or
  calls `metaquest.utils.logging.setup_logging()` itself for console or file output. The CLI is
  unaffected, since `metaquest.cli.main.main()` already calls it explicitly. A second call to
  `setup_logging()` (for example a host application reconfiguring it) now removes only the handlers it
  installed itself, leaving a host's own handlers, or pytest's `caplog` handler, untouched.
- Seven commands that used to load the registry, do their work, and write it back under a lock covering
  only the write (`select`, `blacklist`, `download_metadata`, `parse_metadata`, `sra_profile`,
  `sra_report`, `sra_validate`, and `status --init`/`--reconcile`) now read a snapshot for their own work
  and record their outcome through `registry_update` or a batch, so the file they write is re-read under
  the lock and never overwrites a concurrent `download_sra`'s committed outcome. `status --reconcile` now
  splits its filesystem scan (no lock held) from applying the resulting plan (under the lock), so read
  counting no longer runs while the registry lock is held.
- As part of that same migration, `download_metadata` and `parse_metadata` now log and drop a single
  accession's failing `record_metadata` call instead of letting the exception end the command before
  anything is saved; the command still finishes and exits 0, with every other accession's metadata
  recorded, and its summary line ("Recorded metadata for N accession(s) in the registry; M dropped")
  counts the dropped records. A user relying on either command failing loudly on a bad record should
  watch the log instead.
- `status --reconcile` checks an untracked accession's folder again before recording it as downloaded,
  and uses a spot-count verdict only for the download it was computed on, so a folder removed, or a
  download redone, between the scan and the apply is not recorded from the stale scan.
- `parse_containment` reads a parsed table from disk before taking the registry lock, not under it.
- Every tabular and text output file metaquest writes is now written atomically, through a uniquely named
  temporary file replaced into place with `os.replace`: CSV and TSV tables, the registry, store sidecars,
  the minimap2 index and its record, per-accession metadata XML (and its copy into a store's `metadata/`
  folder), extracted genome FASTA files, the SRA HTML report, the containment HTML report and the
  explorer page; the registry and sidecar writes also call `fsync`. Figures (PNG, PDF, SVG and the
  interactive plotly HTML plots) are still written directly. Temporary file names are dot-prefixed and
  carry the hostname, pid, and a random token, so they no longer collide between two SLURM nodes writing
  into the same folder, and stay invisible to every folder listing. Text metaquest writes is now explicit
  UTF-8 rather than the platform's default locale encoding.
- `--lock-wait` and `--sra-cache`'s help text now cover a plain project's own per-accession lock, not only
  the store's.
- Five more commands stop between units on a signal, record what finished, and exit with status 130:
  `extract_target_reads` (in both the mapping and the assembly loops), `store_adopt` (including
  `--dry-run`, which used to exit 0), `sra_profile`, `download_metadata` and `parse_metadata`.
- New download result strings reach `failed_accessions.txt` and the registry: `interrupted` (the run was
  stopped before or during this accession), `locked: ...` (another process held the accession's lock
  for longer than `--lock-wait`) and `lock lost: ...` (another process took the lock over while this one
  was stalled, so nothing was published). `locked:` is classified `unknown`, so the retry pass tries the
  accession again and waits up to `--lock-wait` for the lock once more.
- New files on disk: `fastq/.locks/` (a plain project's per-accession locks), `<store>/locks/<ACC>.used`
  (its modification time records when a project was last handed the dataset, read by `store_gc`),
  `<lock>.reclaim` guards held for a few file operations during a stale-lock takeover or a release, and,
  rarely, a `<lock>.reclaimed.<hex>` file left when a process is killed in the middle of a takeover.
- `store_gc --json` dataset entries gained a `downloaded` field (the catalogue's download time).
- A write failure at a former `DataFrame.to_csv` site now raises `DataAccessError` with a "Failed to write
  CSV file" message instead of the underlying `OSError`.
- `Bio.Entrez.email` and `Bio.Entrez.api_key` are now set only around each NCBI efetch call and restored
  afterwards, so a library host that sets its own values keeps them.
- A second `extract_target_reads` that needs the minimap2 index another process is building waits, with
  no time limit, for that build to finish and then reuses the index.

### Fixed

- A concurrent write from `download_sra` could be lost when `select`, `blacklist`, `download_metadata`,
  `parse_metadata`, `sra_profile`, `sra_report`, `sra_validate`, or `status --init`/`--reconcile` ran at
  the same time and wrote back a registry snapshot loaded before the download's outcome was recorded.
- Two downloads of the same accession into one plain project could stage into the same
  `<ACCESSION>_temp` folder and delete each other's in-flight output.
- `store_gc --yes` and `store_verify --fix-state` could remove or rewrite a dataset another process had
  since started using, because each checked the lock only once, before doing its own work, rather than
  again immediately before writing.
- A second `Ctrl-C` (or `SIGTERM`/`SIGHUP`) during a command's final registry flush could abandon the
  write outright; it is now logged and the flush is allowed to finish.
- The store catalogue could be written while a process held the project registry lock, so a slow or
  contended catalogue write held up every other command waiting on the registry; `sra_profile`,
  `sra_report`, and `sra_validate` now record store usage after the registry batch that recorded their
  analysis results has already released the lock.
- `terminate_children()`, called after an interrupted download, left the whole process's stop flag set
  for the rest of that process's life, so an in-process caller that ran a second download after an
  interrupted one found its tools refused to start. Each download run now carries its own stop token;
  interrupting one run's tools no longer affects a concurrent run in the same process.
- `pigz`, started to compress a finished download, was not stopped by an interrupt and could keep
  running after its parent command exited; it now receives the same per-run stop token as the download
  tools and is terminated with them, and a `pigz` stopped this way is reported as "compression skipped
  (interrupted)" rather than "compression failed". Without `pigz`, the Python gzip fallback now checks
  the stop token between 1 MiB blocks, so an interrupted download no longer waits for a whole file to be
  compressed.
- A download into a shared store, or a `store_adopt`, whose holder stalled past the lock's stale window
  (a sleeping laptop, a suspended job) while another project took the lock over could still publish its
  staged folder into the store. Both now confirm the lock is still theirs immediately before the
  publishing rename and report `lock lost: ...` with nothing published otherwise. A registry write and a
  store catalogue commit make the same check and raise `LockLost`, leaving a registry batch's queue
  intact for the next flush.
- A per-sample read extraction interrupted while `samtools fastq` was writing left partial FASTQ files
  under their final names, which `status --reconcile` and `status --init` then recorded as a finished
  extraction, so the next `extract_target_reads` skipped the sample. The export now writes into a
  dot-prefixed folder inside the sample's output folder, and each file is renamed into place only once
  the sample is complete. The names published are recorded in a hidden `.<GENOME_ID>.extracted.json`
  beside them; a rerun that writes fewer files removes only the files that genome's previous run
  recorded, never a file another genome's record lists (a genome called `G1_1` next to a paired `G1`),
  and nothing beyond the files it overwrites in a folder written by an earlier version.
- A run that waited for another run's download of the same accession and then found the files counted
  a download attempt and replaced the other run's result message with "already exists"; it now records
  no attempt and leaves an existing record as it was.
- A "locked: ..." or "lock lost: ..." result whose holder pid, time or lock path contained "403" or "404"
  was classified not-found and skipped by the retry pass; lock messages are now classified before the
  error codes, and 403 and 404 count only as whole numbers.
- `store_gc --yes` removing several datasets could delete the catalogue row of one that a download
  published again after gc had moved the old copy aside; such a dataset now keeps its new copy and row
  and is reported `in_use` with "published again during removal".
- A signal arriving while a command was installing or restoring its signal handlers escaped as a
  traceback and could leave a handler installed; the command now logs "Interrupted" and exits 130, and
  every earlier handler is recorded before the new one goes in, so it is always restored.
- An atomic write onto a read-only file (mode 0o444) failed with `PermissionError`; the target's mode is
  now copied onto the temporary file only just before the rename, so such a file is replaced and stays
  read-only.
- A minimap2 that exited 0 without writing its index published a zero-length index that every later run
  reused; an empty index is now refused when built and rebuilt when found.
- Two first-time `store_init` runs on a new store could each write a marker with its own id, the last one
  replacing the first; the marker is now published with a hard link that fails when it exists, so the
  second run keeps the first run's id (a filesystem without hard links falls back to the old rename).
- The taxonomy validation caches are read and appended as UTF-8, matching the UTF-8 written when a cache
  is reset.
- `store_adopt` no longer copies leftover temporary files (`.<name>.<host>.<pid>.<token>.tmp`, or the
  `<name>.tmp.<pid>` of earlier versions) from a project folder into the store, nor counts them in its
  free-space estimate.
- The warning for a catalogue write that failed after a store publish names the store root and the
  `store_reindex --data-root` command to repair it.
- The copy made by `store_link --mode copy` (and by `download_sra` with a copy link mode) is built under a
  dot-prefixed name and renamed into place, so an interrupted copy no longer leaves a partial dataset
  folder in the project.

### Testing

- `make test`: 2707 passed, 4 deselected, 95% coverage (was 2449 passed at 0.5.1); the same 2707 pass
  with no bioinformatics tool on `PATH`.
- `make check` and `make pipeline` pass on the final state of the branch.

### Upgrade notes

- Lock and log wording changed. A waiter now logs "waiting for <what>: held by <holder>" while it waits;
  a dead-holder takeover logs "Took over the lock on <what>: its holder is no longer running (pid N on
  host H)"; a repeated signal logs "Received SIGTERM again; N more will abandon the write" (or `SIGINT`
  or `SIGHUP` in place of `SIGTERM`). A script that greps metaquest's logs for the old "Gave up waiting
  for ..." wording needs to match the new "<what> is locked by pid N on host H since T: <lock>" message
  instead.
- `fastq/<ACCESSION>_temp` leftovers from a version before this one are still recognized, counted as
  transient bytes, and cleaned up on the next download; new downloads stage under
  `fastq/.metaquest-tmp/` instead.
- A registry lock orphaned by a holder killed on a different host is now judged stale only after 120
  seconds (up from 30), so a writer can be blocked behind an orphaned lock for up to 120 seconds; each
  waiter still gives up and reports a failure after its own 30 second wait, so a busy project can see a
  burst of "locked" results that a retry resolves once the 120 seconds pass. A holder that died on the
  same host is still taken over at once.
- A `--sra-cache` folder shared by two projects without a store is still not locked; avoiding concurrent
  prefetch runs into a shared cache directory remains the caller's responsibility, as before this release.

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
  those as zero-valued rows. Rows keep the previous order (sample by sample, genomes in column
  order), so a table without zero or missing cells is the same as before. `--output-format summary`
  now lists only the samples with a positive containment for the requested taxon; a sample whose
  containment is zero for every genome of that taxon was previously listed with 0.0.

### Removed

- `metaquest.data.metadata.get_unique_sample_attributes`; `parse_metadata` collects the attribute
  names in its single pass and no command called the function.

### Fixed

- A registry batch mutation that raises partway through is now rolled back in full: the registry
  is reloaded from the file and the mutations that succeeded before it are replayed, so no
  accession is left with some of its fields updated.
- During `download_sra`, once a SIGTERM or SIGHUP has been turned into an interrupt, both signals
  are ignored until the queued outcomes have been written, so a repeated `kill` no longer loses
  them.
- `scripts/check_ascii.sh` checks every matching file on disk when its root is not a git
  checkout (a `git archive` export, an unpacked sdist) and fails when it finds no source file at
  all, instead of passing after checking nothing.
- `parse_containment` sorts the parsed table by `max_containment` with a stable sort, so accessions
  with the same maximum keep their input order. The default sort ordered ties differently on
  different CPUs, so the same matches gave a different table, screening order and pinned test
  output on Linux and macOS.
- The CI lint step no longer passes `--extend-ignore` on the command line, which replaced the
  setup.cfg list and re-enabled the pydocstyle style codes the project switches off.

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
