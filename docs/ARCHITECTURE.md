# MetaQuest Architecture

This document provides an overview of the MetaQuest architecture and design principles.

## Design Principles

MetaQuest is designed around the following principles:

1. **Separation of Concerns**: Clear separation between data access, business logic, and user interfaces
2. **Extensibility**: Plugin-based design allowing for easy addition of new features
3. **Maintainability**: Well-defined interfaces and abstractions to simplify maintenance
4. **Robustness**: Comprehensive error handling, validation, and numerical stability
5. **Usability**: Intuitive CLI with consistent command patterns
6. **Reliability**: Extensive test coverage with systematic edge case handling

## Architecture Overview

The application is structured using a layered architecture with the following components:

```
+------------------+
|       CLI        |
+------------------+
         |
+------------------+
|    Core Logic    |
+------------------+
         |
+------------------+     +------------------+
|    Data Layer    |<--->|  Plugin System   |
+------------------+     +------------------+
         |
+------------------+
| External Systems |
+------------------+
```

### Layers

#### CLI Layer
The CLI layer provides the command-line interface for the application. It's responsible for:
- Parsing command-line arguments
- Routing commands to the appropriate handlers
- Formatting output for the user

Most commands are one module each under `metaquest/cli/commands/`. Two command groups are large
enough to be their own packages instead: `metaquest/cli/commands/store/` (one module per store
command: `init`, `status`, `reindex`, `adopt`, `verify`, `link`, `usage`, `gc`, plus a shared
`_shared.py`) and `metaquest/cli/commands/status/` (`command.py` for the `StatusCommand` class,
`suggest.py` for the `--next` suggestions, `render_text.py` for the text-output formatters). Every
command writes to stdout through exactly one channel, `BaseCommand.emit`/`emit_raw`/`emit_json`
(`metaquest/cli/base.py`); library modules log or return their output instead of printing, and a
`make check` gate fails on any other `print(` call in `metaquest/`.

#### Core Logic Layer
The core logic layer contains the business logic of the application. It includes:
- Data models and domain objects
- Validation logic
- Processing algorithms
- Error handling

#### Data Layer
The data layer manages access to data sources and external systems:
- File I/O operations
- Data format conversions
- Communication with external APIs
- Caching mechanisms

#### Project registry
`metaquest/data/registry.py` is the project journal: a `metaquest_registry.json` file in the project
root recording screening, selection, exclusion, download, metadata, analysis, extraction, and
assembly outcomes for every accession a project touches. Record functions (`record_screening`,
`record_selection`, `record_exclusion`, `record_download`, `record_metadata`, `record_analysis`,
`record_extraction`, `record_assembly`, `record_genome`) are called by the commands that produce
those outcomes as each one finishes, each inside a `registry_transaction` that loads the file,
applies one record and writes it back under a lock, so a run never reverts another process's edit.
A screening entry carries its own source and query threshold per genome, and the number kept per
genome is capped (`--registry-max-screened`). Presence is never taken from the registry alone:
`status` and the scanning functions (`scan_downloads`, `scan_metadata`, `scan_extractions`,
`scan_assemblies`) re-check the filesystem, so a file removed by hand is reported as missing rather
than done. Writes are atomic (a temporary file renamed into place) and serialized with a lock file
to avoid concurrent corruption. `update_linked` (add or remove one accession from the project's
`store.linked` list) also lives here, next to the other writers.

The registry file carries a `version` field, currently 2. A schema 1 file loads unchanged; any key the
newer schema added is absent and defaults to an empty value. Schema 2 adds two top-level keys: `project`
(`id`, `name`, `path`, `created`, `organisms[]`, `genome_ids[]`), which makes a project self-describing,
and `store` (`root`, `mode`, `linked[]`), which records the shared data store this project uses, if any.
Every path a project registry stores is written relative to the project root when it lies inside that
root, and absolute otherwise, so moving the project directory does not break the registry.

`metaquest/data/registry_blocks.py` defines every block the registry file holds (screening,
selection, exclusion, download, metadata, analysis, extraction, assembly, plus the `project` and
`store` blocks above) as a dataclass derived from a common `RegistryBlock`: `from_dict`/`to_dict`
round-trip any key the block does not itself declare, so a field this version of MetaQuest does not
know about survives a load-and-save cycle unchanged. `registry.py` reads and writes these typed
blocks rather than raw dictionaries; the on-disk JSON layout is unchanged.

#### Optional dependencies
`metaquest/core/optional.py` is the single point where an optional package is imported. Core
runtime dependencies (pandas, numpy, matplotlib, biopython, lxml, requests) are always available;
everything else (scikit-learn and scipy behind the `analysis` extra, plotly and jinja2 behind
`interactive`, cartopy behind `maps`, sourmash behind `sourmash`) is imported at the point of use
with `optional.require(module, extra, purpose)`, never at module import time and never behind a
silent availability flag. A missing package raises `ConfigurationError`, naming the extra and the
interpreter to install it into; a command that catches broad exceptions re-raises
`ConfigurationError` rather than reporting a degraded result.

#### Shared data store
`metaquest/store` is a package, not a single module, because the store's concerns are independent of
each other and each is small enough to test alone:

- **resolve**: finds the store root, in order, from `--data-root`, `METAQUEST_DATA`, `store.root` in
  the project registry, and `[store] data_root` in `~/.config/metaquest/config.toml`; also reads and
  writes that config file.
- **layout**: the store's on-disk shape (`metaquest_store.json` marker, `catalog.sqlite`, `locks/`,
  `tmp/`, `sra/<accession>/`) and the paths derived from it.
- **sidecar**: reads and writes `<accession>.json` next to each dataset's files: state, layout,
  compression, per-file size and md5, read and base counts, the NCBI spot and base counts used for the
  completeness verdict, and the cached `stats` block described below.
- **catalog**: the SQLite database (`datasets`, `files`, `projects`, `usage` tables, plus an
  `unused_datasets` view) that lets a query answer "which projects used this accession" or "how many
  bytes belong to this organism" without walking every sidecar; falls back from WAL to a rollback
  journal on filesystems (network shares in particular) that do not support WAL.
- **link**: creates and removes the per-accession symlink from a project's `fastq/` folder into the
  store, chooses a relative or absolute target, and detects a dangling link.
- **adopt**: folds an existing per-project `fastq/` folder into the store: copies each accession in
  before removing anything from the project, so a copy always exists somewhere during the operation;
  compares byte content when an accession is already in the store so nothing is duplicated.
- **usage**: writes one row per (accession, project, genome, stage) the first and last time each
  combination is used, along with the hostname it ran on, and reports stale projects (whose registry
  is gone, whose `project.id` no longer matches what the catalogue recorded, or that were last seen
  from a different host than the one running the check). On a store shared between machines, a project
  active elsewhere still looks stale from here; `store_gc` leaves a stale project's datasets alone
  unless `--include-stale` is given.
- **locks**: a per-accession lock file with a heartbeat, so two projects downloading the same accession
  at once cooperate rather than corrupt each other's work; a lock with no heartbeat for 10 minutes
  (`DATASET_LOCK_STALE_SECONDS`) is treated as abandoned and taken over. `--lock-wait` on `download_sra`
  and `store_adopt` bounds how long each waits for another project's lock before giving up.
- **stats**: computes and caches the per-dataset statistics block (streaming exact read and base
  counts, plus a sample for per-read metrics such as GC content), invalidated when the FASTQ file's size
  or modification time changes; used by `sra_profile`, `sra_report` and `sra_validate` (through
  `metaquest.sra.dataset_stats.load_dataset_stats` for the first two).

Nothing outside `metaquest/store` depends on its internal layout; other layers call its public
functions (`resolve_store_root`, `link_dataset`, `record_usage_safe`, and so on) and otherwise treat a
project with no store configured exactly as it behaved before the store existed.

#### Locking

Every lock file in metaquest is built on one mechanism, `metaquest/utils/lockfile.py`, rather than each
caller managing its own `O_EXCL` file:

- **Holder record**: creating a lock (`O_EXCL`) writes a JSON record of `{pid, host, started, token,
  pidns}` (`token` a random hex string; `pidns` the inode of `/proc/self/ns/pid` on Linux, absent
  elsewhere). The record lets a waiter describe who holds a lock ("pid N on host H since T") and tell a
  dead holder from a live one.
- **Heartbeat**: one daemon thread per process refreshes the mtime of every lock this process holds, at
  an interval set by the lock's policy, so a holder doing slow work is not mistaken for dead.
  `os.register_at_fork` replaces the heartbeat and its thread-local bookkeeping in a forked child, so a
  fork never refreshes or releases the parent's lock.
- **Guarded reclaim**: a waiter that judges a lock stale (its mtime older than the policy's
  `stale_seconds`) or its holder's process dead (same host, `os.kill(pid, 0)` raising
  `ProcessLookupError`) does not unlink it directly. It first creates `<lock>.reclaim` with `O_EXCL`;
  only the process holding that guard re-observes the lock and removes it, and only if the holder record
  and inode still match what was observed and the lock is still stale or dead. This is what keeps two
  waiters judging one lock stale at the same instant from both "winning": only one of them can create
  the guard file.
- **Release**: a lock is removed only by the process whose token matches the one written in the file, so
  a process that took over a lock never removes a later holder's lock on the same path.

Known limitation: a dataset lock's staging folder (a plain project's `.metaquest-tmp/<ACCESSION>`, or the
store's own staging path) is named after the accession alone, not the holder's token. A takeover of a
holder that is not dead but merely stalled past the stale window (a suspended process, for example) does
not stop that stalled holder from writing into the same staging folder the new holder is now using, so the
two holders' output can mix there while both run. The mixed folder is never published by the stalled
holder: every destructive or publishing step calls `verify_held` first (the plain-project and store
download publish, the `store_adopt` publish, each registry write and each catalogue commit), and a holder
that finds its lock taken over stops with `LockLost` ("lock lost: ..." for a download) and leaves the
staging folder to the new holder. The per-accession lock file itself is never shared this way; only the
staging path underneath it is.

Three `LockPolicy` configurations (`what`, `stale_seconds`, `wait_seconds`, `poll_seconds`,
`heartbeat_seconds`) cover every lock kind:

| Policy | Stale | Wait | Heartbeat |
|---|---|---|---|
| Registry | 120 s | 30 s | 5 s |
| Catalogue | 120 s | 60 s | 5 s |
| Dataset | 600 s | see below | 10 s |

The registry policy covers `<registry>.lock`; the catalogue policy covers
`<store>/catalog.sqlite.lock`; the dataset policy covers four lock files: a store dataset lock
(`<store>/locks/<ACCESSION>.lock`), a plain project's per-accession download lock
(`<fastq>/.locks/<ACCESSION>.lock`), an index build lock (`<index>.lock`), and a per-sample extraction
lock (`<output>/.locks/<ACCESSION>.<GENOME_ID>.lock`).

The dataset policy's wait varies by caller: a store dataset lock and an index build lock wait without a
time limit (for as long as the holder's heartbeat shows it is alive); a plain project's per-accession
download lock is bounded by `--lock-wait` (0, the default, means no limit); a per-sample extraction lock
never waits (`blocking=False`), so a second process or thread finding a sample already locked reports it
skipped instead of mapping it twice.

Compatibility shims keep every existing caller unchanged: `metaquest/store/locks.py` re-exports the
dataset-lock API (`dataset_lock`, `lock_is_held`, `read_holder`, `LockHeld`, `LockWaitStopped`) built on
`held_lock` with the dataset policy, and `metaquest/data/registry.py`'s `_acquire_lock` is a thin
context manager over `held_lock` with the registry policy.

#### Registry write model

To keep a long command from holding the registry lock across slow work (read counting, profiling,
catalogue writes), the registry is written through one of three patterns, never by loading the file,
doing minutes of work, and writing it back under a single lock:

- **`registry_transaction(path)`**: the low-level context manager, used inside the two helpers below and
  directly by a handful of call sites, that loads the registry under the lock, yields it for one quick
  mutation, and writes it back before releasing the lock.
- **`registry_update(path, mutation)`** (`metaquest/data/registry_batch.py`): loads the registry,
  applies `mutation` to it and writes it back, all under one `registry_transaction`, and returns the
  mutation's result. The pattern for a command that works on a read-only snapshot first (ranking,
  parsing, profiling) and then records what it found: `registry_update(path, partial(record_something,
  ...))`. A raising mutation writes nothing.
- **`RegistryBatch`** (`registry_batch(path)`): queues many small mutations (one per accession, for
  example) and applies them in a few transactions instead of one each, flushing on a count or a time
  interval and on `__exit__`; a failing flush is logged rather than raised, since the block's own
  exception, if any, must propagate instead.

`select`, `blacklist`, `download_metadata`, `parse_metadata`, `sra_profile`, `sra_report`,
`sra_validate`, and `status --init`/`--reconcile` all follow this shape: do their own work against a
snapshot loaded without the lock, then record the outcome through `registry_update` or a batch, so the
file they write is re-read under the lock and never overwrites a concurrent `download_sra`'s committed
outcome. `status --reconcile` splits into `registry_reconcile.scan_reconcile` (lists the filesystem once,
no lock, and rehearses the reconcile on a deep copy of the snapshot) and `apply_reconcile` (re-evaluates
the plan's conditions against the registry `registry_update` hands it, reading no files); `reconcile()`
is `apply(scan(...))` for a caller that wants the old one-call behaviour.

#### Atomic writes and temp names

`metaquest/data/file_io.py` provides `atomic_path`, `write_text_atomic`, `write_bytes_atomic`,
`open_atomic`, and `write_csv`, every one of them writing into a uniquely named temporary file next to
the target (`.{name}.{host}.{pid}.{token}.tmp`, dot-prefixed so it is invisible to every folder listing
built on `visible_files`) and publishing it with `os.replace`, so a reader never sees a partially
written file and a crash leaves at most an orphaned, clearly-named temp file. The registry and store
sidecar writes also call `fsync` before replacing; other outputs do not. `scripts/check_atomic_writes.sh`,
run as part of `make check`, fails on a direct `.write_text`, `.write_bytes`, `.to_csv`, or `open(path,
"w...")` call outside the allowlist in `scripts/atomic_writes_allowlist.txt`, which holds only
`file_io.py` itself.

#### Termination

`metaquest/utils/termination.py` installs a signal handler for `SIGINT`, `SIGTERM`, and `SIGHUP` (and
`SIGBREAK` in place of the latter two on Windows) for the length of one command, through
`graceful_termination()`. The first signal sets a `Termination.stop` event and raises
`KeyboardInterrupt`, so `finally` blocks and `with` exits (a registry batch flush, a tool shutdown) run
normally; a second signal is logged and counted, not raised, so it cannot cut that flush short; after
`ABANDON_AFTER_SIGNALS` (3) signals in total the process terminates any running child and exits at once
with status 130 (`EXIT_INTERRUPTED`). `BaseCommand.run` (`metaquest/cli/base.py`) opens this context
around every command's `execute()`, unless a command sets `graceful_shutdown = False`, puts the
`Termination` on `args._termination`, and on a caught `KeyboardInterrupt` logs, calls
`SecureSubprocess.terminate_children()`, and returns 130.

A per-run stop token (an `Event`, not the process-wide `STOP`) follows a download from `download_sra`
through `_project_download`/`_store_download`, `download_accession`, and `compress_fastq` down to
`SecureSubprocess.run_secure`, so interrupting one run's tools (`terminate_children(stop=token)`) does
not affect a concurrent run's tools in the same process. `STOP` and `SecureSubprocess._stopping` remain
the process-wide emergency stop, set only by the final abandon path and never cleared at the start of an
ordinary run, so an in-process caller that starts a further run after an interrupted one must clear it
itself if it wants that further run's own tools to start.

#### Plugin System
The plugin system enables extensibility:
- Format plugins for different file formats
- Visualization plugins for different visualization types
- Processing plugins for different analysis methods

### Key Components

#### Core Components

- **Models**: Data classes representing domain objects like `Containment` and `SRAMetadata`
- **Validation**: Functions for validating input data and configurations
- **Exceptions**: Custom exception hierarchy for clear error reporting

#### Data Access Components

- **file_io**: Abstract file operations for reading/writing various file formats
- **branchwater**: Functionality for working with Branchwater data
- **metadata**: Functions for downloading and processing metadata
- **sra** (`metaquest/data/sra/`): A package, not a single module, for downloading and working with
  SRA data: `fastq` (finding and reading FASTQ files, verifying a download), `cleanup` (transient
  folder handling), `accession` (running `prefetch`/`fasterq-dump` for one accession), `retry`
  (parallel download with retries), `store_handoff` (linking a download into the shared store), and
  `download` (the CLI-facing entry point); the package's `__init__.py` re-exports the public
  functions other layers import, so `from metaquest.data.sra import download_accession` still works
- **registry**: The project journal (`metaquest_registry.json`); records dataset state and
  re-checks presence against the filesystem
- **registry_blocks**: Typed dataclasses for every block the registry file holds, described in
  "Project registry" above
- **store** (`metaquest/store/`): The shared data store package, described in "Shared data store"
  below; a project that never runs `store_init` never touches it

#### Processing Components

- **containment**: Algorithms for analyzing containment data
- **counts**: Functions for counting and summarizing metadata
- **statistics**: Statistical analysis utilities, including `compare_group_means` (t-test for two
  groups, one-way ANOVA for more; used by `sra_report --groups-file`)
- **status_report**: Builds the `status` command's report (present/missing reconciliation, stage
  filtering, store link status) from the registry and the filesystem; kept in `processing/` rather
  than `cli/` so it has no dependency on the CLI layer, and returns data that `cli/commands/status/`
  formats for text or JSON output

#### Visualization Components

- **plots**: Functions for generating various types of plots
- **reporting**: Tools for generating reports

#### Plugin System Components

- **base**: Base classes and registries for plugins
- **formats**: File format plugins
- **visualizers**: Visualization plugins

#### Utility Components

- **logging**: Logging configuration and utilities
- **config**: Configuration management

## Data Flow

1. **Input**: User provides commands via CLI
2. **Command Handling**: CLI routes to appropriate command handler
3. **Data Access**: Commands interact with data sources via the data layer
4. **Processing**: Data is processed using core logic components
5. **Visualization**: Results are visualized or formatted for output
6. **Output**: Results are presented to the user via CLI

## Plugin System

The plugin system uses a registry pattern:

1. Plugins inherit from a base `Plugin` class
2. Plugins are registered in a `PluginRegistry`
3. The application can discover plugins at runtime
4. Plugins provide specific implementations for abstract operations

### Example: Format Plugins

Format plugins handle different file formats:

- `BranchWaterFormatPlugin`: Handles Branchwater CSV files
- `MastiffFormatPlugin`: Handles Mastiff CSV files

Each plugin provides standard methods like:
- `validate_header`: Validates file headers
- `parse_file`: Parses file content
- `extract_metadata`: Extracts metadata from parsed content

## Error Handling

The application uses a comprehensive error handling approach:

1. **Exception Hierarchy**: Custom exception hierarchy starting with `MetaQuestError`
2. **Specific Error Types**: Different exception types for various error categories
3. **Layered Handling**: Each layer handles errors appropriate to its level
4. **Numerical Stability**: Proper handling of edge cases in statistical computations
5. **Input Validation**: Comprehensive validation of user inputs and external data
6. **Graceful Degradation**: Robust handling of degenerate cases and boundary conditions
7. **User-Friendly Display**: CLI layer catches and formats errors for user display

## Configuration Management

The application manages configuration through:

1. Default configuration values
2. Configuration file in user's home directory
3. Environment variables
4. Command-line overrides

## Future Extensions

The architecture is designed to accommodate future extensions:

1. New file format plugins
2. Additional visualization types
3. More analysis algorithms
4. Web interface or API
5. Database integration