# Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MetaQuest correct when several threads, several processes and several hosts work on one project or one shared store at the same time, then close the production gaps a cluster user meets (configuration, exit codes, logging, environment check, resource controls, timing, packaging, HPC documentation), delivered as releases 0.6.0 and 0.7.0.

**Architecture:** One lock mechanism (`utils/lockfile.py`) with heartbeat, guarded reclaim and dead-holder takeover behind every lock file; registry read-modify-write through short batch transactions; atomic writes everywhere; a graceful termination context applied by `BaseCommand.run`; per-run stop tokens; a real multi-process test harness. Phase 2 adds `core/settings.py`, exit codes on the exception classes, logging and progress infrastructure, `metaquest doctor`, resource guards and timing fields, all in new modules so the frozen ones shrink.

**Tech Stack:** Python 3.12 (tomllib, os.register_at_fork), `O_EXCL` lock files, sqlite3, pandas, requests with urllib3 Retry, lxml, pytest with subprocess-spawned CLI processes and fake tools on PATH.

**Spec:** the concurrency inventory and production-gap analysis of 2026-09-30 (in-session; findings are restated in the Context below with file:line references verified against main 5f097f3).

## Review Focus

1. A lock holder that stalls longer than its stale window (a sleeping laptop) must not have its work destroyed by a waiter; callers check `verify_held` before destructive steps.
2. Two waiters judging one lock stale at the same instant must produce exactly one holder (guarded reclaim).
3. A mutation queued by `download_sra` while `select`/`blacklist` runs must survive into the file; no command may write a registry snapshot older than its lock acquisition.
4. A SIGTERM followed by another signal during the final flush must still write the queued outcomes; the third signal abandons with children terminated.
5. An interrupted or killed process must leave no file another process mistakes for a complete output (staging plus one rename, dot-prefixed temp names, no pid-only names).

---

# Production readiness plan: correct under multiple threads and processes (2026-09-30)

## Context

MetaQuest 0.5.1 (main 5f097f3, 2449 tests, CI and nightly smoke green) is fast and well tested, but every
test of concurrency runs threads inside one process or fakes a second process with a hand-written lock
file. A read of every concurrency path found that the software assumes one process per project:

- The registry lock is an `O_EXCL` file reclaimed by age alone (30 s) with no heartbeat, so a holder that
  runs longer than 30 s (store_reindex replay, a slow NFS write, a sleeping laptop) has its lock taken, and
  its unconditional unlink on exit then removes the new holder's lock. Two waiters can also both judge a
  lock stale and both "win" (check-then-unlink race). `registry.py:197-219`, `store/locks.py:126-138`.
- Seven commands hold a loaded registry across seconds or minutes of work and write it back under a lock
  that covers only the write (`select`, `blacklist`, `sra_profile`, `sra_report`, `sra_enhanced`,
  `download_metadata`/`parse_metadata`, `status --init/--reconcile`), so a concurrent `download_sra`'s
  committed outcomes are overwritten by the stale snapshot.
- A plain project (no shared store) has no per-accession lock: two `download_sra` runs on one `fastq/`
  folder both stage into `fastq/<ACC>_temp`, each deleting the other's in-flight output
  (`data/sra/accession.py:356-358`), and a second process sees half-compressed files as "already
  downloaded".
- `store_gc --yes` checks `lock_is_held` once and deletes later without the dataset lock
  (`cli/commands/store/gc.py:213-244, 429-458`); `store_verify` rewrites sidecars without it; a catalogue
  write failure after a successful store publish marks the download failed; read extraction takes the
  store catalogue lock while holding the registry lock.
- Graceful SIGTERM/SIGHUP handling exists only in `download_sra`; a second Ctrl-C during the final flush
  loses queued outcomes; every other long command dies without cleanup.
- Tabular outputs, the user config file and the minimap2 index JSON are written non-atomically; temp
  names are pid-only and collide across SLURM nodes.
- Module globals (`STOP` event, `Bio.Entrez` credentials, `gtdb._session`, `_extra_roots`, logging
  configured on import) break two runs in one process.

A second read found the production gaps a cluster user hits: a fixed 1 h tool timeout, no free-space
check before downloads, dead `--log-file`, per-accession INFO flooding batch logs, a silent 4-worker cap
that ignores the SLURM cpuset, no environment check command, megahit without a cgroup-aware memory cap,
exit code 1 for every failure, no timing records, unpinned tool versions, no HPC documentation.

The user chose: both halves, delivered as two releases (0.6.0 concurrency, 0.7.0 hardening), including
the two visible behaviour changes (exit codes 3 and 4; tool timeout default becomes "no limit" with
`--timeout` and `METAQUEST_TIMEOUT` to set one). Execution: subagent-driven in a worktree, as in the
previous plans (plan committed first, ledger, one implementer and one reviewer per task, whole-branch
review, one fix wave, gates, real-data checks on sekvens2, finishing menu; merge, push and tag are the
user's decisions).

## Global constraints

Python >= 3.12 only; no new runtime dependency (no filelock, psutil, hypothesis); ASCII only in code,
tests, scripts and config (`scripts/check_ascii.sh`); 120-column lines including Markdown; module size
guard (800 lines, MI 20; `registry.py` 1110 and `read_extraction.py` 1210 are frozen exceptions and must
shrink, never grow, so new code goes into new modules); flake8 D101-D103 on every new module (list them
in `setup.cfg`'s comment, never in `per-file-ignores`), no bare `except Exception` (B902); CLI flags with
dashes; plain scientific wording; tests on `tmp_path` with fake tools on `PATH`, never a real tool, store
or network; existing CLI flags keep their meaning (new flags only); commit trailer
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; check trailers before the branch review.

Existing mechanisms to reuse (do not reinvent): `dataset_lock` semantics in `metaquest/store/locks.py`
(heartbeat thread, holder record, stale takeover, `should_stop`), `registry_transaction` and
`RegistryBatch` (`registry_batch.py`, rolls back a failing mutation), temp + `os.replace` writes,
`SecureSubprocess` child tracking and `terminate_children` (`utils/security.py`), `record_usage_safe` /
`record_usage_many` (`store/usage.py`), `BaseCommand.emit/emit_json`, `tests/helpers_extraction.py`
fake-tool pattern, `tests/perf_fixtures.py`, the `perf` marker and `METAQUEST_PERF_SCALE`.

Design decisions already taken (do not reopen): keep the `O_EXCL` lock-file scheme (works on exFAT, NFS,
SMB; `flock` needs lockd and cannot name the holder); keep the catalogue in `DELETE` journal mode (WAL
is unsafe on NFS); no version stamp on the registry (the batch mutation model gives the same safety
without a schema change); never hold the registry lock across ranking, profiling or read counting;
no default timeout on dataset-lock waits; extraction stays sequential per sample (sharding is SLURM's
job); no cross-process NCBI pacing; exit codes 0, 1, 2, 3, 4, 130 only (no code 5).

---

# Phase 1: correct under concurrency (release 0.6.0)

## Task 1: one lock mechanism with heartbeat, safe reclaim and dead-holder takeover

Files: new `metaquest/utils/lockfile.py`; `metaquest/store/locks.py` (thin wrapper, re-exports
`LockWaitStopped`, `dataset_lock`, `lock_is_held`, `read_holder`); `metaquest/data/registry.py`
(`_acquire_lock` becomes a shim over `held_lock`, module shrinks; keep `LOCK_WAIT_SECONDS` and
`LOCK_STALE_SECONDS` as attributes read at call time because tests monkeypatch them);
`metaquest/store/catalog.py` (`catalog_write` uses a catalogue policy); `metaquest/core/constants.py`;
`setup.cfg`; `tests/test_module_sizes.py` (lower registry.py ceiling).

Design:
- Holder record `{pid, host, started, token, pidns}` (`secrets.token_hex(8)`; `pidns` is the inode of
  `/proc/self/ns/pid` on Linux, absent elsewhere). Release compares the token and never removes another
  holder's lock. A non-JSON holder (old versions, fabricated test files) follows the age rule.
- One daemon heartbeat thread per process refreshes every held lock's mtime every `heartbeat_seconds`;
  started lazily; `os.register_at_fork(after_in_child=...)` clears it in a forked child; a refresh that
  finds the file gone or another token marks the lock lost.
- Reclaim only under `<lock>.reclaim` created with `O_EXCL`: re-read holder and age under the guard,
  unlink only if unchanged and still stale or the holder is dead (same host, same pidns, `os.kill(pid, 0)`
  raises `ProcessLookupError`; `PermissionError` means alive); a guard older than 60 s is removed.
- Same-thread re-entry raises `LockReentry` at once (thread-local set of held paths) instead of a 30 s
  hang; other threads contend normally.
- `blocking=False` raises `LockHeld` when the lock is taken.

```python
class LockHeld(DataAccessError): ...
class LockLost(DataAccessError): ...
class LockReentry(DataAccessError): ...
@dataclass(frozen=True)
class LockPolicy:
    what: str; stale_seconds: float; wait_seconds: float = 0.0; poll_seconds: float = 1.0
    heartbeat_seconds: float = 10.0
@contextmanager
def held_lock(lock: Path, policy: LockPolicy, should_stop: Optional[Callable[[], bool]] = None,
              blocking: bool = True) -> Iterator[Path]
def verify_held(lock: Path) -> None            # raises LockLost
def read_holder(lock: Path) -> Dict[str, Any]
def describe_holder(holder: Dict[str, Any]) -> str
def holder_is_dead(holder: Dict[str, Any]) -> bool
```

Policies: dataset lock stale 600 s, heartbeat 10 s, wait 0 (as today); registry wait 30 s (unchanged,
fits SLURM KillWait), stale 120 s, heartbeat 5 s, poll 0.05 s; catalogue `CATALOG_LOCK_WAIT_SECONDS = 60`,
stale 120 s, heartbeat 5 s, error text "Store catalogue is locked by pid N on host H since T: <lock>".
Lock file names unchanged; mixed old/new versions stay safe (new holders heartbeat inside the old 30 s
age).

Tests (`tests/test_lockfile.py`): reclaim race with a forced stale judgement (exactly one winner);
dead same-host pid taken over at once (pid of a finished subprocess); foreign-host holder respected until
stale; release never removes a foreign lock; heartbeat keeps a holder past a 0.3 s stale window;
re-entry raises; `blocking=False` raises `LockHeld`; bare-pid lock follows the age rule (existing
`test_data_registry` and `test_store_locks` cases pass unchanged); 50-thread counter stress with
enter/exit markers asserting no overlap; 8-process stress with the `multiprocessing` spawn context.

## Task 2: atomic write helpers and every writer on them

Files: `metaquest/data/file_io.py`; call sites: `store/resolve.py:81` (config), `read_extraction.py:150`
(index JSON, written after the index replace), `blacklist.py:28`, `select.py:218`, `sra.py:298` (report
file), `retry.py:208` (`failed_accessions.txt`), `branchwater.py:317`, `branchwater_search.py:197-204`,
`sra_metadata.py:475/524`, `taxonomy.py:382/515`, `explore.py:297`, `advanced_analysis.py:94/103`,
`sourmash_plugin.py:198/205`, `profiles.py:60`, `sra_report.py:216`, `sra_profile.py:201`,
`reporting.py:233`, `metadata.py:711`, `processing/counts.py:156-161`, `results.py:75`,
`status/command.py:101-102`; pid-only temp names replaced in `sidecar.py:235`, `metadata.py:61`,
`registry.py:240` (with `fsync=True`, temp removed in `finally` on any exit including
`KeyboardInterrupt`, via a success flag not `except BaseException`), `fastq.py:425` (keep a `.gz.tmp`
prefix or update the `fastq_files` exclusion at `registry.py:823` in the same commit),
`read_extraction.py:144`; new grep guard `scripts/check_atomic_writes.sh` in `make check`.

```python
def unique_temp_path(target: Path) -> Path        # parent / f".{name}.{host}.{pid}.{token_hex(4)}.tmp"
@contextmanager
def atomic_path(target: Union[str, Path], fsync: bool = False) -> Iterator[Path]
def write_text_atomic(path, text: str, encoding: str = "utf-8", fsync: bool = False) -> Path
def write_csv(df, file_path, **kwargs) -> None    # same signature, now atomic; compression inferred from the real suffix
```
Temp created with `os.open(..., O_CREAT|O_EXCL|O_WRONLY, 0o666)` so the umask applies and shared-store
files stay group-readable; an existing target's mode is copied; symlink targets resolved; `os.replace`
retried 5 times over 0.5 s on `PermissionError` (Windows). Dot-prefixed leftovers are invisible to
listings.

Tests: interrupt mid-write leaves old content and no temp; `.tsv.gz` round-trips; group-readable mode
survives; a reader thread polling during 200 rewrites never sees a partial file; the grep guard fails on
`.to_csv(<path>` / `.write_text(` under `metaquest/` outside an allowlist; a `KeyboardInterrupt` from a
patched `os.replace` in `_write_registry` leaves no `.tmp`.

## Task 3: per-accession lock and one-rename publish for plain projects

Files: `data/sra/accession.py`, `retry.py`, `download.py`; move `_publish_store_dataset` from
`store_handoff.py` to `cleanup.py` as `publish_folder` (re-export); `cli/commands/sra.py` (`--lock-wait`
help now covers plain projects; `_transient_folders` and `transient_bytes` include
`fastq/.metaquest-tmp/*` and old `fastq/<ACC>_temp`).

```python
def project_lock_path(fastq_folder, accession) -> Path          # <fastq>/.locks/<ACC>.lock
def _project_download(accession, output_folder, num_threads=4, force=False, temp_folder=None, *,
                      lock_wait: float = 0.0, stop: Optional[threading.Event] = None,
                      **download_kwargs) -> Tuple[bool, str]
def publish_folder(staged: Path, target: Path, tmp: Path) -> None
```
Inside the lock (dataset-lock policy): re-run `_check_existing_download(target)` and return "already
exists" when found; download into `<fastq>/.metaquest-tmp/<ACC>` with `sra_cache` set explicitly to the
old default `<fastq>/.sra-cache`; `verify_held`; `publish_folder` with a single rename so a second
process never sees half-compressed files. `LockWaitStopped` becomes `(False, "interrupted")`.
`_check_existing_downloads` stays as an up-front hint only. Document that a user-supplied `--sra-cache`
shared by two projects without a store is not locked.

Tests (`tests/test_data_sra.py`): two threads on one accession, fake tool runs once, the other reports
"already exists"; a watcher thread polling `accession_has_fastq` never observes an incomplete folder;
`.locks` and `.metaquest-tmp` invisible to `scan_downloads` and `status`; `--lock-wait 1` against a held
lock gives up naming the holder.

## Task 4: registry read-modify-write through batches; catalogue never under the registry lock

Files: `registry_batch.py` (`registry_update`); new `metaquest/data/registry_reconcile.py`
(`reconcile`, `_fill_missing_download_verdicts`, `_store_verdict` move out of `registry.py`, re-exported;
lower the ceiling); commands `select.py`, `blacklist.py`, `sra_profile.py`, `sra_report.py`,
`sra_enhanced.py`, `metadata.py`, `status/command.py`, `read_extraction.py`.

```python
def registry_update(path, mutation: Callable[[Registry], T]) -> T     # load, mutate, save under one short lock
@dataclass
class ReconcilePlan: ...
def scan_reconcile(snapshot: Registry, paths: ProjectPaths) -> ReconcilePlan   # disk and read counting, unlocked
def apply_reconcile(registry: Registry, plan: ReconcilePlan) -> ReconcileReport # fast, re-checks each condition
```
Per command: `select` ranks on a snapshot, writes its output atomically, then
`registry_update(path, partial(record_selection, ...))`; `blacklist` does the whole read-modify-write of
`blacklist.txt` inside `registry_update`; `sra_profile`/`sra_report`/`sra_validate` queue
`record_analysis` on a no-flush batch and call `record_usage_many` after it exits (one catalogue lock,
never inside the registry lock); metadata commands parse outside then queue `record_metadata`;
`status --init` refuses (raises) if the file appeared meanwhile instead of overwriting; `status
--reconcile` = scan, then `registry_update(apply)`; `read_extraction.py:340,486` move
`record_usage_safe` after the `with registry_transaction` block using the returned registry.

Tests: one deterministic test per command (block the long step on an `Event`, record a download for
SRR9 from another thread through `registry_transaction`, release, assert both changes are in the file);
`catalog_write` patched to assert the registry lock file is absent when read extraction records usage;
`apply(scan(r))` equals the old `reconcile` on the existing fixtures.

## Task 5: store_gc removal under the dataset lock; store_verify write-back under it

Files: `cli/commands/store/gc.py`, `store/locks.py`, `store/link.py`, `data/sra/store_handoff.py`,
`core/constants.py`, `cli/commands/store/verify.py`.

```python
def touch_dataset_use(paths, accession) -> None        # utime <store>/locks/<ACC>.used; silent on OSError
def last_dataset_use(paths, accession) -> Optional[float]
GC_RECENT_USE_GRACE_SECONDS = 86400.0
```
`touch_dataset_use` in `_link_result` after `_store_fetch` links, in `store_link` and `store_adopt`.
`_remove_candidates` per dataset: `dataset_lock(..., blocking=False)` (held -> `report["in_use"]`); under
the lock re-verify sidecar `downloaded`, usage project ids, `linked_by` empty, last use older than the
grace; `os.replace(sra_dir, tmp/<ACC>_gc)`; one `catalog_write` deleting the rows (on failure rename
every aside folder back); then remove the aside trees. Leftovers take the same non-blocking lock; `_gc`
added to `_accession_of_leftover`. `store_verify` wraps `write_sidecar` + `catalog_write` in the
non-blocking lock, re-reads the sidecar under it and skips if `downloaded`/`files` changed; held ->
"in use, not updated".

Tests: candidate locked by another thread before removal is reported `in_use` and survives; a dataset
touched a second ago is kept; catalogue failure after the rename restores the folders; verify skips a
dataset replaced between check and write-back. Release note: recently linked datasets are kept one day.

## Task 6: catalogue busy timeout and publish tolerance

Files: `store/catalog.py`, `data/sra/store_handoff.py`, `core/constants.py`.

`sqlite3.connect(str(path), timeout=CATALOG_BUSY_TIMEOUT_SECONDS)` (30 s; the implicit default of 5 s
is shorter than a `store_reindex` replay). Post-publish:
`_catalogue_published(store, sidecar) -> bool` never raises `DataAccessError`, logs a warning naming
`store_reindex`; `_store_fetch` continues to link and returns `f"{message}; catalogue pending; stored"`
(suffix last because `sra.py:434` matches `endswith("; stored")`).

Tests: catalogue lock held past a 0.2 s wait during `_store_fetch` gives success, link exists, warning
logged, later `store_reindex` adds the row; a reader during a 1 s `BEGIN EXCLUSIVE` succeeds;
`test_store_catalog.py:369-402` unchanged.

## Task 7: graceful termination for every command

Files: new `metaquest/utils/termination.py`; `cli/base.py` (`BaseCommand.run`, `graceful_shutdown = True`,
`CommandRegistry._add_parser` sets `func=command.run`); `cli/commands/sra.py` (remove
`_termination_raises_interrupt`, keep the name as an alias for `tests/test_cli_commands.py:3172-3311`);
`registry.py` temp cleanup already covered by Task 2.

```python
ABANDON_AFTER_SIGNALS = 3
@dataclass
class Termination:
    stop: threading.Event; signum: Optional[int] = None; repeats: int = 0
    @property
    def requested(self) -> bool
@contextmanager
def graceful_termination(stop: Optional[threading.Event] = None,
                         abandon_after: int = ABANDON_AFTER_SIGNALS) -> Iterator[Termination]
```
SIGINT, SIGTERM, SIGHUP (Windows: SIGINT, SIGBREAK). First signal: set `stop`, raise
`KeyboardInterrupt`. Later signals: logged with how many more abandon the write, not raised. Third:
`terminate_children(grace=0)` then `os._exit(130)` (safe because of heartbeat locks and the dead-pid
check). No-op off the main thread. `BaseCommand.run` wraps `execute`, puts `term` on
`args._termination`, and on `KeyboardInterrupt` logs "Interrupted (<signal>)", terminates children and
returns 130 (unchanged code; a later phase may add 128 + signum). Commands that must not be wrapped set
`graceful_shutdown = False`.

Tests: `os.kill(os.getpid(), SIGTERM)` inside the context: first raises, second is logged, third calls a
patched `os._exit`, handlers restored; nothing installed off the main thread; a second SIGINT sent from a
patched `RegistryBatch._apply_all` still lets the flush complete; existing signal tests pass through the
alias.

## Task 8: interrupt paths for the other long commands

Files: `cli/commands/read_extraction.py`, `store/adopt.py`, `sra_profile.py`, `metadata.py`.
Read `args._termination.stop` between samples; stage every per-sample output and rename on completion so
an interrupted sample leaves nothing a rerun mistakes for done. `run_secure` already kills its child on
`BaseException` in the waiting thread.

Tests (in-process with fake tools; process-level in Task 12): SIGTERM during a fake minimap2 run gives
exit 130, no child left, finished samples recorded, the interrupted one absent.

## Task 9: per-run stop token through the download chain

Files: `data/sra/accession.py`, `retry.py`, `download.py`, `store_handoff.py`, `utils/security.py`,
`cli/commands/sra.py`.
`download_sra(..., stop: Optional[threading.Event] = None)` creates one when absent; passed to
`_download_with_retries`, `_execute_parallel_downloads`, `_retry_failed_downloads`, the worker,
`download_accession(..., stop=None)`, `_run_download_tool(executable, args, stop)`; `_store_download`
uses `should_stop=stop.is_set`; the stop condition is `stop.is_set() or STOP.is_set()`; `STOP` stays as
a process-wide emergency stop and is no longer cleared at run start (`retry.py:246`).
`run_secure(..., stop=None)`; `terminate_children(grace=5.0, stop=None) -> int` (None = all);
`_children: Dict[Popen, Optional[Event]]`; `_stopping`/`clear_stopping` kept.

Tests: two `download_sra` runs in two threads with fake workers, cancelling A's token stops only A's
children; existing tests setting `accession_mod.STOP` pass.

## Task 10: small global fixes

Files: `utils/security.py` (`_extra_roots_lock`), `data/metadata.py` (`_PACE_LOCK` held while sleeping;
`_ENTREZ_LOCK` and `_entrez_credentials(email, api_key)` context that sets and restores `Entrez.email`
and `Entrez.api_key` around each efetch), `data/gtdb.py` (double-checked lazy init under
`_SESSION_LOCK`), `utils/logging.py` (`setup_logging` tags its handlers `handler._metaquest = True` and
removes only tagged ones), `metaquest/__init__.py` (import-time `setup_logging()` replaced by a
`NullHandler` on the `metaquest` logger; `main()` still calls `setup_logging`).

Tests: `import metaquest` leaves a pre-installed root handler; `setup_logging` twice installs one console
handler and caplog still captures; 20 threads on `add_allowed_root` leave one entry; Entrez credentials
restored after a call. CHANGELOG: library users must call `setup_logging()` themselves.

## Task 11 (optional, do if Tasks 1-10 land cleanly): extraction and index guards

Files: new `metaquest/data/extraction_locks.py` (read_extraction.py must not grow).
`build_index` under `held_lock(<index>.lock, stale 600, heartbeat 10)` so a second process waits and
reuses; per-sample lock `<output_folder>/.locks/<ACC>.<genome>.lock` with `blocking=False`, held ->
"in progress elsewhere, skipped" (right for overlapping SLURM shards).
Tests: two threads extracting one sample, fake minimap2 runs once, one thread reports skipped.

## Task 12: multi-process test harness and process tests

Files: new `tests/helpers_processes.py`, new `tests/test_concurrency_processes.py`, `pyproject.toml`
(marker `multiprocess`, runs by default; skip signal tests on Windows), `CLAUDE.md` testing section.

```python
def cli_env(tmp_path, fake_bin, store=None, **extra) -> Dict[str, str]   # PYTHONPATH=repo, HOME/XDG under tmp_path,
                                                                        # METAQUEST_DATA set or removed, PATH=fake_bin + python dir
def spawn_cli(args, cwd, env) -> subprocess.Popen                       # [sys.executable, "-m", "metaquest.cli.main", ...], start_new_session=True
def run_cli(args, cwd, env, timeout=20.0) -> subprocess.CompletedProcess
def install_fake_tools(bin_dir, barrier_dir) -> None   # fasterq-dump, prefetch, pigz as Python scripts: write <barrier>/<ACC>.<pid>.started,
                                                       # record pid, wait while <barrier>/hold-<ACC> exists, write mates, exit 143 on SIGTERM
def wait_for(predicate, timeout=10.0, interval=0.05) -> None
def alive(pid) -> bool
```
Tests, each bounded to about 10 s with file barriers: (1) same accession, one project: one `started`
file, both exit 0, complete files, registry `downloaded` once, no `.tmp`; (2) same accession, one store,
two projects: one fetch, both linked; (3) registry contention: `download_sra` with three accessions (one
held) while `blacklist` and `select_datasets` run to completion; final registry has the exclusion, the
selection and all three outcomes; (4) SIGTERM then a second SIGTERM 0.2 s later mid-download: exit 130
within 10 s, SRR1 recorded, no fake pid alive, no lock files, SRR2 not published; (5) SIGKILL of a lock
holder: the second process takes over at once with a warning naming the dead pid; (6) `store_gc --yes`
racing a held download: reported `in_use`, survives; (7) in-process: two runs in threads do not share a
stop token; the lock stress tests from Task 1.

## Task 13: release 0.6.0

Files: `pyproject.toml`, `metaquest/__init__.py`, `CHANGELOG.md`, `README.md` (short "Running several
instances" paragraph: per-accession locks, `--lock-wait` for plain projects, signal behaviour), `docs/ARCHITECTURE.md`
(locking section: one lock mechanism, policies, where each lock lives), `CLAUDE.md` (rules: never hold
the registry lock across long work, use `registry_update`/batches, atomic writers only, per-run stop
tokens, process tests in `tests/test_concurrency_processes.py`).
CHANGELOG sections: Added (process tests, `registry_update`), Changed (locks heartbeat and reclaim,
plain-project staging under `fastq/.metaquest-tmp`, `--lock-wait` on plain projects, recently linked
datasets kept one day by gc, logging not configured on import, catalogue-pending downloads count as
success), Fixed (lost updates, temp collisions, gc and verify races, second Ctrl-C, catalogue under the
registry lock). `make check`, `make test`, `make pipeline`, `python -m build`, no tag.

---

# Phase 2: production hardening (release 0.7.0)

## Task 14: runtime settings module

Files: new `metaquest/core/settings.py`; `store/resolve.py` keeps `config_path`/`read_config` as
re-exports (tests import them; `read_config` now maps `TOMLDecodeError`/`OSError` to
`ConfigurationError` naming the file); `tests/conftest.py` autouse `reset_for_tests`.

```python
@dataclass(frozen=True)
class SettingSpec: name; parse: Callable[[str], Any]; default; env: str; doc: str; cli_dest: Optional[str] = None
@dataclass(frozen=True)
class Resolved: name; value; source   # "--timeout" | "METAQUEST_TIMEOUT" | "config [runtime] timeout" | "default"
def resolve_setting(name, cli_value=None) -> Resolved
def resolve_all(args=None) -> Dict[str, Resolved]
def activate(args) -> RuntimeSettings; def active() -> RuntimeSettings; def reset_for_tests() -> None
```
Precedence: CLI flag, `METAQUEST_<NAME>` env, `config.toml [runtime]`, default. Settings:
`subprocess_timeout` (0 = none, flag `--timeout` on download_sra and extract_target_reads),
`max_workers_cap` (env/config, default 4), `lock_wait` (existing flag), `registry_lock_wait/stale`,
`dataset_lock_stale`, `lock_heartbeat`, `catalog_lock_wait`, `ncbi_email`, `ncbi_api_key` (`NCBI_API_KEY`
kept), `temp_folder`, `log_file`, `log_level`, `progress_every` (50), `log_host` (false),
`min_free_gb` (10), `assembly_memory` ("auto"). Flags mapping to settings get `default=None`
(`--lock-wait` in `sra.py:249` and `store/adopt.py:94`); `--email` becomes optional where a setting
resolves. Lock policies in Task 1 read the active settings. The store root keeps its own precedence.
Tests (`tests/test_core_settings.py`): one test per precedence level, source strings, bad env/config
values raise `ConfigurationError` naming the source, malformed TOML, unknown key warns, XDG honoured.

## Task 15: exit codes on exceptions

Files: `core/exceptions.py`, `cli/base.py` (`BaseCommand.fail(error, context) -> int` logs with
`exc_info` and returns `exit_code_for(error)`), `cli/main.py` (both except branches return
`exit_code_for(e)`), `registry.py:209` and `store/locks.py:103` raise `LockTimeoutError`,
`sra_metadata.py` raises `NetworkError` on connection errors, 429 and 5xx, `download_sra` returns 4 when
every failure classifies as network, missing tools raise `ConfigurationError` (Task 17), the 17 command
modules replace `except MetaQuestError ... return 1` with `return self.fail(e, "...")`.

```python
class ExitCode(IntEnum): OK = 0; FAILURE = 1; USAGE = 2; CONFIGURATION = 3; TRANSIENT = 4; INTERRUPTED = 130
class MetaQuestError(Exception): exit_code: int = ExitCode.FAILURE
class ConfigurationError(MetaQuestError): exit_code = ExitCode.CONFIGURATION
class TransientError(DataAccessError): exit_code = ExitCode.TRANSIENT
class LockTimeoutError(TransientError); class NetworkError(TransientError)
def exit_code_for(error: BaseException) -> int
```
Migration: run the suite after the class change and before the command migration; the failing `== 1`
assertions (about 150 exist, fewer than 15 expected to change) are the exact change set. New tests:
`tests/test_exit_codes.py` (parametrized), `main()` returns 3/4/130, `download_sra` returns 4 for
all-network failures with the downloader mocked.

## Task 16: log file, quiet/verbose, PID and host, tracebacks, progress summaries

Files: `utils/logging.py`, `cli/main.py`, `cli/base.py` (`add_global_options(parser, suppress_defaults)`
on the main parser and every subparser with `default=argparse.SUPPRESS`), new `metaquest/utils/progress.py`,
demoted per-item lines in `accession.py:362,380`, `retry.py:169`, `metadata.py:190,325`,
`read_extraction.py` per-sample INFO (after Task 20 frees space), `tests/test_cli_main.py` (four
`assert_called_once_with(level=...)` match on the level only).

`setup_logging(level, log_file=None, *, show_host=False, console_traceback=False)`: file handler appends,
format `%(asctime)s %(hostname)s[%(process)d] %(levelname)s %(name)s: %(message)s`, level
`min(level, INFO)`; console formatter drops the traceback unless DEBUG; the file always has it; the
"Use --log-level DEBUG" hint only when there is no log file. Global flags `--log-file`, `-q/--quiet`,
`-v/--verbose` (mutually exclusive), `--progress-every N`. `main()` logs version, argv, host, PID and
`SLURM_JOB_ID`/`SLURM_ARRAY_TASK_ID` at DEBUG.
```python
class ProgressReporter:
    def __init__(self, label, total, every, min_interval=300.0, logger=..., clock=time.monotonic)
    def update(self, ok: bool, n: int = 1) -> None    # thread-safe; INFO every `every` items or `min_interval` s
    def finish(self) -> None
```
Line: `download_sra: 150/2000 done (148 ok, 2 failed), 3.1/min, about 9 h 57 min left`; `every=0`
restores per-item INFO lines. Wired in the main-thread `on_result` path, `download_metadata`, and
`extract_target_reads`.
Tests (`tests/test_logging_setup.py`, `tests/test_progress.py`): idempotent setup, foreign handler
survives, file appended across runs, PID and host in the file line, `--quiet` hides INFO on console but
not in the file, traceback only in the file, both flag placements parse, `-q -v` is usage error 2;
reporter lines at N, 2N and finish with a fake clock, `min_interval`, 8 threads x 1000 updates,
`download_sra` INFO count at most `total/every + fixed` with per-accession lines at DEBUG.

## Task 17: tool table, `require_tools`, and `metaquest doctor`

Files: new `metaquest/utils/tools.py`, new `metaquest/cli/commands/doctor.py`, new
`metaquest/processing/doctor_report.py`, `cli/main.py` (register; group "Environment" before "Other"),
`Makefile` (`doctor` target), callers `sra.py:648`, `cli/commands/read_extraction.py:491-504`,
`data/genome_download.py:96`, `read_extraction._refuse_old_minimap2` (kept as fallback),
`accession.fasterq_dump_version` and `read_extraction.megahit_version` (thin wrappers), new
`tests/helpers_tools.py` (`fake_tool(tmp_path, name, stdout, stderr, rc)` on PATH).

```python
@dataclass(frozen=True)
class ToolSpec: name; conda_package; min_version: Optional[str]; used_by: Tuple[str, ...]; optional: bool = False
                version_args: Tuple[str, ...] = ("--version",)
TOOLS  # fasterq-dump, prefetch (sra-tools 3.0), minimap2 2.17, samtools 1.10 (coverage subcommand), megahit 1.2.9,
       # pigz, datasets (ncbi-datasets-cli), seqkit optional (check whether `seqkit version` is needed; add to safe params)
def parse_version(text) -> Optional[Tuple[int, ...]]; def probe_tool(name, timeout=30.0) -> ToolStatus
def require_tools(names, check_versions=True) -> None    # ConfigurationError listing every problem with a conda hint
```
Doctor checks (`Check(name, status ok|warn|fail, detail, data)`): Python and metaquest version; every
tool with version and floor (missing is `warn` unless `--for COMMAND` needs it, then `fail`); config
parses and effective settings with sources; store root reachable, marker, writable, free space; free space
at project, temp folder and `.sra-cache` (`warn` below `min_free_gb`); CPUs available vs total, cgroup
memory limit, SLURM variables; nearest registry loadable; `--network` optional (NCBI einfo and
Branchwater, 10 s). Exit 0 or 3; `--json` one document.
Tests (`tests/test_tools.py`, `tests/test_cli_doctor.py`): version parse table (`2.28-r1209`,
`samtools 1.21`, `MEGAHIT v1.2.9`, `fasterq-dump : 3.1.1`, `datasets version: 16.27.0`); minimap2 2.16
refused with exit 3 from `extract_target_reads`; missing tool gives 3 from `download_sra` and
`genome_download`; doctor with fake PATH tools, `--json` parses, `--for` fails on a missing tool, low disk
faked via `shutil.disk_usage` warns, malformed config fails, network skipped without `--network`.

## Task 18: configurable subprocess timeout

Files: `utils/security.py` (`run_secure(..., timeout=_DEFAULT)`; `_DEFAULT` resolves to
`settings.active().subprocess_timeout`; 0 or None means `communicate(timeout=None)`, fixing the
`timeout or MAX` bug at line 338), `core/constants.py` (`DEFAULT_SUBPROCESS_TIMEOUT = 0`, old name kept
as alias, `VERSION_PROBE_TIMEOUT = 30`), `sra.py` and `cli/commands/read_extraction.py` (`--timeout`,
`default=None`, help "seconds before an external tool is stopped; 0 (the default) means no limit").
Message: "Command timed out after N s (set --timeout or METAQUEST_TIMEOUT; 0 disables)", still a
`SecurityError` (classified by its text at `accession.py:420`). Version probes pass 30 s explicitly.
Tests: fake sleeping tool with `--timeout 1` raises and classifies; `timeout=0` unlimited; setting
default honoured; probe uses 30 s. CHANGELOG "Changed" (default was 1 h).

## Task 19: free-space guard, CPU and memory detection, megahit memory

Files: new `metaquest/data/sra/space.py`, new `metaquest/utils/resources.py`, `data/sra/retry.py`
(`_instrumented(worker, guard, timings)` wraps the worker once so both passes use it; first pass aborts
remaining futures on the first disk-full result and marks them "disk-full: not attempted"),
`data/sra/download.py` (keyword arguments), `cli/commands/sra.py` (`--min-free-gb`, run sizes from
`MetadataBlock.run_size` via `_registry_inputs`; help text for `--max-workers`), `download.py:21-31`
(`available_cpus()` and the `max_workers_cap` setting), `store/stats.py:165`
(`min(DEFAULT_NUM_THREADS, available_cpus())`), new `metaquest/data/assembly.py` (Task 20) gets
`--assembly-memory`.

```python
FASTQ_EXPANSION = 5   # calibrate on 3-5 crispatus runs (.sra size, dump temp peak, FASTQ size); state numbers in CHANGELOG
class SpaceGuard:
    def __init__(self, locations: Mapping[str, Path], floor_bytes: int, run_sizes: Mapping[str, int], use_prefetch: bool)
    def reserve(self, accession) -> Optional[str]; def release(self, accession) -> None
    def preflight(self, accessions) -> List[str]
def available_cpus() -> int             # sched_getaffinity, else SLURM_CPUS_PER_TASK, else os.cpu_count() or 1
def memory_limit_bytes() -> Optional[int]   # cgroup v2 memory.max walked to root, v1 limit_in_bytes, SLURM_MEM_PER_NODE, else None
def parse_memory(value: str, limit: Optional[int]) -> Optional[int]   # "auto" | fraction | "32G"/"32000M"/bytes
```
Needs grouped by filesystem (`st_dev`); in-flight reservations subtracted under a lock; requirement is
the estimate when the run size is known else `floor_bytes`; `--min-free-gb 0` disables; unreadable
`disk_usage` means "has room" (same as adopt). `--assembly-memory auto` passes `--memory` as
`0.9 * memory_limit_bytes()` when a limit is detected, else omitted; a fraction passes through; keep the
worker cap at 4 (downloads are network-bound) but compute from `available_cpus()` and say so in help.
Tests (`tests/test_sra_space.py`, `tests/test_resources.py`): fake `disk_usage` per device; two workers
reserving concurrently; unknown-size floor; first-pass disk-full abort; guard off; unmeasurable fs
proceeds; fake `/proc` and `/sys` trees with injectable root (v2 nested minimum, v1, "max", SLURM,
macOS None); `parse_memory` table; megahit argv contains `--memory` only when resolved.

## Task 20: assembly split and timing fields

Files: new `metaquest/data/assembly.py` (`_megahit_args`, `assemble_reads`, `megahit_version`,
`resolve_assembly_threads` move out of `read_extraction.py`, re-exported; lower its ceiling),
`data/registry_blocks.py` (`started`, `seconds` optional on `DownloadBlock`, `ExtractionBlock`,
`AssemblyBlock`; omitted when None so existing registries round-trip byte-identical), new
`metaquest/data/registry_timing.py` (`set_download_timing`, `set_extraction_timing`,
`set_assembly_timing`; registry.py unchanged), `retry.py` (`timings` dict filled by `_instrumented`),
`cli/commands/sra.py` (`_result_recorder` calls `set_download_timing` after `record_download`; a
store-linked dataset gets None which clears a stale value; `--report-file` gains a trailing `seconds`
column), `cli/commands/read_extraction.py` (`ExtractionResult.seconds`, timed around
`_extract_one_sample`), `processing/results.py` (`download_seconds`, `extraction_seconds`,
`assembly_seconds` appended to `RESULTS_COLUMNS`), `processing/status_report.py` (`"timing"` key:
counts, totals, medians) and `status/render_text.py` (one line), `status --export-tsv` columns.
Tests: block without timing round-trips byte-identical; recorder writes seconds and clears them for a
link; report CSV header and row; results columns (update exact-column assertions); status timing on a
fixture; size guard passes.

## Task 21: XML hardening, HTTP retries, dead constants, tool floors, PyPI job

Files: new `metaquest/utils/xml.py` (`SAFE_PARSER = etree.XMLParser(resolve_entities=False,
no_network=True, load_dtd=False, huge_tree=False)`, `parse_xml_file`), `data/metadata.py:603`, new
`metaquest/utils/http.py` (`retrying_session()` moved from `taxonomy._build_retrying_session`,
re-exported), `data/sra_metadata.py` (one session in `__init__`; `NetworkError` after retries),
`core/constants.py` (remove `DEFAULT_MEMORY_LIMIT_GB`, `MAX_FILE_SIZE_MB`, `DEFAULT_PLUGIN_TIMEOUT`,
`ERROR_MESSAGES`, `SUCCESS_MESSAGES`; use `LOG_LEVELS`/`DEFAULT_LOG_LEVEL` in `main.py`),
`environment.yml` (`samtools>=1.10`, `megahit>=1.2.9`, `pigz>=2.4`, `ncbi-datasets-cli>=16`, seqkit
`>=2.0` still commented), new `tests/test_environment_pins.py` (regex parse matches `TOOLS`),
`.github/workflows/release.yml` (build job uploads `dist/`; `pypi` job `if: vars.PYPI_PUBLISH == 'true'`,
`environment: pypi`, `id-token: write`, `pypa/gh-action-pypi-publish@release/v1`), new
`docs/packaging.md` (bioconda `meta.yaml` template and order: PyPI first, then bioconda; no recipe folder
in this repo).
Stdlib `ET.fromstring` sites stay (expat guards in 3.12) with a comment and a billion-laughs test.
To check online before merge: `ncbi-datasets-cli` 16 supports `download genome accession --inputfile
--no-progressbar`; bioconda megahit on osx-arm64; the name `metaquest` on PyPI (else distribution name
`metaquest-bio`, import name unchanged).
Tests: external and internal entities not expanded, billion-laughs rejected quickly, `sra_metadata`
retries on 503 with `tests/fake_http.py` then raises `NetworkError` and the CLI exits 4, grep test for
removed constants.

## Task 22: documentation and release 0.7.0

Files: new `docs/hpc.md` (pre-flight `metaquest doctor --for download_sra`; array template with
`split -n l/$N`, `#SBATCH --array`, `--cpus-per-task=8 --mem=16G --time=24:00:00 --signal=B:TERM@300`,
`exec metaquest --log-file logs/dl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log download_sra
--accessions-file shard --num-threads 4 --max-workers 2 --temp-folder "$TMPDIR" --timeout 0 --lock-wait
3600 --report-file ...`; signals and walltime, what is flushed, resume by rerun; exit codes and
`--dependency=afterok` with a resubmit loop on 4; shared store on a group filesystem, `METAQUEST_DATA`,
dataset locks; NFS caveats (O_EXCL on NFSv4 and recent NFSv3, clock skew and the stale window, no
`--temp-folder` on NFS, one log per process); `--assembly-memory auto`; one extraction job per genome),
new `docs/configuration.md` (`[runtime]` table: type, default, env, flag; `[store] data_root`; both
precedence lists and why they differ; doctor shows sources), `README.md` ("Running on a cluster",
exit-code table, Logging, doctor, external tools with the same floors as `TOOLS`),
`docs/ARCHITECTURE.md` (replace the generic configuration section with `core/settings.py` and
`store/resolve.py`; add `utils/tools.py`, `utils/resources.py`, exit-code and logging policy),
`CLAUDE.md` (drop stale "0%"/"88%+" coverage claims; rules: `self.fail()`, per-item logging at DEBUG,
new settings in `SETTINGS`), `setup.cfg` comment, `pyproject.toml` and `metaquest/__init__.py` 0.7.0,
`CHANGELOG.md` (Added, Changed, Fixed incl. megahit OOM under cgroups and first-pass disk-full abort,
Security, Documentation, Upgrade notes: exit 3/4, timeout default, grep patterns for per-accession lines
need DEBUG). `make check`, `make test`, `make pipeline`, `python -m build`, no tag.

Order inside Phase 2: 14; 15 and 16 in parallel; 17; 18; 20 (split first); 19; 20 (timing); 16
(progress wiring in read_extraction); 21; 22.

---

## Verification

Per task: `conda run -n metaquest make check` and the affected tests; per phase end: `make check`,
`make test` (all markers, `METAQUEST_PERF_SCALE=1` locally), `make pipeline`, `python -m build`; CI matrix
(ubuntu/macos x 3.12/3.13) green on push; nightly smoke green.

Real data (sekvens2 crispatus project, env `metaquest`, `PYTHONPATH` to the worktree, scratch registry
copies only):
- Phase 1: two `download_sra` processes on a scratch copy of the project with an all-present list plus
  one small run (`--blacklist` for the rest), started 1 s apart: one downloads, one reports "already
  exists", registry consistent, no `.tmp` or lock left; `blacklist` and `select_datasets` run during a
  `download_sra` with `--max-downloads 1` of a small run: all three outcomes present afterwards;
  `kill -TERM` during a real fasterq-dump: exit 130, no orphan `fasterq-dump`, outcomes flushed; `status
  --json`, `results_table` and `parse_containment` byte-identical to 0.5.1 apart from the registry path;
  `store_gc` report unchanged on the real store.
- Phase 2: `metaquest doctor --json` in the env lists every tool with a version; `download_sra --timeout
  0` on the 11.3 M-spot run completes; `--min-free-gb 100000` refuses with the disk-full message before
  any tool starts; `extract_target_reads --assembly-memory 4G --force` on SRR23946447 (APFS copy)
  records `seconds` and `--memory` appears in the megahit argv (`--debug-keep-sam` log);
  `--log-file` receives PID, host and the traceback of a forced failure; `-q` prints only warnings;
  exit codes: missing tool 3, held registry lock 4.

## Not in scope

`filelock`/`flock`, WAL, registry version stamp, parallel extraction inside one process, cross-process
NCBI pacing, log rotation or JSON logs, cross-process space reservation, SLURM submission from metaquest
or workflow wrappers, container images (after bioconda), pandas streaming for tables (revisit above
about 10^6 rows), exit code 5, a `config set` command, removal of compatibility names (`STOP`,
`clear_stopping`, `_termination_raises_interrupt`, `_acquire_lock`, `LOCK_*_SECONDS`).
