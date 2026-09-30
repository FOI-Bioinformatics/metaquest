"""
One lock-file mechanism for every lock MetaQuest takes.

A lock is a file created with ``O_EXCL``, which works on local disks, exFAT, NFS and SMB
alike. The file holds a JSON holder record ``{pid, host, started, token, pidns}``: ``token``
is a random value that identifies this one acquisition, and ``pidns`` (Linux only) is the
inode of the holder's pid namespace, so a pid is only interpreted inside the namespace it
belongs to. Each lock is taken under a ``LockPolicy`` that names what it protects and sets
its stale threshold, wait limit, poll interval and heartbeat interval.

While a lock is held, one daemon thread per process refreshes the file's mtime every
``heartbeat_seconds``, so the file's age measures how long ago the holder was last alive,
not how long it has been working. A waiter removes a lock only when that age exceeds
``stale_seconds`` or when the holder is known to be dead (same host, same pid namespace,
and no process with its pid). The removal happens under a second ``O_EXCL`` file,
``<lock>.reclaim``, and only if the lock file is still the one the waiter judged, so two
waiters can never both remove a lock and one of them remove the other's new lock.

Release compares the token and never removes a lock that now belongs to someone else. A
holder record that is not JSON (a bare pid written by an older version, or a test fixture)
is never judged dead; it follows the age rule alone.
"""

import json
import logging
import os
import secrets
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)

# A waiter says it is waiting once the wait passes FIRST_WAIT_LOG_SECONDS (sub-second
# contention stays quiet), then every LOCK_WAIT_LOG_SECONDS.
FIRST_WAIT_LOG_SECONDS = 1.0
LOCK_WAIT_LOG_SECONDS = 30.0
# A reclaim guard is held for a few file operations; one this old was left by a crash.
RECLAIM_GUARD_STALE_SECONDS = 60.0

__all__ = [
    "LockHeld",
    "LockLost",
    "LockPolicy",
    "LockReentry",
    "LockWaitStopped",
    "describe_holder",
    "held_lock",
    "holder_is_dead",
    "read_holder",
    "verify_held",
]


class LockHeld(DataAccessError):
    """Raised when a lock is taken by another holder and the caller will not wait any longer."""


class LockLost(DataAccessError):
    """Raised by ``verify_held`` when a lock this process took is no longer its own."""


class LockReentry(DataAccessError):
    """Raised when a thread asks again for a lock it already holds, which would never be granted."""


class LockWaitStopped(DataAccessError):
    """Raised when a caller's stop predicate ends a wait for a held lock (for example on Ctrl-C)."""


@dataclass(frozen=True)
class LockPolicy:
    """How one kind of lock is waited for, judged stale and kept alive.

    ``what`` names the protected resource in messages. ``wait_seconds`` of zero waits for as
    long as the holder stays alive; a positive value gives up with ``LockHeld`` after that
    long. A lock whose mtime is older than ``stale_seconds`` is taken over.
    """

    what: str
    stale_seconds: float
    wait_seconds: float = 0.0
    poll_seconds: float = 1.0
    heartbeat_seconds: float = 10.0


# ----------------------------------------------------------------- holder record


def _pid_namespace() -> Optional[int]:
    """Inode of this process's pid namespace on Linux, or None where there is none to read."""
    try:
        return os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        return None


def read_holder(lock: Path) -> Dict[str, Any]:
    """The holder record inside ``lock``; ``{"pid": N}`` for a bare pid, ``{}`` when unreadable.

    A lock file caught between creation and its first write, or written in a form this
    version does not know, reads as an empty record: callers then report an unknown holder.
    """
    try:
        data = json.loads(Path(lock).read_text())
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, int) and not isinstance(data, bool):
        return {"pid": data}
    return {}


def describe_holder(holder: Dict[str, Any]) -> str:
    """One-line description of a lock's holder, for a log line or an error message."""
    return (
        f"pid {holder.get('pid', 'unknown')} on host {holder.get('host', 'unknown')} "
        f"since {holder.get('started', 'an unknown time')}"
    )


def holder_is_dead(holder: Dict[str, Any]) -> bool:
    """True only when the holder's process is known to have exited.

    That needs the same host and the same pid namespace as this process and a pid that no
    process has (``ProcessLookupError``). A process of another user (``PermissionError``),
    another host, or a record without a host is treated as alive.
    """
    pid = holder.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if holder.get("host") != socket.gethostname() or holder.get("pidns") != _pid_namespace():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _create(lock: Path) -> Optional[Dict[str, Any]]:
    """Create ``lock`` exclusively and return the holder record written, or None if it exists."""
    holder: Dict[str, Any] = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(),
        "token": secrets.token_hex(8),
    }
    pidns = _pid_namespace()
    if pidns is not None:
        holder["pidns"] = pidns
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    except OSError as e:
        raise DataAccessError(f"Cannot create the lock file {lock}: {e}") from e
    try:
        os.write(handle, json.dumps(holder).encode())
    except OSError as e:
        os.close(handle)
        Path(lock).unlink(missing_ok=True)
        raise DataAccessError(f"Cannot write the lock file {lock}: {e}") from e
    os.close(handle)
    return holder


# ----------------------------------------------------------------------- reclaim


@dataclass(frozen=True)
class _Observation:
    """One reading of a lock file: its holder, inode and age in seconds."""

    holder: Dict[str, Any]
    inode: int
    age: float


def _observe(lock: Path) -> Optional[_Observation]:
    """Read ``lock``'s holder, inode and age, or None when the file no longer exists."""
    try:
        status = os.stat(lock)
    except FileNotFoundError:
        return None
    return _Observation(read_holder(lock), status.st_ino, time.time() - status.st_mtime)


def _judged_stale(age: float, policy: LockPolicy) -> bool:
    """Whether a lock whose heartbeat is ``age`` seconds old has outlived ``policy``."""
    return age > policy.stale_seconds


def _take_guard(guard: Path) -> bool:
    """Create the reclaim guard exclusively; remove one left by a crash and try once more."""
    for _ in range(2):
        try:
            os.close(os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            return True
        except FileExistsError:
            try:
                age = time.time() - guard.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= RECLAIM_GUARD_STALE_SECONDS:
                return False
            logger.warning("Removing the reclaim guard %s (%.0f s old)", guard, age)
            guard.unlink(missing_ok=True)
        except OSError as e:
            raise DataAccessError(f"Cannot create the reclaim guard {guard}: {e}") from e
    return False


def _reclaim(lock: Path, observed: _Observation, policy: LockPolicy) -> bool:
    """Remove ``lock`` if it is still the file ``observed`` and is stale or has a dead holder.

    Runs under ``<lock>.reclaim``, so only one waiter decides at a time, and re-reads the
    lock under it: a lock replaced since the observation (another waiter reclaimed it and
    took it) is left alone. Returns True when this call removed the lock.
    """
    guard = lock.with_name(lock.name + ".reclaim")
    if not _take_guard(guard):
        return False
    try:
        current = _observe(lock)
        if current is None or current.inode != observed.inode or current.holder != observed.holder:
            return False
        if holder_is_dead(current.holder):
            logger.warning(
                "Taking over the lock on %s: its holder is no longer running (%s)",
                policy.what,
                describe_holder(current.holder),
            )
        elif _judged_stale(current.age, policy):
            logger.warning(
                "Removing the stale lock on %s (no heartbeat for %.0f s): held by %s",
                policy.what,
                current.age,
                describe_holder(current.holder),
            )
        else:
            return False
        lock.unlink(missing_ok=True)
        return True
    finally:
        guard.unlink(missing_ok=True)


# --------------------------------------------------------------------- heartbeat


@dataclass
class _Held:
    """One lock held by this process: its file, token and heartbeat state."""

    path: Path
    token: str
    interval: float
    refreshed: float = field(default_factory=time.monotonic)
    lost: bool = False


def _refresh(entry: _Held) -> None:
    """Touch ``entry``'s lock file, or mark it lost when it is gone or holds another token."""
    entry.refreshed = time.monotonic()
    if read_holder(entry.path).get("token") != entry.token:
        if not entry.lost:
            logger.warning("Lost the lock %s: it was removed or taken over", entry.path)
        entry.lost = True
        return
    try:
        os.utime(entry.path, None)
    except FileNotFoundError:
        entry.lost = True
    except OSError as e:
        logger.warning("Cannot refresh the lock %s: %s", entry.path, e)


class _Heartbeat:
    """The one daemon thread per process that refreshes every held lock's mtime."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.held: Dict[str, _Held] = {}
        self.thread: Optional[threading.Thread] = None

    def add(self, key: str, entry: _Held) -> None:
        with self.cond:
            self.held[key] = entry
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="metaquest-lock-heartbeat", daemon=True)
                self.thread.start()
            self.cond.notify_all()

    def remove(self, key: str) -> None:
        with self.cond:
            self.held.pop(key, None)

    def get(self, key: str) -> Optional[_Held]:
        with self.cond:
            return self.held.get(key)

    def held_count(self) -> int:
        with self.cond:
            return len(self.held)

    def _due(self) -> Tuple[List[_Held], Optional[float]]:
        """Entries due for a refresh, and how long to sleep when none is (None: no lock held)."""
        now = time.monotonic()
        live = [entry for entry in self.held.values() if not entry.lost]
        due = [entry for entry in live if now - entry.refreshed >= entry.interval]
        if due or not live:
            return due, None
        return due, min(entry.refreshed + entry.interval for entry in live) - now

    def _run(self) -> None:
        while True:
            with self.cond:
                due, sleep = self._due()
                if not due:
                    self.cond.wait(timeout=sleep)
                    continue
            # File I/O happens outside the condition, so a slow network file system never
            # blocks another thread taking or releasing a lock.
            for entry in due:
                _refresh(entry)


_HEARTBEAT = _Heartbeat()
_THREAD_STATE = threading.local()


def _thread_held() -> Set[str]:
    """Paths of the locks the calling thread holds right now."""
    held: Optional[Set[str]] = getattr(_THREAD_STATE, "held", None)
    if held is None:
        held = set()
        _THREAD_STATE.held = held
    return held


def _after_fork_in_child() -> None:
    """A forked child holds none of its parent's locks and has no heartbeat thread."""
    global _HEARTBEAT, _THREAD_STATE
    _HEARTBEAT = _Heartbeat()
    _THREAD_STATE = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


# -------------------------------------------------------------- acquire, release


def _held_error(lock: Path, policy: LockPolicy) -> LockHeld:
    return LockHeld(f"{policy.what} is locked by {describe_holder(read_holder(lock))}: {lock}")


def _check_stop(policy: LockPolicy, should_stop: Optional[Callable[[], bool]], waited: float) -> None:
    if should_stop is not None and should_stop():
        raise LockWaitStopped(f"Stopped waiting for {policy.what} after {waited:.0f} s: a stop was requested")


def _acquire(
    lock: Path, key: str, policy: LockPolicy, should_stop: Optional[Callable[[], bool]], blocking: bool
) -> Dict[str, Any]:
    """Take ``lock`` under ``policy`` and return the holder record written.

    The wait limit is checked before a reclaim, so a lock truly held by another process is
    reported, not taken, when both thresholds pass on the same poll.
    """
    started = time.monotonic()
    next_log = FIRST_WAIT_LOG_SECONDS
    while True:
        holder = _create(lock)
        if holder is not None:
            return holder
        waited = time.monotonic() - started
        if key in _thread_held():
            # A stop the caller asked for is what it gets; otherwise the wait could never end.
            _check_stop(policy, should_stop, waited)
            raise LockReentry(f"{policy.what} is already locked by this thread: {lock}")
        if blocking and policy.wait_seconds and waited >= policy.wait_seconds:
            raise _held_error(lock, policy)
        observed = _observe(lock)
        if observed is None:
            continue
        if _judged_stale(observed.age, policy) or holder_is_dead(observed.holder):
            if _reclaim(lock, observed, policy):
                continue
        if not blocking:
            raise _held_error(lock, policy)
        _check_stop(policy, should_stop, waited)
        if waited >= next_log:
            logger.info("waiting for %s: held by %s", policy.what, describe_holder(observed.holder))
            next_log = waited + LOCK_WAIT_LOG_SECONDS
        time.sleep(policy.poll_seconds)


def _release(lock: Path, entry: _Held) -> None:
    """Remove ``lock`` only while it still carries this acquisition's token."""
    current = read_holder(lock)
    if current.get("token") == entry.token:
        lock.unlink(missing_ok=True)
        return
    if lock.exists():
        logger.warning("Not removing %s: it is now held by %s", lock, describe_holder(current))
    else:
        logger.warning("The lock %s was removed while it was held", lock)


@contextmanager
def held_lock(
    lock: Path,
    policy: LockPolicy,
    should_stop: Optional[Callable[[], bool]] = None,
    blocking: bool = True,
) -> Iterator[Path]:
    """Hold ``lock`` for the length of the block, under ``policy``.

    Waits while a live holder keeps the lock, up to ``policy.wait_seconds`` (zero: no
    limit), then raises ``LockHeld``; ``blocking=False`` raises it at once instead, after
    any stale or dead-holder takeover. ``should_stop`` ends a wait with ``LockWaitStopped``
    as soon as it returns True; a free lock is taken regardless. A thread asking for a lock
    it already holds gets ``LockReentry`` at once; other threads contend normally. The
    parent folder must exist.
    """
    lock = Path(lock)
    key = os.path.abspath(lock)
    holder = _acquire(lock, key, policy, should_stop, blocking)
    owner = os.getpid()
    entry = _Held(lock, str(holder["token"]), policy.heartbeat_seconds)
    _thread_held().add(key)
    _HEARTBEAT.add(key, entry)
    try:
        yield lock
    finally:
        # A forked child leaves the block too, but the lock is its parent's to release.
        if os.getpid() == owner:
            _thread_held().discard(key)
            _HEARTBEAT.remove(key)
            _release(lock, entry)


def verify_held(lock: Path) -> None:
    """Raise ``LockLost`` unless this process still holds ``lock`` under its own token."""
    entry = _HEARTBEAT.get(os.path.abspath(lock))
    if entry is None:
        raise LockLost(f"{lock} is not held by this process")
    current = read_holder(Path(lock))
    if current.get("token") != entry.token:
        entry.lost = True
    if entry.lost:
        detail = f"now held by {describe_holder(current)}" if current else "removed"
        raise LockLost(f"Lost the lock {lock}: {detail}")
