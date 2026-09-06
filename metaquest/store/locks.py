"""
The per-accession lock that guards one dataset's folder in the shared store.

A store download or adoption owns ``<store>/sra/<ACC>`` for as long as the
transfer takes, which for a large run is minutes to hours. That is far longer
than the registry's own lock (``metaquest.data.registry._acquire_lock``) was
written for: it reclaims any lock file older than half a minute, which would
let a second project delete and rewrite the folder a first one is still
filling.

The lock here is built for that length of stay. The holder writes its pid,
host and start time into ``<store>/locks/<ACC>.lock`` and a daemon thread
refreshes the file's mtime every ``LOCK_HEARTBEAT_SECONDS`` while the lock is
held, so the file's age measures how long ago the holder was last alive rather
than how long it has been working. A waiter therefore waits for as long as the
heartbeat keeps running, with no overall timeout: giving up on a legitimate
multi-hour download would be the bug, not the wait. Only a lock whose mtime is
older than ``DATASET_LOCK_STALE_SECONDS`` (its holder died, or its machine
did) is reclaimed, with a warning. A caller that would rather not wait passes
``wait_seconds``; the resulting error names the accession and the holder.

Registry and catalogue writes are sub-second and keep using their own lock.
"""

import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from metaquest.core.constants import DATASET_LOCK_STALE_SECONDS, LOCK_HEARTBEAT_SECONDS
from metaquest.core.exceptions import DataAccessError
from metaquest.store.layout import StorePaths, lock_path

logger = logging.getLogger(__name__)

# How often a waiter re-checks a lock it is waiting for, and how often it says so.
LOCK_POLL_SECONDS = 1.0
LOCK_WAIT_LOG_SECONDS = 30.0

__all__ = ["dataset_lock", "lock_holder", "lock_is_held"]


def _read_holder(lock: Path) -> Dict[str, Any]:
    """The ``{pid, host, started}`` record inside ``lock``, or ``{}`` when it cannot be read.

    A lock file caught between creation and its first write, or written by a future version,
    reads as an empty record: the caller then reports an unknown holder rather than failing.
    """
    try:
        data = json.loads(lock.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _describe(holder: Dict[str, Any]) -> str:
    """One-line description of a lock's holder, for a log line or an error message."""
    return (
        f"pid {holder.get('pid', 'unknown')} on {holder.get('host', 'unknown host')} "
        f"since {holder.get('started', 'an unknown time')}"
    )


def _lock_age(lock: Path) -> Optional[float]:
    """Seconds since the lock file's last heartbeat, or None when it no longer exists."""
    try:
        return time.time() - lock.stat().st_mtime
    except OSError:
        return None


def _create_lock(lock: Path) -> Optional[Dict[str, Any]]:
    """Create ``lock`` exclusively and return the holder record written, or None if it exists."""
    holder = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    except OSError as e:
        raise DataAccessError(f"Cannot create the lock file {lock}: {e}") from e
    try:
        os.write(handle, json.dumps(holder).encode())
    finally:
        os.close(handle)
    return holder


def _give_up(lock: Path, accession: str, waited: float) -> DataAccessError:
    """The error raised when ``wait_seconds`` runs out, naming accession and holder."""
    return DataAccessError(
        f"Gave up waiting for {accession} after {waited:.0f} s: " f"{lock} is held by {_describe(_read_holder(lock))}"
    )


def _acquire(lock: Path, accession: str, wait_seconds: float) -> Dict[str, Any]:
    """Take ``lock`` for ``accession``, waiting while another live holder keeps it.

    Waits for ever by default; ``wait_seconds`` above zero gives up after that long. A lock
    whose heartbeat stopped more than ``DATASET_LOCK_STALE_SECONDS`` ago is removed and taken
    over, with a warning naming the dead holder.
    """
    started = time.monotonic()
    next_log = 0.0
    while True:
        holder = _create_lock(lock)
        if holder is not None:
            return holder

        age = _lock_age(lock)
        if age is None:
            # Released between the failed create and the age check; try again at once.
            continue
        if age > DATASET_LOCK_STALE_SECONDS:
            logger.warning(
                "Removing the stale lock on %s (no heartbeat for %.0f s): held by %s",
                accession,
                age,
                _describe(_read_holder(lock)),
            )
            lock.unlink(missing_ok=True)
            continue

        waited = time.monotonic() - started
        if wait_seconds and waited >= wait_seconds:
            raise _give_up(lock, accession, waited)
        if waited >= next_log:
            logger.info("waiting for %s: held by %s", accession, _describe(_read_holder(lock)))
            next_log = waited + LOCK_WAIT_LOG_SECONDS
        time.sleep(LOCK_POLL_SECONDS)


class _Heartbeat:
    """Daemon thread refreshing a held lock file's mtime, so waiters see a live holder."""

    def __init__(self, lock: Path) -> None:
        self._lock = lock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"lock-heartbeat-{lock.name}", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(LOCK_HEARTBEAT_SECONDS):
            try:
                os.utime(self._lock, None)
            except OSError as e:
                # The lock is gone (released, or reclaimed as stale): nothing left to refresh.
                logger.debug("Stopped refreshing %s: %s", self._lock, e)
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=LOCK_HEARTBEAT_SECONDS + 5.0)


def _release(lock: Path, holder: Dict[str, Any]) -> None:
    """Remove ``lock``, but only while it still records ``holder`` as its owner.

    A lock reclaimed as stale while we held it (a heartbeat that could not write, a clock
    jump) now belongs to someone else, and removing it would drop that project's protection
    over the same folder.
    """
    current = _read_holder(lock)
    if not lock.exists():
        return
    if current == holder:
        lock.unlink(missing_ok=True)
        return
    logger.warning("Not removing %s: it is now held by %s", lock, _describe(current))


@contextmanager
def dataset_lock(paths: StorePaths, accession: str, wait_seconds: float = 0.0) -> Iterator[Path]:
    """Hold ``<store>/locks/<ACC>.lock`` for the length of the block.

    ``wait_seconds`` of zero (the default) waits for as long as the current holder stays
    alive, since a download legitimately runs for hours; a positive value gives up after
    that many seconds with a ``DataAccessError`` naming the accession and the holder.
    """
    lock = lock_path(paths, accession)
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = _acquire(lock, accession, wait_seconds)
    heartbeat = _Heartbeat(lock)
    heartbeat.start()
    try:
        yield lock
    finally:
        heartbeat.stop()
        _release(lock, holder)


def lock_is_held(paths: StorePaths, accession: str) -> bool:
    """True when another run is working on ``accession`` right now.

    A lock file whose heartbeat stopped more than ``DATASET_LOCK_STALE_SECONDS`` ago belongs
    to a dead holder and does not count: callers that only inspect (garbage collection,
    adoption) must not treat one as a live download for ever.
    """
    age = _lock_age(lock_path(paths, accession))
    return age is not None and age <= DATASET_LOCK_STALE_SECONDS


def lock_holder(paths: StorePaths, accession: str) -> str:
    """One-line description of whoever holds ``accession``'s lock, for a message."""
    return _describe(_read_holder(lock_path(paths, accession)))
