"""
The per-accession lock that guards one dataset's folder in the shared store.

A store download or adoption owns ``<store>/sra/<ACC>`` for as long as the transfer takes,
which for a large run is minutes to hours. The lock is ``<store>/locks/<ACC>.lock``, taken
through ``metaquest.utils.lockfile.held_lock``: the holder's heartbeat refreshes the file's
mtime every ``LOCK_HEARTBEAT_SECONDS``, so a waiter waits for as long as the holder stays
alive, with no overall timeout (giving up on a legitimate multi-hour download would be the
bug, not the wait). A lock whose heartbeat stopped more than ``DATASET_LOCK_STALE_SECONDS``
ago, or whose holder is known to have exited on this host, is taken over with a warning. A
caller that would rather not wait passes ``wait_seconds``; the resulting error names the
accession and the holder.

The limits are read from this module at call time, so tests can shrink them.
"""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from metaquest.core.constants import DATASET_LOCK_STALE_SECONDS, LOCK_HEARTBEAT_SECONDS
from metaquest.store.layout import StorePaths, lock_path
from metaquest.utils.lockfile import (
    LOCK_WAIT_LOG_SECONDS,
    LockPolicy,
    LockWaitStopped,
    describe_holder,
    held_lock,
    holder_is_dead,
    read_holder,
)

# How often a waiter re-checks a lock it is waiting for.
LOCK_POLL_SECONDS = 1.0

__all__ = [
    "DATASET_LOCK_STALE_SECONDS",
    "LOCK_HEARTBEAT_SECONDS",
    "LOCK_WAIT_LOG_SECONDS",
    "LockWaitStopped",
    "dataset_lock",
    "lock_holder",
    "lock_is_held",
    "read_holder",
]


def _dataset_policy(accession: str, wait_seconds: float) -> LockPolicy:
    """The dataset lock's policy, built from this module's current limits."""
    return LockPolicy(
        what=accession,
        stale_seconds=DATASET_LOCK_STALE_SECONDS,
        wait_seconds=wait_seconds,
        poll_seconds=LOCK_POLL_SECONDS,
        heartbeat_seconds=LOCK_HEARTBEAT_SECONDS,
    )


@contextmanager
def dataset_lock(
    paths: StorePaths,
    accession: str,
    wait_seconds: float = 0.0,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Iterator[Path]:
    """Hold ``<store>/locks/<ACC>.lock`` for the length of the block.

    ``wait_seconds`` of zero (the default) waits for as long as the current holder stays
    alive, since a download legitimately runs for hours; a positive value gives up after
    that many seconds with a ``DataAccessError`` naming the accession and the holder.
    ``should_stop``, when given, ends a wait for a held lock with ``LockWaitStopped`` as
    soon as it returns True; a free lock is taken regardless.
    """
    lock = lock_path(paths, accession)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with held_lock(lock, _dataset_policy(accession, wait_seconds), should_stop=should_stop) as held:
        yield held


def lock_is_held(paths: StorePaths, accession: str) -> bool:
    """True when another run is working on ``accession`` right now.

    A lock file whose heartbeat stopped more than ``DATASET_LOCK_STALE_SECONDS`` ago, or
    whose holder has exited on this host, does not count: callers that only inspect (garbage
    collection, adoption) must not treat one as a live download for ever.
    """
    lock = lock_path(paths, accession)
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return False
    return age <= DATASET_LOCK_STALE_SECONDS and not holder_is_dead(read_holder(lock))


def lock_holder(paths: StorePaths, accession: str) -> str:
    """One-line description of whoever holds ``accession``'s lock, for a message."""
    return describe_holder(read_holder(lock_path(paths, accession)))
