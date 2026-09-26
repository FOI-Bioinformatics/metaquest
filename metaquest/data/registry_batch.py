"""Batched writes to the project registry.

Every ``registry_transaction`` reads and rewrites the whole registry file. On a project with
some 15,000 datasets that took about one second per transaction, so a command that records
thousands of accessions one transaction at a time ran for hours. ``registry_batch`` queues
mutations and applies them in a few transactions instead. It lives in its own module, next to
``metaquest.data.registry``, which is held at its current size by the module size ceiling.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from metaquest.data.registry import Registry, registry_path, registry_transaction

logger = logging.getLogger(__name__)

# Indirection so tests can advance the clock a batch uses without touching the lock timing.
_monotonic = time.monotonic

RegistryMutation = Callable[[Registry], Any]


class RegistryBatch:
    """Queue registry mutations and apply them in a few transactions instead of one each.

    A batch queues each mutation (a function that changes a loaded ``Registry`` in place) and
    applies the queue, in the order queued, inside one ``registry_transaction``: when
    ``flush_every`` mutations are waiting, when ``flush_seconds`` have passed since the last
    flush (checked as each mutation is queued), on ``flush()``, and when the ``with`` block
    exits, including by an exception or ``KeyboardInterrupt``.

    The registry lock is held only while a flush runs, so other writers (and other batches)
    are not blocked between flushes; because each flush loads the file inside the lock, their
    changes are never reverted. A mutation must therefore not assume the registry it receives
    is the one an earlier flush saw. ``flush_every`` or ``flush_seconds`` set to None disables
    that trigger, so a batch with both None writes once, on exit.

    Functions passed to ``add_flush_hook`` are called with the written registry after each
    flush has released the lock, so work that takes a different lock (the store catalogue)
    never waits while holding this one. ``registry`` holds the registry the last flush wrote,
    or None before the first flush.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        flush_every: Optional[int] = 50,
        flush_seconds: Optional[float] = 30.0,
    ) -> None:
        """Bind the batch to a registry file; nothing is read or written until the first flush."""
        self.path = registry_path(path)
        self.flush_every = flush_every
        self.flush_seconds = flush_seconds
        self.registry: Optional[Registry] = None
        self._queue: List[RegistryMutation] = []
        self._hooks: List[Callable[[Registry], None]] = []
        self._last_flush = _monotonic()

    def __len__(self) -> int:
        """The number of mutations queued and not yet written."""
        return len(self._queue)

    def add_flush_hook(self, hook: Callable[[Registry], None]) -> None:
        """Call ``hook(registry)`` after every flush that wrote something, once the lock is released."""
        self._hooks.append(hook)

    def apply(self, mutation: RegistryMutation) -> None:
        """Queue ``mutation(registry)``; flush first if the size or time limit has been reached."""
        self._queue.append(mutation)
        if self.flush_every is not None and len(self._queue) >= self.flush_every:
            self.flush()
        elif self.flush_seconds is not None and _monotonic() - self._last_flush >= self.flush_seconds:
            self.flush()

    def flush(self) -> Optional[Registry]:
        """Apply every queued mutation in one transaction and return the written registry.

        Returns None, writing nothing, when the queue is empty. The queue is cleared only once
        the transaction has written the file, so a flush that fails (a lock timeout, a disk
        error, a mutation that raises) keeps the queued mutations for a later attempt.
        """
        if not self._queue:
            return None
        pending = list(self._queue)
        with registry_transaction(self.path) as registry:
            for mutation in pending:
                mutation(registry)
        del self._queue[: len(pending)]
        self._last_flush = _monotonic()
        self.registry = registry
        for hook in self._hooks:
            hook(registry)
        return registry

    def __enter__(self) -> "RegistryBatch":
        """Return the batch itself."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Flush what is queued, whether the block finished or raised.

        When the block raised (a ``KeyboardInterrupt`` included), a failing flush is logged
        and the original exception propagates rather than the flush error replacing it.
        """
        if exc_type is None:
            self.flush()
            return
        try:
            self.flush()
        except Exception as e:  # noqa: B902 - the block's own exception must propagate, not this one
            logger.error("Could not write %d queued registry update(s) to %s: %s", len(self._queue), self.path, e)


def registry_batch(
    path: Optional[Union[str, Path]] = None,
    flush_every: Optional[int] = 50,
    flush_seconds: Optional[float] = 30.0,
) -> RegistryBatch:
    """Return a ``RegistryBatch`` for ``path``, used as ``with registry_batch(path) as batch: batch.apply(fn)``.

    Like ``registry_transaction``, a batch must not be flushed while the same process holds
    that registry's lock (inside a ``registry_transaction`` block for the same file), since
    the flush would wait for a lock this process already holds.
    """
    return RegistryBatch(path, flush_every=flush_every, flush_seconds=flush_seconds)
