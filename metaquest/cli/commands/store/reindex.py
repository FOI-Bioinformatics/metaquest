"""
`store_reindex`: rebuild the SQLite catalogue from the sidecars on disk and replay the journal.
"""

import argparse
from typing import List, Tuple

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands.store._shared import _now, _no_store_hint
from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry_blocks as rb
from metaquest.data.file_io import visible_files
from metaquest.data.registry import load_registry
from metaquest.store import journal
from metaquest.store.catalog import REBUILT_WITHOUT_PROJECTS, catalog_write
from metaquest.store.layout import StorePaths, sidecar_path, store_paths
from metaquest.store.resolve import resolve_store_root
from metaquest.store.sidecar import Sidecar, read_sidecar


class StoreReindexCommand(BaseCommand):
    """Command to rebuild the store catalogue from every dataset's sidecar file."""

    @property
    def name(self) -> str:
        """Return the command name."""
        return "store_reindex"

    @property
    def help(self) -> str:
        """Return the command's help text."""
        return (
            "Rebuild the store catalogue from every dataset's sidecar file "
            "(refuses to run when any sidecar cannot be read)"
        )

    @property
    def group(self) -> str:
        """Return the pipeline-step group this command is listed under."""
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add the command's arguments."""
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )

    @staticmethod
    def _read_all_sidecars(paths: StorePaths) -> Tuple[List[Sidecar], List[str]]:
        """Every dataset's sidecar, plus the accessions whose sidecar could not be read.

        A reindex rebuilds ``datasets`` from exactly what it reads, so a sidecar missed here
        would look like a dataset that no longer exists. The caller stops rather than acting
        on a partial reading.
        """
        sidecars: List[Sidecar] = []
        unreadable: List[str] = []
        for acc_dir in visible_files(paths.sra, dirs=True):
            sidecar = read_sidecar(sidecar_path(paths, acc_dir.name))
            if sidecar is None:
                unreadable.append(acc_dir.name)
                continue
            sidecars.append(sidecar)
        return sidecars, unreadable

    def execute(self, args: argparse.Namespace) -> int:
        """Run the command; return the exit code."""
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, rb.store_block(registry).root)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        try:
            paths = store_paths(root)
            sidecars, unreadable = self._read_all_sidecars(paths)
            if unreadable:
                self.logger.error(
                    "Not reindexing: %d sidecar(s) could not be read, and rebuilding from a "
                    "partial reading would drop those datasets and their usage history: %s",
                    len(unreadable),
                    ", ".join(unreadable),
                )
                return 1
            with catalog_write(paths) as catalog:
                # Replay first: a fresh or rebuilt catalogue has no projects yet, so usage rows
                # restored here can insert "unknown" placeholder datasets for accessions the
                # journal references. reindex() then removes any placeholder (and real) row
                # whose accession is not among the sidecars just read and whose folder is gone
                # from disk, so a dataset gc already removed is never resurrected by replay.
                projects, usage = journal.replay(paths, catalog)
                count = catalog.reindex(sidecars)
                has_datasets = catalog.conn.execute("SELECT 1 FROM datasets LIMIT 1").fetchone() is not None
                if projects == 0 and has_datasets:
                    catalog.set_meta(REBUILT_WITHOUT_PROJECTS, _now())
                else:
                    # Either at least one project was restored, or the store now holds no
                    # dataset at all (nothing for gc to mistake as unused either way): a flag
                    # set by an earlier, emptier reindex must not survive as stale.
                    catalog.delete_meta(REBUILT_WITHOUT_PROJECTS)
            if projects == 0:
                self._warn_no_projects_restored(paths, has_datasets)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        self.emit(f"Reindexed {count} dataset(s); restored {projects} project(s) and {usage} usage record(s)")
        return 0

    def _warn_no_projects_restored(self, paths: StorePaths, flagged: bool) -> None:
        """Warn that no project was restored. With no datasets there is nothing gc could take for
        unused, so the ``rebuilt_without_projects`` flag is neither set nor cleared in that case."""
        if not flagged:
            self.logger.warning("No project records could be restored from %s", paths.journal)
            return
        self.logger.warning(
            "No project records could be restored from %s; every dataset will look unused until each "
            "project runs store_init or store_link again. store_gc now refuses to run (catalogue flag '%s') "
            "until that is done and it is run once with --accept-rebuilt, or until a later store_reindex "
            "restores at least one project",
            paths.journal,
            REBUILT_WITHOUT_PROJECTS,
        )
