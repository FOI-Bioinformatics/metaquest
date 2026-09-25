"""
`store_init`: create (or reuse) a shared data store and record it in the project registry.
"""

import argparse
import uuid
from pathlib import Path
from typing import Any, Dict

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands.store._shared import _now, _gitignore_guard
from metaquest.core.exceptions import DataAccessError, MetaQuestError
from metaquest.data.registry import registry_transaction
from metaquest.store.catalog import catalog_write
from metaquest.store.layout import init_store, read_marker
from metaquest.store.resolve import write_config_data_root


def _refuse_unusable_root(root: Path) -> None:
    """Raise unless ``root`` is either an existing store or a directory safe to make one in.

    ``store_init --data-root ~`` (or any folder already holding unrelated work) would
    otherwise scatter ``sra/``, ``tmp/``, ``locks/``, ``metadata/`` and a marker through it.
    A folder that does not exist yet, an empty one, or one that already carries a store
    marker are all fine.
    """
    if not root.exists() or read_marker(root) is not None:
        return
    if not root.is_dir():
        raise DataAccessError(f"Store root '{root}' is not a directory")
    if any(root.iterdir()):
        raise DataAccessError(
            f"Store root '{root}' is not empty and holds no store marker; "
            "point --data-root at an empty folder or an existing store"
        )


class StoreInitCommand(BaseCommand):
    """Command to initialize a shared data store and bind this project to it."""

    @property
    def name(self) -> str:
        """Return the command name."""
        return "store_init"

    @property
    def help(self) -> str:
        """Return the command's help text."""
        return "Initialize the shared data store and record this project's use of it"

    @property
    def group(self) -> str:
        """Return the pipeline-step group this command is listed under."""
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add the command's arguments."""
        parser.add_argument(
            "--data-root",
            required=True,
            help="Folder to use as the shared data store root (must be empty, or an existing store)",
        )
        parser.add_argument(
            "--project-name",
            default=None,
            help="Name to record for this project (default: the working directory's name)",
        )
        parser.add_argument(
            "--set-default",
            action="store_true",
            help="Also record this root as the user's default store, in the user config file",
        )
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )

    # --------------------------------------------------------------- execute

    def _warn_on_rebinding(self, registry_store: Dict[str, Any], root: Path) -> None:
        """Say so when this project was already bound to a different store root.

        Rebinding leaves the datasets the project links from the old store exactly where they
        are; the links now point outside the store this project records, which is worth one
        line rather than silence.
        """
        previous = (registry_store or {}).get("root")
        if previous and previous != str(root):
            self.logger.warning(
                "This project was bound to the store at %s and is now bound to %s; "
                "datasets it links from the old store are untouched and still linked there",
                previous,
                root,
            )

    def execute(self, args: argparse.Namespace) -> int:
        """Run the command; return the exit code."""
        try:
            root = Path(args.data_root)
            _refuse_unusable_root(root)
            paths = init_store(root)
            # catalog_write migrates the schema itself; opening (and closing) it here is
            # enough to make sure catalog.sqlite exists before anything else touches it.
            with catalog_write(paths):
                pass

            cwd = Path.cwd()
            with registry_transaction(args.registry) as registry:
                existing_project = dict(registry.project) if registry.project else {}
                project_id = existing_project.get("id") or str(uuid.uuid4())
                created = existing_project.get("created") or _now()
                name = args.project_name or cwd.name

                # Keys store_init does not own (e.g. "exports" from results_table) are kept.
                registry.project = {
                    **existing_project,
                    "id": project_id,
                    "name": name,
                    "path": str(cwd.resolve()),
                    "created": created,
                }
                self._warn_on_rebinding(registry.store, root.resolve())
                registry.store = {
                    "root": str(root.resolve()),
                    "mode": "symlink",
                    # Preserved: store_init cannot rebuild the list of datasets this project
                    # links, and resetting it would lose that record silently.
                    "linked": sorted(registry.store.get("linked") or []),
                }
                project_snapshot = dict(registry.project)
                registry_path_str = str(registry.path)

            with catalog_write(paths) as catalog:
                catalog.upsert_project(
                    project_snapshot["id"], project_snapshot["name"], project_snapshot["path"], registry_path_str
                )

            if args.set_default:
                write_config_data_root(root.resolve())
                self.logger.info("Recorded %s as the default store in the user config", root.resolve())

            _gitignore_guard(cwd, self.logger)

            self.logger.info("Store root: %s", root.resolve())
            self.logger.info("Project id: %s (%s)", project_snapshot["id"], project_snapshot["name"])
            return 0
        except MetaQuestError as e:
            self.logger.error("Error initializing store: %s", e)
            return 1
