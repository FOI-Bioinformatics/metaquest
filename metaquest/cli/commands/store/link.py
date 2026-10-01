"""
`store_link` and `store_unlink`: add or remove one accession's project symlink.
"""

import argparse
from pathlib import Path
from typing import List

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands.store._shared import _no_store_hint, _sidecar_completeness
from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry_blocks as rb
from metaquest.data.registry import load_registry, record_download, registry_transaction, update_linked
from metaquest.data.registry_timing import set_download_timing
from metaquest.store.layout import StorePaths, sidecar_path, store_paths
from metaquest.store.link import LINK_MODES, link_dataset, unlink_dataset
from metaquest.store.resolve import resolve_store_root
from metaquest.store.sidecar import read_sidecar
from metaquest.store.usage import ensure_project_identity, record_usage_many


class StoreLinkCommand(BaseCommand):
    """Command to link project accessions to the shared store's copies."""

    @property
    def name(self) -> str:
        """Return the command name."""
        return "store_link"

    @property
    def help(self) -> str:
        """Return the command's help text."""
        return "Link project accessions to the shared store's copies (complete datasets only, unless --accept-partial)"

    @property
    def group(self) -> str:
        """Return the pipeline-step group this command is listed under."""
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add the command's arguments."""
        parser.add_argument("accessions", nargs="+", help="Accessions to link from the store")
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--link-mode",
            choices=list(LINK_MODES),
            default="auto",
            help=(
                "How this project points at the store's copy: a relative or absolute symlink, "
                "a copy of the folder, or auto (relative when the store and the project share a "
                "parent folder)"
            ),
        )
        parser.add_argument(
            "--accept-partial",
            dest="accept_partial",
            action="store_true",
            default=False,
            help="Link a dataset whose store copy is incomplete, failed or undescribed",
        )

    def _refuse_incomplete(self, paths: StorePaths, accession: str, accept_partial: bool) -> bool:
        """True when ``accession`` must not be linked as it stands.

        The store's sidecar is the record of whether a dataset is usable. A ``partial`` or
        ``failed`` one, or one with no sidecar at all (an interrupted run, or a folder put
        there by hand), reads as complete once it is linked into ``fastq/``, so linking it
        silently would feed an unfinished dataset into every later analysis.
        """
        sidecar = read_sidecar(sidecar_path(paths, accession))
        if sidecar is None:
            state = "no sidecar"
        elif sidecar.state in ("partial", "failed", "downloading"):
            state = sidecar.state
        else:
            return False

        if accept_partial:
            self.logger.warning("%s: linking a dataset the store records as %s", accession, state)
            return False
        self.logger.error(
            "%s: the store records this dataset as %s; rerun with --accept-partial to link it anyway",
            accession,
            state,
        )
        return True

    def execute(self, args: argparse.Namespace) -> int:
        """Run the command; return the exit code."""
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, rb.store_block(registry).root)
        except DataAccessError as e:
            return self.fail(e, self.name)

        if root is None:
            _no_store_hint()
            return 1

        paths = store_paths(root)
        accept_partial = getattr(args, "accept_partial", False)
        linked: List[str] = []
        failed: List[str] = []
        for accession in args.accessions:
            if self._refuse_incomplete(paths, accession, accept_partial):
                failed.append(accession)
                continue
            try:
                link_dataset(args.fastq_folder, accession, paths, mode=args.link_mode)
                linked.append(accession)
            except DataAccessError as e:
                self.logger.error("%s: %s", accession, e)
                failed.append(accession)

        if linked:
            with registry_transaction(args.registry) as reg:
                ensure_project_identity(reg)
                for accession in linked:
                    complete = _sidecar_completeness(paths, accession)
                    record_download(
                        reg,
                        accession,
                        "downloaded",
                        args.fastq_folder,
                        attempt=False,
                        complete=complete,
                        source="store",
                        store_name=accession,
                    )
                    # Linked, not downloaded: an earlier download time of this project is not kept.
                    set_download_timing(reg, accession, None, None)
                for accession in linked:
                    update_linked(reg, accession, add=True)
                usage_registry = reg
            # Outside the transaction: see the note in store_adopt.
            record_usage_many(paths, usage_registry, [(acc, "", "linked", "store_link") for acc in linked])

        self.emit(f"Linked {len(linked)} of {len(args.accessions)} accession(s)")
        return 1 if failed else 0


class StoreUnlinkCommand(BaseCommand):
    """Command to remove a project's store link, without touching the store's copy."""

    @property
    def name(self) -> str:
        """Return the command name."""
        return "store_unlink"

    @property
    def help(self) -> str:
        """Return the command's help text."""
        return "Remove a project's link to the store (never removes a real directory)"

    @property
    def group(self) -> str:
        """Return the pipeline-step group this command is listed under."""
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add the command's arguments."""
        parser.add_argument("accessions", nargs="+", help="Accessions to unlink")
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Run the command; return the exit code."""
        removed: List[str] = []
        refused: List[str] = []
        for accession in args.accessions:
            link_path = Path(args.fastq_folder) / accession
            if link_path.exists() and not link_path.is_symlink():
                self.logger.error("%s is a real directory, not a store link; refusing to remove it", accession)
                refused.append(accession)
                continue
            try:
                was_removed = unlink_dataset(args.fastq_folder, accession)
            except DataAccessError as e:
                self.logger.error("%s: %s", self.name, e)
                refused.append(accession)
                continue
            if was_removed:
                removed.append(accession)

        if removed:
            with registry_transaction(args.registry) as reg:
                for accession in removed:
                    record_download(reg, accession, "missing", args.fastq_folder, attempt=False)
                    set_download_timing(reg, accession, None, None)
                for accession in removed:
                    update_linked(reg, accession, add=False)

        self.emit(f"Unlinked {len(removed)} of {len(args.accessions)} accession(s)")
        return 1 if refused else 0
