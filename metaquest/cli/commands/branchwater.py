"""
Branchwater-related CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from pathlib import Path

from metaquest.core.exceptions import MetaQuestError
from metaquest.data.branchwater import (
    process_branchwater_files,
    extract_metadata_from_branchwater,
)


class UseBranchwaterCommand(BaseCommand):
    """Command for processing pre-downloaded Branchwater files."""

    @property
    def name(self) -> str:
        return "use_branchwater"

    @property
    def help(self) -> str:
        return "Process pre-downloaded Branchwater files"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--branchwater-folder",
            required=True,
            help="Folder containing Branchwater output files",
        )
        parser.add_argument(
            "--matches-folder",
            default="matches",
            help="Folder to save processed matches",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            process_branchwater_files(args.branchwater_folder, args.matches_folder)
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error processing Branchwater files: {e}")
            return 1


class ExtractBranchwaterMetadataCommand(BaseCommand):
    """Command for extracting metadata from Branchwater files."""

    @property
    def name(self) -> str:
        return "extract_branchwater_metadata"

    @property
    def help(self) -> str:
        return "Extract metadata from Branchwater files"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--branchwater-folder",
            required=True,
            help="Folder containing Branchwater output files",
        )
        parser.add_argument(
            "--metadata-folder",
            default="metadata",
            help="Folder to save extracted metadata",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            metadata_folder = Path(args.metadata_folder)
            metadata_folder.mkdir(exist_ok=True)
            output_file = metadata_folder / "branchwater_metadata.txt"

            extract_metadata_from_branchwater(args.branchwater_folder, output_file)
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return 1
