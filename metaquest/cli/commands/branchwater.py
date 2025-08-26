"""
Branchwater-related CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands_legacy import (
    process_branchwater_command,
    extract_branchwater_metadata_command,
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
        return process_branchwater_command(args)


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
        return extract_branchwater_metadata_command(args)
