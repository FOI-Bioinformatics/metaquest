"""
Test data CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.processing.containment import download_test_genome


class DownloadTestGenomeCommand(BaseCommand):
    """Command for downloading test genome."""

    @property
    def name(self) -> str:
        return "download_test_genome"

    @property
    def help(self) -> str:
        return "Download test genome"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output-folder",
            default="genomes",
            help="Folder to save the downloaded fasta files",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            download_test_genome(args.output_folder)
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error downloading test genome: {e}")
            return 1
