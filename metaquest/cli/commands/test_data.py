"""
Test data CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands_legacy import download_test_genome_command


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
        return download_test_genome_command(args)
