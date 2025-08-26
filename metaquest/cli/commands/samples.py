"""
Sample analysis CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands_legacy import single_sample_command


class SingleSampleCommand(BaseCommand):
    """Command for analyzing a single sample."""
    
    @property
    def name(self) -> str:
        return "single_sample"
    
    @property
    def help(self) -> str:
        return "Analyze a single sample"
    
    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--summary-file",
            default="parsed_containment.txt",
            help="Path to the summary file"
        )
        parser.add_argument(
            "--metadata-file",
            default="metadata_table.txt",
            help="Path to the metadata file"
        )
        parser.add_argument(
            "--summary-column",
            required=True,
            help="Name of the column in the summary file"
        )
        parser.add_argument(
            "--metadata-column",
            required=True,
            help="Name of the column in the metadata file"
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.1,
            help="Threshold for the column in the summary file"
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=100,
            help="Number of top items to keep"
        )
    
    def execute(self, args: argparse.Namespace) -> int:
        return single_sample_command(args)