"""
Sample analysis CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.processing.containment import count_single_sample


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
            help="Path to the summary file",
        )
        parser.add_argument(
            "--metadata-file",
            default="metadata_table.txt",
            help="Path to the metadata file",
        )
        parser.add_argument(
            "--summary-column",
            required=True,
            help="Name of the column in the summary file",
        )
        parser.add_argument(
            "--metadata-column",
            required=True,
            help="Name of the column in the metadata file",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.1,
            help="Threshold for the column in the summary file",
        )
        parser.add_argument("--top-n", type=int, default=100, help="Number of top items to keep")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            count_dict = count_single_sample(
                summary_file=args.summary_file,
                metadata_file=args.metadata_file,
                summary_column=args.summary_column,
                metadata_column=args.metadata_column,
                threshold=args.threshold,
                top_n=args.top_n,
            )

            if not count_dict:
                self.logger.warning("No data found above threshold")
                return 0

            # Log the top N items
            for key, value in list(count_dict.items())[: args.top_n]:
                self.logger.info(f"{key}: {value}")

            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error analyzing single sample: {e}")
            return 1
