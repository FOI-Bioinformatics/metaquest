"""
Containment-related CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.branchwater import parse_containment_data
from metaquest.visualization.plots import plot_containment as viz_plot_containment


class ParseContainmentCommand(BaseCommand):
    """Command for parsing containment data from match files."""

    @property
    def name(self) -> str:
        return "parse_containment"

    @property
    def help(self) -> str:
        return "Parse containment data from match files"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--matches-folder",
            default="matches",
            help="Folder containing containment match files",
        )
        parser.add_argument(
            "--parsed-containment-file",
            default="parsed_containment.txt",
            help="File where the parsed containment will be stored",
        )
        parser.add_argument(
            "--summary-containment-file",
            default="top_containments.txt",
            help="File where the containment summary will be stored",
        )
        parser.add_argument(
            "--step-size",
            default=0.1,
            type=float,
            help="Size of steps for the containment thresholds",
        )
        parser.add_argument(
            "--file-format",
            default=None,
            choices=["branchwater"],
            help="Format of the input files",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            parse_containment_data(
                args.matches_folder,
                args.parsed_containment_file,
                args.summary_containment_file,
                args.step_size,
            )
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error parsing containment: {e}")
            return 1


class PlotContainmentCommand(BaseCommand):
    """Command for plotting containment data."""

    @property
    def name(self) -> str:
        return "plot_containment"

    @property
    def help(self) -> str:
        return "Plot containment data"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--file-path", required=True, help="Path to the containment file")
        parser.add_argument("--column", default="max_containment", help="Column to plot")
        parser.add_argument("--title", default=None, help="Title of the plot")
        parser.add_argument("--colors", default=None, help="Colors to use in the plot")
        parser.add_argument("--show-title", action="store_true", help="Whether to display the title")
        parser.add_argument(
            "--save-format",
            default=None,
            choices=["png", "jpg", "pdf", "svg"],
            help="Format to save the plot",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="Minimum value to include in the plot",
        )
        parser.add_argument(
            "--plot-type",
            default="rank",
            choices=["rank", "histogram", "box", "violin"],
            help="Type of plot to generate",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            viz_plot_containment(
                file_path=args.file_path,
                column=args.column,
                title=args.title,
                colors=args.colors,
                show_title=args.show_title,
                save_format=args.save_format,
                threshold=args.threshold,
                plot_type=args.plot_type,
            )
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error plotting containment: {e}")
            return 1
