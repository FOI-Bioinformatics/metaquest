"""
Containment-related CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands_legacy import (
    parse_containment_command,
    plot_containment_command
)


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
            help="Folder containing containment match files"
        )
        parser.add_argument(
            "--parsed-containment-file",
            default="parsed_containment.txt",
            help="File where the parsed containment will be stored"
        )
        parser.add_argument(
            "--summary-containment-file", 
            default="top_containments.txt",
            help="File where the containment summary will be stored"
        )
        parser.add_argument(
            "--step-size",
            default=0.1,
            type=float,
            help="Size of steps for the containment thresholds"
        )
        parser.add_argument(
            "--file-format",
            default=None,
            choices=["branchwater", "mastiff"],
            help="Format of the input files (branchwater or mastiff)"
        )
    
    def execute(self, args: argparse.Namespace) -> int:
        return parse_containment_command(args)


class PlotContainmentCommand(BaseCommand):
    """Command for plotting containment data."""
    
    @property
    def name(self) -> str:
        return "plot_containment"
    
    @property  
    def help(self) -> str:
        return "Plot containment data"
    
    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--file-path", 
            required=True,
            help="Path to the containment file"
        )
        parser.add_argument(
            "--column",
            default="max_containment",
            help="Column to plot"
        )
        parser.add_argument(
            "--title",
            default=None,
            help="Title of the plot"
        )
        parser.add_argument(
            "--colors",
            default=None,
            help="Colors to use in the plot"
        )
        parser.add_argument(
            "--show-title",
            action="store_true",
            help="Whether to display the title"
        )
        parser.add_argument(
            "--save-format",
            default=None,
            choices=["png", "jpg", "pdf", "svg"],
            help="Format to save the plot"
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="Minimum value to include in the plot"
        )
        parser.add_argument(
            "--plot-type",
            default="rank",
            choices=["rank", "histogram", "box", "violin"],
            help="Type of plot to generate"
        )
    
    def execute(self, args: argparse.Namespace) -> int:
        return plot_containment_command(args)