"""
Command Line Interface entry point for MetaQuest.

This module provides the main CLI entry point with modular command architecture.
"""

import argparse
import logging
import sys
from typing import List, Optional

from metaquest import __version__
from metaquest.cli.base import command_registry
from metaquest.utils.logging import setup_logging

# Import all command modules to register them
from metaquest.cli.commands import (
    UseBranchwaterCommand,
    ExtractBranchwaterMetadataCommand,
    ParseContainmentCommand,
    PlotContainmentCommand,
    DownloadMetadataCommand,
    ParseMetadataCommand,
    CountMetadataCommand,
    PlotMetadataCountsCommand,
    SingleSampleCommand,
    DownloadSraCommand,
    AssembleDatasetsCommand,
    DownloadTestGenomeCommand,
)
from metaquest.cli.commands.advanced_analysis import (
    DiversityAnalysisCommand,
    InteractivePlotCommand,
    TaxonomyValidationCommand,
    TaxonomicSummaryCommand,
)


def register_all_commands() -> None:
    """Register all available commands with the registry."""
    commands = [
        DownloadTestGenomeCommand(),
        UseBranchwaterCommand(),
        ExtractBranchwaterMetadataCommand(),
        ParseContainmentCommand(),
        DownloadMetadataCommand(),
        ParseMetadataCommand(),
        CountMetadataCommand(),
        SingleSampleCommand(),
        PlotContainmentCommand(),
        PlotMetadataCountsCommand(),
        DownloadSraCommand(),
        AssembleDatasetsCommand(),
        DiversityAnalysisCommand(),
        InteractivePlotCommand(),
        TaxonomyValidationCommand(),
        TaxonomicSummaryCommand(),
    ]

    for command in commands:
        command_registry.register(command)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="MetaQuest: A toolkit for analyzing metagenomic datasets based on genome containment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"MetaQuest v{__version__}"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    # Register all commands and setup subparsers
    register_all_commands()
    command_registry.setup_parsers(parser)

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (if None, sys.argv[1:] is used)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Set up logging
    setup_logging(level=getattr(logging, parsed_args.log_level))

    try:
        # Execute the chosen command
        return parsed_args.func(parsed_args)
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
