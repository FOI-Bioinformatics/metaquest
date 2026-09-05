"""
Command Line Interface entry point for MetaQuest.

This module provides the main CLI entry point with modular command architecture.
"""

import argparse
import logging
import sys
import traceback
from typing import Dict, List, Optional

from metaquest import __version__
from metaquest.cli.base import BaseCommand, DefaultsHelpFormatter, command_registry
from metaquest.core.exceptions import MetaQuestError
from metaquest.utils.logging import setup_logging

# Import all command modules to register them
from metaquest.cli.commands import (
    UseBranchwaterCommand,
    ExtractBranchwaterMetadataCommand,
    ParseContainmentCommand,
    PlotContainmentCommand,
    DownloadMetadataCommand,
    ParseMetadataCommand,
    CheckMetadataAttributesCommand,
    CountMetadataCommand,
    PlotMetadataCountsCommand,
    SingleSampleCommand,
    DownloadSraCommand,
    AssembleDatasetsCommand,
    StatusCommand,
    ExtractTargetReadsCommand,
    DownloadTestGenomeCommand,
    GenomeSearchCommand,
    GenomeDownloadCommand,
    GenomePrepareCommand,
    EnrichTaxonomyCommand,
    ExploreContainmentCommand,
    FindByTaxonomyCommand,
)
from metaquest.cli.commands.select import SelectDatasetsCommand
from metaquest.cli.commands.branchwater_search import BranchwaterSearchCommand
from metaquest.cli.commands.advanced_analysis import (
    DiversityAnalysisCommand,
    InteractivePlotCommand,
    TaxonomyValidationCommand,
    TaxonomicSummaryCommand,
)
from metaquest.cli.commands.sra_enhanced import (
    SRAInfoCommand,
    SRAStatsCommand,
    SRAValidateCommand,
)
from metaquest.cli.commands.sra_intelligent import (
    SRAQualityProfileCommand,
    SRAInteractiveDashboardCommand,
    SRAComparativeAnalysisCommand,
)


def register_all_commands() -> None:
    """Register all available commands with the registry."""
    commands = [
        DownloadTestGenomeCommand(),
        BranchwaterSearchCommand(),
        UseBranchwaterCommand(),
        ExtractBranchwaterMetadataCommand(),
        ParseContainmentCommand(),
        DownloadMetadataCommand(),
        ParseMetadataCommand(),
        CheckMetadataAttributesCommand(),
        CountMetadataCommand(),
        SingleSampleCommand(),
        PlotContainmentCommand(),
        PlotMetadataCountsCommand(),
        DownloadSraCommand(),
        AssembleDatasetsCommand(),
        StatusCommand(),
        ExtractTargetReadsCommand(),
        SelectDatasetsCommand(),
        # Enhanced SRA commands
        SRAInfoCommand(),
        SRAStatsCommand(),
        SRAValidateCommand(),
        # Intelligent SRA commands
        SRAQualityProfileCommand(),
        SRAInteractiveDashboardCommand(),
        SRAComparativeAnalysisCommand(),
        # Genome commands
        GenomeSearchCommand(),
        GenomeDownloadCommand(),
        GenomePrepareCommand(),
        # Taxonomy exploration commands
        EnrichTaxonomyCommand(),
        ExploreContainmentCommand(),
        FindByTaxonomyCommand(),
        # Advanced analysis commands
        DiversityAnalysisCommand(),
        InteractivePlotCommand(),
        TaxonomyValidationCommand(),
        TaxonomicSummaryCommand(),
    ]

    for command in commands:
        command_registry.register(command)


GROUP_ORDER = ["Containment", "Metadata", "Genomes", "Reads", "Analysis", "Other"]


class _HelpFormatter(DefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show option defaults (when they exist) and keep the epilog's line breaks."""


def _commands_epilog(commands: Dict[str, BaseCommand]) -> str:
    """List commands under their pipeline step for the main --help."""
    by_group: Dict[str, List[BaseCommand]] = {}
    for command in commands.values():
        by_group.setdefault(command.group, []).append(command)
    lines = ["commands by pipeline step:"]
    for group in GROUP_ORDER + sorted(set(by_group) - set(GROUP_ORDER)):
        if group not in by_group:
            continue
        lines.append(f"  {group}:")
        for command in by_group[group]:
            lines.append(f"    {command.name:<28} {command.help}")
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    # Register all commands before building the parser so the epilog can list them.
    register_all_commands()

    parser = argparse.ArgumentParser(
        description="MetaQuest: A toolkit for analyzing metagenomic datasets based on genome containment.",
        epilog=_commands_epilog(command_registry.get_all_commands()),
        formatter_class=_HelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"MetaQuest v{__version__}")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        default="INFO",
        help="Set the logging level (one of DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

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
    except MetaQuestError as e:
        logging.error(f"Error: {e}")
        if parsed_args.log_level == "DEBUG":
            logging.debug(traceback.format_exc())
        else:
            logging.info("Use --log-level DEBUG for full traceback.")
        return 1
    except Exception as e:
        logging.error(f"{type(e).__name__}: {e}")
        if parsed_args.log_level == "DEBUG":
            logging.debug(traceback.format_exc())
        else:
            logging.info("Use --log-level DEBUG for full traceback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
