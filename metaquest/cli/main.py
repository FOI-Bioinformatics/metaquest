"""
Command Line Interface entry point for MetaQuest.

This module provides the main CLI entry point with modular command architecture.
"""

import argparse
import logging
import os
import socket
import sys
from typing import Dict, List, Optional

from metaquest import __version__
from metaquest.cli.base import BaseCommand, DefaultsHelpFormatter, add_global_options, command_registry
from metaquest.core import settings
from metaquest.core.exceptions import ConfigurationError, MetaQuestError, exit_code_for
from metaquest.utils.logging import log_traceback_hint, setup_logging

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
    StatusCommand,
    ResultsTableCommand,
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
from metaquest.cli.commands.blacklist import BlacklistCommand
from metaquest.cli.commands.doctor import STARTUP_ERROR_ATTR, DoctorCommand
from metaquest.cli.commands.branchwater_search import BranchwaterSearchCommand
from metaquest.cli.commands.store import (
    StoreAdoptCommand,
    StoreGcCommand,
    StoreInitCommand,
    StoreLinkCommand,
    StoreReindexCommand,
    StoreStatusCommand,
    StoreUnlinkCommand,
    StoreUsageCommand,
    StoreVerifyCommand,
)
from metaquest.cli.commands.advanced_analysis import (
    DiversityAnalysisCommand,
    InteractivePlotCommand,
    TaxonomyValidationCommand,
    TaxonomicSummaryCommand,
)
from metaquest.cli.commands.renamed import renamed_commands
from metaquest.cli.commands.sra_enhanced import SRAInfoCommand, SRAValidateCommand
from metaquest.cli.commands.sra_profile import SRAProfileCommand
from metaquest.cli.commands.sra_report import SRAReportCommand


def register_all_commands() -> None:
    """Register all available commands with the registry."""
    commands = [
        # Containment commands
        BranchwaterSearchCommand(),
        UseBranchwaterCommand(),
        ParseContainmentCommand(),
        PlotContainmentCommand(),
        ExploreContainmentCommand(),
        EnrichTaxonomyCommand(),
        FindByTaxonomyCommand(),
        # Metadata commands
        ExtractBranchwaterMetadataCommand(),
        DownloadMetadataCommand(),
        ParseMetadataCommand(),
        CheckMetadataAttributesCommand(),
        CountMetadataCommand(),
        SingleSampleCommand(),
        PlotMetadataCountsCommand(),
        # Genome commands
        GenomeSearchCommand(),
        GenomeDownloadCommand(),
        GenomePrepareCommand(),
        DownloadTestGenomeCommand(),
        # Reads commands
        SelectDatasetsCommand(),
        BlacklistCommand(),
        DownloadSraCommand(),
        StatusCommand(),
        ResultsTableCommand(),
        SRAInfoCommand(),
        SRAValidateCommand(),
        SRAProfileCommand(),
        SRAReportCommand(),
        ExtractTargetReadsCommand(),
        # Store commands
        StoreInitCommand(),
        StoreStatusCommand(),
        StoreReindexCommand(),
        StoreAdoptCommand(),
        StoreVerifyCommand(),
        StoreLinkCommand(),
        StoreUnlinkCommand(),
        StoreUsageCommand(),
        StoreGcCommand(),
        # Analysis commands
        DiversityAnalysisCommand(),
        InteractivePlotCommand(),
        TaxonomyValidationCommand(),
        TaxonomicSummaryCommand(),
        # Environment commands
        DoctorCommand(),
        # Former names (0.5.0), hidden from the help
        *renamed_commands(),
    ]

    for command in commands:
        command_registry.register(command)


GROUP_ORDER = ["Containment", "Metadata", "Genomes", "Reads", "Store", "Analysis", "Environment", "Other"]


class _HelpFormatter(DefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show option defaults (when they exist) and keep the epilog's line breaks."""


def _commands_epilog(commands: Dict[str, BaseCommand]) -> str:
    """List commands under their pipeline step for the main --help; hidden commands are left out."""
    by_group: Dict[str, List[BaseCommand]] = {}
    for command in commands.values():
        if command.hidden:
            continue
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

    add_global_options(parser, suppress_defaults=False)

    command_registry.setup_parsers(parser)

    return parser


def _flag(args: argparse.Namespace, name: str) -> bool:
    """True only when ``args`` really holds ``name`` set to True (a mock namespace invents attributes)."""
    try:
        return vars(args).get(name) is True
    except TypeError:
        return False


def _apply_quiet_and_verbose(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Turn ``-q``/``-v`` into ``args.log_level``; together (in either placement) they are a usage error."""
    quiet, verbose = _flag(args, "log_quiet"), _flag(args, "log_verbose")
    if quiet and verbose:
        parser.error("argument -q/--quiet: not allowed with argument -v/--verbose")
    if quiet:
        args.log_level = "WARNING"
    elif verbose:
        args.log_level = "DEBUG"


# Flags whose value must not reach a log file shared with other users.
SECRET_FLAGS = ("--api-key",)


def _masked_argv(argv: List[str]) -> List[str]:
    """``argv`` with the value of every secret flag replaced by ``***``."""
    masked: List[str] = []
    hide_next = False
    for item in argv:
        if hide_next:
            masked.append("***")
            hide_next = False
        elif item in SECRET_FLAGS:
            masked.append(item)
            hide_next = True
        elif item.split("=", 1)[0] in SECRET_FLAGS and "=" in item:
            masked.append(item.split("=", 1)[0] + "=***")
        else:
            masked.append(item)
    return masked


def _log_run_header(argv: List[str]) -> None:
    """Log, at DEBUG, what identifies this run in a shared log: version, arguments, host, PID, SLURM IDs."""
    logging.debug("MetaQuest v%s: %s", __version__, " ".join(["metaquest", *_masked_argv(argv)]))
    logging.debug("Host %s, process ID %d", socket.gethostname(), os.getpid())
    slurm = [f"{name}={os.environ[name]}" for name in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID") if os.environ.get(name)]
    if slurm:
        logging.debug("SLURM: %s", ", ".join(slurm))


def _configure_logging(runtime: settings.RuntimeSettings) -> Optional[int]:
    """Set up logging from the runtime settings; an exit code when the log file cannot be opened."""
    level = getattr(logging, runtime.log_level)
    try:
        setup_logging(
            level=level,
            log_file=runtime.log_file,
            show_host=runtime.log_host,
            console_traceback=level <= logging.DEBUG,
        )
    except OSError as e:
        setup_logging(level=level, show_host=runtime.log_host)
        error = ConfigurationError(
            f"Cannot open the log file {runtime.log_file} (from {runtime.source('log_file')}): {e}"
        )
        logging.error(f"Error: {error}")
        return exit_code_for(error)
    return None


def _log_failure(message: str, error: BaseException) -> None:
    """Log a failure that left the command: one line on the console, the traceback in the log file."""
    logging.error(message, exc_info=error)
    log_traceback_hint()


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (if None, sys.argv[1:] is used)

    Returns:
        Exit code: 0 success, 1 failure, 2 usage error, 3 configuration problem, 4 retryable
        failure (network, a lock wait that gave up), 130 interrupted
        (``metaquest.core.exceptions.ExitCode``).
    """
    argv = list(sys.argv[1:] if args is None else args)
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    _apply_quiet_and_verbose(parser, parsed_args)

    # Every runtime setting is resolved once, before logging is set up, since the level is one.
    try:
        runtime = settings.activate(parsed_args)
    except ConfigurationError as e:
        if getattr(parsed_args, "command", None) != "doctor":
            setup_logging(level=logging.INFO)
            logging.error(f"Error: {e}")
            return exit_code_for(e)
        # doctor exists to diagnose exactly this: it runs on the built-in defaults and its
        # config check reports the error (exit code 3 through its own failed check).
        runtime = settings.activate_defaults()
        setattr(parsed_args, STARTUP_ERROR_ATTR, str(e))
    failed = _configure_logging(runtime)
    if failed is not None:
        return failed
    for message in runtime.warnings:
        logging.getLogger("metaquest.core.settings").warning(message)
    _log_run_header(argv)
    logging.debug("Runtime settings (value and source):\n  %s", "\n  ".join(runtime.describe()))

    try:
        # Execute the chosen command
        return parsed_args.func(parsed_args)
    except MetaQuestError as e:
        _log_failure(f"Error: {e}", e)
        return exit_code_for(e)
    except Exception as e:
        _log_failure(f"{type(e).__name__}: {e}", e)
        return exit_code_for(e)
    except KeyboardInterrupt as e:
        # Only a command with graceful_shutdown False gets here; BaseCommand.run handles the rest.
        logging.error("Interrupted")
        return exit_code_for(e)


if __name__ == "__main__":
    sys.exit(main())
