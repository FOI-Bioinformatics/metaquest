"""
Command Line Interface for MetaQuest.

This package provides the command-line interface for the MetaQuest toolkit.
"""

# Import command handlers and CLI entry point
from metaquest.cli.commands import (
    download_test_genome_command,
    process_branchwater_command,
    extract_branchwater_metadata_command,
    parse_containment_command,
    download_metadata_command,
    parse_metadata_command,
    count_metadata_command,
    single_sample_command,
    plot_containment_command,
    plot_metadata_counts_command,
    download_sra_command,
    assemble_datasets_command,
)
from metaquest.cli.main import main, create_parser
