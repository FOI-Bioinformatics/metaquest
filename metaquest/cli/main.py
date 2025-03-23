"""
Command Line Interface entry point for MetaQuest.

This module provides the main CLI entry point and argument parsing.
"""

import argparse
import logging
import sys
from typing import List, Optional

from metaquest import __version__
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
from metaquest.utils.logging import setup_logging


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

    subparsers = parser.add_subparsers(
        title="commands",
        description="valid commands",
        help="additional help",
        dest="command",
    )

    # Make subcommand required
    subparsers.required = True

    # Download test genome command
    parser_download_test_genome = subparsers.add_parser(
        "download_test_genome", help="Download test genome"
    )
    parser_download_test_genome.add_argument(
        "--output-folder",
        default="genomes",
        help="Folder to save the downloaded fasta files",
    )
    parser_download_test_genome.set_defaults(func=download_test_genome_command)

    # Process Branchwater files command
    parser_branchwater = subparsers.add_parser(
        "use_branchwater", help="Process pre-downloaded Branchwater files"
    )
    parser_branchwater.add_argument(
        "--branchwater-folder",
        required=True,
        help="Folder containing Branchwater output files",
    )
    parser_branchwater.add_argument(
        "--matches-folder", default="matches", help="Folder to save processed matches"
    )
    parser_branchwater.set_defaults(func=process_branchwater_command)

    # Extract metadata from Branchwater files command
    parser_extract_metadata = subparsers.add_parser(
        "extract_branchwater_metadata", help="Extract metadata from Branchwater files"
    )
    parser_extract_metadata.add_argument(
        "--branchwater-folder",
        required=True,
        help="Folder containing Branchwater output files",
    )
    parser_extract_metadata.add_argument(
        "--metadata-folder",
        default="metadata",
        help="Folder to save extracted metadata",
    )
    parser_extract_metadata.set_defaults(func=extract_branchwater_metadata_command)

    # Parse containment command
    parser_parse_containment = subparsers.add_parser(
        "parse_containment", help="Parse containment data from match files"
    )
    parser_parse_containment.add_argument(
        "--matches-folder",
        default="matches",
        help="Folder containing containment match files",
    )
    parser_parse_containment.add_argument(
        "--parsed-containment-file",
        default="parsed_containment.txt",
        help="File where the parsed containment will be stored",
    )
    parser_parse_containment.add_argument(
        "--summary-containment-file",
        default="top_containments.txt",
        help="File where the containment summary will be stored",
    )
    parser_parse_containment.add_argument(
        "--step-size",
        default="0.1",
        type=float,
        help="Size of steps for the containment thresholds",
    )
    parser_parse_containment.add_argument(
        "--file-format",
        default=None,
        choices=["branchwater", "mastiff"],
        help="Format of the input files (branchwater or mastiff)",
    )

    parser_parse_containment.set_defaults(func=parse_containment_command)

    # Download metadata command
    parser_download_metadata = subparsers.add_parser(
        "download_metadata", help="Download metadata for SRA accessions"
    )
    parser_download_metadata.add_argument(
        "--email", required=True, help="Your email address for NCBI API access"
    )
    parser_download_metadata.add_argument(
        "--matches-folder", default="matches", help="Folder containing match files"
    )
    parser_download_metadata.add_argument(
        "--metadata-folder",
        default="metadata",
        help="Folder to save downloaded metadata",
    )
    parser_download_metadata.add_argument(
        "--threshold", type=float, default=0.0, help="Threshold for containment values"
    )
    parser_download_metadata.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate number of accessions without downloading",
    )
    parser_download_metadata.set_defaults(func=download_metadata_command)

    # Parse metadata command
    parser_parse_metadata = subparsers.add_parser(
        "parse_metadata", help="Parse downloaded metadata files"
    )
    parser_parse_metadata.add_argument(
        "--metadata-folder", default="metadata", help="Folder containing metadata files"
    )
    parser_parse_metadata.add_argument(
        "--metadata-table-file",
        default="metadata_table.txt",
        help="File where the parsed metadata will be stored",
    )
    parser_parse_metadata.set_defaults(func=parse_metadata_command)

    # Count metadata command
    parser_count_metadata = subparsers.add_parser(
        "count_metadata", help="Count metadata values by genome"
    )
    parser_count_metadata.add_argument(
        "--summary-file",
        default="parsed_containment.txt",
        help="Path to the summary file",
    )
    parser_count_metadata.add_argument(
        "--metadata-file",
        default="metadata_table.txt",
        help="Path to the metadata file",
    )
    parser_count_metadata.add_argument(
        "--metadata-column",
        required=True,
        help="Name of the column in the metadata file",
    )
    parser_count_metadata.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for containment values"
    )
    parser_count_metadata.add_argument(
        "--output-file", default="metadata_counts.txt", help="Path to the output file"
    )
    parser_count_metadata.add_argument(
        "--stat-file", default=None, help="Path to the statistics file"
    )
    parser_count_metadata.set_defaults(func=count_metadata_command)

    # Single sample command
    parser_single_sample = subparsers.add_parser(
        "single_sample", help="Analyze a single sample"
    )
    parser_single_sample.add_argument(
        "--summary-file",
        default="parsed_containment.txt",
        help="Path to the summary file",
    )
    parser_single_sample.add_argument(
        "--metadata-file",
        default="metadata_table.txt",
        help="Path to the metadata file",
    )
    parser_single_sample.add_argument(
        "--summary-column", required=True, help="Name of the column in the summary file"
    )
    parser_single_sample.add_argument(
        "--metadata-column",
        required=True,
        help="Name of the column in the metadata file",
    )
    parser_single_sample.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for the column in the summary file",
    )
    parser_single_sample.add_argument(
        "--top-n", type=int, default=100, help="Number of top items to keep"
    )
    parser_single_sample.set_defaults(func=single_sample_command)

    # Plot containment command
    parser_plot_containment = subparsers.add_parser(
        "plot_containment", help="Plot containment data"
    )
    parser_plot_containment.add_argument(
        "--file-path", required=True, help="Path to the containment file"
    )
    parser_plot_containment.add_argument(
        "--column", default="max_containment", help="Column to plot"
    )
    parser_plot_containment.add_argument(
        "--title", default=None, help="Title of the plot"
    )
    parser_plot_containment.add_argument(
        "--colors", default=None, help="Colors to use in the plot"
    )
    parser_plot_containment.add_argument(
        "--show-title", action="store_true", help="Whether to display the title"
    )
    parser_plot_containment.add_argument(
        "--save-format",
        default=None,
        choices=["png", "jpg", "pdf", "svg"],
        help="Format to save the plot",
    )
    parser_plot_containment.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum value to include in the plot",
    )
    parser_plot_containment.add_argument(
        "--plot-type",
        default="rank",
        choices=["rank", "histogram", "box", "violin"],
        help="Type of plot to generate",
    )
    parser_plot_containment.set_defaults(func=plot_containment_command)

    # Plot metadata counts command
    parser_plot_metadata = subparsers.add_parser(
        "plot_metadata_counts", help="Plot metadata counts"
    )
    parser_plot_metadata.add_argument(
        "--file-path", required=True, help="Path to the metadata counts file"
    )
    parser_plot_metadata.add_argument(
        "--title", default=None, help="Title for the plot"
    )
    parser_plot_metadata.add_argument(
        "--plot-type",
        default="bar",
        choices=["bar", "pie", "radar"],
        help="Type of plot to generate",
    )
    parser_plot_metadata.add_argument(
        "--colors", default=None, help="Colors or colormap name"
    )
    parser_plot_metadata.add_argument(
        "--show-title", action="store_true", help="Whether to display the title"
    )
    parser_plot_metadata.add_argument(
        "--save-format",
        default=None,
        choices=["png", "jpg", "pdf", "svg"],
        help="Format to save the figure",
    )
    parser_plot_metadata.set_defaults(func=plot_metadata_counts_command)

    # Download SRA command
    parser_download_sra = subparsers.add_parser(
        "download_sra", help="Download SRA datasets"
    )
    parser_download_sra.add_argument(
        "--fastq-folder", default="fastq", help="Folder to save downloaded FASTQ files"
    )
    parser_download_sra.add_argument(
        "--accessions-file",
        required=True,
        help="File containing SRA accessions, one per line",
    )
    parser_download_sra.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Maximum number of datasets to download",
    )
    parser_download_sra.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for each fasterq-dump",
    )
    parser_download_sra.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of threads for parallel downloads",
    )
    parser_download_sra.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate number of accessions without downloading",
    )
    parser_download_sra.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if files exist",
    )
    parser_download_sra.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum number of retry attempts for failed downloads",
    )
    parser_download_sra.add_argument(
        "--temp-folder",
        help="Directory to use for fasterq-dump temporary files (must be writable)",
    )
    parser_download_sra.add_argument(
        "--blacklist",
        nargs="+",
        help="One or more files containing blacklisted accessions, one per line",
    )
    parser_download_sra.set_defaults(func=download_sra_command)

    # Assemble datasets command
    parser_assemble_datasets = subparsers.add_parser(
        "assemble_datasets", help="Assemble datasets from fastq files"
    )
    parser_assemble_datasets.add_argument(
        "--data-files", required=True, nargs="+", help="List of paths to data files"
    )
    parser_assemble_datasets.add_argument(
        "--output-file", required=True, help="Path to save the assembled dataset"
    )
    parser_assemble_datasets.set_defaults(func=assemble_datasets_command)

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