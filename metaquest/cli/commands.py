"""
Command handlers for MetaQuest CLI.

This module provides the implementation for all CLI commands.
"""

import argparse
import logging
from pathlib import Path

from metaquest.core.exceptions import MetaQuestError
from metaquest.data.branchwater import (
    process_branchwater_files,
    extract_metadata_from_branchwater,
    parse_containment_data,
)
from metaquest.processing.containment import (
    download_test_genome as dp_download_test_genome,
    count_single_sample as dp_count_single_sample,
)
from metaquest.processing.counts import count_metadata as dp_count_metadata
from metaquest.data.metadata import (
    download_metadata as dp_download_metadata,
    parse_metadata as dp_parse_metadata,
)
from metaquest.data.sra import (
    download_sra as dp_download_sra,
    assemble_datasets as dp_assemble_datasets,
)
from metaquest.visualization.plots import (
    plot_containment as viz_plot_containment,
    plot_metadata_counts as viz_plot_metadata_counts,
)

logger = logging.getLogger(__name__)


def download_test_genome_command(args: argparse.Namespace) -> int:
    """
    Command handler for download_test_genome.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        dp_download_test_genome(args.output_folder)
        return 0
    except MetaQuestError as e:
        logger.error(f"Error downloading test genome: {e}")
        return 1


def process_branchwater_command(args: argparse.Namespace) -> int:
    """
    Command handler for use_branchwater.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        process_branchwater_files(args.branchwater_folder, args.matches_folder)
        return 0
    except MetaQuestError as e:
        logger.error(f"Error processing Branchwater files: {e}")
        return 1


def extract_branchwater_metadata_command(args: argparse.Namespace) -> int:
    """
    Command handler for extract_branchwater_metadata.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        metadata_folder = Path(args.metadata_folder)
        metadata_folder.mkdir(exist_ok=True)
        output_file = metadata_folder / "branchwater_metadata.txt"

        extract_metadata_from_branchwater(args.branchwater_folder, output_file)
        return 0
    except MetaQuestError as e:
        logger.error(f"Error extracting metadata: {e}")
        return 1


def parse_containment_command(args: argparse.Namespace) -> int:
    """
    Command handler for parse_containment.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        parse_containment_data(
            args.matches_folder,
            args.parsed_containment_file,
            args.summary_containment_file,
            args.step_size,
        )
        return 0
    except MetaQuestError as e:
        logger.error(f"Error parsing containment: {e}")
        return 1


def download_metadata_command(args: argparse.Namespace) -> int:
    """
    Command handler for download_metadata.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        dp_download_metadata(
            email=args.email,
            matches_folder=args.matches_folder,
            metadata_folder=args.metadata_folder,
            threshold=args.threshold,
            dry_run=args.dry_run,
        )
        return 0
    except MetaQuestError as e:
        logger.error(f"Error downloading metadata: {e}")
        return 1


def parse_metadata_command(args: argparse.Namespace) -> int:
    """
    Command handler for parse_metadata.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        dp_parse_metadata(args.metadata_folder, args.metadata_table_file)
        return 0
    except MetaQuestError as e:
        logger.error(f"Error parsing metadata: {e}")
        return 1


def count_metadata_command(args: argparse.Namespace) -> int:
    """
    Command handler for count_metadata.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        dp_count_metadata(
            summary_file=args.summary_file,
            metadata_file=args.metadata_file,
            metadata_column=args.metadata_column,
            threshold=args.threshold,
            output_file=args.output_file,
            stat_file=args.stat_file,
        )
        return 0
    except MetaQuestError as e:
        logger.error(f"Error counting metadata: {e}")
        return 1


def single_sample_command(args: argparse.Namespace) -> int:
    """
    Command handler for single_sample.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        count_dict = dp_count_single_sample(
            summary_file=args.summary_file,
            metadata_file=args.metadata_file,
            summary_column=args.summary_column,
            metadata_column=args.metadata_column,
            threshold=args.threshold,
            top_n=args.top_n,
        )

        if not count_dict:
            logger.warning("No data found above threshold")
            return 0

        # Log the top N items
        for key, value in list(count_dict.items())[: args.top_n]:
            logger.info(f"{key}: {value}")

        return 0
    except MetaQuestError as e:
        logger.error(f"Error analyzing single sample: {e}")
        return 1


def plot_containment_command(args: argparse.Namespace) -> int:
    """
    Command handler for plot_containment.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
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
        logger.error(f"Error plotting containment: {e}")
        return 1


def plot_metadata_counts_command(args: argparse.Namespace) -> int:
    """
    Command handler for plot_metadata_counts.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        viz_plot_metadata_counts(
            file_path=args.file_path,
            title=args.title,
            plot_type=args.plot_type,
            colors=args.colors,
            show_title=args.show_title,
            save_format=args.save_format,
        )
        return 0
    except MetaQuestError as e:
        logger.error(f"Error plotting metadata counts: {e}")
        return 1


def download_sra_command(args: argparse.Namespace) -> int:
    """
    Command handler for download_sra.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        download_stats = dp_download_sra(
            fastq_folder=args.fastq_folder,
            accessions_file=args.accessions_file,
            max_downloads=args.max_downloads,
            dry_run=args.dry_run,
            num_threads=args.num_threads,
            max_workers=args.max_workers,
            force=args.force,
            max_retries=args.max_retries,
            temp_folder=args.temp_folder,
            blacklist=args.blacklist,
        )

        if args.dry_run:
            logger.info(
                f"Dry run: would download {download_stats['to_download']} of {download_stats['total']} datasets"
            )
            logger.info(
                f"  {download_stats['already_downloaded']} datasets would be skipped (already downloaded)"
            )
            if 'blacklisted' in download_stats and download_stats['blacklisted'] > 0:
                logger.info(
                    f"  {download_stats['blacklisted']} datasets would be skipped (blacklisted)"
                )
            if "to_download" in download_stats and download_stats["to_download"] > 0:
                logger.info(f"  Output folder would be: {args.fastq_folder}")
                if args.max_downloads:
                    logger.info(f"  Limited to {args.max_downloads} downloads")
        else:
            logger.info("Download summary:")
            logger.info(
                f"  Successfully downloaded: {download_stats['successful']} datasets"
            )
            logger.info(f"  Failed downloads: {download_stats['failed']} datasets")
            logger.info(
                f"  Already downloaded: {download_stats['already_downloaded']} datasets"
            )
            if 'blacklisted' in download_stats and download_stats['blacklisted'] > 0:
                logger.info(
                    f"  Blacklisted: {download_stats['blacklisted']} datasets"
                )
            logger.info(f"  Total processed: {download_stats['total']} datasets")

        # Return error if there were failed downloads
        if not args.dry_run and download_stats["failed"] > 0:
            logger.warning(
                "Some downloads failed. Use --force to retry or --max-retries to enable automatic retry."
            )
            if (
                "failed_accessions" in download_stats
                and download_stats["failed_accessions"]
            ):
                # Write failed accessions to file for easier retry
                failed_file = Path(args.fastq_folder) / "failed_accessions.txt"
                with open(failed_file, "w") as f:
                    for acc in download_stats["failed_accessions"]:
                        f.write(f"{acc}\n")
                logger.info(f"Failed accessions written to {failed_file}")
                logger.info(
                    f"To retry only failed accessions: metaquest download_sra "
                    f"--accessions-file {failed_file} "
                    f"--fastq-folder {args.fastq_folder}"
                )
            return 1  # Return error code

        return 0

    except MetaQuestError as e:
        logger.error(f"Error downloading SRA data: {e}")
        return 1


def assemble_datasets_command(args: argparse.Namespace) -> int:
    """
    Command handler for assemble_datasets.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        dp_assemble_datasets(args)
        return 0
    except MetaQuestError as e:
        logger.error(f"Error assembling datasets: {e}")
        return 1