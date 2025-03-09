"""
Command handlers for MetaQuest CLI.

This module provides the implementation for all CLI commands.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from metaquest.core.exceptions import MetaQuestError
from metaquest.data.branchwater import (
    process_branchwater_files,
    extract_metadata_from_branchwater,
    parse_containment_data
)
from metaquest.processing.containment import (
    download_test_genome as dp_download_test_genome,
    count_single_sample as dp_count_single_sample
)
from metaquest.processing.counts import (
    count_metadata as dp_count_metadata
)
from metaquest.data.metadata import (
    download_metadata as dp_download_metadata,
    parse_metadata as dp_parse_metadata,
    check_metadata_attributes as dp_check_metadata_attributes
)
from metaquest.data.sra import (
    download_sra as dp_download_sra,
    assemble_datasets as dp_assemble_datasets
)
from metaquest.visualization.plots import (
    plot_containment as viz_plot_containment,
    plot_metadata_counts as viz_plot_metadata_counts
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
            args.step_size
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
            dry_run=args.dry_run
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
            stat_file=args.stat_file
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
            top_n=args.top_n
        )
        
        if not count_dict:
            logger.warning("No data found above threshold")
            return 0
            
        # Log the top N items
        for key, value in list(count_dict.items())[:args.top_n]:
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
            plot_type=args.plot_type
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
            save_format=args.save_format
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
        download_count = dp_download_sra(
            fastq_folder=args.fastq_folder,
            accessions_file=args.accessions_file,
            max_downloads=args.max_downloads,
            dry_run=args.dry_run,
            num_threads=args.num_threads,
            max_workers=args.max_workers
        )
        
        if args.dry_run:
            logger.info(f"Would download {download_count} datasets")
        else:
            logger.info(f"Successfully downloaded {download_count} datasets")
            
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