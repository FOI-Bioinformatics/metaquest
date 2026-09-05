"""
Containment analysis for MetaQuest.

This module provides functions for analyzing genome containment data.
"""

import logging
import gzip
import urllib.request
import pandas as pd
from pathlib import Path
from typing import Dict, Union

from metaquest.core.exceptions import ProcessingError
from metaquest.data.file_io import ensure_directory

logger = logging.getLogger(__name__)


def download_test_genome(output_folder: Union[str, Path]) -> Path:
    """
    Download a test genome for demonstration purposes.

    Args:
        output_folder: Folder to save the downloaded genome

    Returns:
        Path to the downloaded genome file

    Raises:
        ProcessingError: If the download fails
    """
    try:
        output_path = ensure_directory(output_folder) / "GCF_000008985.1.fasta"

        # Skip if the file already exists
        if output_path.exists():
            logger.info(f"Test genome already exists at {output_path}")
            return output_path

        logger.info("Downloading test genome")

        # URL for Rickettsia prowazekii genome (small genome for testing)
        url = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/008/985/"
            "GCF_000008985.1_ASM898v1/GCF_000008985.1_ASM898v1_genomic.fna.gz"
        )

        # Download compressed file
        temp_file = output_path.with_suffix(".gz")
        response = urllib.request.urlopen(url, timeout=30)
        with open(temp_file, "wb") as dl_f:
            dl_f.write(response.read())

        # Decompress file
        with gzip.open(temp_file, "rt") as f_in:
            with open(output_path, "w") as f_out:
                f_out.write(f_in.read())

        # Remove temporary file
        temp_file.unlink()

        logger.info(f"Downloaded test genome to {output_path}")
        return output_path

    except Exception as e:
        raise ProcessingError(f"Error downloading test genome: {e}")


def count_single_sample(
    summary_file: Union[str, Path],
    metadata_file: Union[str, Path],
    summary_column: str,
    metadata_column: str,
    threshold: float = 0.1,
    top_n: int = 100,
) -> Dict[str, int]:
    """
    Count occurrences of metadata values in samples matching a genome.

    Args:
        summary_file: Path to the containment summary file
        metadata_file: Path to the metadata table file
        summary_column: Column name in the summary file (usually a genome ID)
        metadata_column: Column name in the metadata file to count
        threshold: Minimum containment threshold
        top_n: Number of top items to keep

    Returns:
        Dictionary mapping metadata values to counts

    Raises:
        ProcessingError: If the operation fails
    """
    try:
        # Load the summary and metadata dataframes
        summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
        metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)

        # Validate column exists in summary dataframe
        if summary_column not in summary_df.columns:
            raise ProcessingError(
                f"Column {summary_column} not found in summary file. "
                f"Available columns: {', '.join(summary_df.columns)}"
            )

        # Validate column exists in metadata dataframe
        if metadata_column not in metadata_df.columns:
            raise ProcessingError(
                f"Column {metadata_column} not found in metadata file. "
                f"Available columns: {', '.join(metadata_df.columns)}"
            )

        # Find accessions at or above threshold
        selected_accessions = summary_df[summary_df[summary_column] >= threshold].index
        logger.info(f"Found {len(selected_accessions)} accessions with {summary_column} >= {threshold}")

        if len(selected_accessions) == 0:
            logger.warning("No accessions found above threshold")
            return {}

        # Filter metadata by selected accessions
        filtered_metadata = metadata_df.loc[metadata_df.index.isin(selected_accessions)]

        if filtered_metadata.empty:
            logger.warning("No matching metadata found for selected accessions")
            return {}

        # Count occurrences of values in the specified column
        value_counts = filtered_metadata[metadata_column].value_counts()

        # Get top N items
        top_items = value_counts.head(top_n).to_dict()

        logger.info(f"Found {len(value_counts)} unique values in {metadata_column}")
        logger.info(f"Top {min(top_n, len(top_items))} values:")

        for key, count in list(top_items.items())[:5]:
            logger.info(f"  {key}: {count}")

        if len(top_items) > 5:
            logger.info(f"  ... and {len(top_items) - 5} more")

        return top_items

    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error counting single sample metadata: {e}")
