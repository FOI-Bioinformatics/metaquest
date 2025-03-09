"""
Containment analysis for MetaQuest.

This module provides functions for analyzing genome containment data.
"""

import logging
import gzip
import urllib.request
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from metaquest.core.exceptions import ProcessingError
from metaquest.core.validation import validate_folder
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
        url = ("https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/008/985/"
               "GCF_000008985.1_ASM898v1/GCF_000008985.1_ASM898v1_genomic.fna.gz")
        
        # Download compressed file
        temp_file = output_path.with_suffix('.gz')
        urllib.request.urlretrieve(url, temp_file)
        
        # Decompress file
        with gzip.open(temp_file, 'rt') as f_in:
            with open(output_path, 'w') as f_out:
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
    top_n: int = 100
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
        
        # Find accessions above threshold
        selected_accessions = summary_df[summary_df[summary_column] > threshold].index
        logger.info(
            f"Found {len(selected_accessions)} accessions with {summary_column} > {threshold}"
        )
        
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


def filter_samples_by_containment(
    summary_file: Union[str, Path],
    threshold: float = 0.5,
    genome_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter samples based on containment threshold.
    
    Args:
        summary_file: Path to the containment summary file
        threshold: Minimum containment threshold
        genome_id: Specific genome ID to filter by (if None, use max_containment)
        
    Returns:
        DataFrame of filtered samples
        
    Raises:
        ProcessingError: If the operation fails
    """
    try:
        # Load the summary dataframe
        summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
        
        # Apply filter
        if genome_id is not None:
            if genome_id not in summary_df.columns:
                raise ProcessingError(
                    f"Genome {genome_id} not found in summary file. "
                    f"Available genomes: {', '.join([col for col in summary_df.columns if col not in ('max_containment', 'max_containment_annotation')])}"
                )
            filtered_df = summary_df[summary_df[genome_id] > threshold]
            logger.info(
                f"Found {len(filtered_df)} samples with {genome_id} > {threshold}"
            )
        else:
            filtered_df = summary_df[summary_df['max_containment'] > threshold]
            logger.info(
                f"Found {len(filtered_df)} samples with max_containment > {threshold}"
            )
        
        return filtered_df
        
    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error filtering samples by containment: {e}")


def find_co_occurring_genomes(
    summary_file: Union[str, Path],
    threshold: float = 0.5,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Find genomes that co-occur in the same samples.
    
    Args:
        summary_file: Path to the containment summary file
        threshold: Minimum containment threshold
        min_samples: Minimum number of samples for a genome to be considered
        
    Returns:
        DataFrame with co-occurrence matrix
        
    Raises:
        ProcessingError: If the operation fails
    """
    try:
        # Load the summary dataframe
        summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
        
        # Get genome columns
        genome_cols = [col for col in summary_df.columns 
                      if col not in ('max_containment', 'max_containment_annotation')
                      and ('GCA' in col or 'GCF' in col)]
        
        if not genome_cols:
            raise ProcessingError("No genome columns found in summary file")
        
        # Create binary presence/absence matrix
        presence_df = pd.DataFrame(index=summary_df.index)
        
        for col in genome_cols:
            presence_df[col] = (summary_df[col] > threshold).astype(int)
        
        # Filter to genomes present in at least min_samples
        genome_counts = presence_df.sum()
        frequent_genomes = genome_counts[genome_counts >= min_samples].index.tolist()
        
        if not frequent_genomes:
            logger.warning(
                f"No genomes found in at least {min_samples} samples at threshold {threshold}"
            )
            return pd.DataFrame()
        
        # Create co-occurrence matrix
        cooccurrence_matrix = pd.DataFrame(index=frequent_genomes, columns=frequent_genomes)
        
        for i in frequent_genomes:
            for j in frequent_genomes:
                # Count samples where both genomes are present
                cooccurrence_matrix.loc[i, j] = ((presence_df[i] == 1) & 
                                               (presence_df[j] == 1)).sum()
        
        logger.info(
            f"Created co-occurrence matrix for {len(frequent_genomes)} genomes"
        )
        return cooccurrence_matrix
        
    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error finding co-occurring genomes: {e}")