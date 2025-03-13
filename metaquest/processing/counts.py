"""
Metadata counting for MetaQuest.

This module provides functions for counting and analyzing metadata.
"""

import logging
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union

from metaquest.core.exceptions import ProcessingError
from metaquest.data.file_io import write_csv

logger = logging.getLogger(__name__)


def count_metadata(
    summary_file: Union[str, Path],
    metadata_file: Union[str, Path],
    metadata_column: str,
    threshold: float,
    output_file: Union[str, Path],
    stat_file: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Count occurrences of metadata values by genome.

    Args:
        summary_file: Path to the summary file
        metadata_file: Path to the metadata file
        metadata_column: Column in metadata file to count
        threshold: Minimum containment threshold
        output_file: Path to save the result table
        stat_file: Path to save the statistics file

    Returns:
        DataFrame with count results

    Raises:
        ProcessingError: If the operation fails
    """
    try:
        # Generate statistics file path if not provided
        if stat_file is None:
            base_name, ext = os.path.splitext(output_file)
            stat_file = f"{base_name}_stats{ext}"

        # Load data
        summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
        metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)

        # Validate metadata column exists
        if metadata_column not in metadata_df.columns:
            available_columns = ", ".join(metadata_df.columns)
            raise ProcessingError(
                f"Column '{metadata_column}' not found in metadata file. "
                f"Available columns: {available_columns}"
            )

        # Get genome columns
        genome_columns = [
            col for col in summary_df.columns if "GCF" in col or "GCA" in col
        ]

        if not genome_columns:
            raise ProcessingError("No genome columns found in summary file")

        logger.info(f"Processing {len(genome_columns)} genome columns")

        # Create result dataframes for each genome
        df_list = []
        unique_accessions = set()

        for genome_column in genome_columns:
            try:
                # Find accessions with containment above threshold
                selected_accessions = summary_df[
                    summary_df[genome_column] > threshold
                ].index

                # Skip if no matching accessions
                if len(selected_accessions) == 0:
                    logger.warning(
                        f"No accessions found for {genome_column} above threshold {threshold}"
                    )
                    continue

                unique_accessions.update(selected_accessions)

                # Filter metadata to selected accessions
                selected_metadata = metadata_df[
                    metadata_df.index.isin(selected_accessions)
                ]

                # Skip if no matching metadata
                if selected_metadata.empty:
                    logger.warning(f"No metadata found for {genome_column} accessions")
                    continue

                # Count values in metadata column
                count_series = selected_metadata[metadata_column].value_counts()

                # Create DataFrame with counts
                count_df = pd.DataFrame({genome_column: count_series})
                df_list.append(count_df)

            except Exception as e:
                logger.error(f"Error processing genome {genome_column}: {e}")

        # Skip if no data frames created
        if not df_list:
            logger.warning("No count data generated")
            return pd.DataFrame()

        # Combine all dataframes
        result_df = pd.concat(df_list, axis=1)

        # Fill NA values with 0 and convert to int
        result_df = result_df.fillna(0).astype(int)

        # Save to file
        write_csv(result_df, output_file, sep="\t")
        logger.info(f"Saved count table to {output_file}")

        # Calculate and save column sums
        column_sums = result_df.sum().sort_values(ascending=False)
        write_csv(column_sums, stat_file, sep="\t", header=False)
        logger.info(f"Saved column statistics to {stat_file}")

        # Log summary information
        total_counts = result_df.values.sum()
        logger.info(f"Total counts in table: {total_counts}")
        logger.info(f"Unique accessions after filtering: {len(unique_accessions)}")

        return result_df

    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error counting metadata: {e}")


def count_metadata_by_category(
    metadata_file: Union[str, Path],
    category_column: str,
    count_column: str,
    output_file: Optional[Union[str, Path]] = None,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Create a contingency table of counts by category.

    Args:
        metadata_file: Path to the metadata file
        category_column: Column to use for categories
        count_column: Column to count within categories
        output_file: Path to save the result table
        min_count: Minimum count for a value to be included

    Returns:
        DataFrame with count results

    Raises:
        ProcessingError: If the operation fails
    """
    try:
        # Load metadata
        metadata_df = pd.read_csv(metadata_file, sep="\t")

        # Validate columns exist
        if category_column not in metadata_df.columns:
            available_columns = ", ".join(metadata_df.columns)
            raise ProcessingError(
                f"Category column '{category_column}' not found in metadata file. "
                f"Available columns: {available_columns}"
            )

        if count_column not in metadata_df.columns:
            available_columns = ", ".join(metadata_df.columns)
            raise ProcessingError(
                f"Count column '{count_column}' not found in metadata file. "
                f"Available columns: {available_columns}"
            )

        # Create contingency table
        contingency_table = pd.crosstab(
            index=metadata_df[category_column], columns=metadata_df[count_column]
        )

        # Filter by minimum count
        contingency_table = contingency_table[
            contingency_table.sum(axis=1) >= min_count
        ]

        # Sort by row sums
        contingency_table = contingency_table.loc[
            contingency_table.sum(axis=1).sort_values(ascending=False).index
        ]

        # Save to file if specified
        if output_file:
            write_csv(contingency_table, output_file, sep="\t")
            logger.info(f"Saved contingency table to {output_file}")

        return contingency_table

    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error creating contingency table: {e}")


def summarize_metadata_column(
    metadata_file: Union[str, Path],
    column: str,
    output_file: Optional[Union[str, Path]] = None,
) -> Dict[str, int]:
    """
    Summarize values in a metadata column.

    Args:
        metadata_file: Path to the metadata file
        column: Column to summarize
        output_file: Path to save the result summary

    Returns:
        Dictionary with value counts

    Raises:
        ProcessingError: If the operation fails
    """
    try:
        # Load metadata
        metadata_df = pd.read_csv(metadata_file, sep="\t")

        # Validate column exists
        if column not in metadata_df.columns:
            available_columns = ", ".join(metadata_df.columns)
            raise ProcessingError(
                f"Column '{column}' not found in metadata file. "
                f"Available columns: {available_columns}"
            )

        # Count values
        value_counts = metadata_df[column].value_counts()

        # Calculate non-null percentage
        non_null_count = value_counts.sum()
        total_count = len(metadata_df)
        non_null_percentage = (non_null_count / total_count) * 100

        logger.info(f"Column: {column}")
        logger.info(
            f"Non-null values: {non_null_count}/{total_count} ({non_null_percentage:.1f}%)"
        )
        logger.info(f"Unique values: {len(value_counts)}")

        # Log top values
        logger.info("Top values:")
        for value, count in value_counts.head(10).items():
            logger.info(f"  {value}: {count}")

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(f"Column: {column}\n")
                f.write(
                    f"Non-null values: {non_null_count}/{total_count} ({non_null_percentage:.1f}%)\n"
                )
                f.write(f"Unique values: {len(value_counts)}\n\n")
                f.write("Value counts:\n")

                for value, count in value_counts.items():
                    f.write(f"{value}\t{count}\n")

            logger.info(f"Saved summary to {output_file}")

        return value_counts.to_dict()

    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error summarizing metadata column: {e}")
