"""
Statistical analysis for MetaQuest.

This module provides functions for statistical analysis of genomic data.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from metaquest.core.exceptions import ProcessingError

logger = logging.getLogger(__name__)


def calculate_enrichment(
    observed_counts: Dict[str, int], expected_counts: Dict[str, int], min_count: int = 5
) -> pd.DataFrame:
    """
    Calculate enrichment of observed counts relative to expected.

    Args:
        observed_counts: Dictionary of observed counts
        expected_counts: Dictionary of expected counts
        min_count: Minimum count for inclusion in results

    Returns:
        DataFrame with enrichment statistics

    Raises:
        ProcessingError: If the calculation fails
    """
    try:
        # Prepare counts for common keys
        common_keys = set(observed_counts.keys()) & set(expected_counts.keys())

        if not common_keys:
            logger.warning("No common keys between observed and expected counts")
            return pd.DataFrame()

        results = []

        # Calculate statistics for each key
        for key in common_keys:
            observed = observed_counts.get(key, 0)
            expected = expected_counts.get(key, 0)

            # Skip items with low counts
            if observed < min_count and expected < min_count:
                continue

            # Calculate fold enrichment
            if expected > 0:
                fold_enrichment = observed / expected
            else:
                fold_enrichment = float("inf") if observed > 0 else 1.0

            # Calculate p-value with Fisher's exact test
            # Create contingency table
            # [[observed, total_observed - observed],
            #  [expected, total_expected - expected]]
            total_observed = sum(observed_counts.values())
            total_expected = sum(expected_counts.values())

            contingency_table = np.array(
                [
                    [observed, total_observed - observed],
                    [expected, total_expected - expected],
                ]
            )

            odds_ratio, p_value = stats.fisher_exact(contingency_table)

            # Store results
            results.append(
                {
                    "key": key,
                    "observed": observed,
                    "expected": expected,
                    "fold_enrichment": fold_enrichment,
                    "p_value": p_value,
                }
            )

        # Create DataFrame and sort by enrichment
        if not results:
            logger.warning("No results after filtering")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Add multiple testing correction
        if len(df) > 1:
            df["adjusted_p_value"] = stats.multipletests(
                df["p_value"], method="fdr_bh"
            )[1]
        else:
            df["adjusted_p_value"] = df["p_value"]

        # Sort by fold enrichment
        df = df.sort_values("fold_enrichment", ascending=False)

        return df

    except Exception as e:
        raise ProcessingError(f"Error calculating enrichment: {e}")


def calculate_distance_matrix(
    summary_file: Union[str, Path], threshold: float = 0.1, method: str = "jaccard"
) -> pd.DataFrame:
    """
    Calculate distance matrix between samples based on genome presence.

    Args:
        summary_file: Path to the summary file
        threshold: Minimum containment threshold
        method: Distance method ('jaccard', 'dice', or 'cosine')

    Returns:
        DataFrame with distance matrix

    Raises:
        ProcessingError: If the calculation fails
    """
    try:
        # Validate method
        valid_methods = ("jaccard", "dice", "cosine")
        if method not in valid_methods:
            raise ProcessingError(
                f"Invalid distance method: {method}. "
                f"Valid methods: {', '.join(valid_methods)}"
            )

        # Load summary data
        summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)

        # Get genome columns
        genome_columns = [
            col for col in summary_df.columns if "GCF" in col or "GCA" in col
        ]

        if not genome_columns:
            raise ProcessingError("No genome columns found in summary file")

        # Create binary presence/absence matrix
        presence_df = pd.DataFrame(index=summary_df.index)

        for col in genome_columns:
            presence_df[col] = (summary_df[col] > threshold).astype(int)

        # Initialize distance matrix
        sample_count = len(presence_df)
        distance_matrix = pd.DataFrame(
            index=presence_df.index, columns=presence_df.index
        )

        # Calculate distances
        for i, sample1 in enumerate(presence_df.index):
            # Self-distance is 0
            distance_matrix.loc[sample1, sample1] = 0.0

            # Calculate distance to other samples
            for sample2 in presence_df.index[i + 1 :]:
                # Get binary vectors
                vec1 = presence_df.loc[sample1].values
                vec2 = presence_df.loc[sample2].values

                # Calculate distance based on method
                if method == "jaccard":
                    # Jaccard distance = 1 - (intersection / union)
                    intersection = np.sum(np.logical_and(vec1, vec2))
                    union = np.sum(np.logical_or(vec1, vec2))

                    if union > 0:
                        distance = 1.0 - (intersection / union)
                    else:
                        distance = 1.0

                elif method == "dice":
                    # Dice distance = 1 - (2 * intersection / (sum1 + sum2))
                    intersection = np.sum(np.logical_and(vec1, vec2))
                    sum1 = np.sum(vec1)
                    sum2 = np.sum(vec2)

                    if (sum1 + sum2) > 0:
                        distance = 1.0 - (2.0 * intersection / (sum1 + sum2))
                    else:
                        distance = 1.0

                elif method == "cosine":
                    # Cosine distance = 1 - (dot product / (norm1 * norm2))
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)

                    if norm1 > 0 and norm2 > 0:
                        distance = 1.0 - (dot_product / (norm1 * norm2))
                    else:
                        distance = 1.0

                # Set distance (symmetric matrix)
                distance_matrix.loc[sample1, sample2] = distance
                distance_matrix.loc[sample2, sample1] = distance

        logger.info(f"Calculated {method} distances for {sample_count} samples")

        return distance_matrix

    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error calculating distance matrix: {e}")


def perform_hypergeometric_test(
    success_in_sample: int,
    sample_size: int,
    success_in_population: int,
    population_size: int,
) -> Tuple[float, float]:
    """
    Perform hypergeometric test for enrichment.

    Args:
        success_in_sample: Number of successes in the sample
        sample_size: Size of the sample
        success_in_population: Number of successes in the population
        population_size: Size of the population

    Returns:
        Tuple of (fold_enrichment, p_value)

    Raises:
        ProcessingError: If the calculation fails
    """
    try:
        # Validate inputs
        if sample_size <= 0 or population_size <= 0:
            raise ProcessingError("Sample and population sizes must be positive")

        if success_in_sample > sample_size:
            raise ProcessingError("Success count cannot exceed sample size")

        if success_in_population > population_size:
            raise ProcessingError("Success count cannot exceed population size")

        # Calculate expected number of successes
        expected = (success_in_population / population_size) * sample_size

        # Calculate fold enrichment
        if expected > 0:
            fold_enrichment = success_in_sample / expected
        else:
            fold_enrichment = float("inf") if success_in_sample > 0 else 1.0

        # Calculate p-value
        # Probability of observing success_in_sample OR MORE successes
        p_value = stats.hypergeom.sf(
            success_in_sample - 1,  # -1 because sf gives P(X > k)
            population_size,
            success_in_population,
            sample_size,
        )

        return fold_enrichment, p_value

    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Error performing hypergeometric test: {e}")
