"""
Data processing modules for MetaQuest.

This package contains modules for processing and analyzing data.
"""

# Import commonly used functions for easier access
from metaquest.processing.containment import download_test_genome, count_single_sample
from metaquest.processing.counts import count_metadata
from metaquest.processing.statistics import (
    calculate_enrichment,
    calculate_distance_matrix,
)
