"""
MetaQuest - A toolkit for analyzing metagenomic datasets based on genome containment.

MetaQuest helps users search through SRA datasets to find containment of specified
genomes and analyze associated metadata.
"""

__version__ = "0.3.1"
__author__ = "Andreas Sjödin"
__email__ = "andreas.sjodin@gmail.com"

# Import commonly used modules for easier access
from metaquest.utils.logging import setup_logging

# Initialize logging by default
setup_logging()
