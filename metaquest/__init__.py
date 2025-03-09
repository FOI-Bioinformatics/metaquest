"""
MetaQuest - A toolkit for analyzing metagenomic datasets based on genome containment.

MetaQuest helps users search through SRA datasets to find containment of specified 
genomes and analyze associated metadata.
"""

__version__ = "0.3.0"
__author__ = "Andreas Sjödin"
__email__ = "andreas.sjodin@gmail.com"

# Import commonly used modules for easier access
from metaquest.core.exceptions import MetaQuestError, ValidationError
from metaquest.utils.config import get_config, set_config
from metaquest.utils.logging import setup_logging

# Initialize logging by default
setup_logging()