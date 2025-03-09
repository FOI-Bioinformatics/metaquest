"""
Core components for MetaQuest.

This package contains the core domain models, validation logic, and exceptions.
"""

# Import commonly used components for easier access
from metaquest.core.exceptions import (
    MetaQuestError, ValidationError, FormatError, 
    DataAccessError, ProcessingError, VisualizationError
)
from metaquest.core.models import (
    Containment, GenomeInfo, SRAMetadata, ContainmentSummary
)
from metaquest.core.validation import (
    validate_csv_file, detect_file_format, validate_accession,
    validate_containment_value, validate_folder
)