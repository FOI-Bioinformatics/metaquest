"""
Data access layer for MetaQuest.

This package contains modules for accessing and manipulating data from various sources.
"""

# Import commonly used functions for easier access
from metaquest.data.file_io import (
    ensure_directory,
    list_files,
    copy_file,
    read_csv,
    write_csv,
    read_json,
    write_json,
)
from metaquest.data.branchwater import (
    process_branchwater_files,
    extract_metadata_from_branchwater,
    parse_containment_data,
)
