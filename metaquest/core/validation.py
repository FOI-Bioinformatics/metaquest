"""
Validation utilities for MetaQuest.

This module provides functions for validating input data and configurations.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from metaquest.core.exceptions import ValidationError, FormatError

logger = logging.getLogger(__name__)


def detect_file_format(file_path: Union[str, Path]) -> str:
    """
    Automatically detect the format of a CSV file.

    Args:
        file_path: Path to the CSV file to analyze

    Returns:
        Detected format: 'branchwater'

    Raises:
        FormatError: If the file format cannot be determined
    """
    try:
        with open(file_path, "r") as f:
            # Read the header line
            header = f.readline().strip()

            # Split the header into column names
            columns = [col.strip() for col in header.split(",")]

            # Check for branchwater format - must have "acc" column
            if "acc" in columns and "containment" in columns:
                return "branchwater"

            # If we can't determine the format
            raise FormatError(
                f"Could not determine file format for {file_path}. "
                f"Missing required columns for known formats. "
                f"Header: {header}"
            )
    except Exception as e:
        if isinstance(e, FormatError):
            raise
        raise FormatError(f"Error reading file {file_path}: {str(e)}")


def validate_accession(accession: str) -> bool:
    """
    Validate if a string looks like a valid SRA accession.

    Args:
        accession: The accession string to validate

    Returns:
        True if the accession is valid, False otherwise
    """
    # Basic validation - SRA accessions follow specific patterns
    from metaquest.core.constants import SRA_VALID_PREFIXES

    valid_prefixes = SRA_VALID_PREFIXES

    if not accession:
        return False

    if not accession.startswith(valid_prefixes):
        return False

    # Check if the rest is numeric
    suffix = accession[3:]
    if not suffix.isdigit():
        return False

    return True


def validate_containment_value(value: Union[str, float]) -> Optional[float]:
    """
    Validate and convert a containment value.

    Args:
        value: The containment value to validate

    Returns:
        The validated containment value as a float, or None if invalid
    """
    try:
        float_value = float(value)

        # Containment should be between 0 and 1
        if 0 <= float_value <= 1:
            return float_value
        else:
            logger.warning(f"Containment value {float_value} outside expected range [0, 1]")
            return None
    except (ValueError, TypeError):
        logger.warning(f"Invalid containment value: {value}")
        return None


def validate_folder(folder_path: Union[str, Path], create: bool = False) -> Path:
    """
    Validate that a folder exists and is accessible.

    Args:
        folder_path: Path to the folder to validate
        create: If True, create the folder if it doesn't exist

    Returns:
        The validated folder path as a Path object

    Raises:
        ValidationError: If the folder doesn't exist and create is False, or if it can't be created
    """
    path = Path(folder_path)

    if path.exists():
        if not path.is_dir():
            raise ValidationError(f"{path} exists but is not a directory")
    elif create:
        try:
            path.mkdir(parents=True)
            logger.info(f"Created directory: {path}")
        except Exception as e:
            raise ValidationError(f"Failed to create directory {path}: {e}")
    else:
        raise ValidationError(f"Directory does not exist: {path}")

    return path
