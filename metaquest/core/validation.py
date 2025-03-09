"""
Validation utilities for MetaQuest.

This module provides functions for validating input data and configurations.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from metaquest.core.exceptions import ValidationError, FormatError

# Required columns for different file formats
BRANCHWATER_REQUIRED_COLS = ['acc', 'containment']
MASTIFF_REQUIRED_COLS = ['SRA accession', 'containment']

logger = logging.getLogger(__name__)


def detect_file_format(file_path: Union[str, Path]) -> str:
    """
    Automatically detect the format of a CSV file (branchwater or mastiff).
    
    Args:
        file_path: Path to the CSV file to analyze
    
    Returns:
        Detected format: 'branchwater' or 'mastiff'
    
    Raises:
        FormatError: If the file format cannot be determined
    """
    try:
        with open(file_path, 'r') as f:
            # Read the header line
            header = f.readline().strip()
            
            # Check for branchwater format
            if 'acc' in header and 'containment' in header:
                return 'branchwater'
            
            # Check for mastiff format
            if 'SRA accession' in header and 'containment' in header:
                return 'mastiff'
            
            # If we can't determine the format
            raise FormatError(f"Could not determine file format for {file_path}. Header: {header}")
    except Exception as e:
        raise FormatError(f"Error reading file {file_path}: {str(e)}")


def validate_csv_file(file_path: Union[str, Path], file_format: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Validate a CSV file by checking required columns and format.
    
    Args:
        file_path: Path to the CSV file to validate
        file_format: The expected format ('branchwater' or 'mastiff')
                     If None, format will be automatically detected
    
    Returns:
        A tuple of (detected format, list of column headers)
    
    Raises:
        ValidationError: If the file does not meet validation requirements
    """
    try:
        # Auto-detect format if not specified
        detected_format = detect_file_format(file_path) if file_format is None else file_format
        
        # Determine required columns based on format
        if detected_format == 'branchwater':
            required_cols = BRANCHWATER_REQUIRED_COLS
        elif detected_format == 'mastiff':
            required_cols = MASTIFF_REQUIRED_COLS
        else:
            raise ValidationError(f"Unsupported file format: {detected_format}")
        
        # Read and validate headers
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                raise ValidationError(f"File {file_path} is empty")
            
            # Check for required columns
            missing_cols = [col for col in required_cols if col not in headers]
            if missing_cols:
                raise ValidationError(
                    f"Missing required columns in {file_path}: {', '.join(missing_cols)}\n"
                    f"Expected format: {detected_format}, Headers found: {', '.join(headers)}"
                )
            
            # Check for data rows
            try:
                next(reader)  # Try to read the first data row
            except StopIteration:
                raise ValidationError(f"File {file_path} contains headers but no data rows")
        
        return detected_format, headers
    
    except FormatError:
        # Re-raise FormatError as is
        raise
    except ValidationError:
        # Re-raise ValidationError as is
        raise
    except Exception as e:
        # Wrap other exceptions in ValidationError
        raise ValidationError(f"Error validating {file_path}: {str(e)}")


def validate_accession(accession: str) -> bool:
    """
    Validate if a string looks like a valid SRA accession.
    
    Args:
        accession: The accession string to validate
        
    Returns:
        True if the accession is valid, False otherwise
    """
    # Basic validation - SRA accessions follow specific patterns
    valid_prefixes = ('SRR', 'ERR', 'DRR')
    
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