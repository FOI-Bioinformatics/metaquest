"""
File I/O utilities for MetaQuest.

This module provides abstract file operations to handle different file types and formats.
"""

import json
import csv
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, TypeVar

import pandas as pd

from metaquest.core.exceptions import DataAccessError
from metaquest.core.validation import validate_folder

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar("T")


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        Path object for the directory

    Raises:
        DataAccessError: If the directory cannot be created
    """
    try:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    except Exception as e:
        raise DataAccessError(f"Failed to create directory {path}: {e}")


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List all files in a directory matching a pattern.

    Args:
        directory: Path to the directory
        pattern: Glob pattern to match files

    Returns:
        List of matching file paths
    """
    try:
        directory = Path(directory)
        return list(directory.glob(pattern))
    except Exception as e:
        logger.warning(
            f"Error listing files in {directory} with pattern {pattern}: {e}"
        )
        return []


def copy_file(source: Union[str, Path], destination: Union[str, Path]) -> Path:
    """
    Copy a file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path

    Returns:
        Path to the destination file

    Raises:
        DataAccessError: If the file cannot be copied
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)

        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        return Path(shutil.copy2(source_path, dest_path))
    except Exception as e:
        raise DataAccessError(f"Failed to copy {source} to {destination}: {e}")


def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a CSV file to a pandas DataFrame.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        Pandas DataFrame

    Raises:
        DataAccessError: If the file cannot be read
    """
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        raise DataAccessError(f"Failed to read CSV file {file_path}: {e}")


def write_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    Write a pandas DataFrame to a CSV file.

    Args:
        df: Pandas DataFrame to write
        file_path: Path to the output CSV file
        **kwargs: Additional arguments to pass to df.to_csv

    Raises:
        DataAccessError: If the file cannot be written
    """
    try:
        # Ensure directory exists
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, **kwargs)
    except Exception as e:
        raise DataAccessError(f"Failed to write CSV file {file_path}: {e}")


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a JSON file to a dictionary.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with the JSON data

    Raises:
        DataAccessError: If the file cannot be read
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise DataAccessError(f"Failed to read JSON file {file_path}: {e}")


def write_json(
    data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2
) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        data: Dictionary to write
        file_path: Path to the output JSON file
        indent: Number of spaces for indentation

    Raises:
        DataAccessError: If the file cannot be written
    """
    try:
        # Ensure directory exists
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise DataAccessError(f"Failed to write JSON file {file_path}: {e}")


def process_files_in_directory(
    directory: Union[str, Path],
    process_func: Callable[[Path], T],
    pattern: str = "*.csv",
    raise_errors: bool = False,
) -> Dict[str, T]:
    """
    Process all files in a directory using a function.

    Args:
        directory: Directory containing files to process
        process_func: Function to apply to each file
        pattern: Glob pattern to match files
        raise_errors: Whether to raise exceptions or log them

    Returns:
        Dictionary mapping filenames to processing results

    Raises:
        DataAccessError: If raise_errors is True and an error occurs
    """
    results = {}
    directory_path = Path(directory)

    try:
        files = list_files(directory_path, pattern)

        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory_path}")
            return results

        for file_path in files:
            try:
                results[file_path.name] = process_func(file_path)
            except Exception as e:
                if raise_errors:
                    raise DataAccessError(f"Error processing {file_path}: {e}")
                logger.error(f"Error processing {file_path}: {e}")

        return results

    except Exception as e:
        if raise_errors:
            raise DataAccessError(f"Error processing files in {directory_path}: {e}")
        logger.error(f"Error processing files in {directory_path}: {e}")
        return results
