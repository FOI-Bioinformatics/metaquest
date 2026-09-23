"""
File I/O utilities for MetaQuest.

This module provides abstract file operations to handle different file types and formats.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)


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


def is_hidden_name(name: str) -> bool:
    """True for a dotfile name, including the ``._<name>`` AppleDouble files macOS writes next to
    every file on a volume without native extended attributes (ExFAT, SMB, some NAS shares)."""
    return name.startswith(".")


def visible_files(directory: Union[str, Path], *patterns: str, dirs: bool = False) -> List[Path]:
    """Entries directly in ``directory`` matching any of ``patterns``, hidden names removed.

    Returns files (or directories when ``dirs`` is True), sorted by name, without duplicates.
    A directory that does not exist yields an empty list. Every folder listing in metaquest
    goes through here so that ``._*`` and ``.DS_Store`` never count as data.
    """
    base = Path(directory)
    if not base.is_dir():
        return []
    found = set()
    for pattern in patterns or ("*",):
        for candidate in base.glob(pattern):
            if is_hidden_name(candidate.name):
                continue
            try:
                keep = candidate.is_dir() if dirs else candidate.is_file()
            except OSError:
                continue
            if keep:
                found.add(candidate)
    return sorted(found)


def list_files(directory: Union[str, Path], pattern: str = "*", include_hidden: bool = False) -> List[Path]:
    """List the files in ``directory`` matching ``pattern``; hidden names are dropped unless asked for."""
    try:
        directory = Path(directory)
        if include_hidden:
            return list(directory.glob(pattern))
        return visible_files(directory, pattern)
    except Exception as e:
        logger.warning(f"Error listing files in {directory} with pattern {pattern}: {e}")
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
