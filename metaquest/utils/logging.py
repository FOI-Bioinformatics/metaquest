"""
Logging utilities for MetaQuest.

This module provides functions to set up and configure logging.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Path to log file (if None, log to stderr only)
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress verbose logging from some libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Log the setup
    logging.debug("Logging configured")