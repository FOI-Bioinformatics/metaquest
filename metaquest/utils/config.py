"""
Configuration utilities for MetaQuest.

This module provides functions to manage application configuration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from metaquest.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "ncbi": {"email": "", "api_key": ""},
    "paths": {
        "genomes_folder": "genomes",
        "matches_folder": "matches",
        "metadata_folder": "metadata",
        "output_folder": "output",
    },
    "processing": {"threads": 4, "default_threshold": 0.5},
    "visualization": {"default_colors": "viridis", "default_format": "png"},
}

# Global configuration store
_config: Dict[str, Any] = {}


def _get_config_path() -> Path:
    """
    Get the path to the configuration file.

    Returns:
        Path to the configuration file
    """
    # Check for config path in environment variable
    env_config_path = os.environ.get("METAQUEST_CONFIG")
    if env_config_path:
        return Path(env_config_path)

    # Otherwise use default location
    home_dir = Path.home()
    return home_dir / ".metaquest" / "config.json"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to the configuration file (if None, use default)

    Returns:
        Loaded configuration dictionary

    Raises:
        ConfigurationError: If the configuration file cannot be loaded
    """
    global _config

    # Use default config path if not provided
    if config_path is None:
        config_path = _get_config_path()
    else:
        config_path = Path(config_path)

    # Start with default configuration
    config = DEFAULT_CONFIG.copy()

    # Try to load from file
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            # Merge with defaults (only override existing keys)
            _merge_config(config, loaded_config)
            logger.debug(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.warning(f"Failed to load configuration from {config_path}: {e}")
        logger.warning("Using default configuration")

    # Update global configuration
    _config = config
    return config


def save_config(
    config: Dict[str, Any], config_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save
        config_path: Path to the configuration file (if None, use default)

    Raises:
        ConfigurationError: If the configuration file cannot be saved
    """
    global _config

    # Use default config path if not provided
    if config_path is None:
        config_path = _get_config_path()
    else:
        config_path = Path(config_path)

    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.debug(f"Saved configuration to {config_path}")

        # Update global configuration
        _config = config
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}")


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.

    Returns:
        Current configuration dictionary
    """
    global _config

    # Load config if not already loaded
    if not _config:
        _config = load_config()

    return _config


def set_config(config: Dict[str, Any], save: bool = True) -> None:
    """
    Set the current configuration.

    Args:
        config: New configuration dictionary
        save: Whether to save the configuration to file

    Raises:
        ConfigurationError: If the configuration cannot be saved
    """
    global _config

    # Update global configuration
    _config = config

    # Save to file if requested
    if save:
        save_config(config)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a value from the configuration by key path.

    Args:
        key_path: Dot-separated path to the configuration value
        default: Default value to return if key not found

    Returns:
        Configuration value or default
    """
    config = get_config()
    keys = key_path.split(".")

    try:
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(key_path: str, value: Any, save: bool = True) -> None:
    """
    Set a value in the configuration by key path.

    Args:
        key_path: Dot-separated path to the configuration value
        value: Value to set
        save: Whether to save the configuration to file

    Raises:
        ConfigurationError: If the configuration cannot be saved
    """
    config = get_config()
    keys = key_path.split(".")

    try:
        # Navigate to the parent object
        target = config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        # Set the value
        target[keys[-1]] = value

        # Update and save
        set_config(config, save)
    except Exception as e:
        raise ConfigurationError(f"Failed to set configuration value {key_path}: {e}")


def _merge_config(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration dictionary
        overlay: Overlay configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            _merge_config(base[key], value)
        else:
            # Override or add values
            base[key] = value

    return base
