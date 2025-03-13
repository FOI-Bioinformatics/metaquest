"""
Custom exception classes for MetaQuest.
"""


class MetaQuestError(Exception):
    """Base exception for all MetaQuest errors."""


class ValidationError(MetaQuestError):
    """Exception raised for validation errors in input data."""


class FormatError(ValidationError):
    """Exception raised for errors related to file formats."""


class DataAccessError(MetaQuestError):
    """Exception raised for errors in data access operations."""


class ProcessingError(MetaQuestError):
    """Exception raised for errors during data processing."""


class VisualizationError(MetaQuestError):
    """Exception raised for errors during visualization generation."""


class PluginError(MetaQuestError):
    """Exception raised for errors related to plugins."""


class ConfigurationError(MetaQuestError):
    """Exception raised for errors in configuration."""
