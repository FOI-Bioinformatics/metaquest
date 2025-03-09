"""
Custom exception classes for MetaQuest.
"""


class MetaQuestError(Exception):
    """Base exception for all MetaQuest errors."""

    pass


class ValidationError(MetaQuestError):
    """Exception raised for validation errors in input data."""

    pass


class FormatError(ValidationError):
    """Exception raised for errors related to file formats."""

    pass


class DataAccessError(MetaQuestError):
    """Exception raised for errors in data access operations."""

    pass


class ProcessingError(MetaQuestError):
    """Exception raised for errors during data processing."""

    pass


class VisualizationError(MetaQuestError):
    """Exception raised for errors during visualization generation."""

    pass


class PluginError(MetaQuestError):
    """Exception raised for errors related to plugins."""

    pass


class ConfigurationError(MetaQuestError):
    """Exception raised for errors in configuration."""

    pass
