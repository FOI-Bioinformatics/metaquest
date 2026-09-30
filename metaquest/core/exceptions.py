"""
Custom exception classes for MetaQuest, and the exit code each one gives a command.

Exit codes: 0 success, 1 failure, 2 usage error, 3 configuration (the environment lacks
something a command needs), 4 a retryable failure (network, a lock wait that gave up), 130
interrupted. A batch script can resubmit on 4 and stop on 1 or 3.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes returned by ``metaquest``."""

    OK = 0
    FAILURE = 1
    USAGE = 2
    CONFIGURATION = 3
    TRANSIENT = 4
    INTERRUPTED = 130


class MetaQuestError(Exception):
    """Base exception for all MetaQuest errors."""

    exit_code: int = ExitCode.FAILURE


class ValidationError(MetaQuestError):
    """Exception raised for validation errors in input data."""


class FormatError(ValidationError):
    """Exception raised for errors related to file formats."""


class DataAccessError(MetaQuestError):
    """Exception raised for errors in data access operations."""


class TransientError(DataAccessError):
    """A data access failure that may succeed if the same command is run again later."""

    exit_code = ExitCode.TRANSIENT


class LockTimeoutError(TransientError):
    """Raised when a wait for a lock held by another live process reaches its time limit."""


class NetworkError(TransientError):
    """Raised when a remote service cannot be reached, times out, or answers 429 or 5xx."""


class ProcessingError(MetaQuestError):
    """Exception raised for errors during data processing."""


class VisualizationError(MetaQuestError):
    """Exception raised for errors during visualization generation."""


class PluginError(MetaQuestError):
    """Exception raised for errors related to plugins."""


class SecurityError(MetaQuestError):
    """Exception raised for security-related errors."""


class ConfigurationError(MetaQuestError):
    """Exception raised when the environment lacks something a command needs, such as an optional package."""

    exit_code = ExitCode.CONFIGURATION


def exit_code_for(error: BaseException) -> int:
    """The process exit code for ``error``: its class's code, 130 for an interrupt, else 1."""
    if isinstance(error, KeyboardInterrupt):
        return int(ExitCode.INTERRUPTED)
    if isinstance(error, MetaQuestError):
        return int(error.exit_code)
    return int(ExitCode.FAILURE)
