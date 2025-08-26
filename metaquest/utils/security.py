"""
Security utilities for MetaQuest.

This module provides secure subprocess handling and input validation.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

from metaquest.core.exceptions import SecurityError
from metaquest.core.validation import validate_accession
from metaquest.core.constants import (
    ALLOWED_BIOINFORMATICS_TOOLS,
    MAX_SUBPROCESS_TIMEOUT,
    DANGEROUS_ENV_VARS,
    SRA_ACCESSION_PATTERN,
    UNSAFE_SHELL_CHARS,
)

logger = logging.getLogger(__name__)


class SecureSubprocess:
    """Secure subprocess wrapper with validation and sanitization."""

    # Get allowed tools from constants
    ALLOWED_EXECUTABLES = set(ALLOWED_BIOINFORMATICS_TOOLS.keys())

    # Get safe parameters from constants
    SAFE_PARAMETERS = {
        tool: config["safe_params"]
        for tool, config in ALLOWED_BIOINFORMATICS_TOOLS.items()
    }

    @staticmethod
    def validate_executable(executable: str) -> str:
        """
        Validate and sanitize executable name.

        Args:
            executable: The executable name to validate

        Returns:
            Validated executable name

        Raises:
            SecurityError: If executable is not allowed
        """
        if executable not in SecureSubprocess.ALLOWED_EXECUTABLES:
            raise SecurityError(f"Executable '{executable}' is not allowed")

        # Additional validation - ensure no path traversal
        if "/" in executable or "\\" in executable or ".." in executable:
            raise SecurityError(f"Invalid executable path: {executable}")

        return executable

    @staticmethod
    def validate_parameter(executable: str, param: str) -> str:
        """
        Validate parameter for given executable.

        Args:
            executable: The executable name
            param: The parameter to validate

        Returns:
            Validated parameter

        Raises:
            SecurityError: If parameter is not safe
        """
        if executable not in SecureSubprocess.SAFE_PARAMETERS:
            raise SecurityError(f"No parameter validation defined for {executable}")

        safe_params = SecureSubprocess.SAFE_PARAMETERS[executable]

        # Check if it's a flag parameter
        if param.startswith("-") and param not in safe_params:
            raise SecurityError(f"Parameter '{param}' not allowed for {executable}")

        # Basic sanitization - no shell metacharacters
        if any(char in param for char in UNSAFE_SHELL_CHARS):
            raise SecurityError(f"Parameter contains unsafe characters: {param}")

        return param

    @staticmethod
    def validate_path(path: Union[str, Path], allow_creation: bool = True) -> Path:
        """
        Validate and sanitize file/directory path.

        Args:
            path: The path to validate
            allow_creation: Whether to allow creation of non-existent paths

        Returns:
            Validated Path object

        Raises:
            SecurityError: If path is unsafe
        """
        path_obj = Path(path).resolve()

        # Check for directory traversal attempts
        try:
            path_obj.relative_to(Path.cwd())
        except ValueError:
            # Allow absolute paths within certain directories
            allowed_roots = [Path.cwd(), Path.home(), Path("/tmp")]
            if not any(str(path_obj).startswith(str(root)) for root in allowed_roots):
                raise SecurityError(f"Path outside allowed directories: {path}")

        # Check for suspicious path components
        unsafe_components = ["..", ".", "~"]
        if any(comp in unsafe_components for comp in path_obj.parts):
            # Allow single '.' and legitimate parent directory traversal
            if not (len(path_obj.parts) == 1 and path_obj.parts[0] == "."):
                if ".." in str(path) and not allow_creation:
                    raise SecurityError(f"Unsafe path component in: {path}")

        return path_obj

    @staticmethod
    def validate_accession_for_subprocess(accession: str) -> str:
        """
        Validate SRA accession for subprocess usage.

        Args:
            accession: The SRA accession to validate

        Returns:
            Validated accession

        Raises:
            SecurityError: If accession is invalid or unsafe
        """
        if not validate_accession(accession):
            raise SecurityError(f"Invalid SRA accession format: {accession}")

        # Additional security check - only alphanumeric characters allowed
        if not re.match(SRA_ACCESSION_PATTERN, accession):
            raise SecurityError(
                f"SRA accession contains invalid characters: {accession}"
            )

        return accession

    @classmethod
    def run_secure(
        cls,
        executable: str,
        args: List[str],
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """
        Run subprocess with security validations.

        Args:
            executable: The executable to run
            args: List of arguments
            cwd: Working directory
            env: Environment variables
            timeout: Timeout in seconds
            **kwargs: Additional subprocess.run arguments

        Returns:
            CompletedProcess result

        Raises:
            SecurityError: If any validation fails
        """
        # Validate executable
        safe_executable = cls.validate_executable(executable)

        # Build and validate command
        cmd = [safe_executable]

        i = 0
        while i < len(args):
            arg = args[i]

            # If it's a parameter flag, validate it
            if arg.startswith("-"):
                cls.validate_parameter(executable, arg)
                cmd.append(arg)

                # If this parameter takes a value, validate the value too
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    i += 1
                    value = args[i]

                    # Special validation for path arguments
                    if arg in ["-O", "-o", "--out-dir", "--temp", "-1", "-2"]:
                        value = str(cls.validate_path(value))
                    elif executable == "fasterq-dump" and i == 1:  # SRA accession
                        value = cls.validate_accession_for_subprocess(value)

                    cmd.append(value)
            else:
                # Standalone argument (like SRA accession)
                if executable == "fasterq-dump" and not arg.startswith("-"):
                    arg = cls.validate_accession_for_subprocess(arg)
                cmd.append(arg)

            i += 1

        # Validate working directory if provided
        if cwd:
            cwd = cls.validate_path(cwd)

        # Set safe environment
        safe_env = os.environ.copy() if env is None else env.copy()
        # Remove potentially dangerous environment variables
        for var in DANGEROUS_ENV_VARS:
            safe_env.pop(var, None)

        logger.debug(f"Running secure command: {' '.join(cmd)}")

        try:
            # Use secure defaults
            secure_kwargs = {
                "check": True,
                "capture_output": True,
                "text": True,
                "cwd": cwd,
                "env": safe_env,
                "timeout": timeout or MAX_SUBPROCESS_TIMEOUT,
                **kwargs,
            }

            return subprocess.run(cmd, **secure_kwargs)

        except subprocess.TimeoutExpired as e:
            raise SecurityError(f"Command timed out: {e}")
        except subprocess.CalledProcessError as e:
            # Re-raise as the original exception type for compatibility
            raise e
        except Exception as e:
            raise SecurityError(f"Subprocess execution failed: {e}")


def validate_file_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate a file path for security.

    Args:
        path: The file path to validate
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        SecurityError: If path is unsafe
    """
    return SecureSubprocess.validate_path(path, allow_creation=not must_exist)
