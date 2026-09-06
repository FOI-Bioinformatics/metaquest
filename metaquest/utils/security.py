"""
Security utilities for MetaQuest.

This module provides secure subprocess handling and input validation.
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Union

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

# Per-tool flags that never take a value; any other allowlisted flag for that
# tool consumes the following token as its value.
BOOLEAN_FLAGS: Dict[str, FrozenSet[str]] = {
    "fasterq-dump": frozenset(
        {"--progress", "--split-files", "--split-3", "--skip-technical", "--include-technical", "--force", "--version"}
    ),
    "prefetch": frozenset({"--progress", "--resume", "--version"}),
    "pigz": frozenset({"-f", "-k", "--version"}),
    "minimap2": frozenset({"-a", "--version"}),
    "samtools": frozenset({"-b", "-c", "--version"}),
    "megahit": frozenset({"--no-mercy", "--version"}),
}
# Kept for backward compatibility with any caller importing the old name.
FASTERQ_DUMP_BOOLEAN_FLAGS = BOOLEAN_FLAGS["fasterq-dump"]
# fasterq-dump flags whose value must be a non-negative integer.
FASTERQ_DUMP_INTEGER_FLAGS = frozenset({"--threads"})
# Flags (any tool) whose value is a filesystem path and must pass validate_path.
PATH_VALUE_FLAGS = frozenset({"-O", "-o", "--out-dir", "--temp", "-1", "-2", "-0", "-s"})
# Tools whose positional argument is either an SRA accession or a .sra file path.
SRA_POSITIONAL_TOOLS = frozenset({"fasterq-dump", "prefetch"})


class SecureSubprocess:
    """Secure subprocess wrapper with validation and sanitization."""

    # Get allowed tools from constants
    ALLOWED_EXECUTABLES = set(ALLOWED_BIOINFORMATICS_TOOLS.keys())

    # Get safe parameters from constants
    SAFE_PARAMETERS = {tool: config["safe_params"] for tool, config in ALLOWED_BIOINFORMATICS_TOOLS.items()}

    # Directories registered at runtime from user-supplied output or temp folders.
    _extra_roots: List[Path] = []

    @classmethod
    def allowed_roots(cls) -> List[Path]:
        """Directories under which validated paths may fall."""
        roots = [Path.cwd(), Path.home(), Path("/tmp"), Path(tempfile.gettempdir())]
        roots.extend(cls._extra_roots)
        return [root.resolve() for root in roots]

    @classmethod
    def add_allowed_root(cls, path: Union[str, Path]) -> Path:
        """Permit paths under a directory the user chose on the command line."""
        resolved = Path(path).resolve()
        if resolved not in cls._extra_roots:
            cls._extra_roots.append(resolved)
        return resolved

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

    @classmethod
    def validate_path(cls, path: Union[str, Path], allow_creation: bool = True) -> Path:
        """
        Validate and sanitize a file or directory path.

        A path is accepted when it lies under the working directory, the home
        directory, the system temp directory, or a root registered with
        ``add_allowed_root``. Parent-directory segments are rejected when the
        caller does not create the path.

        Raises:
            SecurityError: If the path is unsafe
        """
        path_obj = Path(path).resolve()

        if not any(path_obj == root or root in path_obj.parents for root in cls.allowed_roots()):
            raise SecurityError(f"Path outside allowed directories: {path}")

        if not allow_creation and ".." in Path(path).parts:
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
            raise SecurityError(f"SRA accession contains invalid characters: {accession}")

        return accession

    @classmethod
    def _build_validated_command(cls, executable: str, args: List[str]) -> List[str]:
        """Validate the executable and each argument, returning the command list to run.

        Flags are checked against the per-tool allowlist; path-valued flags and
        SRA accessions are validated/sanitized. Raises SecurityError on any
        disallowed component.
        """
        cmd = [cls.validate_executable(executable)]

        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith("-"):
                cls.validate_parameter(executable, arg)
                cmd.append(arg)

                takes_value = arg not in BOOLEAN_FLAGS.get(executable, frozenset())
                if takes_value and i + 1 < len(args) and not args[i + 1].startswith("-"):
                    i += 1
                    value = args[i]
                    if arg in PATH_VALUE_FLAGS:
                        value = str(cls.validate_path(value))
                    elif executable == "fasterq-dump" and arg in FASTERQ_DUMP_INTEGER_FLAGS:
                        if not value.isdigit():
                            raise SecurityError(f"Invalid integer value for {arg}: {value}")
                    cmd.append(value)
            else:
                # Positional argument: for fasterq-dump and prefetch this is either the SRA
                # accession or a path to an already-downloaded archive. NCBI serves some runs
                # only as .sralite, which fasterq-dump reads like a .sra file.
                if executable in SRA_POSITIONAL_TOOLS:
                    if arg.endswith((".sra", ".sralite")):
                        arg = str(cls.validate_path(arg))
                    else:
                        arg = cls.validate_accession_for_subprocess(arg)
                cmd.append(arg)

            i += 1

        return cmd

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
        cmd = cls._build_validated_command(executable, args)

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
