"""
Base classes for CLI command system.

This module provides the foundation for a modular command architecture.
"""

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseCommand(ABC):
    """Base class for all CLI commands."""

    def __init__(self):
        """Initialize the command with a logger."""
        self.logger = logging.getLogger(self.__class__.__module__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the command name."""
        pass

    @property
    @abstractmethod
    def help(self) -> str:
        """Return the help text for this command."""
        pass

    @property
    def aliases(self) -> List[str]:
        """Return alternate names this command also responds to (default: none)."""
        return []

    @property
    def group(self) -> str:
        """Pipeline step the command belongs to, used to group the main help listing."""
        return "Other"

    @abstractmethod
    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure the argument parser for this command."""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the command with parsed arguments."""
        pass


class CommandRegistry:
    """Registry for managing CLI commands."""

    def __init__(self):
        self._commands: Dict[str, BaseCommand] = {}

    def register(self, command: BaseCommand) -> None:
        """Register a command in the registry."""
        self._commands[command.name] = command

    def get_command(self, name: str) -> Optional[BaseCommand]:
        """Get a command by name."""
        return self._commands.get(name)

    def get_all_commands(self) -> Dict[str, BaseCommand]:
        """Get all registered commands."""
        return self._commands.copy()

    def setup_parsers(self, main_parser: argparse.ArgumentParser) -> None:
        """Add one subparser per command plus a hidden subparser per alias."""
        subparsers = main_parser.add_subparsers(
            title="commands",
            dest="command",
            metavar="COMMAND",
            help="Run 'metaquest COMMAND --help' for the options of one command",
        )
        subparsers.required = True

        for command in self._commands.values():
            self._add_parser(subparsers, command, command.name, command.help)
            for alias in command.aliases:
                # Omit the `help` kwarg entirely rather than passing argparse.SUPPRESS: on
                # some Python versions SUPPRESS is rendered literally as "==SUPPRESS==" for
                # subparser choices instead of being hidden. Not passing `help` at all means
                # argparse never records a choices entry for this alias, so it is left out
                # of the listing while still parsing normally.
                self._add_parser(subparsers, command, alias, None)

    @staticmethod
    def _add_parser(subparsers, command: BaseCommand, name: str, help_text: Optional[str]) -> None:
        kwargs = {} if help_text is None else {"help": help_text}
        subparser = subparsers.add_parser(
            name,
            description=command.help,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            **kwargs,
        )
        command.configure_parser(subparser)
        subparser.set_defaults(func=command.execute)


# Global registry instance
command_registry = CommandRegistry()
