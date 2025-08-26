"""
Base classes for CLI command system.

This module provides the foundation for a modular command architecture.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseCommand(ABC):
    """Base class for all CLI commands."""

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
        """Setup subparsers for all registered commands."""
        subparsers = main_parser.add_subparsers(
            title="commands",
            description="valid commands",
            help="additional help",
            dest="command",
        )
        subparsers.required = True

        for command in self._commands.values():
            subparser = subparsers.add_parser(command.name, help=command.help)
            command.configure_parser(subparser)
            subparser.set_defaults(func=command.execute)


# Global registry instance
command_registry = CommandRegistry()
