"""
Base classes for CLI command system.

This module provides the foundation for a modular command architecture.
"""

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.utils.security import SecureSubprocess
from metaquest.utils.termination import EXIT_INTERRUPTED, graceful_termination

if TYPE_CHECKING:
    from metaquest.data.registry import Registry
    from metaquest.store.layout import StorePaths


def read_accessions_file(path: Union[str, Path]) -> List[str]:
    """Accessions listed in ``path``, one per line, in file order.

    Surrounding whitespace is stripped; blank lines and lines starting with ``#`` are skipped.
    A file that does not exist or cannot be read raises ``DataAccessError`` naming it.
    """
    try:
        with open(path, "r") as handle:
            lines = handle.read().splitlines()
    except OSError as e:
        raise DataAccessError(f"Cannot read accessions file {path}: {e}") from e
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def accessions_from_args(accessions_file: Optional[str], accessions: Optional[Sequence[str]]) -> List[str]:
    """The accessions named by ``--accessions-file`` and repeated ``--accession`` flags together.

    File entries come first, then the flag values, each accession once in first-seen order.
    Empty when neither was given.
    """
    named = read_accessions_file(accessions_file) if accessions_file else []
    named += list(accessions or [])
    return list(dict.fromkeys(named))


def resolve_command_store(args: argparse.Namespace, registry: "Registry") -> Optional["StorePaths"]:
    """The shared data store (if any) for a command's ``--data-root`` and registry.

    ``getattr`` guards ``args.data_root`` so a namespace built without that attribute is not
    broken by it. A store that cannot be reached only costs the usage record, so it is
    logged and skipped rather than failing an analysis the project can run on its own files.
    """
    from metaquest.data import registry_blocks as rb
    from metaquest.store.resolve import resolve_optional_store

    return resolve_optional_store(getattr(args, "data_root", None), rb.store_block(registry).root)


class DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Append '(default: ...)' only when the option has a real default."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None or action.default is argparse.SUPPRESS:
            return action.help or ""
        return super()._get_help_string(action) or ""


def emit_error_json(message: str) -> None:
    """Write ``{"error": message}`` to stdout as one JSON document.

    For module-level helpers that have no command instance at hand. The message should also
    be logged by the caller when a human reader needs it; the log goes to stderr.
    """
    print(json.dumps({"error": message}, indent=2), file=sys.stdout)


class BaseCommand(ABC):
    """Base class for all CLI commands."""

    # False for a command that must keep the default signal behaviour; ``run`` then calls
    # ``execute`` directly.
    graceful_shutdown: bool = True

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

    @property
    def hidden(self) -> bool:
        """True for a command that parses but is left out of the main help (e.g. a renamed one)."""
        return False

    @abstractmethod
    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure the argument parser for this command."""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the command with parsed arguments."""
        pass

    def run(self, args: argparse.Namespace) -> int:
        """Run ``execute`` with SIGINT, SIGTERM and SIGHUP handled; the parser calls this, not ``execute``.

        The first signal (or Ctrl-C) raises ``KeyboardInterrupt`` inside ``execute``, so its
        ``finally`` blocks and ``with`` exits still write what the command has done; later
        signals are logged, not raised (see ``metaquest.utils.termination``). An interrupt that
        leaves ``execute`` is logged, running tools are stopped and 130 is returned. The
        ``Termination`` is available to ``execute`` as ``args._termination``; its ``stop`` is
        this run's stop token.

        On an interrupt only the tools started under that token are terminated
        (``terminate_children(stop=term.stop)``), so another command running in the same
        process keeps its own. A tool started without a token on the main thread needs no
        such call: ``run_secure`` kills its child when the interrupt reaches the waiting
        thread. The process-wide stopping flag is neither set nor cleared here, so no
        ``clear_stopping`` call is needed at the start of a run.
        """
        if not self.graceful_shutdown:
            return self.execute(args)
        with graceful_termination() as term:
            args._termination = term
            try:
                return self.execute(args)
            except KeyboardInterrupt:
                self.logger.error("Interrupted (%s)", term.cause)
                SecureSubprocess.terminate_children(stop=term.stop)
                return EXIT_INTERRUPTED

    # Output. stdout carries the command's result (tables, JSON); stderr carries logging.
    # These methods and ``emit_error_json`` are the only places in the package that write to
    # stdout (``scripts/check_no_print.sh`` enforces this).

    def emit(self, text: str = "") -> None:
        """Write one newline-terminated line of the command's result to stdout."""
        print(text, file=sys.stdout)

    def emit_raw(self, text: str) -> None:
        """Write ``text`` to stdout as is, without adding a newline (e.g. ``DataFrame.to_csv`` output)."""
        sys.stdout.write(text)

    def emit_json(self, payload: Any) -> None:
        """Write ``payload`` to stdout as exactly one JSON document (indent 2, trailing newline)."""
        print(json.dumps(payload, indent=2), file=sys.stdout)


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
            # Omit the `help` kwarg for the canonical name too: passing it makes argparse
            # print a second, flat listing of every command above the grouped epilog in
            # `metaquest.cli.main._commands_epilog`. `description` (set unconditionally in
            # `_add_parser`) still drives the per-command `metaquest COMMAND --help` text.
            self._add_parser(subparsers, command, command.name, None)
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
            formatter_class=DefaultsHelpFormatter,
            **kwargs,
        )
        command.configure_parser(subparser)
        subparser.set_defaults(func=command.run)


# Global registry instance
command_registry = CommandRegistry()
