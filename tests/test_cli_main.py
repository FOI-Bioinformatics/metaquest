"""
Test CLI main module functionality.

Tests for the main CLI entry point, command registration, and argument parsing.
"""

import argparse
import logging
from unittest.mock import Mock, patch
from io import StringIO

import pytest

from metaquest.cli.main import register_all_commands, create_parser, main
from metaquest.cli.base import command_registry


class TestRegisterAllCommands:
    """Test command registration functionality."""

    def test_register_all_commands(self):
        """Test that all commands are registered successfully."""
        # Clear registry first
        command_registry._commands.clear()

        # Register all commands
        register_all_commands()

        # Verify commands are registered
        assert len(command_registry._commands) > 0

        # Check for key commands
        command_names = [cmd.name for cmd in command_registry._commands.values()]
        expected_commands = [
            "download_test_genome",
            "use_branchwater",
            "extract_branchwater_metadata",
            "parse_containment",
            "download_metadata",
            "parse_metadata",
            "check_metadata_attributes",
            "count_metadata",
            "single_sample",
            "plot_containment",
            "plot_metadata_counts",
            "download_sra",
            "assemble_datasets",
            "sra_info",
            "sra_download",
            "sra_stats",
            "sra_validate",
            "diversity_analysis",
            "interactive_plot",
            "validate_taxonomy",
            "taxonomic_summary",
        ]

        for expected_cmd in expected_commands:
            assert expected_cmd in command_names

    @staticmethod
    def _subcommand_choices(parser):
        """Return the set of accepted subcommand names (including aliases)."""
        action = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None))
        return set(action.choices)

    def test_snake_case_aliases_resolve(self):
        """The kebab-case intelligent commands also accept snake_case names."""
        choices = self._subcommand_choices(create_parser())
        for kebab, snake in (
            ("sra-download-intelligent", "sra_download_intelligent"),
            ("sra-profile-quality", "sra_profile_quality"),
            ("sra-dashboard", "sra_dashboard"),
            ("sra-compare", "sra_compare"),
        ):
            assert kebab in choices and snake in choices

    def test_check_metadata_attributes_is_registered(self):
        """check_metadata_attributes is a real command (README step 7)."""
        assert "check_metadata_attributes" in self._subcommand_choices(create_parser())

    def test_register_all_commands_idempotent(self):
        """Test that registering commands multiple times doesn't cause issues."""
        # Clear registry first
        command_registry._commands.clear()

        # Register commands twice
        register_all_commands()
        first_count = len(command_registry._commands)

        register_all_commands()
        second_count = len(command_registry._commands)

        # Should be the same count (commands should be replaced, not duplicated)
        assert first_count == second_count


class TestCreateParser:
    """Test argument parser creation."""

    def test_create_parser_basic(self):
        """Test basic parser creation."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "MetaQuest" in parser.description

    def test_create_parser_version_argument(self):
        """Test version argument is configured."""
        parser = create_parser()

        # Test version argument
        with pytest.raises(SystemExit):
            with patch("sys.stdout", new_callable=StringIO):
                parser.parse_args(["--version"])

    def test_create_parser_log_level_argument(self):
        """Test log level argument configuration."""
        parser = create_parser()

        # Test valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            args = parser.parse_args(["--log-level", level, "download_test_genome"])
            assert args.log_level == level

    def test_create_parser_invalid_log_level(self):
        """Test invalid log level raises error."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--log-level", "INVALID"])

    def test_create_parser_has_subcommands(self):
        """Test that subcommands are properly configured."""
        parser = create_parser()

        # Parse a known command
        args = parser.parse_args(["download_test_genome"])
        assert hasattr(args, "func")
        assert callable(args.func)

    def test_create_parser_no_arguments_shows_help(self):
        """Test that no arguments shows help."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestMainFunction:
    """Test main function execution."""

    @patch("metaquest.cli.main.setup_logging")
    @patch("metaquest.cli.main.create_parser")
    def test_main_successful_execution(self, mock_create_parser, mock_setup_logging):
        """Test successful command execution."""
        # Mock parser and args
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.log_level = "INFO"
        mock_args.func.return_value = 0

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        # Test main function
        result = main(["test_command"])

        assert result == 0
        mock_setup_logging.assert_called_once_with(level=logging.INFO)
        mock_args.func.assert_called_once_with(mock_args)

    @patch("metaquest.cli.main.setup_logging")
    @patch("metaquest.cli.main.create_parser")
    def test_main_command_failure(self, mock_create_parser, mock_setup_logging):
        """Test command execution failure."""
        # Mock parser and args with failing command
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.log_level = "ERROR"
        mock_args.func.return_value = 1

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        # Test main function
        result = main(["failing_command"])

        assert result == 1
        mock_setup_logging.assert_called_once_with(level=logging.ERROR)
        mock_args.func.assert_called_once_with(mock_args)

    @patch("metaquest.cli.main.logging.error")
    @patch("metaquest.cli.main.setup_logging")
    @patch("metaquest.cli.main.create_parser")
    def test_main_exception_handling(self, mock_create_parser, mock_setup_logging, mock_log_error):
        """Test exception handling in main function."""
        # Mock parser and args that raise exception
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.log_level = "DEBUG"
        mock_args.func.side_effect = Exception("Test error")

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        # Test main function
        result = main(["error_command"])

        assert result == 1
        mock_setup_logging.assert_called_once_with(level=logging.DEBUG)
        mock_log_error.assert_called_once_with("Exception: Test error")

    @patch("metaquest.cli.main.setup_logging")
    @patch("metaquest.cli.main.create_parser")
    def test_main_different_log_levels(self, mock_create_parser, mock_setup_logging):
        """Test main function with different log levels."""
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in log_levels:
            # Reset mocks
            mock_setup_logging.reset_mock()

            # Mock parser and args
            mock_parser = Mock()
            mock_args = Mock()
            mock_args.log_level = level
            mock_args.func.return_value = 0

            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            # Test main function
            main(["test_command"])

            expected_level = getattr(logging, level)
            mock_setup_logging.assert_called_once_with(level=expected_level)

    @patch("metaquest.cli.main.sys.argv", ["metaquest"])
    def test_main_no_args_uses_sys_argv(self):
        """Test that main() with no args uses sys.argv."""
        with patch("metaquest.cli.main.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.side_effect = SystemExit(2)  # Help message exit
            mock_create_parser.return_value = mock_parser

            with pytest.raises(SystemExit):
                main()

            # Should call parse_args with None (which uses sys.argv[1:])
            mock_parser.parse_args.assert_called_once_with(None)

    def test_main_with_explicit_args(self):
        """Test main() with explicit argument list."""
        with patch("metaquest.cli.main.create_parser") as mock_create_parser:
            with patch("metaquest.cli.main.setup_logging"):
                mock_parser = Mock()
                mock_args = Mock()
                mock_args.log_level = "INFO"
                mock_args.func.return_value = 0

                mock_parser.parse_args.return_value = mock_args
                mock_create_parser.return_value = mock_parser

                test_args = ["--log-level", "DEBUG", "download_test_genome"]
                result = main(test_args)

                assert result == 0
                mock_parser.parse_args.assert_called_once_with(test_args)


class TestMainIntegration:
    """Integration tests for main function with real components."""

    def test_main_integration_version(self):
        """Test version argument integration."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                main(["--version"])

            assert exc_info.value.code == 0
            output = mock_stdout.getvalue()
            assert "MetaQuest v" in output

    def test_main_integration_help(self):
        """Test help display integration."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                main(["--help"])

            assert exc_info.value.code == 0
            output = mock_stdout.getvalue()
            assert "MetaQuest" in output
            assert "usage:" in output

    def test_main_integration_invalid_command(self):
        """Test invalid command handling."""
        with patch("sys.stderr", new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                main(["invalid_command"])

            assert exc_info.value.code != 0

    @patch("metaquest.cli.main.setup_logging")
    def test_main_integration_valid_command_structure(self, mock_setup_logging):
        """Test that a valid command can be parsed (without executing)."""
        # This tests the full parser setup without actually running commands
        with patch("metaquest.cli.main.register_all_commands"):
            with patch.object(command_registry, "setup_parsers"):
                # Mock a simple command execution
                with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                    mock_args = Mock()
                    mock_args.log_level = "INFO"
                    mock_args.func.return_value = 0
                    mock_parse.return_value = mock_args

                    result = main(["download_test_genome"])

                    assert result == 0
                    mock_setup_logging.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
