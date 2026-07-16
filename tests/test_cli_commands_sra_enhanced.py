"""
Test CLI enhanced SRA commands functionality.

Tests for enhanced SRA command classes including info, download, stats, and validation,
focusing on argument parsing, validation, and proper delegation with mocked dependencies.
"""

import argparse
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys

import pytest

# Store original modules before mocking
_original_sra_enhanced = sys.modules.get("metaquest.data.sra_enhanced")
_original_sra_metadata = sys.modules.get("metaquest.data.sra_metadata")

# Mock dependencies to avoid import issues
mock_sra_enhanced = Mock()
mock_sra_enhanced.EnhancedSRADownloader = MagicMock()
mock_sra_enhanced.verify_sra_tools = Mock(return_value=True)
mock_sra_enhanced.estimate_download_time = Mock(return_value=2.5)
mock_sra_enhanced.create_download_report = Mock()

mock_sra_metadata = Mock()
mock_sra_metadata.save_metadata_report = Mock()
mock_sra_metadata.generate_statistics_report = Mock()

sys.modules["metaquest.data.sra_enhanced"] = mock_sra_enhanced
sys.modules["metaquest.data.sra_metadata"] = mock_sra_metadata

from metaquest.cli.commands.sra_enhanced import (  # noqa: E402
    SRAInfoCommand,
    SRADownloadEnhancedCommand,
    SRAStatsCommand,
    SRAValidateCommand,
)

# Restore original modules immediately after importing the commands under test.
# This prevents mock pollution of sys.modules that would affect other test files.
if _original_sra_enhanced is not None:
    sys.modules["metaquest.data.sra_enhanced"] = _original_sra_enhanced
else:
    sys.modules.pop("metaquest.data.sra_enhanced", None)

if _original_sra_metadata is not None:
    sys.modules["metaquest.data.sra_metadata"] = _original_sra_metadata
else:
    sys.modules.pop("metaquest.data.sra_metadata", None)


class TestSRAInfoCommand:
    """Test SRAInfoCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = SRAInfoCommand()
        assert command.name == "sra_info"
        assert "information" in command.help.lower()
        assert "sra" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = SRAInfoCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test required arguments
        args = parser.parse_args(["--accessions-file", "test.txt", "--email", "test@example.com"])
        assert args.accessions_file == "test.txt"
        assert args.email == "test@example.com"

        # Test optional arguments with defaults
        assert args.output_report == "sra_info_report.csv"
        assert args.bandwidth_mbps == 100.0

    def test_configure_parser_with_optional_args(self):
        """Test parser with optional arguments."""
        command = SRAInfoCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--accessions-file",
                "test.txt",
                "--email",
                "test@example.com",
                "--api-key",
                "mykey123",
                "--output-report",
                "custom_report.csv",
                "--bandwidth-mbps",
                "250.5",
            ]
        )

        assert args.api_key == "mykey123"
        assert args.output_report == "custom_report.csv"
        assert args.bandwidth_mbps == 250.5

    @patch("builtins.print")
    def test_execute_success(self, mock_print, tmp_path):
        """Test successful execution."""
        command = SRAInfoCommand()
        args = argparse.Namespace(
            accessions_file="test_accessions.txt",
            email="test@example.com",
            api_key="test_key",
            output_report="test_report.csv",
            bandwidth_mbps=100.0,
        )

        # Mock the downloader and its methods
        mock_downloader = Mock()
        mock_metadata = {
            "SRR123456": Mock(platform="ILLUMINA", layout="PAIRED", size_mb=1024),
            "SRR789012": Mock(platform="ILLUMINA", layout="SINGLE", size_mb=512),
        }
        mock_tech_counts = {"RNA-Seq": 2}
        mock_total_size = 1.5

        mock_downloader.preview_downloads.return_value = (mock_metadata, mock_tech_counts, mock_total_size)

        # Properly configure the mock
        mock_sra_enhanced.EnhancedSRADownloader.return_value = mock_downloader

        # Mock file reading
        mock_file_content = "SRR123456\nSRR789012\n"
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = command.execute(args)

        assert result == 0
        mock_sra_enhanced.EnhancedSRADownloader.assert_called_once_with("test@example.com", "test_key")
        mock_downloader.preview_downloads.assert_called_once()
        mock_sra_metadata.save_metadata_report.assert_called_once()

    @patch("builtins.print")
    def test_execute_no_accessions(self, mock_print):
        """Test execution with empty accessions file."""
        command = SRAInfoCommand()
        args = argparse.Namespace(
            accessions_file="empty.txt",
            email="test@example.com",
            api_key=None,
            output_report="report.csv",
            bandwidth_mbps=100.0,
        )

        with patch("builtins.open", mock_open(read_data="")):
            result = command.execute(args)

        assert result == 1
        mock_print.assert_called_with("No accessions found in file")

    @patch("builtins.print")
    def test_execute_no_metadata(self, mock_print):
        """Test execution when no metadata can be fetched."""
        command = SRAInfoCommand()
        args = argparse.Namespace(
            accessions_file="test.txt",
            email="test@example.com",
            api_key=None,
            output_report="report.csv",
            bandwidth_mbps=100.0,
        )

        mock_downloader = Mock()
        mock_downloader.preview_downloads.return_value = ({}, {}, 0.0)

        # Properly configure the mock
        mock_sra_enhanced.EnhancedSRADownloader.return_value = mock_downloader

        with patch("builtins.open", mock_open(read_data="SRR123456\n")):
            result = command.execute(args)

        assert result == 1
        # Check that the specific error message was printed (might not be the last call)
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("Could not fetch metadata" in call for call in calls)

    @patch("metaquest.cli.commands.sra_enhanced.logger")
    @patch("builtins.open", side_effect=FileNotFoundError())
    def test_execute_file_error(self, mock_open, mock_logger):
        """Test execution with file reading error."""
        command = SRAInfoCommand()
        args = argparse.Namespace(
            accessions_file="nonexistent.txt",
            email="test@example.com",
            api_key=None,
            output_report="report.csv",
            bandwidth_mbps=100.0,
        )

        result = command.execute(args)

        assert result == 1
        mock_logger.error.assert_called_once()


class TestSRADownloadEnhancedCommand:
    """Test SRADownloadEnhancedCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = SRADownloadEnhancedCommand()
        assert command.name == "sra_download"
        assert "download" in command.help.lower()
        assert "enhanced" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = SRADownloadEnhancedCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(["--accessions-file", "test.txt", "--email", "test@example.com"])

        assert args.accessions_file == "test.txt"
        assert args.email == "test@example.com"
        assert args.fastq_folder == "fastq"
        assert args.num_threads == 4
        assert args.max_workers == 4
        assert not args.dry_run
        assert not args.force
        assert not args.verify_tools

    def test_configure_parser_all_options(self):
        """Test parser with all options."""
        command = SRADownloadEnhancedCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--accessions-file",
                "test.txt",
                "--email",
                "test@example.com",
                "--fastq-folder",
                "custom_fastq",
                "--api-key",
                "mykey",
                "--max-downloads",
                "100",
                "--num-threads",
                "8",
                "--max-workers",
                "2",
                "--dry-run",
                "--force",
                "--temp-folder",
                "/tmp/sra",
                "--blacklist",
                "bad1.txt",
                "bad2.txt",
                "--verify-tools",
                "--report-file",
                "custom_report.csv",
            ]
        )

        assert args.fastq_folder == "custom_fastq"
        assert args.api_key == "mykey"
        assert args.max_downloads == 100
        assert args.num_threads == 8
        assert args.max_workers == 2
        assert args.dry_run
        assert args.force
        assert args.temp_folder == "/tmp/sra"
        assert args.blacklist == ["bad1.txt", "bad2.txt"]
        assert args.verify_tools
        assert args.report_file == "custom_report.csv"

    def test_verify_tools_success(self):
        """Test successful tools verification."""
        command = SRADownloadEnhancedCommand()
        args = argparse.Namespace(verify_tools=True)

        mock_sra_enhanced.verify_sra_tools.return_value = True

        with patch("builtins.print"):
            result = command._verify_tools(args)

        assert result is True
        mock_sra_enhanced.verify_sra_tools.assert_called_once()

    def test_verify_tools_failure(self):
        """Test failed tools verification."""
        command = SRADownloadEnhancedCommand()
        args = argparse.Namespace(verify_tools=True)

        mock_sra_enhanced.verify_sra_tools.return_value = False

        with patch("builtins.print"):
            result = command._verify_tools(args)

        assert result is False

    def test_verify_tools_skip(self):
        """Test skipping tools verification."""
        command = SRADownloadEnhancedCommand()
        args = argparse.Namespace(verify_tools=False)

        result = command._verify_tools(args)

        assert result is True

    def test_read_accessions_success(self):
        """Test successful accessions reading."""
        command = SRADownloadEnhancedCommand()

        with patch("builtins.open", mock_open(read_data="SRR123\nSRR456\n")):
            result = command._read_accessions("test.txt")

        assert result == ["SRR123", "SRR456"]

    @patch("builtins.print")
    def test_read_accessions_empty(self, mock_print):
        """Test reading empty accessions file."""
        command = SRADownloadEnhancedCommand()

        with patch("builtins.open", mock_open(read_data="")):
            result = command._read_accessions("empty.txt")

        assert result is None
        mock_print.assert_called_with("No accessions found in file")

    def test_read_blacklist_success(self, tmp_path):
        """Test successful blacklist reading."""
        command = SRADownloadEnhancedCommand()

        # Create test blacklist files
        blacklist1 = tmp_path / "blacklist1.txt"
        blacklist1.write_text("SRR999\nSRR888\n")
        blacklist2 = tmp_path / "blacklist2.txt"
        blacklist2.write_text("SRR777\n")

        blacklist_files = [str(blacklist1), str(blacklist2)]

        with patch("metaquest.cli.commands.sra_enhanced.logger"):
            result = command._read_blacklist(blacklist_files)

        assert result == {"SRR999", "SRR888", "SRR777"}

    def test_read_blacklist_empty(self):
        """Test reading empty blacklist."""
        command = SRADownloadEnhancedCommand()

        result = command._read_blacklist(None)

        assert result == set()

    @patch("builtins.print")
    def test_handle_dry_run(self, mock_print):
        """Test dry run handling."""
        command = SRADownloadEnhancedCommand()

        mock_downloader = Mock()
        mock_downloader.preview_downloads.return_value = ({"SRR123": Mock()}, {"RNA-Seq": 1}, 5.0)

        command._handle_dry_run(mock_downloader, ["SRR123"])

        mock_downloader.preview_downloads.assert_called_once_with(["SRR123"])

    @patch("builtins.print")
    def test_print_results(self, mock_print):
        """Test results printing."""
        command = SRADownloadEnhancedCommand()

        results = {
            "total": 10,
            "successful": 8,
            "failed": 2,
            "technology_summary": {"RNA-Seq": 5, "ChIP-Seq": 3},
            "failed_accessions": ["SRR999"],
        }

        command._print_results(results)

        # Check that print was called multiple times
        assert mock_print.call_count >= 5

    @patch("builtins.print")
    def test_execute_dry_run(self, mock_print):
        """Test execution in dry run mode."""
        command = SRADownloadEnhancedCommand()
        args = argparse.Namespace(
            verify_tools=False,
            accessions_file="test.txt",
            blacklist=None,
            email="test@example.com",
            api_key=None,
            num_threads=4,
            max_workers=4,
            temp_folder=None,
            dry_run=True,
        )

        mock_downloader = Mock()
        mock_downloader.preview_downloads.return_value = ({}, {}, 0.0)

        # Properly configure the mock
        mock_sra_enhanced.EnhancedSRADownloader.return_value = mock_downloader

        with patch("builtins.open", mock_open(read_data="SRR123\nSRR456\n")):
            result = command.execute(args)

        assert result == 0
        mock_downloader.preview_downloads.assert_called_once()

    @patch("builtins.print")
    def test_execute_download_success(self, mock_print, tmp_path):
        """Test successful download execution."""
        command = SRADownloadEnhancedCommand()
        args = argparse.Namespace(
            verify_tools=False,
            accessions_file="test.txt",
            blacklist=None,
            email="test@example.com",
            api_key=None,
            num_threads=4,
            max_workers=4,
            temp_folder=None,
            dry_run=False,
            fastq_folder="fastq",
            force=False,
            max_downloads=None,
            report_file="report.csv",
        )

        mock_downloader = Mock()
        mock_results = {"total": 1, "successful": 1, "failed": 0, "technology_summary": {}, "failed_accessions": []}
        mock_downloader.download_batch_enhanced.return_value = mock_results

        # Properly configure the mock
        mock_sra_enhanced.EnhancedSRADownloader.return_value = mock_downloader

        with patch("builtins.open", mock_open(read_data="SRR123\n")):
            result = command.execute(args)

        assert result == 0
        mock_downloader.download_batch_enhanced.assert_called_once()
        mock_sra_enhanced.create_download_report.assert_called_once()

    @patch("builtins.print")
    def test_execute_download_with_failures(self, mock_print, tmp_path):
        """Test download execution with some failures."""
        command = SRADownloadEnhancedCommand()
        args = argparse.Namespace(
            verify_tools=False,
            accessions_file="test.txt",
            blacklist=None,
            email="test@example.com",
            api_key=None,
            num_threads=4,
            max_workers=4,
            temp_folder=None,
            dry_run=False,
            fastq_folder="fastq",
            force=False,
            max_downloads=None,
            report_file="report.csv",
        )

        mock_downloader = Mock()
        mock_results = {
            "total": 2,
            "successful": 1,
            "failed": 1,
            "technology_summary": {},
            "failed_accessions": ["SRR999"],
        }
        mock_downloader.download_batch_enhanced.return_value = mock_results

        # Properly configure the mock
        mock_sra_enhanced.EnhancedSRADownloader.return_value = mock_downloader

        with patch("builtins.open", mock_open(read_data="SRR123\n")):
            with patch("pathlib.Path") as mock_path_class:
                mock_path_instance = Mock()
                mock_failed_file = Mock()
                mock_failed_file.__enter__ = Mock(return_value=Mock())
                mock_failed_file.__exit__ = Mock(return_value=None)
                mock_path_instance.__truediv__ = Mock(return_value=mock_failed_file)
                mock_path_class.return_value = mock_path_instance

                result = command.execute(args)

        assert result == 1  # Should return 1 when there are failures

    @patch("metaquest.cli.commands.sra_enhanced.logger")
    def test_execute_exception_handling(self, mock_logger):
        """Test exception handling during execution."""
        command = SRADownloadEnhancedCommand()
        args = argparse.Namespace(verify_tools=False)

        # Mock _read_accessions to raise an exception
        with patch.object(command, "_read_accessions", side_effect=Exception("Test error")):
            result = command.execute(args)

        assert result == 1
        mock_logger.error.assert_called_once()


class TestSRAStatsCommand:
    """Test SRAStatsCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = SRAStatsCommand()
        assert command.name == "sra_stats"
        assert "statistics" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = SRAStatsCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args([])
        assert args.fastq_folder == "fastq"
        assert args.output_report == "sra_statistics.csv"
        assert args.accessions is None

    def test_configure_parser_with_options(self):
        """Test parser with all options."""
        command = SRAStatsCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--fastq-folder",
                "custom_fastq",
                "--output-report",
                "custom_stats.csv",
                "--accessions",
                "SRR123",
                "SRR456",
            ]
        )

        assert args.fastq_folder == "custom_fastq"
        assert args.output_report == "custom_stats.csv"
        assert args.accessions == ["SRR123", "SRR456"]

    @patch("builtins.print")
    def test_execute_success(self, mock_print, tmp_path):
        """Test successful execution."""
        command = SRAStatsCommand()

        # Create test fastq folder
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()

        args = argparse.Namespace(fastq_folder=str(fastq_folder), output_report="stats.csv", accessions=None)

        result = command.execute(args)

        assert result == 0
        mock_sra_metadata.generate_statistics_report.assert_called_once_with(fastq_folder, "stats.csv")

    @patch("builtins.print")
    def test_execute_folder_not_exists(self, mock_print):
        """Test execution when fastq folder doesn't exist."""
        command = SRAStatsCommand()
        args = argparse.Namespace(fastq_folder="/nonexistent/folder", output_report="stats.csv", accessions=None)

        result = command.execute(args)

        assert result == 1
        mock_print.assert_called_with("FASTQ folder /nonexistent/folder does not exist")

    @patch("metaquest.cli.commands.sra_enhanced.logger")
    def test_execute_exception_handling(self, mock_logger, tmp_path):
        """Test exception handling during execution."""
        command = SRAStatsCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()

        args = argparse.Namespace(fastq_folder=str(fastq_folder), output_report="stats.csv", accessions=None)

        # Mock generate_statistics_report to raise an exception
        mock_sra_metadata.generate_statistics_report.side_effect = Exception("Test error")

        result = command.execute(args)

        assert result == 1
        mock_logger.error.assert_called_once()


class TestSRAValidateCommand:
    """Test SRAValidateCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = SRAValidateCommand()
        assert command.name == "sra_validate"
        assert "validate" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = SRAValidateCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args([])
        assert args.fastq_folder == "fastq"
        assert args.accessions is None
        assert not args.check_pairs

    def test_configure_parser_with_options(self):
        """Test parser with all options."""
        command = SRAValidateCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            ["--fastq-folder", "custom_fastq", "--accessions", "SRR123", "SRR456", "--check-pairs"]
        )

        assert args.fastq_folder == "custom_fastq"
        assert args.accessions == ["SRR123", "SRR456"]
        assert args.check_pairs

    def test_find_accession_dirs_all(self, tmp_path):
        """Test finding all accession directories."""
        command = SRAValidateCommand()

        # Create test directories
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        (fastq_folder / "SRR123").mkdir()
        (fastq_folder / "SRR456").mkdir()
        (fastq_folder / "not_accession.txt").touch()  # Should be ignored

        result = command._find_accession_dirs(fastq_folder)

        dir_names = {d.name for d in result}
        assert dir_names == {"SRR123", "SRR456"}

    def test_find_accession_dirs_specific(self, tmp_path):
        """Test finding specific accession directories."""
        command = SRAValidateCommand()

        # Create test directories
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        (fastq_folder / "SRR123").mkdir()
        (fastq_folder / "SRR456").mkdir()
        (fastq_folder / "SRR789").mkdir()

        result = command._find_accession_dirs(fastq_folder, ["SRR123", "SRR789"])

        dir_names = {d.name for d in result}
        assert dir_names == {"SRR123", "SRR789"}

    @patch("builtins.print")
    def test_validate_directory_no_files(self, mock_print, tmp_path):
        """Test validation of directory with no FASTQ files."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        result = command._validate_directory(acc_dir)

        assert result["accession"] == "SRR123"
        assert result["status"] == "FAILED"
        assert "No FASTQ files found" in result["issues"]
        assert result["num_files"] == 0

    @patch("builtins.print")
    def test_validate_directory_success(self, mock_print, tmp_path):
        """Test successful directory validation."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        # Create test FASTQ file with valid content
        fastq_file = acc_dir / "test.fastq"
        fastq_file.write_text("@read1\nACGT\n+\n!!!!\n")

        # Mock SeqIO import within the method
        def mock_import(name, *args, **kwargs):
            if name == "Bio":
                mock_bio = Mock()
                mock_bio.SeqIO = Mock()
                mock_bio.SeqIO.parse.return_value = [Mock()]  # Mock valid records
                return mock_bio
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = command._validate_directory(acc_dir)

        assert result["accession"] == "SRR123"
        assert result["status"] == "PASSED"
        assert result["issues"] == "None"
        assert result["num_files"] == 1

    @patch("builtins.print")
    def test_validate_directory_empty_files(self, mock_print, tmp_path):
        """Test validation with empty files."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        # Create empty FASTQ file
        fastq_file = acc_dir / "test.fastq"
        fastq_file.touch()

        result = command._validate_directory(acc_dir)

        assert result["status"] == "FAILED"
        assert "Empty file: test.fastq" in result["issues"]

    @patch("builtins.print")
    def test_validate_directory_paired_end_mismatch(self, mock_print, tmp_path):
        """Test validation with mismatched paired-end files."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        # Create mismatched paired-end files
        (acc_dir / "test_R1.fastq").write_text("content")
        (acc_dir / "test_R2.fastq").write_text("content")
        (acc_dir / "test2_R1.fastq").write_text("content")
        # Missing test2_R2.fastq

        result = command._validate_directory(acc_dir, check_pairs=True)

        assert result["status"] == "FAILED"
        assert "Mismatched paired-end files" in result["issues"]

    @patch("builtins.print")
    def test_validate_directory_format_error(self, mock_print, tmp_path):
        """Test validation with FASTQ format errors."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        fastq_file = acc_dir / "test.fastq"
        fastq_file.write_text("invalid fastq content")

        # Mock SeqIO to raise an exception
        def mock_import(name, *args, **kwargs):
            if name == "Bio":
                mock_bio = Mock()
                mock_bio.SeqIO = Mock()
                mock_bio.SeqIO.parse.side_effect = Exception("Format error")
                return mock_bio
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = command._validate_directory(acc_dir)

        assert result["status"] == "FAILED"
        assert "FASTQ format error" in result["issues"]

    @patch("builtins.print")
    def test_print_validation_results_success(self, mock_print):
        """Test printing successful validation results."""
        command = SRAValidateCommand()

        validation_results = [
            {"accession": "SRR123", "status": "PASSED", "issues": "None"},
            {"accession": "SRR456", "status": "PASSED", "issues": "None"},
        ]

        result = command._print_validation_results(validation_results)

        assert result is True

    @patch("builtins.print")
    def test_print_validation_results_with_failures(self, mock_print):
        """Test printing validation results with failures."""
        command = SRAValidateCommand()

        validation_results = [
            {"accession": "SRR123", "status": "PASSED", "issues": "None"},
            {"accession": "SRR456", "status": "FAILED", "issues": "Empty files"},
        ]

        result = command._print_validation_results(validation_results)

        assert result is False

    @patch("builtins.print")
    def test_execute_success(self, mock_print, tmp_path):
        """Test successful execution."""
        command = SRAValidateCommand()

        # Create test structure
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.mkdir()
        (acc_dir / "test.fastq").write_text("content")

        args = argparse.Namespace(fastq_folder=str(fastq_folder), accessions=None, check_pairs=False)

        # Mock _validate_directory to return success
        with patch.object(command, "_validate_directory") as mock_validate:
            mock_validate.return_value = {"accession": "SRR123", "status": "PASSED", "issues": "None"}

            result = command.execute(args)

        assert result == 0
        mock_validate.assert_called_once()

    @patch("builtins.print")
    def test_execute_folder_not_exists(self, mock_print):
        """Test execution when fastq folder doesn't exist."""
        command = SRAValidateCommand()
        args = argparse.Namespace(fastq_folder="/nonexistent/folder", accessions=None, check_pairs=False)

        result = command.execute(args)

        assert result == 1
        mock_print.assert_called_with("FASTQ folder /nonexistent/folder does not exist")

    @patch("builtins.print")
    def test_execute_no_accession_dirs(self, mock_print, tmp_path):
        """Test execution when no accession directories found."""
        command = SRAValidateCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()

        args = argparse.Namespace(fastq_folder=str(fastq_folder), accessions=None, check_pairs=False)

        result = command.execute(args)

        assert result == 1
        mock_print.assert_called_with("No accession directories found")

    @patch("metaquest.cli.commands.sra_enhanced.logger")
    def test_execute_exception_handling(self, mock_logger, tmp_path):
        """Test exception handling during execution."""
        command = SRAValidateCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.mkdir()

        args = argparse.Namespace(fastq_folder=str(fastq_folder), accessions=None, check_pairs=False)

        # Mock _validate_directory to raise an exception
        with patch.object(command, "_validate_directory", side_effect=Exception("Test error")):
            result = command.execute(args)

        assert result == 1
        mock_logger.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
