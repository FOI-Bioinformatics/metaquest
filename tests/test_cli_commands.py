"""
Test CLI commands functionality.

This module tests the CLI command classes and their execute methods,
focusing on argument parsing, validation, and proper delegation.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from metaquest.cli.commands.branchwater import (
    UseBranchwaterCommand,
    ExtractBranchwaterMetadataCommand,
)
from metaquest.cli.commands.containment import ParseContainmentCommand, PlotContainmentCommand
from metaquest.cli.commands.metadata import (
    DownloadMetadataCommand,
    ParseMetadataCommand,
    CountMetadataCommand,
    PlotMetadataCountsCommand,
)
from metaquest.cli.commands.sra import DownloadSraCommand, AssembleDatasetsCommand
from metaquest.cli.commands.samples import SingleSampleCommand
from metaquest.cli.commands.test_data import DownloadTestGenomeCommand
from metaquest.core.exceptions import MetaQuestError


class TestUseBranchwaterCommand:
    """Test UseBranchwaterCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = UseBranchwaterCommand()
        assert command.name == "use_branchwater"
        assert "process" in command.help.lower()
        assert "branchwater" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = UseBranchwaterCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Check required arguments are added
        args = parser.parse_args(["--branchwater-folder", "test_folder"])
        assert args.branchwater_folder == "test_folder"
        assert args.matches_folder == "matches"  # default value

        # Test with custom matches folder
        args = parser.parse_args(["--branchwater-folder", "test_folder", "--matches-folder", "custom_matches"])
        assert args.matches_folder == "custom_matches"

    def test_configure_parser_missing_required(self):
        """Test parser with missing required arguments."""
        command = UseBranchwaterCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing required --branchwater-folder

    @patch("metaquest.cli.commands.branchwater.process_branchwater_files")
    def test_execute_success(self, mock_command):
        """Test successful command execution."""
        mock_command.return_value = {"file1": Path("test")}
        command = UseBranchwaterCommand()

        args = argparse.Namespace(branchwater_folder="test_folder", matches_folder="matches")

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with("test_folder", "matches")

    @patch("metaquest.cli.commands.branchwater.process_branchwater_files")
    def test_execute_failure(self, mock_command):
        """Test command execution failure."""
        mock_command.side_effect = MetaQuestError("Test error")
        command = UseBranchwaterCommand()

        args = argparse.Namespace(branchwater_folder="test_folder", matches_folder="matches")

        result = command.execute(args)
        assert result == 1
        mock_command.assert_called_once_with("test_folder", "matches")


class TestExtractBranchwaterMetadataCommand:
    """Test ExtractBranchwaterMetadataCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = ExtractBranchwaterMetadataCommand()
        assert command.name == "extract_branchwater_metadata"
        assert "extract" in command.help.lower()
        assert "metadata" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = ExtractBranchwaterMetadataCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(["--branchwater-folder", "test_folder"])
        assert args.branchwater_folder == "test_folder"
        assert args.metadata_folder == "metadata"

    @patch("metaquest.cli.commands.branchwater.extract_metadata_from_branchwater")
    @patch("pathlib.Path.mkdir")
    def test_execute(self, mock_mkdir, mock_command):
        """Test command execution."""
        import pandas as pd

        mock_command.return_value = pd.DataFrame()
        command = ExtractBranchwaterMetadataCommand()

        args = argparse.Namespace(branchwater_folder="test_folder", metadata_folder="metadata")

        result = command.execute(args)
        assert result == 0
        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_command.assert_called_once_with("test_folder", Path("metadata/branchwater_metadata.txt"))


class TestParseContainmentCommand:
    """Test ParseContainmentCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = ParseContainmentCommand()
        assert command.name == "parse_containment"
        assert "parse" in command.help.lower()
        assert "containment" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = ParseContainmentCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required argument
        args = parser.parse_args(["--matches-folder", "test_matches"])
        assert args.matches_folder == "test_matches"
        assert args.parsed_containment_file == "parsed_containment.txt"
        assert args.summary_containment_file == "top_containments.txt"
        assert args.step_size == 0.1
        assert args.file_format is None

    def test_configure_parser_with_optional_args(self):
        """Test parser with optional arguments."""
        command = ParseContainmentCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--matches-folder",
                "test_matches",
                "--parsed-containment-file",
                "custom_parsed.txt",
                "--summary-containment-file",
                "custom_summary.txt",
                "--step-size",
                "0.1",
                "--file-format",
                "branchwater",
            ]
        )

        assert args.parsed_containment_file == "custom_parsed.txt"
        assert args.summary_containment_file == "custom_summary.txt"
        assert args.step_size == 0.1
        assert args.file_format == "branchwater"

    @patch("metaquest.cli.commands.containment.parse_containment_data")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = None
        command = ParseContainmentCommand()

        args = argparse.Namespace(
            matches_folder="test_matches",
            parsed_containment_file="parsed.txt",
            summary_containment_file="summary.txt",
            step_size=0.05,
            file_format=None,
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with("test_matches", "parsed.txt", "summary.txt", 0.05)


class TestDownloadMetadataCommand:
    """Test DownloadMetadataCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = DownloadMetadataCommand()
        assert command.name == "download_metadata"
        assert "download" in command.help.lower()
        assert "metadata" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = DownloadMetadataCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required email
        args = parser.parse_args(["--email", "test@example.com"])
        assert args.email == "test@example.com"
        assert args.matches_folder == "matches"
        assert args.metadata_folder == "metadata"
        assert args.threshold == 0.0
        assert args.dry_run is False

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = DownloadMetadataCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--email",
                "test@example.com",
                "--matches-folder",
                "custom_matches",
                "--metadata-folder",
                "custom_metadata",
                "--threshold",
                "0.5",
                "--dry-run",
            ]
        )

        assert args.threshold == 0.5
        assert args.dry_run is True

    @patch("metaquest.cli.commands.metadata.download_metadata")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = None
        command = DownloadMetadataCommand()

        args = argparse.Namespace(
            email="test@example.com", matches_folder="matches", metadata_folder="metadata", threshold=0.0, dry_run=False
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(
            email="test@example.com", matches_folder="matches", metadata_folder="metadata", threshold=0.0, dry_run=False
        )


class TestDownloadTestGenomeCommand:
    """Test DownloadTestGenomeCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = DownloadTestGenomeCommand()
        assert command.name == "download_test_genome"
        assert "download" in command.help.lower()
        assert "test" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = DownloadTestGenomeCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test default output folder
        args = parser.parse_args([])
        assert args.output_folder == "genomes"

        # Test custom output folder
        args = parser.parse_args(["--output-folder", "custom_genomes"])
        assert args.output_folder == "custom_genomes"

    @patch("metaquest.cli.commands.test_data.download_test_genome")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = None
        command = DownloadTestGenomeCommand()

        args = argparse.Namespace(output_folder="genomes")
        result = command.execute(args)

        assert result == 0
        mock_command.assert_called_once_with("genomes")


class TestDownloadSraCommand:
    """Test DownloadSraCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = DownloadSraCommand()
        assert command.name == "download_sra"
        assert "download" in command.help.lower()
        assert "sra" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = DownloadSraCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required accessions file
        args = parser.parse_args(["--accessions-file", "accessions.txt"])
        assert args.accessions_file == "accessions.txt"
        assert args.fastq_folder == "fastq"
        assert args.num_threads == 4
        assert args.max_workers == 4
        assert args.dry_run is False
        assert args.force is False

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = DownloadSraCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--accessions-file",
                "accessions.txt",
                "--fastq-folder",
                "custom_fastq",
                "--num-threads",
                "8",
                "--max-workers",
                "2",
                "--max-downloads",
                "10",
                "--dry-run",
                "--force",
            ]
        )

        assert args.fastq_folder == "custom_fastq"
        assert args.num_threads == 8
        assert args.max_workers == 2
        assert args.max_downloads == 10
        assert args.dry_run is True
        assert args.force is True

    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = {
            "total": 1,
            "to_download": 1,
            "already_downloaded": 0,
            "successful": 1,
            "failed": 0,
            "failed_accessions": [],
        }
        command = DownloadSraCommand()

        args = argparse.Namespace(
            accessions_file="accessions.txt",
            fastq_folder="fastq",
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(
            fastq_folder="fastq",
            accessions_file="accessions.txt",
            max_downloads=None,
            dry_run=False,
            num_threads=4,
            max_workers=4,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
        )


class TestSingleSampleCommand:
    """Test SingleSampleCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = SingleSampleCommand()
        assert command.name == "single_sample"
        assert "single" in command.help.lower()
        assert "sample" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = SingleSampleCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required arguments
        args = parser.parse_args(["--summary-column", "test_genome", "--metadata-column", "organism"])

        assert args.summary_column == "test_genome"
        assert args.metadata_column == "organism"
        assert args.summary_file == "parsed_containment.txt"
        assert args.metadata_file == "metadata_table.txt"
        assert args.threshold == 0.1
        assert args.top_n == 100

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = SingleSampleCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--summary-column",
                "test_genome",
                "--metadata-column",
                "organism",
                "--summary-file",
                "custom_summary.txt",
                "--metadata-file",
                "custom_metadata.txt",
                "--threshold",
                "0.5",
                "--top-n",
                "50",
            ]
        )

        assert args.summary_file == "custom_summary.txt"
        assert args.metadata_file == "custom_metadata.txt"
        assert args.threshold == 0.5
        assert args.top_n == 50

    @patch("metaquest.cli.commands.samples.count_single_sample")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = {"organism1": 5, "organism2": 3}
        command = SingleSampleCommand()

        args = argparse.Namespace(
            summary_column="test_genome",
            metadata_column="organism",
            summary_file="summary.txt",
            metadata_file="metadata.txt",
            threshold=0.1,
            top_n=100,
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(
            summary_file="summary.txt",
            metadata_file="metadata.txt",
            summary_column="test_genome",
            metadata_column="organism",
            threshold=0.1,
            top_n=100,
        )


class TestAssembleDatasetsCommand:
    """Test AssembleDatasetsCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = AssembleDatasetsCommand()
        assert command.name == "assemble_datasets"
        assert "assemble" in command.help.lower()
        assert "datasets" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = AssembleDatasetsCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required arguments
        args = parser.parse_args(["--data-files", "file1.txt", "file2.txt", "--output-file", "assembled.txt"])

        assert args.data_files == ["file1.txt", "file2.txt"]
        assert args.output_file == "assembled.txt"

    @patch("metaquest.cli.commands.sra.assemble_datasets")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = None
        command = AssembleDatasetsCommand()

        args = argparse.Namespace(data_files=["file1.txt", "file2.txt"], output_file="assembled.txt")

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(args)


class TestParseMetadataCommand:
    """Test ParseMetadataCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = ParseMetadataCommand()
        assert command.name == "parse_metadata"
        assert "parse" in command.help.lower()
        assert "metadata" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = ParseMetadataCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test default values
        args = parser.parse_args([])
        assert args.metadata_folder == "metadata"
        assert args.metadata_table_file == "metadata_table.txt"

        # Test custom values
        args = parser.parse_args(["--metadata-folder", "custom_metadata", "--metadata-table-file", "custom_table.txt"])
        assert args.metadata_folder == "custom_metadata"
        assert args.metadata_table_file == "custom_table.txt"

    @patch("metaquest.cli.commands.metadata.parse_metadata")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = None
        command = ParseMetadataCommand()

        args = argparse.Namespace(metadata_folder="metadata", metadata_table_file="metadata_table.txt")

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with("metadata", "metadata_table.txt")


class TestCountMetadataCommand:
    """Test CountMetadataCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = CountMetadataCommand()
        assert command.name == "count_metadata"
        assert "count" in command.help.lower()
        assert "metadata" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = CountMetadataCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required metadata column
        args = parser.parse_args(["--metadata-column", "organism"])
        assert args.metadata_column == "organism"
        assert args.summary_file == "parsed_containment.txt"
        assert args.metadata_file == "metadata_table.txt"
        assert args.threshold == 0.5
        assert args.output_file == "metadata_counts.txt"
        assert args.stat_file is None

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = CountMetadataCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--metadata-column",
                "organism",
                "--summary-file",
                "custom_summary.txt",
                "--threshold",
                "0.8",
                "--output-file",
                "custom_counts.txt",
                "--stat-file",
                "statistics.txt",
            ]
        )

        assert args.summary_file == "custom_summary.txt"
        assert args.threshold == 0.8
        assert args.output_file == "custom_counts.txt"
        assert args.stat_file == "statistics.txt"

    @patch("metaquest.cli.commands.metadata.count_metadata")
    def test_execute(self, mock_command):
        """Test command execution."""
        mock_command.return_value = None
        command = CountMetadataCommand()

        args = argparse.Namespace(
            metadata_column="organism",
            summary_file="summary.txt",
            metadata_file="metadata.txt",
            threshold=0.5,
            output_file="counts.txt",
            stat_file=None,
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(
            summary_file="summary.txt",
            metadata_file="metadata.txt",
            metadata_column="organism",
            threshold=0.5,
            output_file="counts.txt",
            stat_file=None,
        )


class TestPlotContainmentCommand:
    """Test PlotContainmentCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = PlotContainmentCommand()
        assert command.name == "plot_containment"
        assert "plot" in command.help.lower()
        assert "containment" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = PlotContainmentCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required file path
        args = parser.parse_args(["--file-path", "containment.txt"])
        assert args.file_path == "containment.txt"
        assert args.column == "max_containment"
        assert args.plot_type == "rank"
        assert args.title is None
        assert args.colors is None
        assert args.save_format is None

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = PlotContainmentCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--file-path",
                "containment.txt",
                "--column",
                "custom_column",
                "--plot-type",
                "histogram",
                "--title",
                "Custom Title",
                "--colors",
                "red",
                "--save-format",
                "png",
                "--threshold",
                "0.5",
                "--show-title",
            ]
        )

        assert args.column == "custom_column"
        assert args.plot_type == "histogram"
        assert args.title == "Custom Title"
        assert args.colors == "red"
        assert args.save_format == "png"
        assert args.threshold == 0.5
        assert args.show_title is True

    @patch("metaquest.cli.commands.containment.viz_plot_containment")
    def test_execute(self, mock_command):
        """Test command execution."""
        import matplotlib.pyplot as plt

        mock_command.return_value = plt.figure()
        command = PlotContainmentCommand()

        args = argparse.Namespace(
            file_path="containment.txt",
            column="max_containment",
            plot_type="rank",
            title=None,
            colors=None,
            save_format=None,
            threshold=None,
            show_title=False,
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(
            file_path="containment.txt",
            column="max_containment",
            title=None,
            colors=None,
            show_title=False,
            save_format=None,
            threshold=None,
            plot_type="rank",
        )


class TestPlotMetadataCountsCommand:
    """Test PlotMetadataCountsCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = PlotMetadataCountsCommand()
        assert command.name == "plot_metadata_counts"
        assert "plot" in command.help.lower()
        assert "metadata" in command.help.lower()
        assert "counts" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = PlotMetadataCountsCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test with required file path
        args = parser.parse_args(["--file-path", "counts.txt"])
        assert args.file_path == "counts.txt"
        assert args.plot_type == "bar"
        assert args.title is None
        assert args.colors is None
        assert args.save_format is None
        assert args.show_title is False

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = PlotMetadataCountsCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--file-path",
                "counts.txt",
                "--plot-type",
                "pie",
                "--title",
                "Metadata Counts",
                "--colors",
                "viridis",
                "--save-format",
                "pdf",
                "--show-title",
            ]
        )

        assert args.plot_type == "pie"
        assert args.title == "Metadata Counts"
        assert args.colors == "viridis"
        assert args.save_format == "pdf"
        assert args.show_title is True

    @patch("metaquest.cli.commands.metadata.plot_metadata_counts")
    def test_execute(self, mock_command):
        """Test command execution."""
        import matplotlib.pyplot as plt

        mock_command.return_value = plt.figure()
        command = PlotMetadataCountsCommand()

        args = argparse.Namespace(
            file_path="counts.txt", plot_type="bar", title=None, colors=None, save_format=None, show_title=False
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(
            file_path="counts.txt", title=None, plot_type="bar", colors=None, show_title=False, save_format=None
        )
