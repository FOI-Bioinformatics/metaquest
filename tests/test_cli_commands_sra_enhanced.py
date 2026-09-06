"""
Test CLI enhanced SRA commands functionality.

Tests for the sra_info, sra_stats, and sra_validate command classes,
focusing on argument parsing, validation, and proper delegation with mocked dependencies.
"""

import argparse
import json
from unittest.mock import Mock, patch, mock_open

import pandas as pd_module
import pytest

from metaquest.cli.commands.sra_enhanced import (
    SRAInfoCommand,
    SRAStatsCommand,
    SRAValidateCommand,
)


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

    @patch("metaquest.cli.commands.sra_enhanced.save_metadata_report")
    @patch("metaquest.cli.commands.sra_enhanced.create_download_preview")
    @patch("metaquest.cli.commands.sra_enhanced.SRAMetadataClient")
    @patch("builtins.print")
    def test_execute_success(self, mock_print, mock_client_class, mock_create_preview, mock_save_report, tmp_path):
        """Test successful execution."""
        command = SRAInfoCommand()
        args = argparse.Namespace(
            accessions_file="test_accessions.txt",
            email="test@example.com",
            api_key="test_key",
            output_report="test_report.csv",
            bandwidth_mbps=100.0,
        )

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_metadata = {
            "SRR123456": Mock(platform="ILLUMINA", layout="PAIRED", size_mb=1024),
            "SRR789012": Mock(platform="ILLUMINA", layout="SINGLE", size_mb=512),
        }
        mock_tech_counts = {"RNA-Seq": 2}
        mock_total_size = 1.5

        mock_create_preview.return_value = (mock_metadata, mock_tech_counts, mock_total_size)

        # Mock file reading
        mock_file_content = "SRR123456\nSRR789012\n"
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = command.execute(args)

        assert result == 0
        mock_client_class.assert_called_once_with("test@example.com", "test_key")
        mock_create_preview.assert_called_once_with(["SRR123456", "SRR789012"], mock_client)
        mock_save_report.assert_called_once()

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

    @patch("metaquest.cli.commands.sra_enhanced.create_download_preview")
    @patch("metaquest.cli.commands.sra_enhanced.SRAMetadataClient")
    @patch("builtins.print")
    def test_execute_no_metadata(self, mock_print, mock_client_class, mock_create_preview):
        """Test execution when no metadata can be fetched."""
        command = SRAInfoCommand()
        args = argparse.Namespace(
            accessions_file="test.txt",
            email="test@example.com",
            api_key=None,
            output_report="report.csv",
            bandwidth_mbps=100.0,
        )

        mock_create_preview.return_value = ({}, {}, 0.0)

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

    def test_sample_size_defaults_and_rejects_non_positive_values(self):
        command = SRAStatsCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        assert parser.parse_args([]).sample_size == 10000
        assert parser.parse_args(["--sample-size", "500"]).sample_size == 500
        with pytest.raises(SystemExit):
            parser.parse_args(["--sample-size", "0"])

    @patch("metaquest.cli.commands.sra_enhanced.generate_statistics_report")
    @patch("builtins.print")
    def test_execute_passes_sample_size_through(self, mock_print, mock_generate_report, tmp_path):
        """--sample-size reaches the report generator, which uses it for both the shared
        statistics record and the streaming per-read sample."""
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        args = argparse.Namespace(
            fastq_folder=str(fastq_folder), output_report="stats.csv", accessions=None, sample_size=500
        )

        assert SRAStatsCommand().execute(args) == 0

        mock_generate_report.assert_called_once_with(fastq_folder, "stats.csv", sample_size=500)

    @patch("metaquest.cli.commands.sra_enhanced.generate_statistics_report")
    @patch("builtins.print")
    def test_execute_success(self, mock_print, mock_generate_report, tmp_path):
        """Test successful execution."""
        command = SRAStatsCommand()

        # Create test fastq folder
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()

        args = argparse.Namespace(fastq_folder=str(fastq_folder), output_report="stats.csv", accessions=None)

        result = command.execute(args)

        assert result == 0
        mock_generate_report.assert_called_once_with(fastq_folder, "stats.csv", sample_size=10000)

    @patch("builtins.print")
    def test_execute_records_analyses_in_registry(self, mock_print, tmp_path):
        """Each accession in the (mocked) statistics report is recorded as an sra_stats analysis."""
        command = SRAStatsCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        report_path = tmp_path / "stats.csv"
        registry_path = tmp_path / "metaquest_registry.json"

        def fake_generate_report(folder, output_report, sample_size=None):
            pd_module.DataFrame(
                [
                    {"accession": "SRR1", "total_reads": 1000, "gc_content": 45.0, "avg_read_length": 150.0},
                    {"accession": "SRR2", "total_reads": 2000, "gc_content": 50.0, "avg_read_length": 151.0},
                ]
            ).to_csv(output_report, index=False)

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            output_report=str(report_path),
            accessions=None,
            registry=str(registry_path),
            data_root=None,
        )

        with patch("metaquest.cli.commands.sra_enhanced.generate_statistics_report", side_effect=fake_generate_report):
            result = command.execute(args)

        assert result == 0
        registry = json.loads(registry_path.read_text())
        for acc, total_reads in (("SRR1", 1000), ("SRR2", 2000)):
            analysis = registry["datasets"][acc]["analyses"]["sra_stats"]
            # report_path lives under the project root (the registry's own folder), so the
            # registry records it relative to it, which keeps the project movable.
            assert analysis["output"] == "stats.csv"
            assert analysis["summary"]["total_reads"] == total_reads

    @patch("builtins.print")
    def test_execute_records_usage_in_store_catalogue(self, mock_print, tmp_path):
        """Each accession is also recorded as 'analysed' usage in the store catalogue."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.catalog import Catalog
        from metaquest.store.layout import init_store, store_paths

        command = SRAStatsCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        report_path = tmp_path / "stats.csv"
        registry_path = tmp_path / "metaquest_registry.json"
        store_root = tmp_path / "store"
        init_store(store_root)

        registry = _load(registry_path)
        registry.project = {"id": "proj1", "name": "demo", "path": str(tmp_path), "created": "now"}
        _save(registry)

        def fake_generate_report(folder, output_report, sample_size=None):
            pd_module.DataFrame(
                [{"accession": "SRR1", "total_reads": 1000, "gc_content": 45.0, "avg_read_length": 150.0}]
            ).to_csv(output_report, index=False)

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            output_report=str(report_path),
            accessions=None,
            registry=str(registry_path),
            data_root=str(store_root),
        )

        with patch("metaquest.cli.commands.sra_enhanced.generate_statistics_report", side_effect=fake_generate_report):
            result = command.execute(args)

        assert result == 0
        with Catalog(store_paths(store_root)) as catalog:
            row = catalog.conn.execute(
                "SELECT stage FROM usage WHERE accession = ? AND project_id = ?", ("SRR1", "proj1")
            ).fetchone()
        assert row["stage"] == "analysed"

    @patch("builtins.print")
    def test_catalog_failure_leaves_analysis_outcome_unchanged(self, mock_print, tmp_path):
        """A broken catalogue write never changes the sra_stats registry outcome or exit code."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.layout import init_store

        command = SRAStatsCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        report_path = tmp_path / "stats.csv"
        registry_path = tmp_path / "metaquest_registry.json"
        store_root = tmp_path / "store"
        init_store(store_root)

        registry = _load(registry_path)
        registry.project = {"id": "proj1", "name": "demo", "path": str(tmp_path), "created": "now"}
        _save(registry)

        def fake_generate_report(folder, output_report, sample_size=None):
            pd_module.DataFrame(
                [{"accession": "SRR1", "total_reads": 1000, "gc_content": 45.0, "avg_read_length": 150.0}]
            ).to_csv(output_report, index=False)

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            output_report=str(report_path),
            accessions=None,
            registry=str(registry_path),
            data_root=str(store_root),
        )

        with patch("metaquest.cli.commands.sra_enhanced.generate_statistics_report", side_effect=fake_generate_report):
            with patch("metaquest.store.usage.catalog_write", side_effect=RuntimeError("locked")):
                result = command.execute(args)

        assert result == 0
        registry_after = json.loads(registry_path.read_text())
        analysis = registry_after["datasets"]["SRR1"]["analyses"]["sra_stats"]
        assert analysis["summary"]["total_reads"] == 1000

    @patch("builtins.print")
    def test_execute_folder_not_exists(self, mock_print):
        """Test execution when fastq folder doesn't exist."""
        command = SRAStatsCommand()
        args = argparse.Namespace(fastq_folder="/nonexistent/folder", output_report="stats.csv", accessions=None)

        result = command.execute(args)

        assert result == 1
        mock_print.assert_called_with("FASTQ folder /nonexistent/folder does not exist")

    @patch("metaquest.cli.commands.sra_enhanced.generate_statistics_report")
    @patch("metaquest.cli.commands.sra_enhanced.logger")
    def test_execute_exception_handling(self, mock_logger, mock_generate_report, tmp_path):
        """Test exception handling during execution."""
        command = SRAStatsCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()

        args = argparse.Namespace(fastq_folder=str(fastq_folder), output_report="stats.csv", accessions=None)

        # Mock generate_statistics_report to raise an exception
        mock_generate_report.side_effect = Exception("Test error")

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
        assert not args.md5

    def test_configure_parser_with_options(self):
        """Test parser with all options."""
        command = SRAValidateCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            ["--fastq-folder", "custom_fastq", "--accessions", "SRR123", "SRR456", "--check-pairs", "--md5"]
        )

        assert args.fastq_folder == "custom_fastq"
        assert args.accessions == ["SRR123", "SRR456"]
        assert args.check_pairs
        assert args.md5

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

        result = command._validate_directory(acc_dir)

        assert result["accession"] == "SRR123"
        assert result["status"] == "PASSED"
        assert result["issues"] == "None"
        assert result["issues_list"] == []
        assert result["num_files"] == 1
        assert result["checks"] == ["empty_files", "format", "completeness"]

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
    def test_validate_directory_mate_count_mismatch(self, mock_print, tmp_path):
        """A paired-end dataset whose mates have different read counts is flagged, with the
        read counts named in the message."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        (acc_dir / "SRR123_1.fastq").write_text("@r\nACGT\n+\nIIII\n" * 2)
        (acc_dir / "SRR123_2.fastq").write_text("@r\nACGT\n+\nIIII\n" * 1)

        result = command._validate_directory(acc_dir, check_pairs=True)

        assert result["status"] == "FAILED"
        assert "mate files differ (2 vs 1)" in result["issues"]
        assert "mate_counts" in result["checks"]

    @patch("metaquest.cli.commands.sra_enhanced.cached_stats")
    @patch("metaquest.cli.commands.sra_enhanced.count_fastq_reads")
    @patch("builtins.print")
    def test_validate_directory_mate_count_uses_cached_stats(self, mock_print, mock_count, mock_cached, tmp_path):
        """When a cached stats record is available, mate counts come from its
        ``reads_per_file`` rather than a fresh ``count_fastq_reads`` pass."""
        mock_cached.return_value = {"reads_per_file": {"SRR123_1.fastq": 5, "SRR123_2.fastq": 5}}

        command = SRAValidateCommand()
        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()
        (acc_dir / "SRR123_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        (acc_dir / "SRR123_2.fastq").write_text("@r\nACGT\n+\nIIII\n")

        result = command._validate_directory(acc_dir, check_pairs=True)

        assert result["status"] == "PASSED"
        mock_count.assert_not_called()

    @patch("builtins.print")
    def test_validate_directory_format_error_broken_header(self, mock_print, tmp_path):
        """A file whose first line is not a FASTQ header is caught."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        fastq_file = acc_dir / "test.fastq"
        fastq_file.write_text("invalid fastq content\nACGT\n+\nIIII\n")

        result = command._validate_directory(acc_dir)

        assert result["status"] == "FAILED"
        assert "FASTQ format error" in result["issues"]
        assert "header does not start with" in result["issues"]

    @patch("builtins.print")
    def test_validate_directory_format_error_length_mismatch(self, mock_print, tmp_path):
        """A first record whose sequence and quality strings differ in length is caught."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        fastq_file = acc_dir / "test.fastq"
        fastq_file.write_text("@read1\nACGTACGT\n+\nIII\n")

        result = command._validate_directory(acc_dir)

        assert result["status"] == "FAILED"
        assert "sequence/quality length mismatch" in result["issues"]

    @patch("builtins.print")
    def test_validate_directory_format_check_does_not_read_a_large_file_in_full(self, mock_print, tmp_path):
        """The first-record shape check never falls back to a full-file read: a large file
        with a broken header is rejected without ``count_fastq_reads`` (a full streaming
        pass) ever being called."""
        command = SRAValidateCommand()

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()

        fastq_file = acc_dir / "test.fastq"
        # A large file (many valid-looking records) but a broken first header.
        fastq_file.write_text("not-a-header\n" + ("@r\nACGT\n+\nIIII\n" * 50000))

        with patch("metaquest.cli.commands.sra_enhanced.count_fastq_reads") as mock_count:
            result = command._validate_directory(acc_dir)

        assert result["status"] == "FAILED"
        assert "header does not start with" in result["issues"]
        mock_count.assert_not_called()

    @patch("builtins.print")
    def test_validate_directory_partial_sidecar_reports_spots(self, mock_print, tmp_path):
        """A store-linked accession whose sidecar records a partial download is failed with
        the reads-on-disk-vs-spots-at-NCBI message."""
        from metaquest.store.sidecar import Sidecar, write_sidecar

        store_acc_dir = tmp_path / "store" / "sra" / "SRR123"
        store_acc_dir.mkdir(parents=True)
        (store_acc_dir / "SRR123.fastq").write_text("@r\nACGT\n+\nIIII\n")
        write_sidecar(
            store_acc_dir / "SRR123.json",
            Sidecar(
                accession="SRR123",
                state="partial",
                reads_per_mate=5,
                ncbi={"spots": 100},
            ),
        )

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.symlink_to(store_acc_dir)

        command = SRAValidateCommand()
        result = command._validate_directory(acc_dir)

        assert result["status"] == "FAILED"
        assert "partial: 5 reads on disk vs 100 spots at NCBI" in result["issues"]

    @pytest.mark.parametrize(
        "state, expected",
        [("failed", "failed at NCBI download"), ("downloading", "download in progress elsewhere")],
    )
    @patch("builtins.print")
    def test_validate_directory_flags_failed_and_downloading_sidecars(self, mock_print, tmp_path, state, expected):
        """Only a complete or adopted dataset passes: a failed download, and one another
        project is still downloading, are not finished datasets even when their files parse."""
        from metaquest.store.sidecar import Sidecar, write_sidecar

        store_acc_dir = tmp_path / "store" / "sra" / "SRR123"
        store_acc_dir.mkdir(parents=True)
        (store_acc_dir / "SRR123.fastq").write_text("@r\nACGT\n+\nIIII\n")
        write_sidecar(store_acc_dir / "SRR123.json", Sidecar(accession="SRR123", state=state))

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.symlink_to(store_acc_dir)

        result = SRAValidateCommand()._validate_directory(acc_dir)

        assert result["status"] == "FAILED"
        assert expected in result["issues"]

    @patch("builtins.print")
    def test_validate_directory_passes_a_complete_sidecar(self, mock_print, tmp_path):
        from metaquest.store.sidecar import Sidecar, write_sidecar

        store_acc_dir = tmp_path / "store" / "sra" / "SRR123"
        store_acc_dir.mkdir(parents=True)
        (store_acc_dir / "SRR123.fastq").write_text("@r\nACGT\n+\nIIII\n")
        write_sidecar(store_acc_dir / "SRR123.json", Sidecar(accession="SRR123", state="complete"))

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.symlink_to(store_acc_dir)

        assert SRAValidateCommand()._validate_directory(acc_dir)["status"] == "PASSED"

    @patch("builtins.print")
    def test_validate_directory_registry_verdict_truncated_reports_spots(self, mock_print, tmp_path):
        """A plain project directory (no store sidecar) falls back to the registry's own
        download verdict for the same completeness check."""
        from metaquest.data.registry import Registry

        acc_dir = tmp_path / "SRR123"
        acc_dir.mkdir()
        (acc_dir / "SRR123.fastq").write_text("@r\nACGT\n+\nIIII\n")

        registry = Registry()
        registry.datasets["SRR123"] = {
            "download": {"complete": {"verdict": "truncated", "reads_r1": 5, "expected_spots": 100}}
        }

        command = SRAValidateCommand()
        result = command._validate_directory(acc_dir, registry)

        assert result["status"] == "FAILED"
        assert "partial: 5 reads on disk vs 100 spots at NCBI" in result["issues"]

    @patch("builtins.print")
    def test_validate_directory_md5_mismatch(self, mock_print, tmp_path):
        """--md5 fails a file whose content no longer matches the sidecar's recorded md5."""
        from metaquest.store.sidecar import Sidecar, write_sidecar

        store_acc_dir = tmp_path / "store" / "sra" / "SRR123"
        store_acc_dir.mkdir(parents=True)
        fastq_path = store_acc_dir / "SRR123.fastq"
        fastq_path.write_text("@r\nACGT\n+\nIIII\n")
        write_sidecar(
            store_acc_dir / "SRR123.json",
            Sidecar(
                accession="SRR123",
                files=[{"name": "SRR123.fastq", "bytes": fastq_path.stat().st_size, "md5": "0" * 32, "reads": 1}],
            ),
        )

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.symlink_to(store_acc_dir)

        command = SRAValidateCommand()
        result = command._validate_directory(acc_dir, check_md5=True)

        assert result["status"] == "FAILED"
        assert "md5 mismatch: SRR123.fastq" in result["issues"]
        assert "md5" in result["checks"]

    @patch("builtins.print")
    def test_validate_directory_md5_match_passes(self, mock_print, tmp_path):
        """--md5 passes when the file's md5 matches the sidecar's recorded value."""
        from metaquest.store.sidecar import Sidecar, md5_file, write_sidecar

        store_acc_dir = tmp_path / "store" / "sra" / "SRR123"
        store_acc_dir.mkdir(parents=True)
        fastq_path = store_acc_dir / "SRR123.fastq"
        fastq_path.write_text("@r\nACGT\n+\nIIII\n")
        write_sidecar(
            store_acc_dir / "SRR123.json",
            Sidecar(
                accession="SRR123",
                files=[
                    {
                        "name": "SRR123.fastq",
                        "bytes": fastq_path.stat().st_size,
                        "md5": md5_file(fastq_path),
                        "reads": 1,
                    }
                ],
            ),
        )

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.symlink_to(store_acc_dir)

        command = SRAValidateCommand()
        result = command._validate_directory(acc_dir, check_md5=True)

        assert result["status"] == "PASSED"

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

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            accessions=None,
            check_pairs=False,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        # Mock _validate_directory to return success
        with patch.object(command, "_validate_directory") as mock_validate:
            mock_validate.return_value = {
                "accession": "SRR123",
                "status": "PASSED",
                "issues": "None",
                "num_files": 1,
            }

            result = command.execute(args)

        assert result == 0
        mock_validate.assert_called_once()

    @patch("builtins.print")
    def test_execute_records_analyses_in_registry(self, mock_print, tmp_path):
        """Each validated accession is recorded with its pass/fail status and file count."""
        command = SRAValidateCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.mkdir()
        (acc_dir / "test.fastq").write_text("@r\nACGT\n+\n!!!!\n")
        registry_path = tmp_path / "metaquest_registry.json"

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            accessions=None,
            check_pairs=False,
            registry=str(registry_path),
            data_root=None,
        )

        with patch.object(command, "_validate_directory") as mock_validate:
            mock_validate.return_value = {
                "accession": "SRR123",
                "status": "PASSED",
                "issues": "None",
                "issues_list": [],
                "num_files": 1,
                "checks": ["empty_files", "format", "completeness"],
            }
            result = command.execute(args)

        assert result == 0
        registry = json.loads(registry_path.read_text())
        assert registry["datasets"]["SRR123"]["analyses"]["validate"]["summary"] == {
            "passed": True,
            "files": 1,
            "issues": [],
        }

    @patch("builtins.print")
    def test_execute_records_usage_in_store_catalogue(self, mock_print, tmp_path):
        """A validated accession is also recorded as 'analysed' usage in the store catalogue."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.catalog import Catalog
        from metaquest.store.layout import init_store, store_paths

        command = SRAValidateCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.mkdir()
        (acc_dir / "test.fastq").write_text("@r\nACGT\n+\n!!!!\n")
        registry_path = tmp_path / "metaquest_registry.json"
        store_root = tmp_path / "store"
        init_store(store_root)

        registry = _load(registry_path)
        registry.project = {"id": "proj1", "name": "demo", "path": str(tmp_path), "created": "now"}
        _save(registry)

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            accessions=None,
            check_pairs=False,
            registry=str(registry_path),
            data_root=str(store_root),
        )

        with patch.object(command, "_validate_directory") as mock_validate:
            mock_validate.return_value = {
                "accession": "SRR123",
                "status": "PASSED",
                "issues": "None",
                "num_files": 1,
            }
            result = command.execute(args)

        assert result == 0
        with Catalog(store_paths(store_root)) as catalog:
            row = catalog.conn.execute(
                "SELECT stage FROM usage WHERE accession = ? AND project_id = ?", ("SRR123", "proj1")
            ).fetchone()
        assert row["stage"] == "analysed"

    @patch("builtins.print")
    def test_catalog_failure_leaves_validate_outcome_unchanged(self, mock_print, tmp_path):
        """A broken catalogue write never changes validate's registry outcome or exit code."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.layout import init_store

        command = SRAValidateCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.mkdir()
        (acc_dir / "test.fastq").write_text("@r\nACGT\n+\n!!!!\n")
        registry_path = tmp_path / "metaquest_registry.json"
        store_root = tmp_path / "store"
        init_store(store_root)

        registry = _load(registry_path)
        registry.project = {"id": "proj1", "name": "demo", "path": str(tmp_path), "created": "now"}
        _save(registry)

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            accessions=None,
            check_pairs=False,
            registry=str(registry_path),
            data_root=str(store_root),
        )

        with patch.object(command, "_validate_directory") as mock_validate:
            mock_validate.return_value = {
                "accession": "SRR123",
                "status": "PASSED",
                "issues": "None",
                "num_files": 1,
            }
            with patch("metaquest.store.usage.catalog_write", side_effect=RuntimeError("locked")):
                result = command.execute(args)

        assert result == 0
        registry_after = json.loads(registry_path.read_text())
        assert registry_after["datasets"]["SRR123"]["analyses"]["validate"]["summary"] == {
            "passed": True,
            "files": 1,
            "issues": [],
        }

    @patch("builtins.print")
    def test_execute_records_failing_issues_in_registry_summary(self, mock_print, tmp_path):
        """A failed accession's registry summary carries the raw issue list, not just the
        joined display string, and names the checks that ran."""
        command = SRAValidateCommand()

        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        acc_dir = fastq_folder / "SRR123"
        acc_dir.mkdir()
        (acc_dir / "test.fastq").touch()  # empty file: fails the "empty_files" check
        registry_path = tmp_path / "metaquest_registry.json"

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            accessions=None,
            check_pairs=False,
            md5=False,
            registry=str(registry_path),
            data_root=None,
        )

        result = command.execute(args)

        assert result == 1
        registry = json.loads(registry_path.read_text())
        summary = registry["datasets"]["SRR123"]["analyses"]["validate"]["summary"]
        assert summary["passed"] is False
        assert summary["files"] == 1
        assert summary["issues"] == ["Empty file: test.fastq"]

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

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            accessions=None,
            check_pairs=False,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        # Mock _validate_directory to raise an exception
        with patch.object(command, "_validate_directory", side_effect=Exception("Test error")):
            result = command.execute(args)

        assert result == 1
        mock_logger.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])


class TestAnalysisWithoutAReachableStore:
    """A store that cannot be read costs the usage record, never the analysis."""

    @patch("builtins.print")
    def test_sra_stats_warns_and_completes_when_the_store_root_is_gone(self, mock_print, tmp_path, caplog):
        import logging

        command = SRAStatsCommand()
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        report_path = tmp_path / "stats.csv"
        registry_path = tmp_path / "metaquest_registry.json"
        gone = tmp_path / "unmounted"

        def fake_generate_report(folder, output_report, sample_size=None):
            pd_module.DataFrame(
                [{"accession": "SRR1", "total_reads": 1000, "gc_content": 45.0, "avg_read_length": 150.0}]
            ).to_csv(output_report, index=False)

        args = argparse.Namespace(
            fastq_folder=str(fastq_folder),
            output_report=str(report_path),
            accessions=None,
            registry=str(registry_path),
            data_root=str(gone),
        )

        with caplog.at_level(logging.WARNING):
            with patch(
                "metaquest.cli.commands.sra_enhanced.generate_statistics_report", side_effect=fake_generate_report
            ):
                result = command.execute(args)

        assert result == 0
        assert report_path.is_file()
        assert any("store unavailable" in record.message for record in caplog.records)
