"""
Test CLI commands functionality.

This module tests the CLI command classes and their execute methods,
focusing on argument parsing, validation, and proper delegation.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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
from metaquest.cli.commands.sra import DownloadSraCommand
from metaquest.cli.commands.samples import SingleSampleCommand
from metaquest.cli.commands.test_data import DownloadTestGenomeCommand
from metaquest.core.constants import DEFAULT_REGISTRY_MAX_SCREENED, FAILED_ACCESSIONS_FILE
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.registry import load_registry, record_download, record_exclusion, record_metadata, save_registry


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
        assert args.registry_max_screened == DEFAULT_REGISTRY_MAX_SCREENED

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
            ]
        )

        assert args.parsed_containment_file == "custom_parsed.txt"
        assert args.summary_containment_file == "custom_summary.txt"
        assert args.step_size == 0.1

    @patch("metaquest.cli.commands.containment.parse_containment_data")
    def test_execute(self, mock_command, tmp_path):
        """Test command execution."""
        mock_command.return_value = None
        command = ParseContainmentCommand()

        args = argparse.Namespace(
            matches_folder="test_matches",
            parsed_containment_file=str(tmp_path / "parsed.txt"),
            summary_containment_file="summary.txt",
            step_size=0.05,
            registry=str(tmp_path / "metaquest_registry.json"),
            registry_max_screened=DEFAULT_REGISTRY_MAX_SCREENED,
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with("test_matches", str(tmp_path / "parsed.txt"), "summary.txt", 0.05)

    def test_execute_records_screening_in_registry(self, tmp_path):
        """Every accession in the parsed containment table is recorded as screened from matches."""
        matches_folder = tmp_path / "matches"
        matches_folder.mkdir()
        (matches_folder / "GCF_A.csv").write_text("acc,containment\nSRR1,0.9\nSRR2,0.1\n")

        command = ParseContainmentCommand()
        args = argparse.Namespace(
            matches_folder=str(matches_folder),
            parsed_containment_file=str(tmp_path / "parsed.txt"),
            summary_containment_file=str(tmp_path / "summary.txt"),
            step_size=0.1,
            registry=str(tmp_path / "metaquest_registry.json"),
            registry_max_screened=DEFAULT_REGISTRY_MAX_SCREENED,
        )

        result = command.execute(args)
        assert result == 0

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        for acc in ("SRR1", "SRR2"):
            screening = data["datasets"][acc]["screening"]
            assert "GCF_A" in screening["genomes"]
            assert screening["genomes"]["GCF_A"]["source"] == "matches"


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
    def test_execute(self, mock_command, tmp_path):
        """Test command execution."""
        mock_command.return_value = {}
        command = DownloadMetadataCommand()

        args = argparse.Namespace(
            email="test@example.com",
            matches_folder="matches",
            metadata_folder="metadata",
            threshold=0.0,
            dry_run=False,
            accessions_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with(
            email="test@example.com",
            matches_folder="matches",
            metadata_folder="metadata",
            threshold=0.0,
            dry_run=False,
            accessions_file=None,
        )

    def test_execute_records_metadata_in_registry(self, tmp_path):
        """Every accession download_metadata reports downloaded is recorded with its XML path."""
        metadata_folder = tmp_path / "metadata"
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR1\nSRR2\n")

        command = DownloadMetadataCommand()
        args = argparse.Namespace(
            email="test@example.com",
            matches_folder=str(tmp_path / "matches"),
            metadata_folder=str(metadata_folder),
            threshold=0.0,
            dry_run=False,
            accessions_file=str(accessions_file),
            registry=str(tmp_path / "metaquest_registry.json"),
        )

        def fake_download(accession, metadata_path, entrez_email):
            xml_path = metadata_path / f"{accession}_metadata.xml"
            xml_path.write_text("<root/>")
            return True, xml_path

        with patch("metaquest.data.metadata._download_single_metadata", side_effect=fake_download):
            result = command.execute(args)

        assert result == 0
        registry = json.loads((tmp_path / "metaquest_registry.json").read_text())
        for acc in ("SRR1", "SRR2"):
            assert registry["datasets"][acc]["metadata"]["xml"].endswith(f"{acc}_metadata.xml")

    def test_execute_dry_run_records_nothing(self, tmp_path):
        """A dry run does not touch the registry."""
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR1\n")

        command = DownloadMetadataCommand()
        args = argparse.Namespace(
            email="test@example.com",
            matches_folder=str(tmp_path / "matches"),
            metadata_folder=str(tmp_path / "metadata"),
            threshold=0.0,
            dry_run=True,
            accessions_file=str(accessions_file),
            registry=str(tmp_path / "metaquest_registry.json"),
        )

        result = command.execute(args)
        assert result == 0
        assert not (tmp_path / "metaquest_registry.json").exists()


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
        assert args.max_workers is None
        assert args.dry_run is False
        assert args.force is False
        assert args.verify_downloads is True
        assert args.redownload_truncated is False

    def test_configure_parser_verify_downloads_flags(self):
        """--no-verify-downloads flips the default; --redownload-truncated is off by default."""
        command = DownloadSraCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(["--accessions-file", "a.txt", "--no-verify-downloads"])
        assert args.verify_downloads is False

        args = parser.parse_args(["--accessions-file", "a.txt", "--redownload-truncated"])
        assert args.redownload_truncated is True

    def test_configure_parser_prefetch_and_compress_flags(self):
        """--sra-cache, --keep-sra, --no-prefetch, --no-compress default sensibly and parse."""
        command = DownloadSraCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(["--accessions-file", "a.txt"])
        assert args.sra_cache is None
        assert args.use_prefetch is True
        assert args.keep_sra is False
        assert args.compress is True

        args = parser.parse_args(
            [
                "--accessions-file",
                "a.txt",
                "--sra-cache",
                "/tmp/cache",
                "--keep-sra",
                "--no-prefetch",
                "--no-compress",
            ]
        )
        assert args.sra_cache == "/tmp/cache"
        assert args.keep_sra is True
        assert args.use_prefetch is False
        assert args.compress is False

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

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute(self, mock_command, _which, tmp_path):
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
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        result = command.execute(args)
        assert result == 0
        call_kwargs = dict(mock_command.call_args.kwargs)
        on_result = call_kwargs.pop("on_result")
        assert callable(on_result)
        assert call_kwargs == {
            "fastq_folder": str(tmp_path / "fastq"),
            "accessions_file": "accessions.txt",
            "max_downloads": None,
            "dry_run": False,
            "num_threads": 4,
            "max_workers": 4,
            "force": False,
            "max_retries": 1,
            "temp_folder": None,
            "blacklist": None,
            "blacklist_accessions": set(),
            "expected_spots": {},
            "redownload_truncated": False,
            "truncated_accessions": set(),
            "sra_cache": None,
            "use_prefetch": True,
            "keep_sra": False,
            "compress": True,
        }

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_forwards_prefetch_and_compress_flags(self, mock_command, _which, tmp_path):
        """--sra-cache, --keep-sra, --no-prefetch and --no-compress reach download_sra unchanged."""
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
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
            sra_cache="/tmp/custom-cache",
            use_prefetch=False,
            keep_sra=True,
            compress=False,
        )

        result = command.execute(args)
        assert result == 0
        call_kwargs = mock_command.call_args.kwargs
        assert call_kwargs["sra_cache"] == "/tmp/custom-cache"
        assert call_kwargs["use_prefetch"] is False
        assert call_kwargs["keep_sra"] is True
        assert call_kwargs["compress"] is False

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_computes_default_max_workers_from_cpu_count(self, mock_command, _which, tmp_path):
        """When --max-workers is not given, it is computed from the CPU count and thread count."""
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
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=None,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        with patch("metaquest.cli.commands.sra.os.cpu_count", return_value=8):
            result = command.execute(args)

        assert result == 0
        assert mock_command.call_args.kwargs["max_workers"] == 2

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_warns_when_workers_oversubscribe_cpu(self, mock_command, _which, tmp_path, caplog):
        """An explicit --max-workers x --num-threads exceeding the CPU count logs a warning."""
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
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        with patch("metaquest.cli.commands.sra.os.cpu_count", return_value=4):
            with caplog.at_level("WARNING"):
                result = command.execute(args)

        assert result == 0
        assert "exceeds" in caplog.text

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_does_not_warn_about_the_derived_default(self, mock_command, _which, tmp_path, caplog):
        """The CPU-derived default is this tool's own choice; only an explicit --max-workers is warned about."""
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
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=None,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        # One worker is already the floor of the derived default, so on a 2-CPU machine a
        # single 4-thread download oversubscribes and there is nothing the user could change.
        with patch("metaquest.cli.commands.sra.os.cpu_count", return_value=2):
            with caplog.at_level("WARNING"):
                result = command.execute(args)

        assert result == 0
        assert mock_command.call_args.kwargs["max_workers"] == 1
        assert "exceeds" not in caplog.text

    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_dry_run(self, mock_command, tmp_path):
        """Dry-run logs the plan (incl. blacklisted and max-downloads branches) and returns 0."""
        mock_command.return_value = {
            "total": 10,
            "to_download": 5,
            "already_downloaded": 3,
            "blacklisted": 2,
            "successful": 0,
            "failed": 0,
        }
        command = DownloadSraCommand()
        args = argparse.Namespace(
            accessions_file="accessions.txt",
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=5,
            num_threads=4,
            max_workers=4,
            dry_run=True,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=["bl.txt"],
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        result = command.execute(args)
        assert result == 0
        assert not (tmp_path / "metaquest_registry.json").exists()

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_with_failures_writes_failed_file(self, mock_command, _which, tmp_path, caplog):
        """Failed downloads return 1; the retry hint is logged and the CLI itself writes no file."""
        failed_file = tmp_path / "fastq" / FAILED_ACCESSIONS_FILE

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR999", False, "Download failed: t")
            return {
                "total": 3,
                "already_downloaded": 1,
                "blacklisted": 1,
                "successful": 1,
                "failed": 1,
                "failed_accessions": ["SRR999"],
            }

        mock_command.side_effect = fake_download_sra
        command = DownloadSraCommand()
        args = argparse.Namespace(
            accessions_file="accessions.txt",
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        with caplog.at_level("INFO"):
            result = command.execute(args)
        assert result == 1
        # The data layer (fake download_sra here) is the only writer of this file; the CLI
        # itself never creates it.
        assert not failed_file.exists()
        assert str(failed_file) in caplog.text
        assert "--accessions-file" in caplog.text

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_reports_disk_full_abort(self, mock_command, _which, tmp_path, caplog):
        """A disk-full abort logs the reason distinctly and returns 1, even though outcomes are recorded."""

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", False, "disk-full: No space left on device")
            kwargs["on_result"]("SRR2", False, "disk-full: not attempted")
            return {
                "total": 2,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 0,
                "failed": 2,
                "failed_accessions": ["SRR1", "SRR2"],
                "aborted": "disk-full",
            }

        mock_command.side_effect = fake_download_sra
        command = DownloadSraCommand()
        args = argparse.Namespace(
            accessions_file="accessions.txt",
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=2,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        with caplog.at_level("ERROR"):
            result = command.execute(args)

        assert result == 1
        assert "disk-full" in caplog.text

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_metaquest_error(self, mock_command, _which, tmp_path):
        """A MetaQuestError from the backend is caught and returns 1."""
        mock_command.side_effect = MetaQuestError("backend boom")
        command = DownloadSraCommand()
        args = argparse.Namespace(
            accessions_file="accessions.txt",
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        result = command.execute(args)
        assert result == 1

    @patch("metaquest.cli.commands.sra.shutil.which", return_value=None)
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_missing_fasterq_dump_exits_1(self, mock_download, _which, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            fastq_folder=str(tmp_path / "fastq"),
            accessions_file=str(acc),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )
        assert DownloadSraCommand().execute(args) == 1
        mock_download.assert_not_called()

    @patch("metaquest.cli.commands.sra.shutil.which", return_value=None)
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_dry_run_skips_tool_check(self, mock_download, _which, tmp_path):
        mock_download.return_value = {
            "total": 1,
            "already_downloaded": 0,
            "blacklisted": 0,
            "to_download": 1,
            "successful": 0,
            "failed": 0,
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            fastq_folder=str(tmp_path / "fastq"),
            accessions_file=str(acc),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=True,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )
        assert DownloadSraCommand().execute(args) == 0

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_report_file_lists_every_status(self, mock_download, _which, tmp_path):
        mock_download.return_value = {
            "total": 4,
            "already_downloaded": 1,
            "blacklisted": 1,
            "successful": 1,
            "failed": 1,
            "failed_accessions": ["SRR2"],
            "results": {"SRR1": "Downloaded 2 files", "SRR2": "Download failed: timeout"},
            "already_downloaded_accessions": ["SRR3"],
            "blacklisted_accessions": ["SRR4"],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\nSRR3\nSRR4\n")
        report = tmp_path / "reports" / "download_report.csv"
        args = argparse.Namespace(
            fastq_folder=str(tmp_path / "fastq"),
            accessions_file=str(acc),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=str(report),
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )
        DownloadSraCommand().execute(args)
        assert report.read_text().splitlines() == [
            "accession,status,message",
            "SRR1,downloaded,Downloaded 2 files",
            "SRR2,failed,Download failed: timeout",
            "SRR3,already_present,",
            "SRR4,blacklisted,",
        ]

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_execute_records_outcomes_in_registry(self, mock_download, _which, tmp_path):
        """Every accession's outcome ends up in the registry; the CLI never touches failed_accessions.txt."""
        fastq_folder = tmp_path / "fastq"
        (fastq_folder / "SRR1").mkdir(parents=True)
        (fastq_folder / "SRR1" / "SRR1_1.fastq").write_text("@r\nA\n+\nI\n")
        registry_path = tmp_path / "metaquest_registry.json"
        failed_file = fastq_folder / FAILED_ACCESSIONS_FILE

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, "Downloaded 2 files")
            kwargs["on_result"]("SRR2", False, "Download failed: t")
            failed_file.parent.mkdir(parents=True, exist_ok=True)
            failed_file.write_text("SRR2\n")
            return {
                "total": 5,
                "already_downloaded": 1,
                "blacklisted": 1,
                "successful": 1,
                "failed": 1,
                "failed_accessions": ["SRR2"],
                "results": {"SRR1": "Downloaded 2 files", "SRR2": "Download failed: t"},
                "already_downloaded_accessions": ["SRR3"],
                "blacklisted_accessions": ["SRR4"],
                "skipped_accessions": ["SRR5"],
            }

        mock_download.side_effect = fake_download_sra

        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\nSRR3\nSRR4\nSRR5\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_path),
            data_root=None,
        )

        result = DownloadSraCommand().execute(args)
        assert result == 1

        registry = json.loads(registry_path.read_text())
        datasets = registry["datasets"]

        assert datasets["SRR1"]["download"]["state"] == "downloaded"
        assert datasets["SRR1"]["download"]["files"][0]["path"].endswith("SRR1_1.fastq")
        assert "inferred" not in datasets["SRR1"]["download"]

        assert datasets["SRR2"]["download"]["state"] == "failed"
        assert datasets["SRR2"]["download"]["attempts"] == 1

        assert datasets["SRR3"]["download"]["state"] == "downloaded"
        assert "inferred" not in datasets["SRR3"]["download"]
        # Already present on disk, not a real attempt: attempts stays at 0.
        assert datasets["SRR3"]["download"]["attempts"] == 0

        assert datasets["SRR4"]["download"]["state"] == "skipped"
        assert datasets["SRR4"]["download"]["message"] == "blacklisted"

        assert datasets["SRR5"]["download"]["state"] == "skipped"
        assert datasets["SRR5"]["download"]["message"] == "--max-downloads"

        # The failed_accessions.txt written by the fake download_sra (standing in for the
        # data layer) is untouched by the CLI.
        assert failed_file.read_text() == "SRR2\n"

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_verdict_from_message_is_recorded_in_registry(self, mock_download, _which, tmp_path):
        """The verdict encoded in on_result's message ends up as download.complete in the registry."""
        fastq_folder = tmp_path / "fastq"
        registry_path = tmp_path / "metaquest_registry.json"

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, "Downloaded 1 files, truncated (300000 of 48000000 spots)")
            return {
                "total": 1,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 1,
                "failed": 0,
                "failed_accessions": [],
            }

        mock_download.side_effect = fake_download_sra
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_path),
            data_root=None,
        )

        assert DownloadSraCommand().execute(args) == 0
        datasets = json.loads(registry_path.read_text())["datasets"]
        assert datasets["SRR1"]["download"]["complete"]["verdict"] == "truncated"
        assert datasets["SRR1"]["download"]["complete"]["reads_r1"] == 300000
        assert datasets["SRR1"]["download"]["complete"]["expected_spots"] == 48000000

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_expected_spots_read_from_registry_metadata(self, mock_download, _which, tmp_path):
        """--verify-downloads (the default) reads run_total_spots from the registry into expected_spots."""
        registry_path = tmp_path / "metaquest_registry.json"
        seeded = load_registry(registry_path)
        record_metadata(seeded, "SRR1", tmp_path / "SRR1.xml", {"run_total_spots": 12345})
        save_registry(seeded)

        mock_download.return_value = {
            "total": 1,
            "already_downloaded": 0,
            "blacklisted": 0,
            "successful": 0,
            "failed": 0,
            "failed_accessions": [],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_path),
            data_root=None,
            verify_downloads=True,
        )

        DownloadSraCommand().execute(args)
        assert mock_download.call_args.kwargs["expected_spots"] == {"SRR1": 12345}

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_no_verify_downloads_passes_no_expected_spots(self, mock_download, _which, tmp_path):
        registry_path = tmp_path / "metaquest_registry.json"
        seeded = load_registry(registry_path)
        record_metadata(seeded, "SRR1", tmp_path / "SRR1.xml", {"run_total_spots": 12345})
        save_registry(seeded)

        mock_download.return_value = {
            "total": 1,
            "already_downloaded": 0,
            "blacklisted": 0,
            "successful": 0,
            "failed": 0,
            "failed_accessions": [],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_path),
            data_root=None,
            verify_downloads=False,
        )

        DownloadSraCommand().execute(args)
        assert mock_download.call_args.kwargs["expected_spots"] is None

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_redownload_truncated_computes_set_from_registry_verdicts(self, mock_download, _which, tmp_path):
        registry_path = tmp_path / "metaquest_registry.json"
        seeded = load_registry(registry_path)
        record_download(seeded, "SRR1", "downloaded", tmp_path / "fastq")
        seeded.datasets["SRR1"]["download"]["complete"] = {"verdict": "truncated"}
        record_download(seeded, "SRR2", "downloaded", tmp_path / "fastq")
        seeded.datasets["SRR2"]["download"]["complete"] = {"verdict": "complete"}
        save_registry(seeded)

        mock_download.return_value = {
            "total": 2,
            "already_downloaded": 0,
            "blacklisted": 0,
            "successful": 0,
            "failed": 0,
            "failed_accessions": [],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_path),
            data_root=None,
            redownload_truncated=True,
        )

        DownloadSraCommand().execute(args)
        assert mock_download.call_args.kwargs["truncated_accessions"] == {"SRR1"}
        assert mock_download.call_args.kwargs["redownload_truncated"] is True

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_blacklisted_accession_keeps_its_downloaded_record(self, mock_download, _which, tmp_path):
        """An accession downloaded earlier and blacklisted later keeps its files and sizes."""
        fastq_folder = tmp_path / "fastq"
        (fastq_folder / "SRR1").mkdir(parents=True)
        (fastq_folder / "SRR1" / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        registry_path = tmp_path / "metaquest_registry.json"

        seeded = load_registry(registry_path)
        record_download(seeded, "SRR1", "downloaded", fastq_folder)
        record_exclusion(seeded, "SRR1", "16S amplicon")
        save_registry(seeded)
        before = json.loads(registry_path.read_text())["datasets"]["SRR1"]["download"]

        mock_download.return_value = {
            "total": 1,
            "already_downloaded": 0,
            "blacklisted": 1,
            "successful": 0,
            "failed": 0,
            "failed_accessions": [],
            "blacklisted_accessions": ["SRR1"],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_path),
            data_root=None,
        )

        assert DownloadSraCommand().execute(args) == 0
        after = json.loads(registry_path.read_text())["datasets"]["SRR1"]["download"]
        assert after["state"] == "downloaded"
        assert after["files"] == before["files"]
        assert after["bytes_total"] == before["bytes_total"] > 0

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_concurrent_registry_edit_survives_the_run(self, mock_download, _which, tmp_path):
        """An exclusion written by another process mid-run is not reverted by the download's writes."""
        fastq_folder = tmp_path / "fastq"
        registry_path = tmp_path / "metaquest_registry.json"

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, "Downloaded 1 file")
            # Stand in for a second terminal running `metaquest blacklist --add SRR9`.
            other = load_registry(registry_path)
            record_exclusion(other, "SRR9", "16S amplicon")
            save_registry(other)
            kwargs["on_result"]("SRR2", True, "Downloaded 1 file")
            return {
                "total": 2,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 2,
                "failed": 0,
                "failed_accessions": [],
            }

        mock_download.side_effect = fake_download_sra
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_path),
            data_root=None,
        )

        assert DownloadSraCommand().execute(args) == 0
        datasets = json.loads(registry_path.read_text())["datasets"]
        assert datasets["SRR9"]["exclusion"]["excluded"] is True
        assert datasets["SRR1"]["download"]["state"] == "downloaded"
        assert datasets["SRR2"]["download"]["state"] == "downloaded"

    @patch("metaquest.cli.commands.sra.download_sra")
    def test_dry_run_resolves_and_logs_data_root(self, mock_download, tmp_path, caplog):
        """--dry-run --data-root resolves the store root (no fasterq-dump check needed) and logs it."""
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        init_store(store_root)
        mock_download.return_value = {
            "total": 1,
            "to_download": 1,
            "already_downloaded": 0,
            "successful": 0,
            "failed": 0,
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=True,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=str(store_root),
        )

        with caplog.at_level("INFO"):
            result = DownloadSraCommand().execute(args)

        assert result == 0
        assert any(str(store_root.resolve()) in message for message in caplog.messages)

    # ------------------------------------------------------------ shared store

    def test_configure_parser_store_flags(self):
        """--link-mode, --accept-partial and --no-resume-partial default sensibly and parse."""
        parser = argparse.ArgumentParser()
        DownloadSraCommand().configure_parser(parser)

        args = parser.parse_args(["--accessions-file", "a.txt"])
        assert args.link_mode == "auto"
        assert args.accept_partial is False
        assert args.resume_partial is True

        args = parser.parse_args(
            ["--accessions-file", "a.txt", "--link-mode", "copy", "--accept-partial", "--no-resume-partial"]
        )
        assert args.link_mode == "copy"
        assert args.accept_partial is True
        assert args.resume_partial is False

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_store_paths_and_link_options_reach_download_sra(self, mock_download, _which, tmp_path):
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        init_store(store_root)
        mock_download.return_value = {
            "total": 1,
            "to_download": 1,
            "already_downloaded": 0,
            "successful": 1,
            "failed": 0,
            "failed_accessions": [],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=str(store_root),
            link_mode="relative",
            accept_partial=True,
            resume_partial=False,
        )

        assert DownloadSraCommand().execute(args) == 0

        kwargs = mock_download.call_args.kwargs
        assert kwargs["store"].root == store_root.resolve()
        assert kwargs["link_mode"] == "relative"
        assert kwargs["accept_partial"] is True
        assert kwargs["resume_partial"] is False
        assert Path(kwargs["store_metadata"][0]) == tmp_path / "metadata"
        assert Path(kwargs["store_metadata"][1]) == store_root.resolve() / "metadata"

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_no_store_leaves_the_download_call_unchanged(self, mock_download, _which, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.delenv("METAQUEST_DATA", raising=False)
        mock_download.return_value = {
            "total": 1,
            "to_download": 1,
            "already_downloaded": 0,
            "successful": 1,
            "failed": 0,
            "failed_accessions": [],
        }
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        args = argparse.Namespace(
            accessions_file=str(acc),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        assert DownloadSraCommand().execute(args) == 0
        assert "store" not in mock_download.call_args.kwargs

    @pytest.mark.parametrize(
        "message,attempts",
        [("linked from store, 1 files", 0), ("Downloaded 1 files, complete (1 of 1 spots); stored", 1)],
    )
    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_store_backed_results_record_source_and_linked_list(
        self, mock_download, _which, tmp_path, message, attempts
    ):
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = paths.sra / "SRR1"
        acc_dir.mkdir(parents=True)
        (acc_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        os.symlink(acc_dir, fastq_folder / "SRR1")

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, message)
            return {
                "total": 1,
                "to_download": 1,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 1,
                "failed": 0,
                "failed_accessions": [],
                "results": {"SRR1": message},
            }

        mock_download.side_effect = fake_download_sra
        registry_file = tmp_path / "metaquest_registry.json"
        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_file),
            data_root=str(store_root),
        )

        assert DownloadSraCommand().execute(args) == 0

        written = json.loads(registry_file.read_text())
        download = written["datasets"]["SRR1"]["download"]
        assert download["state"] == "downloaded"
        assert download["source"] == "store"
        assert download["store_name"] == "SRR1"
        assert download["attempts"] == attempts
        assert written["store"]["linked"] == ["SRR1"]

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_a_plain_download_is_not_recorded_as_store_backed(self, mock_download, _which, tmp_path):
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        init_store(store_root)
        fastq_folder = tmp_path / "fastq"
        (fastq_folder / "SRR1").mkdir(parents=True)
        (fastq_folder / "SRR1" / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, "Downloaded 1 files, unverified")
            return {
                "total": 1,
                "to_download": 1,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 1,
                "failed": 0,
                "failed_accessions": [],
                "results": {"SRR1": "Downloaded 1 files, unverified"},
            }

        mock_download.side_effect = fake_download_sra
        registry_file = tmp_path / "metaquest_registry.json"
        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_file),
            data_root=str(store_root),
        )

        assert DownloadSraCommand().execute(args) == 0

        written = json.loads(registry_file.read_text())
        assert "source" not in written["datasets"]["SRR1"]["download"]
        assert written["store"].get("linked", []) == []

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_an_already_linked_dataset_is_recorded_as_store_backed(self, mock_download, _which, tmp_path):
        """A link left by an earlier run is recorded with its source, not as a plain download."""
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = paths.sra / "SRR1"
        acc_dir.mkdir(parents=True)
        (acc_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        os.symlink(acc_dir, fastq_folder / "SRR1")

        mock_download.return_value = {
            "total": 1,
            "to_download": 0,
            "already_downloaded": 1,
            "blacklisted": 0,
            "successful": 0,
            "failed": 0,
            "failed_accessions": [],
            "results": {},
            "already_downloaded_accessions": ["SRR1"],
        }
        registry_file = tmp_path / "metaquest_registry.json"
        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_file),
            data_root=str(store_root),
        )

        assert DownloadSraCommand().execute(args) == 0

        written = json.loads(registry_file.read_text())
        assert written["datasets"]["SRR1"]["download"]["source"] == "store"
        assert written["store"]["linked"] == ["SRR1"]

    # -------------------------------------------------------- store usage catalogue

    @staticmethod
    def _seed_project(registry_path, project_id="proj1"):
        from metaquest.data.registry import load_registry as _load, save_registry as _save

        registry = _load(registry_path)
        registry.project = {"id": project_id, "name": "demo", "path": str(registry_path.parent), "created": "now"}
        _save(registry)

    @staticmethod
    def _usage_rows(store_root):
        from metaquest.store.catalog import Catalog
        from metaquest.store.layout import store_paths

        with Catalog(store_paths(store_root)) as catalog:
            catalog.migrate()
            return [
                dict(r)
                for r in catalog.conn.execute("SELECT accession, project_id, genome_id, stage FROM usage").fetchall()
            ]

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_fresh_store_download_records_usage_stage_downloaded(self, mock_download, _which, tmp_path):
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        init_store(store_root)
        registry_file = tmp_path / "metaquest_registry.json"
        self._seed_project(registry_file)
        message = "Downloaded 1 files, complete (1 of 1 spots); stored"

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, message)
            return {
                "total": 1,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 1,
                "failed": 0,
                "failed_accessions": [],
                "results": {"SRR1": message},
            }

        mock_download.side_effect = fake_download_sra
        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_file),
            data_root=str(store_root),
        )

        assert DownloadSraCommand().execute(args) == 0

        rows = self._usage_rows(store_root)
        assert rows == [{"accession": "SRR1", "project_id": "proj1", "genome_id": "", "stage": "downloaded"}]

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_linked_from_store_records_usage_stage_linked(self, mock_download, _which, tmp_path):
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        init_store(store_root)
        registry_file = tmp_path / "metaquest_registry.json"
        self._seed_project(registry_file)
        message = "linked from store, 1 files"

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, message)
            return {
                "total": 1,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 1,
                "failed": 0,
                "failed_accessions": [],
                "results": {"SRR1": message},
            }

        mock_download.side_effect = fake_download_sra
        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_file),
            data_root=str(store_root),
        )

        assert DownloadSraCommand().execute(args) == 0

        rows = self._usage_rows(store_root)
        assert rows == [{"accession": "SRR1", "project_id": "proj1", "genome_id": "", "stage": "linked"}]

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_already_downloaded_store_backed_records_usage_stage_linked(self, mock_download, _which, tmp_path):
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = paths.sra / "SRR1"
        acc_dir.mkdir(parents=True)
        (acc_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        fastq_folder = tmp_path / "fastq"
        fastq_folder.mkdir()
        os.symlink(acc_dir, fastq_folder / "SRR1")

        registry_file = tmp_path / "metaquest_registry.json"
        self._seed_project(registry_file)

        mock_download.return_value = {
            "total": 1,
            "to_download": 0,
            "already_downloaded": 1,
            "blacklisted": 0,
            "successful": 0,
            "failed": 0,
            "failed_accessions": [],
            "results": {},
            "already_downloaded_accessions": ["SRR1"],
        }
        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_file),
            data_root=str(store_root),
        )

        assert DownloadSraCommand().execute(args) == 0

        rows = self._usage_rows(store_root)
        assert rows == [{"accession": "SRR1", "project_id": "proj1", "genome_id": "", "stage": "linked"}]

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_catalog_failure_leaves_download_outcome_unchanged(self, mock_download, _which, tmp_path):
        """A broken catalogue write never changes the download's exit code or registry record."""
        from metaquest.store.layout import init_store

        store_root = tmp_path / "store"
        init_store(store_root)
        registry_file = tmp_path / "metaquest_registry.json"
        self._seed_project(registry_file)
        message = "Downloaded 1 files, complete (1 of 1 spots); stored"

        def fake_download_sra(**kwargs):
            kwargs["on_result"]("SRR1", True, message)
            return {
                "total": 1,
                "already_downloaded": 0,
                "blacklisted": 0,
                "successful": 1,
                "failed": 0,
                "failed_accessions": [],
                "results": {"SRR1": message},
            }

        mock_download.side_effect = fake_download_sra
        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(tmp_path / "fastq"),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(registry_file),
            data_root=str(store_root),
        )

        with patch("metaquest.store.usage.catalog_write", side_effect=RuntimeError("catalogue is locked")):
            result = DownloadSraCommand().execute(args)

        assert result == 0
        written = json.loads(registry_file.read_text())
        download = written["datasets"]["SRR1"]["download"]
        assert download["state"] == "downloaded"
        assert download["source"] == "store"
        assert download["store_name"] == "SRR1"
        assert written["store"]["linked"] == ["SRR1"]

    # ------------------------------------------------------- transient bytes warning

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_warns_when_transient_bytes_exceed_threshold(self, mock_download, _which, tmp_path, caplog, monkeypatch):
        """A kept .sra-cache bigger than the (patched, small) threshold is named in a warning."""
        monkeypatch.setattr("metaquest.cli.commands.sra.TRANSIENT_BYTES_WARN_THRESHOLD", 10)
        mock_download.return_value = {
            "total": 1,
            "already_downloaded": 0,
            "blacklisted": 0,
            "successful": 1,
            "failed": 0,
            "failed_accessions": [],
        }
        fastq_folder = tmp_path / "fastq"
        cache = fastq_folder / ".sra-cache"
        cache.mkdir(parents=True)
        (cache / "SRR1.sra").write_bytes(b"x" * 100)

        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        with caplog.at_level("WARNING"):
            assert DownloadSraCommand().execute(args) == 0

        assert str(fastq_folder) in caplog.text
        assert "100 bytes" in caplog.text

    @patch("metaquest.cli.commands.sra.shutil.which", return_value="/usr/bin/fasterq-dump")
    @patch("metaquest.cli.commands.sra.download_sra")
    def test_no_warning_when_transient_bytes_under_threshold(self, mock_download, _which, tmp_path, caplog):
        """The default 1 GB threshold is not tripped by a small leftover cache file."""
        mock_download.return_value = {
            "total": 1,
            "already_downloaded": 0,
            "blacklisted": 0,
            "successful": 1,
            "failed": 0,
            "failed_accessions": [],
        }
        fastq_folder = tmp_path / "fastq"
        cache = fastq_folder / ".sra-cache"
        cache.mkdir(parents=True)
        (cache / "SRR1.sra").write_bytes(b"x" * 100)

        args = argparse.Namespace(
            accessions_file=str(tmp_path / "acc.txt"),
            fastq_folder=str(fastq_folder),
            max_downloads=None,
            num_threads=4,
            max_workers=4,
            dry_run=False,
            force=False,
            max_retries=1,
            temp_folder=None,
            blacklist=None,
            report_file=None,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        with caplog.at_level("WARNING"):
            assert DownloadSraCommand().execute(args) == 0

        assert "transient" not in caplog.text.lower()


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
        assert args.metadata_file is None
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

    @patch("metaquest.cli.commands.samples.resolve_metadata_table")
    @patch("metaquest.cli.commands.samples.count_single_sample")
    def test_execute(self, mock_command, mock_resolve):
        """Test command execution."""
        mock_command.return_value = {"organism1": 5, "organism2": 3}
        mock_resolve.return_value = Path("metadata.txt")
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
        mock_resolve.assert_called_once_with("metadata.txt")
        mock_command.assert_called_once_with(
            summary_file="summary.txt",
            metadata_file="metadata.txt",
            summary_column="test_genome",
            metadata_column="organism",
            threshold=0.1,
            top_n=100,
        )


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
    def test_execute(self, mock_command, tmp_path):
        """Test command execution."""
        mock_command.return_value = pd.DataFrame()
        command = ParseMetadataCommand()

        args = argparse.Namespace(
            metadata_folder="metadata",
            metadata_table_file="metadata_table.txt",
            registry=str(tmp_path / "metaquest_registry.json"),
        )

        result = command.execute(args)
        assert result == 0
        mock_command.assert_called_once_with("metadata", "metadata_table.txt")

    def test_execute_records_metadata_in_registry(self, tmp_path):
        """Parsing a metadata folder records run_size/run_md5 for each accession."""
        metadata_folder = tmp_path / "metadata"
        metadata_folder.mkdir()
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <SAMPLE>
                    <IDENTIFIERS>
                        <PRIMARY_ID>SAMN123</PRIMARY_ID>
                    </IDENTIFIERS>
                    <SAMPLE_NAME>
                        <SCIENTIFIC_NAME>Escherichia coli</SCIENTIFIC_NAME>
                    </SAMPLE_NAME>
                </SAMPLE>
                <RUN_SET>
                    <RUN>
                        <IDENTIFIERS>
                            <PRIMARY_ID>SRR123</PRIMARY_ID>
                        </IDENTIFIERS>
                        <size>12345</size>
                        <md5>abcdef0123456789</md5>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""
        (metadata_folder / "SRR123_metadata.xml").write_text(xml_content)

        command = ParseMetadataCommand()
        args = argparse.Namespace(
            metadata_folder=str(metadata_folder),
            metadata_table_file=str(tmp_path / "metadata_table.txt"),
            registry=str(tmp_path / "metaquest_registry.json"),
        )

        result = command.execute(args)
        assert result == 0

        registry = json.loads((tmp_path / "metaquest_registry.json").read_text())
        metadata = registry["datasets"]["SRR123"]["metadata"]
        assert metadata["run_size"] == "12345"
        assert metadata["run_md5"] == "abcdef0123456789"

    def test_execute_records_ncbi_attribute_spots_in_registry(self, tmp_path):
        """Real NCBI efetch XML holds spots/bases/md5 as attributes; parse_metadata must record them."""
        metadata_folder = tmp_path / "metadata"
        metadata_folder.mkdir()
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <EXPERIMENT>
                    <IDENTIFIERS>
                        <PRIMARY_ID>EXP1</PRIMARY_ID>
                    </IDENTIFIERS>
                    <LIBRARY_DESCRIPTOR>
                        <LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY>
                        <LIBRARY_LAYOUT>
                            <PAIRED/>
                        </LIBRARY_LAYOUT>
                    </LIBRARY_DESCRIPTOR>
                    <PLATFORM>
                        <ILLUMINA>
                            <INSTRUMENT_MODEL>Illumina HiSeq 2500</INSTRUMENT_MODEL>
                        </ILLUMINA>
                    </PLATFORM>
                </EXPERIMENT>
                <SAMPLE>
                    <IDENTIFIERS>
                        <PRIMARY_ID>SAMN123</PRIMARY_ID>
                    </IDENTIFIERS>
                </SAMPLE>
                <RUN_SET>
                    <RUN accession="SRR1" total_spots="47964651" total_bases="14389395300" size="4744553813">
                        <IDENTIFIERS>
                            <PRIMARY_ID>SRR1</PRIMARY_ID>
                        </IDENTIFIERS>
                        <SRAFiles>
                            <SRAFile filename="SRR1" md5="abc" semantic_name="run"/>
                        </SRAFiles>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""
        (metadata_folder / "SRR1_metadata.xml").write_text(xml_content)

        command = ParseMetadataCommand()
        args = argparse.Namespace(
            metadata_folder=str(metadata_folder),
            metadata_table_file=str(tmp_path / "metadata_table.txt"),
            registry=str(tmp_path / "metaquest_registry.json"),
        )

        result = command.execute(args)
        assert result == 0

        registry = json.loads((tmp_path / "metaquest_registry.json").read_text())
        metadata = registry["datasets"]["SRR1"]["metadata"]
        assert metadata["run_total_spots"] == 47964651
        assert metadata["run_total_bases"] == 14389395300
        assert metadata["run_size"] == "4744553813"
        assert metadata["run_md5"] == "abc"
        assert metadata["library_layout"] == "PAIRED"
        assert metadata["platform"] == "ILLUMINA"
        assert metadata["library_strategy"] == "WGS"


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
        assert args.metadata_file is None
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

    @patch("metaquest.cli.commands.metadata.resolve_metadata_table")
    @patch("metaquest.cli.commands.metadata.count_metadata")
    def test_execute(self, mock_command, mock_resolve):
        """Test command execution."""
        mock_command.return_value = None
        mock_resolve.return_value = Path("metadata.txt")
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
        mock_resolve.assert_called_once_with("metadata.txt")
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
