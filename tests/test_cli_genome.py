"""Tests for genome CLI commands (search, download, prepare)."""

import argparse
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from metaquest.cli.commands.genome import (
    GenomeSearchCommand,
    GenomeDownloadCommand,
    GenomePrepareCommand,
)
from metaquest.core.exceptions import MetaQuestError


class TestGenomeSearchCommand:
    """Tests for GenomeSearchCommand."""

    def test_command_properties(self):
        cmd = GenomeSearchCommand()
        assert cmd.name == "genome_search"
        assert "search" in cmd.help.lower()

    def test_configure_parser(self):
        cmd = GenomeSearchCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args(["--species", "Salmonella enterica"])
        assert args.species == "Salmonella enterica"
        assert args.genus is None
        assert args.representative_only is True
        assert args.output_format == "list"

    def test_configure_parser_genus(self):
        cmd = GenomeSearchCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args(["--genus", "Salmonella"])
        assert args.genus == "Salmonella"
        assert args.species is None

    def test_configure_parser_mutually_exclusive(self):
        cmd = GenomeSearchCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--species", "Foo", "--genus", "Bar"])

    def test_configure_parser_missing_required(self):
        cmd = GenomeSearchCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_configure_parser_all_flag(self):
        cmd = GenomeSearchCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args(["--species", "Foo", "--all"])
        assert args.representative_only is False

    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_species_search(self, mock_get):
        mock_get.return_value = ["GCF_000006945.2", "GCF_000007545.1"]
        cmd = GenomeSearchCommand()
        args = argparse.Namespace(
            species="Salmonella enterica",
            genus=None,
            output=None,
            representative_only=True,
            output_format="list",
        )
        result = cmd.execute(args)
        assert result == 0
        mock_get.assert_called_once_with("Salmonella enterica", representative_only=True)

    @patch("metaquest.cli.commands.genome.get_accessions_for_genus")
    def test_execute_genus_search(self, mock_get):
        mock_get.return_value = ["GCF_000006945.2"]
        cmd = GenomeSearchCommand()
        args = argparse.Namespace(
            species=None,
            genus="Salmonella",
            output=None,
            representative_only=True,
            output_format="list",
        )
        result = cmd.execute(args)
        assert result == 0
        mock_get.assert_called_once_with("Salmonella", representative_only=True)

    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_output_to_file(self, mock_get):
        mock_get.return_value = ["GCF_000006945.2"]
        cmd = GenomeSearchCommand()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            outpath = f.name

        try:
            args = argparse.Namespace(
                species="Salmonella enterica",
                genus=None,
                output=outpath,
                representative_only=True,
                output_format="list",
            )
            result = cmd.execute(args)
            assert result == 0
            content = Path(outpath).read_text()
            assert "GCF_000006945.2" in content
        finally:
            Path(outpath).unlink(missing_ok=True)

    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_tsv_format(self, mock_get):
        mock_get.return_value = ["GCF_000006945.2"]
        cmd = GenomeSearchCommand()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            outpath = f.name

        try:
            args = argparse.Namespace(
                species="Salmonella enterica",
                genus=None,
                output=outpath,
                representative_only=True,
                output_format="tsv",
            )
            result = cmd.execute(args)
            assert result == 0
            content = Path(outpath).read_text()
            assert "GCF_000006945.2\tSalmonella enterica" in content
        finally:
            Path(outpath).unlink(missing_ok=True)

    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_no_results(self, mock_get):
        mock_get.return_value = []
        cmd = GenomeSearchCommand()
        args = argparse.Namespace(
            species="Nonexistent species",
            genus=None,
            output=None,
            representative_only=True,
            output_format="list",
        )
        result = cmd.execute(args)
        assert result == 0

    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_api_failure(self, mock_get):
        mock_get.side_effect = MetaQuestError("API error")
        cmd = GenomeSearchCommand()
        args = argparse.Namespace(
            species="Salmonella enterica",
            genus=None,
            output=None,
            representative_only=True,
            output_format="list",
        )
        result = cmd.execute(args)
        assert result == 1


class TestGenomeDownloadCommand:
    """Tests for GenomeDownloadCommand."""

    def test_command_properties(self):
        cmd = GenomeDownloadCommand()
        assert cmd.name == "genome_download"
        assert "download" in cmd.help.lower()

    def test_configure_parser(self):
        cmd = GenomeDownloadCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args(["--accessions", "GCF_000006945.2"])
        assert args.accessions == ["GCF_000006945.2"]
        assert args.output_dir == "genomes/"
        assert args.representative_only is True

    def test_configure_parser_multiple_accessions(self):
        cmd = GenomeDownloadCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args(["--accessions", "GCF_000006945.2", "GCF_000007545.1"])
        assert len(args.accessions) == 2

    def test_execute_missing_all_sources(self):
        cmd = GenomeDownloadCommand()
        args = argparse.Namespace(
            accessions=None,
            accession_file=None,
            species=None,
            genus=None,
            output_dir="genomes/",
            representative_only=True,
            assembly_level=None,
        )
        result = cmd.execute(args)
        assert result == 1

    @patch("metaquest.cli.commands.genome.extract_and_organize")
    @patch("metaquest.cli.commands.genome.download_genomes")
    def test_execute_with_accessions(self, mock_download, mock_extract):
        mock_download.return_value = Path("genomes/download.zip")
        mock_extract.return_value = {"GCF_000006945.2": Path("genomes/GCF_000006945.2.fna.gz")}
        cmd = GenomeDownloadCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                accessions=["GCF_000006945.2"],
                accession_file=None,
                species=None,
                genus=None,
                output_dir=tmpdir,
                representative_only=True,
                assembly_level=None,
            )
            result = cmd.execute(args)
            assert result == 0
            mock_download.assert_called_once()
            mock_extract.assert_called_once()

    @patch("metaquest.cli.commands.genome.extract_and_organize")
    @patch("metaquest.cli.commands.genome.download_genomes")
    def test_execute_with_accession_file(self, mock_download, mock_extract):
        mock_download.return_value = Path("genomes/download.zip")
        mock_extract.return_value = {"GCF_000006945.2": Path("genomes/GCF_000006945.2.fna.gz")}
        cmd = GenomeDownloadCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            acc_file = Path(tmpdir) / "accessions.txt"
            acc_file.write_text("GCF_000006945.2\nGCF_000007545.1\n")
            args = argparse.Namespace(
                accessions=None,
                accession_file=str(acc_file),
                species=None,
                genus=None,
                output_dir=tmpdir,
                representative_only=True,
                assembly_level=None,
            )
            result = cmd.execute(args)
            assert result == 0

    @patch("metaquest.cli.commands.genome.extract_and_organize")
    @patch("metaquest.cli.commands.genome.download_genomes")
    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_with_species(self, mock_gtdb, mock_download, mock_extract):
        mock_gtdb.return_value = ["GCF_000006945.2"]
        mock_download.return_value = Path("genomes/download.zip")
        mock_extract.return_value = {"GCF_000006945.2": Path("genomes/GCF_000006945.2.fna.gz")}
        cmd = GenomeDownloadCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                accessions=None,
                accession_file=None,
                species="Salmonella enterica",
                genus=None,
                output_dir=tmpdir,
                representative_only=True,
                assembly_level=None,
            )
            result = cmd.execute(args)
            assert result == 0
            mock_gtdb.assert_called_once_with("Salmonella enterica", representative_only=True)

    def test_execute_accession_file_not_found(self):
        cmd = GenomeDownloadCommand()
        args = argparse.Namespace(
            accessions=None,
            accession_file="/nonexistent/file.txt",
            species=None,
            genus=None,
            output_dir="genomes/",
            representative_only=True,
            assembly_level=None,
        )
        result = cmd.execute(args)
        assert result == 1

    @patch("metaquest.cli.commands.genome.download_genomes")
    def test_execute_download_failure(self, mock_download):
        mock_download.side_effect = MetaQuestError("Download failed")
        cmd = GenomeDownloadCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                accessions=["GCF_000006945.2"],
                accession_file=None,
                species=None,
                genus=None,
                output_dir=tmpdir,
                representative_only=True,
                assembly_level=None,
            )
            result = cmd.execute(args)
            assert result == 1


class TestGenomePrepareCommand:
    """Tests for GenomePrepareCommand."""

    def test_command_properties(self):
        cmd = GenomePrepareCommand()
        assert cmd.name == "genome_prepare"
        assert "manifest" in cmd.help.lower() or "prepare" in cmd.help.lower()

    def test_configure_parser(self):
        cmd = GenomePrepareCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args(["--species", "Salmonella enterica"])
        assert args.species == "Salmonella enterica"
        assert args.manifest_file == "genome_manifest.csv"
        assert args.skip_download is False

    @patch("metaquest.cli.commands.genome.extract_and_organize")
    @patch("metaquest.cli.commands.genome.download_genomes")
    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_full_pipeline(self, mock_gtdb, mock_download, mock_extract):
        mock_gtdb.return_value = ["GCF_000006945.2"]
        mock_download.return_value = Path("download.zip")
        mock_extract.return_value = {"GCF_000006945.2": Path("GCF_000006945.2.fna.gz")}
        cmd = GenomePrepareCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake genome file so manifest picks it up
            genome_file = Path(tmpdir) / "GCF_000006945.2.fna.gz"
            genome_file.touch()
            manifest = Path(tmpdir) / "manifest.csv"
            args = argparse.Namespace(
                species="Salmonella enterica",
                genus=None,
                accession_file=None,
                output_dir=tmpdir,
                representative_only=True,
                manifest_file=str(manifest),
                skip_download=False,
            )
            result = cmd.execute(args)
            assert result == 0
            assert manifest.exists()
            content = manifest.read_text()
            assert "name,genome_filename,protein_filename" in content
            assert "GCF_000006945.2" in content

    def test_execute_skip_download(self):
        cmd = GenomePrepareCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake genome files
            (Path(tmpdir) / "genome1.fna.gz").touch()
            (Path(tmpdir) / "genome2.fna.gz").touch()
            manifest = Path(tmpdir) / "manifest.csv"
            args = argparse.Namespace(
                species=None,
                genus=None,
                accession_file=None,
                output_dir=tmpdir,
                representative_only=True,
                manifest_file=str(manifest),
                skip_download=True,
            )
            result = cmd.execute(args)
            assert result == 0
            assert manifest.exists()
            with open(manifest) as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert len(rows) == 3  # header + 2 entries

    def test_execute_missing_sources_no_skip(self):
        cmd = GenomePrepareCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                species=None,
                genus=None,
                accession_file=None,
                output_dir=tmpdir,
                representative_only=True,
                manifest_file="manifest.csv",
                skip_download=False,
            )
            result = cmd.execute(args)
            assert result == 1

    @patch("metaquest.cli.commands.genome.download_genomes")
    @patch("metaquest.cli.commands.genome.get_accessions_for_species")
    def test_execute_api_failure(self, mock_gtdb, mock_download):
        mock_gtdb.side_effect = MetaQuestError("API error")
        cmd = GenomePrepareCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                species="Salmonella enterica",
                genus=None,
                accession_file=None,
                output_dir=tmpdir,
                representative_only=True,
                manifest_file="manifest.csv",
                skip_download=False,
            )
            result = cmd.execute(args)
            assert result == 1

    def test_execute_skip_download_no_genomes(self):
        cmd = GenomePrepareCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.csv"
            args = argparse.Namespace(
                species=None,
                genus=None,
                accession_file=None,
                output_dir=tmpdir,
                representative_only=True,
                manifest_file=str(manifest),
                skip_download=True,
            )
            result = cmd.execute(args)
            assert result == 0

    @patch("metaquest.cli.commands.genome.extract_and_organize")
    @patch("metaquest.cli.commands.genome.download_genomes")
    @patch("metaquest.cli.commands.genome.get_accessions_for_genus")
    def test_execute_with_genus(self, mock_gtdb, mock_download, mock_extract):
        mock_gtdb.return_value = ["GCF_000006945.2", "GCF_000007545.1"]
        mock_download.return_value = Path("download.zip")
        mock_extract.return_value = {}
        cmd = GenomePrepareCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.csv"
            args = argparse.Namespace(
                species=None,
                genus="Salmonella",
                accession_file=None,
                output_dir=tmpdir,
                representative_only=True,
                manifest_file=str(manifest),
                skip_download=False,
            )
            result = cmd.execute(args)
            assert result == 0
            mock_gtdb.assert_called_once_with("Salmonella", representative_only=True)

    @patch("metaquest.cli.commands.genome.extract_and_organize")
    @patch("metaquest.cli.commands.genome.download_genomes")
    def test_execute_with_accession_file(self, mock_download, mock_extract):
        mock_download.return_value = Path("download.zip")
        mock_extract.return_value = {}
        cmd = GenomePrepareCommand()
        with tempfile.TemporaryDirectory() as tmpdir:
            acc_file = Path(tmpdir) / "accessions.txt"
            acc_file.write_text("GCF_000006945.2\n# comment\nGCF_000007545.1\n")
            manifest = Path(tmpdir) / "manifest.csv"
            args = argparse.Namespace(
                species=None,
                genus=None,
                accession_file=str(acc_file),
                output_dir=tmpdir,
                representative_only=True,
                manifest_file=str(manifest),
                skip_download=False,
            )
            result = cmd.execute(args)
            assert result == 0
