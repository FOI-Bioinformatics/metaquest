"""Tests for the genome download module."""

import csv
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.data.genome_download import (
    _validate_genome_accession,
    check_datasets_available,
    create_genome_manifest,
    download_from_file,
    download_genomes,
    extract_and_organize,
    read_accession_file,
)

# --- _validate_genome_accession ---


class TestValidateGenomeAccession:
    def test_valid_gcf_accession(self):
        assert _validate_genome_accession("GCF_000005845.2") == "GCF_000005845.2"

    def test_valid_gca_accession(self):
        assert _validate_genome_accession("GCA_000005845.2") == "GCA_000005845.2"

    def test_invalid_prefix(self):
        with pytest.raises(DataAccessError, match="Invalid genome accession prefix"):
            _validate_genome_accession("SRR1234567")

    def test_invalid_format_no_version(self):
        with pytest.raises(DataAccessError, match="Invalid genome accession format"):
            _validate_genome_accession("GCF_000005845")

    def test_invalid_format_short_digits(self):
        with pytest.raises(DataAccessError, match="Invalid genome accession format"):
            _validate_genome_accession("GCF_12345.1")

    def test_empty_string(self):
        with pytest.raises(DataAccessError, match="Invalid genome accession prefix"):
            _validate_genome_accession("")


# --- check_datasets_available ---


class TestCheckDatasetsAvailable:
    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_available(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert check_datasets_available() is True
        mock_run.assert_called_once_with("datasets", ["--version"])

    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_not_available(self, mock_run):
        mock_run.side_effect = FileNotFoundError("datasets not found")
        assert check_datasets_available() is False

    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_security_error(self, mock_run):
        from metaquest.core.exceptions import SecurityError

        mock_run.side_effect = SecurityError("not allowed")
        assert check_datasets_available() is False


# --- download_genomes ---


class TestDownloadGenomes:
    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_single_accession(self, mock_run, tmp_path):
        # Create the zip file that would be produced by the command
        zip_path = tmp_path / "ncbi_dataset.zip"
        zip_path.touch()

        mock_run.return_value = MagicMock(returncode=0)

        result = download_genomes(["GCF_000005845.2"], tmp_path)

        assert result == zip_path
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        assert call_args[0] == "datasets"
        assert "GCF_000005845.2" in call_args[1]
        assert "--include" in call_args[1]
        assert "genome" in call_args[1]

    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_multiple_accessions(self, mock_run, tmp_path):
        zip_path = tmp_path / "ncbi_dataset.zip"
        zip_path.touch()

        mock_run.return_value = MagicMock(returncode=0)
        accessions = ["GCF_000005845.2", "GCA_000001405.1"]

        result = download_genomes(accessions, tmp_path)

        assert result == zip_path
        call_args = mock_run.call_args[0][1]
        assert "GCF_000005845.2" in call_args
        assert "GCA_000001405.1" in call_args

    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_with_assembly_level(self, mock_run, tmp_path):
        zip_path = tmp_path / "ncbi_dataset.zip"
        zip_path.touch()

        mock_run.return_value = MagicMock(returncode=0)

        download_genomes(["GCF_000005845.2"], tmp_path, assembly_level="complete")

        call_args = mock_run.call_args[0][1]
        assert "--assembly-level" in call_args
        assert "complete" in call_args

    def test_empty_accessions(self, tmp_path):
        with pytest.raises(DataAccessError, match="No accessions provided"):
            download_genomes([], tmp_path)

    def test_invalid_accession(self, tmp_path):
        with pytest.raises(DataAccessError, match="Invalid genome accession"):
            download_genomes(["INVALID_123"], tmp_path)

    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_download_failure(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, "datasets", stderr="Connection failed")
        with pytest.raises(DataAccessError, match="Failed to download genomes"):
            download_genomes(["GCF_000005845.2"], tmp_path)

    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_zip_not_created(self, mock_run, tmp_path):
        # Simulate command success but no zip produced
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(DataAccessError, match="zip file was not created"):
            download_genomes(["GCF_000005845.2"], tmp_path)


# --- download_from_file ---


class TestDownloadFromFile:
    @patch("metaquest.data.genome_download.SecureSubprocess.run_secure")
    def test_download_from_file(self, mock_run, tmp_path):
        acc_file = tmp_path / "accessions.txt"
        acc_file.write_text("GCF_000005845.2\nGCA_000001405.1\n")

        output_dir = tmp_path / "output"
        zip_path = output_dir / "ncbi_dataset.zip"

        def create_zip(*args, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            zip_path.touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = create_zip

        result = download_from_file(acc_file, output_dir)

        assert result == zip_path
        call_args = mock_run.call_args[0][1]
        assert "--inputfile" in call_args
        assert str(acc_file) in call_args

    def test_missing_file(self, tmp_path):
        with pytest.raises(DataAccessError, match="Accession file not found"):
            download_from_file(tmp_path / "nonexistent.txt", tmp_path / "out")

    def test_empty_file(self, tmp_path):
        acc_file = tmp_path / "empty.txt"
        acc_file.write_text("\n\n")

        with pytest.raises(DataAccessError, match="No valid accessions"):
            download_from_file(acc_file, tmp_path / "out")

    def test_file_with_invalid_accessions(self, tmp_path):
        acc_file = tmp_path / "bad.txt"
        acc_file.write_text("INVALID_123\n")

        with pytest.raises(DataAccessError, match="Invalid genome accession"):
            download_from_file(acc_file, tmp_path / "out")


# --- extract_and_organize ---


def _create_mock_ncbi_zip(zip_path, accessions_and_files):
    """Helper to create a mock NCBI datasets zip file.

    Args:
        zip_path: Path for the zip file.
        accessions_and_files: Dict mapping accession to FASTA content.
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        for accession, content in accessions_and_files.items():
            fasta_name = f"ncbi_dataset/data/{accession}/{accession}_genomic.fna"
            zf.writestr(fasta_name, content)


class TestExtractAndOrganize:
    def test_single_genome(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        _create_mock_ncbi_zip(zip_path, {"GCF_000005845.2": ">seq1\nATCG\n"})

        output_dir = tmp_path / "genomes"
        result = extract_and_organize(zip_path, output_dir)

        assert "GCF_000005845.2" in result
        fasta = result["GCF_000005845.2"]
        assert fasta.exists()
        assert fasta.name == "GCF_000005845.2.fna"
        assert ">seq1" in fasta.read_text()

    def test_multiple_genomes(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        _create_mock_ncbi_zip(
            zip_path,
            {
                "GCF_000005845.2": ">ecoli\nATCG\n",
                "GCA_000001405.1": ">human\nGGCC\n",
            },
        )

        output_dir = tmp_path / "genomes"
        result = extract_and_organize(zip_path, output_dir)

        assert len(result) == 2
        assert "GCF_000005845.2" in result
        assert "GCA_000001405.1" in result

    def test_skips_non_accession_dirs(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_genomic.fna",
                ">seq\nATCG\n",
            )
            zf.writestr(
                "ncbi_dataset/data/assembly_data_report.jsonl",
                '{"report": true}\n',
            )

        output_dir = tmp_path / "genomes"
        result = extract_and_organize(zip_path, output_dir)

        assert len(result) == 1
        assert "GCF_000005845.2" in result

    def test_missing_zip(self, tmp_path):
        with pytest.raises(DataAccessError, match="Zip file not found"):
            extract_and_organize(tmp_path / "missing.zip", tmp_path / "out")

    def test_invalid_zip(self, tmp_path):
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_text("not a zip file")

        with pytest.raises(DataAccessError, match="Not a valid zip file"):
            extract_and_organize(bad_zip, tmp_path / "out")

    def test_corrupt_zip(self, tmp_path):
        bad_zip = tmp_path / "corrupt.zip"
        # Write something that looks like a zip header but is corrupt
        bad_zip.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        with pytest.raises(DataAccessError):
            extract_and_organize(bad_zip, tmp_path / "out")

    def test_unexpected_structure(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("some_other_dir/file.txt", "data")

        with pytest.raises(DataAccessError, match="Unexpected zip structure"):
            extract_and_organize(zip_path, tmp_path / "out")

    def test_no_fasta_in_accession_dir(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "ncbi_dataset/data/GCF_000005845.2/report.json",
                '{"report": true}',
            )

        output_dir = tmp_path / "genomes"
        result = extract_and_organize(zip_path, output_dir)

        assert len(result) == 0

    def test_cleanup_on_success(self, tmp_path):
        """Verify temp extraction dir is cleaned up after success."""
        zip_path = tmp_path / "test.zip"
        _create_mock_ncbi_zip(zip_path, {"GCF_000005845.2": ">seq\nATCG\n"})

        extract_and_organize(zip_path, tmp_path / "genomes")

        extract_dir = tmp_path / "ncbi_extract_tmp"
        assert not extract_dir.exists()


# --- read_accession_file ---


class TestReadAccessionFile:
    def test_basic_file(self, tmp_path):
        f = tmp_path / "acc.txt"
        f.write_text("GCF_000005845.2\nGCA_000001405.1\n")

        result = read_accession_file(f)
        assert result == ["GCF_000005845.2", "GCA_000001405.1"]

    def test_skips_empty_lines(self, tmp_path):
        f = tmp_path / "acc.txt"
        f.write_text("GCF_000005845.2\n\n\nGCA_000001405.1\n\n")

        result = read_accession_file(f)
        assert result == ["GCF_000005845.2", "GCA_000001405.1"]

    def test_skips_comments(self, tmp_path):
        f = tmp_path / "acc.txt"
        f.write_text("# This is a comment\nGCF_000005845.2\n# Another comment\n")

        result = read_accession_file(f)
        assert result == ["GCF_000005845.2"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "acc.txt"
        f.write_text("  GCF_000005845.2  \n  GCA_000001405.1\t\n")

        result = read_accession_file(f)
        assert result == ["GCF_000005845.2", "GCA_000001405.1"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(DataAccessError, match="Accession file not found"):
            read_accession_file(tmp_path / "missing.txt")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")

        result = read_accession_file(f)
        assert result == []


# --- create_genome_manifest ---


class TestCreateGenomeManifest:
    def test_basic_manifest(self, tmp_path):
        genome_paths = {
            "GCF_000005845.2": tmp_path / "GCF_000005845.2.fna",
            "GCA_000001405.1": tmp_path / "GCA_000001405.1.fna",
        }
        manifest = tmp_path / "manifest.csv"

        result = create_genome_manifest(genome_paths, manifest)

        assert result == manifest
        assert manifest.exists()

        with open(manifest) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["name", "genome_filename", "protein_filename"]
        assert len(rows) == 3
        # Sorted by accession
        assert rows[1][0] == "GCA_000001405.1"
        assert rows[2][0] == "GCF_000005845.2"
        # Protein column should be empty
        assert rows[1][2] == ""
        assert rows[2][2] == ""

    def test_empty_genome_paths(self, tmp_path):
        with pytest.raises(DataAccessError, match="No genome paths provided"):
            create_genome_manifest({}, tmp_path / "manifest.csv")

    def test_creates_parent_directory(self, tmp_path):
        genome_paths = {
            "GCF_000005845.2": tmp_path / "GCF_000005845.2.fna",
        }
        manifest = tmp_path / "subdir" / "manifest.csv"

        result = create_genome_manifest(genome_paths, manifest)

        assert result.exists()

    def test_single_entry(self, tmp_path):
        genome_paths = {
            "GCF_000005845.2": Path("/data/GCF_000005845.2.fna"),
        }
        manifest = tmp_path / "manifest.csv"

        create_genome_manifest(genome_paths, manifest)

        with open(manifest) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[1] == [
            "GCF_000005845.2",
            "/data/GCF_000005845.2.fna",
            "",
        ]
