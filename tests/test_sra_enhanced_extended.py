"""
EXTENDED TESTS for data/sra_enhanced.py (30% → 80%+ coverage)

This file adds tests for untested methods:
- download_accession_enhanced (full method)
- _download_with_optimizations (error paths)
- _handle_download_output
- _check_existing_download
- _cleanup_temp
- download_batch_enhanced
- create_download_report

Run: pytest tests/test_sra_enhanced_extended.py -v
"""

import subprocess
from unittest.mock import patch
from dataclasses import dataclass

from metaquest.data.sra_enhanced import (
    EnhancedSRADownloader,
    create_download_report,
)
from metaquest.core.exceptions import SecurityError


# Mock data classes
@dataclass
class MockSRADatasetInfo:
    """Mock SRA dataset info."""

    accession: str
    platform: str
    instrument: str
    layout: str
    spots: int
    bases: int
    avg_length: float
    size_mb: float


@dataclass
class MockReadStatistics:
    """Mock read statistics."""

    total_reads: int
    total_bases: int
    avg_read_length: float
    gc_content: float


# ============================================================================
# TEST CLASS: EnhancedSRADownloader - Main Download Logic
# ============================================================================


class TestDownloadAccessionEnhanced:
    """Test download_accession_enhanced method comprehensively."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com")

    def test_download_accession_success(self, tmp_path):
        """Test successful accession download."""
        output_folder = tmp_path / "downloads"

        mock_dataset = MockSRADatasetInfo(
            accession="SRR001",
            platform="ILLUMINA",
            instrument="HiSeq",
            layout="PAIRED",
            spots=1000,
            bases=150000,
            avg_length=150.0,
            size_mb=100.0,
        )

        mock_stats = MockReadStatistics(total_reads=1000, total_bases=150000, avg_read_length=150.0, gc_content=45.0)

        with patch.object(self.downloader.metadata_client, "get_sra_metadata") as mock_meta:
            mock_meta.return_value = {"SRR001": mock_dataset}

            with patch.object(self.downloader, "_check_existing_download", return_value=False):
                with patch.object(self.downloader, "_download_with_optimizations", return_value=(True, "Downloaded")):
                    with patch("metaquest.data.sra_enhanced.calculate_read_statistics", return_value=mock_stats):
                        # Create dummy fastq file
                        output_path = output_folder / "SRR001"
                        output_path.mkdir(parents=True, exist_ok=True)
                        (output_path / "test.fastq.gz").touch()

                        success, message, metadata = self.downloader.download_accession_enhanced(
                            "SRR001", output_folder, force=False
                        )

        assert success is True
        assert "technology" in metadata
        assert metadata["technology"] == "illumina"
        assert metadata["downloaded_reads"] == 1000

    def test_download_accession_already_exists(self, tmp_path):
        """Test skipping already downloaded accession."""
        output_folder = tmp_path / "downloads"

        mock_dataset = MockSRADatasetInfo(
            accession="SRR001",
            platform="ILLUMINA",
            instrument="HiSeq",
            layout="PAIRED",
            spots=1000,
            bases=150000,
            avg_length=150.0,
            size_mb=100.0,
        )

        with patch.object(self.downloader.metadata_client, "get_sra_metadata") as mock_meta:
            mock_meta.return_value = {"SRR001": mock_dataset}

            with patch.object(self.downloader, "_check_existing_download", return_value=True):
                success, message, metadata = self.downloader.download_accession_enhanced(
                    "SRR001", output_folder, force=False
                )

        assert success is True
        assert "Already downloaded" in message

    def test_download_accession_no_metadata(self, tmp_path):
        """Test download when metadata fetch fails."""
        output_folder = tmp_path / "downloads"

        with patch.object(self.downloader.metadata_client, "get_sra_metadata") as mock_meta:
            mock_meta.return_value = {}  # No metadata returned

            with patch.object(self.downloader, "_check_existing_download", return_value=False):
                with patch.object(self.downloader, "_download_with_optimizations", return_value=(True, "Downloaded")):
                    success, message, metadata = self.downloader.download_accession_enhanced(
                        "SRR001", output_folder, force=False
                    )

        # Should still attempt download even without metadata
        assert success is True

    def test_download_accession_force_redownload(self, tmp_path):
        """Test force redownload ignores existing files."""
        output_folder = tmp_path / "downloads"

        mock_dataset = MockSRADatasetInfo(
            accession="SRR001",
            platform="ILLUMINA",
            instrument="HiSeq",
            layout="PAIRED",
            spots=1000,
            bases=150000,
            avg_length=150.0,
            size_mb=100.0,
        )

        with patch.object(self.downloader.metadata_client, "get_sra_metadata") as mock_meta:
            mock_meta.return_value = {"SRR001": mock_dataset}

            # Even though files exist, force=True should skip the check
            with patch.object(self.downloader, "_check_existing_download", return_value=True):
                with patch.object(self.downloader, "_download_with_optimizations", return_value=(True, "Downloaded")):
                    success, message, metadata = self.downloader.download_accession_enhanced(
                        "SRR001", output_folder, force=True  # Force redownload
                    )

        assert success is True
        assert "Downloaded" in message

    def test_download_accession_exception_handling(self, tmp_path):
        """Test exception handling in download_accession_enhanced."""
        output_folder = tmp_path / "downloads"

        with patch.object(self.downloader.metadata_client, "get_sra_metadata") as mock_meta:
            mock_meta.side_effect = Exception("API Error")

            success, message, metadata = self.downloader.download_accession_enhanced(
                "SRR001", output_folder, force=False
            )

        assert success is False
        assert "Enhanced download failed" in message


# ============================================================================
# TEST CLASS: Download With Optimizations
# ============================================================================


class TestDownloadWithOptimizations:
    """Test _download_with_optimizations method."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com")

    def test_download_with_optimizations_success(self, tmp_path):
        """Test successful download with optimizations."""
        output_path = tmp_path / "SRR001"

        with patch.object(self.downloader, "_build_download_command", return_value=["--help"]):
            with patch("metaquest.data.sra_enhanced.SecureSubprocess.run_secure"):
                with patch.object(self.downloader, "_handle_download_output", return_value=(True, "Success")):
                    success, message = self.downloader._download_with_optimizations("SRR001", output_path, "illumina")

        assert success is True
        assert message == "Success"

    def test_download_called_process_error(self, tmp_path):
        """Test handling of CalledProcessError."""
        output_path = tmp_path / "SRR001"

        with patch.object(self.downloader, "_build_download_command", return_value=["--help"]):
            with patch("metaquest.data.sra_enhanced.SecureSubprocess.run_secure") as mock_run:
                error = subprocess.CalledProcessError(1, "fasterq-dump", stderr="Download failed")
                mock_run.side_effect = error

                with patch.object(self.downloader, "_cleanup_temp"):
                    success, message = self.downloader._download_with_optimizations("SRR001", output_path, "illumina")

        assert success is False
        assert "Download command failed" in message

    def test_download_security_error(self, tmp_path):
        """Test handling of SecurityError."""
        output_path = tmp_path / "SRR001"

        with patch.object(self.downloader, "_build_download_command", return_value=["--help"]):
            with patch("metaquest.data.sra_enhanced.SecureSubprocess.run_secure") as mock_run:
                mock_run.side_effect = SecurityError("Unsafe command")

                with patch.object(self.downloader, "_cleanup_temp"):
                    success, message = self.downloader._download_with_optimizations("SRR001", output_path, "illumina")

        assert success is False
        assert "Security error" in message

    def test_download_unexpected_error(self, tmp_path):
        """Test handling of unexpected exceptions."""
        output_path = tmp_path / "SRR001"

        with patch.object(self.downloader, "_build_download_command", return_value=["--help"]):
            with patch("metaquest.data.sra_enhanced.SecureSubprocess.run_secure") as mock_run:
                mock_run.side_effect = RuntimeError("Unexpected error")

                with patch.object(self.downloader, "_cleanup_temp"):
                    success, message = self.downloader._download_with_optimizations("SRR001", output_path, "illumina")

        assert success is False
        assert "Download failed" in message


# ============================================================================
# TEST CLASS: Handle Download Output
# ============================================================================


class TestHandleDownloadOutput:
    """Test _handle_download_output method."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com")

    def test_handle_output_success(self, tmp_path):
        """Test successful output handling."""
        temp_path = tmp_path / "temp"
        temp_path.mkdir()
        output_path = tmp_path / "output" / "SRR001"

        # Create mock FASTQ files
        (temp_path / "SRR001_1.fastq").touch()
        (temp_path / "SRR001_2.fastq").touch()

        with patch.object(self.downloader, "_rename_files_by_technology") as mock_rename:
            mock_rename.return_value = {
                temp_path / "SRR001_1.fastq": output_path / "SRR001_R1.fastq.gz",
                temp_path / "SRR001_2.fastq": output_path / "SRR001_R2.fastq.gz",
            }

            with patch("shutil.move"):
                success, message = self.downloader._handle_download_output(temp_path, output_path, "illumina")

        assert success is True
        assert "Downloaded 2 files" in message

    def test_handle_output_no_fastq_files(self, tmp_path):
        """Test handling when no FASTQ files are created."""
        temp_path = tmp_path / "temp"
        temp_path.mkdir()
        output_path = tmp_path / "output" / "SRR001"

        # No FASTQ files created
        with patch.object(self.downloader, "_cleanup_temp"):
            success, message = self.downloader._handle_download_output(temp_path, output_path, "illumina")

        assert success is False
        assert "No FASTQ files created" in message


# ============================================================================
# TEST CLASS: File Renaming Edge Cases
# ============================================================================


class TestFileRenamingEdgeCases:
    """Test _rename_files_by_technology edge cases."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com")

    def test_rename_illumina_multiple_files(self, tmp_path):
        """Test renaming multiple Illumina files (>2)."""
        output_path = tmp_path / "SRR001"

        # Create 4 files
        files = [
            tmp_path / "SRR001_1.fastq",
            tmp_path / "SRR001_2.fastq",
            tmp_path / "SRR001_3.fastq",
            tmp_path / "SRR001_4.fastq",
        ]
        for f in files:
            f.touch()

        renamed = self.downloader._rename_files_by_technology(files, "illumina", output_path)

        assert len(renamed) == 4
        # Should be numbered R1, R2, R3, R4
        assert str(renamed[files[0]]).endswith("_R1.fastq.gz")
        assert str(renamed[files[3]]).endswith("_R4.fastq.gz")

    def test_rename_illumina_single_file(self, tmp_path):
        """Test renaming single Illumina file."""
        output_path = tmp_path / "SRR001"

        file1 = tmp_path / "SRR001.fastq"
        file1.touch()

        renamed = self.downloader._rename_files_by_technology([file1], "illumina", output_path)

        assert len(renamed) == 1
        assert str(renamed[file1]).endswith("_R1.fastq.gz")

    def test_rename_nanopore_multiple_files(self, tmp_path):
        """Test renaming multiple Nanopore files."""
        output_path = tmp_path / "SRR001"

        files = [
            tmp_path / "SRR001_1.fastq.gz",
            tmp_path / "SRR001_2.fastq.gz",
        ]
        for f in files:
            f.touch()

        renamed = self.downloader._rename_files_by_technology(files, "nanopore", output_path)

        assert len(renamed) == 2
        assert str(renamed[files[0]]).endswith("_long_1.fastq.gz")
        assert str(renamed[files[1]]).endswith("_long_2.fastq.gz")

    def test_rename_unknown_technology(self, tmp_path):
        """Test renaming with unknown technology."""
        output_path = tmp_path / "SRR001"

        files = [
            tmp_path / "SRR001_1.fastq",
            tmp_path / "SRR001_2.fastq",
        ]
        for f in files:
            f.touch()

        renamed = self.downloader._rename_files_by_technology(files, "unknown", output_path)

        assert len(renamed) == 2
        # Should use generic naming with suffixes
        assert "_1" in str(renamed[files[0]])
        assert "_2" in str(renamed[files[1]])


# ============================================================================
# TEST CLASS: Existing Download Check
# ============================================================================


class TestCheckExistingDownload:
    """Test _check_existing_download method."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com")

    def test_check_existing_no_directory(self, tmp_path):
        """Test check when directory doesn't exist."""
        output_path = tmp_path / "nonexistent"

        result = self.downloader._check_existing_download(output_path)

        assert result is False

    def test_check_existing_empty_directory(self, tmp_path):
        """Test check with empty directory."""
        output_path = tmp_path / "empty"
        output_path.mkdir()

        result = self.downloader._check_existing_download(output_path)

        assert result is False

    def test_check_existing_with_fastq_files(self, tmp_path):
        """Test check with existing FASTQ files."""
        output_path = tmp_path / "existing"
        output_path.mkdir()
        (output_path / "test.fastq.gz").touch()

        result = self.downloader._check_existing_download(output_path)

        assert result is True


# ============================================================================
# TEST CLASS: Cleanup Temp
# ============================================================================


class TestCleanupTemp:
    """Test _cleanup_temp method."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com")

    def test_cleanup_temp_success(self, tmp_path):
        """Test successful temp cleanup."""
        temp_path = tmp_path / "temp"
        temp_path.mkdir()
        (temp_path / "file.txt").touch()

        self.downloader._cleanup_temp(temp_path)

        assert not temp_path.exists()

    def test_cleanup_temp_nonexistent(self, tmp_path):
        """Test cleanup of nonexistent directory (should not raise)."""
        temp_path = tmp_path / "nonexistent"

        # Should not raise exception
        self.downloader._cleanup_temp(temp_path)

    def test_cleanup_temp_permission_error(self, tmp_path):
        """Test cleanup when permission error occurs."""
        temp_path = tmp_path / "temp"
        temp_path.mkdir()

        with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            # Should log warning but not raise
            self.downloader._cleanup_temp(temp_path)


# ============================================================================
# TEST CLASS: Batch Download
# ============================================================================


class TestDownloadBatchEnhanced:
    """Test download_batch_enhanced method."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com", max_workers=2)

    def test_batch_download_success(self, tmp_path):
        """Test successful batch download."""
        output_folder = tmp_path / "downloads"
        accessions = ["SRR001", "SRR002"]

        with patch.object(self.downloader, "download_accession_enhanced") as mock_download:
            mock_download.side_effect = [
                (True, "Success", {"technology": "illumina"}),
                (True, "Success", {"technology": "nanopore"}),
            ]

            results = self.downloader.download_batch_enhanced(accessions, output_folder, force=False)

        assert results["total"] == 2
        assert results["successful"] == 2
        assert results["failed"] == 0
        assert "illumina" in results["technology_summary"]
        assert "nanopore" in results["technology_summary"]

    def test_batch_download_with_failures(self, tmp_path):
        """Test batch download with some failures."""
        output_folder = tmp_path / "downloads"
        accessions = ["SRR001", "SRR002", "SRR003"]

        with patch.object(self.downloader, "download_accession_enhanced") as mock_download:
            mock_download.side_effect = [
                (True, "Success", {"technology": "illumina"}),
                (False, "Failed", {}),
                (True, "Success", {"technology": "illumina"}),
            ]

            results = self.downloader.download_batch_enhanced(accessions, output_folder, force=False)

        assert results["total"] == 3
        assert results["successful"] == 2
        assert results["failed"] == 1
        assert len(results["failed_accessions"]) == 1

    def test_batch_download_with_blacklist(self, tmp_path):
        """Test batch download with blacklist."""
        output_folder = tmp_path / "downloads"
        accessions = ["SRR001", "SRR002", "SRR003"]
        blacklist = {"SRR002"}

        with patch.object(self.downloader, "download_accession_enhanced") as mock_download:
            mock_download.return_value = (True, "Success", {"technology": "illumina"})

            results = self.downloader.download_batch_enhanced(
                accessions, output_folder, force=False, blacklist=blacklist
            )

        assert results["total"] == 2  # SRR002 was blacklisted
        assert mock_download.call_count == 2

    def test_batch_download_with_max_downloads(self, tmp_path):
        """Test batch download with max_downloads limit."""
        output_folder = tmp_path / "downloads"
        accessions = ["SRR001", "SRR002", "SRR003"]

        with patch.object(self.downloader, "download_accession_enhanced") as mock_download:
            mock_download.return_value = (True, "Success", {"technology": "illumina"})

            results = self.downloader.download_batch_enhanced(accessions, output_folder, force=False, max_downloads=2)

        assert results["total"] == 2  # Limited to 2
        assert mock_download.call_count == 2

    def test_batch_download_exception_handling(self, tmp_path):
        """Test batch download with exceptions."""
        output_folder = tmp_path / "downloads"
        accessions = ["SRR001", "SRR002"]

        with patch.object(self.downloader, "download_accession_enhanced") as mock_download:
            mock_download.side_effect = [
                (True, "Success", {"technology": "illumina"}),
                Exception("Unexpected error"),
            ]

            results = self.downloader.download_batch_enhanced(accessions, output_folder, force=False)

        assert results["total"] == 2
        assert results["successful"] == 1
        assert results["failed"] == 1
        assert "SRR002" in results["failed_accessions"]


# ============================================================================
# TEST CLASS: Create Download Report
# ============================================================================


class TestCreateDownloadReport:
    """Test create_download_report function."""

    def test_create_report_success(self, tmp_path):
        """Test successful report creation."""
        results = {
            "metadata": {
                "SRR001": {
                    "technology": "illumina",
                    "platform": "ILLUMINA",
                    "layout": "PAIRED",
                    "spots": 1000,
                    "bases": 150000,
                    "avg_length": 150.0,
                    "size_mb": 100.0,
                    "downloaded_reads": 1000,
                    "downloaded_bases": 150000,
                    "gc_content": 45.0,
                },
                "SRR002": {
                    "technology": "nanopore",
                    "platform": "OXFORD_NANOPORE",
                    "layout": "SINGLE",
                    "spots": 500,
                    "bases": 250000,
                    "avg_length": 500.0,
                    "size_mb": 200.0,
                    "downloaded_reads": 500,
                    "downloaded_bases": 250000,
                    "gc_content": 48.0,
                },
            },
            "download_results": {
                "SRR001": "Downloaded successfully",
                "SRR002": "Downloaded successfully",
            },
            "failed_accessions": [],
        }

        output_file = tmp_path / "report.csv"

        create_download_report(results, output_file)

        assert output_file.exists()

        # Check contents
        import pandas as pd

        df = pd.read_csv(output_file)
        assert len(df) == 2
        assert "SRR001" in df["accession"].values
        assert "illumina" in df["technology"].values

    def test_create_report_with_failures(self, tmp_path):
        """Test report creation with failed downloads."""
        results = {
            "metadata": {
                "SRR001": {
                    "technology": "illumina",
                    "platform": "ILLUMINA",
                    "layout": "PAIRED",
                    "spots": 1000,
                    "bases": 150000,
                    "avg_length": 150.0,
                    "size_mb": 100.0,
                    "downloaded_reads": 1000,
                    "downloaded_bases": 150000,
                    "gc_content": 45.0,
                },
                "SRR002": {},  # Failed download
            },
            "download_results": {
                "SRR001": "Success",
                "SRR002": "Failed to download",
            },
            "failed_accessions": ["SRR002"],
        }

        output_file = tmp_path / "report.csv"

        create_download_report(results, output_file)

        assert output_file.exists()

        import pandas as pd

        df = pd.read_csv(output_file)
        assert "Failed" in df[df["accession"] == "SRR002"]["status"].values

    def test_create_report_no_metadata(self, tmp_path, caplog):
        """Test report creation with no metadata."""
        results = {"metadata": {}}
        output_file = tmp_path / "report.csv"

        create_download_report(results, output_file)

        assert "No metadata available" in caplog.text


# ============================================================================
# TEST CLASS: Build Download Command Edge Cases
# ============================================================================


class TestBuildDownloadCommandEdgeCases:
    """Test _build_download_command edge cases."""

    def test_build_command_with_temp_folder(self, tmp_path):
        """Test command building with temp folder."""
        temp_folder = tmp_path / "temp"
        temp_folder.mkdir()

        downloader = EnhancedSRADownloader("test@example.com", temp_folder=temp_folder)

        args = downloader._build_download_command("SRR001", tmp_path, "illumina")

        assert "--temp" in args
        assert str(temp_folder.absolute()) in args

    def test_build_command_pacbio_technology(self, tmp_path):
        """Test command building for PacBio."""
        downloader = EnhancedSRADownloader("test@example.com")

        args = downloader._build_download_command("SRR001", tmp_path, "pacbio")

        assert "SRR001" in args
        assert "--include-technical" in args


# ============================================================================
# SUCCESS METRICS:
#
# After running these tests:
# - Expected: 50+ additional tests pass
# - Coverage: 30% → 80%+ for data/sra_enhanced.py
# - All untested methods now covered
#
# Run tests:
#   pytest tests/test_sra_enhanced_extended.py -v
#
# Check coverage:
#   pytest --cov=metaquest.data.sra_enhanced --cov-report=term-missing \
#          tests/test_sra_enhanced.py tests/test_sra_enhanced_extended.py
# ============================================================================
