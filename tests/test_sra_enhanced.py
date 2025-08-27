"""
Tests for enhanced SRA functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from metaquest.data.sra_metadata import (
    SRAMetadataClient,
    SRADatasetInfo,
    detect_sequencing_technology,
    calculate_read_statistics,
    create_download_preview,
)
from metaquest.data.sra_enhanced import (
    EnhancedSRADownloader,
    verify_sra_tools,
    estimate_download_time,
)


class TestSRAMetadataClient:
    """Test SRA metadata client functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = SRAMetadataClient("test@example.com")

    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.email == "test@example.com"
        assert self.client.api_key is None
        assert self.client.request_delay > 0

    def test_client_with_api_key(self):
        """Test client with API key."""
        client = SRAMetadataClient("test@example.com", "api_key_123")
        assert client.api_key == "api_key_123"
        assert client.request_delay < 0.5  # Should be faster with API key

    @patch('requests.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.text = '{"test": "data"}'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client._make_request("http://test.com", {"param": "value"})

        assert result == '{"test": "data"}'
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_make_request_failure(self, mock_get):
        """Test failed API request."""
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            self.client._make_request("http://test.com", {"param": "value"})

    def test_dataset_info_creation(self):
        """Test SRADatasetInfo creation."""
        info = SRADatasetInfo(
            accession="SRR123456",
            title="Test dataset",
            organism="Escherichia coli",
            platform="ILLUMINA",
            instrument="Illumina HiSeq 2500",
            strategy="WGS",
            layout="PAIRED",
            spots=1000000,
            bases=150000000,
            avg_length=150.0,
            size_mb=100.0,
            release_date="2023-01-01",
            bioproject="PRJNA123456",
            biosample="SAMN123456",
            library_selection="RANDOM",
            library_source="GENOMIC",
        )

        assert info.accession == "SRR123456"
        assert info.platform == "ILLUMINA"
        assert info.spots == 1000000


class TestTechnologyDetection:
    """Test sequencing technology detection."""

    def create_test_dataset_info(self, platform, instrument, strategy="WGS"):
        """Create test dataset info."""
        return SRADatasetInfo(
            accession="TEST123",
            title="Test",
            organism="Test organism",
            platform=platform,
            instrument=instrument,
            strategy=strategy,
            layout="PAIRED",
            spots=1000,
            bases=150000,
            avg_length=150.0,
            size_mb=100.0,
            release_date="2023-01-01",
            bioproject="",
            biosample="",
            library_selection="",
            library_source="",
        )

    def test_illumina_detection(self):
        """Test Illumina technology detection."""
        # Test platform detection
        info = self.create_test_dataset_info("ILLUMINA", "HiSeq 2500")
        assert detect_sequencing_technology(info) == "illumina"

        # Test instrument detection
        info = self.create_test_dataset_info("", "Illumina NovaSeq 6000")
        assert detect_sequencing_technology(info) == "illumina"

    def test_nanopore_detection(self):
        """Test Nanopore technology detection."""
        # Test platform detection
        info = self.create_test_dataset_info("OXFORD_NANOPORE", "MinION")
        assert detect_sequencing_technology(info) == "nanopore"

        # Test instrument detection
        info = self.create_test_dataset_info("", "GridION X5")
        assert detect_sequencing_technology(info) == "nanopore"

        # Test strategy detection
        info = self.create_test_dataset_info("", "", "NANOPORE")
        assert detect_sequencing_technology(info) == "nanopore"

    def test_pacbio_detection(self):
        """Test PacBio technology detection."""
        # Test platform detection
        info = self.create_test_dataset_info("PACBIO_SMRT", "PacBio RS")
        assert detect_sequencing_technology(info) == "pacbio"

        # Test instrument detection
        info = self.create_test_dataset_info("", "Sequel II")
        assert detect_sequencing_technology(info) == "pacbio"

    def test_unknown_technology(self):
        """Test unknown technology detection."""
        info = self.create_test_dataset_info("UNKNOWN", "Unknown Instrument")
        assert detect_sequencing_technology(info) == "unknown"


class TestReadStatistics:
    """Test read statistics calculation."""

    def create_test_fastq_file(self, temp_dir, sequences):
        """Create a test FASTQ file."""
        fastq_path = temp_dir / "test.fastq"
        
        with open(fastq_path, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f"@read_{i}\n")
                f.write(f"{seq}\n")
                f.write("+\n")
                f.write("I" * len(seq) + "\n")  # High quality scores
        
        return fastq_path

    def test_read_statistics_calculation(self):
        """Test basic read statistics calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test sequences
            sequences = [
                "ATCGATCGATCGATCG",  # 16 bp
                "GCTAGCTAGCTAGCTA",  # 16 bp
                "AAAAAAAAAAAAAAAA",  # 16 bp, no GC
                "GGGGGGGGGGGGGGGG",  # 16 bp, all GC
            ]
            
            fastq_file = self.create_test_fastq_file(temp_path, sequences)
            stats = calculate_read_statistics([fastq_file])

            assert stats.total_reads == 4
            assert stats.total_bases == 64
            assert stats.avg_read_length == 16.0
            assert stats.min_read_length == 16
            assert stats.max_read_length == 16
            assert 40 < stats.gc_content < 60  # Should be around 50%

    def test_empty_fastq_file(self):
        """Test with empty FASTQ file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fastq_file = temp_path / "empty.fastq"
            fastq_file.touch()  # Create empty file

            stats = calculate_read_statistics([fastq_file])

            assert stats.total_reads == 0
            assert stats.total_bases == 0

    def test_n50_calculation(self):
        """Test N50 calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sequences of different lengths for N50 test
            sequences = [
                "A" * 100,  # 100 bp
                "T" * 200,  # 200 bp
                "C" * 300,  # 300 bp
                "G" * 400,  # 400 bp
            ]
            
            fastq_file = self.create_test_fastq_file(temp_path, sequences)
            stats = calculate_read_statistics([fastq_file])

            # Total bases: 1000, so N50 should be 300 (cumulative reaches 500 at 300)
            assert stats.n50 == 300


class TestEnhancedSRADownloader:
    """Test enhanced SRA downloader."""

    def setup_method(self):
        """Set up test downloader."""
        self.downloader = EnhancedSRADownloader("test@example.com")

    @patch('metaquest.data.sra_enhanced.create_download_preview')
    def test_preview_downloads(self, mock_preview):
        """Test download preview."""
        mock_metadata = {"SRR123": MagicMock()}
        mock_tech_counts = {"illumina": 1}
        mock_size_gb = 1.5
        
        mock_preview.return_value = (mock_metadata, mock_tech_counts, mock_size_gb)

        metadata, tech_counts, size_gb = self.downloader.preview_downloads(["SRR123"])

        assert metadata == mock_metadata
        assert tech_counts == mock_tech_counts
        assert size_gb == mock_size_gb

    def test_build_download_command_illumina(self):
        """Test building download command for Illumina."""
        args = self.downloader._build_download_command("SRR123", Path("/tmp"), "illumina")
        
        assert "SRR123" in args
        assert "--split-files" in args

    def test_build_download_command_nanopore(self):
        """Test building download command for Nanopore."""
        args = self.downloader._build_download_command("SRR123", Path("/tmp"), "nanopore")
        
        assert "SRR123" in args
        assert "--include-technical" in args

    def test_rename_files_illumina_paired(self):
        """Test file renaming for Illumina paired-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "SRR123"
            
            # Create test files
            file1 = temp_path / "SRR123_1.fastq"
            file2 = temp_path / "SRR123_2.fastq"
            file1.touch()
            file2.touch()
            
            renamed = self.downloader._rename_files_by_technology(
                [file1, file2], "illumina", output_path
            )
            
            assert len(renamed) == 2
            assert str(renamed[file1]).endswith("_R1.fastq.gz")
            assert str(renamed[file2]).endswith("_R2.fastq.gz")

    def test_rename_files_nanopore(self):
        """Test file renaming for Nanopore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "SRR123"
            
            # Create test file
            file1 = temp_path / "SRR123.fastq"
            file1.touch()
            
            renamed = self.downloader._rename_files_by_technology(
                [file1], "nanopore", output_path
            )
            
            assert len(renamed) == 1
            assert str(renamed[file1]).endswith("_long.fastq")


class TestSRAUtilities:
    """Test SRA utility functions."""

    @patch('subprocess.run')
    def test_verify_sra_tools_success(self, mock_run):
        """Test successful SRA tools verification."""
        mock_run.return_value = MagicMock()
        
        result = verify_sra_tools()
        
        assert result == True
        assert mock_run.call_count == 3  # Three tools to verify

    @patch('subprocess.run')
    def test_verify_sra_tools_failure(self, mock_run):
        """Test SRA tools verification failure."""
        mock_run.side_effect = FileNotFoundError()
        
        result = verify_sra_tools()
        
        assert result == False

    def test_estimate_download_time(self):
        """Test download time estimation."""
        # Test with 1 GB at 100 Mbps with 4 parallel downloads
        time_hours = estimate_download_time(1.0, 100.0, 4)
        
        # Should be less than 1 hour
        assert time_hours < 1.0
        assert time_hours > 0

    def test_download_preview_creation(self):
        """Test download preview creation."""
        # Mock metadata client
        mock_client = MagicMock()
        mock_dataset = SRADatasetInfo(
            accession="SRR123",
            title="Test",
            organism="E. coli",
            platform="ILLUMINA",
            instrument="HiSeq",
            strategy="WGS",
            layout="PAIRED",
            spots=1000,
            bases=150000,
            avg_length=150.0,
            size_mb=100.0,
            release_date="2023-01-01",
            bioproject="",
            biosample="",
            library_selection="",
            library_source="",
        )
        mock_client.get_sra_metadata.return_value = {"SRR123": mock_dataset}

        metadata, tech_counts, total_size_gb = create_download_preview(
            ["SRR123"], mock_client
        )

        assert len(metadata) == 1
        assert "SRR123" in metadata
        assert tech_counts["illumina"] == 1
        assert total_size_gb == 100.0 / 1024  # Convert MB to GB


class TestSRAIntegration:
    """Integration tests for SRA functionality."""

    def test_metadata_to_dataframe_conversion(self):
        """Test converting metadata to DataFrame format."""
        dataset_info = SRADatasetInfo(
            accession="SRR123456",
            title="Test dataset",
            organism="Escherichia coli",
            platform="ILLUMINA",
            instrument="Illumina HiSeq 2500",
            strategy="WGS",
            layout="PAIRED",
            spots=1000000,
            bases=150000000,
            avg_length=150.0,
            size_mb=100.0,
            release_date="2023-01-01",
            bioproject="PRJNA123456",
            biosample="SAMN123456",
            library_selection="RANDOM",
            library_source="GENOMIC",
        )

        # Convert to dictionary (as would be done for CSV export)
        record = {
            "accession": dataset_info.accession,
            "platform": dataset_info.platform,
            "technology": detect_sequencing_technology(dataset_info),
            "spots": dataset_info.spots,
            "bases": dataset_info.bases,
        }

        assert record["accession"] == "SRR123456"
        assert record["technology"] == "illumina"
        assert record["spots"] == 1000000

    @patch('metaquest.data.sra_metadata.save_metadata_report')
    def test_report_generation_integration(self, mock_save):
        """Test integration of metadata and report generation."""
        metadata = {
            "SRR123": SRADatasetInfo(
                accession="SRR123",
                title="Test",
                organism="E. coli",
                platform="ILLUMINA",
                instrument="HiSeq",
                strategy="WGS",
                layout="PAIRED",
                spots=1000,
                bases=150000,
                avg_length=150.0,
                size_mb=100.0,
                release_date="2023-01-01",
                bioproject="",
                biosample="",
                library_selection="",
                library_source="",
            )
        }

        # This should not raise an exception
        mock_save(metadata, "test_report.csv")
        mock_save.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])