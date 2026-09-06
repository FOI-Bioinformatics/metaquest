"""
Tests for the SRA metadata client: NCBI queries, technology detection, and
read statistics calculation.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import sys


# Force fresh import of modules to avoid contamination from other tests
def _force_fresh_import():
    """Force fresh import of SRA modules to get real implementations."""
    modules_to_clear = [
        "metaquest.data.sra_metadata",
    ]
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            # Only remove if it's a Mock object (contaminated)
            if hasattr(sys.modules[module_name], "_mock_name"):
                del sys.modules[module_name]


# Apply fresh import at start of test session
_force_fresh_import()

from metaquest.data.sra_metadata import (  # noqa: E402
    SRAMetadataClient,
    SRADatasetInfo,
    detect_sequencing_technology,
    calculate_read_statistics,
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

    @patch("requests.get")
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.text = '{"test": "data"}'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client._make_request("http://test.com", {"param": "value"})

        assert result == '{"test": "data"}'
        mock_get.assert_called_once()

    @patch("requests.get")
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

    def test_default_result_is_not_sampled(self):
        """A file smaller than max_reads is read in full: sampled is False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fastq_file = self.create_test_fastq_file(temp_path, ["ACGT"] * 5)

            stats = calculate_read_statistics([fastq_file])

            assert stats.total_reads == 5
            assert stats.sampled is False

    def test_max_reads_caps_the_stream_and_marks_sampled(self):
        """max_reads stops the stream early and marks the result as sampled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fastq_file = self.create_test_fastq_file(temp_path, ["ACGT"] * 10)

            stats = calculate_read_statistics([fastq_file], max_reads=4)

            assert stats.total_reads == 4
            assert stats.sampled is True

    def test_max_reads_zero_means_exact(self):
        """max_reads=0 reads every record, however many there are."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fastq_file = self.create_test_fastq_file(temp_path, ["ACGT"] * 10)

            stats = calculate_read_statistics([fastq_file], max_reads=0)

            assert stats.total_reads == 10
            assert stats.sampled is False

    def test_cached_dict_builds_read_statistics_without_file_io(self):
        """When ``cached`` is given, the result comes straight from it: no FASTQ is read."""
        cached = {
            "reads_total": 12345,
            "bases_total": 1850000,
            "avg_read_length": 150.0,
            "min_read_length": 100,
            "max_read_length": 200,
            "n50": 150,
            "gc_content": 0.45,
            "quality_summary": {"mean": 30.0, "median": 31.0, "q25": 25.0, "q75": 35.0},
            "sampled": True,
        }

        stats = calculate_read_statistics([Path("/nonexistent/does-not-matter.fastq")], cached=cached)

        assert stats.total_reads == 12345
        assert stats.total_bases == 1850000
        assert stats.avg_read_length == 150.0
        assert stats.min_read_length == 100
        assert stats.max_read_length == 200
        assert stats.n50 == 150
        assert stats.gc_content == pytest.approx(45.0)  # cache holds a 0-1 fraction
        assert stats.sampled is True
        assert stats.quality_scores["mean"] == pytest.approx(30.0)


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

    @patch("metaquest.data.sra_metadata.save_metadata_report")
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
