"""
Test processing.containment module functionality.

Tests for containment analysis including test genome download
and containment data analysis functions.
"""

import io
import tempfile
import gzip
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest
import pandas as pd

from metaquest.processing.containment import (
    download_test_genome,
    count_single_sample,
    filter_samples_by_containment,
    find_co_occurring_genomes,
)
from metaquest.core.exceptions import ProcessingError


class TestDownloadTestGenome:
    """Test download_test_genome function."""

    def test_download_test_genome_file_exists(self, tmp_path):
        """Test when test genome file already exists."""
        # Create existing file
        output_path = tmp_path / "GCF_000008985.1.fasta"
        output_path.write_text("existing genome content")

        result = download_test_genome(tmp_path)
        
        assert result == output_path
        assert output_path.exists()
        assert output_path.read_text() == "existing genome content"

    @patch("metaquest.processing.containment.urllib.request.urlopen")
    def test_download_test_genome_success(self, mock_urlopen, tmp_path):
        """Test successful genome download."""
        # Create temporary compressed file content
        genome_content = ">test_genome\nACGTACGTACGT\n"
        compressed = io.BytesIO()
        with gzip.open(compressed, 'wt') as f:
            f.write(genome_content)

        # Mock urlopen to return our compressed data
        mock_response = Mock()
        mock_response.read.return_value = compressed.getvalue()
        mock_urlopen.return_value = mock_response

        result = download_test_genome(tmp_path)

        expected_path = tmp_path / "GCF_000008985.1.fasta"
        assert result == expected_path
        assert expected_path.exists()
        assert expected_path.read_text() == genome_content

        # Verify temporary gz file was cleaned up
        temp_gz_path = expected_path.with_suffix(".gz")
        assert not temp_gz_path.exists()

    @patch("metaquest.processing.containment.urllib.request.urlopen")
    def test_download_test_genome_download_failure(self, mock_urlopen, tmp_path):
        """Test failure during download."""
        mock_urlopen.side_effect = Exception("Download failed")

        with pytest.raises(ProcessingError, match="Error downloading test genome"):
            download_test_genome(tmp_path)

    @patch("metaquest.processing.containment.urllib.request.urlopen")
    def test_download_test_genome_decompression_failure(self, mock_urlopen, tmp_path):
        """Test failure during decompression."""
        # Mock urlopen to return invalid gzip data
        mock_response = Mock()
        mock_response.read.return_value = b"not a valid gzip file"
        mock_urlopen.return_value = mock_response

        with pytest.raises(ProcessingError, match="Error downloading test genome"):
            download_test_genome(tmp_path)

    def test_download_test_genome_invalid_output_folder(self):
        """Test with invalid output folder."""
        # Test with path that cannot be created (e.g., permission denied scenario)
        with patch("metaquest.processing.containment.ensure_directory") as mock_ensure:
            mock_ensure.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(ProcessingError):
                download_test_genome("/invalid/path")

    @patch("metaquest.processing.containment.urllib.request.urlretrieve")
    def test_download_test_genome_creates_directory(self, mock_urlretrieve, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        # Use a subdirectory that doesn't exist yet
        output_dir = tmp_path / "subdir" / "genomes"
        
        # Create temporary compressed file content
        genome_content = ">test_genome\nACGTACGTACGT\n"
        temp_gz_file = tmp_path / "temp.gz"
        
        with gzip.open(temp_gz_file, 'wt') as f:
            f.write(genome_content)

        def mock_download(url, filename):
            temp_gz_path = Path(filename)
            temp_gz_path.parent.mkdir(parents=True, exist_ok=True)
            temp_gz_path.write_bytes(temp_gz_file.read_bytes())

        mock_urlretrieve.side_effect = mock_download

        result = download_test_genome(output_dir)
        
        expected_path = output_dir / "GCF_000008985.1.fasta"
        assert result == expected_path
        assert expected_path.exists()
        assert output_dir.exists()

    @patch("metaquest.processing.containment.urllib.request.urlretrieve")
    @patch("metaquest.processing.containment.logger")
    def test_download_test_genome_logging(self, mock_logger, mock_urlretrieve, tmp_path):
        """Test that appropriate logging messages are generated."""
        # Test logging when file exists
        output_path = tmp_path / "GCF_000008985.1.fasta"
        output_path.write_text("existing")

        download_test_genome(tmp_path)
        mock_logger.info.assert_called_with(f"Test genome already exists at {output_path}")

        # Reset mock and test logging during download
        mock_logger.reset_mock()
        output_path.unlink()  # Remove existing file

        # Create temporary compressed file
        genome_content = ">test_genome\nACGT\n"
        temp_gz_file = tmp_path / "temp.gz"
        
        with gzip.open(temp_gz_file, 'wt') as f:
            f.write(genome_content)

        def mock_download(url, filename):
            Path(filename).write_bytes(temp_gz_file.read_bytes())

        mock_urlretrieve.side_effect = mock_download

        download_test_genome(tmp_path)
        
        # Check that download-related logging occurred
        mock_logger.info.assert_any_call("Downloading test genome")
        mock_logger.info.assert_any_call(f"Downloaded test genome to {output_path}")

    def test_download_test_genome_path_handling(self, tmp_path):
        """Test that function handles both string and Path inputs."""
        # Test with string path
        output_path_str = str(tmp_path)
        
        with patch("metaquest.processing.containment.urllib.request.urlretrieve") as mock_urlretrieve:
            genome_content = ">test\nACGT\n"
            temp_gz_file = tmp_path / "temp.gz"
            
            with gzip.open(temp_gz_file, 'wt') as f:
                f.write(genome_content)

            def mock_download(url, filename):
                Path(filename).write_bytes(temp_gz_file.read_bytes())

            mock_urlretrieve.side_effect = mock_download

            result = download_test_genome(output_path_str)
            
            expected_path = tmp_path / "GCF_000008985.1.fasta"
            assert result == expected_path
            assert isinstance(result, Path)


class TestCountSingleSample:
    """Test count_single_sample function."""

    def setup_method(self):
        """Set up test data."""
        # Sample summary data
        self.summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.2, 0.6, 0.05, 0.9],
            'GCF_000002.1': [0.1, 0.7, 0.3, 0.15, 0.2],
            'max_containment': [0.8, 0.7, 0.6, 0.15, 0.9]
        }, index=['SRR001', 'SRR002', 'SRR003', 'SRR004', 'SRR005'])
        
        # Sample metadata
        self.metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'E. coli', 'B. subtilis', 'E. coli'],
            'country': ['USA', 'UK', 'Canada', 'Germany', 'USA'],
            'year': [2020, 2019, 2021, 2020, 2022]
        }, index=['SRR001', 'SRR002', 'SRR003', 'SRR004', 'SRR005'])

    def test_count_single_sample_success(self, tmp_path):
        """Test successful metadata counting."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        self.summary_data.to_csv(summary_file, sep='\t')
        self.metadata_data.to_csv(metadata_file, sep='\t')
        
        result = count_single_sample(
            summary_file, metadata_file, 'GCF_000001.1', 'organism',
            threshold=0.5, top_n=10
        )
        
        # Expected: SRR001 (0.8), SRR003 (0.6), SRR005 (0.9) pass threshold
        # All have 'E. coli' in metadata
        assert result == {'E. coli': 3}

    def test_count_single_sample_multiple_organisms(self, tmp_path):
        """Test counting with multiple different organisms."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        # Modify metadata to have different organisms for samples above threshold
        metadata_mixed = self.metadata_data.copy()
        metadata_mixed.loc['SRR001', 'organism'] = 'E. coli'
        metadata_mixed.loc['SRR003', 'organism'] = 'B. subtilis'
        metadata_mixed.loc['SRR005', 'organism'] = 'E. coli'
        
        self.summary_data.to_csv(summary_file, sep='\t')
        metadata_mixed.to_csv(metadata_file, sep='\t')
        
        result = count_single_sample(
            summary_file, metadata_file, 'GCF_000001.1', 'organism',
            threshold=0.5, top_n=10
        )
        
        assert result == {'E. coli': 2, 'B. subtilis': 1}

    def test_count_single_sample_no_matches_above_threshold(self, tmp_path):
        """Test behavior when no samples above threshold."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        self.summary_data.to_csv(summary_file, sep='\t')
        self.metadata_data.to_csv(metadata_file, sep='\t')
        
        result = count_single_sample(
            summary_file, metadata_file, 'GCF_000001.1', 'organism',
            threshold=0.95, top_n=10
        )
        
        assert result == {}

    def test_count_single_sample_missing_summary_column(self, tmp_path):
        """Test error when summary column doesn't exist."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        self.summary_data.to_csv(summary_file, sep='\t')
        self.metadata_data.to_csv(metadata_file, sep='\t')
        
        with pytest.raises(ProcessingError, match="Column MISSING not found in summary file"):
            count_single_sample(
                summary_file, metadata_file, 'MISSING', 'organism'
            )

    def test_count_single_sample_missing_metadata_column(self, tmp_path):
        """Test error when metadata column doesn't exist."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        self.summary_data.to_csv(summary_file, sep='\t')
        self.metadata_data.to_csv(metadata_file, sep='\t')
        
        with pytest.raises(ProcessingError, match="Column missing not found in metadata file"):
            count_single_sample(
                summary_file, metadata_file, 'GCF_000001.1', 'missing'
            )

    def test_count_single_sample_no_metadata_matches(self, tmp_path):
        """Test behavior when no metadata matches selected accessions."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        # Create metadata with different index
        different_metadata = pd.DataFrame({
            'organism': ['X. test']
        }, index=['SRR999'])
        
        self.summary_data.to_csv(summary_file, sep='\t')
        different_metadata.to_csv(metadata_file, sep='\t')
        
        result = count_single_sample(
            summary_file, metadata_file, 'GCF_000001.1', 'organism',
            threshold=0.5
        )
        
        assert result == {}

    def test_count_single_sample_top_n_limit(self, tmp_path):
        """Test top_n parameter limits results."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        # Create data with many different values
        large_metadata = pd.DataFrame({
            'organism': [f'Species_{i}' for i in range(20)]
        }, index=[f'SRR{i:03d}' for i in range(1, 21)])
        
        large_summary = pd.DataFrame({
            'GCF_000001.1': [0.8] * 20  # All above threshold
        }, index=[f'SRR{i:03d}' for i in range(1, 21)])
        
        large_summary.to_csv(summary_file, sep='\t')
        large_metadata.to_csv(metadata_file, sep='\t')
        
        result = count_single_sample(
            summary_file, metadata_file, 'GCF_000001.1', 'organism',
            threshold=0.5, top_n=5
        )
        
        assert len(result) == 5

    def test_count_single_sample_file_read_error(self, tmp_path):
        """Test handling of file reading errors."""
        summary_file = tmp_path / "missing_summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        self.metadata_data.to_csv(metadata_file, sep='\t')
        
        with pytest.raises(ProcessingError, match="Error counting single sample metadata"):
            count_single_sample(
                summary_file, metadata_file, 'GCF_000001.1', 'organism'
            )

    def test_count_single_sample_string_paths(self, tmp_path):
        """Test function accepts string paths."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        
        self.summary_data.to_csv(summary_file, sep='\t')
        self.metadata_data.to_csv(metadata_file, sep='\t')
        
        result = count_single_sample(
            str(summary_file), str(metadata_file), 'GCF_000001.1', 'organism',
            threshold=0.5
        )
        
        assert result == {'E. coli': 3}


class TestFilterSamplesByContainment:
    """Test filter_samples_by_containment function."""

    def setup_method(self):
        """Set up test data."""
        self.summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.2, 0.6, 0.05, 0.9],
            'GCF_000002.1': [0.1, 0.7, 0.3, 0.15, 0.2],
            'max_containment': [0.8, 0.7, 0.6, 0.15, 0.9],
            'max_containment_annotation': ['G1', 'G2', 'G1', 'G2', 'G1']
        }, index=['SRR001', 'SRR002', 'SRR003', 'SRR004', 'SRR005'])

    def test_filter_samples_by_specific_genome(self, tmp_path):
        """Test filtering by specific genome ID."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = filter_samples_by_containment(
            summary_file, threshold=0.5, genome_id='GCF_000001.1'
        )
        
        # Should return SRR001 (0.8), SRR003 (0.6), SRR005 (0.9)
        assert len(result) == 3
        assert 'SRR001' in result.index
        assert 'SRR003' in result.index
        assert 'SRR005' in result.index

    def test_filter_samples_by_max_containment(self, tmp_path):
        """Test filtering by max_containment column."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = filter_samples_by_containment(
            summary_file, threshold=0.65, genome_id=None
        )
        
        # Should return samples with max_containment > 0.65
        # SRR001 (0.8), SRR002 (0.7), SRR005 (0.9)
        assert len(result) == 3
        assert 'SRR001' in result.index
        assert 'SRR002' in result.index
        assert 'SRR005' in result.index

    def test_filter_samples_missing_genome(self, tmp_path):
        """Test error when specified genome doesn't exist."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        with pytest.raises(ProcessingError, match="Genome MISSING not found in summary file"):
            filter_samples_by_containment(
                summary_file, threshold=0.5, genome_id='MISSING'
            )

    def test_filter_samples_no_matches(self, tmp_path):
        """Test behavior when no samples meet threshold."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = filter_samples_by_containment(
            summary_file, threshold=0.95, genome_id='GCF_000001.1'
        )
        
        assert len(result) == 0

    def test_filter_samples_file_error(self, tmp_path):
        """Test handling of file reading errors."""
        summary_file = tmp_path / "missing.tsv"
        
        with pytest.raises(ProcessingError, match="Error filtering samples by containment"):
            filter_samples_by_containment(summary_file, threshold=0.5)

    def test_filter_samples_string_path(self, tmp_path):
        """Test function accepts string path."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = filter_samples_by_containment(
            str(summary_file), threshold=0.5, genome_id='GCF_000001.1'
        )
        
        assert len(result) == 3

    def test_filter_samples_edge_case_exact_threshold(self, tmp_path):
        """Test filtering with samples exactly at threshold."""
        summary_file = tmp_path / "summary.tsv"
        # Modify data to have exact threshold values
        edge_data = self.summary_data.copy()
        edge_data.loc['SRR003', 'GCF_000001.1'] = 0.5  # Exactly at threshold
        edge_data.to_csv(summary_file, sep='\t')
        
        result = filter_samples_by_containment(
            summary_file, threshold=0.5, genome_id='GCF_000001.1'
        )
        
        # Should NOT include SRR003 since it's not > 0.5
        assert len(result) == 2
        assert 'SRR001' in result.index
        assert 'SRR005' in result.index
        assert 'SRR003' not in result.index


class TestFindCoOccurringGenomes:
    """Test find_co_occurring_genomes function."""

    def setup_method(self):
        """Set up test data."""
        # Create data where genomes co-occur in some samples
        self.summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.2, 0.6, 0.05, 0.9, 0.7],  # Present in 4 samples
            'GCF_000002.1': [0.6, 0.7, 0.3, 0.15, 0.2, 0.8],  # Present in 3 samples  
            'GCF_000003.1': [0.1, 0.8, 0.9, 0.05, 0.6, 0.2],  # Present in 3 samples
            'max_containment': [0.8, 0.8, 0.9, 0.15, 0.9, 0.8],
            'max_containment_annotation': ['G1', 'G2', 'G3', 'G2', 'G1', 'G2']
        }, index=['SRR001', 'SRR002', 'SRR003', 'SRR004', 'SRR005', 'SRR006'])

    def test_find_co_occurring_genomes_success(self, tmp_path):
        """Test successful co-occurrence analysis."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = find_co_occurring_genomes(
            summary_file, threshold=0.5, min_samples=3
        )
        
        # Should include genomes present in at least 3 samples
        expected_genomes = ['GCF_000001.1', 'GCF_000002.1', 'GCF_000003.1']
        assert list(result.index) == expected_genomes
        assert list(result.columns) == expected_genomes
        
        # Check co-occurrence counts (samples where both are present)
        # GCF_000001.1 and GCF_000002.1 co-occur in SRR001, SRR006 (threshold 0.5)
        assert int(result.loc['GCF_000001.1', 'GCF_000002.1']) == 2

    def test_find_co_occurring_genomes_no_frequent_genomes(self, tmp_path):
        """Test behavior when no genomes meet min_samples requirement."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = find_co_occurring_genomes(
            summary_file, threshold=0.5, min_samples=10
        )
        
        assert result.empty

    def test_find_co_occurring_genomes_no_genome_columns(self, tmp_path):
        """Test error when no valid genome columns found."""
        # Create data with only known metadata columns
        invalid_data = pd.DataFrame({
            'max_containment': [0.8, 0.7],
            'max_containment_annotation': ['A', 'B']
        }, index=['SRR001', 'SRR002'])

        summary_file = tmp_path / "summary.tsv"
        invalid_data.to_csv(summary_file, sep='\t')

        with pytest.raises(ProcessingError, match="No genome columns found in summary file"):
            find_co_occurring_genomes(summary_file)

    def test_find_co_occurring_genomes_high_threshold(self, tmp_path):
        """Test with high threshold that excludes most samples."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = find_co_occurring_genomes(
            summary_file, threshold=0.85, min_samples=1
        )
        
        # Only very high values should pass
        # Should still create matrix but with fewer co-occurrences
        assert not result.empty

    def test_find_co_occurring_genomes_self_cooccurrence(self, tmp_path):
        """Test that diagonal shows correct self co-occurrence."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = find_co_occurring_genomes(
            summary_file, threshold=0.5, min_samples=3
        )
        
        # Diagonal should show number of samples where each genome is present
        for genome in result.index:
            self_count = int(result.loc[genome, genome])
            # Count manually from test data
            genome_presence = (self.summary_data[genome] > 0.5).sum()
            assert self_count == genome_presence

    def test_find_co_occurring_genomes_file_error(self, tmp_path):
        """Test handling of file reading errors."""
        summary_file = tmp_path / "missing.tsv"
        
        with pytest.raises(ProcessingError, match="Error finding co-occurring genomes"):
            find_co_occurring_genomes(summary_file)

    def test_find_co_occurring_genomes_string_path(self, tmp_path):
        """Test function accepts string path."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = find_co_occurring_genomes(
            str(summary_file), threshold=0.5, min_samples=3
        )
        
        assert not result.empty

    def test_find_co_occurring_genomes_matrix_symmetry(self, tmp_path):
        """Test that co-occurrence matrix is symmetric."""
        summary_file = tmp_path / "summary.tsv"
        self.summary_data.to_csv(summary_file, sep='\t')
        
        result = find_co_occurring_genomes(
            summary_file, threshold=0.5, min_samples=3
        )
        
        # Matrix should be symmetric
        for i in result.index:
            for j in result.columns:
                assert result.loc[i, j] == result.loc[j, i]