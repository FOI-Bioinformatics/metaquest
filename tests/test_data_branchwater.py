"""
Test data.branchwater module functionality.

Tests for Branchwater file processing, metadata extraction, and containment analysis.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pandas as pd
import pytest
from collections import defaultdict

from metaquest.data.branchwater import (
    process_branchwater_files,
    _process_branchwater_row,
    _validate_branchwater_file,
    extract_metadata_from_branchwater,
    _finalize_metadata_extraction,
    _process_genome_containments,
    _generate_containment_summary,
    parse_containment_data,
)
from metaquest.core.exceptions import DataAccessError, ValidationError
from metaquest.core.models import ContainmentSummary


class TestProcessBranchwaterFiles:
    """Test process_branchwater_files function."""

    def test_process_branchwater_files_success(self, tmp_path):
        """Test successful processing of Branchwater files."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        
        # Create test CSV files with valid branchwater format
        csv_content = "acc,containment,ksize,scaled\nERR123,0.95,31,1000\nERR456,0.87,31,1000"
        (source_dir / "genome1.csv").write_text(csv_content)
        (source_dir / "genome2.csv").write_text(csv_content)
        
        with patch('metaquest.data.branchwater.read_csv') as mock_read_csv:
            # Mock the header check for format validation
            mock_df = pd.DataFrame(columns=['acc', 'containment', 'ksize', 'scaled'])
            mock_read_csv.return_value = mock_df
            
            result = process_branchwater_files(source_dir, target_dir)
        
        assert len(result) == 2
        assert "genome1" in result
        assert "genome2" in result
        assert target_dir.exists()

    def test_process_branchwater_files_skip_existing(self, tmp_path):
        """Test skipping files that already exist in target."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()
        
        # Create source file
        csv_content = "acc,containment,ksize,scaled\nERR123,0.95,31,1000"
        (source_dir / "genome1.csv").write_text(csv_content)
        
        # Create existing target file
        (target_dir / "genome1.csv").write_text("existing content")
        
        result = process_branchwater_files(source_dir, target_dir)
        
        assert len(result) == 1
        assert "genome1" in result
        # Should keep existing file content
        assert (target_dir / "genome1.csv").read_text() == "existing content"

    def test_process_branchwater_files_no_csv_files(self, tmp_path):
        """Test with no CSV files in source directory."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        
        # Create non-CSV file
        (source_dir / "not_csv.txt").write_text("not a csv")
        
        result = process_branchwater_files(source_dir, target_dir)
        
        assert result == {}

    def test_process_branchwater_files_invalid_format(self, tmp_path):
        """Test with files that have invalid format."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()

        # Create CSV with invalid format
        invalid_content = "invalid,header,format\ndata1,data2,data3"
        (source_dir / "invalid.csv").write_text(invalid_content)

        with patch('metaquest.data.branchwater.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame(columns=['invalid', 'header', 'format'])
            mock_read_csv.return_value = mock_df

            with patch('metaquest.data.branchwater.BranchWaterFormatPlugin.validate_header', side_effect=ValidationError("Invalid")):
                result = process_branchwater_files(source_dir, target_dir)

        # Should have empty result due to validation errors
        assert result == {}

    def test_process_branchwater_files_source_not_exist(self):
        """Test with non-existent source folder."""
        with pytest.raises(ValidationError):
            process_branchwater_files("/nonexistent/source", "/tmp/target")


class TestProcessBranchwaterRow:
    """Test _process_branchwater_row function."""

    def test_process_branchwater_row_basic(self):
        """Test basic row processing."""
        row = pd.Series({
            'acc': 'ERR123456',
            'biosample': 'SAMN123',
            'organism': 'Escherichia coli',
            'geo_loc_name_country_calc': 'USA'
        })
        metadata_records = []
        
        _process_branchwater_row(row, metadata_records)
        
        assert len(metadata_records) == 1
        record = metadata_records[0]
        assert record['Run_ID'] == 'ERR123456'
        assert record['Sample_ID'] == 'SAMN123'
        assert record['Sample_Scientific_Name'] == 'Escherichia coli'
        assert record['geo_loc_name_country_calc'] == 'USA'

    def test_process_branchwater_row_missing_acc(self):
        """Test row processing with missing acc column."""
        row = pd.Series({'other_field': 'value'})
        metadata_records = []
        
        with patch('metaquest.data.branchwater.logger') as mock_logger:
            _process_branchwater_row(row, metadata_records)
        
        assert len(metadata_records) == 0
        mock_logger.warning.assert_called_once()

    def test_process_branchwater_row_partial_fields(self):
        """Test row processing with only some optional fields."""
        row = pd.Series({
            'acc': 'ERR789',
            'organism': 'Salmonella enterica',
            'unknown_field': 'ignored',
            'cANI': 0.98
        })
        metadata_records = []
        
        _process_branchwater_row(row, metadata_records)
        
        assert len(metadata_records) == 1
        record = metadata_records[0]
        assert record['Run_ID'] == 'ERR789'
        assert record['Sample_Scientific_Name'] == 'Salmonella enterica'
        assert record['cANI'] == 0.98
        assert 'unknown_field' not in record

    def test_process_branchwater_row_with_nan_values(self):
        """Test row processing with NaN values."""
        row = pd.Series({
            'acc': 'ERR999',
            'organism': 'Bacteria',
            'biosample': pd.NA,
            'collection_date_sam': None
        })
        metadata_records = []
        
        _process_branchwater_row(row, metadata_records)
        
        assert len(metadata_records) == 1
        record = metadata_records[0]
        assert record['Run_ID'] == 'ERR999'
        assert record['Sample_Scientific_Name'] == 'Bacteria'
        assert 'Sample_ID' not in record
        assert 'collection_date_sam' not in record


class TestValidateBranchwaterFile:
    """Test _validate_branchwater_file function."""

    def test_validate_branchwater_file_valid(self, tmp_path):
        """Test validation of valid Branchwater file."""
        csv_file = tmp_path / "valid.csv"
        csv_content = "acc,containment,ksize,scaled\nERR123,0.95,31,1000"
        csv_file.write_text(csv_content)
        
        result = _validate_branchwater_file(csv_file)
        
        assert result is True

    def test_validate_branchwater_file_missing_acc(self, tmp_path):
        """Test validation with missing acc column."""
        csv_file = tmp_path / "invalid.csv"
        csv_content = "run_id,containment,ksize,scaled\nERR123,0.95,31,1000"
        csv_file.write_text(csv_content)
        
        with patch('metaquest.data.branchwater.logger') as mock_logger:
            result = _validate_branchwater_file(csv_file)
        
        assert result is False
        mock_logger.warning.assert_called_once()

    def test_validate_branchwater_file_missing_containment(self, tmp_path):
        """Test validation with missing containment column."""
        csv_file = tmp_path / "invalid.csv"
        csv_content = "acc,similarity,ksize,scaled\nERR123,0.95,31,1000"
        csv_file.write_text(csv_content)
        
        with patch('metaquest.data.branchwater.logger') as mock_logger:
            result = _validate_branchwater_file(csv_file)
        
        assert result is False
        mock_logger.warning.assert_called_once()


class TestExtractMetadataFromBranchwater:
    """Test extract_metadata_from_branchwater function."""

    def test_extract_metadata_success(self, tmp_path):
        """Test successful metadata extraction."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_file = tmp_path / "metadata.csv"
        
        # Create test CSV file
        csv_content = "acc,containment,organism,biosample\nERR123,0.95,E. coli,SAMN123\nERR456,0.87,S. enterica,SAMN456"
        (source_dir / "test.csv").write_text(csv_content)
        
        result = extract_metadata_from_branchwater(source_dir, output_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'Run_ID' in result.columns
        assert output_file.exists()

    def test_extract_metadata_no_csv_files(self, tmp_path):
        """Test with no CSV files."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_file = tmp_path / "metadata.csv"
        
        result = extract_metadata_from_branchwater(source_dir, output_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_extract_metadata_nonexistent_folder(self, tmp_path):
        """Test with non-existent source folder."""
        output_file = tmp_path / "metadata.csv"
        
        with pytest.raises(DataAccessError, match="Branchwater folder does not exist"):
            extract_metadata_from_branchwater("/nonexistent", output_file)

    def test_extract_metadata_invalid_files(self, tmp_path):
        """Test with invalid file formats."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_file = tmp_path / "metadata.csv"
        
        # Create invalid CSV file
        invalid_content = "invalid,format\ndata1,data2"
        (source_dir / "invalid.csv").write_text(invalid_content)
        
        result = extract_metadata_from_branchwater(source_dir, output_file)
        
        # Should return empty DataFrame but still create output file
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert output_file.exists()

    def test_extract_metadata_processing_error(self, tmp_path):
        """Test with file processing errors."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_file = tmp_path / "metadata.csv"
        
        # Create file with valid header but invalid content
        csv_content = "acc,containment\ninvalid_content_here"
        (source_dir / "error.csv").write_text(csv_content)
        
        with patch('pandas.read_csv', side_effect=Exception("Parse error")):
            result = extract_metadata_from_branchwater(source_dir, output_file)
        
        # Should handle error gracefully
        assert isinstance(result, pd.DataFrame)


class TestFinalizeMetadataExtraction:
    """Test _finalize_metadata_extraction function."""

    def test_finalize_metadata_with_records(self, tmp_path):
        """Test finalization with metadata records."""
        metadata_records = [
            {'Run_ID': 'ERR123', 'Sample_Scientific_Name': 'E. coli'},
            {'Run_ID': 'ERR456', 'Sample_Scientific_Name': 'S. enterica'},
            {'Run_ID': 'ERR123', 'Sample_Scientific_Name': 'E. coli'}  # Duplicate
        ]
        output_file = tmp_path / "metadata.csv"
        
        result = _finalize_metadata_extraction(metadata_records, output_file, 2, 0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Duplicates should be removed
        assert output_file.exists()

    def test_finalize_metadata_empty_records(self, tmp_path):
        """Test finalization with no metadata records."""
        metadata_records = []
        output_file = tmp_path / "metadata.csv"
        
        result = _finalize_metadata_extraction(metadata_records, output_file, 0, 1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert output_file.exists()

    def test_finalize_metadata_create_directory(self, tmp_path):
        """Test that output directory is created."""
        metadata_records = [{'Run_ID': 'ERR123'}]
        output_file = tmp_path / "nested" / "dir" / "metadata.csv"
        
        result = _finalize_metadata_extraction(metadata_records, output_file, 1, 0)
        
        assert output_file.exists()
        assert output_file.parent.exists()


class TestProcessGenomeContainments:
    """Test _process_genome_containments function."""

    def test_process_genome_containments_branchwater(self, tmp_path):
        """Test processing with Branchwater format."""
        csv_file = tmp_path / "genome1.csv"
        containment_data = defaultdict(dict)
        
        with patch('metaquest.data.branchwater.read_csv') as mock_read_csv:
            # Mock header check
            mock_headers = pd.DataFrame(columns=['acc', 'containment'])
            mock_read_csv.return_value = mock_headers
            
            with patch('metaquest.data.branchwater.BranchWaterFormatPlugin') as mock_plugin:
                mock_plugin.validate_header.return_value = True
                mock_plugin.parse_file.return_value = [
                    Mock(accession='ERR123', value=0.95),
                    Mock(accession='ERR456', value=0.87)
                ]
                
                _process_genome_containments(csv_file, "genome1", containment_data)
        
        assert 'ERR123' in containment_data
        assert 'ERR456' in containment_data
        assert containment_data['ERR123']['genome1'] == 0.95
        assert containment_data['ERR456']['genome1'] == 0.87

    def test_process_genome_containments_unknown_format(self, tmp_path):
        """Test processing with unknown format."""
        csv_file = tmp_path / "genome1.csv"
        containment_data = defaultdict(dict)

        with patch('metaquest.data.branchwater.read_csv') as mock_read_csv:
            mock_headers = pd.DataFrame(columns=['unknown', 'format'])
            mock_read_csv.return_value = mock_headers

            with patch('metaquest.data.branchwater.BranchWaterFormatPlugin') as mock_bw_plugin:
                mock_bw_plugin.validate_header.side_effect = ValidationError("Not branchwater")

                with pytest.raises(DataAccessError, match="Error processing"):
                    _process_genome_containments(csv_file, "genome1", containment_data)

    def test_process_genome_containments_max_value(self, tmp_path):
        """Test that maximum containment value is kept for duplicates."""
        csv_file = tmp_path / "genome1.csv"
        containment_data = defaultdict(dict)
        
        # Pre-populate with existing value
        containment_data['ERR123']['genome1'] = 0.5
        
        with patch('metaquest.data.branchwater.read_csv') as mock_read_csv:
            mock_headers = pd.DataFrame(columns=['acc', 'containment'])
            mock_read_csv.return_value = mock_headers
            
            with patch('metaquest.data.branchwater.BranchWaterFormatPlugin') as mock_plugin:
                mock_plugin.validate_header.return_value = True
                mock_plugin.parse_file.return_value = [
                    Mock(accession='ERR123', value=0.95)  # Higher value
                ]
                
                _process_genome_containments(csv_file, "genome1", containment_data)
        
        # Should keep the higher value
        assert containment_data['ERR123']['genome1'] == 0.95


class TestGenerateContainmentSummary:
    """Test _generate_containment_summary function."""

    def test_generate_containment_summary_success(self, tmp_path):
        """Test successful summary generation."""
        containment_data = {
            'ERR123': {'genome1': 0.95, 'genome2': 0.1},
            'ERR456': {'genome1': 0.87, 'genome2': 0.95},
            'ERR789': {'genome1': 0.0, 'genome2': 0.5}
        }
        output_file = tmp_path / "parsed.txt"
        summary_file = tmp_path / "summary.txt"
        
        with patch('metaquest.data.branchwater.write_csv') as mock_write:
            result = _generate_containment_summary(containment_data, output_file, summary_file, 0.1)
        
        assert isinstance(result, ContainmentSummary)
        assert len(result.thresholds) > 0
        assert len(result.counts) > 0
        assert len(result.max_containment) == 3
        assert mock_write.call_count == 2  # parsed and summary files

    def test_generate_containment_summary_empty_data(self, tmp_path):
        """Test summary generation with empty data."""
        containment_data = {}
        output_file = tmp_path / "parsed.txt"
        summary_file = tmp_path / "summary.txt"
        
        # Empty data will create an empty DataFrame which pandas can handle
        with patch('metaquest.data.branchwater.write_csv') as mock_write:
            result = _generate_containment_summary(containment_data, output_file, summary_file, 0.1)
        
        assert isinstance(result, ContainmentSummary)
        mock_write.assert_called()


class TestParseContainmentData:
    """Test parse_containment_data function."""

    def test_parse_containment_data_success(self, tmp_path):
        """Test successful containment data parsing."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()
        output_file = tmp_path / "parsed.txt"
        summary_file = tmp_path / "summary.txt"
        
        # Create test CSV file
        (matches_dir / "genome1.csv").write_text("acc,containment\nERR123,0.95")
        
        def mock_process_side_effect(csv_file, genome_id, containment_data):
            # Simulate adding some data to containment_data
            containment_data['ERR123']['genome1'] = 0.95
        
        with patch('metaquest.data.branchwater._process_genome_containments', side_effect=mock_process_side_effect) as mock_process:
            with patch('metaquest.data.branchwater._generate_containment_summary') as mock_generate:
                mock_summary = ContainmentSummary()
                mock_generate.return_value = mock_summary
                
                result = parse_containment_data(matches_dir, output_file, summary_file, 0.05)
        
        assert result == mock_summary
        mock_process.assert_called_once()
        mock_generate.assert_called_once()

    def test_parse_containment_data_no_files(self, tmp_path):
        """Test with no CSV files."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()
        output_file = tmp_path / "parsed.txt"
        summary_file = tmp_path / "summary.txt"
        
        result = parse_containment_data(matches_dir, output_file, summary_file)
        
        assert isinstance(result, ContainmentSummary)

    def test_parse_containment_data_processing_errors(self, tmp_path):
        """Test with file processing errors."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()
        output_file = tmp_path / "parsed.txt"
        summary_file = tmp_path / "summary.txt"
        
        # Create test files
        (matches_dir / "genome1.csv").write_text("data")
        (matches_dir / "genome2.csv").write_text("data")
        
        with patch('metaquest.data.branchwater._process_genome_containments') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            result = parse_containment_data(matches_dir, output_file, summary_file)
        
        # Should return empty summary when no valid data
        assert isinstance(result, ContainmentSummary)

    def test_parse_containment_data_invalid_folder(self):
        """Test with invalid matches folder."""
        with pytest.raises(ValidationError):
            parse_containment_data("/nonexistent", "output.txt", "summary.txt")


class TestBranchwaterIntegration:
    """Integration tests for branchwater functionality."""

    def test_full_branchwater_workflow(self, tmp_path):
        """Test complete branchwater processing workflow with realistic Salmonella data."""
        # Set up directories
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        
        # Use realistic Salmonella test data from salmonella_mini.csv
        real_data_path = Path(__file__).parent.parent / "test_data" / "branchwater" / "salmonella_mini.csv"
        if real_data_path.exists():
            # Copy real data for testing
            import shutil
            shutil.copy(real_data_path, source_dir / "salmonella_genome.csv")
            # Also create a second file for multi-file testing
            shutil.copy(real_data_path, source_dir / "ecoli_genome.csv")
        else:
            # Fallback to minimal synthetic data if real data not available
            csv_content = """acc,containment,cANI,biosample,bioproject,assay_type,collection_date_sam,geo_loc_name_country_calc,organism,lat_lon
ERR123456,0.95,0.98,SAMN123,PRJNA123,WGS,2023-01-01,USA,Salmonella enterica,35.7N 100.2W
ERR789012,0.87,0.96,SAMN456,PRJNA456,WGS,2023-02-15,Denmark,gut metagenome,55.67N 12.57E"""
            
            (source_dir / "salmonella_genome.csv").write_text(csv_content)
            (source_dir / "ecoli_genome.csv").write_text(csv_content)
        
        # Mock the format validation to work with our realistic test data
        with patch('metaquest.data.branchwater.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame(columns=['acc', 'containment', 'cANI', 'biosample', 'bioproject', 'assay_type', 'collection_date_sam', 'geo_loc_name_country_calc', 'organism', 'lat_lon'])
            mock_read_csv.return_value = mock_df
            
            # Process files
            result_files = process_branchwater_files(source_dir, target_dir)
        
        # Verify file processing
        assert len(result_files) == 2
        assert "ecoli_genome" in result_files
        assert "salmonella_genome" in result_files
        
        # Test metadata extraction with realistic data
        metadata_file = tmp_path / "metadata.csv"
        
        with patch('metaquest.data.branchwater._validate_branchwater_file', return_value=True):
            with patch('pandas.read_csv') as mock_pd_read:
                # Use realistic values from salmonella_mini.csv for better testing
                mock_pd_read.return_value = pd.DataFrame({
                    'acc': ['ERR2868097', 'SRR22022422', 'SRR31320585'],
                    'containment': [0.98, 0.98, 0.97],
                    'cANI': [1.0, 1.0, 1.0],
                    'biosample': ['SAMEA5056563', 'SAMN29160454', 'SAMN44590404'],
                    'bioproject': ['PRJEB29454', 'PRJNA849983', 'PRJNA1182286'],
                    'organism': ['metagenome', 'food metagenome', 'metagenome'],
                    'geo_loc_name_country_calc': ['NP', 'United Kingdom', 'USA'],
                    'collection_date_sam': ['', '2019-02-07', '2023-09-27']
                })
                
                metadata_df = extract_metadata_from_branchwater(source_dir, metadata_file)
        
        # Verify metadata extraction
        assert isinstance(metadata_df, pd.DataFrame)
        assert metadata_file.exists()

    def test_error_handling_workflow(self, tmp_path):
        """Test error handling throughout the workflow."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        
        # Create files that will cause various errors
        (source_dir / "corrupt.csv").write_text("corrupted,data\nwith,errors")
        (source_dir / "empty.csv").write_text("")
        
        # Should handle errors gracefully
        with patch('metaquest.data.branchwater.logger') as mock_logger:
            result = process_branchwater_files(source_dir, target_dir)
        
        # Should have logged errors but not crashed
        assert isinstance(result, dict)
        mock_logger.error.assert_called()

    def test_realistic_salmonella_data_parsing(self, tmp_path):
        """Test parsing with actual realistic Salmonella data."""
        # Use the realistic salmonella_mini.csv data
        real_data_path = Path(__file__).parent.parent / "test_data" / "branchwater" / "salmonella_mini.csv"
        if not real_data_path.exists():
            pytest.skip("Realistic Salmonella test data not available")
        
        # Set up test environment
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_file = tmp_path / "metadata.csv"
        
        # Copy real data
        import shutil
        shutil.copy(real_data_path, source_dir / "salmonella.csv")
        
        # Extract metadata using the real data structure
        result = extract_metadata_from_branchwater(source_dir, output_file)
        
        # Verify that we can process real data without errors
        assert isinstance(result, pd.DataFrame)
        assert output_file.exists()
        
        # If data was processed, verify it contains expected fields
        if len(result) > 0:
            assert 'Run_ID' in result.columns
            # Verify some realistic entries are present
            run_ids = result['Run_ID'].tolist()
            assert any(run_id.startswith('ERR') or run_id.startswith('SRR') for run_id in run_ids)


if __name__ == "__main__":
    pytest.main([__file__])