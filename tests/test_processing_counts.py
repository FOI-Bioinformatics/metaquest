"""
Test processing.counts module functionality.

Tests for metadata counting and analysis functions.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import pandas as pd

from metaquest.processing.counts import (
    _validate_metadata_column,
    _get_genome_columns, 
    _process_genome_accessions,
    count_metadata,
    count_metadata_by_category,
    summarize_metadata_column
)
from metaquest.core.exceptions import ProcessingError


class TestValidateMetadataColumn:
    """Test _validate_metadata_column function."""

    def test_validate_existing_column(self):
        """Test validation with existing column."""
        df = pd.DataFrame({'organism': ['E. coli', 'S. aureus'], 'location': ['USA', 'UK']})
        
        # Should not raise exception
        _validate_metadata_column(df, 'organism')
        _validate_metadata_column(df, 'location')

    def test_validate_nonexistent_column(self):
        """Test validation with non-existent column."""
        df = pd.DataFrame({'organism': ['E. coli', 'S. aureus'], 'location': ['USA', 'UK']})
        
        with pytest.raises(ProcessingError, match="Column 'nonexistent' not found"):
            _validate_metadata_column(df, 'nonexistent')

    def test_validate_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ProcessingError, match="Column 'any_column' not found"):
            _validate_metadata_column(df, 'any_column')


class TestGetGenomeColumns:
    """Test _get_genome_columns function."""

    def test_get_gcf_columns(self):
        """Test getting GCF genome columns."""
        df = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9],
            'GCF_000002.1': [0.7, 0.6],
            'max_containment': [0.8, 0.9],
        })

        genome_cols = _get_genome_columns(df)
        assert set(genome_cols) == {'GCF_000001.1', 'GCF_000002.1'}

    def test_get_gca_columns(self):
        """Test getting GCA genome columns."""
        df = pd.DataFrame({
            'GCA_000001.1': [0.8, 0.9],
            'GCA_000002.1': [0.7, 0.6],
            'max_containment': [0.8, 0.9],
        })

        genome_cols = _get_genome_columns(df)
        assert set(genome_cols) == {'GCA_000001.1', 'GCA_000002.1'}

    def test_get_mixed_genome_columns(self):
        """Test getting mixed GCF and GCA columns."""
        df = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9],
            'GCA_000002.1': [0.7, 0.6],
            'max_containment': [0.8, 0.9],
        })

        genome_cols = _get_genome_columns(df)
        assert set(genome_cols) == {'GCF_000001.1', 'GCA_000002.1'}

    def test_no_genome_columns(self):
        """Test when no genome columns are found."""
        df = pd.DataFrame({
            'max_containment': [0.5, 0.6],
            'max_containment_annotation': ['A', 'B'],
        })

        with pytest.raises(ProcessingError, match="No genome columns found"):
            _get_genome_columns(df)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ProcessingError, match="No genome columns found"):
            _get_genome_columns(df)


class TestProcessGenomeAccessions:
    """Test _process_genome_accessions function."""

    def test_process_genome_accessions_success(self):
        """Test successful processing of genome accessions."""
        # Create test data
        summary_df = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9, 0.3, 0.7],
        }, index=['SRR1', 'SRR2', 'SRR3', 'SRR4'])
        
        metadata_df = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'B. subtilis', 'E. coli'],
        }, index=['SRR1', 'SRR2', 'SRR3', 'SRR4'])
        
        df_list = []
        threshold = 0.5
        
        result = _process_genome_accessions(
            'GCF_000001.1', summary_df, threshold, metadata_df, 'organism', df_list
        )
        
        # Should process 3 accessions above threshold (0.8, 0.9, 0.7)
        assert result == 3
        assert len(df_list) == 1
        
        # Check the counts DataFrame
        count_df = df_list[0]
        assert 'GCF_000001.1' in count_df.columns
        assert count_df['GCF_000001.1']['E. coli'] == 2  # SRR1 and SRR4
        assert count_df['GCF_000001.1']['S. aureus'] == 1  # SRR2

    def test_process_genome_accessions_no_matches(self):
        """Test when no accessions match the threshold."""
        summary_df = pd.DataFrame({
            'GCF_000001.1': [0.1, 0.2, 0.3],
        }, index=['SRR1', 'SRR2', 'SRR3'])
        
        metadata_df = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'B. subtilis'],
        }, index=['SRR1', 'SRR2', 'SRR3'])
        
        df_list = []
        threshold = 0.5
        
        result = _process_genome_accessions(
            'GCF_000001.1', summary_df, threshold, metadata_df, 'organism', df_list
        )
        
        assert result == 0
        assert len(df_list) == 0

    def test_process_genome_accessions_no_metadata_match(self):
        """Test when accessions don't have matching metadata."""
        summary_df = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9],
        }, index=['SRR1', 'SRR2'])
        
        metadata_df = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus'],
        }, index=['SRR3', 'SRR4'])  # Different indices
        
        df_list = []
        threshold = 0.5
        
        result = _process_genome_accessions(
            'GCF_000001.1', summary_df, threshold, metadata_df, 'organism', df_list
        )
        
        assert result == 0
        assert len(df_list) == 0

    @patch('metaquest.processing.counts.logger')
    def test_process_genome_accessions_logging(self, mock_logger):
        """Test that appropriate warning messages are logged."""
        summary_df = pd.DataFrame({
            'GCF_000001.1': [0.1, 0.2],
        }, index=['SRR1', 'SRR2'])
        
        metadata_df = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus'],
        }, index=['SRR1', 'SRR2'])
        
        df_list = []
        threshold = 0.5
        
        _process_genome_accessions(
            'GCF_000001.1', summary_df, threshold, metadata_df, 'organism', df_list
        )
        
        mock_logger.warning.assert_called_with(
            "No accessions found for GCF_000001.1 above threshold 0.5"
        )


class TestCountGenomeMetadata:
    """Test count_metadata function."""

    def test_count_metadata_success(self, tmp_path):
        """Test successful metadata counting."""
        # Create test files
        summary_file = tmp_path / "summary.txt"
        metadata_file = tmp_path / "metadata.txt"
        output_file = tmp_path / "counts.txt"
        
        # Create summary data
        summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9, 0.3, 0.7],
            'GCF_000002.1': [0.6, 0.2, 0.8, 0.9],
        }, index=['SRR1', 'SRR2', 'SRR3', 'SRR4'])
        summary_data.to_csv(summary_file, sep='\t')
        
        # Create metadata data
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'B. subtilis', 'E. coli'],
        }, index=['SRR1', 'SRR2', 'SRR3', 'SRR4'])
        metadata_data.to_csv(metadata_file, sep='\t', index_label='accession')
        
        result = count_metadata(
            summary_file, metadata_file, 'organism', 0.5, output_file
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'GCF_000001.1' in result.columns
        assert 'GCF_000002.1' in result.columns
        assert output_file.exists()

    def test_count_metadata_file_not_found(self, tmp_path):
        """Test with non-existent files."""
        with pytest.raises(ProcessingError, match="Error counting metadata"):
            count_metadata(
                "nonexistent_summary.txt",
                "nonexistent_metadata.txt", 
                "organism",
                0.5,
                tmp_path / "output.txt"
            )

    def test_count_metadata_invalid_column(self, tmp_path):
        """Test with invalid metadata column."""
        summary_file = tmp_path / "summary.txt"
        metadata_file = tmp_path / "metadata.txt"
        
        # Create test data
        summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9],
        }, index=['SRR1', 'SRR2'])
        summary_data.to_csv(summary_file, sep='\t')
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus'],
        }, index=['SRR1', 'SRR2'])
        metadata_data.to_csv(metadata_file, sep='\t', index_label='accession')
        
        with pytest.raises(ProcessingError, match="Column 'invalid_column' not found"):
            count_metadata(
                summary_file, metadata_file, 'invalid_column', 0.5, tmp_path / "output.txt"
            )


class TestCountMetadataByCategory:
    """Test count_metadata_by_category function."""

    def test_count_metadata_by_category_success(self, tmp_path):
        """Test successful contingency table creation."""
        metadata_file = tmp_path / "metadata.txt"
        output_file = tmp_path / "contingency.txt"
        
        # Create test metadata
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'E. coli', 'S. aureus', 'B. subtilis', 'E. coli'],
            'location': ['USA', 'UK', 'USA', 'Germany', 'USA'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        result = count_metadata_by_category(
            metadata_file, 'organism', 'location', output_file, min_count=1
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'USA' in result.columns
        assert 'UK' in result.columns
        assert 'E. coli' in result.index
        assert result.loc['E. coli', 'USA'] == 2
        assert output_file.exists()

    def test_count_metadata_by_category_min_count_filter(self, tmp_path):
        """Test contingency table with minimum count filtering."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'B. subtilis'],
            'location': ['USA', 'UK', 'Germany'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        result = count_metadata_by_category(
            metadata_file, 'organism', 'location', min_count=2
        )
        
        # Should filter out all rows since each organism appears only once
        assert len(result) == 0

    def test_count_metadata_by_category_invalid_columns(self, tmp_path):
        """Test with invalid column names."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus'],
            'location': ['USA', 'UK'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        with pytest.raises(ProcessingError, match="Category column 'invalid' not found"):
            count_metadata_by_category(metadata_file, 'invalid', 'location')


class TestSummarizeMetadataColumn:
    """Test summarize_metadata_column function."""

    def test_summarize_metadata_column_success(self, tmp_path):
        """Test successful column value counting."""
        metadata_file = tmp_path / "metadata.txt"
        output_file = tmp_path / "counts.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'E. coli', 'S. aureus', 'B. subtilis', 'E. coli'],
            'location': ['USA', 'UK', 'USA', 'Germany', 'USA'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        summarize_metadata_column(metadata_file, 'organism', output_file)
        
        assert output_file.exists()
        
        # Check output file content
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'E. coli' in content
            assert 'S. aureus' in content

    def test_summarize_metadata_column_invalid_column(self, tmp_path):
        """Test with invalid column name."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        with pytest.raises(ProcessingError, match="Column 'invalid' not found"):
            summarize_metadata_column(metadata_file, 'invalid')

    @patch('metaquest.processing.counts.logger')
    def test_summarize_metadata_column_logging(self, mock_logger, tmp_path):
        """Test that appropriate logging occurs."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'E. coli', 'S. aureus'],
            'location': ['USA', 'UK', 'USA'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        summarize_metadata_column(metadata_file, 'organism')
        
        # Check that summary information was logged
        mock_logger.info.assert_any_call("Column: organism")
        mock_logger.info.assert_any_call("Unique values: 2")

    def test_summarize_metadata_column_return_value(self, tmp_path):
        """Test that function returns correct value counts dictionary."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'E. coli', 'S. aureus', 'B. subtilis', 'E. coli'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        result = summarize_metadata_column(metadata_file, 'organism')
        
        assert isinstance(result, dict)
        assert result['E. coli'] == 3
        assert result['S. aureus'] == 1
        assert result['B. subtilis'] == 1

    def test_summarize_metadata_column_with_nulls(self, tmp_path):
        """Test summarization with null values."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', None, 'S. aureus', 'E. coli', None],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        result = summarize_metadata_column(metadata_file, 'organism')
        
        # Should only count non-null values
        assert result['E. coli'] == 2
        assert result['S. aureus'] == 1
        assert len(result) == 2  # No null key


class TestCountMetadataIntegration:
    """Integration tests for count_metadata with more complex scenarios."""

    def test_count_metadata_multiple_genomes_success(self, tmp_path):
        """Test counting with multiple genomes and complex metadata."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        output_file = tmp_path / "counts.tsv"
        
        # Create complex summary data
        summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.2, 0.6, 0.05, 0.9, 0.7, 0.1, 0.8],
            'GCF_000002.1': [0.1, 0.7, 0.3, 0.85, 0.2, 0.6, 0.9, 0.4],
            'GCA_000003.1': [0.6, 0.8, 0.9, 0.1, 0.7, 0.2, 0.5, 0.8],
            'max_containment': [0.8, 0.8, 0.9, 0.85, 0.9, 0.7, 0.9, 0.8]
        }, index=['SRR001', 'SRR002', 'SRR003', 'SRR004', 'SRR005', 'SRR006', 'SRR007', 'SRR008'])
        
        # Create complex metadata
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'E. coli', 'B. subtilis', 'E. coli', 
                        'S. aureus', 'P. aeruginosa', 'E. coli'],
            'country': ['USA', 'UK', 'Canada', 'Germany', 'USA', 'France', 'Japan', 'Australia'],
            'source': ['clinical', 'environmental', 'clinical', 'food', 'clinical', 
                      'environmental', 'clinical', 'environmental']
        }, index=['SRR001', 'SRR002', 'SRR003', 'SRR004', 'SRR005', 'SRR006', 'SRR007', 'SRR008'])
        
        summary_data.to_csv(summary_file, sep='\t')
        metadata_data.to_csv(metadata_file, sep='\t')
        
        result = count_metadata(
            summary_file, metadata_file, 'organism', 0.5, output_file
        )
        
        # Verify result structure and content
        assert not result.empty
        assert 'GCF_000001.1' in result.columns
        assert 'GCF_000002.1' in result.columns
        assert 'GCA_000003.1' in result.columns
        
        # Verify that output and stats files were created
        assert output_file.exists()
        stats_file = output_file.with_name(output_file.stem + '_stats' + output_file.suffix)
        assert stats_file.exists()

    @patch('metaquest.processing.counts.write_csv')
    def test_count_metadata_stat_file_generation(self, mock_write_csv, tmp_path):
        """Test automatic generation of statistics file."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        output_file = tmp_path / "counts.tsv"
        
        # Simple test data
        summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.6, 0.9]
        }, index=['SRR001', 'SRR002', 'SRR003'])
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'E. coli']
        }, index=['SRR001', 'SRR002', 'SRR003'])
        
        summary_data.to_csv(summary_file, sep='\t')
        metadata_data.to_csv(metadata_file, sep='\t')
        
        count_metadata(summary_file, metadata_file, 'organism', 0.5, output_file)
        
        # Should call write_csv twice: once for counts, once for stats
        assert mock_write_csv.call_count == 2
        
        # Check that stats file path was auto-generated
        call_args = mock_write_csv.call_args_list
        stats_call = call_args[1]  # Second call should be for stats
        stats_path = str(stats_call[0][1])  # Second argument is the file path
        assert 'counts_stats.tsv' in stats_path

    def test_count_metadata_edge_case_all_low_values(self, tmp_path):
        """Test when all containment values are below threshold."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        output_file = tmp_path / "counts.tsv"
        
        # All containment values below threshold
        summary_data = pd.DataFrame({
            'GCF_000001.1': [0.1, 0.2, 0.3],
            'GCF_000002.1': [0.05, 0.15, 0.25]
        }, index=['SRR001', 'SRR002', 'SRR003'])
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'B. subtilis']
        }, index=['SRR001', 'SRR002', 'SRR003'])
        
        summary_data.to_csv(summary_file, sep='\t')
        metadata_data.to_csv(metadata_file, sep='\t')
        
        result = count_metadata(
            summary_file, metadata_file, 'organism', 0.5, output_file
        )
        
        # Should return empty DataFrame
        assert result.empty

    def test_count_metadata_partial_metadata_coverage(self, tmp_path):
        """Test when only some samples have metadata."""
        summary_file = tmp_path / "summary.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        output_file = tmp_path / "counts.tsv"
        
        # Summary has 5 samples
        summary_data = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.2, 0.6, 0.9, 0.7]
        }, index=['SRR001', 'SRR002', 'SRR003', 'SRR004', 'SRR005'])
        
        # Metadata only has 3 samples
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'E. coli']
        }, index=['SRR001', 'SRR003', 'SRR005'])  # Missing SRR002, SRR004
        
        summary_data.to_csv(summary_file, sep='\t')
        metadata_data.to_csv(metadata_file, sep='\t')
        
        result = count_metadata(
            summary_file, metadata_file, 'organism', 0.5, output_file
        )
        
        # Should still work, just with fewer samples
        assert not result.empty
        assert 'GCF_000001.1' in result.columns
        # Should count E. coli twice (SRR001=0.8, SRR005=0.7 both > 0.5 and have metadata)
        assert result['GCF_000001.1']['E. coli'] == 2


class TestCountMetadataByCategoryExtended:
    """Extended tests for count_metadata_by_category function."""

    def test_count_metadata_by_category_no_output_file(self, tmp_path):
        """Test function works without specifying output file."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus', 'E. coli'],
            'location': ['USA', 'UK', 'USA'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        result = count_metadata_by_category(
            metadata_file, 'organism', 'location'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_count_metadata_by_category_large_dataset(self, tmp_path):
        """Test with larger dataset to verify sorting and filtering."""
        metadata_file = tmp_path / "metadata.txt"
        
        # Create larger dataset
        organisms = ['E. coli'] * 20 + ['S. aureus'] * 15 + ['B. subtilis'] * 10 + ['P. aeruginosa'] * 5
        locations = ['USA'] * 25 + ['UK'] * 15 + ['Germany'] * 10
        
        metadata_data = pd.DataFrame({
            'organism': organisms,
            'location': locations,
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        result = count_metadata_by_category(
            metadata_file, 'organism', 'location', min_count=8
        )
        
        # Should filter out organisms with < 8 total occurrences
        # E. coli (20), S. aureus (15), B. subtilis (10) should remain
        # P. aeruginosa (5) should be filtered out
        assert 'E. coli' in result.index
        assert 'S. aureus' in result.index  
        assert 'B. subtilis' in result.index
        assert len(result) == 3  # P. aeruginosa should be filtered out

    def test_count_metadata_by_category_string_path(self, tmp_path):
        """Test that function accepts string paths."""
        metadata_file = tmp_path / "metadata.txt"
        
        metadata_data = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus'],
            'location': ['USA', 'UK'],
        })
        metadata_data.to_csv(metadata_file, sep='\t', index=False)
        
        result = count_metadata_by_category(
            str(metadata_file), 'organism', 'location'
        )
        
        assert isinstance(result, pd.DataFrame)

    def test_count_metadata_by_category_file_error(self, tmp_path):
        """Test error handling for file operations."""
        nonexistent_file = tmp_path / "missing.txt"
        
        with pytest.raises(ProcessingError, match="Error creating contingency table"):
            count_metadata_by_category(nonexistent_file, 'organism', 'location')


class TestProcessGenomeAccessionsExtended:
    """Extended tests for _process_genome_accessions function."""

    @patch('metaquest.processing.counts.logger')
    def test_process_genome_accessions_exception_handling(self, mock_logger):
        """Test exception handling in _process_genome_accessions."""
        # Create invalid data that will cause an exception
        summary_df = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9]
        }, index=['SRR1', 'SRR2'])
        
        # Create metadata with wrong type to trigger exception
        metadata_df = "invalid_metadata"  # Not a DataFrame
        
        df_list = []
        
        result = _process_genome_accessions(
            'GCF_000001.1', summary_df, 0.5, metadata_df, 'organism', df_list
        )
        
        # Should return 0 and log error
        assert result == 0
        mock_logger.error.assert_called()

    def test_process_genome_accessions_empty_metadata_after_filter(self):
        """Test when filtered metadata is empty due to no matching indices."""
        summary_df = pd.DataFrame({
            'GCF_000001.1': [0.8, 0.9]
        }, index=['SRR1', 'SRR2'])
        
        metadata_df = pd.DataFrame({
            'organism': ['E. coli', 'S. aureus']
        }, index=['SRR3', 'SRR4'])  # No matching indices
        
        df_list = []
        
        result = _process_genome_accessions(
            'GCF_000001.1', summary_df, 0.5, metadata_df, 'organism', df_list
        )
        
        assert result == 0
        assert len(df_list) == 0