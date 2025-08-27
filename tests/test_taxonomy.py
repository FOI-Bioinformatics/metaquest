"""
Tests for taxonomy integration functions.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from metaquest.data.taxonomy import (
    NCBITaxonomyClient,
    validate_taxonomic_assignments,
    create_taxonomic_summary,
    analyze_taxonomic_composition
)


class TestNCBITaxonomyClient:
    """Test NCBI taxonomy client functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = NCBITaxonomyClient("test@example.com")
        
    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.email == "test@example.com"
        assert self.client.api_key is None
        assert self.client.request_delay > 0
        
    def test_client_with_api_key(self):
        """Test client with API key."""
        client = NCBITaxonomyClient("test@example.com", "api_key_123")
        assert client.api_key == "api_key_123"
        assert client.request_delay < 0.5  # Should be faster with API key
        
    def test_clean_species_name(self):
        """Test species name cleaning."""
        # Test various cleaning operations
        assert self.client._clean_species_name("candidate Bacteria") == "Bacteria"
        assert self.client._clean_species_name("uncultured bacterium") == "bacterium"
        assert self.client._clean_species_name("Escherichia coli strain K12") == "Escherichia coli"
        assert self.client._clean_species_name("Bacteria sp. ABC123") == "Bacteria"
        assert self.client._clean_species_name("Species [with brackets]") == "Species"
        assert self.client._clean_species_name("Species (with parentheses)") == "Species"
        
    @patch('requests.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.text = "<xml>test</xml>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client._make_request("http://test.com", {"param": "value"})
        
        assert result == "<xml>test</xml>"
        mock_get.assert_called_once()
        
    @patch('requests.get')
    def test_make_request_failure(self, mock_get):
        """Test failed API request."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            self.client._make_request("http://test.com", {"param": "value"})
            
    @patch.object(NCBITaxonomyClient, '_make_request')
    def test_search_taxonomy(self, mock_request):
        """Test taxonomy search."""
        # Mock search response
        search_xml = """
        <eSearchResult>
            <IdList>
                <Id>12345</Id>
                <Id>67890</Id>
            </IdList>
        </eSearchResult>
        """
        
        # Mock fetch response
        fetch_xml = """
        <TaxaSet>
            <Taxon>
                <TaxId>12345</TaxId>
                <ScientificName>Escherichia coli</ScientificName>
                <Rank>species</Rank>
                <LineageEx>
                    <Taxon>
                        <ScientificName>Bacteria</ScientificName>
                        <Rank>superkingdom</Rank>
                    </Taxon>
                </LineageEx>
            </Taxon>
        </TaxaSet>
        """
        
        mock_request.side_effect = [search_xml, fetch_xml]
        
        results = self.client.search_taxonomy("Escherichia coli")
        
        assert len(results) == 1
        assert results[0]['tax_id'] == '12345'
        assert results[0]['scientific_name'] == 'Escherichia coli'
        assert results[0]['rank'] == 'species'
        
    def test_validate_species_name_mock(self):
        """Test species name validation with mocked data."""
        with patch.object(self.client, 'search_taxonomy') as mock_search:
            mock_search.return_value = [{
                'tax_id': '562',
                'scientific_name': 'Escherichia coli',
                'rank': 'species',
                'lineage': 'superkingdom:Bacteria;phylum:Proteobacteria'
            }]
            
            result = self.client.validate_species_name("E. coli")
            
            assert result['is_valid'] == True
            assert result['tax_id'] == '562'
            assert result['validated_name'] == 'Escherichia coli'
            assert 'lineage' in result
            
    def test_validate_species_name_not_found(self):
        """Test species name validation when not found."""
        with patch.object(self.client, 'search_taxonomy') as mock_search:
            mock_search.return_value = []
            
            result = self.client.validate_species_name("Nonexistent species")
            
            assert result['is_valid'] == False
            assert result['validated_name'] == ''
            assert result['tax_id'] == ''


class TestTaxonomyValidation:
    """Test taxonomy validation functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.species_list = [
            "Escherichia coli",
            "Bacillus subtilis",
            "Invalid species name"
        ]
        
    @patch('metaquest.data.taxonomy.NCBITaxonomyClient')
    def test_validate_taxonomic_assignments(self, mock_client_class):
        """Test batch taxonomy validation."""
        # Mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock validation results
        mock_client.validate_species_name.side_effect = [
            {'original_name': 'Escherichia coli', 'is_valid': True, 'tax_id': '562'},
            {'original_name': 'Bacillus subtilis', 'is_valid': True, 'tax_id': '1423'},
            {'original_name': 'Invalid species name', 'is_valid': False, 'tax_id': ''}
        ]
        
        result = validate_taxonomic_assignments(
            self.species_list,
            email="test@example.com"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result['is_valid'].sum() == 2  # Two valid species
        
    def test_create_taxonomic_summary(self):
        """Test taxonomic summary creation."""
        # Create test data
        abundance_data = pd.DataFrame({
            'Species A': [0.5, 0.3, 0.2],
            'Species B': [0.2, 0.4, 0.1],
            'Species C': [0.3, 0.3, 0.7]
        }, index=['Sample1', 'Sample2', 'Sample3'])
        
        taxonomy_data = pd.DataFrame({
            'original_name': ['Species A', 'Species B', 'Species C'],
            'validated_name': ['Genus alpha sp1', 'Genus beta sp1', 'Genus alpha sp2'],
            'is_valid': [True, True, True],
            'lineage': [
                'genus:Genus alpha;family:Family X',
                'genus:Genus beta;family:Family Y',
                'genus:Genus alpha;family:Family X'
            ]
        })
        
        result = create_taxonomic_summary(
            abundance_data, taxonomy_data, level='genus'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3  # 3 samples
        assert 'Genus alpha' in result.columns
        assert 'Genus beta' in result.columns
        
        # Check that Species A and C are grouped under Genus alpha
        assert result.loc['Sample1', 'Genus alpha'] == 0.8  # 0.5 + 0.3
        
    def test_analyze_taxonomic_composition(self):
        """Test multi-level taxonomic composition analysis."""
        # Create test data
        abundance_data = pd.DataFrame({
            'Species A': [1.0, 0.5],
            'Species B': [0.5, 1.0]
        }, index=['Sample1', 'Sample2'])
        
        taxonomy_data = pd.DataFrame({
            'original_name': ['Species A', 'Species B'],
            'validated_name': ['Species A', 'Species B'],
            'is_valid': [True, True],
            'lineage': [
                'phylum:Phylum1;genus:Genus1',
                'phylum:Phylum2;genus:Genus2'
            ]
        })
        
        result = analyze_taxonomic_composition(
            abundance_data, taxonomy_data, levels=['phylum', 'genus']
        )
        
        assert isinstance(result, dict)
        assert 'phylum' in result
        assert 'genus' in result
        assert isinstance(result['phylum'], pd.DataFrame)
        assert isinstance(result['genus'], pd.DataFrame)


class TestTaxonomyEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_species_list(self):
        """Test with empty species list."""
        with patch('metaquest.data.taxonomy.NCBITaxonomyClient'):
            result = validate_taxonomic_assignments(
                [], email="test@example.com"
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            
    def test_create_summary_invalid_taxonomy(self):
        """Test taxonomic summary with invalid taxonomy data."""
        abundance_data = pd.DataFrame({
            'Species A': [1.0],
        }, index=['Sample1'])
        
        taxonomy_data = pd.DataFrame({
            'original_name': ['Species A'],
            'validated_name': ['Species A'],
            'is_valid': [False],  # Invalid taxonomy
            'lineage': ['']
        })
        
        result = create_taxonomic_summary(
            abundance_data, taxonomy_data, level='genus'
        )
        
        # Should handle invalid taxonomy gracefully
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 1  # 1 sample


if __name__ == "__main__":
    pytest.main([__file__])