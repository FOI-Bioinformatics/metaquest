"""Tests for the GTDB API client module."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from metaquest.core.exceptions import DataAccessError
from metaquest.data.gtdb import (
    get_accessions_for_genus,
    get_accessions_for_species,
    search_species,
    search_taxon,
)

# --- Realistic mock data ---

SPECIES_RESULT_LIST = [
    {
        "accession": "GCF_000005845.2",
        "species": "s__Escherichia coli",
        "isRep": True,
        "name": "Escherichia coli K-12",
    },
    {
        "accession": "GCF_000008865.2",
        "species": "s__Escherichia coli",
        "isRep": False,
        "name": "Escherichia coli O157:H7",
    },
]

SPECIES_RESULT_DICT = {
    "genomes": [
        {
            "gid": "GCF_000005845.2",
            "species": "s__Escherichia coli",
            "is_representative": True,
        },
    ]
}

TAXON_RESULT_LIST = [
    {
        "accession": "GCF_000005845.2",
        "species": "s__Escherichia coli",
        "isRep": True,
    },
    {
        "accession": "GCF_000240185.1",
        "species": "s__Escherichia fergusonii",
        "isRep": True,
    },
    {
        "accession": "GCF_001234567.1",
        "species": "s__Escherichia fergusonii",
        "isRep": False,
    },
]


# --- search_species ---


class TestSearchSpecies:
    @patch("metaquest.data.gtdb.requests.get")
    def test_returns_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = SPECIES_RESULT_LIST
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_species("Escherichia coli")

        assert len(result) == 2
        assert result[0]["accession"] == "GCF_000005845.2"

    @patch("metaquest.data.gtdb.requests.get")
    def test_returns_dict_with_genomes(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = SPECIES_RESULT_DICT
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_species("Escherichia coli")

        assert len(result) == 1
        assert result[0]["gid"] == "GCF_000005845.2"

    @patch("metaquest.data.gtdb.requests.get")
    def test_returns_dict_without_genomes_key(self, mock_get):
        """A dict without 'genomes' wraps itself in a list."""
        single_record = {"accession": "GCF_000005845.2", "species": "E. coli"}
        mock_response = MagicMock()
        mock_response.json.return_value = single_record
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_species("Escherichia coli")

        assert len(result) == 1
        assert result[0]["accession"] == "GCF_000005845.2"

    @patch("metaquest.data.gtdb.requests.get")
    def test_empty_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_species("Nonexistent species")

        assert result == []

    @patch("metaquest.data.gtdb.requests.get")
    def test_none_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = None
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_species("Nonexistent species")

        assert result == []

    @patch("metaquest.data.gtdb.requests.get")
    def test_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(DataAccessError, match="GTDB API error"):
            search_species("Escherichia coli")

    @patch("metaquest.data.gtdb.requests.get")
    def test_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        with pytest.raises(DataAccessError, match="GTDB API error"):
            search_species("Escherichia coli")

    @patch("metaquest.data.gtdb.requests.get")
    def test_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        with pytest.raises(DataAccessError, match="GTDB API error"):
            search_species("Escherichia coli")

    @patch("metaquest.data.gtdb.requests.get")
    def test_unexpected_data_type(self, mock_get):
        """Non-list, non-dict response returns empty list."""
        mock_response = MagicMock()
        mock_response.json.return_value = "unexpected string"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_species("Escherichia coli")

        assert result == []


# --- search_taxon ---


class TestSearchTaxon:
    @patch("metaquest.data.gtdb.requests.get")
    def test_returns_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = TAXON_RESULT_LIST
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_taxon("Escherichia")

        assert len(result) == 3

    @patch("metaquest.data.gtdb.requests.get")
    def test_with_limit(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = TAXON_RESULT_LIST[:1]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_taxon("Escherichia", limit=1)

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["limit"] == 1
        assert len(result) == 1

    @patch("metaquest.data.gtdb.requests.get")
    def test_returns_dict_with_results_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": TAXON_RESULT_LIST}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_taxon("Escherichia")

        assert len(result) == 3

    @patch("metaquest.data.gtdb.requests.get")
    def test_returns_dict_without_results_key(self, mock_get):
        single_record = {"accession": "GCF_000005845.2", "name": "Escherichia"}
        mock_response = MagicMock()
        mock_response.json.return_value = single_record
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_taxon("Escherichia")

        assert len(result) == 1

    @patch("metaquest.data.gtdb.requests.get")
    def test_empty_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_taxon("Nonexistent")

        assert result == []

    @patch("metaquest.data.gtdb.requests.get")
    def test_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed")

        with pytest.raises(DataAccessError, match="GTDB API error"):
            search_taxon("Escherichia")

    @patch("metaquest.data.gtdb.requests.get")
    def test_unexpected_data_type(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = 42
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = search_taxon("Escherichia")

        assert result == []


# --- get_accessions_for_species ---


class TestGetAccessionsForSpecies:
    @patch("metaquest.data.gtdb.search_species")
    def test_representative_only(self, mock_search):
        mock_search.return_value = SPECIES_RESULT_LIST

        result = get_accessions_for_species("Escherichia coli", representative_only=True)

        assert result == ["GCF_000005845.2"]

    @patch("metaquest.data.gtdb.search_species")
    def test_all_genomes(self, mock_search):
        mock_search.return_value = SPECIES_RESULT_LIST

        result = get_accessions_for_species("Escherichia coli", representative_only=False)

        assert "GCF_000005845.2" in result
        assert "GCF_000008865.2" in result
        assert len(result) == 2

    @patch("metaquest.data.gtdb.search_species")
    def test_no_representative_falls_back_to_first(self, mock_search):
        """When representative_only but none flagged, returns the first accession."""
        mock_search.return_value = [
            {"accession": "GCF_111111111.1", "isRep": False},
            {"accession": "GCF_222222222.1", "isRep": False},
        ]

        result = get_accessions_for_species("Some species", representative_only=True)

        assert result == ["GCF_111111111.1"]

    @patch("metaquest.data.gtdb.search_species")
    def test_empty_results(self, mock_search):
        mock_search.return_value = []

        result = get_accessions_for_species("Nonexistent species")

        assert result == []

    @patch("metaquest.data.gtdb.search_species")
    def test_alternative_accession_keys(self, mock_search):
        """Tests gid and ncbi_accession fallback keys."""
        mock_search.return_value = [
            {"gid": "GCF_000005845.2", "isRep": True},
        ]

        result = get_accessions_for_species("E. coli", representative_only=True)

        assert result == ["GCF_000005845.2"]

    @patch("metaquest.data.gtdb.search_species")
    def test_is_representative_key(self, mock_search):
        """Tests the is_representative key variant."""
        mock_search.return_value = [
            {"accession": "GCF_000005845.2", "is_representative": True},
        ]

        result = get_accessions_for_species("E. coli", representative_only=True)

        assert result == ["GCF_000005845.2"]

    @patch("metaquest.data.gtdb.search_species")
    def test_records_without_accession(self, mock_search):
        mock_search.return_value = [
            {"species": "E. coli", "isRep": True},
        ]

        result = get_accessions_for_species("E. coli", representative_only=False)

        assert result == []


# --- get_accessions_for_genus ---


class TestGetAccessionsForGenus:
    @patch("metaquest.data.gtdb.search_taxon")
    def test_representative_only(self, mock_search):
        mock_search.return_value = TAXON_RESULT_LIST

        result = get_accessions_for_genus("Escherichia", representative_only=True)

        # Should get one per species (only representatives)
        assert "GCF_000005845.2" in result
        assert "GCF_000240185.1" in result
        assert len(result) == 2

    @patch("metaquest.data.gtdb.search_taxon")
    def test_deduplicates_species(self, mock_search):
        """Only one accession per unique species name."""
        mock_search.return_value = TAXON_RESULT_LIST

        result = get_accessions_for_genus("Escherichia", representative_only=True)

        # E. fergusonii appears twice but only the representative should be included
        assert result.count("GCF_000240185.1") == 1

    @patch("metaquest.data.gtdb.search_taxon")
    def test_empty_results(self, mock_search):
        mock_search.return_value = []

        result = get_accessions_for_genus("Nonexistent")

        assert result == []

    @patch("metaquest.data.gtdb.search_taxon")
    def test_no_representatives_falls_back(self, mock_search):
        """When no records are flagged as representative, collects all accessions."""
        mock_search.return_value = [
            {"accession": "GCF_111111111.1", "species": "sp1", "isRep": False},
            {"accession": "GCF_222222222.1", "species": "sp2", "isRep": False},
        ]

        result = get_accessions_for_genus("SomeGenus", representative_only=True)

        assert "GCF_111111111.1" in result
        assert "GCF_222222222.1" in result

    @patch("metaquest.data.gtdb.search_taxon")
    def test_all_genomes(self, mock_search):
        """With representative_only=False, no filtering happens."""
        mock_search.return_value = [
            {"accession": "GCF_000005845.2", "species": "sp1", "isRep": True},
            {"accession": "GCF_000008865.2", "species": "sp1", "isRep": False},
        ]

        result = get_accessions_for_genus("Escherichia", representative_only=False)

        # First record included; second has same species so skipped by dedup
        assert result == ["GCF_000005845.2"]

    @patch("metaquest.data.gtdb.search_taxon")
    def test_uses_name_key_for_species(self, mock_search):
        """Falls back to 'name' key when 'species' is missing."""
        mock_search.return_value = [
            {"accession": "GCF_000005845.2", "name": "sp1", "isRep": True},
            {"accession": "GCF_000240185.1", "name": "sp2", "isRep": True},
        ]

        result = get_accessions_for_genus("Escherichia", representative_only=True)

        assert len(result) == 2
