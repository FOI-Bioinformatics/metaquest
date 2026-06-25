"""Tests for metaquest/data/genome_taxonomy.py.

Covers GTDB taxonomy parsing, API-based enrichment, cache round-trips,
long-format annotation, taxonomy filtering, and aggregation.

Run: pytest tests/test_genome_taxonomy.py -v
"""

import pytest
import pandas as pd
from unittest.mock import patch

from metaquest.core.models import TaxonomyInfo
from metaquest.core.exceptions import DataAccessError
from metaquest.data.genome_taxonomy import (
    parse_gtdb_taxonomy_string,
    enrich_genomes_with_taxonomy,
    load_taxonomy_cache,
    save_taxonomy_cache,
    annotate_containment_with_taxonomy,
    filter_by_taxonomy,
    summarize_by_taxonomy,
)

# ============================================================================
# parse_gtdb_taxonomy_string
# ============================================================================


class TestParseGtdbTaxonomyString:

    def test_valid_full_string(self):
        s = "d__Bacteria;p__Firmicutes;c__Bacilli;" "o__Bacillales;f__Bacillaceae;g__Bacillus;s__Bacillus subtilis"
        info = parse_gtdb_taxonomy_string(s, "GCF_000009045.1")
        assert info.genome_id == "GCF_000009045.1"
        assert info.phylum == "Firmicutes"
        assert info.class_name == "Bacilli"
        assert info.order == "Bacillales"
        assert info.family == "Bacillaceae"
        assert info.genus == "Bacillus"
        assert info.species == "Bacillus subtilis"

    def test_empty_string(self):
        info = parse_gtdb_taxonomy_string("", "GCF_000")
        assert info.genome_id == "GCF_000"
        assert info.species is None
        assert info.family is None

    def test_partial_string(self):
        info = parse_gtdb_taxonomy_string("d__Bacteria;p__Proteobacteria", "GCF_001")
        assert info.phylum == "Proteobacteria"
        assert info.family is None

    def test_malformed_tokens_ignored(self):
        info = parse_gtdb_taxonomy_string("bad_token;f__Fam1;also_bad", "g1")
        assert info.family == "Fam1"
        assert info.genus is None

    def test_empty_values_not_set(self):
        info = parse_gtdb_taxonomy_string("f__;g__Genus1", "g2")
        assert info.family is None
        assert info.genus == "Genus1"

    def test_whitespace_tokens(self):
        info = parse_gtdb_taxonomy_string(" f__Fam1 ; g__Gen1 ", "g3")
        assert info.family == "Fam1"
        assert info.genus == "Gen1"


# ============================================================================
# enrich_genomes_with_taxonomy (mocked API)
# ============================================================================


class TestEnrichGenomesWithTaxonomy:

    @patch("metaquest.data.genome_taxonomy._lookup_genome_taxonomy_gtdb")
    def test_basic_enrichment(self, mock_lookup):
        mock_lookup.return_value = TaxonomyInfo(genome_id="GCF_001", family="Bacillaceae", genus="Bacillus")
        result = enrich_genomes_with_taxonomy(["GCF_001"])
        assert "GCF_001" in result
        assert result["GCF_001"].family == "Bacillaceae"
        mock_lookup.assert_called_once_with("GCF_001")

    @patch("metaquest.data.genome_taxonomy._lookup_genome_taxonomy_gtdb")
    def test_uses_cache(self, mock_lookup, tmp_path):
        cache_file = tmp_path / "cache.tsv"
        tax = {"GCF_A": TaxonomyInfo(genome_id="GCF_A", family="FamA")}
        save_taxonomy_cache(tax, cache_file)

        result = enrich_genomes_with_taxonomy(["GCF_A"], cache_file=cache_file)
        assert result["GCF_A"].family == "FamA"
        mock_lookup.assert_not_called()

    @patch("metaquest.data.genome_taxonomy._lookup_genome_taxonomy_gtdb")
    def test_api_failure_returns_empty_info(self, mock_lookup):
        mock_lookup.side_effect = DataAccessError("API down")
        result = enrich_genomes_with_taxonomy(["GCF_FAIL"])
        assert "GCF_FAIL" in result
        assert result["GCF_FAIL"].family is None

    @patch("metaquest.data.genome_taxonomy._lookup_genome_taxonomy_gtdb")
    def test_none_result_gives_empty_info(self, mock_lookup):
        mock_lookup.return_value = None
        result = enrich_genomes_with_taxonomy(["GCF_NONE"])
        assert result["GCF_NONE"].family is None


# ============================================================================
# load_taxonomy_cache / save_taxonomy_cache
# ============================================================================


class TestTaxonomyCache:

    def test_roundtrip(self, tmp_path):
        cache_file = tmp_path / "tax.tsv"
        original = {
            "GCF_001": TaxonomyInfo(
                genome_id="GCF_001",
                species="Sp1",
                genus="Gen1",
                family="Fam1",
                order="Ord1",
                class_name="Cls1",
                phylum="Phy1",
                organism="Org1",
                tax_id="12345",
            ),
            "GCF_002": TaxonomyInfo(genome_id="GCF_002", family="Fam2"),
        }
        save_taxonomy_cache(original, cache_file)
        loaded = load_taxonomy_cache(cache_file)

        assert set(loaded.keys()) == {"GCF_001", "GCF_002"}
        assert loaded["GCF_001"].species == "Sp1"
        assert loaded["GCF_001"].tax_id == "12345"
        assert loaded["GCF_002"].family == "Fam2"
        assert loaded["GCF_002"].species is None

    def test_load_nonexistent_file(self, tmp_path):
        result = load_taxonomy_cache(tmp_path / "no_such_file.tsv")
        assert result == {}


# ============================================================================
# annotate_containment_with_taxonomy
# ============================================================================


class TestAnnotateContainment:

    def _make_df(self):
        return pd.DataFrame(
            {"genA": [0.5, 0.1], "genB": [0.8, 0.0]},
            index=["S1", "S2"],
        )

    def test_long_format_conversion(self):
        tax = {
            "genA": TaxonomyInfo(genome_id="genA", family="F1", genus="G1", species="Sp1"),
            "genB": TaxonomyInfo(genome_id="genB", family="F2", genus="G2", species="Sp2"),
        }
        result = annotate_containment_with_taxonomy(self._make_df(), tax)
        assert set(result.columns) == {"sample", "genome", "containment", "species", "genus", "family"}
        assert len(result) == 4  # 2 samples x 2 genomes

    def test_missing_taxonomy_gives_none(self):
        result = annotate_containment_with_taxonomy(self._make_df(), {})
        assert result.iloc[0]["family"] is None


# ============================================================================
# filter_by_taxonomy
# ============================================================================


class TestFilterByTaxonomy:

    @pytest.fixture()
    def annotated(self):
        return pd.DataFrame(
            {
                "sample": ["S1", "S1", "S2"],
                "genome": ["gA", "gB", "gA"],
                "containment": [0.5, 0.8, 0.1],
                "species": ["Sp1", "Sp2", "Sp1"],
                "genus": ["Gen1", "Gen2", "Gen1"],
                "family": ["Fam1", "Fam2", "Fam1"],
            }
        )

    def test_filter_by_family(self, annotated):
        result = filter_by_taxonomy(annotated, family="Fam1")
        assert len(result) == 2
        assert set(result["family"]) == {"Fam1"}

    def test_filter_by_genus(self, annotated):
        result = filter_by_taxonomy(annotated, genus="Gen2")
        assert len(result) == 1

    def test_filter_by_species(self, annotated):
        result = filter_by_taxonomy(annotated, species="sp1")
        assert len(result) == 2  # case-insensitive

    def test_filter_by_min_containment(self, annotated):
        result = filter_by_taxonomy(annotated, min_containment=0.5)
        assert len(result) == 2

    def test_combined_filters(self, annotated):
        result = filter_by_taxonomy(annotated, family="Fam1", min_containment=0.3)
        assert len(result) == 1
        assert result.iloc[0]["sample"] == "S1"


# ============================================================================
# summarize_by_taxonomy
# ============================================================================


class TestSummarizeByTaxonomy:

    @pytest.fixture()
    def annotated(self):
        return pd.DataFrame(
            {
                "sample": ["S1", "S1", "S2", "S2"],
                "genome": ["gA", "gB", "gA", "gB"],
                "containment": [0.5, 0.8, 0.1, 0.3],
                "family": ["Fam1", "Fam2", "Fam1", "Fam2"],
                "genus": ["G1", "G2", "G1", "G2"],
                "species": ["Sp1", "Sp2", "Sp1", "Sp2"],
            }
        )

    def test_summarize_family(self, annotated):
        result = summarize_by_taxonomy(annotated, level="family")
        assert "Fam1" in result.columns
        assert "Fam2" in result.columns
        assert result.loc["S1", "Fam1"] == 0.5

    def test_summarize_genus(self, annotated):
        result = summarize_by_taxonomy(annotated, level="genus")
        assert "G1" in result.columns

    def test_invalid_level_raises(self, annotated):
        with pytest.raises(ValueError, match="level must be"):
            summarize_by_taxonomy(annotated, level="kingdom")

    def test_empty_after_dropna(self):
        df = pd.DataFrame(
            {
                "sample": ["S1"],
                "genome": ["g1"],
                "containment": [0.5],
                "family": [None],
                "genus": [None],
                "species": [None],
            }
        )
        result = summarize_by_taxonomy(df, level="family")
        assert result.empty
