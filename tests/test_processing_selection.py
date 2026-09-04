"""Tests for accession selection from containment and metadata tables."""

import pytest

from metaquest.core.exceptions import ProcessingError
from metaquest.processing.selection import select_accessions


@pytest.fixture
def tables(tmp_path):
    cont = tmp_path / "parsed_containment.txt"
    cont.write_text(
        "\tGCF_A\tGCF_B\tmax_containment\tmax_containment_annotation\n"
        "SRR1\t0.95\t0.10\t0.95\tGCF_A\n"
        "SRR2\t0.20\t0.80\t0.80\tGCF_B\n"
        "SRR3\t0.05\t0.02\t0.05\tGCF_A\n"
    )
    meta = tmp_path / "branchwater_metadata.txt"
    meta.write_text(
        "Run_ID\tSample_Scientific_Name\tgeo_loc_name_country_calc\n"
        "SRR1\tWolbachia pipientis\tAustria\n"
        "SRR2\tmosquito metagenome\tFrance\n"
        "SRR3\tmosquito metagenome\tFrance\n"
    )
    return cont, meta


def test_default_uses_max_containment(tables):
    cont, _ = tables
    assert select_accessions(cont, threshold=0.5) == ["SRR1", "SRR2"]


def test_genome_column_and_threshold(tables):
    cont, _ = tables
    assert select_accessions(cont, genome_id="GCF_B", threshold=0.5) == ["SRR2"]


def test_metadata_filter_is_case_insensitive(tables):
    cont, meta = tables
    result = select_accessions(
        cont, threshold=0.0, metadata_file=meta, metadata_column="geo_loc_name_country_calc", metadata_value="france"
    )
    assert result == ["SRR2", "SRR3"]


def test_unknown_genome_lists_columns(tables):
    cont, _ = tables
    with pytest.raises(ProcessingError, match="GCF_A"):
        select_accessions(cont, genome_id="GCF_Z")


def test_metadata_column_without_value_raises(tables):
    cont, meta = tables
    with pytest.raises(ProcessingError, match="metadata_value"):
        select_accessions(cont, metadata_file=meta, metadata_column="Sample_Scientific_Name")
