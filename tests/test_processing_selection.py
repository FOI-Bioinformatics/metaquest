"""Tests for accession selection from containment and metadata tables."""

import pytest

from metaquest.core.exceptions import ProcessingError
from metaquest.processing.selection import select_accessions, select_accessions_ranked


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


def test_metadata_join_ignores_whitespace_in_run_ids(tmp_path):
    cont = tmp_path / "parsed_containment.txt"
    cont.write_text("\tGCF_A\tmax_containment\n" "SRR1 \t0.95\t0.95\n" "SRR2\t0.80\t0.80\n")
    meta = tmp_path / "branchwater_metadata.txt"
    meta.write_text("Run_ID\tgeo_loc_name_country_calc\n" " SRR1\tFrance\n" "SRR2\tFrance\n")
    result = select_accessions(
        cont, threshold=0.0, metadata_file=meta, metadata_column="geo_loc_name_country_calc", metadata_value="france"
    )
    assert result == ["SRR1", "SRR2"]


def test_top_n_applies_after_threshold(tables):
    cont, _ = tables
    assert select_accessions(cont, threshold=0.0, top_n=1) == ["SRR1"]


def test_top_n_applies_after_exclusions(tables):
    cont, _ = tables
    # SRR1 ranks first; excluding it should let SRR2 take the single top_n slot,
    # not truncate before exclusion is applied.
    result = select_accessions(cont, threshold=0.0, top_n=1, exclude={"SRR1"})
    assert result == ["SRR2"]


def test_top_n_applies_after_metadata_filter(tables):
    cont, meta = tables
    result = select_accessions(
        cont,
        threshold=0.0,
        top_n=1,
        metadata_file=meta,
        metadata_column="geo_loc_name_country_calc",
        metadata_value="france",
    )
    assert result == ["SRR2"]


def test_exclude_removes_accessions(tables):
    cont, _ = tables
    assert select_accessions(cont, threshold=0.0, exclude={"SRR2"}) == ["SRR1", "SRR3"]


def test_genome_id_and_genome_ids_are_mutually_exclusive(tables):
    cont, _ = tables
    with pytest.raises(ProcessingError, match="genome_id"):
        select_accessions(cont, genome_id="GCF_A", genome_ids=["GCF_A", "GCF_B"])


def test_genome_ids_any_ranks_on_row_wise_max(tables):
    cont, _ = tables
    # SRR1: max(0.95, 0.10)=0.95; SRR2: max(0.20, 0.80)=0.80; SRR3: max(0.05, 0.02)=0.05
    result = select_accessions(cont, genome_ids=["GCF_A", "GCF_B"], threshold=0.5, require="any")
    assert result == ["SRR1", "SRR2"]


def test_genome_ids_all_requires_every_column_at_threshold(tables):
    cont, _ = tables
    # SRR1 min(0.95, 0.10)=0.10, SRR2 min(0.20, 0.80)=0.20, SRR3 min(0.05, 0.02)=0.02 (excluded);
    # both remaining accessions rank on their min, best first.
    result = select_accessions(cont, genome_ids=["GCF_A", "GCF_B"], threshold=0.1, require="all")
    assert result == ["SRR2", "SRR1"]


def test_genome_ids_all_excludes_rows_below_threshold_on_any_column(tables):
    cont, _ = tables
    result = select_accessions(cont, genome_ids=["GCF_A", "GCF_B"], threshold=0.5, require="all")
    assert result == []


def test_require_invalid_value_raises(tables):
    cont, _ = tables
    with pytest.raises(ProcessingError, match="require"):
        select_accessions(cont, genome_ids=["GCF_A", "GCF_B"], threshold=0.1, require="both")


def test_top_n_zero_raises(tables):
    cont, _ = tables
    with pytest.raises(ProcessingError, match="--top-n must be a positive integer"):
        select_accessions(cont, threshold=0.0, top_n=0)


def test_top_n_negative_raises(tables):
    cont, _ = tables
    with pytest.raises(ProcessingError, match="--top-n must be a positive integer"):
        select_accessions(cont, threshold=0.0, top_n=-1)


def test_select_accessions_ranked_returns_accession_column_value(tables):
    cont, _ = tables
    ranked = select_accessions_ranked(cont, threshold=0.5)
    assert ranked == [("SRR1", "max_containment", 0.95), ("SRR2", "max_containment", 0.80)]


def test_select_accessions_ranked_respects_top_n_and_exclude(tables):
    cont, _ = tables
    ranked = select_accessions_ranked(cont, threshold=0.0, top_n=1, exclude={"SRR1"})
    assert ranked == [("SRR2", "max_containment", 0.80)]


def test_select_accessions_wraps_ranked(tables):
    cont, _ = tables
    assert select_accessions(cont, threshold=0.5) == [
        acc for acc, _, _ in select_accessions_ranked(cont, threshold=0.5)
    ]
