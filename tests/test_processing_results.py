"""Tests for the consolidated per-accession, per-genome results table."""

import pandas as pd
import pytest

from metaquest.data import registry as reg
from metaquest.processing.results import RESULTS_COLUMNS, results_dataframe, results_rows, screened_pairs


def _registry(tmp_path):
    r = reg.load_registry(tmp_path / "metaquest_registry.json")
    reg.record_screening(r, "SRR1", "GCF_A", 0.9, 0.99, "matches", 0.0, None)
    reg.record_screening(r, "SRR1", "GCF_B", 0.2, None, "matches", 0.0, None)
    reg.record_screening(r, "SRR2", "GCF_A", 0.5, None, "matches", 0.0, None)
    reg.record_selection(r, ["SRR1"], {"column": "GCF_A", "threshold": 0.5}, tmp_path / "accessions.txt")
    reg.record_exclusion(r, "SRR2", "16S amplicon")
    reg.record_download(r, "SRR1", "downloaded", tmp_path / "fastq", attempt=False)
    reg.record_metadata(r, "SRR1", tmp_path / "metadata" / "SRR1.xml", {"run_size": "4000", "run_total_spots": "1000"})
    coverage = {"breadth": 0.8, "mean_depth": 12.5, "coverage_tsv": tmp_path / "targeted" / "cov.tsv"}
    reg.record_extraction(r, "SRR1", "GCF_A", [], 250, False, {}, coverage=coverage)
    stats = {"contigs": 12, "total_bp": 90000, "n50": 8000, "genome_fraction_estimate": 0.7, "mapping_rate": 0.95}
    reg.record_assembly(r, "SRR1", "GCF_A", tmp_path / "targeted" / "SRR1" / "GCF_A_assembly", stats, "v1", {})
    return r


def _by_pair(rows):
    return {(row["accession"], row["genome_id"]): row for row in rows}


def test_columns_in_stated_order():
    assert RESULTS_COLUMNS == [
        "accession",
        "genome_id",
        "containment",
        "selected",
        "excluded",
        "exclusion_reason",
        "download_state",
        "run_total_spots",
        "run_size",
        "mapped_reads",
        "mapping_rate_to_reference",
        "breadth",
        "mean_depth",
        "contigs",
        "total_bp",
        "n50",
        "genome_fraction_estimate",
        "assembly_mapping_rate",
    ]


def test_one_row_per_pair_sorted_by_containment(tmp_path):
    rows = results_rows(_registry(tmp_path))
    assert [(row["accession"], row["genome_id"]) for row in rows] == [
        ("SRR1", "GCF_A"),
        ("SRR2", "GCF_A"),
        ("SRR1", "GCF_B"),
    ]
    assert all(list(row) == RESULTS_COLUMNS for row in rows)


def test_row_fields_come_from_registry(tmp_path):
    rows = _by_pair(results_rows(_registry(tmp_path)))
    row = rows[("SRR1", "GCF_A")]
    assert row["containment"] == 0.9
    assert row["selected"] is True and row["excluded"] is False and row["exclusion_reason"] is None
    assert row["download_state"] == "downloaded"
    assert row["run_total_spots"] == 1000 and row["run_size"] == 4000
    assert row["mapped_reads"] == 250
    assert row["mapping_rate_to_reference"] == pytest.approx(0.25)
    assert row["breadth"] == 0.8 and row["mean_depth"] == 12.5
    assert (row["contigs"], row["total_bp"], row["n50"]) == (12, 90000, 8000)
    assert row["genome_fraction_estimate"] == 0.7 and row["assembly_mapping_rate"] == 0.95
    excluded = rows[("SRR2", "GCF_A")]
    assert excluded["selected"] is False and excluded["excluded"] is True
    assert excluded["exclusion_reason"] == "16S amplicon"
    assert excluded["download_state"] is None and excluded["mapped_reads"] is None


def test_min_containment_drops_low_pairs(tmp_path):
    rows = results_rows(_registry(tmp_path), min_containment=0.5)
    assert {(row["accession"], row["genome_id"]) for row in rows} == {("SRR1", "GCF_A"), ("SRR2", "GCF_A")}


def test_genome_id_restricts_rows(tmp_path):
    rows = results_rows(_registry(tmp_path), genome_id="GCF_B")
    assert [(row["accession"], row["genome_id"]) for row in rows] == [("SRR1", "GCF_B")]


def test_mapping_rate_none_without_spots(tmp_path):
    r = reg.load_registry(tmp_path / "metaquest_registry.json")
    reg.record_screening(r, "SRR3", "GCF_A", 0.4, None, "matches", 0.0, None)
    reg.record_extraction(r, "SRR3", "GCF_A", [], 50, False, {})
    row = results_rows(r)[0]
    assert row["mapped_reads"] == 50 and row["run_total_spots"] is None
    assert row["mapping_rate_to_reference"] is None


def test_mapping_rate_none_with_zero_mapped(tmp_path):
    r = reg.load_registry(tmp_path / "metaquest_registry.json")
    reg.record_metadata(r, "SRR3", tmp_path / "SRR3.xml", {"run_total_spots": 100})
    reg.record_extraction(r, "SRR3", "GCF_A", [], 0, False, {})
    assert results_rows(r)[0]["mapping_rate_to_reference"] is None


def test_older_extraction_without_coverage_keys_gives_empty_values(tmp_path):
    r = reg.load_registry(tmp_path / "metaquest_registry.json")
    r.datasets["SRR4"] = {"extractions": {"GCF_A": {"mapped_reads": 5, "files": []}}}
    row = results_rows(r)[0]
    assert row["breadth"] is None and row["mean_depth"] is None and row["contigs"] is None


def test_parsed_table_adds_pairs_the_registry_capped(tmp_path):
    r = _registry(tmp_path)
    table = pd.DataFrame(
        {
            "GCF_A": [0.91234, 0.0, 0.3],
            "GCF_B": [0.2, 0.0, 0.0],
            "max_containment": [0.91234, 0.0, 0.3],
            "max_containment_annotation": ["GCF_A", "", "GCF_A"],
        },
        index=["SRR1", "SRR5", "SRR6"],
    )
    pairs = screened_pairs(r, table)
    assert pairs[("SRR1", "GCF_A")] == 0.91234
    assert pairs[("SRR6", "GCF_A")] == 0.3
    assert pairs[("SRR2", "GCF_A")] == 0.5
    assert ("SRR5", "GCF_A") not in pairs
    assert not any(genome.startswith("max_containment") for _, genome in pairs)
    rows = _by_pair(results_rows(r, parsed_table=table))
    assert rows[("SRR6", "GCF_A")]["containment"] == 0.3
    assert rows[("SRR1", "GCF_A")]["containment"] == 0.91234


def test_extraction_without_screening_appears_with_empty_containment(tmp_path):
    r = _registry(tmp_path)
    reg.record_extraction(r, "SRR9", "GCF_C", [], 7, False, {})
    rows = results_rows(r)
    assert (rows[-1]["accession"], rows[-1]["genome_id"]) == ("SRR9", "GCF_C")
    assert rows[-1]["containment"] is None and rows[-1]["mapped_reads"] == 7


def test_min_containment_above_zero_drops_pairs_with_unknown_containment(tmp_path):
    r = _registry(tmp_path)
    reg.record_extraction(r, "SRR9", "GCF_C", [], 7, False, {})
    assert ("SRR9", "GCF_C") not in _by_pair(results_rows(r, min_containment=0.1))
    assert ("SRR9", "GCF_C") in _by_pair(results_rows(r))


def test_dataframe_has_exact_columns_and_is_header_only_when_empty(tmp_path):
    df = results_dataframe(results_rows(_registry(tmp_path)))
    assert list(df.columns) == RESULTS_COLUMNS and len(df) == 3
    empty = results_dataframe([])
    assert list(empty.columns) == RESULTS_COLUMNS and empty.empty


def test_dataframe_keeps_integers_as_integers(tmp_path):
    df = results_dataframe(results_rows(_registry(tmp_path)))
    text = df.to_csv(sep="\t", index=False)
    line = next(line for line in text.splitlines() if line.startswith("SRR1\tGCF_A"))
    fields = dict(zip(RESULTS_COLUMNS, line.split("\t")))
    assert fields["run_total_spots"] == "1000" and fields["contigs"] == "12"
    missing = next(line for line in text.splitlines() if line.startswith("SRR2\tGCF_A"))
    assert dict(zip(RESULTS_COLUMNS, missing.split("\t")))["mapped_reads"] == ""
