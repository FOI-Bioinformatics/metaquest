"""Tests for default-file resolution shared by CLI commands."""

from pathlib import Path

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.data.defaults import read_matrix, resolve_metadata_table


class TestResolveMetadataTable:
    def test_explicit_path_wins(self, tmp_path):
        table = tmp_path / "mine.txt"
        table.write_text("Run_ID\tx\n")
        assert resolve_metadata_table(str(table)) == table

    def test_ncbi_table_preferred_over_branchwater(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "metadata_table.txt").write_text("Run_ID\tx\n")
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "branchwater_metadata.txt").write_text("Run_ID\tx\n")
        assert resolve_metadata_table(None) == Path("metadata_table.txt")

    def test_falls_back_to_branchwater_metadata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "branchwater_metadata.txt").write_text("Run_ID\tx\n")
        assert resolve_metadata_table(None) == Path("metadata/branchwater_metadata.txt")

    def test_missing_lists_candidates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(DataAccessError, match="metadata_table.txt.*branchwater_metadata.txt"):
            resolve_metadata_table(None)

    def test_explicit_missing_raises(self, tmp_path):
        with pytest.raises(DataAccessError, match="nope.txt"):
            resolve_metadata_table(str(tmp_path / "nope.txt"))


class TestReadMatrix:
    def test_tsv_from_parse_containment_drops_text_columns(self, tmp_path):
        f = tmp_path / "parsed_containment.txt"
        f.write_text("\tGCF_A\tGCF_B\tmax_containment\tmax_containment_annotation\nSRR1\t0.9\t0.1\t0.9\tGCF_A\n")
        df = read_matrix(f)
        assert list(df.columns) == ["GCF_A", "GCF_B", "max_containment"]
        assert list(df.index) == ["SRR1"]

    def test_csv(self, tmp_path):
        f = tmp_path / "abundance.csv"
        f.write_text(",sp1,sp2\nS1,1,2\n")
        df = read_matrix(f)
        assert df.loc["S1", "sp2"] == 2
