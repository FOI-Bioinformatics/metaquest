"""Tests for metaquest.data.branchwater_search (no network; sourmash optional)."""

import json
import random
from unittest.mock import Mock, patch

import pytest
import requests

from metaquest.core.exceptions import DataAccessError
from metaquest.data.branchwater_search import (
    BRANCHWATER_COLUMNS,
    DEFAULT_SERVER,
    load_signature,
    search_index,
    sketch_fasta,
    write_branchwater_csv,
)

SIG_OBJECT = {
    "class": "sourmash_signature",
    "name": "wmel",
    "signatures": [
        {"num": 0, "ksize": 21, "seed": 42, "max_hash": 18446744073709552, "mins": [1, 2, 3], "molecule": "DNA"}
    ],
}


def _write_fasta(path, seed=7, length=5000):
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGT") for _ in range(length))
    path.write_text(f">contig1 test\n{seq[:2500]}\n>contig2\n{seq[2500:]}\n")


class TestSketchFasta:
    def test_builds_k21_scaled1000_signature(self, tmp_path):
        pytest.importorskip("sourmash")
        fasta = tmp_path / "GCF_000000001.1.fna"
        _write_fasta(fasta)
        signature = sketch_fasta(fasta)
        sketch = signature["signatures"][0]
        assert signature["name"] == "GCF_000000001.1"
        assert sketch["ksize"] == 21
        assert round(2**64 / sketch["max_hash"]) == 1000
        assert len(sketch["mins"]) > 0

    def test_missing_fasta_raises(self, tmp_path):
        pytest.importorskip("sourmash")
        with pytest.raises(DataAccessError, match="not found"):
            sketch_fasta(tmp_path / "missing.fna")

    def test_empty_fasta_raises(self, tmp_path):
        pytest.importorskip("sourmash")
        fasta = tmp_path / "empty.fna"
        fasta.write_text("")
        with pytest.raises(DataAccessError, match="No sequences"):
            sketch_fasta(fasta)

    def test_missing_sourmash_gives_install_hint(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sourmash"):
                raise ImportError("no sourmash")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        fasta = tmp_path / "x.fna"
        _write_fasta(fasta)
        with pytest.raises(DataAccessError, match=r"metaquest\[sourmash\]"):
            sketch_fasta(fasta)


class TestLoadSignature:
    def test_list_form(self, tmp_path):
        path = tmp_path / "wmel.sig"
        path.write_text(json.dumps([SIG_OBJECT]))
        assert load_signature(path)["name"] == "wmel"

    def test_object_form(self, tmp_path):
        path = tmp_path / "wmel.sig"
        path.write_text(json.dumps(SIG_OBJECT))
        assert load_signature(path)["signatures"][0]["ksize"] == 21

    def test_wrong_parameters_raise(self, tmp_path):
        bad = json.loads(json.dumps(SIG_OBJECT))
        bad["signatures"][0]["ksize"] = 31
        path = tmp_path / "k31.sig"
        path.write_text(json.dumps([bad]))
        with pytest.raises(DataAccessError, match="k=21, scaled=1000"):
            load_signature(path)

    def test_missing_or_invalid(self, tmp_path):
        with pytest.raises(DataAccessError, match="not found"):
            load_signature(tmp_path / "none.sig")
        path = tmp_path / "bad.sig"
        path.write_text("{not json")
        with pytest.raises(DataAccessError, match="not valid JSON"):
            load_signature(path)


class TestSearchIndex:
    @patch("metaquest.data.branchwater_search.requests.post")
    def test_posts_signature_and_parses_csv(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text="SRA accession,containment\nSRR1,0.5\nSRR2,0.9\n")
        matches = search_index(SIG_OBJECT, 0.1)
        assert matches == [("SRR2", 0.9), ("SRR1", 0.5)]
        args, kwargs = mock_post.call_args
        assert args[0] == f"{DEFAULT_SERVER}/search"
        assert kwargs["json"] == {"threshold": 0.1, "signature": SIG_OBJECT}
        assert kwargs["timeout"] == 600

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_http_error_raises(self, mock_post):
        mock_post.return_value = Mock(status_code=500, text="boom")
        with pytest.raises(DataAccessError, match="HTTP 500"):
            search_index(SIG_OBJECT, 0.1)

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_connection_error_raises(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("down")
        with pytest.raises(DataAccessError, match="search failed"):
            search_index(SIG_OBJECT, 0.1, server="https://example.org/")

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_unexpected_header_raises(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text="<html>redirect</html>")
        with pytest.raises(DataAccessError, match="Unexpected Branchwater response"):
            search_index(SIG_OBJECT, 0.1)

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_empty_result(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text="SRA accession,containment\n")
        assert search_index(SIG_OBJECT, 0.1) == []

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_rows_below_threshold_are_dropped(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200, text="SRA accession,containment\nSRR1,0.0009\nSRR2,0.5\nSRR3,0.1\n"
        )
        matches = search_index(SIG_OBJECT, 0.1)
        assert matches == [("SRR2", 0.5), ("SRR3", 0.1)]

    @patch("metaquest.data.branchwater_search.requests.post")
    def test_all_rows_below_threshold_gives_empty(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text="SRA accession,containment\nSRR1,0.0009\nSRR2,0.001\n")
        assert search_index(SIG_OBJECT, 0.1) == []


class TestWriteBranchwaterCsv:
    def test_header_and_cani(self, tmp_path):
        out = write_branchwater_csv([("SRR11011981", 0.9461), ("SRR2", 0.0)], tmp_path / "bw" / "wmel.csv")
        lines = out.read_text().splitlines()
        assert lines[0] == ",".join(BRANCHWATER_COLUMNS)
        assert lines[1] == "SRR11011981,0.9461,0.9974,,,,,,,"
        assert lines[2] == "SRR2,0.0000,0.0000,,,,,,,"

    def test_empty_matches_write_header_only(self, tmp_path):
        out = write_branchwater_csv([], tmp_path / "empty.csv")
        assert out.read_text().splitlines() == [",".join(BRANCHWATER_COLUMNS)]
