"""Tests for the branchwater_search CLI command."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from metaquest.cli.commands.branchwater_search import BranchwaterSearchCommand
from metaquest.core.constants import DEFAULT_REGISTRY_MAX_SCREENED
from metaquest.core.exceptions import DataAccessError


def _args(tmp_path=None, **kwargs):
    registry_dir = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    base = dict(
        genome_fasta=None,
        signature=None,
        threshold=0.1,
        branchwater_folder="branchwater",
        output=None,
        server="https://s",
        registry=str(registry_dir / "metaquest_registry.json"),
        registry_max_screened=DEFAULT_REGISTRY_MAX_SCREENED,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestBranchwaterSearchCommand:
    def test_properties(self):
        cmd = BranchwaterSearchCommand()
        assert cmd.name == "branchwater_search"
        assert "Branchwater" in cmd.help

    def test_parser_requires_one_input(self):
        parser = argparse.ArgumentParser()
        BranchwaterSearchCommand().configure_parser(parser)
        args = parser.parse_args(["--genome-fasta", "g.fna"])
        assert args.threshold == 0.1 and args.branchwater_folder == "branchwater" and args.output is None
        assert args.registry_max_screened == DEFAULT_REGISTRY_MAX_SCREENED
        try:
            parser.parse_args([])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("one of --genome-fasta/--signature must be required")

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index")
    @patch("metaquest.cli.commands.branchwater_search.sketch_fasta")
    def test_fasta_default_output_path(self, mock_sketch, mock_search, mock_write, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_sketch.return_value = {"signatures": []}
        mock_search.return_value = [("SRR1", 0.9)]
        rc = BranchwaterSearchCommand().execute(_args(tmp_path, genome_fasta="genomes/GCF_000008025.1.fna"))
        assert rc == 0
        mock_search.assert_called_once_with({"signatures": []}, 0.1, server="https://s")
        mock_write.assert_called_once_with([("SRR1", 0.9)], Path("branchwater") / "GCF_000008025.1.csv")
        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        screening = data["datasets"]["SRR1"]["screening"]
        assert screening["genomes"]["GCF_000008025.1"]["source"] == "branchwater"
        assert screening["genomes"]["GCF_000008025.1"]["containment"] == 0.9

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index")
    @patch("metaquest.cli.commands.branchwater_search.load_signature")
    def test_signature_and_explicit_output(self, mock_load, mock_search, mock_write, tmp_path):
        mock_load.return_value = {"signatures": []}
        mock_search.return_value = []
        rc = BranchwaterSearchCommand().execute(_args(tmp_path, signature="wmel.sig", output=str(tmp_path / "out.csv")))
        assert rc == 0
        mock_write.assert_called_once_with([], Path(tmp_path / "out.csv"))
        assert (tmp_path / "metaquest_registry.json").exists()

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index", return_value=[])
    @patch("metaquest.cli.commands.branchwater_search.load_signature", return_value={"signatures": []})
    def test_zero_matches_warns(self, _load, _search, _write, caplog):
        with caplog.at_level("WARNING"):
            assert BranchwaterSearchCommand().execute(_args(signature="wmel.sig")) == 0
        assert "control genome" in caplog.text

    @patch("metaquest.cli.commands.branchwater_search.load_signature", side_effect=DataAccessError("bad sig"))
    def test_error_returns_1(self, _load):
        assert BranchwaterSearchCommand().execute(_args(signature="wmel.sig")) == 1

    @patch("metaquest.data.branchwater_search.requests.post")
    @patch("metaquest.cli.commands.branchwater_search.load_signature", return_value={"signatures": []})
    def test_server_ignoring_threshold_is_filtered_locally(self, _load, mock_post, tmp_path, caplog):
        mock_post.return_value = Mock(
            status_code=200, text="SRA accession,containment\nSRR1,0.0009\nSRR2,0.001\nSRR3,0.0005\n"
        )
        output = tmp_path / "out.csv"
        with caplog.at_level("WARNING"):
            rc = BranchwaterSearchCommand().execute(_args(tmp_path, signature="wmel.sig", output=str(output)))
        assert rc == 0
        assert "control genome" in caplog.text
        assert output.read_text().splitlines() == [
            "acc,containment,cANI,biosample,bioproject,assay_type,collection_date_sam,"
            "geo_loc_name_country_calc,organism,lat_lon"
        ]

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index")
    @patch("metaquest.cli.commands.branchwater_search.load_signature")
    def test_screening_entries_are_capped(self, mock_load, mock_search, _write, tmp_path, caplog):
        """A broad search must not fill the registry; only the best matches are kept."""
        mock_load.return_value = {"signatures": []}
        mock_search.return_value = [("SRR1", 0.9), ("SRR2", 0.5), ("SRR3", 0.2)]
        with caplog.at_level("WARNING"):
            rc = BranchwaterSearchCommand().execute(
                _args(tmp_path, signature="wmel.sig", output=str(tmp_path / "out.csv"), registry_max_screened=2)
            )
        assert rc == 0
        datasets = json.loads((tmp_path / "metaquest_registry.json").read_text())["datasets"]
        assert sorted(datasets) == ["SRR1", "SRR2"]
        assert "2" in caplog.text

    def test_registered(self):
        from metaquest.cli.main import create_parser, register_all_commands

        register_all_commands()
        parser = create_parser()
        action = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None))
        assert "branchwater_search" in action.choices
