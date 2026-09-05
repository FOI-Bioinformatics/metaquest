"""Tests for the branchwater_search CLI command."""

import argparse
from pathlib import Path
from unittest.mock import patch

from metaquest.cli.commands.branchwater_search import BranchwaterSearchCommand
from metaquest.core.exceptions import DataAccessError


def _args(**kwargs):
    base = dict(
        genome_fasta=None,
        signature=None,
        threshold=0.1,
        branchwater_folder="branchwater",
        output=None,
        server="https://s",
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
        rc = BranchwaterSearchCommand().execute(_args(genome_fasta="genomes/GCF_000008025.1.fna"))
        assert rc == 0
        mock_search.assert_called_once_with({"signatures": []}, 0.1, server="https://s")
        mock_write.assert_called_once_with([("SRR1", 0.9)], Path("branchwater") / "GCF_000008025.1.csv")

    @patch("metaquest.cli.commands.branchwater_search.write_branchwater_csv")
    @patch("metaquest.cli.commands.branchwater_search.search_index")
    @patch("metaquest.cli.commands.branchwater_search.load_signature")
    def test_signature_and_explicit_output(self, mock_load, mock_search, mock_write, tmp_path):
        mock_load.return_value = {"signatures": []}
        mock_search.return_value = []
        rc = BranchwaterSearchCommand().execute(_args(signature="wmel.sig", output=str(tmp_path / "out.csv")))
        assert rc == 0
        mock_write.assert_called_once_with([], Path(tmp_path / "out.csv"))

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

    def test_registered(self):
        from metaquest.cli.main import create_parser, register_all_commands

        register_all_commands()
        parser = create_parser()
        action = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None))
        assert "branchwater_search" in action.choices
