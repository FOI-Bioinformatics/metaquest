"""Tests for the select_datasets CLI command."""

import argparse
import json

from metaquest.cli.commands.select import SelectDatasetsCommand


def _args(tmp_path, **kwargs):
    base = dict(
        parsed_containment=str(tmp_path / "parsed_containment.txt"),
        genome_id=None,
        threshold=0.5,
        metadata_file=None,
        metadata_column=None,
        metadata_value=None,
        output=str(tmp_path / "accessions.txt"),
        registry=str(tmp_path / "metaquest_registry.json"),
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_writes_one_accession_per_line(tmp_path):
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    rc = SelectDatasetsCommand().execute(_args(tmp_path))
    assert rc == 0
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\n"


def test_missing_table_returns_1(tmp_path):
    rc = SelectDatasetsCommand().execute(_args(tmp_path))
    assert rc == 1


def test_selection_is_recorded_in_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    rc = SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A", threshold=0.5, registry=None))
    assert rc == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    sel = data["datasets"]["SRR1"]["selection"]
    assert sel["selected"] is True and sel["criteria"]["column"] == "GCF_A" and sel["criteria"]["threshold"] == 0.5
    assert "SRR2" not in data["datasets"] or not data["datasets"]["SRR2"]["selection"]["selected"]


def test_command_is_registered():
    from metaquest.cli.main import create_parser, register_all_commands

    register_all_commands()
    parser = create_parser()
    action = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None))
    assert "select_datasets" in action.choices
