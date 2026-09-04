"""Tests for the select_datasets CLI command."""

import argparse

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


def test_command_is_registered():
    from metaquest.cli.main import create_parser, register_all_commands

    register_all_commands()
    parser = create_parser()
    action = next(a for a in parser._subparsers._group_actions if getattr(a, "choices", None))
    assert "select_datasets" in action.choices
