"""Tests for the select_datasets CLI command."""

import argparse
import json

import pytest

from metaquest.cli.commands.select import SelectDatasetsCommand
from metaquest.data.registry import load_registry, record_download, record_exclusion, save_registry


def _args(tmp_path, **kwargs):
    base = dict(
        parsed_containment=str(tmp_path / "parsed_containment.txt"),
        genome_id=None,
        genome_ids=None,
        require="any",
        threshold=0.5,
        top_n=None,
        metadata_file=None,
        metadata_column=None,
        metadata_value=None,
        output=str(tmp_path / "accessions.txt"),
        registry=str(tmp_path / "metaquest_registry.json"),
        skip_excluded=True,
        skip_downloaded=False,
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


def test_excluded_accessions_removed_by_default(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.8\t0.8\n")
    registry_path = tmp_path / "metaquest_registry.json"
    registry = load_registry(registry_path)
    record_exclusion(registry, "SRR2", "16S amplicon")
    save_registry(registry)

    with caplog.at_level("INFO"):
        rc = SelectDatasetsCommand().execute(_args(tmp_path, registry=str(registry_path)))
    assert rc == 0
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\n"
    assert any("already downloaded" in r.message and "excluded" in r.message for r in caplog.records)


def test_downloaded_accessions_removed_with_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.8\t0.8\n")
    registry_path = tmp_path / "metaquest_registry.json"
    registry = load_registry(registry_path)
    record_download(registry, "SRR2", "downloaded", tmp_path / "fastq")
    save_registry(registry)

    rc = SelectDatasetsCommand().execute(_args(tmp_path, registry=str(registry_path), skip_downloaded=True))
    assert rc == 0
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\n"


def test_no_skip_excluded_keeps_excluded_accessions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.8\t0.8\n")
    registry_path = tmp_path / "metaquest_registry.json"
    registry = load_registry(registry_path)
    record_exclusion(registry, "SRR2", "16S amplicon")
    save_registry(registry)

    rc = SelectDatasetsCommand().execute(_args(tmp_path, registry=str(registry_path), skip_excluded=False))
    assert rc == 0
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\nSRR2\n"


def test_argparse_rejects_genome_id_with_genome_ids():
    from metaquest.cli.main import create_parser, register_all_commands

    register_all_commands()
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "select_datasets",
                "--genome-id",
                "GCF_A",
                "--genome-ids",
                "GCF_A",
                "GCF_B",
            ]
        )


def test_argparse_rejects_top_n_zero():
    from metaquest.cli.main import create_parser, register_all_commands

    register_all_commands()
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["select_datasets", "--top-n", "0"])


def test_argparse_rejects_top_n_negative():
    from metaquest.cli.main import create_parser, register_all_commands

    register_all_commands()
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["select_datasets", "--top-n", "-1"])


def test_registry_records_ranked_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.8\t0.8\n")
    registry_path = tmp_path / "metaquest_registry.json"
    rc = SelectDatasetsCommand().execute(_args(tmp_path, registry=str(registry_path)))
    assert rc == 0
    data = json.loads(registry_path.read_text())
    ranked = data["datasets"]["SRR1"]["selection"]["ranked"]
    assert ranked == [{"accession": "SRR1", "rank": 1, "column": "max_containment", "value": 0.9}]
