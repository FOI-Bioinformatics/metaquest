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
        no_record=False,
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


def test_selection_records_resolved_metadata_file(tmp_path, monkeypatch):
    """The recorded criteria must carry the resolved --metadata-file path, so status --next
    can reproduce it on a reselect (tests/test_cli_status.py's reselect suite)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    meta = tmp_path / "meta.txt"
    meta.write_text("\tcountry\nSRR1\tSweden\nSRR2\tNorway\n")
    rc = SelectDatasetsCommand().execute(
        _args(
            tmp_path,
            genome_id="GCF_A",
            threshold=0.5,
            registry=None,
            metadata_file=str(meta),
            metadata_column="country",
            metadata_value="Sweden",
        )
    )
    assert rc == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    criteria = data["datasets"]["SRR1"]["selection"]["criteria"]
    assert criteria["metadata_file"] == str(meta.resolve())


def test_selection_records_autodetected_metadata_file(tmp_path, monkeypatch):
    """When --metadata-file is not given but --metadata-column is, execute() autodetects the
    default metadata table via resolve_metadata_table (see lines ~114-116); the recorded
    criteria must carry that table's resolved path too, not just an explicit --metadata-file,
    so a reselect built from the criteria does not depend on cwd-relative autodetection
    happening again."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    meta = tmp_path / "metadata_table.txt"
    meta.write_text("\tcountry\nSRR1\tSweden\nSRR2\tNorway\n")
    rc = SelectDatasetsCommand().execute(
        _args(
            tmp_path,
            genome_id="GCF_A",
            threshold=0.5,
            registry=None,
            metadata_column="country",
            metadata_value="Sweden",
        )
    )
    assert rc == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    criteria = data["datasets"]["SRR1"]["selection"]["criteria"]
    assert criteria["metadata_file"] == str(meta.resolve())


def test_selection_without_metadata_file_records_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    rc = SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A", threshold=0.5, registry=None))
    assert rc == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    assert data["datasets"]["SRR1"]["selection"]["criteria"]["metadata_file"] is None


def test_selection_records_no_metadata_file_without_a_metadata_column(tmp_path, monkeypatch):
    """--metadata-file without --metadata-column has no effect on the run (execute() only
    resolves it when args.metadata_column is set), so it must not be recorded either: a
    reselect command built from it would carry a --metadata-file flag that does nothing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    meta = tmp_path / "meta.txt"
    meta.write_text("\tcountry\nSRR1\tSweden\nSRR2\tNorway\n")
    rc = SelectDatasetsCommand().execute(
        _args(tmp_path, genome_id="GCF_A", threshold=0.5, registry=None, metadata_file=str(meta))
    )
    assert rc == 0
    data = json.loads((tmp_path / "metaquest_registry.json").read_text())
    assert data["datasets"]["SRR1"]["selection"]["criteria"]["metadata_file"] is None


def test_no_record_leaves_registry_untouched(tmp_path, monkeypatch):
    """--no-record writes the output file but must not redefine the project's target list:
    a later, differently-thresholded --no-record run leaves the registry's selected set
    (recorded by the first, normal run) exactly as it was."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    registry_path = tmp_path / "metaquest_registry.json"

    rc = SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A", threshold=0.5, registry=str(registry_path)))
    assert rc == 0
    before = registry_path.read_text()

    rc = SelectDatasetsCommand().execute(
        _args(
            tmp_path,
            genome_id="GCF_A",
            threshold=0.05,
            registry=str(registry_path),
            no_record=True,
            output=str(tmp_path / "exploratory.txt"),
        )
    )
    assert rc == 0
    assert (tmp_path / "exploratory.txt").read_text() == "SRR1\nSRR2\n"
    assert registry_path.read_text() == before


def test_no_record_refuses_the_recorded_selection_file(tmp_path, monkeypatch, caplog):
    """A --no-record run that would write to the file a recorded selection names is refused:
    the file keeps the recorded list and the registry stays as it was."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    registry_path = tmp_path / "metaquest_registry.json"
    rc = SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A", threshold=0.5, output="accessions.txt"))
    assert rc == 0
    before_registry = registry_path.read_text()
    before_file = (tmp_path / "accessions.txt").read_text()

    with caplog.at_level("ERROR"):
        rc = SelectDatasetsCommand().execute(
            _args(tmp_path, genome_id="GCF_A", threshold=0.05, no_record=True, output="accessions.txt")
        )
    assert rc == 1
    assert (tmp_path / "accessions.txt").read_text() == before_file == "SRR1\n"
    assert registry_path.read_text() == before_registry
    assert any("recorded selection" in r.message and "--output" in r.message for r in caplog.records)

    # The same file named by an absolute path is refused as well.
    rc = SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A", threshold=0.05, no_record=True))
    assert rc == 1
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\n"


def test_no_record_with_other_output_proceeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    registry_path = tmp_path / "metaquest_registry.json"
    assert SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A")) == 0
    before = registry_path.read_text()
    rc = SelectDatasetsCommand().execute(
        _args(tmp_path, genome_id="GCF_A", threshold=0.05, no_record=True, output="other.txt")
    )
    assert rc == 0
    assert (tmp_path / "other.txt").read_text() == "SRR1\nSRR2\n"
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\n"
    assert registry_path.read_text() == before


def test_no_record_default_output_without_recorded_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text("\tGCF_A\tmax_containment\nSRR1\t0.9\t0.9\nSRR2\t0.1\t0.1\n")
    rc = SelectDatasetsCommand().execute(_args(tmp_path, genome_id="GCF_A", no_record=True))
    assert rc == 0
    assert (tmp_path / "accessions.txt").read_text() == "SRR1\n"
    assert not (tmp_path / "metaquest_registry.json").exists()


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


def test_registry_records_the_combined_column_for_genome_ids(tmp_path, monkeypatch):
    """--genome-ids ranks on a combined column, and that is what the criteria must name.

    Recording the single --genome-id fallback here would describe a ranking that never ran.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parsed_containment.txt").write_text(
        "\tGCF_A\tGCF_B\tmax_containment\nSRR1\t0.9\t0.7\t0.9\nSRR2\t0.1\t0.1\t0.1\n"
    )
    registry_path = tmp_path / "metaquest_registry.json"
    rc = SelectDatasetsCommand().execute(
        _args(tmp_path, genome_ids=["GCF_A", "GCF_B"], require="any", registry=str(registry_path))
    )

    assert rc == 0
    data = json.loads(registry_path.read_text())
    column = data["datasets"]["SRR1"]["selection"]["ranked"][0]["column"]
    assert data["datasets"]["SRR1"]["selection"]["criteria"]["column"] == column
    assert "GCF_A" in column and "GCF_B" in column
