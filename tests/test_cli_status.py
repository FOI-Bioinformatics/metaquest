"""Tests for the `status` (local inventory) CLI command."""

import argparse
import gzip
import json
import shlex
import tempfile
from pathlib import Path

import pytest

from metaquest.cli.commands.status import StatusCommand
from metaquest.data.registry import (
    Registry,
    SCHEMA_VERSION,
    load_registry,
    record_exclusion,
    record_extraction,
    record_genome,
    record_selection,
    registry_transaction,
    save_registry,
    upsert_dataset,
)


def _args(registry, **kwargs):
    base = dict(
        registry=str(registry),
        data_root=None,
        fastq_folder="fastq",
        metadata_folder="metadata",
        genomes_folder="genomes",
        accessions_file=None,
        parsed_containment=None,
        list_missing=False,
        json=False,
        targeted_folder="targeted",
        matches_folder="matches",
        stage=None,
        genome=None,
        init=False,
        reconcile=False,
        export_tsv=None,
        next=False,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def _make_tree(tmp):
    """Create a small fixture tree: one accession fully present, one absent."""
    root = Path(tmp)
    (root / "fastq" / "SRR1").mkdir(parents=True)
    (root / "fastq" / "SRR1" / "SRR1.fastq.gz").write_text("@r\nACGT\n+\nIIII\n")
    (root / "metadata").mkdir()
    (root / "metadata" / "SRR1_metadata.xml").write_text("<xml/>")
    (root / "genomes").mkdir()
    (root / "genomes" / "GCF_000006945.2.fna").write_text(">s\nACGT\n")
    return root


class TestStatusCommand:
    def test_command_properties(self):
        cmd = StatusCommand()
        assert cmd.name == "status"
        assert "local" in cmd.help.lower()

    def test_on_disk_inventory(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                )
            )
        assert result == 0
        out = capsys.readouterr().out
        assert "FASTQ accessions on disk : 1" in out
        assert "Metadata XML on disk     : 1" in out
        assert "Genome FASTA on disk     : 1" in out

    def test_reconcile_present_and_missing(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            accs = root / "accs.txt"
            accs.write_text("SRR1\nSRR2\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    accessions_file=str(accs),
                    list_missing=True,
                )
            )
        assert result == 0
        out = capsys.readouterr().out
        assert "FASTQ    : 1 present, 1 missing" in out
        assert "Metadata : 1 present, 1 missing" in out
        assert "SRR2" in out

    def test_json_output(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            accs = root / "accs.txt"
            accs.write_text("SRR1\nSRR2\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    accessions_file=str(accs),
                    json=True,
                )
            )
        assert result == 0
        report = json.loads(capsys.readouterr().out)
        assert report["on_disk"]["fastq_accessions"] == 1
        assert report["wanted"]["total"] == 2
        assert report["wanted"]["fastq_missing"] == ["SRR2"]

    def test_on_disk_inventory_ignores_transient_temp_folder(self, capsys):
        """A <acc>_temp folder holding a partial FASTQ must not inflate the on-disk count."""
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            (root / "fastq" / "SRR2_temp").mkdir(parents=True)
            (root / "fastq" / "SRR2_temp" / "SRR2_temp_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    json=True,
                )
            )
        assert result == 0
        report = json.loads(capsys.readouterr().out)
        assert report["on_disk"]["fastq_accessions"] == 1

    def test_parsed_containment_supplies_wanted_list(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            table = root / "parsed_containment.txt"
            table.write_text("accession\tGCF_x\nSRR1\t0.9\nSRR2\t0.4\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    parsed_containment=str(table),
                    json=True,
                )
            )
        assert result == 0
        report = json.loads(capsys.readouterr().out)
        assert report["wanted"]["total"] == 2

    def test_missing_accessions_file_errors(self, tmp_path):
        cmd = StatusCommand()
        result = cmd.execute(_args(tmp_path / "metaquest_registry.json", accessions_file="/nonexistent/accs.txt"))
        assert result == 1


def _status_args(root, **overrides):
    base = dict(
        fastq_folder=str(root / "fastq"),
        metadata_folder=str(root / "metadata"),
        genomes_folder=str(root / "genomes"),
        targeted_folder=str(root / "targeted"),
        matches_folder=str(root / "matches"),
        registry=str(root / "metaquest_registry.json"),
        data_root=None,
        accessions_file=None,
        parsed_containment=None,
        stage=None,
        genome=None,
        init=False,
        reconcile=False,
        export_tsv=None,
        next=False,
        list_missing=False,
        json=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_inventory_ignores_appledouble(tmp_path):
    root = _make_tree(tmp_path)
    (root / "metadata" / "._SRR1_metadata.xml").write_bytes(b"\x00\x05")
    (root / "genomes" / "._g.fna").write_bytes(b"\x00\x05")
    # A hidden accession directory with a real, non-empty FASTQ file inside: only the
    # directory's own dotted name should exclude it, not an empty-folder accident.
    hidden_acc = root / "fastq" / "._SRR1"
    hidden_acc.mkdir()
    (hidden_acc / "reads.fastq.gz").write_bytes(b"x" * 10)
    report = StatusCommand()._inventory_report(_status_args(root), Registry())
    assert report["on_disk"]["metadata_xml"] == 1
    assert report["on_disk"]["genome_fasta"] == 1
    assert report["on_disk"]["fastq_accessions"] == 1


def _project_tree(root):
    for acc in ("SRR1", "SRR2"):
        d = root / "fastq" / acc
        d.mkdir(parents=True)
        (d / f"{acc}_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
    (root / "matches").mkdir()
    (root / "matches" / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\nSRR2,0.4,0.9\nSRR3,0.2,0.8\n")
    (root / "accessions.txt").write_text("SRR1\nSRR2\nSRR3\n")


class TestStatusWithRegistry:
    def test_without_registry_bootstraps_in_memory_and_hints(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(_status_args(tmp_path, accessions_file=str(tmp_path / "accessions.txt")))
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["registry"]["exists"] is False
        assert out["stages"]["screened"]["count"] == 3 and out["stages"]["downloaded"]["count"] == 2
        assert out["wanted"]["total"] == 3 and out["on_disk"]["fastq_accessions"] == 2
        assert not (tmp_path / "metaquest_registry.json").exists()

    def test_init_persists_bootstrap(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(
            _status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt"))
        )
        assert rc == 0 and (tmp_path / "metaquest_registry.json").exists()
        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR1"]["download"]["inferred"] is True

    def test_second_init_refuses_to_overwrite(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        capsys.readouterr()
        registry_path = tmp_path / "metaquest_registry.json"
        before = registry_path.read_text()
        rc = StatusCommand().execute(_status_args(tmp_path, init=True))
        assert rc == 1
        assert registry_path.read_text() == before

    def test_stage_and_genome_filters_list_accessions(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        rc = StatusCommand().execute(_status_args(tmp_path, stage="screened", genome=["GCF_1"], json=False))
        out = capsys.readouterr().out
        assert rc == 0 and "SRR1" in out and "SRR3" in out

    def test_next_suggests_download_and_extraction(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        capsys.readouterr()
        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        commands = [n["command"] for n in out["next"]]
        assert any(c.startswith("metaquest download_sra") for c in commands)  # SRR3 selected, not downloaded
        assert any(
            "extract_target_reads" in c and "GCF_1" in c for c in commands
        )  # SRR1/SRR2 downloaded, not extracted

    def test_next_does_not_suggest_a_no_skip_excluded_list(self, tmp_path, monkeypatch):
        """A selection recorded with --no-skip-excluded may still list an excluded accession,
        so --next must not suggest downloading that selection's output file directly; it
        should instead point at re-running select_datasets with --skip-excluded.

        SRR1 and SRR2 already have FASTQ on disk in `_project_tree`, so `--init` marks them
        downloaded and `_download_next_steps` would drop them regardless of selection
        criteria; SRR3 has no FASTQ on disk, so it is the accession left to download and the
        one whose selection criteria this test exercises."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_exclusion(reg, "SRR2", "isolate")
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {"skip_excluded": False, "column": "GCF_A", "threshold": 0.5},
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]
        download_commands = [c for c in commands if c.startswith("metaquest download_sra")]
        # sel_noskip.txt is never downloaded directly; it is only the --output of the
        # runnable reselect command (checked in test_reselect_suggestion_is_runnable).
        assert not any("sel_noskip.txt" in c for c in download_commands)
        assert any("select_datasets" in c and "--skip-excluded" in c for c in commands)

    def test_mixed_skip_and_no_skip_selections_yield_both_a_download_and_a_reselect(self, tmp_path, monkeypatch):
        """A registry holding both an ordinary (--skip-excluded) selection for one accession
        and a --no-skip-excluded selection for another, each still to download, must surface
        both kinds of next step at once: a direct download command for the ordinary
        selection's output file, and a reselect suggestion for the --no-skip-excluded one.
        Neither must swallow or replace the other.

        SRR4's selection is written directly rather than through a second `record_selection`
        call: that helper unselects anything selected earlier but absent from its own
        `accessions` argument, which would otherwise clear SRR3's selection made just above."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_selection(
                reg, ["SRR3"], {"skip_excluded": True, "column": "GCF_A", "threshold": 0.5}, "sel_normal.txt"
            )
            upsert_dataset(reg, "SRR4")["selection"] = {
                "selected": True,
                "date": "2026-01-01T00:00:00+00:00",
                "criteria": {"skip_excluded": False, "column": "GCF_B", "threshold": 0.3},
                "output": "sel_noskip.txt",
            }
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]

        download_commands = [c for c in commands if c.startswith("metaquest download_sra")]
        reselect_commands = [c for c in commands if c.startswith("metaquest select_datasets")]

        assert any("sel_normal.txt" in c for c in download_commands)
        assert len(reselect_commands) == 1
        assert "sel_noskip.txt" in reselect_commands[0] and "--skip-excluded" in reselect_commands[0]

    def test_reselect_suggestion_is_runnable(self, tmp_path, monkeypatch):
        """The reselect suggestion for a --no-skip-excluded selection is not a placeholder: it
        is built from the recorded criteria and can be run as-is to redo the selection with
        --skip-excluded, writing back to the same output file."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_exclusion(reg, "SRR2", "isolate")
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {"skip_excluded": False, "column": "GCF_A", "threshold": 0.5},
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]
        reselect = next(c for c in commands if c.startswith("metaquest select_datasets"))
        assert reselect == (
            "metaquest select_datasets --genome-id GCF_A --threshold 0.5 --skip-excluded --output sel_noskip.txt"
        )

    def test_reselect_suggestion_uses_genome_ids_when_recorded(self, tmp_path, monkeypatch):
        """A --no-skip-excluded selection made with --genome-ids reselects the same way."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {"skip_excluded": False, "genome_ids": ["GCF_A", "GCF_B"], "require": "all", "threshold": 0.3},
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]
        reselect = next(c for c in commands if c.startswith("metaquest select_datasets"))
        assert reselect == (
            "metaquest select_datasets --genome-ids GCF_A GCF_B --require all "
            "--threshold 0.3 --skip-excluded --output sel_noskip.txt"
        )

    def test_reselect_suggestion_omits_malformed_numbers_and_require(self, tmp_path, monkeypatch):
        """Values read back from a hand-edited registry are coerced (threshold to float, top-N
        to a positive int, require to any/all); a value that does not coerce is omitted rather
        than pasted into the command unquoted."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {
                    "skip_excluded": False,
                    "genome_ids": ["GCF_A", "GCF_B"],
                    "require": "all; touch pwned",
                    "threshold": "0.5; touch pwned",
                    "top_n": "3 && touch pwned",
                },
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        reselect = next(s["command"] for s in steps if s["command"].startswith("metaquest select_datasets"))
        assert "pwned" not in reselect
        assert reselect == (
            "metaquest select_datasets --genome-ids GCF_A GCF_B --skip-excluded --output sel_noskip.txt"
        )

    def test_reselect_command_warns_on_malformed_threshold(self, caplog):
        """A recorded threshold that does not coerce to a float is dropped from the suggested
        command (so select_datasets falls back to its own default on rerun) and now also logs
        a warning naming the bad value, instead of failing silently."""
        with caplog.at_level("WARNING"):
            command = StatusCommand._reselect_command({"column": "GCF_A", "threshold": "abc"}, "out.txt")
        assert "threshold" in caplog.text and "not a number" in caplog.text
        assert "'abc'" in caplog.text or '"abc"' in caplog.text
        assert "--threshold" not in command

    def test_reselect_command_warns_on_unknown_require(self, caplog):
        """A recorded require value outside any/all is dropped from the suggested command and
        logs a warning with its own wording (not "is not a number", which does not describe an
        enum value), naming the bad value, instead of failing silently."""
        with caplog.at_level("WARNING"):
            command = StatusCommand._reselect_command({"genome_ids": ["GCF_A", "GCF_B"], "require": "maybe"}, "out.txt")
        assert "Recorded require 'maybe' is not 'any' or 'all'; the suggested command omits --require" in caplog.text
        assert "--require" not in command

    def test_reselect_suggestion_includes_metadata_file(self, tmp_path, monkeypatch):
        """A selection recorded with --metadata-file must reproduce that flag on rerun, next
        to the metadata column/value it already reproduces, so the reselect actually rereads
        the same metadata table rather than falling back to metadata table autodetection. The
        path holds a space, so this also exercises that it is shell-quoted like the metadata
        value already is (test_reselect_suggestion_quotes_values_with_spaces)."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        meta_path = str(root / "my meta.txt")
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {
                    "skip_excluded": False,
                    "column": "GCF_A",
                    "threshold": 0.5,
                    "metadata_file": meta_path,
                    "metadata_column": "country",
                    "metadata_value": "Sweden",
                },
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]
        reselect = next(c for c in commands if c.startswith("metaquest select_datasets"))
        assert f"--metadata-file {shlex.quote(meta_path)}" in reselect
        assert "--metadata-column country --metadata-value Sweden" in reselect

    def test_reselect_suggestion_coerces_numeric_strings(self):
        command = StatusCommand._reselect_command(
            {"column": "GCF_A", "threshold": "0.25", "top_n": "7", "require": "any"}, "out.txt"
        )
        assert command == (
            "metaquest select_datasets --genome-id GCF_A --threshold 0.25 --skip-excluded --output out.txt --top-n 7"
        )

    def test_reselect_suggestion_reproduces_run_filters(self):
        """A selection recorded with the run size, spot count and platform filters reselects with
        the same four flags: the size as an integer byte count, the platform shell-quoted."""
        command = StatusCommand._reselect_command(
            {
                "column": "GCF_A",
                "threshold": 0.5,
                "max_run_size": 500000000,
                "min_spots": "1000",
                "max_spots": 2000000,
                "platform": "OXFORD NANOPORE",
            },
            "out.txt",
        )
        assert command == (
            "metaquest select_datasets --genome-id GCF_A --threshold 0.5 --skip-excluded --output out.txt"
            " --max-run-size 500000000 --min-spots 1000 --max-spots 2000000 --platform 'OXFORD NANOPORE'"
        )
        assert shlex.split(command)[-2:] == ["--platform", "OXFORD NANOPORE"]

    def test_reselect_suggestion_keeps_min_spots_zero(self):
        """--min-spots 0 is a valid flag (it drops runs with no spot count), so it is reproduced."""
        command = StatusCommand._reselect_command({"column": "GCF_A", "min_spots": 0}, "out.txt")
        assert command.endswith(" --min-spots 0")

    def test_reselect_command_warns_on_malformed_max_run_size(self, caplog):
        with caplog.at_level("WARNING"):
            command = StatusCommand._reselect_command(
                {"column": "GCF_A", "max_run_size": "big; rm -rf ~", "max_spots": -3}, "out.txt"
            )
        assert "Recorded max_run_size 'big; rm -rf ~'" in caplog.text
        assert "Recorded max_spots -3" in caplog.text
        assert "--max-run-size" not in command and "--max-spots" not in command and "rm -rf" not in command

    def test_reselect_suggestion_reproduces_metadata_filter_top_n_and_table(self, tmp_path, monkeypatch):
        """A selection made with a metadata filter, a top-N cap and a non-default containment
        table must reselect the same way: dropping any of those would reproduce a different,
        looser selection or read the wrong file."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {
                    "skip_excluded": False,
                    "column": "GCF_A",
                    "threshold": 0.5,
                    "metadata_column": "country",
                    "metadata_value": "Sweden",
                    "top_n": 10,
                    "table": "custom_containment.txt",
                },
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]
        reselect = next(c for c in commands if c.startswith("metaquest select_datasets"))
        assert reselect == (
            "metaquest select_datasets --genome-id GCF_A --threshold 0.5 --skip-excluded "
            "--output sel_noskip.txt --metadata-column country --metadata-value Sweden "
            "--top-n 10 --parsed-containment custom_containment.txt"
        )

    def test_reselect_suggestion_omits_table_flag_when_it_is_the_default(self, tmp_path, monkeypatch):
        """A recorded table equal to select_datasets' own default must not be echoed back as
        --parsed-containment; only a non-default table needs to be named explicitly."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {"skip_excluded": False, "column": "GCF_A", "threshold": 0.5, "table": "parsed_containment.txt"},
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]
        reselect = next(c for c in commands if c.startswith("metaquest select_datasets"))
        assert reselect == (
            "metaquest select_datasets --genome-id GCF_A --threshold 0.5 --skip-excluded --output sel_noskip.txt"
        )

    def test_reselect_suggestion_quotes_values_with_spaces(self, tmp_path, monkeypatch):
        """A metadata value containing a space must be shell-quoted, so the suggested command
        is actually safe to paste and run rather than breaking the shell's argument split."""
        root = tmp_path
        _project_tree(root)
        monkeypatch.chdir(root)
        StatusCommand().execute(_status_args(root, init=True))
        with registry_transaction(str(root / "metaquest_registry.json")) as reg:
            record_selection(
                reg,
                ["SRR1", "SRR2", "SRR3"],
                {
                    "skip_excluded": False,
                    "column": "GCF_A",
                    "threshold": 0.5,
                    "metadata_column": "city",
                    "metadata_value": "New York",
                },
                "sel_noskip.txt",
            )
        steps = StatusCommand._download_next_steps(load_registry(str(root / "metaquest_registry.json")))
        commands = [s["command"] for s in steps]
        reselect = next(c for c in commands if c.startswith("metaquest select_datasets"))

        assert "--metadata-value 'New York'" in reselect
        assert shlex.split(reselect) == [
            "metaquest",
            "select_datasets",
            "--genome-id",
            "GCF_A",
            "--threshold",
            "0.5",
            "--skip-excluded",
            "--output",
            "sel_noskip.txt",
            "--metadata-column",
            "city",
            "--metadata-value",
            "New York",
        ]

    def test_next_extraction_command_is_runnable(self, tmp_path, capsys):
        """The extract suggestion carries the table it was selected from and a FASTA that exists."""
        _project_tree(tmp_path)
        (tmp_path / "genomes").mkdir()
        (tmp_path / "genomes" / "GCF_1.fasta").write_text(">s\nACGT\n")
        table = tmp_path / "tables" / "containment.txt"
        table.parent.mkdir()
        table.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.4\n")
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_selection(seeded, ["SRR1", "SRR2"], {"column": "GCF_1", "table": str(table)}, tmp_path / "acc.txt")
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        extract = next(s for s in out["next"] if "extract_target_reads" in s["command"])
        assert f"--parsed-containment {table}" in extract["command"]
        assert f"--genome-fasta {tmp_path / 'genomes' / 'GCF_1.fasta'}" in extract["command"]

    def test_next_uses_the_recorded_genome_fasta(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_genome(seeded, "GCF_1", tmp_path / "refs" / "wMel.fna", tmp_path / "manifest.csv")
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        extract = next(s for s in out["next"] if "extract_target_reads" in s["command"])
        assert f"--genome-fasta {tmp_path / 'refs' / 'wMel.fna'}" in extract["command"]

    def test_a_moved_project_still_reports_the_recorded_genome_fasta(self, tmp_path, capsys):
        """A registry recorded under the old project location still resolves its genome FASTA
        after the project directory is renamed, since the path is stored relative to the
        registry file rather than to the working directory at record time."""
        old_root = tmp_path / "a"
        _project_tree(old_root)
        StatusCommand().execute(_status_args(old_root, init=True))
        seeded = load_registry(old_root / "metaquest_registry.json")
        record_genome(seeded, "GCF_1", old_root / "refs" / "wMel.fna", old_root / "manifest.csv")
        save_registry(seeded)
        capsys.readouterr()

        new_root = tmp_path / "b"
        old_root.rename(new_root)

        StatusCommand().execute(_status_args(new_root, next=True))
        out = json.loads(capsys.readouterr().out)
        extract = next(s for s in out["next"] if "extract_target_reads" in s["command"])
        assert f"--genome-fasta {new_root / 'refs' / 'wMel.fna'}" in extract["command"]

    def test_next_drops_excluded_and_already_extracted_accessions(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_exclusion(seeded, "SRR3", "16S amplicon")  # selected, not downloaded
        record_exclusion(seeded, "SRR2", "16S amplicon")  # downloaded
        record_extraction(seeded, "SRR1", "GCF_1", [], 0, False, {})  # zero mapped, but recorded
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        download = [s for s in out["next"] if "download_sra" in s["command"]]
        extract = [
            s for s in out["next"] if "extract_target_reads" in s["command"] and "--assemble" not in s["command"]
        ]
        assert download == []  # SRR3 was the only one left to download
        assert extract == []  # SRR1 has a record, SRR2 is excluded

    def test_reconcile_marks_missing_and_untracked(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        (tmp_path / "fastq" / "SRR2" / "SRR2_1.fastq").unlink()
        (tmp_path / "fastq" / "SRR8").mkdir()
        (tmp_path / "fastq" / "SRR8" / "SRR8_1.fastq").write_text("@r\nA\n+\nI\n")
        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        out = json.loads(capsys.readouterr().out)
        assert out["drift"]["recorded_missing"] == ["SRR2"] and out["drift"]["untracked_fastq"] == ["SRR8"]
        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR2"]["download"]["state"] == "missing"

    def test_reconcile_registers_untracked_work(self, tmp_path, capsys):
        """Untracked FASTQ and extractions are recorded as inferred, and still reported."""
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        (tmp_path / "fastq" / "SRR8").mkdir()
        (tmp_path / "fastq" / "SRR8" / "SRR8_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        extracted = tmp_path / "targeted" / "SRR1"
        extracted.mkdir(parents=True)
        for name in ("GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"):
            with gzip.open(extracted / name, "wt") as handle:
                handle.write("@r1\nACGT\n+\nIIII\n")

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        out = json.loads(capsys.readouterr().out)
        assert out["drift"]["untracked_fastq"] == ["SRR8"]
        assert out["drift"]["untracked_extractions"] == [["SRR1", "GCF_1"]]

        datasets = json.loads((tmp_path / "metaquest_registry.json").read_text())["datasets"]
        assert datasets["SRR8"]["download"]["state"] == "downloaded"
        assert datasets["SRR8"]["download"]["inferred"] is True
        assert datasets["SRR8"]["download"]["attempts"] == 0
        extraction = datasets["SRR1"]["extractions"]["GCF_1"]
        assert extraction["mapped_reads"] > 0 and extraction["inferred"] is True

    def test_reconcile_without_a_registry_refuses(self, tmp_path, caplog):
        _project_tree(tmp_path)
        with caplog.at_level("ERROR"):
            rc = StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        assert rc == 1
        assert not (tmp_path / "metaquest_registry.json").exists()
        assert "status --init" in caplog.text

    def test_report_carries_the_schema_version(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        out = json.loads(capsys.readouterr().out)
        assert out["registry"]["version"] == SCHEMA_VERSION

    def test_text_report_shows_selection_criteria_and_exclusion_reasons(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_selection(
            seeded,
            ["SRR1"],
            {"column": "GCF_1", "threshold": 0.5, "metadata_column": "organism", "metadata_value": "soil"},
            tmp_path / "accessions.txt",
        )
        record_exclusion(seeded, "SRR2", "16S amplicon")
        record_exclusion(seeded, "SRR3", "16S amplicon")
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "column GCF_1" in out and "threshold 0.5" in out and "organism = soil" in out
        assert "16S amplicon: 2" in out

    def test_text_report_shows_the_selection_run_filters(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_selection(
            seeded,
            ["SRR1"],
            {
                "column": "GCF_1",
                "threshold": 0.5,
                "max_run_size": 600_000_000,
                "min_spots": 3_000_000,
                "max_spots": 9_000_000,
                "platform": "illumina",
            },
            tmp_path / "accessions.txt",
        )
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert (
            "column GCF_1, threshold 0.5, max size 600 MB, spots >= 3000000, spots <= 9000000, platform illumina" in out
        )

    @pytest.mark.parametrize(
        "size,text",
        [(2_000_000_000, "max size 2 GB"), (1_500_000, "max size 1.5 MB"), (512, "max size 512 bytes")],
    )
    def test_selection_detail_formats_the_run_size(self, tmp_path, size, text):
        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_selection(registry, ["SRR1"], {"column": "GCF_1", "max_run_size": size}, tmp_path / "a.txt")
        assert text in StatusCommand._selection_detail(registry)

    def test_selection_detail_leaves_out_a_malformed_run_filter(self, tmp_path):
        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_selection(
            registry, ["SRR1"], {"column": "GCF_1", "max_run_size": "lots", "min_spots": -3}, tmp_path / "a.txt"
        )
        detail = StatusCommand._selection_detail(registry)
        assert "max size" not in detail and "spots" not in detail

    def test_export_tsv(self, tmp_path, capsys, caplog):
        _project_tree(tmp_path)
        extracted = tmp_path / "targeted" / "SRR1"
        extracted.mkdir(parents=True)
        for name in ("GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"):
            with gzip.open(extracted / name, "wt") as handle:
                handle.write("@r1\nACGT\n+\nIIII\n")

        with caplog.at_level("INFO"):
            StatusCommand().execute(_status_args(tmp_path, init=True, export_tsv=str(tmp_path / "registry")))

        assert (tmp_path / "registry_datasets.tsv").exists() and (tmp_path / "registry_extractions.tsv").exists()
        ext = (tmp_path / "registry_extractions.tsv").read_text().splitlines()[0]
        assert ext.startswith("accession\t")
        assert "registry_datasets.tsv" in caplog.text and "registry_extractions.tsv" in caplog.text

    def test_text_report_shows_stage_matrix(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "Local inventory" in out and "Stages" in out and "screened" in out and "GCF_1" in out

    def test_downloads_report_lists_truncated_and_unverified(self, tmp_path, capsys):
        """report['downloads'] surfaces registry verdicts; stages.downloaded is unaffected."""
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        seeded.datasets["SRR1"]["download"]["complete"] = {"verdict": "truncated"}
        seeded.datasets["SRR2"]["download"]["complete"] = {"verdict": "unverified"}
        save_registry(seeded)
        capsys.readouterr()

        rc = StatusCommand().execute(_status_args(tmp_path))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["downloads"]["truncated"] == ["SRR1"]
        assert out["downloads"]["unverified"] == ["SRR2"]
        assert out["stages"]["downloaded"]["count"] == 2

    def test_text_report_shows_truncated_downloads_line(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        seeded.datasets["SRR1"]["download"]["complete"] = {"verdict": "truncated"}
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "truncated downloads: 1 (SRR1)" in out


class TestStatusStorePlumbing:
    def test_no_store_configured_omits_store_section(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(_status_args(tmp_path))
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert "store" not in out

    def test_no_store_configured_text_report_has_no_store_block(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "Store" not in out

    def test_data_root_flag_adds_store_section(self, tmp_path, capsys):
        from metaquest.store.catalog import catalog_write
        from metaquest.store.layout import init_store

        _project_tree(tmp_path)
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "P1", "/p1", "reg1")

        rc = StatusCommand().execute(_status_args(tmp_path, data_root=str(store_root)))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["store"]["root"] == str(store_root.resolve())
        assert out["store"]["datasets"] == {}

    def test_data_root_flag_adds_text_store_block_after_local_inventory(self, tmp_path, capsys):
        from metaquest.store.layout import init_store

        _project_tree(tmp_path)
        store_root = tmp_path / "store"
        init_store(store_root)

        StatusCommand().execute(_status_args(tmp_path, data_root=str(store_root), json=False))
        out = capsys.readouterr().out

        assert "Local inventory" in out
        assert "Store" in out
        assert out.index("Local inventory") < out.index("Store")

    def test_registry_recorded_store_root_is_used_when_no_flag(self, tmp_path, capsys):
        from metaquest.store.layout import init_store

        _project_tree(tmp_path)
        store_root = tmp_path / "store"
        init_store(store_root)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        seeded.store = {"root": str(store_root), "mode": "symlink", "linked": []}
        save_registry(seeded)
        capsys.readouterr()

        rc = StatusCommand().execute(_status_args(tmp_path))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["store"]["root"] == str(store_root.resolve())


class TestStatusIncompleteStoreLinks:
    def test_status_names_links_to_incomplete_store_datasets(self, tmp_path, capsys):
        """A fastq/<ACC> symlink into a store dataset whose sidecar state is not ready (here
        'failed') is missing, but status must say why rather than reporting it as an ordinary
        missing accession."""
        from metaquest.store.layout import init_store, sidecar_path, sra_dir
        from metaquest.store.link import link_dataset
        from metaquest.store.sidecar import Sidecar, write_sidecar

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        dataset_dir = sra_dir(paths, "SRR1")
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "SRR1.fastq.gz").write_bytes(b"x" * 10)
        write_sidecar(sidecar_path(paths, "SRR1"), Sidecar(accession="SRR1", state="failed"))

        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()
        link_dataset(fastq_dir, "SRR1", paths, mode="absolute")

        (tmp_path / "matches").mkdir()
        (tmp_path / "matches" / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\n")
        (tmp_path / "accessions.txt").write_text("SRR1\n")

        rc = StatusCommand().execute(
            _status_args(
                tmp_path,
                accessions_file=str(tmp_path / "accessions.txt"),
                data_root=str(store_root),
                json=False,
            )
        )
        out = capsys.readouterr().out

        assert rc == 0
        assert "linked to a store dataset that is not complete" in out

    def test_json_report_lists_incomplete_store_links(self, tmp_path, capsys):
        from metaquest.store.layout import init_store, sidecar_path, sra_dir
        from metaquest.store.link import link_dataset
        from metaquest.store.sidecar import Sidecar, write_sidecar

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        dataset_dir = sra_dir(paths, "SRR1")
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "SRR1.fastq.gz").write_bytes(b"x" * 10)
        write_sidecar(sidecar_path(paths, "SRR1"), Sidecar(accession="SRR1", state="partial"))

        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()
        link_dataset(fastq_dir, "SRR1", paths, mode="absolute")

        (tmp_path / "matches").mkdir()
        (tmp_path / "matches" / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\n")
        (tmp_path / "accessions.txt").write_text("SRR1\n")

        rc = StatusCommand().execute(
            _status_args(
                tmp_path,
                accessions_file=str(tmp_path / "accessions.txt"),
                data_root=str(store_root),
            )
        )
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["wanted"]["fastq_incomplete_store_links"] == ["SRR1"]
        assert "SRR1" in out["wanted"]["fastq_missing"]

    def test_a_complete_store_link_is_not_reported_as_incomplete(self, tmp_path, capsys):
        from metaquest.store.layout import init_store, sidecar_path, sra_dir
        from metaquest.store.link import link_dataset
        from metaquest.store.sidecar import Sidecar, write_sidecar

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        dataset_dir = sra_dir(paths, "SRR1")
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "SRR1.fastq.gz").write_bytes(b"x" * 10)
        write_sidecar(sidecar_path(paths, "SRR1"), Sidecar(accession="SRR1", state="complete"))

        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()
        link_dataset(fastq_dir, "SRR1", paths, mode="absolute")

        (tmp_path / "matches").mkdir()
        (tmp_path / "matches" / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\n")
        (tmp_path / "accessions.txt").write_text("SRR1\n")

        rc = StatusCommand().execute(
            _status_args(
                tmp_path,
                accessions_file=str(tmp_path / "accessions.txt"),
                data_root=str(store_root),
                json=False,
            )
        )
        out = capsys.readouterr().out

        assert rc == 0
        assert "linked to a store dataset that is not complete" not in out


class TestStatusReconcileVerdict:
    def test_reconcile_fills_missing_download_verdict(self, tmp_path, capsys):
        """A download recorded before completeness verification existed (no `download.complete`)
        gets a verdict computed from what is on disk, once the registry records NCBI's spot count."""
        from metaquest.data.registry import record_metadata

        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()

        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_metadata(registry, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_total_spots": 100})
        save_registry(registry)

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        capsys.readouterr()

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        complete = data["datasets"]["SRR1"]["download"]["complete"]
        assert complete["verdict"] == "truncated"

    def test_reconcile_backfilling_a_verdict_does_not_disturb_the_recorded_date(self, tmp_path, capsys):
        """Filling in a missing verdict uses set_download_verdict, not record_download, so the
        download's recorded date (and files/bytes_total/message) survive untouched."""
        from metaquest.data.registry import record_metadata

        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()

        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_metadata(registry, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_total_spots": 100})
        original_date = registry.datasets["SRR1"]["download"]["date"]
        save_registry(registry)

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        capsys.readouterr()

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR1"]["download"]["date"] == original_date
        assert data["datasets"]["SRR1"]["download"]["complete"]["verdict"] == "truncated"

    def test_reconcile_leaves_existing_verdict_alone(self, tmp_path, capsys):
        """A verdict already on file (e.g. from a fresh download run) is never recomputed."""
        from metaquest.data.registry import record_download, record_metadata

        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()

        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_metadata(registry, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_total_spots": 100})
        record_download(
            registry,
            "SRR1",
            "downloaded",
            tmp_path / "fastq",
            attempt=False,
            complete={"method": "spots", "ratio": 1.0, "verdict": "complete"},
        )
        save_registry(registry)

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        capsys.readouterr()

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR1"]["download"]["complete"]["verdict"] == "complete"

    def test_reconcile_skips_store_sourced_downloads(self, tmp_path, capsys):
        """A download whose reads came from the shared store already has its own verdict
        pipeline; reconcile must not overwrite it."""
        from metaquest.data.registry import record_download, record_metadata

        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()

        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_metadata(registry, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_total_spots": 100})
        record_download(
            registry, "SRR1", "downloaded", tmp_path / "fastq", attempt=False, source="store", store_name="SRR1"
        )
        save_registry(registry)

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        capsys.readouterr()

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert "complete" not in data["datasets"]["SRR1"]["download"]


class TestStatusWithoutAReachableStore:
    """An unmounted or moved store must not stop a report about the project."""

    def test_status_reports_the_store_as_unavailable_and_still_lists_dangling_links(self, tmp_path, capsys):
        _project_tree(tmp_path)
        # The link points into a store that is not there: exactly what an unmounted volume
        # leaves behind, and exactly the thing status exists to report.
        gone = tmp_path / "unmounted"
        (tmp_path / "fastq" / "SRR3").symlink_to(gone / "sra" / "SRR3")

        assert StatusCommand().execute(_status_args(tmp_path, init=True)) == 0
        capsys.readouterr()

        rc = StatusCommand().execute(_status_args(tmp_path, data_root=str(gone), reconcile=True))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["store"]["available"] is False
        assert out["store"]["root"] == str(gone.resolve())
        assert "SRR3" in out["drift"]["dangling_links"]

    def test_text_report_says_the_store_could_not_be_read(self, tmp_path, capsys):
        _project_tree(tmp_path)
        gone = tmp_path / "unmounted"

        rc = StatusCommand().execute(_status_args(tmp_path, data_root=str(gone), json=False))
        out = capsys.readouterr().out

        assert rc == 0
        assert "Unavailable" in out

    def test_a_store_root_with_no_catalogue_is_reported_as_unavailable(self, tmp_path, capsys):
        from metaquest.store.layout import init_store

        _project_tree(tmp_path)
        store_root = tmp_path / "store"
        init_store(store_root)
        # A store folder that has never been written to has no catalog.sqlite yet.

        rc = StatusCommand().execute(_status_args(tmp_path, data_root=str(store_root)))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["store"]["available"] is False
