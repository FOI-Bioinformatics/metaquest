"""Tests for the `status` (local inventory) CLI command."""

import argparse
import json
import tempfile
from pathlib import Path

from metaquest.cli.commands.status import StatusCommand


def _args(**kwargs):
    base = dict(
        fastq_folder="fastq",
        metadata_folder="metadata",
        genomes_folder="genomes",
        accessions_file=None,
        parsed_containment=None,
        list_missing=False,
        json=False,
        targeted_folder="targeted",
        matches_folder="matches",
        registry=None,
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

    def test_parsed_containment_supplies_wanted_list(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            table = root / "parsed_containment.txt"
            table.write_text("accession\tGCF_x\nSRR1\t0.9\nSRR2\t0.4\n")
            result = cmd.execute(
                _args(
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

    def test_missing_accessions_file_errors(self):
        cmd = StatusCommand()
        result = cmd.execute(_args(accessions_file="/nonexistent/accs.txt"))
        assert result == 1


def _status_args(root, **overrides):
    base = dict(
        fastq_folder=str(root / "fastq"),
        metadata_folder=str(root / "metadata"),
        genomes_folder=str(root / "genomes"),
        targeted_folder=str(root / "targeted"),
        matches_folder=str(root / "matches"),
        registry=str(root / "metaquest_registry.json"),
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

    def test_export_tsv(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, export_tsv=str(tmp_path / "registry")))
        assert (tmp_path / "registry_datasets.tsv").exists() and (tmp_path / "registry_extractions.tsv").exists()

    def test_text_report_shows_stage_matrix(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "Local inventory" in out and "Stages" in out and "screened" in out and "GCF_1" in out
