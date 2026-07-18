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
