"""Tests for the extract_target_reads CLI command."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

from metaquest.cli.commands.read_extraction import ExtractTargetReadsCommand


def _args(**kwargs):
    base = dict(
        parsed_containment="parsed_containment.txt",
        genome_id="GCF_1",
        genome_fasta="GCF_1.fna",
        fastq_folder="fastq",
        output_folder="targeted",
        threshold=0.1,
        preset="sr",
        threads=4,
        assemble=False,
        assembly_threads=None,
        min_contig_len=None,
        dry_run=False,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def _tree(tmp):
    root = Path(tmp)
    table = root / "parsed_containment.txt"
    table.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.05\n")
    d = root / "fastq" / "SRR1"
    d.mkdir(parents=True)
    (d / "SRR1_1.fastq.gz").write_text("x")
    (d / "SRR1_2.fastq.gz").write_text("x")
    genome = root / "GCF_1.fna"
    genome.write_text(">s\nACGT\n")
    return root, table, genome


class TestExtractTargetReadsCommand:
    def test_command_properties(self):
        cmd = ExtractTargetReadsCommand()
        assert cmd.name == "extract_target_reads"
        assert "target" in cmd.help.lower()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_extracts(self, mock_run):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                )
            )
        assert rc == 0
        assert mock_run.called

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_dry_run(self, mock_run):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    dry_run=True,
                )
            )
        assert rc == 0
        mock_run.assert_not_called()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_with_assembly(self, mock_run):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    assemble=True,
                )
            )
        assert rc == 0
        tools = [c.args[0] for c in mock_run.call_args_list]
        assert "megahit" in tools

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_assembly_single_thread_on_macos(self, mock_run, monkeypatch):
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    threads=4,
                    assemble=True,
                )
            )
        assert rc == 0
        megahit_call = next(c for c in mock_run.call_args_list if c.args[0] == "megahit")
        args = megahit_call.args[1]
        assert args[args.index("--num-cpu-threads") + 1] == "1"

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_assembly_threads_override_on_macos(self, mock_run, monkeypatch):
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    threads=4,
                    assemble=True,
                    assembly_threads=6,
                )
            )
        assert rc == 0
        megahit_call = next(c for c in mock_run.call_args_list if c.args[0] == "megahit")
        args = megahit_call.args[1]
        assert args[args.index("--num-cpu-threads") + 1] == "6"

    def test_execute_missing_genome_column_returns_1(self):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    parsed_containment=str(table),
                    genome_id="GCF_absent",
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                )
            )
        assert rc == 1
