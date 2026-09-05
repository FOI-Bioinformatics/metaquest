"""Tests for the extract_target_reads CLI command."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from metaquest.cli.commands.read_extraction import ExtractTargetReadsCommand
from metaquest.core.exceptions import ProcessingError
from helpers_extraction import _fake_tools


def _args(tmp, **kwargs):
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
        force=False,
        registry=str(Path(tmp) / "registry.json"),
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


def _two_sample_tree(tmp):
    """Like _tree, but with two samples above the threshold."""
    root = Path(tmp)
    table = root / "parsed_containment.txt"
    table.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.8\n")
    for acc in ("SRR1", "SRR2"):
        d = root / "fastq" / acc
        d.mkdir(parents=True)
        (d / f"{acc}_1.fastq.gz").write_text("x")
        (d / f"{acc}_2.fastq.gz").write_text("x")
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
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
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
    def test_execute_returns_1_when_no_sample_yields_reads(self, mock_run, caplog):
        mock_run.side_effect = _fake_tools({"mapped": 0})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            with caplog.at_level("ERROR"):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                    )
                )
        assert rc == 1
        assert "No reads mapped to GCF_1" in caplog.text

    def test_execute_returns_1_when_no_sample_meets_threshold(self, caplog):
        """threshold above every sample's containment -> nothing selected."""
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            with caplog.at_level("ERROR"):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=2.0,
                    )
                )
        assert rc == 1
        assert "No sample meets containment >= 2.0 for GCF_1" in caplog.text

    def test_execute_returns_1_when_no_fastq_for_selected_samples(self, caplog):
        """A sample is selected but --fastq-folder does not contain its reads."""
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            with caplog.at_level("ERROR"):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "no-such-fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                    )
                )
        assert rc == 1
        assert "No FASTQ files found for the 1 selected sample(s) under" in caplog.text

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_dry_run(self, mock_run):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
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
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
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
        mock_run.side_effect = _fake_tools({})
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
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
        megahit_call = next(c for c in mock_run.call_args_list if c.args[0] == "megahit" and "-o" in c.args[1])
        args = megahit_call.args[1]
        assert args[args.index("--num-cpu-threads") + 1] == "1"

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_assembly_threads_override_on_macos(self, mock_run, monkeypatch):
        mock_run.side_effect = _fake_tools({})
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
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
        megahit_call = next(c for c in mock_run.call_args_list if c.args[0] == "megahit" and "-o" in c.args[1])
        args = megahit_call.args[1]
        assert args[args.index("--num-cpu-threads") + 1] == "6"

    def test_execute_missing_genome_column_returns_1(self):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_id="GCF_absent",
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                )
            )
        assert rc == 1

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_records_extraction_and_assembly(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    assemble=True,
                    registry=str(registry_file),
                )
            )
            assert rc == 0
            data = json.loads(registry_file.read_text())
        extraction = data["datasets"]["SRR1"]["extractions"]["GCF_1"]
        assert extraction["mapped_reads"] > 0
        assert extraction["assembly"]["contigs"] == 2

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_records_are_checkpointed_when_a_later_assembly_fails(self, mock_run):
        """Each extraction and assembly is written as it completes, so a later failure keeps them."""
        mock_run.side_effect = _fake_tools({})

        def fail_on_second(reads, out_dir, **kwargs):
            if "SRR2" in str(out_dir):
                raise ProcessingError("megahit crashed")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "final.contigs.fa").write_text(">c1 len=100\nACGT\n")
            return Path(out_dir), True

        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _two_sample_tree(tmp)
            registry_file = root / "registry.json"
            with patch("metaquest.cli.commands.read_extraction.assemble_extracted_reads", side_effect=fail_on_second):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                        assemble=True,
                        registry=str(registry_file),
                    )
                )
            assert rc == 1
            datasets = json.loads(registry_file.read_text())["datasets"]
        assert datasets["SRR1"]["extractions"]["GCF_1"]["mapped_reads"] > 0
        assert datasets["SRR2"]["extractions"]["GCF_1"]["mapped_reads"] > 0
        assert datasets["SRR1"]["extractions"]["GCF_1"]["assembly"]["contigs"] == 1
        assert datasets["SRR2"]["extractions"]["GCF_1"].get("assembly") is None

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_second_run_skips_and_returns_0(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            args = _args(
                tmp,
                parsed_containment=str(table),
                genome_fasta=str(genome),
                fastq_folder=str(root / "fastq"),
                output_folder=str(root / "targeted"),
                threshold=0.5,
                registry=str(registry_file),
            )
            rc = cmd.execute(args)
            assert rc == 0
            calls_before = len(mock_run.call_args_list)

            rc = cmd.execute(args)
            assert rc == 0
            tools_after_second_run = [c.args[0] for c in mock_run.call_args_list[calls_before:]]
            assert "minimap2" not in tools_after_second_run
            assert "samtools" not in tools_after_second_run

            forced_args = _args(
                tmp,
                parsed_containment=str(table),
                genome_fasta=str(genome),
                fastq_folder=str(root / "fastq"),
                output_folder=str(root / "targeted"),
                threshold=0.5,
                registry=str(registry_file),
                force=True,
            )
            calls_before_forced = len(mock_run.call_args_list)
            rc = cmd.execute(forced_args)
            assert rc == 0
            tools_after_forced_run = [c.args[0] for c in mock_run.call_args_list[calls_before_forced:]]
            assert "minimap2" in tools_after_forced_run
            assert "samtools" in tools_after_forced_run
