"""Tests for targeted read extraction (metaquest.data.read_extraction)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from metaquest.core.exceptions import DataAccessError, ProcessingError
from metaquest.data.read_extraction import (
    assemble_extracted_reads,
    extract_target_reads,
    select_samples_for_genome,
)


def _make_tree(tmp, paired=True):
    """Build a fixture: containment table, per-accession FASTQ, and a genome FASTA."""
    root = Path(tmp)
    table = root / "parsed_containment.txt"
    table.write_text(
        "\tGCF_1\tGCF_2\tmax_containment\n"
        "SRR1\t0.90\t0.0\t0.90\n"
        "SRR2\t0.40\t0.85\t0.85\n"
        "SRR3\t0.05\t0.0\t0.05\n"
    )
    for acc in ("SRR1", "SRR2", "SRR3"):
        d = root / "fastq" / acc
        d.mkdir(parents=True)
        (d / f"{acc}_1.fastq.gz").write_text("x")
        if paired:
            (d / f"{acc}_2.fastq.gz").write_text("x")
    genome = root / "GCF_1.fna"
    genome.write_text(">s\nACGT\n")
    return root, table, genome


class TestSelectSamples:
    def test_threshold_selection(self):
        df = pd.DataFrame({"GCF_1": [0.9, 0.4, 0.05]}, index=["SRR1", "SRR2", "SRR3"])
        assert select_samples_for_genome(df, "GCF_1", 0.5) == ["SRR1"]
        assert select_samples_for_genome(df, "GCF_1", 0.3) == ["SRR1", "SRR2"]

    def test_missing_genome_column(self):
        df = pd.DataFrame({"GCF_1": [0.9]}, index=["SRR1"])
        with pytest.raises(ProcessingError):
            select_samples_for_genome(df, "GCF_missing", 0.1)


class TestExtractTargetReads:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_paired_extraction_command_construction(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
            )
        # Only SRR1 clears threshold 0.5 for GCF_1.
        assert list(results) == ["SRR1"]
        assert [str(p.name) for p in results["SRR1"]] == ["GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"]

        tools = [c.args[0] for c in mock_run.call_args_list]
        assert tools == ["minimap2", "samtools", "samtools"]
        # samtools fastq for paired input uses -1/-2.
        fastq_args = mock_run.call_args_list[2].args[1]
        assert "-1" in fastq_args and "-2" in fastq_args

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_single_end_uses_flag_0(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=False)
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
            )
        assert [p.name for p in results["SRR1"]] == ["GCF_1.fastq.gz"]
        fastq_args = mock_run.call_args_list[2].args[1]
        assert "-0" in fastq_args

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_dry_run_runs_no_tools(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp)
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.3,
                dry_run=True,
            )
        assert set(results) == {"SRR1", "SRR2"}
        mock_run.assert_not_called()

    def test_missing_table_raises(self):
        with pytest.raises(DataAccessError):
            extract_target_reads("/no/such/table.txt", "GCF_1", "/no/genome.fna")

    def test_bad_preset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp)
            with pytest.raises(ProcessingError):
                extract_target_reads(table, "GCF_1", genome, preset="not-a-preset")

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_missing_genome_fasta_raises(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp:
            root, table, _ = _make_tree(tmp)
            with pytest.raises(DataAccessError):
                extract_target_reads(table, "GCF_1", root / "absent.fna", fastq_folder=root / "fastq")


class TestAssembleExtractedReads:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_paired_uses_1_2(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        out = assemble_extracted_reads([Path("a_1.fastq.gz"), Path("a_2.fastq.gz")], "asm")
        assert out == Path("asm")
        args = mock_run.call_args.args[1]
        assert "-1" in args and "-2" in args and "-r" not in args

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_single_uses_r(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assemble_extracted_reads([Path("a.fastq.gz")], "asm", min_contig_len=500)
        args = mock_run.call_args.args[1]
        assert "-r" in args and "--min-contig-len" in args

    def test_bad_read_count_raises(self):
        with pytest.raises(ProcessingError):
            assemble_extracted_reads([], "asm")
