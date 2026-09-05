"""Tests for targeted read extraction (metaquest.data.read_extraction)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from metaquest.core.exceptions import DataAccessError, ProcessingError
from metaquest.data.read_extraction import (
    ExtractionResult,
    assemble_extracted_reads,
    extract_target_reads,
    resolve_assembly_threads,
    select_samples_for_genome,
)
from helpers_extraction import _fake_tools


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
        state = {}
        mock_run.side_effect = _fake_tools(state)
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
            assert list(results) == ["SRR1"]
            assert [p.name for p in results["SRR1"].files] == ["GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"]
            # The orphan file (-0) is removed when empty.
            assert not (root / "targeted" / "SRR1" / "GCF_1_0.fastq.gz").exists()

        tools = [c[0] for c in state["calls"]]
        assert tools == ["minimap2", "samtools", "samtools", "samtools"]
        assert state["calls"][2][1][:2] == ["view", "-c"]
        fastq_args = state["calls"][3][1]
        assert fastq_args[0] == "fastq"
        assert all(flag in fastq_args for flag in ("-1", "-2", "-s", "-0"))

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_single_end_uses_flag_0(self, mock_run):
        state = {"nonempty": ("-0",)}
        mock_run.side_effect = _fake_tools(state)
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
            assert [p.name for p in results["SRR1"].files] == ["GCF_1.fastq.gz"]
        fastq_args = state["calls"][3][1]
        assert "-0" in fastq_args

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_zero_mapped_records_writes_nothing(self, mock_run, caplog):
        state = {"mapped": 0}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            with caplog.at_level("WARNING"):
                results = extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                )
            assert results == {"SRR1": ExtractionResult([], 0)}
            assert list((root / "targeted" / "SRR1").glob("*.fastq.gz")) == []
        assert "No reads from SRR1 mapped to GCF_1" in caplog.text
        # samtools fastq is never run when nothing mapped.
        assert [c[0] for c in state["calls"]] == ["minimap2", "samtools", "samtools"]

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_unequal_mates_fall_back_to_orphan_file(self, mock_run, caplog):
        state = {"unequal": True, "nonempty": ("-0",)}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            with caplog.at_level("WARNING"):
                results = extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                )
            assert [p.name for p in results["SRR1"].files] == ["GCF_1_0.fastq.gz"]
        assert "different read counts" in caplog.text

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_mapped_count_is_logged(self, mock_run, caplog):
        state = {"mapped": 1234}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            with caplog.at_level("INFO"):
                extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                )
        assert "1234 mapped records for SRR1" in caplog.text

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


class TestExtractionIdempotency:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_recorded_extraction_is_skipped_unless_forced(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            first = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
            )
            record = {
                "SRR1": {
                    "genome_fasta": str(genome),
                    "preset": "sr",
                    "threshold": 0.5,
                    "mapped_reads": first["SRR1"].mapped_records,
                    "unequal_mates": False,
                    "files": [str(p) for p in first["SRR1"].files],
                }
            }
            calls_before = len(state["calls"])
            again = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                already_done=record,
            )
            assert again["SRR1"].skipped is True and len(state["calls"]) == calls_before
            forced = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                already_done=record,
                force=True,
            )
            assert forced["SRR1"].skipped is False and len(state["calls"]) > calls_before

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_assembly_skips_existing_contigs_and_refuses_empty_dir(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asm"
            out.mkdir()
            (out / "final.contigs.fa").write_text(">c len=4\nACGT\n")
            assemble_extracted_reads([Path(tmp) / "r1.fq.gz", Path(tmp) / "r2.fq.gz"], out)
            assert not mock_run.called
            (out / "final.contigs.fa").unlink()
            with pytest.raises(ProcessingError, match="rerun with --force"):
                assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out)
            assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out, force=True)
            assert mock_run.called


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


class TestResolveAssemblyThreads:
    def test_explicit_request_wins(self, monkeypatch):
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        assert resolve_assembly_threads(8, 4) == 8

    def test_macos_defaults_to_single_thread(self, monkeypatch):
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        assert resolve_assembly_threads(None, 4) == 1

    def test_non_macos_uses_fallback(self, monkeypatch):
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Linux")
        assert resolve_assembly_threads(None, 4) == 4
