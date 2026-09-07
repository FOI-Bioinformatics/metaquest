"""Tests for targeted read extraction (metaquest.data.read_extraction)."""

import gzip
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from metaquest.core.exceptions import DataAccessError, ProcessingError, SecurityError
from metaquest.data.read_extraction import (
    ExtractionResult,
    _run_minimap2,
    _sample_reads,
    assemble_extracted_reads,
    assembly_coverage,
    build_index,
    extract_target_reads,
    fasta_length,
    resolve_assembly_threads,
    resolve_index_path,
    select_samples_for_genome,
    summarise_contigs,
)
from metaquest.data.registry import load_registry, record_extraction, resolve_project_path, save_registry
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


class TestSampleReads:
    """_sample_reads: which files of an accession folder are handed to minimap2."""

    def _acc_dir(self, tmp_path, *names):
        acc_dir = tmp_path / "fastq" / "SRR1"
        acc_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (acc_dir / name).write_text("@r\nACGT\n+\nIIII\n")
        return tmp_path / "fastq"

    def test_paired_folder_returns_mates_in_order(self, tmp_path):
        folder = self._acc_dir(tmp_path, "SRR1_2.fastq.gz", "SRR1_1.fastq.gz")

        assert [p.name for p in _sample_reads(folder, "SRR1")] == ["SRR1_1.fastq.gz", "SRR1_2.fastq.gz"]

    def test_orphan_file_is_not_passed_to_the_aligner(self, tmp_path):
        """--split-3 writes unpaired spots to a bare <acc>.fastq; minimap2 gets the mates only."""
        folder = self._acc_dir(tmp_path, "SRR1.fastq", "SRR1_1.fastq", "SRR1_2.fastq")

        assert [p.name for p in _sample_reads(folder, "SRR1")] == ["SRR1_1.fastq", "SRR1_2.fastq"]

    def test_single_end_folder_returns_the_bare_file(self, tmp_path):
        folder = self._acc_dir(tmp_path, "SRR1.fastq")

        assert [p.name for p in _sample_reads(folder, "SRR1")] == ["SRR1.fastq"]

    def test_partial_download_leftovers_are_ignored(self, tmp_path):
        """Zero-byte files and .gz.tmp.<pid> leftovers are not reads."""
        folder = self._acc_dir(tmp_path, "SRR1_1.fastq")
        (folder / "SRR1" / "SRR1_2.fastq").write_text("")
        (folder / "SRR1" / "SRR1_1.fastq.gz.tmp.4242").write_text("junk")

        assert [p.name for p in _sample_reads(folder, "SRR1")] == ["SRR1_1.fastq"]

    def test_missing_folder_returns_empty(self, tmp_path):
        assert _sample_reads(tmp_path / "fastq", "SRR1") == []


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
        # First minimap2 call builds the shared index (-d); the second aligns the sample.
        assert tools == ["minimap2", "minimap2", "samtools", "samtools", "samtools", "samtools"]
        assert "-d" in state["calls"][0][1]
        assert state["calls"][1][1][:2] == ["-a", "-x"]
        assert state["calls"][2][1][:2] == ["view", "-c"]
        assert state["calls"][3][1][:4] == ["view", "-b", "-F", "0x904"]
        assert state["calls"][4][1][:2] == ["view", "-c"]
        fastq_args = state["calls"][5][1]
        assert fastq_args[0] == "fastq"
        assert all(flag in fastq_args for flag in ("-1", "-2", "-s", "-0", "-@"))

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
        fastq_args = state["calls"][5][1]
        assert "-0" in fastq_args
        assert "-@" in fastq_args

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
        # Nothing mapped (per the SAM count), so filtering/counting the BAM and running
        # samtools fastq are all skipped.
        assert [c[0] for c in state["calls"]] == ["minimap2", "minimap2", "samtools"]

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
        seen = []
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
                on_result=lambda acc, result: seen.append(acc),
            )
        assert set(results) == {"SRR1", "SRR2"}
        mock_run.assert_not_called()
        assert seen == []  # a dry run records nothing

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_on_result_runs_per_sample_and_failures_are_isolated(self, mock_run, caplog):
        """A callback that raises for one sample must not stop the next sample's extraction."""
        mock_run.side_effect = _fake_tools({})
        seen = []

        def on_result(accession, result):
            seen.append((accession, result.mapped_records))
            if accession == "SRR1":
                raise RuntimeError("registry is locked by another process")

        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            with caplog.at_level("WARNING"):
                results = extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.3,
                    on_result=on_result,
                )
            assert [p.name for p in results["SRR2"].files] == ["GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"]
        assert [acc for acc, _ in seen] == ["SRR1", "SRR2"]
        assert set(results) == {"SRR1", "SRR2"}
        assert "Recording the extraction result for SRR1 failed" in caplog.text

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


class TestBuildIndex:
    """build_index/resolve_index_path: one shared minimap2 index per genome/preset."""

    def test_index_built_once_and_reused(self, tmp_path):
        genome = tmp_path / "g.fna"
        genome.write_text(">s\nACGT\n")
        index_dir = tmp_path / ".index"
        state = {}
        with patch("metaquest.data.read_extraction.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
            first = build_index(genome, "sr", index_dir)
            assert len(state["calls"]) == 1
            executable, args = state["calls"][0]
            assert executable == "minimap2"
            # Built under a temporary name and moved into place, so a concurrent run never
            # reads a half-written index.
            assert args[:3] == ["-x", "sr", "-d"]
            assert args[3].startswith(f"{first}.tmp.")
            assert args[4] == str(genome)
            assert first.is_file()
            again = build_index(genome, "sr", index_dir)
        assert again == first
        assert len(state["calls"]) == 1  # the FASTA has not changed, so the index is reused
        assert not list(index_dir.glob("*.tmp.*"))

    def test_index_rebuilt_for_a_different_genome_with_the_same_name(self, tmp_path):
        """Two genome files can share a stem, so the index is keyed on the FASTA's identity.

        The second FASTA deliberately carries an older mtime, which is what a copy restored
        from a tarball or moved with ``cp -p`` looks like; an mtime comparison alone would
        map it against the first genome's index.
        """
        old = tmp_path / "genomes_v1" / "wMel.fna"
        new = tmp_path / "genomes_v2" / "wMel.fna"
        old.parent.mkdir(parents=True)
        new.parent.mkdir(parents=True)
        old.write_text(">old\nAAAACCCC\n")
        new.write_text(">new\nCCCC\n")
        past = old.stat().st_mtime - 3600
        os.utime(new, (past, past))

        index_dir = tmp_path / ".index"
        state = {}
        with patch("metaquest.data.read_extraction.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
            first = build_index(old, "sr", index_dir)
            second = build_index(new, "sr", index_dir)

        assert first == second  # same stem, so the same index path
        assert len([c for c in state["calls"] if "-d" in c[1]]) == 2
        record = json.loads(index_dir.joinpath("wMel.sr.mmi.json").read_text())
        assert record["fasta"] == str(new.resolve())

    def test_a_failed_build_leaves_no_index(self, tmp_path):
        genome = tmp_path / "g.fna"
        genome.write_text(">s\nACGT\n")
        index_dir = tmp_path / ".index"

        def failing_build(executable, args, **kwargs):
            Path(args[args.index("-d") + 1]).write_bytes(b"half")  # a partial index
            raise SecurityError("minimap2 refused the command")

        with patch("metaquest.data.read_extraction.SecureSubprocess.run_secure", side_effect=failing_build):
            with pytest.raises(SecurityError):
                build_index(genome, "sr", index_dir)

        assert not (index_dir / "g.sr.mmi").exists()
        assert not list(index_dir.glob("*.tmp.*"))

    def test_index_rebuilt_when_fasta_touched_newer(self, tmp_path):
        genome = tmp_path / "g.fna"
        genome.write_text(">s\nACGT\n")
        index_dir = tmp_path / ".index"
        state = {}
        with patch("metaquest.data.read_extraction.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
            index_path = build_index(genome, "sr", index_dir)
            assert len(state["calls"]) == 1

            newer = index_path.stat().st_mtime + 10
            os.utime(genome, (newer, newer))

            build_index(genome, "sr", index_dir)
        assert len(state["calls"]) == 2

    def test_resolve_index_path_names_the_file_by_genome_stem_and_preset(self, tmp_path):
        path = resolve_index_path(tmp_path / "genomes" / "GCF_1.fna", "map-ont", tmp_path / ".index")
        assert path == tmp_path / ".index" / "GCF_1.map-ont.mmi"


class TestIndexReuseAcrossSamples:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_index_built_once_for_three_samples_and_reused(self, mock_run):
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
                threshold=0.05,
            )
            assert set(results) == {"SRR1", "SRR2", "SRR3"}
        index_calls = [c for c in state["calls"] if c[0] == "minimap2" and "-d" in c[1]]
        assert len(index_calls) == 1


class TestMapqAndSamLifetime:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_min_mapq_passed_to_the_filter_and_logged(self, mock_run, caplog):
        state = {"mapped_total": 100, "mapped": 80}
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
                    min_mapq=20,
                )
        filter_call = next(c for c in state["calls"] if c[0] == "samtools" and c[1][:2] == ["view", "-b"])
        args = filter_call[1]
        assert args[args.index("-F") + 1] == "0x904"
        assert args[args.index("-q") + 1] == "20"
        assert "kept 80 of 100 mapped records (secondary/supplementary and MAPQ below 20 removed: 20)" in caplog.text

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_no_min_mapq_omits_the_q_flag(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
            )
        filter_call = next(c for c in state["calls"] if c[0] == "samtools" and c[1][:2] == ["view", "-b"])
        assert "-q" not in filter_call[1]

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_sam_is_removed_after_the_run(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
            )
            assert list((root / "targeted" / "SRR1").glob("*.sam")) == []

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_sam_is_kept_with_the_debug_flag(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                keep_sam=True,
            )
            assert len(list((root / "targeted" / "SRR1").glob("*.sam"))) == 1

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_sam_written_under_temp_folder_when_given(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            temp_folder = root / "scratch"
            extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                temp_folder=temp_folder,
                keep_sam=True,
            )
            assert list(temp_folder.glob("*.sam"))
            assert list((root / "targeted" / "SRR1").glob("*.sam")) == []


class TestTruncatedDownloadSkip:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_truncated_sample_is_skipped_by_default(self, mock_run, caplog):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            truncated = {"SRR1": {"verdict": "truncated", "reads_r1": 5, "expected_spots": 20}}
            with caplog.at_level("WARNING"):
                results = extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                    truncated_downloads=truncated,
                )
        assert "SRR1" not in results
        assert not mock_run.called
        assert "skipped SRR1: download truncated (5 of 20 spots); use --allow-truncated" in caplog.text

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_allow_truncated_extracts_anyway(self, mock_run):
        state = {}
        mock_run.side_effect = _fake_tools(state)
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            truncated = {"SRR1": {"verdict": "truncated", "reads_r1": 5, "expected_spots": 20}}
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                truncated_downloads=truncated,
                allow_truncated=True,
            )
        assert "SRR1" in results
        assert mock_run.called


class TestMateCountMismatch:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_mismatched_mate_counts_map_single_end_without_the_stderr_marker(self, mock_run, caplog):
        """mate_counts alone (no minimap2 stderr marker) is enough to trigger the fallback."""
        state = {"nonempty": ("-0",)}
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
                    mate_counts={"SRR1": (100, 80)},
                )
        assert results["SRR1"].unequal_mates is True
        assert [p.name for p in results["SRR1"].files] == ["GCF_1_0.fastq.gz"]
        assert "mapped independently as single-end" in caplog.text

        minimap2_calls = [c for c in state["calls"] if c[0] == "minimap2"]
        # one shared index build, plus one alignment per mate file
        assert len(minimap2_calls) == 3
        alignment_calls = [c for c in minimap2_calls if "-a" in c[1]]
        assert all(len([r for r in c[1] if str(r).endswith(".fastq.gz")]) == 1 for c in alignment_calls)

        cat_calls = [c for c in state["calls"] if c[0] == "samtools" and c[1][:1] == ["cat"]]
        assert len(cat_calls) == 1

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_matching_mate_counts_map_as_a_pair(self, mock_run):
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
                mate_counts={"SRR1": (100, 100)},
            )
        assert results["SRR1"].unequal_mates is False
        alignment_calls = [c for c in state["calls"] if c[0] == "minimap2" and "-a" in c[1]]
        assert len(alignment_calls) == 1


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
            notified = []
            again = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                already_done=record,
                on_result=lambda acc, result: notified.append((acc, result.skipped)),
            )
            assert again["SRR1"].skipped is True and len(state["calls"]) == calls_before
            assert notified == [("SRR1", True)]  # skips are reported too
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
            _, ran = assemble_extracted_reads([Path(tmp) / "r1.fq.gz", Path(tmp) / "r2.fq.gz"], out)
            assert not mock_run.called and ran is False
            (out / "final.contigs.fa").unlink()
            with pytest.raises(ProcessingError, match="rerun with --force"):
                assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out)
            _, ran = assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out, force=True)
            assert mock_run.called and ran is True

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_inferred_record_without_parameters_is_skipped(self, mock_run):
        """A record bootstrapped from disk has no parameters; each missing one is a wildcard."""
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
            inferred = {
                "SRR1": {
                    "mapped_reads": 5,
                    "files": [str(p) for p in first["SRR1"].files],
                    "unequal_mates": False,
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
                already_done=inferred,
            )
            assert again["SRR1"].skipped is True and len(state["calls"]) == calls_before

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_inferred_zero_mapped_record_is_skipped(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta=genome,
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                already_done={"SRR1": {"mapped_reads": 0, "files": [], "unequal_mates": False}},
            )
        assert results["SRR1"].skipped is True
        mock_run.assert_not_called()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_relative_genome_path_matches_recorded_path(self, mock_run, monkeypatch):
        """./genomes/x.fna and genomes/x.fna are the same file, so the record still matches."""
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            monkeypatch.chdir(root)
            record = {
                "SRR1": {
                    "genome_fasta": "./GCF_1.fna",
                    "preset": "sr",
                    "threshold": 0.5,
                    "mapped_reads": 0,
                    "files": [],
                    "unequal_mates": False,
                }
            }
            results = extract_target_reads(
                parsed_containment=table,
                genome_id="GCF_1",
                genome_fasta="GCF_1.fna",
                fastq_folder=root / "fastq",
                output_folder=root / "targeted",
                threshold=0.5,
                already_done=record,
            )
        assert results["SRR1"].skipped is True
        mock_run.assert_not_called()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_resolved_record_still_matches_after_the_project_directory_is_renamed(self, mock_run, tmp_path):
        """A registry record is written under the project's original location, the project
        directory is then renamed, and the registry (now loaded from the new location) is
        resolved through resolve_project_path exactly as the CLI does; the resolved record
        must still match so extract_target_reads skips instead of remapping."""
        mock_run.side_effect = _fake_tools({})
        old_root = tmp_path / "a"
        old_root.mkdir()
        _make_tree(old_root, paired=True)
        genome = old_root / "GCF_1.fna"
        extracted_dir = old_root / "targeted" / "SRR1"
        extracted_dir.mkdir(parents=True)
        extracted_files = [extracted_dir / "GCF_1_1.fastq.gz", extracted_dir / "GCF_1_2.fastq.gz"]
        for f in extracted_files:
            f.write_text("@r\nACGT\n+\nIIII\n")

        registry_file = old_root / "metaquest_registry.json"
        registry = load_registry(registry_file)
        record_extraction(
            registry,
            "SRR1",
            "GCF_1",
            extracted_files,
            42,
            False,
            {"genome_fasta": genome, "preset": "sr", "threshold": 0.5},
        )
        save_registry(registry)

        new_root = tmp_path / "b"
        old_root.rename(new_root)

        reloaded = load_registry(new_root / "metaquest_registry.json")
        stored = reloaded.datasets["SRR1"]["extractions"]["GCF_1"]
        resolved_record = {
            "SRR1": {
                **stored,
                "genome_fasta": str(resolve_project_path(reloaded, stored["genome_fasta"])),
                "files": [str(resolve_project_path(reloaded, p)) for p in stored["files"]],
            }
        }

        results = extract_target_reads(
            parsed_containment=new_root / "parsed_containment.txt",
            genome_id="GCF_1",
            genome_fasta=new_root / "GCF_1.fna",
            fastq_folder=new_root / "fastq",
            output_folder=new_root / "targeted",
            threshold=0.5,
            already_done=resolved_record,
        )
        assert results["SRR1"].skipped is True
        mock_run.assert_not_called()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_dry_run_reports_would_be_skips(self, mock_run, caplog):
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _make_tree(tmp, paired=True)
            record = {
                "SRR1": {
                    "genome_fasta": str(genome),
                    "preset": "sr",
                    "threshold": 0.5,
                    "mapped_reads": 42,
                    "files": [],
                    "unequal_mates": False,
                }
            }
            with caplog.at_level("INFO"):
                results = extract_target_reads(
                    parsed_containment=table,
                    genome_id="GCF_1",
                    genome_fasta=genome,
                    fastq_folder=root / "fastq",
                    output_folder=root / "targeted",
                    threshold=0.5,
                    dry_run=True,
                    already_done=record,
                )
        assert results["SRR1"].skipped is True and results["SRR1"].mapped_records == 42
        assert "would skip SRR1 (already extracted, 42 mapped reads); use --force to redo" in caplog.text
        mock_run.assert_not_called()


class TestAssembleExtractedReads:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_paired_uses_1_2(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        out, ran = assemble_extracted_reads([Path("a_1.fastq.gz"), Path("a_2.fastq.gz")], "asm")
        assert out == Path("asm") and ran is True
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

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_default_preset_added_to_the_megahit_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assemble_extracted_reads([Path("a.fastq.gz")], "asm")
        args = mock_run.call_args.args[1]
        assert args[args.index("--presets") + 1] == "meta-sensitive"

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_explicit_preset_added_to_the_megahit_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assemble_extracted_reads([Path("a.fastq.gz")], "asm", preset="meta-large")
        args = mock_run.call_args.args[1]
        assert args[args.index("--presets") + 1] == "meta-large"

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_default_and_none_preset_omit_the_flag(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assemble_extracted_reads([Path("a.fastq.gz")], "asm", preset="default")
        assert "--presets" not in mock_run.call_args.args[1]
        assemble_extracted_reads([Path("a.fastq.gz")], "asm", preset=None)
        assert "--presets" not in mock_run.call_args.args[1]

    def test_preset_with_explicit_k_values_raises(self):
        with pytest.raises(ProcessingError, match="cannot be combined"):
            assemble_extracted_reads([Path("a.fastq.gz")], "asm", preset="meta-sensitive", k_flags={"k-min": 21})

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_explicit_k_values_used_without_a_preset(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assemble_extracted_reads([Path("a.fastq.gz")], "asm", preset=None, k_flags={"k-min": 21, "k-max": 141})
        args = mock_run.call_args.args[1]
        assert args[args.index("--k-min") + 1] == "21"
        assert args[args.index("--k-max") + 1] == "141"
        assert "--presets" not in args

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_intermediate_contigs_removed_by_default(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asm"
            assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out)
            assert (out / "final.contigs.fa").exists()
            assert not (out / "intermediate_contigs").exists()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_intermediate_contigs_kept_when_requested(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asm"
            assemble_extracted_reads([Path(tmp) / "r1.fq.gz"], out, keep_intermediate=True)
            assert (out / "intermediate_contigs").exists()


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


class TestRunMinimap2Retry:
    """_run_minimap2 retries once against the FASTA when the prebuilt index fails."""

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_index_failure_retries_against_the_fasta_and_warns(self, mock_run, tmp_path, caplog):
        genome = tmp_path / "g.fna"
        genome.write_text(">s\nACGT\n")
        index_path = tmp_path / "g.sr.mmi"
        sam_path = tmp_path / "out.sam"
        reads = [tmp_path / "r1.fastq.gz"]
        ok_result = MagicMock(returncode=0, stdout="", stderr="")
        mock_run.side_effect = [RuntimeError("index built by an incompatible minimap2 version"), ok_result]

        with caplog.at_level("WARNING"):
            result = _run_minimap2("SRR1", "sr", 4, sam_path, index_path, genome, reads)

        assert result is ok_result
        assert mock_run.call_count == 2
        first_call, second_call = mock_run.call_args_list
        assert first_call.args[0] == "minimap2" and str(index_path) in first_call.args[1]
        assert second_call.args[0] == "minimap2" and str(genome) in second_call.args[1]
        assert str(index_path) not in second_call.args[1]
        assert "retrying against the FASTA directly" in caplog.text


class TestSummariseContigsRicherStats:
    def test_n90_gc_and_size_bucket_from_real_sequence(self, tmp_path):
        contigs = tmp_path / "contigs.fa"
        # One 1000 bp contig, all G/C, and one 100 bp contig, all A/T.
        contigs.write_text(">c1\n" + "GC" * 500 + "\n>c2\n" + "AT" * 50 + "\n")

        stats = summarise_contigs(contigs)

        assert stats["contigs"] == 2
        assert stats["total_bp"] == 1100
        assert stats["largest"] == 1000
        assert stats["n50"] == 1000
        assert stats["n90"] == 1000
        assert stats["gc"] == round(1000 / 1100, 4)
        assert stats["contigs_ge_1kb"] == 1

    def test_missing_file_returns_a_zeroed_dict_with_the_new_keys(self, tmp_path):
        stats = summarise_contigs(tmp_path / "missing.fa")
        assert stats == {
            "contigs": 0,
            "total_bp": 0,
            "n50": 0,
            "n90": 0,
            "largest": 0,
            "gc": 0.0,
            "contigs_ge_1kb": 0,
        }

    def test_header_length_is_still_honoured_for_total_bp(self, tmp_path):
        """megahit's len= header value still wins for length even though gc is read from the
        (here deliberately short) sequence lines."""
        contigs = tmp_path / "contigs.fa"
        contigs.write_text(">c1 len=100\nACGT\n>c2 len=50\nACGT\n")
        stats = summarise_contigs(contigs)
        assert stats["total_bp"] == 150
        assert stats["largest"] == 100


class TestFastaLength:
    def test_counts_sequence_bases_only(self, tmp_path):
        fasta = tmp_path / "g.fna"
        fasta.write_text(">chr1 some description\nACGTACGT\nACGT\n>chr2\nTTTT\n")
        assert fasta_length(fasta) == 16


class TestAssemblyCoverage:
    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_coverage_numbers_from_faked_counts(self, mock_run, tmp_path):
        mock_run.side_effect = _fake_tools({"coverage_mapped": 8})
        contigs = tmp_path / "final.contigs.fa"
        contigs.write_text(">c1 len=100\nACGT\n>c2 len=50\nACGT\n")
        reads_file = tmp_path / "r_1.fastq.gz"
        with gzip.open(reads_file, "wt") as handle:
            handle.write("@r\nACGTACGTAC\n+\nIIIIIIIIII\n" * 5)

        result = assembly_coverage(contigs, [reads_file], "sr", 4, tmp_path, mapped_reads=40)

        assert result["reads_mapped"] == 8
        assert result["mapping_rate"] == pytest.approx(0.2)
        assert result["mean_depth_estimate"] == pytest.approx(8 * 10 / 150)
        assert not (tmp_path / "coverage.sam").exists()
        assert not (tmp_path / "coverage.bam").exists()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_mapping_rate_is_none_when_mapped_reads_is_unknown_or_zero(self, mock_run, tmp_path):
        mock_run.side_effect = _fake_tools({})
        contigs = tmp_path / "final.contigs.fa"
        contigs.write_text(">c1 len=10\nACGT\n")
        reads_file = tmp_path / "r.fastq.gz"
        with gzip.open(reads_file, "wt") as handle:
            handle.write("@r\nACGT\n+\nIIII\n")

        assert assembly_coverage(contigs, [reads_file], "sr", 2, tmp_path, mapped_reads=None)["mapping_rate"] is None
        assert assembly_coverage(contigs, [reads_file], "sr", 2, tmp_path, mapped_reads=0)["mapping_rate"] is None

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_minimap2_and_samtools_args_shape(self, mock_run, tmp_path):
        mock_run.side_effect = _fake_tools({})
        contigs = tmp_path / "final.contigs.fa"
        contigs.write_text(">c1 len=10\nACGT\n")
        reads_file = tmp_path / "r.fastq.gz"
        with gzip.open(reads_file, "wt") as handle:
            handle.write("@r\nACGT\n+\nIIII\n")

        assembly_coverage(contigs, [reads_file], "map-ont", 3, tmp_path, mapped_reads=10)

        minimap2_call = next(c for c in mock_run.call_args_list if c.args[0] == "minimap2")
        args = minimap2_call.args[1]
        assert args[:3] == ["-a", "-x", "map-ont"]
        assert args[args.index("-t") + 1] == "3"
        assert str(contigs) in args and str(reads_file) in args

        filter_call = next(c for c in mock_run.call_args_list if c.args[0] == "samtools" and c.args[1][0] == "view")
        fargs = filter_call.args[1]
        assert fargs[fargs.index("-F") + 1] == "0x904"
        assert fargs[fargs.index("-@") + 1] == "3"
