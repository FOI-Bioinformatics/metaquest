"""
Tests for metaquest.store.stats: one cached FASTQ statistics record per dataset, computed
once and shared by sra_stats, sra_profile_quality and sra_compare instead of each command
re-parsing every read on its own.
"""

import gzip
from pathlib import Path
from unittest.mock import Mock, patch

from metaquest.store.sidecar import Sidecar, read_sidecar, write_sidecar
from metaquest.store.stats import cached_stats, compute_dataset_stats, store_stats


def _write_fastq(path: Path, reads, gz: bool = False) -> None:
    """Write a synthetic FASTQ file: one record per string in ``reads``, quality all 'I'."""
    lines = []
    for i, seq in enumerate(reads):
        lines.append(f"@read{i}")
        lines.append(seq)
        lines.append("+")
        lines.append("I" * len(seq))
    content = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt") as f:
            f.write(content)
    else:
        path.write_text(content)


class TestComputeDatasetStatsCounts:
    """Read counts must be correct for both plain and gzip-compressed FASTQ files."""

    def test_counts_plain_file(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, ["ACGTACGTAC"] * 20)

        stats = compute_dataset_stats([fastq], sample_size=1000, use_seqkit=False)

        assert stats["reads_per_file"][fastq.name] == 20
        assert stats["reads_total"] == 20
        assert stats["sample_size"] == 1000
        assert stats["sampled"] is False
        assert "computed" in stats and stats["computed"]

    def test_counts_gz_file(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq.gz"
        _write_fastq(fastq, ["ACGTACGTAC"] * 20, gz=True)

        stats = compute_dataset_stats([fastq], sample_size=1000, use_seqkit=False)

        assert stats["reads_per_file"][fastq.name] == 20
        assert stats["reads_total"] == 20

    def test_signature_tracks_size_and_mtime(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, ["ACGT"] * 5)

        stats = compute_dataset_stats([fastq], use_seqkit=False)

        st = fastq.stat()
        assert stats["signature"][fastq.name] == [st.st_size, st.st_mtime]

    def test_multiple_files_aggregate(self, tmp_path):
        f1 = tmp_path / "SRR1_1.fastq"
        f2 = tmp_path / "SRR1_2.fastq"
        _write_fastq(f1, ["ACGT"] * 7)
        _write_fastq(f2, ["ACGT"] * 3)

        stats = compute_dataset_stats([f1, f2], use_seqkit=False)

        assert stats["reads_per_file"][f1.name] == 7
        assert stats["reads_per_file"][f2.name] == 3
        assert stats["reads_total"] == 10


class TestUniformSampling:
    """The reservoir sample must be uniform over the whole file, not biased to its head."""

    def test_sampled_flag_set_when_file_exceeds_sample_size(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, ["ACGTACGT"] * 2000)

        stats = compute_dataset_stats([fastq], sample_size=500, use_seqkit=False)

        assert stats["reads_total"] == 2000
        assert stats["sampled"] is True
        assert stats["sample_size"] == 500

    def test_sample_reflects_tail_composition_of_a_50000_read_file(self, tmp_path):
        """40,000 head reads are all-AT (GC=0); 10,000 tail reads are all-GC (GC=1).

        A sampler biased toward the head of the file would see gc_content == 0.0; a
        uniform reservoir sample of 1,000 reads drawn from 50,000 must pick up some of
        the tail's GC-rich reads too (expected ~20% of the sample, well above noise).
        """
        head_reads = ["AAAAAAAA"] * 40000
        tail_reads = ["CCCCCCCC"] * 10000
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, head_reads + tail_reads)

        stats = compute_dataset_stats([fastq], sample_size=1000, use_seqkit=False)

        assert stats["reads_total"] == 50000
        assert stats["sampled"] is True
        assert stats["gc_content"] > 0.05


class TestCachedStats:
    """``cached_stats`` returns the sidecar's stats only while the signature matches."""

    def _seed(self, acc_dir: Path, reads):
        acc_dir.mkdir(parents=True, exist_ok=True)
        fastq = acc_dir / f"{acc_dir.name}.fastq"
        _write_fastq(fastq, reads)
        stats = compute_dataset_stats([fastq], use_seqkit=False)
        sidecar_path = acc_dir / f"{acc_dir.name}.json"
        write_sidecar(sidecar_path, Sidecar(accession=acc_dir.name, stats=stats))
        return fastq, sidecar_path, stats

    def test_cache_hit_when_signature_matches(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        _fastq, sidecar_path, stats = self._seed(acc_dir, ["ACGT"] * 10)

        hit = cached_stats(acc_dir, sidecar_path)

        assert hit == stats

    def test_cache_invalidated_by_size_change(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        fastq, sidecar_path, _stats = self._seed(acc_dir, ["ACGT"] * 10)

        # Rewrite the file with different content -> different size (and mtime).
        _write_fastq(fastq, ["ACGTACGTAC"] * 20)

        assert cached_stats(acc_dir, sidecar_path) is None

    def test_no_sidecar_path_returns_none(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()

        assert cached_stats(acc_dir, None) is None

    def test_missing_sidecar_file_returns_none(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()

        assert cached_stats(acc_dir, acc_dir / "missing.json") is None

    def test_sidecar_without_stats_returns_none(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        sidecar_path = acc_dir / "SRR1.json"
        write_sidecar(sidecar_path, Sidecar(accession="SRR1"))

        assert cached_stats(acc_dir, sidecar_path) is None


class TestStoreStats:
    """``store_stats`` writes through the sidecar, and is a no-op without one."""

    def test_writes_stats_and_timestamp_into_sidecar(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        sidecar_path = acc_dir / "SRR1.json"
        write_sidecar(sidecar_path, Sidecar(accession="SRR1"))

        stats = {"reads_total": 10, "signature": {}}
        store_stats(sidecar_path, stats)

        sidecar = read_sidecar(sidecar_path)
        assert sidecar.stats == stats
        assert sidecar.stats_computed is not None

    def test_noop_without_sidecar_path(self):
        # Must not raise: a project without a shared store has no sidecar to update.
        store_stats(None, {"reads_total": 1})

    def test_noop_when_sidecar_file_missing(self, tmp_path):
        store_stats(tmp_path / "missing.json", {"reads_total": 1})
        assert not (tmp_path / "missing.json").exists()


class TestSeqkitPath:
    """When seqkit is installed, exact per-file counts come from ``seqkit stats -T``."""

    def test_uses_seqkit_when_available(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, ["ACGTACGTAC"] * 20)  # 20 reads x 10bp = 200 bases

        table = (
            "file\tformat\ttype\tnum_seqs\tsum_len\tmin_len\tavg_len\tmax_len\n"
            f"{fastq}\tFASTQ\tDNA\t20\t200\t10\t10.0\t10\n"
        )
        fake_result = Mock(stdout=table, returncode=0)

        with patch("metaquest.store.stats.shutil.which", return_value="/usr/bin/seqkit"):
            with patch("metaquest.store.stats.SecureSubprocess.run_secure", return_value=fake_result) as mock_run:
                stats = compute_dataset_stats([fastq], sample_size=1000, use_seqkit=True)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        assert call_args[0] == "seqkit"
        assert call_args[1][:3] == ["stats", "-T", "-j"]

        assert stats["reads_total"] == 20
        assert stats["bases_total"] == 200
        assert stats["avg_read_length"] == 10.0
        assert stats["min_read_length"] == 10
        assert stats["max_read_length"] == 10

    def test_falls_back_when_seqkit_not_installed(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, ["ACGTACGTAC"] * 5)

        with patch("metaquest.store.stats.shutil.which", return_value=None):
            with patch("metaquest.store.stats.SecureSubprocess.run_secure") as mock_run:
                stats = compute_dataset_stats([fastq], use_seqkit=True)

        mock_run.assert_not_called()
        assert stats["reads_total"] == 5

    def test_use_seqkit_false_skips_seqkit_even_if_installed(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, ["ACGT"] * 3)

        with patch("metaquest.store.stats.shutil.which", return_value="/usr/bin/seqkit"):
            with patch("metaquest.store.stats.SecureSubprocess.run_secure") as mock_run:
                stats = compute_dataset_stats([fastq], use_seqkit=False)

        mock_run.assert_not_called()
        assert stats["reads_total"] == 3

    def test_seqkit_failure_falls_back_to_streaming_counts(self, tmp_path):
        fastq = tmp_path / "SRR1.fastq"
        _write_fastq(fastq, ["ACGT"] * 4)

        with patch("metaquest.store.stats.shutil.which", return_value="/usr/bin/seqkit"):
            with patch("metaquest.store.stats.SecureSubprocess.run_secure", side_effect=RuntimeError("boom")):
                stats = compute_dataset_stats([fastq], use_seqkit=True)

        assert stats["reads_total"] == 4
