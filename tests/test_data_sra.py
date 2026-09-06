"""
Tests for metaquest.data.sra module.
"""

import gzip
import inspect
import json

import pytest
from pathlib import Path
from unittest.mock import Mock, call, patch

from metaquest.core.exceptions import DataAccessError, SecurityError
from metaquest.data.sra import (
    _read_blacklist_files,
    _prepare_temp_folder,
    _check_existing_download,
    _cached_sra_archive,
    _handle_download_output,
    download_accession,
    _check_existing_downloads,
    _process_download_results,
    _retry_failed_downloads,
    _handle_download_failure,
    download_sra,
    accession_has_fastq,
    fastq_files,
    primary_fastq,
    orphan_fastq,
    count_fastq_reads,
    verify_download,
    parse_verdict_message,
    classify_download_error,
    default_max_workers,
    fasterq_dump_version,
    is_transient_folder,
    compress_fastq,
    transient_bytes,
)
from helpers_extraction import _fake_tools


class TestReadBlacklistFiles:
    """Test _read_blacklist_files function."""

    def test_read_blacklist_files_success(self, tmp_path):
        """Test successful blacklist file reading."""
        blacklist1 = tmp_path / "blacklist1.txt"
        blacklist2 = tmp_path / "blacklist2.txt"

        blacklist1.write_text("SRR123\nSRR456\n\n")  # Include empty line
        blacklist2.write_text("SRR789\nSRR123\n")  # Include duplicate

        result = _read_blacklist_files([blacklist1, blacklist2])

        assert result == {"SRR123", "SRR456", "SRR789"}

    def test_read_blacklist_files_empty_list(self):
        """Test with empty blacklist files list."""
        result = _read_blacklist_files([])
        assert result == set()

    def test_read_blacklist_files_none(self):
        """Test with None blacklist files."""
        result = _read_blacklist_files(None)
        assert result == set()

    def test_read_blacklist_files_nonexistent_file(self, tmp_path):
        """Test handling nonexistent blacklist file."""
        nonexistent = tmp_path / "nonexistent.txt"

        with patch("metaquest.data.sra.logger") as mock_logger:
            result = _read_blacklist_files([nonexistent])

        assert result == set()
        mock_logger.warning.assert_called_once()

    def test_read_blacklist_files_empty_file(self, tmp_path):
        """Test handling empty blacklist file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = _read_blacklist_files([empty_file])
        assert result == set()

    def test_read_blacklist_files_whitespace_handling(self, tmp_path):
        """Test proper whitespace handling."""
        blacklist = tmp_path / "blacklist.txt"
        blacklist.write_text("  SRR123  \n\t SRR456 \t\n   \n")

        result = _read_blacklist_files([blacklist])
        assert result == {"SRR123", "SRR456"}

    def test_blacklist_reader_ignores_inline_reasons_and_comment_lines(self, tmp_path):
        from metaquest.data.sra import _read_blacklist_files

        bl = tmp_path / "blacklist.txt"
        bl.write_text("# written by metaquest blacklist\nSRR1  # amplicon\nSRR2\n\n")
        assert _read_blacklist_files([bl]) == {"SRR1", "SRR2"}


class TestPrepareTempFolder:
    """Test _prepare_temp_folder function."""

    def test_prepare_temp_folder_create_new(self, tmp_path):
        """Test creating new temp folder."""
        temp_base = tmp_path / "temp"

        result = _prepare_temp_folder(temp_base)

        assert result.exists()
        assert result.is_dir()

    def test_prepare_temp_folder_existing_folder(self, tmp_path):
        """Test with existing temp folder."""
        temp_base = tmp_path / "temp"
        temp_base.mkdir()

        result = _prepare_temp_folder(temp_base)

        assert result.exists()
        assert result.is_dir()

    def test_prepare_temp_folder_none_input(self):
        """Test with None temp folder."""

        with patch("tempfile.mkdtemp") as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/test_temp"

            result = _prepare_temp_folder(None)

            mock_mkdtemp.assert_called_once()
            assert str(result) == "/tmp/test_temp"


class TestCheckExistingDownload:
    """Test _check_existing_download function."""

    def test_check_existing_download_no_files(self, tmp_path):
        """Test when no files exist."""
        output_path = tmp_path / "downloads" / "SRR123"

        result = _check_existing_download(output_path, force=False)
        assert result is False

    def test_check_existing_download_with_files_no_force(self, tmp_path):
        """Test when files exist and force=False."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("test data")

        result = _check_existing_download(output_path, force=False)
        assert result is True

    def test_check_existing_download_with_files_force(self, tmp_path):
        """Test when files exist and force=True."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("test data")

        with patch("shutil.rmtree") as mock_rmtree:
            result = _check_existing_download(output_path, force=True)

        assert result is False
        mock_rmtree.assert_called_once_with(output_path)

    def test_check_existing_download_empty_directory(self, tmp_path):
        """Test when directory exists but is empty."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)

        with patch("metaquest.data.sra.logger"):
            result = _check_existing_download(output_path, force=False)

        assert result is False
        # Directory should be removed
        assert not output_path.exists()

    def test_check_existing_download_rmdir_error(self, tmp_path):
        """Test handling rmdir error."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)

        with patch.object(Path, "rmdir", side_effect=OSError("Permission denied")):
            with patch("metaquest.data.sra.logger") as mock_logger:
                result = _check_existing_download(output_path, force=False)

        assert result is False
        mock_logger.warning.assert_called()

    def test_check_existing_download_dangling_symlink(self, tmp_path):
        """A broken symlink left by an interrupted run must be cleared, never raise."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.parent.mkdir(parents=True)
        output_path.symlink_to(tmp_path / "gone-target")

        result = _check_existing_download(output_path, force=False)

        assert result is False
        assert not output_path.is_symlink()

    def test_check_existing_download_symlinked_empty_directory_is_unlinked(self, tmp_path):
        """A symlink to an empty directory is unlinked, not rmdir'd (rmdir rejects symlinks)."""
        real_dir = tmp_path / "real_empty"
        real_dir.mkdir()
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.parent.mkdir(parents=True)
        output_path.symlink_to(real_dir)

        result = _check_existing_download(output_path, force=False)

        assert result is False
        assert not output_path.is_symlink()

    def test_check_existing_download_redownload_truncated_forces_fresh(self, tmp_path):
        """force=True (standing in for a truncated accession) removes the directory."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("partial")

        result = _check_existing_download(output_path, force=True)

        assert result is False
        assert not output_path.exists()


class TestFastqFiles:
    """Test fastq_files, the single source of truth for what counts as a FASTQ file on disk."""

    def test_returns_sorted_nonempty_matches(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1_2.fastq").write_text("@r\nACGT\n+\nIIII\n")
        (acc_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        (acc_dir / "SRR1_empty.fastq").write_text("")
        (acc_dir / "notes.txt").write_text("not fastq")

        result = fastq_files(acc_dir)

        assert [p.name for p in result] == ["SRR1_1.fastq", "SRR1_2.fastq"]

    def test_matches_fq_gz(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        with gzip.open(acc_dir / "SRR1.fq.gz", "wt") as handle:
            handle.write("@r\nACGT\n+\nIIII\n")

        assert [p.name for p in fastq_files(acc_dir)] == ["SRR1.fq.gz"]

    def test_missing_directory_returns_empty(self, tmp_path):
        assert fastq_files(tmp_path / "missing") == []

    def test_follows_symlinked_directory(self, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        link = tmp_path / "SRR1"
        link.symlink_to(real_dir)

        assert [p.name for p in fastq_files(link)] == ["SRR1_1.fastq"]


class TestAccessionHasFastq:
    """Test accession_has_fastq, the single source of truth for "already downloaded"."""

    def test_false_for_zero_byte_file(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1.fastq").write_text("")
        assert accession_has_fastq(acc_dir) is False

    def test_true_for_fq_gz(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        with gzip.open(acc_dir / "SRR1.fq.gz", "wt") as handle:
            handle.write("@r\nACGT\n+\nIIII\n")
        assert accession_has_fastq(acc_dir) is True

    def test_false_when_sidecar_marks_partial(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        (acc_dir / "SRR1.json").write_text(json.dumps({"state": "partial"}))
        assert accession_has_fastq(acc_dir) is False

    def test_false_when_sidecar_marks_failed_or_downloading(self, tmp_path):
        for state in ("failed", "downloading"):
            acc_dir = tmp_path / f"SRR_{state}"
            acc_dir.mkdir()
            (acc_dir / f"{acc_dir.name}.fastq").write_text("@r\nACGT\n+\nIIII\n")
            (acc_dir / f"{acc_dir.name}.json").write_text(json.dumps({"state": state}))
            assert accession_has_fastq(acc_dir) is False

    def test_true_when_sidecar_marks_complete(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        (acc_dir / "SRR1.json").write_text(json.dumps({"state": "complete"}))
        assert accession_has_fastq(acc_dir) is True

    def test_true_when_sidecar_is_unreadable(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        (acc_dir / "SRR1.json").write_text("not valid json{")
        assert accession_has_fastq(acc_dir) is True

    def test_false_for_missing_directory(self, tmp_path):
        assert accession_has_fastq(tmp_path / "missing") is False


class TestCountFastqReads:
    """count_fastq_reads: a chunked binary newline count, // 4 per FASTQ record."""

    def test_plain_and_gz(self, tmp_path):
        plain = tmp_path / "a.fastq"
        plain.write_text("".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(3)))
        assert count_fastq_reads(plain) == 3

        gz = tmp_path / "b.fastq.gz"
        with gzip.open(gz, "wt") as handle:
            handle.write("".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(5)))
        assert count_fastq_reads(gz) == 5

    def test_multiple_internal_blocks(self, tmp_path):
        """A file over 1 MiB must still be counted correctly across chunk boundaries."""
        record = "@r{0}\n" + "A" * 32 + "\n+\n" + "I" * 32 + "\n"
        path = tmp_path / "big.fastq"
        with open(path, "w") as handle:
            for i in range(20000):
                handle.write(record.format(i))
        assert path.stat().st_size > 1024 * 1024
        assert count_fastq_reads(path) == 20000

    def test_missing_trailing_newline_plain(self, tmp_path):
        """A file whose last record has no trailing newline must not be undercounted."""
        content = "".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(2))
        content = content[:-1]  # drop the final newline
        path = tmp_path / "a.fastq"
        path.write_text(content)
        assert count_fastq_reads(path) == 2

    def test_missing_trailing_newline_gz(self, tmp_path):
        content = "".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(2))
        content = content[:-1]
        path = tmp_path / "a.fastq.gz"
        with gzip.open(path, "wt") as handle:
            handle.write(content)
        assert count_fastq_reads(path) == 2

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.fastq"
        path.write_text("")
        assert count_fastq_reads(path) == 0


class TestPrimaryAndOrphanFastq:
    """primary_fastq / orphan_fastq: which file in an accession folder holds the spots."""

    def _write(self, acc_dir, *names):
        acc_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (acc_dir / name).write_text("@r\nACGT\n+\nIIII\n")

    def test_primary_prefers_mate_one_over_bare_file(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        self._write(acc_dir, "SRR1.fastq", "SRR1_1.fastq", "SRR1_2.fastq")

        assert primary_fastq(acc_dir) == acc_dir / "SRR1_1.fastq"

    def test_primary_accepts_r1_naming(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        self._write(acc_dir, "SRR1_R1.fastq.gz", "SRR1_R2.fastq.gz")

        assert primary_fastq(acc_dir) == acc_dir / "SRR1_R1.fastq.gz"

    def test_primary_is_bare_file_for_single_end(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        self._write(acc_dir, "SRR1.fastq")

        assert primary_fastq(acc_dir) == acc_dir / "SRR1.fastq"

    def test_primary_none_for_empty_folder(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()

        assert primary_fastq(acc_dir) is None
        assert orphan_fastq(acc_dir) is None

    def test_orphan_only_when_mates_present(self, tmp_path):
        paired = tmp_path / "SRR1"
        self._write(paired, "SRR1.fastq", "SRR1_1.fastq", "SRR1_2.fastq")
        single = tmp_path / "SRR2"
        self._write(single, "SRR2.fastq")

        assert orphan_fastq(paired) == paired / "SRR1.fastq"
        assert orphan_fastq(single) is None


class TestVerifyDownload:
    """verify_download: reads_r1 against NCBI's expected spot count."""

    def _acc_dir(self, tmp_path, name="SRR1"):
        acc_dir = tmp_path / name
        acc_dir.mkdir()
        (acc_dir / f"{name}_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        return acc_dir

    def test_truncated(self, tmp_path, monkeypatch):
        acc_dir = self._acc_dir(tmp_path)
        monkeypatch.setattr("metaquest.data.sra.count_fastq_reads", lambda path: 300000)

        result = verify_download("SRR1", acc_dir, expected_spots=48000000)

        assert result["verdict"] == "truncated"
        assert result["reads_r1"] == 300000
        assert result["expected_spots"] == 48000000
        assert result["ratio"] == round(300000 / 48000000, 4)

    def test_complete(self, tmp_path, monkeypatch):
        acc_dir = self._acc_dir(tmp_path)
        monkeypatch.setattr("metaquest.data.sra.count_fastq_reads", lambda path: 999)

        result = verify_download("SRR1", acc_dir, expected_spots=1000)

        assert result["verdict"] == "complete"
        assert result["ratio"] == 0.999

    def test_complete_at_exact_threshold(self, tmp_path, monkeypatch):
        acc_dir = self._acc_dir(tmp_path)
        monkeypatch.setattr("metaquest.data.sra.count_fastq_reads", lambda path: 990)

        result = verify_download("SRR1", acc_dir, expected_spots=1000)

        assert result["verdict"] == "complete"

    def test_truncated_just_below_threshold(self, tmp_path, monkeypatch):
        acc_dir = self._acc_dir(tmp_path)
        monkeypatch.setattr("metaquest.data.sra.count_fastq_reads", lambda path: 989)

        result = verify_download("SRR1", acc_dir, expected_spots=1000)

        assert result["verdict"] == "truncated"

    def test_unverified_when_spots_unknown(self, tmp_path):
        acc_dir = self._acc_dir(tmp_path)

        result = verify_download("SRR1", acc_dir, expected_spots=None)

        assert result["verdict"] == "unverified"
        assert result["ratio"] is None
        assert result["expected_spots"] is None

    def test_counts_mate_one_plus_orphan_file(self, tmp_path):
        """--split-3 writes unpaired spots to a bare <acc>.fastq; those spots count too."""
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        record = "@r\nACGT\n+\nIIII\n"
        (acc_dir / "SRR1_1.fastq").write_text(record * 1000)
        (acc_dir / "SRR1_2.fastq").write_text(record * 1000)
        (acc_dir / "SRR1.fastq").write_text(record * 7)

        result = verify_download("SRR1", acc_dir, expected_spots=1007)

        assert result["reads_r1"] == 1007
        assert result["ratio"] == 1.0
        assert result["verdict"] == "complete"

    def test_single_end_counts_the_bare_file(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n" * 500)

        result = verify_download("SRR1", acc_dir, expected_spots=500)

        assert result["reads_r1"] == 500
        assert result["verdict"] == "complete"

    def test_no_expected_bytes_parameter(self):
        """NCBI's recorded size is the .sra archive size, so it never constrained the FASTQ bytes."""
        assert "expected_bytes" not in inspect.signature(verify_download).parameters

    def test_bytes_total_sums_fastq_files(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        (acc_dir / "SRR1_2.fastq").write_text("@r\nACGT\n+\nIIII\n")
        expected_bytes = sum(p.stat().st_size for p in acc_dir.glob("*.fastq"))

        result = verify_download("SRR1", acc_dir, expected_spots=None)

        assert result["bytes_total"] == expected_bytes


class TestCachedSraArchive:
    """_cached_sra_archive: which prefetch output fasterq-dump is pointed at."""

    def test_prefers_sra_over_sralite(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1.sralite").write_bytes(b"")
        (acc_dir / "SRR1.sra").write_bytes(b"")

        assert _cached_sra_archive(acc_dir, "SRR1") == acc_dir / "SRR1.sra"

    def test_finds_a_sralite_only_download(self, tmp_path):
        acc_dir = tmp_path / "SRR1"
        acc_dir.mkdir()
        (acc_dir / "SRR1.sralite").write_bytes(b"")

        assert _cached_sra_archive(acc_dir, "SRR1") == acc_dir / "SRR1.sralite"

    def test_falls_back_to_the_expected_name_when_nothing_was_written(self, tmp_path):
        """An empty cache folder yields the .sra path, so fasterq-dump reports the missing file."""
        assert _cached_sra_archive(tmp_path / "SRR1", "SRR1") == tmp_path / "SRR1" / "SRR1.sra"


class TestParseVerdictMessage:
    def test_complete(self):
        result = parse_verdict_message("Downloaded 2 files, complete (300000 of 300000 spots)")
        assert result == {"verdict": "complete", "reads_r1": 300000, "expected_spots": 300000, "ratio": 1.0}

    def test_truncated(self):
        result = parse_verdict_message("Downloaded 1 files, truncated (300000 of 48000000 spots)")
        assert result["verdict"] == "truncated"
        assert result["reads_r1"] == 300000
        assert result["expected_spots"] == 48000000

    def test_truncated_with_retry_prefix(self):
        result = parse_verdict_message("Retry 1: Downloaded 1 files, truncated (300000 of 48000000 spots)")
        assert result["verdict"] == "truncated"

    def test_unverified(self):
        assert parse_verdict_message("Downloaded 2 files, unverified") == {"verdict": "unverified"}

    def test_unrelated_message_returns_none(self):
        assert parse_verdict_message("Download failed: timeout") is None
        assert parse_verdict_message("already exists") is None

    def test_verdict_words_outside_the_message_format_are_ignored(self):
        """Only the ", <verdict>" text _handle_download_output writes counts as a verdict."""
        assert parse_verdict_message("Download failed: unverified checksum for SRR1") is None
        assert parse_verdict_message("not-found: could not complete (0 of 5 spots) for SRR1") is None


class TestHandleDownloadOutput:
    """Test _handle_download_output function."""

    def test_handle_download_output_success(self, tmp_path):
        """Test successful file moving."""
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output"
        temp_path.mkdir()

        # Create test FASTQ files
        (temp_path / "SRR123.fastq").write_text("@seq1\nACGT\n+\nIIII\n")
        (temp_path / "SRR123_2.fastq").write_text("@seq2\nTGCA\n+\nIIII\n")

        success, message = _handle_download_output(temp_path, output_path)

        assert success is True
        assert "2 files" in message
        assert (output_path / "SRR123.fastq").exists()
        assert (output_path / "SRR123_2.fastq").exists()
        assert not temp_path.exists()

    def test_handle_download_output_no_fastq_files(self, tmp_path):
        """No FASTQ files found: message is classified and the temp folder is kept for inspection."""
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output"
        temp_path.mkdir()

        # Create non-FASTQ file
        (temp_path / "other.txt").write_text("not a fastq")

        with patch("shutil.rmtree") as mock_rmtree:
            with patch("metaquest.data.sra.logger") as mock_logger:
                success, message = _handle_download_output(temp_path, output_path)

        assert success is False
        assert message == "unknown: No FASTQ files created"
        mock_logger.error.assert_called()
        mock_rmtree.assert_not_called()
        assert temp_path.exists()

    def test_handle_download_output_rmdir_error(self, tmp_path):
        """Test handling rmtree error during cleanup."""
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output"
        temp_path.mkdir()

        (temp_path / "SRR123.fastq").write_text("@seq1\nACGT\n+\nIIII\n")

        with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
            with patch("shutil.move"):
                with patch("metaquest.data.sra.logger") as mock_logger:
                    success, message = _handle_download_output(temp_path, output_path)

        assert success is True
        mock_logger.warning.assert_called()

    def test_handle_download_output_complete_verdict(self, tmp_path, monkeypatch):
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output" / "SRR123"
        temp_path.mkdir()
        (temp_path / "SRR123_1.fastq").write_text("@seq1\nACGT\n+\nIIII\n")
        (temp_path / "SRR123_2.fastq").write_text("@seq2\nTGCA\n+\nIIII\n")
        monkeypatch.setattr("metaquest.data.sra.count_fastq_reads", lambda path: 300000)

        success, message = _handle_download_output(temp_path, output_path, expected_spots=300000)

        assert success is True
        assert message == "Downloaded 2 files, complete (300000 of 300000 spots)"

    def test_handle_download_output_truncated_verdict_logs_warning(self, tmp_path, monkeypatch):
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output" / "SRR123"
        temp_path.mkdir()
        (temp_path / "SRR123_1.fastq").write_text("@seq1\nACGT\n+\nIIII\n")
        monkeypatch.setattr("metaquest.data.sra.count_fastq_reads", lambda path: 300000)

        with patch("metaquest.data.sra.logger") as mock_logger:
            success, message = _handle_download_output(temp_path, output_path, expected_spots=48000000)

        assert success is True
        assert message == "Downloaded 1 files, truncated (300000 of 48000000 spots)"
        mock_logger.warning.assert_called()

    def test_handle_download_output_unverified_without_expected_spots(self, tmp_path):
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output" / "SRR123"
        temp_path.mkdir()
        (temp_path / "SRR123_1.fastq").write_text("@seq1\nACGT\n+\nIIII\n")

        success, message = _handle_download_output(temp_path, output_path)

        assert success is True
        assert message == "Downloaded 1 files, unverified"

    def test_handle_download_output_reports_compression_failure_in_message(self, tmp_path):
        """A per-file compression failure must not fail the download, but must be visible."""
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output" / "SRR123"
        temp_path.mkdir()
        (temp_path / "SRR123_1.fastq").write_text("@seq1\nACGT\n+\nIIII\n")

        with patch("metaquest.data.sra.compress_fastq", side_effect=OSError("disk full")):
            with patch("metaquest.data.sra.logger") as mock_logger:
                success, message = _handle_download_output(temp_path, output_path, compress=True)

        assert success is True
        assert "compression failed for SRR123_1.fastq" in message
        # The verdict text itself (parseable by parse_verdict_message) must still be present.
        assert message.startswith("Downloaded 1 files, unverified")
        mock_logger.warning.assert_called()


class TestDownloadAccession:
    """Test download_accession function."""

    def test_download_accession_existing_skip(self, tmp_path):
        """Test skipping existing download."""
        output_folder = tmp_path / "downloads"
        output_path = output_folder / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("existing")

        success, message = download_accession("SRR123", output_folder, force=False)

        assert success is True
        assert "already exists" in message

    def test_download_accession_force_existing(self, tmp_path):
        """Test forcing download of existing accession."""
        output_folder = tmp_path / "downloads"
        output_path = output_folder / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("existing")

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                with patch("metaquest.data.sra._handle_download_output") as mock_handle:
                    mock_prep.return_value = tmp_path / "temp"
                    mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")
                    mock_handle.return_value = (True, "Downloaded 1 files")

                    success, message = download_accession("SRR123", output_folder, force=True)

        assert "Downloaded 1 files" in message

    def test_download_accession_security_error(self, tmp_path):
        """Test handling security error."""
        output_folder = tmp_path / "downloads"

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                mock_prep.return_value = tmp_path / "temp"
                mock_run.side_effect = SecurityError("Command blocked")

                success, message = download_accession("SRR123", output_folder)

        assert success is False
        assert "Security error" in message

    def test_download_accession_command_failure(self, tmp_path):
        """Test handling command failure."""
        import subprocess

        output_folder = tmp_path / "downloads"

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                mock_prep.return_value = tmp_path / "temp"
                mock_run.side_effect = subprocess.CalledProcessError(1, "fasterq-dump", stderr="Error occurred")

                success, message = download_accession("SRR123", output_folder)

        assert success is False
        assert "Download failed" in message

    def test_download_accession_success(self, tmp_path):
        """Test successful download."""
        output_folder = tmp_path / "downloads"
        temp_path = tmp_path / "temp"

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                with patch("metaquest.data.sra._handle_download_output") as mock_handle:
                    mock_prep.return_value = temp_path
                    mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")
                    mock_handle.return_value = (True, "Downloaded 2 files")

                    success, message = download_accession("SRR123", output_folder, num_threads=8)

        assert success is True
        assert "Downloaded 2 files" in message
        # Verify command was called with correct arguments
        mock_run.assert_called_once()
        executable = mock_run.call_args[0][0]
        args = mock_run.call_args[0][1]
        assert executable == "fasterq-dump"
        assert "SRR123" in args
        assert "--threads" in args
        assert "8" in args

    def test_download_accession_command_passes_validation(self, tmp_path, monkeypatch):
        """The command download_accession builds must survive SecureSubprocess validation unmocked.

        No prefetch on PATH in this test environment, so this exercises the direct
        fasterq-dump fallback with --split-3/--skip-technical appended.
        """
        monkeypatch.chdir(tmp_path)
        with patch("metaquest.utils.security.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            with patch("metaquest.data.sra._handle_download_output", return_value=(True, "Downloaded 2 files")):
                success, message = download_accession(
                    "SRR2517620", tmp_path / "fastq", num_threads=4, temp_folder=tmp_path / "tmp"
                )
        assert success is True, message
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "fasterq-dump"
        assert cmd[1:4] == ["--threads", "4", "--progress"]
        assert cmd[4] == "SRR2517620"
        assert "--split-3" in cmd
        assert "--skip-technical" in cmd

    def test_download_accession_registers_output_and_temp_roots(self, tmp_path, monkeypatch):
        from metaquest.utils.security import SecureSubprocess

        SecureSubprocess._extra_roots.clear()
        monkeypatch.chdir(tmp_path)
        with patch("metaquest.utils.security.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            with patch("metaquest.data.sra._handle_download_output", return_value=(True, "Downloaded 2 files")):
                download_accession("SRR2517620", tmp_path / "fastq", temp_folder=tmp_path / "scratch")
        assert (tmp_path / "fastq").resolve() in SecureSubprocess._extra_roots
        assert (tmp_path / "scratch").resolve() in SecureSubprocess._extra_roots
        SecureSubprocess._extra_roots.clear()

    def test_download_accession_passes_expected_spots_to_handle_output(self, tmp_path):
        """expected_spots reaches _handle_download_output so the verdict can be computed."""
        output_folder = tmp_path / "downloads"

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                with patch("metaquest.data.sra._handle_download_output") as mock_handle:
                    mock_prep.return_value = tmp_path / "temp"
                    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                    mock_handle.return_value = (True, "Downloaded 1 files, complete (10 of 10 spots)")

                    download_accession("SRR123", output_folder, expected_spots=10)

        mock_handle.assert_called_once()
        assert mock_handle.call_args.kwargs.get("expected_spots") == 10 or 10 in mock_handle.call_args.args

    def test_download_accession_redownload_truncated_forces_fresh_download(self, tmp_path):
        """A prior partial download on disk must not short-circuit as 'already exists'."""
        output_folder = tmp_path / "downloads"
        output_path = output_folder / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("partial")

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                with patch("metaquest.data.sra._handle_download_output") as mock_handle:
                    mock_prep.return_value = tmp_path / "temp"
                    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                    mock_handle.return_value = (True, "Downloaded 1 files, complete (10 of 10 spots)")

                    success, message = download_accession(
                        "SRR123", output_folder, force=False, redownload_truncated=True
                    )

        assert "already exists" not in message
        mock_run.assert_called_once()

    def test_download_accession_failure_keeps_temp_folder_for_resume(self, tmp_path):
        """A failed download must not wipe <acc>_temp; a later attempt resumes into it."""
        import subprocess

        output_folder = tmp_path / "downloads"

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                mock_prep.return_value = tmp_path / "temp"
                mock_run.side_effect = subprocess.CalledProcessError(1, "fasterq-dump", stderr="Connection timed out")

                success, message = download_accession("SRR123", output_folder)

        assert success is False
        temp_path = output_folder / "SRR123_temp"
        assert temp_path.exists()

    def test_download_accession_command_failure_classifies_network_error(self, tmp_path):
        import subprocess

        output_folder = tmp_path / "downloads"

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                mock_prep.return_value = tmp_path / "temp"
                mock_run.side_effect = subprocess.CalledProcessError(1, "fasterq-dump", stderr="Connection timed out")

                success, message = download_accession("SRR123", output_folder)

        assert success is False
        assert message.startswith("network:")

    def test_download_accession_security_error_keeps_temp_folder(self, tmp_path):
        output_folder = tmp_path / "downloads"

        with patch("metaquest.data.sra._prepare_temp_folder") as mock_prep:
            with patch("metaquest.utils.security.SecureSubprocess.run_secure") as mock_run:
                mock_prep.return_value = tmp_path / "temp"
                mock_run.side_effect = SecurityError("Command blocked")

                success, message = download_accession("SRR123", output_folder)

        assert success is False
        temp_path = output_folder / "SRR123_temp"
        assert temp_path.exists()
        assert message.startswith("unknown:")

    def test_download_accession_prefetch_then_fasterq_dump_sequence(self, tmp_path):
        """With prefetch on PATH: prefetch runs first, then fasterq-dump --split-3 on the .sra file."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4}

        with patch("metaquest.data.sra.shutil.which", side_effect=lambda tool: f"/usr/bin/{tool}"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=False)

        assert success is True, message
        executables = [c[0] for c in state["calls"]]
        assert executables == ["prefetch", "fasterq-dump"]

        prefetch_args = state["calls"][0][1]
        assert prefetch_args[:2] == ["-O", str(output_folder / ".sra-cache")]
        assert "--max-size" in prefetch_args and "100G" in prefetch_args
        assert "--progress" in prefetch_args
        assert prefetch_args[-1] == "SRR123"

        dump_args = state["calls"][1][1]
        assert dump_args[0] == "--split-3"
        assert dump_args[1] == "--skip-technical"
        assert dump_args[-1] == str(output_folder / ".sra-cache" / "SRR123" / "SRR123.sra")

        assert accession_has_fastq(output_folder / "SRR123")
        # The .sra archive is removed after a successful, verified download.
        assert not (output_folder / ".sra-cache" / "SRR123").exists()

    def test_download_accession_keep_sra_preserves_cache(self, tmp_path):
        """keep_sra=True leaves the downloaded .sra archive in place after success."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4}

        with patch("metaquest.data.sra.shutil.which", side_effect=lambda tool: f"/usr/bin/{tool}"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=False, keep_sra=True)

        assert success is True, message
        assert (output_folder / ".sra-cache" / "SRR123" / "SRR123.sra").exists()

    def test_download_accession_force_clears_cached_archive_before_prefetch(self, tmp_path):
        """A forced download refetches the .sra instead of reusing a possibly truncated one."""
        output_folder = tmp_path / "downloads"
        cache_dir = output_folder / ".sra-cache" / "SRR123"
        cache_dir.mkdir(parents=True)
        (cache_dir / "SRR123.sra").write_bytes(b"stale archive")
        acc_dir = output_folder / "SRR123"
        acc_dir.mkdir(parents=True)
        (acc_dir / "SRR123_1.fastq").write_text("@r\nACGT\n+\nIIII\n")

        fake = _fake_tools({"reads": 4})
        cache_present_at_prefetch = []

        def run(executable, args, **kwargs):
            if executable == "prefetch":
                cache_present_at_prefetch.append(cache_dir.exists())
            return fake(executable, args, **kwargs)

        with patch("metaquest.data.sra.shutil.which", side_effect=lambda tool: f"/usr/bin/{tool}"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=run):
                success, message = download_accession("SRR123", output_folder, compress=False, force=True)

        assert success is True, message
        assert cache_present_at_prefetch == [False]

    def test_download_accession_truncated_removes_cached_archive(self, tmp_path):
        """A truncated verdict leaves no .sra behind, so the next attempt fetches it again."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4}

        with patch("metaquest.data.sra.shutil.which", side_effect=lambda tool: f"/usr/bin/{tool}"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=False, expected_spots=1000)

        assert success is True, message
        assert "truncated" in message
        assert not (output_folder / ".sra-cache" / "SRR123").exists()

    def test_download_accession_falls_back_without_prefetch_binary(self, tmp_path):
        """shutil.which('prefetch') is None: fall back to the direct fasterq-dump call."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4}

        with patch("metaquest.data.sra.shutil.which", return_value=None):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=False)

        assert success is True, message
        executables = [c[0] for c in state["calls"]]
        assert executables == ["fasterq-dump"]

        dump_args = state["calls"][0][1]
        assert "SRR123" in dump_args
        assert "--split-3" in dump_args
        assert "--skip-technical" in dump_args
        # No cache directory was ever created.
        assert not (output_folder / ".sra-cache").exists()
        assert accession_has_fastq(output_folder / "SRR123")

    def test_download_accession_logs_the_direct_call_when_prefetch_is_missing(self, tmp_path, caplog):
        """The fallback changes where the data comes from, so it is recorded at info level."""
        state = {"reads": 4}

        with patch("metaquest.data.sra.shutil.which", return_value=None):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                with caplog.at_level("INFO", logger="metaquest.data.sra"):
                    success, message = download_accession("SRR123", tmp_path / "downloads", compress=False)

        assert success is True, message
        assert "prefetch not found on PATH" in caplog.text

    def test_download_accession_dumps_a_sralite_archive(self, tmp_path):
        """Some runs are served only as .sralite; fasterq-dump must be pointed at that file."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4, "sralite": True}

        with patch("metaquest.data.sra.shutil.which", side_effect=lambda tool: f"/usr/bin/{tool}"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=False)

        assert success is True, message
        dump_args = state["calls"][1][1]
        assert dump_args[-1] == str(output_folder / ".sra-cache" / "SRR123" / "SRR123.sralite")
        assert accession_has_fastq(output_folder / "SRR123")

    def test_download_accession_use_prefetch_false_skips_prefetch(self, tmp_path):
        """use_prefetch=False falls back to the direct call even when prefetch is on PATH."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4}

        with patch("metaquest.data.sra.shutil.which", side_effect=lambda tool: f"/usr/bin/{tool}"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=False, use_prefetch=False)

        assert success is True, message
        executables = [c[0] for c in state["calls"]]
        assert executables == ["fasterq-dump"]

    def test_download_accession_single_end_output(self, tmp_path):
        """fasterq-dump --split-3 on a single-end run writes <acc>.fastq; that must be recognized."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4, "single": True}

        with patch("metaquest.data.sra.shutil.which", return_value=None):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=False)

        assert success is True, message
        assert (output_folder / "SRR123" / "SRR123.fastq").exists()
        assert accession_has_fastq(output_folder / "SRR123")

    def test_download_accession_compresses_with_python_gzip_fallback(self, tmp_path):
        """No pigz on PATH: compression falls back to Python gzip; read count survives."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4}

        with patch("metaquest.data.sra.shutil.which", return_value=None):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=True)

        assert success is True, message
        acc_dir = output_folder / "SRR123"
        r1 = acc_dir / "SRR123_1.fastq.gz"
        r2 = acc_dir / "SRR123_2.fastq.gz"
        assert r1.exists() and r2.exists()
        assert not (acc_dir / "SRR123_1.fastq").exists()
        assert count_fastq_reads(r1) == 4
        assert accession_has_fastq(acc_dir)

    def test_download_accession_compresses_with_pigz_when_available(self, tmp_path):
        """pigz on PATH: compress_fastq shells out to pigz -p N -f <file> instead of Python gzip."""
        output_folder = tmp_path / "downloads"
        state = {"reads": 4}

        def which(tool):
            return None if tool == "prefetch" else f"/usr/bin/{tool}"

        with patch("metaquest.data.sra.shutil.which", side_effect=which):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)):
                success, message = download_accession("SRR123", output_folder, compress=True, num_threads=4)

        assert success is True, message
        pigz_calls = [c for c in state["calls"] if c[0] == "pigz"]
        assert len(pigz_calls) == 2
        for _, args in pigz_calls:
            assert args[0] == "-p"
            assert args[1] == "4"
            assert "-f" in args

        acc_dir = output_folder / "SRR123"
        assert (acc_dir / "SRR123_1.fastq.gz").exists()
        assert (acc_dir / "SRR123_2.fastq.gz").exists()


class TestCompressFastq:
    """Test the module-level compress_fastq helper directly."""

    def test_compress_fastq_python_fallback(self, tmp_path):
        path = tmp_path / "SRR1_1.fastq"
        path.write_text("@r\nACGT\n+\nIIII\n" * 3)

        with patch("metaquest.data.sra.shutil.which", return_value=None):
            result = compress_fastq(path, threads=4)

        assert result == tmp_path / "SRR1_1.fastq.gz"
        assert result.exists()
        assert not path.exists()
        assert count_fastq_reads(result) == 3

    def test_compress_fastq_uses_pigz_when_available(self, tmp_path):
        path = tmp_path / "SRR1_1.fastq"
        path.write_text("@r\nACGT\n+\nIIII\n")
        state = {}

        with patch("metaquest.data.sra.shutil.which", return_value="/usr/bin/pigz"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=_fake_tools(state)) as mock_run:
                result = compress_fastq(path, threads=8)

        assert result == tmp_path / "SRR1_1.fastq.gz"
        mock_run.assert_called_once()
        executable, args = mock_run.call_args[0][0], mock_run.call_args[0][1]
        assert executable == "pigz"
        assert args == ["-p", "8", "-f", str(path)]

    def test_compress_fastq_python_fallback_atomic_on_mid_write_failure(self, tmp_path):
        """A write failure partway through must not leave a truncated .gz beside the plain file."""
        path = tmp_path / "SRR1_1.fastq"
        path.write_text("@r\nACGT\n+\nIIII\n" * 3)

        real_gzip_open = gzip.open
        write_calls = {"n": 0}

        def flaky_open(*args, **kwargs):
            handle = real_gzip_open(*args, **kwargs)
            real_write = handle.write

            def flaky_write(data):
                write_calls["n"] += 1
                if write_calls["n"] == 1:
                    raise OSError("disk full")
                return real_write(data)

            handle.write = flaky_write
            return handle

        with patch("metaquest.data.sra.shutil.which", return_value=None):
            with patch("metaquest.data.sra.gzip.open", side_effect=flaky_open):
                with pytest.raises(OSError):
                    compress_fastq(path, threads=4)

        # The plain file survives untouched, no partial .gz and no leftover temp file.
        assert path.exists()
        assert fastq_files(tmp_path) == [path]
        assert not (tmp_path / "SRR1_1.fastq.gz").exists()
        assert not any(p.name.startswith("SRR1_1.fastq.gz.tmp.") for p in tmp_path.iterdir())

    def test_compress_fastq_pigz_failure_leaves_no_partial_gz(self, tmp_path):
        """If pigz (via run_secure) raises, any partial .gz it left behind is removed."""
        path = tmp_path / "SRR1_1.fastq"
        path.write_text("@r\nACGT\n+\nIIII\n")
        target = tmp_path / "SRR1_1.fastq.gz"

        def failing_run(executable, args, **kwargs):
            # Simulate pigz having written a partial .gz before dying mid-compression.
            target.write_bytes(b"partial-garbage")
            raise SecurityError("pigz failed")

        with patch("metaquest.data.sra.shutil.which", return_value="/usr/bin/pigz"):
            with patch("metaquest.data.sra.SecureSubprocess.run_secure", side_effect=failing_run):
                with pytest.raises(SecurityError):
                    compress_fastq(path, threads=4)

        assert path.exists()
        assert not target.exists()


class TestClassifyDownloadError:
    """Test classify_download_error pure helper."""

    @pytest.mark.parametrize(
        "text,expected_class",
        [
            ("Connection timed out", "network"),
            ("curl: (7) Failed to connect to host", "network"),
            ("Could not resolve host ftp.ncbi.nlm.nih.gov", "network"),
            ("SSL handshake failed", "network"),
            ("network is unreachable", "network"),
            ("No space left on device", "disk-full"),
            ("write failed: ENOSPC", "disk-full"),
            ("Disk full while writing output", "disk-full"),
            ("SRR000000 not found", "not-found"),
            ("Invalid accession format", "not-found"),
            ("403 Forbidden", "not-found"),
            ("404 Not Found", "not-found"),
            ("accession does not exist", "not-found"),
            ("Some completely unrelated failure", "unknown"),
            ("", "unknown"),
            # Messages as sra-tools actually prints them.
            (
                "prefetch.3.1.1 err: name not found while resolving query within virtual file "
                "system module - failed to resolve accession 'SRR1' - no data ( 404 )",
                "not-found",
            ),
            (
                "fasterq-dump.3.1.1 err: storage exhausted while writing file within file system module",
                "disk-full",
            ),
            ("fasterq-dump.3.1.1 fatal: disk-limit exeeded!", "disk-full"),
            # A class prefix classifies back to its own class, so a message that has already
            # been prefixed once is not reclassified into a different class on the retry path.
            ("disk-full: not attempted", "disk-full"),
            ("not-found: Download failed: no data", "not-found"),
            ("network: Download failed: connection reset by peer", "network"),
            ("unknown: Download failed: something else entirely", "unknown"),
        ],
    )
    def test_classify_download_error_table(self, text, expected_class):
        assert classify_download_error(text) == expected_class

    def test_classify_download_error_case_insensitive(self):
        assert classify_download_error("CONNECTION TIMED OUT") == "network"
        assert classify_download_error("NO SPACE LEFT") == "disk-full"
        assert classify_download_error("NOT FOUND") == "not-found"


class TestDefaultMaxWorkers:
    """Test default_max_workers pure helper."""

    def test_default_max_workers_scales_with_cpu_and_threads(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        assert default_max_workers(4) == 2

    def test_default_max_workers_capped_by_default_max_workers_constant(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 64)
        assert default_max_workers(1) == 4

    def test_default_max_workers_capped_by_max_concurrent_downloads(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert default_max_workers(1) <= 10

    def test_default_max_workers_at_least_one(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 2)
        assert default_max_workers(16) == 1


class TestIsTransientFolder:
    """Test is_transient_folder pure helper."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("SRR123_temp", True),
            ("SRR000000_temp", True),
            (".sra-cache", True),
            ("SRR123", False),
            ("SRR123_temporary", False),
            ("temp", False),
            ("SRR123.sra-cache", False),
        ],
    )
    def test_is_transient_folder(self, name, expected):
        assert is_transient_folder(name) == expected


class TestTransientBytes:
    """Test transient_bytes: total size of .sra-cache and <ACC>_temp folders under one folder."""

    def test_missing_folder_is_zero(self, tmp_path):
        assert transient_bytes(tmp_path / "does-not-exist") == 0

    def test_empty_folder_is_zero(self, tmp_path):
        assert transient_bytes(tmp_path) == 0

    def test_sums_sra_cache_contents(self, tmp_path):
        cache = tmp_path / ".sra-cache"
        cache.mkdir()
        (cache / "SRR1.sra").write_bytes(b"x" * 100)
        (cache / "SRR2.sra").write_bytes(b"x" * 50)
        assert transient_bytes(tmp_path) == 150

    def test_sums_acc_temp_contents(self, tmp_path):
        temp = tmp_path / "SRR1_temp"
        temp.mkdir()
        (temp / "SRR1.fastq").write_bytes(b"x" * 200)
        assert transient_bytes(tmp_path) == 200

    def test_sums_across_multiple_transient_folders(self, tmp_path):
        cache = tmp_path / ".sra-cache"
        cache.mkdir()
        (cache / "SRR1.sra").write_bytes(b"x" * 100)
        temp = tmp_path / "SRR2_temp"
        temp.mkdir()
        (temp / "SRR2.fastq").write_bytes(b"x" * 200)
        assert transient_bytes(tmp_path) == 300

    def test_ignores_non_transient_entries(self, tmp_path):
        real = tmp_path / "SRR1"
        real.mkdir()
        (real / "SRR1_1.fastq").write_bytes(b"x" * 999)
        assert transient_bytes(tmp_path) == 0

    def test_recurses_into_subdirectories(self, tmp_path):
        cache = tmp_path / ".sra-cache"
        nested = cache / "nested"
        nested.mkdir(parents=True)
        (nested / "SRR1.sra").write_bytes(b"x" * 42)
        assert transient_bytes(tmp_path) == 42


class TestCheckExistingDownloads:
    """Test _check_existing_downloads function."""

    def test_check_existing_downloads_none_exist(self, tmp_path):
        """Test when no downloads exist."""
        output_folder = tmp_path / "downloads"
        accessions = ["SRR123", "SRR456"]

        already_downloaded, to_download, blacklisted = _check_existing_downloads(accessions, output_folder, force=False)

        assert to_download == accessions
        assert already_downloaded == []

    def test_check_existing_downloads_some_exist(self, tmp_path):
        """Test when some downloads exist."""
        output_folder = tmp_path / "downloads"
        (output_folder / "SRR123").mkdir(parents=True)
        (output_folder / "SRR123" / "SRR123.fastq").write_text("data")

        accessions = ["SRR123", "SRR456"]

        already_downloaded, to_download, blacklisted = _check_existing_downloads(accessions, output_folder, force=False)

        assert to_download == ["SRR456"]
        assert already_downloaded == ["SRR123"]

    def test_check_existing_downloads_force_all(self, tmp_path):
        """Test with force=True."""
        output_folder = tmp_path / "downloads"
        (output_folder / "SRR123").mkdir(parents=True)
        (output_folder / "SRR123" / "SRR123.fastq").write_text("data")

        accessions = ["SRR123", "SRR456"]

        with patch("shutil.rmtree"):
            already_downloaded, to_download, blacklisted = _check_existing_downloads(
                accessions, output_folder, force=True
            )

        assert to_download == accessions
        assert already_downloaded == []

    def test_check_existing_downloads_truncated_accession_is_not_already_downloaded(self, tmp_path):
        """An accession with a 'truncated' registry verdict must be redownloaded, not skipped."""
        output_folder = tmp_path / "downloads"
        (output_folder / "SRR123").mkdir(parents=True)
        (output_folder / "SRR123" / "SRR123.fastq").write_text("partial")

        accessions = ["SRR123", "SRR456"]

        already_downloaded, to_download, blacklisted = _check_existing_downloads(
            accessions, output_folder, force=False, truncated_accessions={"SRR123"}
        )

        assert to_download == ["SRR123", "SRR456"]
        assert already_downloaded == []


class TestProcessDownloadResults:
    """Test _process_download_results function."""

    def test_process_download_results_all_success(self):
        """Test processing all successful results."""
        # Create mock futures
        future1 = Mock()
        future1.result.return_value = (True, "Success message 1")
        future2 = Mock()
        future2.result.return_value = (True, "Success message 2")

        futures_results = [("SRR123", (True, "Success message 1")), ("SRR456", (True, "Success message 2"))]
        accessions_to_download = ["SRR123", "SRR456"]
        download_results = {}
        failed_accessions = []

        _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions)

        assert download_results == {"SRR123": "Success message 1", "SRR456": "Success message 2"}
        assert failed_accessions == []

    def test_process_download_results_some_failures(self):
        """Test processing mixed success/failure results."""
        future1 = Mock()
        future1.result.return_value = (True, "Success message")
        future2 = Mock()
        future2.result.return_value = (False, "Error message")

        futures_results = [("SRR123", (True, "Success message")), ("SRR456", (False, "Error message"))]
        accessions_to_download = ["SRR123", "SRR456"]
        download_results = {}
        failed_accessions = []

        _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions)

        assert download_results == {"SRR123": "Success message", "SRR456": "Error message"}
        assert failed_accessions == ["SRR456"]

    def test_process_download_results_exception(self):
        """Test handling exceptions in futures."""
        future1 = Mock()
        future1.result.return_value = (True, "Success message")
        future2 = Mock()
        future2.result.side_effect = Exception("Future failed")

        futures_results = [("SRR123", (True, "Success message")), ("SRR456", Exception("Future failed"))]
        accessions_to_download = ["SRR123", "SRR456"]
        download_results = {}
        failed_accessions = []

        with patch("metaquest.data.sra.logger") as mock_logger:
            _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions)

        assert download_results == {
            "SRR123": "Success message",
            "SRR456": "Error: cannot unpack non-iterable Exception object",
        }
        assert failed_accessions == ["SRR456"]
        mock_logger.error.assert_called()


class TestRetryFailedDownloads:
    """Test _retry_failed_downloads function."""

    def test_retry_failed_downloads_success(self, tmp_path):
        """Test successful retry of failed downloads."""
        failed_accessions = ["SRR123", "SRR456"]
        download_results = {}

        with patch("metaquest.data.sra.download_accession") as mock_download:
            mock_download.side_effect = [(True, "Retry success 1"), (True, "Retry success 2")]

            retried_successful, updated_failed, abort_reason = _retry_failed_downloads(
                failed_accessions,
                max_retries=2,
                fastq_path=tmp_path,
                num_threads=4,
                temp_folder=None,
                download_results=download_results,
            )

        assert download_results == {"SRR123": "Retry 1: Retry success 1", "SRR456": "Retry 1: Retry success 2"}
        assert len(updated_failed) == 0
        assert retried_successful == 2
        assert abort_reason is None

    def test_retry_failed_downloads_uses_force_false(self, tmp_path):
        """A retry must not wipe a partially-downloaded temp folder, so it must pass force=False."""
        failed_accessions = ["SRR123"]
        download_results = {}

        with patch("metaquest.data.sra.download_accession") as mock_download:
            mock_download.return_value = (True, "Retry success")

            _retry_failed_downloads(
                failed_accessions,
                max_retries=1,
                fastq_path=tmp_path,
                num_threads=4,
                temp_folder=None,
                download_results=download_results,
            )

        mock_download.assert_called_once()
        assert mock_download.call_args.kwargs.get("force") is False

    def test_retry_failed_downloads_partial_success(self, tmp_path):
        """Test partial success in retry."""
        failed_accessions = ["SRR123", "SRR456"]
        download_results = {}

        with patch("metaquest.data.sra.download_accession") as mock_download:
            with patch("metaquest.data.sra.time.sleep") as mock_sleep:
                mock_download.side_effect = [(True, "Retry success"), (False, "Retry failed")]

                retried_successful, updated_failed, abort_reason = _retry_failed_downloads(
                    failed_accessions,
                    max_retries=2,
                    fastq_path=tmp_path,
                    num_threads=4,
                    temp_folder=None,
                    download_results=download_results,
                )

        assert download_results == {"SRR123": "Retry 1: Retry success", "SRR456": "Retry 2 error: "}
        assert updated_failed == ["SRR456"]
        assert retried_successful == 1
        assert abort_reason is None
        mock_sleep.assert_called_once_with(1)

    def test_retry_failed_downloads_skips_not_found(self, tmp_path):
        """An accession already classified as not-found must not be retried again."""
        failed_accessions = ["SRR123", "SRR404"]
        download_results = {"SRR123": "network: Connection timed out", "SRR404": "not-found: 404 Not Found"}

        with patch("metaquest.data.sra.download_accession") as mock_download:
            mock_download.return_value = (True, "Retry success")

            retried_successful, updated_failed, abort_reason = _retry_failed_downloads(
                failed_accessions,
                max_retries=1,
                fastq_path=tmp_path,
                num_threads=4,
                temp_folder=None,
                download_results=download_results,
            )

        mock_download.assert_called_once_with(
            "SRR123",
            tmp_path,
            4,
            force=False,
            temp_folder=None,
            expected_spots=None,
            redownload_truncated=False,
            sra_cache=None,
            use_prefetch=True,
            keep_sra=False,
            compress=True,
        )
        assert updated_failed == ["SRR404"]
        assert retried_successful == 1
        assert abort_reason is None

    def test_retry_failed_downloads_sleeps_exponentially_between_rounds(self, tmp_path):
        """Between retry rounds, sleep 2**attempt seconds so a flaky network gets a backoff."""
        failed_accessions = ["SRR123"]
        download_results = {}

        with patch("metaquest.data.sra.download_accession") as mock_download:
            with patch("metaquest.data.sra.time.sleep") as mock_sleep:
                mock_download.return_value = (False, "network: Connection timed out")

                _retry_failed_downloads(
                    failed_accessions,
                    max_retries=3,
                    fastq_path=tmp_path,
                    num_threads=4,
                    temp_folder=None,
                    download_results=download_results,
                )

        assert mock_sleep.call_args_list == [call(1), call(2)]

    def test_retry_failed_downloads_disk_full_aborts_without_raising(self, tmp_path):
        """A disk-full failure aborts the run but must not raise, so the caller keeps its state."""
        failed_accessions = ["SRR1", "SRR2", "SRR3"]
        download_results = {}
        notified = []

        def _on_result(accession, success, message):
            notified.append((accession, success, message))

        with patch("metaquest.data.sra.download_accession") as mock_download:
            with patch("metaquest.data.sra.time.sleep") as mock_sleep:
                mock_download.return_value = (False, "disk-full: No space left on device")

                retried_successful, updated_failed, abort_reason = _retry_failed_downloads(
                    failed_accessions,
                    max_retries=2,
                    fastq_path=tmp_path,
                    num_threads=4,
                    temp_folder=None,
                    download_results=download_results,
                    on_result=_on_result,
                )

        # Only the triggering accession is actually attempted; the rest of the round is
        # marked failed without ever calling download_accession.
        mock_download.assert_called_once()
        assert retried_successful == 0
        assert abort_reason == "disk-full"
        assert updated_failed == ["SRR1", "SRR2", "SRR3"]
        assert download_results["SRR1"] == "Retry 1: disk-full: No space left on device"
        assert download_results["SRR2"] == "disk-full: not attempted"
        assert download_results["SRR3"] == "disk-full: not attempted"
        # Every accession in the round is still notified, including the ones never attempted.
        assert notified == [
            ("SRR1", False, "Retry 1: disk-full: No space left on device"),
            ("SRR2", False, "disk-full: not attempted"),
            ("SRR3", False, "disk-full: not attempted"),
        ]
        # No further retry round runs, so no backoff sleep either.
        mock_sleep.assert_not_called()


class TestHandleDownloadFailure:
    """Test _handle_download_failure function."""

    def test_handle_download_failure_no_failures(self, tmp_path):
        """Test when no failures occurred."""
        fastq_path = tmp_path / "downloads"
        failed_accessions = []

        with patch("metaquest.data.sra.logger") as mock_logger:
            _handle_download_failure(fastq_path, failed_accessions)

        # No logging should occur when there are no failures
        mock_logger.info.assert_not_called()

    def test_handle_download_failure_with_failures(self, tmp_path):
        """Test when failures occurred."""
        fastq_path = tmp_path / "downloads"
        fastq_path.mkdir(parents=True, exist_ok=True)
        failed_accessions = ["SRR123", "SRR456"]

        with patch("metaquest.data.sra.logger") as mock_logger:
            _handle_download_failure(fastq_path, failed_accessions)

        mock_logger.info.assert_called()
        # Check that failed accessions file was created
        failed_file = fastq_path / "failed_accessions.txt"
        assert failed_file.exists()
        content = failed_file.read_text()
        assert "SRR123" in content
        assert "SRR456" in content


class TestDownloadSra:
    """Test download_sra function."""

    def test_download_sra_from_file(self, tmp_path):
        """Test downloading SRA data from accessions file."""
        accessions_file = tmp_path / "accessions.txt"
        output_folder = tmp_path / "downloads"
        accessions_file.write_text("SRR123\nSRR456\n")

        with patch("metaquest.data.sra._check_existing_downloads") as mock_check:
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
                with patch("metaquest.data.sra._process_download_results") as mock_process:
                    with patch("metaquest.data.sra._handle_download_failure"):
                        mock_check.return_value = ([], ["SRR123", "SRR456"], [])
                        mock_executor_instance = Mock()
                        mock_executor.return_value.__enter__.return_value = mock_executor_instance
                        mock_executor_instance.submit.return_value = Mock()
                        mock_process.return_value = (2, 0)

                        result = download_sra(output_folder, accessions_file, num_threads=4, max_workers=2)

        assert isinstance(result, dict)
        mock_check.assert_called_once()

    def test_download_sra_propagates_abort_reason(self, tmp_path):
        """A disk-full abort reported by the retry pass surfaces as stats['aborted']."""
        accessions_file = tmp_path / "accessions.txt"
        output_folder = tmp_path / "downloads"
        accessions_file.write_text("SRR123\n")

        with patch("metaquest.data.sra._check_existing_downloads") as mock_check:
            with patch("metaquest.data.sra._download_with_retries") as mock_retries:
                with patch("metaquest.data.sra._handle_download_failure"):
                    mock_check.return_value = ([], ["SRR123"], [])
                    mock_retries.return_value = (
                        0,
                        1,
                        ["SRR123"],
                        {"SRR123": "disk-full: not attempted"},
                        "disk-full",
                    )

                    result = download_sra(output_folder, accessions_file)

        assert result["aborted"] == "disk-full"
        assert result["failed"] == 1

    def test_download_sra_aborted_is_none_when_run_completes_normally(self, tmp_path):
        accessions_file = tmp_path / "accessions.txt"
        output_folder = tmp_path / "downloads"
        accessions_file.write_text("SRR123\n")

        with patch("metaquest.data.sra._check_existing_downloads") as mock_check:
            with patch("metaquest.data.sra._download_with_retries") as mock_retries:
                with patch("metaquest.data.sra._handle_download_failure"):
                    mock_check.return_value = ([], ["SRR123"], [])
                    mock_retries.return_value = (1, 0, [], {"SRR123": "Downloaded 1 files"}, None)

                    result = download_sra(output_folder, accessions_file)

        assert result["aborted"] is None

    def test_download_sra_with_blacklist(self, tmp_path):
        """Test downloading with blacklist filtering."""
        accessions_file = tmp_path / "accessions.txt"
        blacklist_file = tmp_path / "blacklist.txt"
        output_folder = tmp_path / "downloads"

        accessions_file.write_text("SRR123\nSRR456\nSRR789\n")
        blacklist_file.write_text("SRR456\n")

        with patch("metaquest.data.sra._check_existing_downloads") as mock_check:
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
                with patch("metaquest.data.sra._process_download_results") as mock_process:
                    with patch("metaquest.data.sra._handle_download_failure"):
                        mock_check.return_value = ([], ["SRR123", "SRR789"], [])
                        mock_executor_instance = Mock()
                        mock_executor.return_value.__enter__.return_value = mock_executor_instance
                        mock_process.return_value = (2, 0)

                        download_sra(output_folder, accessions_file, blacklist=[blacklist_file])

        # Verify that blacklist filtering works correctly
        check_call_args = mock_check.call_args
        all_accessions = check_call_args[0][0]  # First positional argument
        blacklisted_accessions = check_call_args[0][3]  # Fourth positional argument (set of blacklisted)

        # All accessions should be passed to the function
        assert "SRR123" in all_accessions
        assert "SRR456" in all_accessions
        assert "SRR789" in all_accessions
        # But SRR456 should be in the blacklist set
        assert "SRR456" in blacklisted_accessions

    def test_download_sra_nonexistent_file(self, tmp_path):
        """Test error handling for nonexistent accessions file."""
        nonexistent_file = tmp_path / "nonexistent.txt"
        output_folder = tmp_path / "downloads"

        with pytest.raises(DataAccessError):
            download_sra(output_folder, nonexistent_file)

    def test_download_sra_dry_run(self, tmp_path):
        """Test dry run mode."""
        accessions_file = tmp_path / "accessions.txt"
        output_folder = tmp_path / "downloads"
        accessions_file.write_text("SRR123\nSRR456\n")

        with patch("metaquest.data.sra._check_existing_downloads") as mock_check:
            with patch("metaquest.data.sra.logger") as mock_logger:
                mock_check.return_value = ([], ["SRR123", "SRR456"], [])
                result = download_sra(output_folder, accessions_file, dry_run=True)

        expected_result = {
            "total": 2,
            "already_downloaded": 0,
            "blacklisted": 0,
            "to_download": 2,
            "successful": 0,
            "failed": 0,
            "already_downloaded_accessions": [],
            "blacklisted_accessions": [],
            "skipped_accessions": [],
            "aborted": None,
        }
        assert result == expected_result
        mock_logger.info.assert_called_with("Dry run: would download 2 accessions")

    def test_download_sra_returns_accession_lists(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        (tmp_path / "fastq" / "SRR2").mkdir(parents=True)
        (tmp_path / "fastq" / "SRR2" / "SRR2_1.fastq").write_text("@r\nA\n+\nI\n")
        black = tmp_path / "black.txt"
        black.write_text("SRR1\n")
        stats = download_sra(tmp_path / "fastq", acc, dry_run=True, blacklist=[black])
        assert stats["already_downloaded_accessions"] == ["SRR2"]
        assert stats["blacklisted_accessions"] == ["SRR1"]

    def test_on_result_called_on_main_thread_per_accession(self, tmp_path, monkeypatch):
        import threading

        seen = []
        main = threading.get_ident()

        def on_result(acc, ok, message):
            seen.append((acc, ok, threading.get_ident() == main))

        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        with patch(
            "metaquest.data.sra.download_accession",
            side_effect=[(True, "Downloaded 2 files"), (False, "Download failed: x")],
        ):
            stats = download_sra(tmp_path / "fastq", acc, max_workers=2, max_retries=0, on_result=on_result)
        assert sorted(a for a, _, _ in seen) == ["SRR1", "SRR2"] and all(on_main for _, _, on_main in seen)
        assert stats["failed_accessions"] == ["SRR2"] or stats["failed_accessions"] == ["SRR1"]

    def test_on_result_failure_does_not_corrupt_the_tally(self, tmp_path):
        """A raising on_result (e.g. a registry lock timeout) must not be mistaken for a download failure."""
        calls = []

        def flaky_on_result(acc, ok, message):
            calls.append(acc)
            if len(calls) == 1:
                raise RuntimeError("registry locked")

        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        with patch("metaquest.data.sra.download_accession", return_value=(True, "Downloaded 2 files")):
            stats = download_sra(tmp_path / "fastq", acc, max_retries=0, on_result=flaky_on_result)
        assert stats["successful"] == 2
        assert stats["failed"] == 0
        assert stats["failed_accessions"] == []
        assert len(calls) == 2

    def test_max_downloads_cutoffs_are_returned(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\nSRR3\n")
        with patch("metaquest.data.sra.download_accession", return_value=(True, "Downloaded 2 files")):
            stats = download_sra(tmp_path / "fastq", acc, max_downloads=1)
        assert len(stats["skipped_accessions"]) == 2

    def test_registry_exclusions_join_the_blacklist(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        stats = download_sra(tmp_path / "fastq", acc, dry_run=True, blacklist_accessions={"SRR2"})
        assert stats["blacklisted_accessions"] == ["SRR2"]

    def test_expected_spots_reaches_download_accession_per_accession(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\nSRR2\n")
        calls = {}

        def fake_download_accession(accession, *args, **kwargs):
            calls[accession] = kwargs.get("expected_spots")
            return True, "Downloaded 1 files, unverified"

        with patch("metaquest.data.sra.download_accession", side_effect=fake_download_accession):
            download_sra(tmp_path / "fastq", acc, expected_spots={"SRR1": 1000})

        assert calls == {"SRR1": 1000, "SRR2": None}

    def test_download_sra_records_verdict_through_on_result(self, tmp_path, monkeypatch):
        """A real download_accession run (files created, verified) reports its verdict via on_result."""
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        fastq_folder = tmp_path / "fastq"

        def fake_run_secure(executable, args, **kwargs):
            out_dir = Path(args[args.index("-O") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
            return Mock(returncode=0, stdout="", stderr="")

        seen = []
        monkeypatch.setattr("metaquest.utils.security.SecureSubprocess.run_secure", fake_run_secure)
        monkeypatch.setattr("metaquest.data.sra.count_fastq_reads", lambda path: 1)

        download_sra(
            fastq_folder,
            acc,
            expected_spots={"SRR1": 2},
            on_result=lambda a, ok, msg: seen.append((a, ok, msg)),
        )

        assert seen == [("SRR1", True, "Downloaded 1 files, truncated (1 of 2 spots)")]

    def test_truncated_accessions_are_redownloaded_not_skipped(self, tmp_path):
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        fastq_folder = tmp_path / "fastq"
        (fastq_folder / "SRR1").mkdir(parents=True)
        (fastq_folder / "SRR1" / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")

        with patch(
            "metaquest.data.sra.download_accession", return_value=(True, "Downloaded 1 files, complete (2 of 2 spots)")
        ) as mock_download:
            stats = download_sra(fastq_folder, acc, truncated_accessions={"SRR1"})

        mock_download.assert_called_once()
        assert stats["already_downloaded"] == 0
        assert stats["successful"] == 1


if __name__ == "__main__":
    pytest.main([__file__])


class TestDownloadSraStore:
    """download_sra with a shared store: link what is already there, download what is not."""

    @staticmethod
    def _store(tmp_path):
        from metaquest.store.layout import init_store

        return init_store(tmp_path / "store")

    @staticmethod
    def _accessions(tmp_path, *accessions):
        acc = tmp_path / "acc.txt"
        acc.write_text("".join(f"{a}\n" for a in accessions))
        return acc

    @staticmethod
    def _store_dataset(paths, accession="SRR1", state="complete"):
        from metaquest.store.sidecar import Sidecar, write_sidecar

        acc_dir = paths.sra / accession
        acc_dir.mkdir(parents=True, exist_ok=True)
        (acc_dir / f"{accession}_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        write_sidecar(acc_dir / f"{accession}.json", Sidecar(accession=accession, state=state))
        return acc_dir

    @staticmethod
    def _fake_download(calls):
        """A download_accession stand-in that writes one FASTQ under <output_folder>/<acc>."""

        def _download(accession, output_folder, *args, **kwargs):
            calls.append((accession, Path(output_folder), kwargs))
            acc_dir = Path(output_folder) / accession
            acc_dir.mkdir(parents=True, exist_ok=True)
            (acc_dir / f"{accession}_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
            return True, "Downloaded 1 files, unverified"

        return _download

    def test_complete_in_store_links_without_downloading(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths)
        fastq_folder = tmp_path / "project" / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert calls == []
        assert stats["successful"] == 1
        assert stats["results"]["SRR1"].startswith("linked from store")
        assert (fastq_folder / "SRR1").is_symlink()
        assert (fastq_folder / "SRR1").resolve() == (paths.sra / "SRR1").resolve()

    def test_unverified_sidecar_also_links(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths, state="unverified")
        fastq_folder = tmp_path / "project" / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert calls == []
        assert stats["successful"] == 1
        assert (fastq_folder / "SRR1").is_symlink()

    def test_absent_downloads_into_the_store_and_links(self, tmp_path):
        from metaquest.store.catalog import Catalog
        from metaquest.store.sidecar import read_sidecar

        paths = self._store(tmp_path)
        fastq_folder = tmp_path / "project" / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert len(calls) == 1
        accession, output_folder, kwargs = calls[0]
        assert accession == "SRR1"
        assert output_folder == paths.sra
        assert Path(kwargs["staging_folder"]) == paths.tmp
        assert stats["successful"] == 1
        assert stats["results"]["SRR1"].endswith("; stored")
        assert "Downloaded 1 files, unverified" in stats["results"]["SRR1"]

        sidecar = read_sidecar(paths.sra / "SRR1" / "SRR1.json")
        assert sidecar is not None and sidecar.accession == "SRR1"
        assert sidecar.compression == "none"
        with Catalog(paths) as catalog:
            assert catalog.get_dataset("SRR1") is not None
        assert (fastq_folder / "SRR1").is_symlink()
        assert not (paths.locks / "SRR1.lock").exists()

    def test_gzipped_store_download_records_gzip_compression(self, tmp_path):
        from metaquest.store.sidecar import read_sidecar

        paths = self._store(tmp_path)
        fastq_folder = tmp_path / "project" / "fastq"

        def _download(accession, output_folder, *args, **kwargs):
            acc_dir = Path(output_folder) / accession
            acc_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(acc_dir / f"{accession}_1.fastq.gz", "wt") as handle:
                handle.write("@r\nACGT\n+\nIIII\n")
            return True, "Downloaded 1 files, unverified"

        with patch("metaquest.data.sra.download_accession", side_effect=_download):
            download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert read_sidecar(paths.sra / "SRR1" / "SRR1.json").compression == "gzip"

    def test_store_download_uses_metadata_xml_for_expected_spots(self, tmp_path):
        from metaquest.store.sidecar import read_sidecar

        paths = self._store(tmp_path)
        fastq_folder = tmp_path / "project" / "fastq"
        project_metadata = tmp_path / "project" / "metadata"
        project_metadata.mkdir(parents=True)
        (project_metadata / "SRR1_metadata.xml").write_text(
            '<RunSet><RUN total_spots="1" total_bases="4" size="10"/></RunSet>'
        )
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            download_sra(
                fastq_folder,
                self._accessions(tmp_path, "SRR1"),
                store=paths,
                store_metadata=project_metadata,
                max_retries=0,
            )

        sidecar = read_sidecar(paths.sra / "SRR1" / "SRR1.json")
        assert sidecar.ncbi["spots"] == 1
        assert sidecar.completeness["verdict"] == "complete"
        assert sidecar.state == "complete"

    def test_partial_in_store_is_redownloaded_when_resume_partial(self, tmp_path):
        from metaquest.store.sidecar import read_sidecar

        paths = self._store(tmp_path)
        self._store_dataset(paths, state="partial")
        fastq_folder = tmp_path / "project" / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert len(calls) == 1
        # The replacement is built beside the store, then moved over the partial copy.
        assert calls[0][1] == paths.tmp
        assert stats["successful"] == 1
        assert read_sidecar(paths.sra / "SRR1" / "SRR1.json").state == "complete"
        assert (fastq_folder / "SRR1").is_symlink()
        assert not (paths.tmp / "SRR1").exists()

    def test_files_without_a_sidecar_are_redownloaded(self, tmp_path):
        paths = self._store(tmp_path)
        acc_dir = paths.sra / "SRR1"
        acc_dir.mkdir(parents=True)
        (acc_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        fastq_folder = tmp_path / "project" / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert len(calls) == 1

    def test_partial_in_store_is_refused_without_resume_partial(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths, state="partial")
        fastq_folder = tmp_path / "project" / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(
                fastq_folder,
                self._accessions(tmp_path, "SRR1"),
                store=paths,
                resume_partial=False,
                max_retries=0,
            )

        assert calls == []
        assert stats["failed"] == 1
        assert "rerun with --resume-partial" in stats["results"]["SRR1"]
        assert not (fastq_folder / "SRR1").exists()

    def test_accept_partial_links_the_partial_copy(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths, state="partial")
        fastq_folder = tmp_path / "project" / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(
                fastq_folder,
                self._accessions(tmp_path, "SRR1"),
                store=paths,
                resume_partial=False,
                accept_partial=True,
                max_retries=0,
            )

        assert calls == []
        assert stats["successful"] == 1
        assert stats["results"]["SRR1"].startswith("linked from store")
        assert (fastq_folder / "SRR1").is_symlink()

    def test_failed_store_download_keeps_the_partial_copy(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths, state="partial")
        fastq_folder = tmp_path / "project" / "fastq"

        with patch("metaquest.data.sra.download_accession", return_value=(False, "network: Download failed: timeout")):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert stats["failed"] == 1
        assert (paths.sra / "SRR1" / "SRR1_1.fastq").exists()
        assert not (fastq_folder / "SRR1").exists()

    def test_second_project_links_without_downloading_again(self, tmp_path):
        paths = self._store(tmp_path)
        first = tmp_path / "project_a" / "fastq"
        second = tmp_path / "project_b" / "fastq"
        acc = self._accessions(tmp_path, "SRR1")
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            download_sra(first, acc, store=paths, max_retries=0)
            download_sra(second, acc, store=paths, max_retries=0)

        assert len(calls) == 1
        assert (first / "SRR1").resolve() == (second / "SRR1").resolve() == (paths.sra / "SRR1").resolve()

    def test_link_mode_copy_copies_instead_of_linking(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths)
        fastq_folder = tmp_path / "project" / "fastq"

        with patch("metaquest.data.sra.download_accession", side_effect=AssertionError("no download expected")):
            download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, link_mode="copy", max_retries=0)

        assert (fastq_folder / "SRR1").is_dir() and not (fastq_folder / "SRR1").is_symlink()

    def test_an_existing_project_link_counts_as_already_downloaded(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths)
        fastq_folder = tmp_path / "project" / "fastq"
        from metaquest.store.link import link_dataset

        link_dataset(fastq_folder, "SRR1", paths)

        with patch("metaquest.data.sra.download_accession", side_effect=AssertionError("no download expected")):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert stats["already_downloaded_accessions"] == ["SRR1"]

    def test_a_link_to_a_partial_dataset_is_not_already_downloaded(self, tmp_path):
        paths = self._store(tmp_path)
        self._store_dataset(paths, state="partial")
        fastq_folder = tmp_path / "project" / "fastq"
        from metaquest.store.link import link_dataset

        link_dataset(fastq_folder, "SRR1", paths)
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), store=paths, max_retries=0)

        assert stats["already_downloaded_accessions"] == []
        assert len(calls) == 1

    def test_without_a_store_nothing_changes(self, tmp_path):
        fastq_folder = tmp_path / "fastq"
        calls = []

        with patch("metaquest.data.sra.download_accession", side_effect=self._fake_download(calls)):
            stats = download_sra(fastq_folder, self._accessions(tmp_path, "SRR1"), max_retries=0)

        assert calls[0][1] == fastq_folder
        assert stats["results"]["SRR1"] == "Downloaded 1 files, unverified"


class TestFasterqDumpVersion:
    """The tool version recorded in a store dataset's sidecar."""

    def test_returns_the_last_non_empty_line(self):
        with patch(
            "metaquest.utils.security.SecureSubprocess.run_secure",
            return_value=Mock(stdout="\nfasterq-dump : 3.0.10\n"),
        ) as mock_run:
            assert fasterq_dump_version() == "fasterq-dump : 3.0.10"
        mock_run.assert_called_once_with("fasterq-dump", ["--version"])

    def test_returns_empty_when_the_tool_cannot_be_run(self):
        with patch("metaquest.utils.security.SecureSubprocess.run_secure", side_effect=SecurityError("not installed")):
            assert fasterq_dump_version() == ""


class TestStoreDownloadLockWait:
    """download_sra's --lock-wait: give up on an accession another project is downloading."""

    @staticmethod
    def _store(tmp_path):
        from metaquest.store.layout import init_store

        return init_store(tmp_path / "store")

    def test_lock_wait_gives_up_naming_the_accession_and_holder(self, tmp_path):
        import json as json_module

        from metaquest.store.layout import lock_path

        paths = self._store(tmp_path)
        lock = lock_path(paths, "SRR1")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json_module.dumps({"pid": 4242, "host": "otherhost", "started": "2026-01-01T00:00:00+00:00"}))
        acc = tmp_path / "acc.txt"
        acc.write_text("SRR1\n")
        fastq_folder = tmp_path / "project" / "fastq"

        with patch("metaquest.data.sra.download_accession") as mock_download:
            stats = download_sra(
                fastq_folder, acc, store=paths, max_retries=0, lock_wait=0.1, num_threads=1, max_workers=1
            )

        mock_download.assert_not_called()
        assert stats["failed"] == 1
        message = stats["results"]["SRR1"]
        assert "SRR1" in message
        assert "4242" in message and "otherhost" in message
        # The other project's lock is left exactly as it was.
        assert json_module.loads(lock.read_text())["pid"] == 4242
