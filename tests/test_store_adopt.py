"""Tests for `metaquest.store.adopt`: folding project-owned FASTQ folders into the store."""

import gzip
from pathlib import Path
from unittest.mock import patch

import pytest

from metaquest.store.adopt import adopt
from metaquest.store.catalog import Catalog, catalog_write
from metaquest.store.layout import init_store, sidecar_path, sra_dir
from metaquest.store.sidecar import build_sidecar, read_sidecar, write_sidecar


def _write_fastq(path: Path, text: str = "@r\nACGT\n+\nIIII\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_fastq_gz(path: Path, text: str = "@r\nACGT\n+\nIIII\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(text)


class TestAdoptFresh:
    def test_move_removes_project_copy_writes_sidecar_and_links(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert report.copied == []
        assert report.deduplicated == []
        assert report.conflicts == []

        link = project_fastq / "SRR1"
        assert link.is_symlink()
        assert (sra_dir(paths, "SRR1")).is_dir()
        sidecar = read_sidecar(sidecar_path(paths, "SRR1"))
        assert sidecar is not None
        assert sidecar.tool_version == "adopted"
        assert sidecar.compression == "gzip"
        assert (sra_dir(paths, "SRR1") / "SRR1.fastq.gz").is_file()

        with Catalog(paths) as cat:
            cat.migrate()
            assert cat.get_dataset("SRR1") is not None

    def test_copy_leaves_project_folder_untouched_and_unlinked(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq(entry / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=False, dry_run=False)

        assert report.copied == ["SRR1"]
        assert report.adopted == []
        # The project's own folder is untouched: still a real directory, never linked.
        assert entry.is_dir() and not entry.is_symlink()
        assert (entry / "SRR1.fastq").is_file()
        # The store now holds its own copy too.
        assert (sra_dir(paths, "SRR1")).is_dir()
        sidecar = read_sidecar(sidecar_path(paths, "SRR1"))
        assert sidecar is not None

    def test_no_compress_leaves_files_plain(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        adopt(project_fastq, paths, move=True, dry_run=False, compress=False)

        assert (sra_dir(paths, "SRR1") / "SRR1.fastq").is_file()
        sidecar = read_sidecar(sidecar_path(paths, "SRR1"))
        assert sidecar.compression == "none"

    def test_transient_folder_is_ignored(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        (project_fastq / "SRR1_temp").mkdir(parents=True)
        (project_fastq / ".sra-cache").mkdir(parents=True)

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == []
        assert report.skipped == []
        assert (project_fastq / "SRR1_temp").is_dir()

    def test_already_linked_entry_is_skipped(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")
        adopt(project_fastq, paths, move=True, dry_run=False)

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == []
        assert report.skipped == ["SRR1"]

    def test_metadata_folder_supplies_ncbi_spot_count(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")
        metadata_dir = tmp_path / "project" / "metadata"
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "SRR1_metadata.xml").write_text('<root><RUN total_spots="1" total_bases="4"></RUN></root>')

        adopt(project_fastq, paths, move=True, dry_run=False, metadata_folders=[metadata_dir])

        sidecar = read_sidecar(sidecar_path(paths, "SRR1"))
        assert sidecar.ncbi.get("spots") == 1


class TestAdoptDedupAndConflict:
    def _seed_store_dataset(self, paths, accession, text="@r\nACGT\n+\nIIII\n"):
        acc_dir = sra_dir(paths, accession)
        _write_fastq_gz(acc_dir / f"{accession}.fastq.gz", text)
        sidecar = build_sidecar(accession, acc_dir, {}, "adopted", "gzip")
        write_sidecar(sidecar_path(paths, accession), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)
        return sidecar

    def test_identical_copy_is_deduplicated(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        self._seed_store_dataset(paths, "SRR1")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq_gz(project_fastq / "SRR1" / "SRR1.fastq.gz")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.deduplicated == ["SRR1"]
        assert report.adopted == []
        assert (project_fastq / "SRR1").is_symlink()

    def test_identical_content_dedups_across_differing_compression(self, tmp_path):
        """A plain project copy of the same reads as a gzipped store copy is a duplicate, not
        a conflict: files are compared by decompressed content, not raw bytes/name."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        self._seed_store_dataset(paths, "SRR1", text="@r\nACGT\n+\nIIII\n")

        project_fastq = tmp_path / "project" / "fastq"
        # Same reads, but plain (not gzipped) and named without the .gz suffix.
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq", text="@r\nACGT\n+\nIIII\n")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.deduplicated == ["SRR1"]
        assert report.conflicts == []
        assert (project_fastq / "SRR1").is_symlink()

    def test_conflicting_copy_keeps_both(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        self._seed_store_dataset(paths, "SRR1", text="@r\nACGT\n+\nIIII\n")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq_gz(project_fastq / "SRR1" / "SRR1.fastq.gz", text="@r\nTTTT\n+\nIIII\n")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.conflicts == ["SRR1"]
        assert report.adopted == []
        assert report.deduplicated == []
        # Both copies survive untouched.
        assert (project_fastq / "SRR1").is_dir() and not (project_fastq / "SRR1").is_symlink()
        assert (sra_dir(paths, "SRR1") / "SRR1.fastq.gz").is_file()

    def test_conflicting_read_count_across_differing_compression_is_not_deduplicated(self, tmp_path):
        """Different content (and so a different read count) across differing compression must
        still be flagged as a conflict, not silently treated as a match."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        self._seed_store_dataset(paths, "SRR1", text="@r\nACGT\n+\nIIII\n")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq", text="@r\nTTTT\n+\nIIII\n")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.conflicts == ["SRR1"]
        assert report.deduplicated == []


class TestAdoptDryRun:
    def test_dry_run_writes_nothing(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=True)

        assert report.planned == ["SRR1"]
        assert report.adopted == []
        # Nothing moved, nothing created in the store.
        assert (project_fastq / "SRR1" / "SRR1.fastq").is_file()
        assert not sra_dir(paths, "SRR1").exists()
        with Catalog(paths) as cat:
            cat.migrate()
            assert cat.get_dataset("SRR1") is None

    def test_dry_run_still_reports_conflicts(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz", text="@r\nACGT\n+\nIIII\n")
        sidecar = build_sidecar("SRR1", acc_dir, {}, "adopted", "gzip")
        write_sidecar(sidecar_path(paths, "SRR1"), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq_gz(project_fastq / "SRR1" / "SRR1.fastq.gz", text="@r\nTTTT\n+\nIIII\n")

        report = adopt(project_fastq, paths, move=True, dry_run=True)

        assert report.conflicts == ["SRR1"]
        assert report.planned == []
        # Original project copy untouched.
        assert (project_fastq / "SRR1" / "SRR1.fastq.gz").is_file()

    def test_dry_run_does_not_sweep_stale_staging(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        stale = paths.tmp / "SRR1_adopt"
        _write_fastq_gz(stale / "SRR1.fastq.gz")
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=True)

        assert report.resumed == []
        assert stale.is_dir()


class TestAdoptRestart:
    def test_interrupted_after_final_move_before_sidecar_move_mode(self, tmp_path):
        """The store's final move succeeded (files sit in sra/<ACC>) but the process died
        before the sidecar was written; the project's folder is still there too, since it is
        only ever removed after the sidecar exists. Rerunning with --move finishes the sidecar,
        removes the project's folder, and links it."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq_gz(project_fastq / "SRR1" / "SRR1.fastq.gz")

        assert not sidecar_path(paths, "SRR1").exists()

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert sidecar_path(paths, "SRR1").exists()
        assert (project_fastq / "SRR1").is_symlink()
        with Catalog(paths) as cat:
            cat.migrate()
            assert cat.get_dataset("SRR1") is not None

    def test_interrupted_after_final_move_before_sidecar_copy_mode(self, tmp_path):
        """Same interruption, but this run asked for --copy: the sidecar is finished, and the
        project's folder is left exactly as it was (not removed, not linked)."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq_gz(entry / "SRR1.fastq.gz")

        report = adopt(project_fastq, paths, move=False, dry_run=False)

        assert report.copied == ["SRR1"]
        assert sidecar_path(paths, "SRR1").exists()
        assert entry.is_dir() and not entry.is_symlink()

    def test_project_folder_missing_entirely_still_finishes_via_defensive_scan(self, tmp_path):
        """Edge case: the store has the files and no sidecar, and the project's folder is gone
        by some means outside adopt() (e.g. removed manually). The defensive scan still finishes
        the sidecar and links it."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        project_fastq.mkdir(parents=True)

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert (project_fastq / "SRR1").is_symlink()


class TestAdoptStagingCrashSafety:
    def test_compress_failure_leaves_project_folder_intact_and_no_store_folder(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq(entry / "SRR1.fastq")

        with patch("metaquest.store.adopt.compress_fastq", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                adopt(project_fastq, paths, move=True, dry_run=False)

        # The project's original folder was never touched: staging always copies.
        assert entry.is_dir() and not entry.is_symlink()
        assert (entry / "SRR1.fastq").is_file()
        # The store never got a finished folder for this accession.
        assert not sra_dir(paths, "SRR1").exists()
        assert not sidecar_path(paths, "SRR1").exists()
        # Only the staging leftover exists, under tmp.
        assert (paths.tmp / "SRR1_adopt").is_dir()

    def test_rerun_after_compress_failure_adopts_cleanly_with_no_leftover(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq(entry / "SRR1.fastq")

        with patch("metaquest.store.adopt.compress_fastq", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                adopt(project_fastq, paths, move=True, dry_run=False)

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.resumed == ["SRR1"]
        assert report.adopted == ["SRR1"]
        assert not (paths.tmp / "SRR1_adopt").exists()
        assert (project_fastq / "SRR1").is_symlink()
        sidecar = read_sidecar(sidecar_path(paths, "SRR1"))
        assert sidecar is not None
