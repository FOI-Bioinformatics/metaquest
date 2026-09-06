"""Tests for `metaquest.store.adopt`: folding project-owned FASTQ folders into the store."""

import gzip
from pathlib import Path
from unittest.mock import patch

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.store.adopt import adopt
from metaquest.store.catalog import Catalog, catalog_write
from metaquest.store.layout import init_store, lock_path, sidecar_path, sra_dir
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
        assert sidecar.tool == "adopted"
        assert sidecar.compression == "gzip"
        assert (sra_dir(paths, "SRR1") / "SRR1.fastq.gz").is_file()

        with Catalog(paths, create=True) as cat:
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
        # Not even the catalogue was created: a dry run opens nothing for writing.
        assert not paths.catalog.exists()

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
        with Catalog(paths, create=True) as cat:
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

    def test_a_foreign_sidecar_less_folder_is_reported_and_left_alone(self, tmp_path):
        """A sidecar-less sra/<ACC> for an accession this project never had belongs to
        someone else's interrupted run, most likely one still in flight: adoption reports it
        and touches neither the folder nor the project."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR-FOREIGN")
        _write_fastq_gz(acc_dir / "SRR-FOREIGN.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert report.foreign == ["SRR-FOREIGN"]
        assert not (project_fastq / "SRR-FOREIGN").exists()
        assert not sidecar_path(paths, "SRR-FOREIGN").exists()
        assert (acc_dir / "SRR-FOREIGN.fastq.gz").is_file()

    def test_an_accession_locked_by_another_run_is_skipped_as_in_progress(self, tmp_path):
        """The store holds our accession without a sidecar and its lock is held: another run
        is publishing it right now, so this one leaves both copies alone."""
        import json

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq_gz(entry / "SRR1.fastq.gz")

        lock = lock_path(paths, "SRR1")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": 999999, "host": "otherhost", "started": "2026-01-01T00:00:00+00:00"}))

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.in_progress == ["SRR1"]
        assert report.adopted == []
        assert not sidecar_path(paths, "SRR1").exists()
        assert entry.is_dir() and not entry.is_symlink()

    def test_adoption_refuses_an_accession_without_room_for_it(self, tmp_path):
        """Adoption peaks at three copies of one accession; without room for the staging copy
        it refuses that accession with a message rather than filling the store's filesystem."""
        import shutil as shutil_module
        from collections import namedtuple

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq(entry / "SRR1.fastq")

        usage = namedtuple("usage", "total used free")
        with patch.object(shutil_module, "disk_usage", return_value=usage(1000, 1000, 1)):
            report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.refused == ["SRR1"]
        assert report.adopted == []
        assert not sra_dir(paths, "SRR1").exists()
        assert (entry / "SRR1.fastq").is_file()

    def test_an_adopted_sidecar_records_the_tool_and_the_files_own_age(self, tmp_path):
        """An adopted dataset was not downloaded now, and not by fasterq-dump: its sidecar
        says so, so --older-than counts from the reads' own age rather than from adoption."""
        import os
        import time
        from datetime import datetime, timezone

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq_gz(entry / "SRR1.fastq.gz")
        old = time.time() - 90 * 86400
        os.utime(entry / "SRR1.fastq.gz", (old, old))

        adopt(project_fastq, paths, move=True, dry_run=False)

        sidecar = read_sidecar(sidecar_path(paths, "SRR1"))
        assert sidecar.tool == "adopted"
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(sidecar.downloaded)).days
        assert age_days >= 89


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


class TestAdoptPerAccessionLocking:
    """The sweep of a stale staging folder is scoped to one accession's own lock, not a
    store-wide sweep: two concurrent adopts must never touch each other's staging folder."""

    def test_stale_staging_for_the_adopted_accession_is_removed_and_reported(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        stale = paths.tmp / "SRR1_adopt"
        _write_fastq_gz(stale / "SRR1.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.resumed == ["SRR1"]
        assert report.adopted == ["SRR1"]
        assert not stale.exists()
        assert (project_fastq / "SRR1").is_symlink()

    def test_stale_staging_for_a_different_accession_is_left_untouched(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        other_stale = paths.tmp / "SRR-OTHER_adopt"
        _write_fastq_gz(other_stale / "SRR-OTHER.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.resumed == []
        assert report.adopted == ["SRR1"]
        # A stale staging folder for an accession this run never touches survives untouched.
        assert other_stale.is_dir()
        assert (other_stale / "SRR-OTHER.fastq.gz").is_file()

    def test_lock_file_is_created_and_released_on_success(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert not lock_path(paths, "SRR1").exists()

    def test_a_lock_held_by_a_live_holder_makes_lock_wait_give_up(self, tmp_path):
        import json

        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        # A live holder (a fresh lock file, so not stale): with no --lock-wait, adopt would
        # rightly wait for as long as that project keeps working, so this run gives up instead.
        lock = lock_path(paths, "SRR1")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": 999999, "host": "otherhost", "started": "2026-01-01T00:00:00+00:00"}))

        with pytest.raises(DataAccessError, match="SRR1"):
            adopt(project_fastq, paths, move=True, dry_run=False, lock_wait=0.1)

        # The lock was held by "another process": adopt() must not remove a lock it did not
        # create itself.
        assert lock.exists()
        assert json.loads(lock.read_text())["pid"] == 999999
        # Nothing was staged or moved, since the lock was never acquired.
        assert not sra_dir(paths, "SRR1").exists()
        assert (project_fastq / "SRR1" / "SRR1.fastq").is_file()
