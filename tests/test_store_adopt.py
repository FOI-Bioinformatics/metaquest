"""Tests for `metaquest.store.adopt`: folding project-owned FASTQ folders into the store."""

import gzip
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.store.adopt import adopt
from metaquest.store.catalog import Catalog
from metaquest.store.layout import init_store, sidecar_path, sra_dir
from metaquest.store.sidecar import read_sidecar, write_sidecar


def _write_fastq(path: Path, text: str = "@r\nACGT\n+\nIIII\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_fastq_gz(path: Path, text: str = "@r\nACGT\n+\nIIII\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(text)


class TestAdoptFresh:
    def test_move_relocates_files_writes_sidecar_and_links(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == ["SRR1"]
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

    def test_copy_leaves_no_original_and_no_double_storage(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq(project_fastq / "SRR1" / "SRR1.fastq")

        report = adopt(project_fastq, paths, move=False, dry_run=False)

        assert report.adopted == ["SRR1"]
        link = project_fastq / "SRR1"
        assert link.is_symlink()
        assert (sra_dir(paths, "SRR1")).is_dir()

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
        from metaquest.store.sidecar import build_sidecar

        sidecar = build_sidecar(accession, acc_dir, {}, "adopted", "gzip")
        write_sidecar(sidecar_path(paths, accession), sidecar)
        with Catalog(paths) as cat:
            cat.migrate()
        from metaquest.store.catalog import catalog_write

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
        assert not (project_fastq / "SRR1").is_dir() or (project_fastq / "SRR1").is_symlink()
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
        from metaquest.store.sidecar import build_sidecar
        from metaquest.store.catalog import catalog_write

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


class TestAdoptRestart:
    def test_restart_completes_a_half_moved_accession(self, tmp_path):
        """The store already has the files (a previous move finished) but no sidecar was
        written before the process was interrupted; the project's folder is already gone
        (the move already relocated it). Rerunning adopt finishes the sidecar and links it."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        project_fastq.mkdir(parents=True)

        assert not sidecar_path(paths, "SRR1").exists()

        report = adopt(project_fastq, paths, move=True, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert sidecar_path(paths, "SRR1").exists()
        assert (project_fastq / "SRR1").is_symlink()
        with Catalog(paths) as cat:
            cat.migrate()
            assert cat.get_dataset("SRR1") is not None

    def test_restart_with_project_copy_still_present_finishes_and_links(self, tmp_path):
        """The move into the store's final location succeeded, and (unlike the pure-move
        scenario) the project's original folder is still there too (an interrupted copy);
        rerunning finishes the sidecar, removes the leftover project copy, and links."""
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")

        project_fastq = tmp_path / "project" / "fastq"
        _write_fastq_gz(project_fastq / "SRR1" / "SRR1.fastq.gz")

        report = adopt(project_fastq, paths, move=False, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert (project_fastq / "SRR1").is_symlink()
        assert sidecar_path(paths, "SRR1").exists()


class TestAdoptCrossFilesystem:
    def test_move_refuses_across_filesystems(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq(entry / "SRR1.fastq")

        real_stat = os.stat
        entry_str = str(entry)

        class _ShiftedDevStat:
            """Proxies every attribute of a real stat_result except a bumped st_dev."""

            def __init__(self, real_result):
                self._real = real_result

            def __getattr__(self, name):
                return getattr(self._real, name)

            @property
            def st_dev(self):
                return self._real.st_dev + 1

        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if os.fspath(path) == entry_str:
                return _ShiftedDevStat(result)
            return result

        with patch("metaquest.store.adopt.os.stat", side_effect=fake_stat):
            with pytest.raises(DataAccessError, match="different filesystems"):
                adopt(project_fastq, paths, move=True, dry_run=False)

        # Nothing was touched: the original folder is still there, untouched.
        assert (entry / "SRR1.fastq").is_file()
        assert not sra_dir(paths, "SRR1").exists()

    def test_copy_across_filesystems_succeeds(self, tmp_path):
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        project_fastq = tmp_path / "project" / "fastq"
        entry = project_fastq / "SRR1"
        _write_fastq(entry / "SRR1.fastq")

        real_stat = os.stat
        entry_str = str(entry)

        class _ShiftedDevStat:
            """Proxies every attribute of a real stat_result except a bumped st_dev."""

            def __init__(self, real_result):
                self._real = real_result

            def __getattr__(self, name):
                return getattr(self._real, name)

            @property
            def st_dev(self):
                return self._real.st_dev + 1

        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if os.fspath(path) == entry_str:
                return _ShiftedDevStat(result)
            return result

        with patch("metaquest.store.adopt.os.stat", side_effect=fake_stat):
            report = adopt(project_fastq, paths, move=False, dry_run=False)

        assert report.adopted == ["SRR1"]
        assert (project_fastq / "SRR1").is_symlink()
