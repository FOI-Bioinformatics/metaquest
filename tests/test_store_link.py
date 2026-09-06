"""
Tests for metaquest.store.link: the symlink a project keeps into the shared store.
"""

import os
import shutil
from pathlib import Path

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.data.registry import scan_downloads
from metaquest.data.sra import accession_has_fastq
from metaquest.store.layout import init_store
from metaquest.store.link import dangling_links, is_store_link, link_dataset, unlink_dataset


@pytest.fixture
def paths(tmp_path):
    return init_store(tmp_path / "store")


def _store_dataset(paths, accession="SRR1", state="complete"):
    """Create a store dataset folder with one FASTQ file and a sidecar in ``state``."""
    acc_dir = paths.sra / accession
    acc_dir.mkdir(parents=True, exist_ok=True)
    (acc_dir / f"{accession}_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
    (acc_dir / f"{accession}.json").write_text('{"accession": "%s", "state": "%s"}' % (accession, state))
    return acc_dir


# ------------------------------------------------------------------ link_dataset


def test_link_dataset_creates_a_relative_symlink_under_a_shared_parent(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"

    link = link_dataset(project_fastq, "SRR1", paths)

    assert link == project_fastq / "SRR1"
    assert link.is_symlink()
    assert not os.path.isabs(os.readlink(link))
    assert link.resolve() == (paths.sra / "SRR1").resolve()


def test_link_dataset_absolute_mode_writes_an_absolute_target(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"

    link = link_dataset(project_fastq, "SRR1", paths, mode="absolute")

    assert os.path.isabs(os.readlink(link))
    assert link.resolve() == (paths.sra / "SRR1").resolve()


def test_link_dataset_relative_mode_writes_a_relative_target(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"

    link = link_dataset(project_fastq, "SRR1", paths, mode="relative")

    assert not os.path.isabs(os.readlink(link))


def test_link_dataset_auto_falls_back_to_absolute_without_a_shared_parent(tmp_path, paths, monkeypatch):
    """Two paths with no common parent above the filesystem root get an absolute link."""
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    monkeypatch.setattr("metaquest.store.link.os.path.commonpath", lambda paths_: os.sep)

    link = link_dataset(project_fastq, "SRR1", paths)

    assert os.path.isabs(os.readlink(link))


def test_link_dataset_auto_falls_back_to_absolute_on_separate_drives(tmp_path, paths, monkeypatch):
    """commonpath raises for paths on different drives; the link is absolute rather than failing."""
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"

    def _raise(_paths):
        raise ValueError("paths do not share a drive")

    monkeypatch.setattr("metaquest.store.link.os.path.commonpath", _raise)

    link = link_dataset(project_fastq, "SRR1", paths)

    assert os.path.isabs(os.readlink(link))


def test_link_dataset_copy_mode_copies_the_folder(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"

    copied = link_dataset(project_fastq, "SRR1", paths, mode="copy")

    assert copied.is_dir() and not copied.is_symlink()
    assert (copied / "SRR1_1.fastq").read_text() == "@r\nACGT\n+\nIIII\n"
    # The store copy is untouched.
    assert (paths.sra / "SRR1" / "SRR1_1.fastq").exists()


def test_link_dataset_refuses_to_replace_a_real_directory(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    real_dir = project_fastq / "SRR1"
    real_dir.mkdir(parents=True)
    (real_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")

    with pytest.raises(DataAccessError):
        link_dataset(project_fastq, "SRR1", paths)

    assert (real_dir / "SRR1_1.fastq").exists()


def test_link_dataset_replaces_an_existing_symlink(tmp_path, paths):
    _store_dataset(paths)
    _store_dataset(paths, accession="SRR2")
    project_fastq = tmp_path / "project" / "fastq"
    project_fastq.mkdir(parents=True)
    os.symlink(paths.sra / "SRR2", project_fastq / "SRR1")

    link = link_dataset(project_fastq, "SRR1", paths)

    assert link.resolve() == (paths.sra / "SRR1").resolve()


def test_link_dataset_raises_when_the_store_dataset_is_missing(tmp_path, paths):
    with pytest.raises(DataAccessError):
        link_dataset(tmp_path / "project" / "fastq", "SRR404", paths)


def test_link_dataset_rejects_an_unknown_mode(tmp_path, paths):
    _store_dataset(paths)
    with pytest.raises(DataAccessError):
        link_dataset(tmp_path / "project" / "fastq", "SRR1", paths, mode="hardlink")


# ---------------------------------------------------------------- unlink_dataset


def test_unlink_dataset_removes_only_symlinks(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)

    assert unlink_dataset(project_fastq, "SRR1") is True
    assert not (project_fastq / "SRR1").exists()
    # The store keeps its copy.
    assert (paths.sra / "SRR1" / "SRR1_1.fastq").exists()


def test_unlink_dataset_leaves_a_real_directory_alone(tmp_path, paths):
    project_fastq = tmp_path / "project" / "fastq"
    real_dir = project_fastq / "SRR1"
    real_dir.mkdir(parents=True)
    (real_dir / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")

    assert unlink_dataset(project_fastq, "SRR1") is False
    assert (real_dir / "SRR1_1.fastq").exists()


def test_unlink_dataset_returns_false_when_nothing_is_there(tmp_path):
    assert unlink_dataset(tmp_path / "fastq", "SRR1") is False


def test_unlink_dataset_removes_a_dangling_symlink(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)
    shutil.rmtree(paths.sra / "SRR1")

    assert unlink_dataset(project_fastq, "SRR1") is True
    assert not (project_fastq / "SRR1").is_symlink()


# ----------------------------------------------------------------- is_store_link


def test_is_store_link_distinguishes_store_links_from_other_entries(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)

    other = tmp_path / "elsewhere" / "SRR2"
    other.mkdir(parents=True)
    os.symlink(other, project_fastq / "SRR2")

    real_dir = project_fastq / "SRR3"
    real_dir.mkdir()

    assert is_store_link(project_fastq / "SRR1", paths) is True
    assert is_store_link(project_fastq / "SRR2", paths) is False
    assert is_store_link(real_dir, paths) is False
    assert is_store_link(project_fastq / "SRR9", paths) is False


def test_is_store_link_true_for_a_dangling_store_link(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)
    shutil.rmtree(paths.sra / "SRR1")

    assert is_store_link(project_fastq / "SRR1", paths) is True


# --------------------------------------------------------------- dangling_links


def test_dangling_links_lists_links_whose_target_is_gone(tmp_path, paths):
    _store_dataset(paths)
    _store_dataset(paths, accession="SRR2")
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)
    link_dataset(project_fastq, "SRR2", paths)
    (project_fastq / "SRR3").mkdir()
    shutil.rmtree(paths.sra / "SRR2")

    assert dangling_links(project_fastq) == ["SRR2"]


def test_dangling_links_empty_for_a_missing_folder(tmp_path):
    assert dangling_links(tmp_path / "nope") == []


# -------------------------------------------------- the rest of the tool sees it


def test_scan_downloads_counts_a_linked_dataset(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)

    found = scan_downloads(project_fastq)

    assert list(found) == ["SRR1"]
    assert found["SRR1"][0] == 1
    assert accession_has_fastq(project_fastq / "SRR1") is True


def test_scan_downloads_skips_a_link_to_a_partial_dataset(tmp_path, paths):
    _store_dataset(paths, state="partial")
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)

    assert scan_downloads(project_fastq) == {}
    assert accession_has_fastq(project_fastq / "SRR1") is False


def test_scan_downloads_skips_a_dangling_link(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)
    shutil.rmtree(paths.sra / "SRR1")

    assert scan_downloads(project_fastq) == {}


def test_linked_dataset_is_readable_through_the_project_path(tmp_path, paths):
    _store_dataset(paths)
    project_fastq = tmp_path / "project" / "fastq"
    link_dataset(project_fastq, "SRR1", paths)

    assert (Path(project_fastq) / "SRR1" / "SRR1_1.fastq").read_text().startswith("@r")


def test_link_dataset_relative_link_for_a_relative_project_folder(tmp_path, paths, monkeypatch):
    """A relative --fastq-folder under a shared parent still gets a working relative link."""
    _store_dataset(paths)
    monkeypatch.chdir(tmp_path)

    link = link_dataset(Path("project") / "fastq", "SRR1", paths)

    assert not os.path.isabs(os.readlink(link))
    assert link.resolve() == (paths.sra / "SRR1").resolve()
    assert (link / "SRR1_1.fastq").is_file()
