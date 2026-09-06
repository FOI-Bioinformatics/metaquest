"""Tests for the `store_init`, `store_status` and `store_reindex` CLI commands.

Every test runs under tmp_path and monkeypatches HOME/XDG_CONFIG_HOME/METAQUEST_DATA so
nothing here reads or writes the real user config, matching the isolation pattern used in
tests/test_store_resolve.py.
"""

import argparse
import gzip
import json
import subprocess
from unittest.mock import patch

import pytest

from metaquest.cli.commands.store import (
    StoreAdoptCommand,
    StoreInitCommand,
    StoreLinkCommand,
    StoreReindexCommand,
    StoreStatusCommand,
    StoreUnlinkCommand,
    StoreVerifyCommand,
)
from metaquest.core.constants import STORE_ENV
from metaquest.data.registry import load_registry
from metaquest.store.catalog import Catalog, catalog_write
from metaquest.store.layout import init_store, read_marker, sidecar_path, sra_dir, store_paths
from metaquest.store.sidecar import Sidecar, read_sidecar, write_sidecar
from metaquest.utils.security import SecureSubprocess


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv(STORE_ENV, raising=False)
    yield


@pytest.fixture(autouse=True)
def reset_allowed_roots():
    SecureSubprocess._extra_roots = []
    yield
    SecureSubprocess._extra_roots = []


def _init_args(root, project_dir, **overrides):
    base = dict(
        data_root=str(root),
        project_name=None,
        set_default=False,
        registry=str(project_dir / "metaquest_registry.json"),
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _status_args(**overrides):
    base = dict(data_root=None, registry=None, json=False, verbose=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _reindex_args(**overrides):
    base = dict(data_root=None, registry=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _adopt_args(**overrides):
    base = dict(
        fastq_folder="fastq",
        data_root=None,
        registry=None,
        move=True,
        dry_run=False,
        compress=True,
        metadata_folder="metadata",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _verify_args(accessions=None, **overrides):
    base = dict(
        accessions=list(accessions or []), data_root=None, registry=None, md5=False, spots=False, fix_state=False
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _link_args(accessions, **overrides):
    base = dict(accessions=list(accessions), fastq_folder="fastq", registry=None, data_root=None, link_mode="auto")
    base.update(overrides)
    return argparse.Namespace(**base)


def _unlink_args(accessions, **overrides):
    base = dict(accessions=list(accessions), fastq_folder="fastq", registry=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _sidecar(accession="SRR1", state="complete"):
    return Sidecar(
        accession=accession,
        state=state,
        layout="SINGLE",
        downloaded="2026-09-06T00:00:00+00:00",
        tool="fasterq-dump",
        tool_version="3.0.0",
        compression="gzip",
        files=[{"name": f"{accession}.fastq.gz", "bytes": 100, "md5": "a" * 32, "reads": 5}],
        reads_per_mate=5,
        bases_total=500,
        ncbi={"spots": 5, "bases": 500, "size": 1000, "layout": "SINGLE", "files": []},
        completeness={"method": "spots", "ratio": 1.0, "verdict": "complete"},
    )


class TestStoreInitCommand:
    def test_command_properties(self):
        cmd = StoreInitCommand()
        assert cmd.name == "store_init"
        assert cmd.group == "Store"
        assert "store" in cmd.help.lower()

    def test_configure_parser_requires_data_root(self):
        cmd = StoreInitCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(["--data-root", "/x"])
        assert args.data_root == "/x"
        assert args.project_name is None
        assert args.set_default is False

    def test_execute_creates_store_and_records_project(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreInitCommand().execute(_init_args(root, project_dir))
        assert rc == 0

        marker = read_marker(root)
        assert marker is not None

        paths = store_paths(root)
        with Catalog(paths) as cat:
            tables = {
                row["name"]
                for row in cat.conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
            }
        assert {"datasets", "projects", "usage"} <= tables

        registry = load_registry(project_dir / "metaquest_registry.json")
        assert registry.store["root"] == str(root.resolve())
        assert registry.store["mode"] == "symlink"
        assert registry.store["linked"] == []
        assert registry.project["name"]
        assert registry.project["id"]
        assert registry.project["path"] == str(project_dir.resolve())

    def test_execute_uses_project_name_flag(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreInitCommand().execute(_init_args(root, project_dir, project_name="Wolbachia"))
        assert rc == 0

        registry = load_registry(project_dir / "metaquest_registry.json")
        assert registry.project["name"] == "Wolbachia"

    def test_execute_defaults_project_name_to_cwd_name(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreInitCommand().execute(_init_args(root, project_dir))
        assert rc == 0

        registry = load_registry(project_dir / "metaquest_registry.json")
        assert registry.project["name"] == "my-project"

    def test_execute_keeps_project_id_on_second_run(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        StoreInitCommand().execute(_init_args(root, project_dir))
        first_id = load_registry(project_dir / "metaquest_registry.json").project["id"]

        StoreInitCommand().execute(_init_args(root, project_dir))
        second_id = load_registry(project_dir / "metaquest_registry.json").project["id"]

        assert first_id == second_id

    def test_execute_set_default_writes_config(self, tmp_path, monkeypatch):
        from metaquest.store.resolve import read_config

        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreInitCommand().execute(_init_args(root, project_dir, set_default=True))
        assert rc == 0
        assert read_config()["store"]["data_root"] == root.resolve().as_posix()

    def test_execute_without_set_default_does_not_write_config(self, tmp_path, monkeypatch):
        from metaquest.store.resolve import read_config

        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        StoreInitCommand().execute(_init_args(root, project_dir, set_default=False))
        assert read_config() == {}

    def test_gitignore_guard_appends_fastq_when_git_present(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()
        monkeypatch.chdir(project_dir)

        with patch("metaquest.cli.commands.store.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            rc = StoreInitCommand().execute(_init_args(root, project_dir))
        assert rc == 0

        gitignore = (project_dir / ".gitignore").read_text()
        assert "fastq/" in gitignore.splitlines()
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["git", "ls-files", "fastq"]

    def test_gitignore_guard_skips_when_already_present(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()
        (project_dir / ".gitignore").write_text("fastq/\nother\n")
        monkeypatch.chdir(project_dir)

        with patch("metaquest.cli.commands.store.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            StoreInitCommand().execute(_init_args(root, project_dir))

        gitignore = (project_dir / ".gitignore").read_text()
        assert gitignore.count("fastq/") == 1

    def test_gitignore_guard_warns_when_fastq_tracked(self, tmp_path, monkeypatch, caplog):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()
        monkeypatch.chdir(project_dir)

        with patch("metaquest.cli.commands.store.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="fastq/SRR1/SRR1.fastq\n", stderr=""
            )
            with caplog.at_level("WARNING"):
                StoreInitCommand().execute(_init_args(root, project_dir))

        assert any("git" in message.lower() and "fastq" in message.lower() for message in caplog.messages)

    def test_gitignore_guard_never_runs_when_no_git_dir(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        with patch("metaquest.cli.commands.store.subprocess.run") as mock_run:
            rc = StoreInitCommand().execute(_init_args(root, project_dir))
        assert rc == 0
        mock_run.assert_not_called()
        assert not (project_dir / ".gitignore").exists()


class TestStoreStatusCommand:
    def test_command_properties(self):
        cmd = StoreStatusCommand()
        assert cmd.name == "store_status"
        assert cmd.group == "Store"

    def test_no_store_configured_prints_hint_and_returns_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = StoreStatusCommand().execute(_status_args(registry=str(tmp_path / "metaquest_registry.json")))
        out = capsys.readouterr().out
        assert rc == 1
        assert "store_init" in out

    def test_reports_counts_bytes_and_projects(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "Wolbachia", str(tmp_path / "proj1"), "reg1")
            cat.upsert_dataset(_sidecar("SRR1", state="complete"))
            cat.upsert_dataset(_sidecar("SRR2", state="partial"))
            cat.record_usage("SRR1", "proj1", "wMel", "downloaded")
        (tmp_path / "proj1").mkdir()

        rc = StoreStatusCommand().execute(_status_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["root"] == str(root.resolve())
        assert report["datasets"] == {"complete": 1, "partial": 1}
        assert report["bytes_total"] == 200
        assert report["projects"] == 1
        assert report["stale_projects"] == []
        assert "datasets_list" not in report

    def test_verbose_lists_datasets(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "Wolbachia", str(tmp_path / "proj1"), "reg1")
            cat.upsert_dataset(_sidecar("SRR1"))
            cat.record_usage("SRR1", "proj1", "wMel", "downloaded")

        rc = StoreStatusCommand().execute(_status_args(data_root=str(root), json=True, verbose=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["datasets_list"] == [{"accession": "SRR1", "state": "complete", "bytes": 100, "projects": 1}]

    def test_stale_project_detected_when_path_missing(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        missing_path = tmp_path / "gone"
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "Wolbachia", str(missing_path), "reg1")

        rc = StoreStatusCommand().execute(_status_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["stale_projects"] == ["proj1"]

    def test_text_output_includes_store_header(self, tmp_path, capsys):
        root = tmp_path / "store"
        init_store(root)

        rc = StoreStatusCommand().execute(_status_args(data_root=str(root)))
        out = capsys.readouterr().out

        assert rc == 0
        assert "Store" in out


class TestStoreReindexCommand:
    def test_command_properties(self):
        cmd = StoreReindexCommand()
        assert cmd.name == "store_reindex"
        assert cmd.group == "Store"

    def test_no_store_configured_returns_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = StoreReindexCommand().execute(_reindex_args(registry=str(tmp_path / "metaquest_registry.json")))
        assert rc == 1

    def test_rebuilds_from_sidecars_and_prints_count(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        for acc in ("SRR1", "SRR2"):
            sc_path = sidecar_path(paths, acc)
            write_sidecar(sc_path, _sidecar(acc))

        rc = StoreReindexCommand().execute(_reindex_args(data_root=str(root)))
        out = capsys.readouterr().out

        assert rc == 0
        assert "2" in out

        with Catalog(paths) as cat:
            assert cat.get_dataset("SRR1") is not None
            assert cat.get_dataset("SRR2") is not None

    def test_reindex_drops_removed_accessions(self, tmp_path):
        root = tmp_path / "store"
        paths = init_store(root)
        write_sidecar(sidecar_path(paths, "SRR1"), _sidecar("SRR1"))
        StoreReindexCommand().execute(_reindex_args(data_root=str(root)))

        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR-stale"))

        StoreReindexCommand().execute(_reindex_args(data_root=str(root)))

        with Catalog(paths) as cat:
            assert cat.get_dataset("SRR-stale") is None
            assert cat.get_dataset("SRR1") is not None


def _write_fastq_gz(path, text="@r\nACGT\n+\nIIII\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def _sidecar_matching_disk(acc_dir, accession, state="complete", reads=5):
    """A sidecar whose one file record has the accession's real on-disk size, so a plain
    `store_verify` (no --md5) reports it as healthy."""
    file_path = acc_dir / f"{accession}.fastq.gz"
    sidecar = _sidecar(accession, state=state)
    sidecar.files = [{"name": file_path.name, "bytes": file_path.stat().st_size, "md5": "ignored", "reads": reads}]
    return sidecar


class TestStoreAdoptCommand:
    def test_command_properties(self):
        cmd = StoreAdoptCommand()
        assert cmd.name == "store_adopt"
        assert cmd.group == "Store"

    def test_no_store_configured_returns_1(self, tmp_path, monkeypatch, capsys):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreAdoptCommand().execute(_adopt_args(registry=str(project_dir / "metaquest_registry.json")))
        out = capsys.readouterr().out

        assert rc == 1
        assert "store_init" in out

    def test_adopts_moves_links_and_records_registry(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        init_store(root)
        project_dir = tmp_path / "project"
        (project_dir / "fastq" / "SRR1").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1" / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)

        registry_path = project_dir / "metaquest_registry.json"
        rc = StoreAdoptCommand().execute(_adopt_args(data_root=str(root), registry=str(registry_path)))

        assert rc == 0
        assert (project_dir / "fastq" / "SRR1").is_symlink()

        paths = store_paths(root)
        assert sra_dir(paths, "SRR1").is_dir()
        with Catalog(paths) as cat:
            assert cat.get_dataset("SRR1") is not None

        registry = load_registry(registry_path)
        assert registry.datasets["SRR1"]["download"]["state"] == "downloaded"
        assert registry.datasets["SRR1"]["download"]["source"] == "store"
        assert registry.store["linked"] == ["SRR1"]

    def test_dry_run_changes_nothing(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        init_store(root)
        project_dir = tmp_path / "project"
        (project_dir / "fastq" / "SRR1").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1" / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)

        registry_path = project_dir / "metaquest_registry.json"
        rc = StoreAdoptCommand().execute(_adopt_args(data_root=str(root), registry=str(registry_path), dry_run=True))

        assert rc == 0
        assert (project_dir / "fastq" / "SRR1" / "SRR1.fastq").is_file()
        assert not (project_dir / "fastq" / "SRR1").is_symlink()
        assert not registry_path.exists()

    def test_copy_mode_leaves_project_folder_untouched_and_unlinked(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        init_store(root)
        project_dir = tmp_path / "project"
        entry = project_dir / "fastq" / "SRR1"
        entry.mkdir(parents=True)
        (entry / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)

        registry_path = project_dir / "metaquest_registry.json"
        rc = StoreAdoptCommand().execute(_adopt_args(data_root=str(root), registry=str(registry_path), move=False))

        assert rc == 0
        # --copy: the project's own folder is left exactly as it was, not linked.
        assert entry.is_dir() and not entry.is_symlink()
        assert (entry / "SRR1.fastq").is_file()

        paths = store_paths(root)
        assert sra_dir(paths, "SRR1").is_dir()

        # No registry write is needed: the project's own download record did not change.
        assert not registry_path.exists()


class TestStoreVerifyCommand:
    def test_command_properties(self):
        cmd = StoreVerifyCommand()
        assert cmd.name == "store_verify"
        assert cmd.group == "Store"

    def test_no_store_configured_returns_1(self, tmp_path, monkeypatch, capsys):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreVerifyCommand().execute(_verify_args(registry=str(project_dir / "metaquest_registry.json")))
        assert rc == 1

    def test_verify_healthy_dataset_reports_ok(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")
        sidecar = _sidecar_matching_disk(acc_dir, "SRR1")
        write_sidecar(sidecar_path(paths, "SRR1"), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreVerifyCommand().execute(
            _verify_args(data_root=str(root), registry=str(project_dir / "metaquest_registry.json"))
        )
        out = capsys.readouterr().out

        assert rc == 0
        assert "SRR1" in out
        assert "ok" in out

    def test_verify_flags_truncated_dataset_and_fix_state_updates_sidecar(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        # Only 1 read on disk, but the sidecar's recorded NCBI spot count says there should be 5.
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz", text="@r\nACGT\n+\nIIII\n")
        sidecar = _sidecar("SRR1", state="complete")
        sidecar.files = [
            {
                "name": "SRR1.fastq.gz",
                "bytes": acc_dir.joinpath("SRR1.fastq.gz").stat().st_size,
                "md5": "ignored",
                "reads": 1,
            }
        ]
        write_sidecar(sidecar_path(paths, "SRR1"), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreVerifyCommand().execute(
            _verify_args(
                data_root=str(root), registry=str(project_dir / "metaquest_registry.json"), spots=True, fix_state=True
            )
        )
        out = capsys.readouterr().out

        assert rc == 1
        assert "truncated" in out

        fixed = read_sidecar(sidecar_path(paths, "SRR1"))
        assert fixed.state == "partial"
        assert fixed.completeness["verdict"] == "truncated"

        with Catalog(paths) as cat:
            row = cat.get_dataset("SRR1")
            assert row["state"] == "partial"

    def test_verify_flags_bad_md5_as_failed_regardless_of_spots(self, tmp_path, monkeypatch, capsys):
        """A bytes/md5 mismatch always wins over the spots check: a corrupted file can still
        happen to contain the right number of reads, so --fix-state must mark it failed, not
        complete."""
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz", text="@r\nACGT\n+\nIIII\n")
        sidecar = _sidecar("SRR1", state="complete")
        sidecar.files = [
            {
                "name": "SRR1.fastq.gz",
                "bytes": acc_dir.joinpath("SRR1.fastq.gz").stat().st_size,
                # Wrong md5 on purpose; read count/spots still match (ncbi.spots defaults to 5).
                "md5": "0" * 32,
                "reads": 5,
            }
        ]
        write_sidecar(sidecar_path(paths, "SRR1"), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreVerifyCommand().execute(
            _verify_args(
                data_root=str(root),
                registry=str(project_dir / "metaquest_registry.json"),
                md5=True,
                spots=True,
                fix_state=True,
            )
        )
        out = capsys.readouterr().out

        assert rc == 1
        assert "corrupt" in out

        fixed = read_sidecar(sidecar_path(paths, "SRR1"))
        assert fixed.state == "failed"
        assert fixed.error and "md5 mismatch" in fixed.error

        with Catalog(paths) as cat:
            row = cat.get_dataset("SRR1")
            assert row["state"] == "failed"

    def test_verify_specific_accessions_only(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        for acc in ("SRR1", "SRR2"):
            acc_dir = sra_dir(paths, acc)
            _write_fastq_gz(acc_dir / f"{acc}.fastq.gz")
            sidecar = _sidecar_matching_disk(acc_dir, acc)
            write_sidecar(sidecar_path(paths, acc), sidecar)
            with catalog_write(paths) as cat:
                cat.upsert_dataset(sidecar)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreVerifyCommand().execute(
            _verify_args(
                accessions=["SRR2"], data_root=str(root), registry=str(project_dir / "metaquest_registry.json")
            )
        )
        out = capsys.readouterr().out

        assert rc == 0
        assert "SRR2" in out
        assert "SRR1" not in out

    def test_missing_dataset_fails(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "store"
        init_store(root)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreVerifyCommand().execute(
            _verify_args(
                accessions=["SRR-ghost"],
                data_root=str(root),
                registry=str(project_dir / "metaquest_registry.json"),
            )
        )
        out = capsys.readouterr().out

        assert rc == 1
        assert "missing" in out

    def test_corrupt_gzip_reported_as_corrupt_without_crashing(self, tmp_path, monkeypatch, capsys):
        """A truncated/corrupt gzip file makes read-count verification raise inside the data
        layer; --spots must catch that and report it, not crash the whole command."""
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        acc_dir.mkdir(parents=True)
        # Looks like a gzip file (magic bytes) but is not valid gzip data.
        (acc_dir / "SRR1.fastq.gz").write_bytes(b"\x1f\x8b\x00not-really-gzip")
        sidecar = _sidecar_matching_disk(acc_dir, "SRR1")
        write_sidecar(sidecar_path(paths, "SRR1"), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreVerifyCommand().execute(
            _verify_args(data_root=str(root), registry=str(project_dir / "metaquest_registry.json"), spots=True)
        )
        out = capsys.readouterr().out

        assert rc == 1
        assert "corrupt" in out


class TestStoreLinkCommand:
    def test_command_properties(self):
        cmd = StoreLinkCommand()
        assert cmd.name == "store_link"
        assert cmd.group == "Store"

    def test_no_store_configured_returns_1(self, tmp_path, monkeypatch, capsys):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreLinkCommand().execute(_link_args(["SRR1"], registry=str(project_dir / "metaquest_registry.json")))
        assert rc == 1

    def test_links_accession_and_updates_registry(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")
        sidecar = _sidecar("SRR1")
        write_sidecar(sidecar_path(paths, "SRR1"), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"

        rc = StoreLinkCommand().execute(
            _link_args(
                ["SRR1"], data_root=str(root), registry=str(registry_path), fastq_folder=str(project_dir / "fastq")
            )
        )

        assert rc == 0
        assert (project_dir / "fastq" / "SRR1").is_symlink()

        registry = load_registry(registry_path)
        assert registry.datasets["SRR1"]["download"]["state"] == "downloaded"
        assert registry.datasets["SRR1"]["download"]["source"] == "store"
        assert registry.store["linked"] == ["SRR1"]

    def test_link_failure_reports_nonzero(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        init_store(root)  # no SRR1 dataset in the store

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"

        rc = StoreLinkCommand().execute(
            _link_args(
                ["SRR1"], data_root=str(root), registry=str(registry_path), fastq_folder=str(project_dir / "fastq")
            )
        )

        assert rc == 1
        assert not registry_path.exists()


class TestStoreUnlinkCommand:
    def test_command_properties(self):
        cmd = StoreUnlinkCommand()
        assert cmd.name == "store_unlink"
        assert cmd.group == "Store"

    def test_unlinks_symlink_and_updates_registry(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        _write_fastq_gz(acc_dir / "SRR1.fastq.gz")
        sidecar = _sidecar("SRR1")
        write_sidecar(sidecar_path(paths, "SRR1"), sidecar)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(sidecar)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"
        fastq_dir = project_dir / "fastq"

        StoreLinkCommand().execute(
            _link_args(["SRR1"], data_root=str(root), registry=str(registry_path), fastq_folder=str(fastq_dir))
        )

        rc = StoreUnlinkCommand().execute(
            _unlink_args(["SRR1"], registry=str(registry_path), fastq_folder=str(fastq_dir))
        )

        assert rc == 0
        assert not (fastq_dir / "SRR1").exists()

        registry = load_registry(registry_path)
        assert registry.datasets["SRR1"]["download"]["state"] == "missing"
        assert registry.store["linked"] == []

    def test_refuses_a_real_directory(self, tmp_path, monkeypatch):
        project_dir = tmp_path / "project"
        (project_dir / "fastq" / "SRR1").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1" / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"
        fastq_dir = project_dir / "fastq"

        rc = StoreUnlinkCommand().execute(
            _unlink_args(["SRR1"], registry=str(registry_path), fastq_folder=str(fastq_dir))
        )

        assert rc == 1
        assert (fastq_dir / "SRR1").is_dir() and not (fastq_dir / "SRR1").is_symlink()
        assert (fastq_dir / "SRR1" / "SRR1.fastq").is_file()
