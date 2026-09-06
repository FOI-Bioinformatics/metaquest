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
    StoreUsageCommand,
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
        lock_wait=0.0,
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
    base = dict(
        accessions=list(accessions),
        fastq_folder="fastq",
        registry=None,
        data_root=None,
        link_mode="auto",
        accept_partial=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _unlink_args(accessions, **overrides):
    base = dict(accessions=list(accessions), fastq_folder="fastq", registry=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _usage_args(**overrides):
    base = dict(
        data_root=None,
        registry=None,
        accession=None,
        project=None,
        organism=None,
        unused=False,
        bytes_by_organism=False,
        json=False,
    )
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

    def test_store_status_counts_project_after_init(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "store"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"

        assert StoreInitCommand().execute(_init_args(root, project_dir, registry=str(registry_path))) == 0

        rc = StoreStatusCommand().execute(_status_args(data_root=str(root), json=True, registry=str(registry_path)))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["projects"] == 1

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
        project_dir = tmp_path / "proj1"
        project_dir.mkdir()
        registry_file = project_dir / "metaquest_registry.json"
        registry_file.write_text(json.dumps({"project": {"id": "proj1"}}))
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "Wolbachia", str(project_dir), str(registry_file))
            cat.upsert_dataset(_sidecar("SRR1", state="complete"))
            cat.upsert_dataset(_sidecar("SRR2", state="partial"))
            cat.record_usage("SRR1", "proj1", "wMel", "downloaded")

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

    def test_stale_project_detected_when_registry_missing(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        missing_path = tmp_path / "gone"
        missing_registry = tmp_path / "gone" / "metaquest_registry.json"
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "Wolbachia", str(missing_path), str(missing_registry))

        rc = StoreStatusCommand().execute(_status_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert len(report["stale_projects"]) == 1
        row = report["stale_projects"][0]
        assert (row["project_id"], row["name"], row["registry"]) == ("proj1", "Wolbachia", str(missing_registry))
        assert row["reason"] == "registry missing"

    def test_stale_project_detected_when_registry_id_differs(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        project_dir = tmp_path / "proj1"
        project_dir.mkdir()
        registry_file = project_dir / "metaquest_registry.json"
        registry_file.write_text(json.dumps({"project": {"id": "some-other-id"}}))
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "Wolbachia", str(project_dir), str(registry_file))

        rc = StoreStatusCommand().execute(_status_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert len(report["stale_projects"]) == 1
        row = report["stale_projects"][0]
        assert (row["project_id"], row["name"], row["registry"]) == ("proj1", "Wolbachia", str(registry_file))
        # A registry is there; it simply belongs to a different project now. Saying "registry
        # missing" for that case sends the reader looking for a file that exists.
        assert row["reason"] == "project id differs"

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

    def test_adopt_copies_sidecar_completeness_and_records_usage(self, tmp_path, monkeypatch):
        """A newly adopted accession's registry record carries its sidecar's completeness
        verdict, and the store catalogue gets a 'linked' usage row for this project."""
        root = tmp_path / "store"
        init_store(root)
        project_dir = tmp_path / "project"
        (project_dir / "fastq" / "SRR1").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1" / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"

        assert StoreInitCommand().execute(_init_args(root, project_dir, registry=str(registry_path))) == 0

        rc = StoreAdoptCommand().execute(_adopt_args(data_root=str(root), registry=str(registry_path)))
        assert rc == 0

        registry = load_registry(registry_path)
        complete = registry.datasets["SRR1"]["download"]["complete"]
        # No metadata XML is present, so the sidecar's own verify against NCBI's spot count
        # is unverified; the registry record must carry exactly that verdict, not silently
        # invent one.
        assert complete["verdict"] == "unverified"
        assert complete["expected_spots"] is None
        assert complete["reads_r1"] == 1

        paths = store_paths(root)
        with Catalog(paths) as cat:
            row = cat.conn.execute(
                "SELECT stage FROM usage WHERE accession = ? AND project_id = ?",
                ("SRR1", registry.project["id"]),
            ).fetchone()
        assert row["stage"] == "linked"

    def test_catalog_failure_leaves_adopt_outcome_unchanged(self, tmp_path, monkeypatch):
        """A broken catalogue write never changes store_adopt's registry outcome or exit code."""
        root = tmp_path / "store"
        init_store(root)
        project_dir = tmp_path / "project"
        (project_dir / "fastq" / "SRR1").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1" / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"

        assert StoreInitCommand().execute(_init_args(root, project_dir, registry=str(registry_path))) == 0

        with patch("metaquest.store.usage.catalog_write", side_effect=RuntimeError("locked")):
            rc = StoreAdoptCommand().execute(_adopt_args(data_root=str(root), registry=str(registry_path)))

        assert rc == 0
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

    def test_link_copies_sidecar_completeness_and_records_usage(self, tmp_path, monkeypatch):
        """A linked accession's registry record carries the store sidecar's completeness
        verdict, and the store catalogue gets a 'linked' usage row for this project."""
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

        assert StoreInitCommand().execute(_init_args(root, project_dir, registry=str(registry_path))) == 0

        rc = StoreLinkCommand().execute(
            _link_args(
                ["SRR1"], data_root=str(root), registry=str(registry_path), fastq_folder=str(project_dir / "fastq")
            )
        )
        assert rc == 0

        registry = load_registry(registry_path)
        complete = registry.datasets["SRR1"]["download"]["complete"]
        assert complete == {"verdict": "complete", "ratio": 1.0, "expected_spots": 5, "reads_r1": 5}

        with Catalog(paths) as cat:
            row = cat.conn.execute(
                "SELECT stage FROM usage WHERE accession = ? AND project_id = ?",
                ("SRR1", registry.project["id"]),
            ).fetchone()
        assert row["stage"] == "linked"

    def test_catalog_failure_leaves_link_outcome_unchanged(self, tmp_path, monkeypatch):
        """A broken catalogue write never changes store_link's registry outcome or exit code."""
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

        assert StoreInitCommand().execute(_init_args(root, project_dir, registry=str(registry_path))) == 0

        with patch("metaquest.store.usage.catalog_write", side_effect=RuntimeError("locked")):
            rc = StoreLinkCommand().execute(
                _link_args(
                    ["SRR1"],
                    data_root=str(root),
                    registry=str(registry_path),
                    fastq_folder=str(project_dir / "fastq"),
                )
            )

        assert rc == 0
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


class TestStoreUsageCommand:
    def test_command_properties(self):
        cmd = StoreUsageCommand()
        assert cmd.name == "store_usage"
        assert cmd.group == "Store"

    def test_configure_parser_requires_one_selector(self):
        cmd = StoreUsageCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_configure_parser_rejects_two_selectors(self):
        cmd = StoreUsageCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--accession", "SRR1", "--unused"])

    def test_no_store_configured_returns_1(self, tmp_path, monkeypatch, capsys):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreUsageCommand().execute(
            _usage_args(unused=True, registry=str(project_dir / "metaquest_registry.json"))
        )
        assert rc == 1

    def _seed(self, tmp_path):
        root = tmp_path / "store"
        paths = init_store(root)
        proj_a_dir = tmp_path / "proja"
        proj_a_dir.mkdir()
        proj_b_dir = tmp_path / "projb"
        proj_b_dir.mkdir()
        with catalog_write(paths) as cat:
            cat.upsert_project("proja", "Wolbachia", str(proj_a_dir), str(proj_a_dir / "metaquest_registry.json"))
            cat.upsert_project("projb", "Rickettsia", str(proj_b_dir), str(proj_b_dir / "metaquest_registry.json"))
            cat.upsert_dataset(_sidecar("SRR1"))
            cat.upsert_dataset(_sidecar("SRR2"))
            cat.record_usage("SRR1", "proja", "wMel", "downloaded")
            cat.record_usage("SRR1", "projb", "wMel", "analysed")
            cat.record_usage("SRR2", "proja", "wRi", "downloaded")
        return root

    def test_selector_by_accession(self, tmp_path, capsys):
        root = self._seed(tmp_path)
        rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), accession="SRR1", json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["selector"] == "accession"
        project_ids = {row["project_id"] for row in report["rows"]}
        assert project_ids == {"proja", "projb"}
        for row in report["rows"]:
            assert set(row) == {"project_name", "project_id", "genome_id", "stage", "first_used", "last_used"}

    def test_selector_by_project_name(self, tmp_path, capsys):
        root = self._seed(tmp_path)
        rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), project="Wolbachia", json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["selector"] == "project"
        accessions = {row["accession"] for row in report["rows"]}
        assert accessions == {"SRR1", "SRR2"}

    def test_selector_by_project_id(self, tmp_path, capsys):
        root = self._seed(tmp_path)
        rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), project="projb", json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        accessions = {row["accession"] for row in report["rows"]}
        assert accessions == {"SRR1"}

    def test_selector_by_ambiguous_project_name_exits_1(self, tmp_path, caplog):
        root = self._seed(tmp_path)
        paths = store_paths(root)
        with catalog_write(paths) as cat:
            proj_c_dir = tmp_path / "projc"
            proj_c_dir.mkdir()
            cat.upsert_project("projc", "Wolbachia", str(proj_c_dir), str(proj_c_dir / "metaquest_registry.json"))

        with caplog.at_level("ERROR"):
            rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), project="Wolbachia"))

        assert rc == 1
        assert any("proja" in message and "projc" in message for message in caplog.messages)

    def test_selector_by_organism(self, tmp_path, capsys):
        root = self._seed(tmp_path)
        rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), organism="wMel", json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["selector"] == "organism"
        accessions = {row["accession"] for row in report["rows"]}
        assert accessions == {"SRR1"}

    def test_selector_unused(self, tmp_path, capsys):
        root = self._seed(tmp_path)
        paths = store_paths(root)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR3"))

        rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), unused=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["selector"] == "unused"
        assert report["rows"] == [{"accession": "SRR3", "state": "complete", "bytes": 100}]

    def test_selector_bytes_by_organism(self, tmp_path, capsys):
        root = self._seed(tmp_path)
        rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), bytes_by_organism=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["selector"] == "bytes-by-organism"
        by_genome = {row["genome_id"]: row for row in report["rows"]}
        assert by_genome["wMel"]["datasets"] == 1
        assert by_genome["wMel"]["bytes"] == 100
        assert by_genome["wRi"]["datasets"] == 1
        assert by_genome["wRi"]["bytes"] == 100

    def test_text_output_lists_projects(self, tmp_path, capsys):
        root = self._seed(tmp_path)
        rc = StoreUsageCommand().execute(_usage_args(data_root=str(root), accession="SRR1"))
        out = capsys.readouterr().out

        assert rc == 0
        assert "proja" in out and "projb" in out


class TestStoreInitRebinding:
    """store_init is run again: keep what the project already links, and say what changed."""

    def test_rebinding_to_another_root_preserves_linked_and_warns(self, tmp_path, monkeypatch, caplog):
        import logging

        first = tmp_path / "store-a"
        second = tmp_path / "store-b"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"

        assert StoreInitCommand().execute(_init_args(first, project_dir)) == 0
        registry = load_registry(registry_path)
        registry.store["linked"] = ["SRR1", "SRR2"]
        from metaquest.data.registry import save_registry

        save_registry(registry, registry_path)

        with caplog.at_level(logging.WARNING):
            assert StoreInitCommand().execute(_init_args(second, project_dir)) == 0

        registry = load_registry(registry_path)
        assert registry.store["root"] == str(second.resolve())
        # The list of what this project links is not something store_init knows how to rebuild.
        assert registry.store["linked"] == ["SRR1", "SRR2"]
        assert any("was bound to" in record.message for record in caplog.records)

    def test_refuses_a_non_empty_folder_that_is_not_a_store(self, tmp_path, monkeypatch):
        home_like = tmp_path / "documents"
        home_like.mkdir()
        (home_like / "thesis.txt").write_text("chapter one")
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreInitCommand().execute(_init_args(home_like, project_dir))

        assert rc == 1
        assert not (home_like / "sra").exists()
        assert not (home_like / "metaquest_store.json").exists()

    def test_accepts_an_empty_folder(self, tmp_path, monkeypatch):
        empty = tmp_path / "new-store"
        empty.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        assert StoreInitCommand().execute(_init_args(empty, project_dir)) == 0
        assert read_marker(empty) is not None


class TestStoreLinkGuardsIncompleteDatasets:
    def _store_with(self, tmp_path, state):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        acc_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(acc_dir / "SRR1.fastq.gz", "wt") as handle:
            handle.write("@r\nACGT\n+\nIIII\n")
        if state is not None:
            write_sidecar(sidecar_path(paths, "SRR1"), Sidecar(accession="SRR1", state=state))
        return root, paths

    @pytest.mark.parametrize("state", ["partial", "failed", None])
    def test_an_incomplete_dataset_needs_accept_partial(self, tmp_path, monkeypatch, capsys, state):
        root, paths = self._store_with(tmp_path, state)
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreLinkCommand().execute(_link_args(["SRR1"], data_root=str(root)))

        assert rc == 1
        assert not (project_dir / "fastq" / "SRR1").exists()

        rc = StoreLinkCommand().execute(_link_args(["SRR1"], data_root=str(root), accept_partial=True))

        assert rc == 0
        assert (project_dir / "fastq" / "SRR1").is_symlink()

    def test_a_complete_dataset_links_without_the_flag(self, tmp_path, monkeypatch):
        root, paths = self._store_with(tmp_path, "complete")
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        assert StoreLinkCommand().execute(_link_args(["SRR1"], data_root=str(root))) == 0
        assert (project_dir / "fastq" / "SRR1").is_symlink()


class TestLinkersAlwaysHaveAProjectIdentity:
    """A project that never ran store_init still records who it is, so gc can see its links."""

    def test_store_link_mints_a_project_identity(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = sra_dir(paths, "SRR1")
        acc_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(acc_dir / "SRR1.fastq.gz", "wt") as handle:
            handle.write("@r\nACGT\n+\nIIII\n")
        write_sidecar(sidecar_path(paths, "SRR1"), Sidecar(accession="SRR1", state="complete"))

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        registry_path = project_dir / "metaquest_registry.json"

        # No store_init: the store was found through --data-root alone.
        assert StoreLinkCommand().execute(_link_args(["SRR1"], data_root=str(root))) == 0

        registry = load_registry(registry_path)
        assert registry.project["id"]
        assert registry.project["name"] == project_dir.name
        with Catalog(paths) as catalog:
            rows = catalog.conn.execute("SELECT project_id FROM projects").fetchall()
        assert [row["project_id"] for row in rows] == [registry.project["id"]]

    def test_store_adopt_mints_a_project_identity(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        paths = init_store(root)
        project_dir = tmp_path / "project"
        (project_dir / "fastq" / "SRR1").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1" / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)

        rc = StoreAdoptCommand().execute(_adopt_args(data_root=str(root)))

        assert rc == 0
        registry = load_registry(project_dir / "metaquest_registry.json")
        assert registry.project["id"]
        with Catalog(paths) as catalog:
            assert catalog.conn.execute("SELECT COUNT(*) AS n FROM usage").fetchone()["n"] == 1


class TestStoreCommandsWithoutAMarker:
    """A store root that is not a store: the error names the rule that produced the root."""

    def test_store_status_reports_the_missing_marker(self, tmp_path, monkeypatch, caplog):
        import logging

        not_a_store = tmp_path / "not-a-store"
        not_a_store.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        with caplog.at_level(logging.ERROR):
            rc = StoreStatusCommand().execute(_status_args(data_root=str(not_a_store)))

        assert rc == 1
        assert any("no store marker" in record.message for record in caplog.records)


class TestStoreAdoptGitignoreGuard:
    def test_adopt_adds_fastq_to_gitignore(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        init_store(root)
        project_dir = tmp_path / "project"
        (project_dir / ".git").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1" / "SRR1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        monkeypatch.chdir(project_dir)

        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            assert StoreAdoptCommand().execute(_adopt_args(data_root=str(root))) == 0

        assert "fastq/" in (project_dir / ".gitignore").read_text()
