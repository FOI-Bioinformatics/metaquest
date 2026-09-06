"""Tests for the `store_gc` CLI command.

Every test runs under tmp_path and monkeypatches HOME/XDG_CONFIG_HOME/METAQUEST_DATA so
nothing here reads or writes the real user config, matching the isolation pattern used in
tests/test_cli_store.py and tests/test_store_resolve.py.
"""

import argparse
import json

import pytest

from metaquest.cli.commands.store import StoreGcCommand
from metaquest.core.constants import STORE_ENV
from metaquest.store.catalog import Catalog, catalog_write
from metaquest.store.layout import init_store, sra_dir
from metaquest.store.sidecar import Sidecar


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv(STORE_ENV, raising=False)
    yield


def _gc_args(**overrides):
    base = dict(
        data_root=None,
        registry=None,
        dry_run=False,
        yes=False,
        older_than=None,
        keep_partial=False,
        include_stale=False,
        json=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _sidecar(accession="SRR1", state="complete", downloaded="2026-09-06T00:00:00+00:00"):
    return Sidecar(
        accession=accession,
        state=state,
        layout="SINGLE",
        downloaded=downloaded,
        tool="fasterq-dump",
        tool_version="3.0.0",
        compression="gzip",
        files=[{"name": f"{accession}.fastq.gz", "bytes": 100, "md5": "a" * 32, "reads": 5}],
        reads_per_mate=5,
        bases_total=500,
        ncbi={"spots": 5, "bases": 500, "size": 1000, "layout": "SINGLE", "files": []},
        completeness={"method": "spots", "ratio": 1.0, "verdict": "complete"},
    )


def _write_dataset_dir(paths, accession):
    acc_dir = sra_dir(paths, accession)
    acc_dir.mkdir(parents=True, exist_ok=True)
    (acc_dir / f"{accession}.fastq.gz").write_bytes(b"x" * 100)
    return acc_dir


class TestStoreGcCommand:
    def test_command_properties(self):
        cmd = StoreGcCommand()
        assert cmd.name == "store_gc"
        assert cmd.group == "Store"

    def test_no_store_configured_returns_1(self, tmp_path, monkeypatch, capsys):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        rc = StoreGcCommand().execute(_gc_args(registry=str(project_dir / "metaquest_registry.json")))
        assert rc == 1

    def test_unused_dataset_listed_with_bytes(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        _write_dataset_dir(paths, "SRR1")
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR1"))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        accessions = {row["accession"]: row for row in report["datasets"]}
        assert accessions["SRR1"]["bytes"] == 100
        assert accessions["SRR1"]["reason"] == "unused"

    def test_dataset_used_only_by_stale_project_needs_include_stale(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        _write_dataset_dir(paths, "SRR1")
        stale_dir = tmp_path / "stale_proj"
        stale_dir.mkdir()
        with catalog_write(paths) as cat:
            cat.upsert_project("stale1", "OldProject", str(stale_dir), str(stale_dir / "metaquest_registry.json"))
            cat.upsert_dataset(_sidecar("SRR1"))
            cat.record_usage("SRR1", "stale1", "wMel", "downloaded")

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        # A project is "stale" whenever its registry is not a file on this machine, which is
        # true of every project on another workstation of a shared store: by default such a
        # dataset is kept and only reported.
        assert rc == 0
        assert report["datasets"] == []
        kept = {row["accession"]: row for row in report["kept_stale"]}
        assert "OldProject" in kept["SRR1"]["reason"]

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), include_stale=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        accessions = {row["accession"]: row for row in report["datasets"]}
        assert "OldProject" in accessions["SRR1"]["reason"]
        assert "stale" in accessions["SRR1"]["reason"]

    def test_live_linked_dataset_never_listed(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        _write_dataset_dir(paths, "SRR1")
        live_dir = tmp_path / "live_proj"
        live_dir.mkdir()
        registry_file = live_dir / "metaquest_registry.json"
        registry_file.write_text(json.dumps({"project": {"id": "live1"}}))
        with catalog_write(paths) as cat:
            cat.upsert_project("live1", "LiveProject", str(live_dir), str(registry_file))
            cat.upsert_dataset(_sidecar("SRR1"))
            cat.record_usage("SRR1", "live1", "wMel", "downloaded")

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["datasets"] == []

    def test_keep_partial_spares_partial_datasets(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        _write_dataset_dir(paths, "SRR1")
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR1", state="partial"))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), keep_partial=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["datasets"] == []

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), keep_partial=False, json=True))
        report = json.loads(capsys.readouterr().out)
        assert len(report["datasets"]) == 1

    def test_older_than_filters_recent_datasets(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        _write_dataset_dir(paths, "SRR1")
        from datetime import datetime, timezone

        recent = datetime.now(timezone.utc).isoformat()
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR1", downloaded=recent))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), older_than=30, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["datasets"] == []

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)
        assert len(report["datasets"]) == 1

    def test_leftovers_listed_with_sizes(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        (paths.tmp / "SRR9_temp").mkdir(parents=True)
        (paths.tmp / "SRR9_temp" / "partial.fastq").write_bytes(b"x" * 50)
        (paths.tmp / "SRR8_adopt").mkdir(parents=True)
        (paths.tmp / "SRR8_adopt" / "SRR8.fastq").write_bytes(b"y" * 30)
        (paths.tmp / ".sra-cache" / "SRR7.sra").parent.mkdir(parents=True, exist_ok=True)
        (paths.tmp / ".sra-cache" / "SRR7.sra").write_bytes(b"z" * 20)
        (paths.sra / ".sra-cache").mkdir(parents=True, exist_ok=True)
        (paths.sra / ".sra-cache" / "leftover.sra").write_bytes(b"w" * 10)

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        total_bytes = sum(entry["bytes"] for entry in report["leftovers"])
        assert total_bytes == 50 + 30 + 20 + 10
        assert len(report["leftovers"]) == 4
        assert all(entry["reason"] == "leftover" for entry in report["leftovers"])

    def test_dry_run_deletes_nothing(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = _write_dataset_dir(paths, "SRR1")
        (paths.tmp / "SRR9_temp").mkdir(parents=True)
        (paths.tmp / "SRR9_temp" / "f").write_bytes(b"x" * 10)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR1"))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root)))
        assert rc == 0

        assert acc_dir.is_dir()
        assert (paths.tmp / "SRR9_temp").is_dir()
        with Catalog(paths) as cat:
            assert cat.get_dataset("SRR1") is not None

    def test_yes_removes_folders_catalogue_rows_and_leftovers(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = _write_dataset_dir(paths, "SRR1")
        (paths.tmp / "SRR9_temp").mkdir(parents=True)
        (paths.tmp / "SRR9_temp" / "f").write_bytes(b"x" * 10)
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR1"))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), dry_run=False, yes=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert not acc_dir.exists()
        assert not (paths.tmp / "SRR9_temp").exists()
        with Catalog(paths) as cat:
            assert cat.get_dataset("SRR1") is None
        assert "SRR1" in report["removed_datasets"]
        assert any("SRR9_temp" in entry for entry in report["removed_leftovers"])

    def test_dry_run_and_yes_together_refuses_and_removes_nothing(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = _write_dataset_dir(paths, "SRR1")
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR1"))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), dry_run=True, yes=True))

        assert rc == 1
        assert acc_dir.is_dir()
        with Catalog(paths) as cat:
            assert cat.get_dataset("SRR1") is not None

    def test_live_project_symlink_without_usage_row_keeps_dataset(self, tmp_path, capsys):
        """A dataset a live project still symlinks to must never be removed, even when it has
        no usage row at all (usage tracking predating that project, or a failed hook)."""
        root = tmp_path / "store"
        paths = init_store(root)
        acc_dir = _write_dataset_dir(paths, "SRR1")

        project_dir = tmp_path / "live_proj"
        (project_dir / "fastq").mkdir(parents=True)
        (project_dir / "fastq" / "SRR1").symlink_to(acc_dir, target_is_directory=True)
        registry_file = project_dir / "metaquest_registry.json"
        registry_file.write_text(json.dumps({"project": {"id": "live1"}}))

        with catalog_write(paths) as cat:
            cat.upsert_project("live1", "LiveProject", str(project_dir), str(registry_file))
            cat.upsert_dataset(_sidecar("SRR1"))
            # Deliberately no usage row recorded for SRR1.

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["datasets"] == []
        still_linked = {row["accession"]: row for row in report["still_linked"]}
        assert "LiveProject" in still_linked["SRR1"]["reason"]

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), yes=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert acc_dir.is_dir()
        assert "SRR1" not in report["removed_datasets"]
        with Catalog(paths) as cat:
            assert cat.get_dataset("SRR1") is not None

    def test_stale_projects_named_in_report(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        stale_dir = tmp_path / "stale_proj"
        stale_dir.mkdir()
        missing_registry = stale_dir / "metaquest_registry.json"
        with catalog_write(paths) as cat:
            cat.upsert_project("stale1", "OldProject", str(stale_dir), str(missing_registry))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        row = report["stale_projects"][0]
        assert len(report["stale_projects"]) == 1
        assert row["project_id"] == "stale1"
        assert row["name"] == "OldProject"
        assert row["registry"] == str(missing_registry)
        assert row["reason"] == "registry missing"


class TestStoreGcRespectsLocksAndPlaceholders:
    """Never remove what another run is working on, or a row that stands for no files."""

    @staticmethod
    def _hold(paths, accession):
        import json as json_module

        from metaquest.store.layout import lock_path

        lock = lock_path(paths, accession)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json_module.dumps({"pid": 4242, "host": "otherhost", "started": "2026-01-01T00:00:00+00:00"}))
        return lock

    def test_a_locked_dataset_is_kept_and_reported_as_in_use(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        _write_dataset_dir(paths, "SRR1")
        with catalog_write(paths) as cat:
            cat.upsert_dataset(_sidecar("SRR1"))
        self._hold(paths, "SRR1")

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), yes=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["datasets"] == []
        assert [row["accession"] for row in report["in_use"]] == ["SRR1"]
        assert sra_dir(paths, "SRR1").is_dir()

    def test_leftovers_of_a_locked_accession_are_kept(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        building = paths.tmp / "SRR1_temp"
        building.mkdir(parents=True)
        (building / "SRR1_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        cached = paths.tmp / ".sra-cache" / "SRR1"
        cached.mkdir(parents=True)
        (cached / "SRR1.sra").write_bytes(b"x" * 10)
        self._hold(paths, "SRR1")

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), yes=True, json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["leftovers"] == []
        assert building.is_dir()
        assert cached.is_dir()

    def test_a_placeholder_row_is_never_a_candidate(self, tmp_path, capsys):
        """A usage row for an accession that is not catalogued yet inserts a state="unknown"
        placeholder; it stands for no files, so gc must not offer to remove it."""
        root = tmp_path / "store"
        paths = init_store(root)
        proj = tmp_path / "proj"
        proj.mkdir()
        with catalog_write(paths) as cat:
            cat.upsert_project("p1", "Proj", str(proj), str(proj / "metaquest_registry.json"))
            cat.record_usage("SRR-PLACEHOLDER", "p1", "wMel", "linked")

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert [row["accession"] for row in report["datasets"]] == []

    def test_stale_projects_report_their_host_and_reason(self, tmp_path, capsys):
        root = tmp_path / "store"
        paths = init_store(root)
        gone = tmp_path / "gone"
        gone.mkdir()
        with catalog_write(paths) as cat:
            cat.upsert_project("p1", "Gone", str(gone), str(gone / "metaquest_registry.json"))

        rc = StoreGcCommand().execute(_gc_args(data_root=str(root), json=True))
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        row = report["stale_projects"][0]
        assert row["reason"] == "registry missing"
        assert row["hostname"]
