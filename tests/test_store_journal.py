import json

from metaquest.store.catalog import catalog_write
from metaquest.store.layout import init_store
from metaquest.store import journal


def test_upsert_project_and_usage_are_journaled(tmp_path):
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        c.upsert_project("pid1", "proj", str(tmp_path / "proj"), str(tmp_path / "proj" / "r.json"))
        c.record_usage("SRR1", "pid1", "GCF_1", "linked", "test")
    projects = [json.loads(line) for line in (paths.journal / "projects.jsonl").read_text().splitlines()]
    usage = [json.loads(line) for line in (paths.journal / "usage.jsonl").read_text().splitlines()]
    assert projects[0]["project_id"] == "pid1" and projects[0]["name"] == "proj"
    assert usage[0]["accession"] == "SRR1" and usage[0]["stage"] == "linked"


def test_replay_restores_projects_and_usage_into_a_fresh_catalog(tmp_path):
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        c.upsert_project("pid1", "proj", str(tmp_path / "proj"), "r.json")
        c.record_usage("SRR1", "pid1", "", "downloaded", "")
    (paths.root / "catalog.sqlite").unlink()
    with catalog_write(paths) as c:
        restored = journal.replay(paths, c)
        assert restored == (1, 1)
        assert c.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
        assert c.conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0] == 1


def test_replay_without_journal_returns_zero(tmp_path):
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        assert journal.replay(paths, c) == (0, 0)
