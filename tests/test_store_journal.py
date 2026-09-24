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


def test_replay_skips_usage_whose_project_was_not_restored(tmp_path, caplog):
    """A usage line for a project whose own line is missing must not abort the whole replay."""
    import logging

    paths = init_store(tmp_path / "store")
    journal.append_usage(paths, "SRR1", "pid-missing", "", "linked", "")

    with caplog.at_level(logging.WARNING):
        with catalog_write(paths) as c:
            restored = journal.replay(paths, c)
            assert restored == (0, 0)
            assert c.conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0] == 0
    assert any("Skipped 1 usage record" in r.message for r in caplog.records)


def test_replay_skips_malformed_journal_lines_without_raising(tmp_path):
    """A line that parses but is not a usable record (not an object, or missing a required
    key) is skipped rather than raised through KeyError/TypeError."""
    paths = init_store(tmp_path / "store")
    paths.journal.mkdir(parents=True, exist_ok=True)
    (paths.journal / "projects.jsonl").write_text('[1, 2, 3]\n{"name": "no id here"}\n')
    (paths.journal / "usage.jsonl").write_text('"just a string"\n{"accession": "SRR1"}\n')

    with catalog_write(paths) as c:
        assert journal.replay(paths, c) == (0, 0)


def test_catalog_write_backfills_a_pre_journal_store_once(tmp_path):
    """A pre-journal store's ``projects``/``usage`` rows are copied into the journal the first
    time ``catalog_write`` opens it afterwards (``catalog_write`` calls ``backfill_from_catalog``
    itself, right after ``migrate()``), and never duplicated on a later open."""
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        c.journal_enabled = False  # simulate a store written before the journal existed
        c.upsert_project("pid1", "proj", str(tmp_path / "proj"), "r.json")
        c.record_usage("SRR1", "pid1", "", "linked", "")
    assert not (paths.journal / "projects.jsonl").exists()

    with catalog_write(paths):
        pass  # backfill runs on entry, before this block's body

    projects = [json.loads(line) for line in (paths.journal / "projects.jsonl").read_text().splitlines()]
    usage = [json.loads(line) for line in (paths.journal / "usage.jsonl").read_text().splitlines()]
    assert [p["project_id"] for p in projects] == ["pid1"]
    assert projects[0]["name"] == "proj" and projects[0]["registry"] == "r.json"
    assert [line["accession"] for line in usage] == ["SRR1"]

    with catalog_write(paths):
        pass  # a second open must not duplicate the lines already backfilled
    assert len((paths.journal / "projects.jsonl").read_text().splitlines()) == 1
    assert len((paths.journal / "usage.jsonl").read_text().splitlines()) == 1


def test_backfill_from_catalog_is_a_noop_on_an_empty_catalog(tmp_path):
    """No project rows yet (a freshly initialised store) means nothing to backfill."""
    paths = init_store(tmp_path / "store")
    with catalog_write(paths) as c:
        assert journal.backfill_from_catalog(paths, c) == (0, 0)
    assert not (paths.journal / "projects.jsonl").exists()
