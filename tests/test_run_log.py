"""Tests for the per-project run log (``metaquest/data/run_log.py``) and its hook in ``cli/main.py``."""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from metaquest.cli.commands.blacklist import BlacklistCommand
from metaquest.cli.commands.metadata import DownloadMetadataCommand
from metaquest.cli.main import main
from metaquest.core import settings
from metaquest.core.exceptions import DataAccessError, ProcessingError, ValidationError
from metaquest.data import run_log
from metaquest.data.registry import REGISTRY_FILENAME

STARTED = datetime(2026, 10, 1, 12, 5, 1, tzinfo=timezone.utc)


@pytest.fixture
def run_log_on(monkeypatch):
    """Turn the run log on (``tests/conftest.py`` turns it off for every other test)."""
    monkeypatch.setenv(settings.SETTINGS["run_log"].env, "true")
    settings.reset_for_tests()


@pytest.fixture
def project(tmp_path):
    """A project folder holding a registry file."""
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / REGISTRY_FILENAME).write_text("{}")
    return folder


def _args(**values):
    return argparse.Namespace(**values)


def _record(project, command="sra_profile", detail=None, summary=None, exit_code=0, started=STARTED):
    args = _args(registry=None, threads=2)
    if summary or detail:
        run_log.note_run(args, summary=summary, detail=detail)
    return run_log.record_run(project, command, [command, "--threads", "2"], args, started, 2.5, exit_code)


# --- data layer ----------------------------------------------------------------------------


def test_setting_defaults_to_on_and_is_off_in_tests():
    assert settings.SETTINGS["run_log"].default is True
    assert settings.SETTINGS["run_log"].env == "METAQUEST_RUN_LOG"
    assert settings.active().run_log is False


def test_round_trip(project, run_log_on):
    args = _args(
        registry=project / REGISTRY_FILENAME,
        threads=2,
        func=print,
        _termination=object(),
        accessions=("SRR1", "SRR2"),
    )
    run_log.note_run(args, summary={"profiled": 3})
    run_log.note_run(args, summary={"failed": 1}, detail={"rows": [1, 2]})
    record = run_log.record_run(project, "sra_profile", ["sra_profile", "--threads", "2"], args, STARTED, 2.5, 0)

    assert record is not None
    assert record.run_id.startswith("20261001T120501Z-sra_profile-")
    assert record.started == "2026-10-01T12:05:01Z"
    assert record.finished == "2026-10-01T12:05:03Z"
    assert record.seconds == 2.5
    assert record.args == {"accessions": ["SRR1", "SRR2"], "registry": str(project / REGISTRY_FILENAME), "threads": 2}
    assert record.summary == {"profiled": 3, "failed": 1}
    assert record.detail == f"{record.run_id}.json"
    assert record.schema == run_log.RUN_LOG_SCHEMA

    records = run_log.read_runs(project)
    assert records == [record]
    assert run_log.RunRecord.from_dict(record.to_dict()) == record
    assert run_log.read_detail(project, record) == {"rows": [1, 2]}
    assert (run_log.runs_dir(project) / run_log.RUNS_FILE).is_file()
    assert run_log.runs_dir(project) == project / ".metaquest" / "runs"


def test_record_without_detail_writes_no_detail_file(project, run_log_on):
    record = _record(project)
    assert record.detail is None
    assert run_log.read_detail(project, record) is None
    assert sorted(p.name for p in run_log.runs_dir(project).iterdir() if p.suffix == ".json") == []


def test_read_runs_filters_by_command(project, run_log_on):
    _record(project, "sra_profile")
    _record(project, "sra_report")
    assert [r.command for r in run_log.read_runs(project)] == ["sra_profile", "sra_report"]
    assert [r.command for r in run_log.read_runs(project, "sra_report")] == ["sra_report"]
    assert run_log.read_runs(project.parent / "nowhere") == []


def test_malformed_last_line_is_skipped_with_one_warning(project, run_log_on, caplog):
    _record(project)
    _record(project)
    log = run_log.runs_dir(project) / run_log.RUNS_FILE
    with open(log, "a") as handle:
        handle.write('{"run_id": "2026')  # an interrupted append: no closing brace, no newline

    with caplog.at_level(logging.WARNING, logger="metaquest.data.run_log"):
        assert len(run_log.read_runs(project)) == 2
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    # A later append starts on a line of its own, so the new record is readable.
    third = _record(project)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="metaquest.data.run_log"):
        records = run_log.read_runs(project)
    assert len(records) == 3 and records[-1] == third
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_details_pruned_per_command(project, run_log_on):
    kept = run_log.DETAILS_KEPT_PER_COMMAND
    profiles = [_record(project, "sra_profile", detail={"n": i}) for i in range(kept + 2)]
    reports = [_record(project, "sra_report", detail={"n": i}) for i in range(2)]

    records = run_log.read_runs(project)
    assert len(records) == kept + 4
    by_id = {r.run_id: r for r in records}
    for old in profiles[:2]:
        assert by_id[old.run_id].detail is None
        assert not (run_log.runs_dir(project) / f"{old.run_id}.json").exists()
        assert run_log.read_detail(project, by_id[old.run_id]) is None
    for i, new in enumerate(profiles[2:], start=2):
        assert run_log.read_detail(project, by_id[new.run_id]) == {"n": i}
    for i, report in enumerate(reports):
        assert run_log.read_detail(project, by_id[report.run_id]) == {"n": i}

    lines = (run_log.runs_dir(project) / run_log.RUNS_FILE).read_text().splitlines()
    assert sum(1 for line in lines if json.loads(line)["detail"] is None) == 2


def test_resolve_run_selectors(project, run_log_on):
    first = _record(project, started=datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc))
    second = _record(project, started=datetime(2026, 10, 2, 9, 0, 0, tzinfo=timezone.utc))
    records = run_log.read_runs(project)

    assert run_log.resolve_run(records, first.run_id) == first
    assert run_log.resolve_run(records, "20261001") == first
    assert run_log.resolve_run(records, "latest") == second
    assert run_log.resolve_run(records, "previous") == first
    with pytest.raises(ValidationError, match="matches 2 runs"):
        run_log.resolve_run(records, "2026100")
    with pytest.raises(ValidationError, match="No run"):
        run_log.resolve_run(records, "2025")
    with pytest.raises(ValidationError, match="previous"):
        run_log.resolve_run(records[:1], "previous")
    with pytest.raises(ValidationError, match="no runs"):
        run_log.resolve_run([], "latest")


def test_new_run_id_shape():
    run_id = run_log.new_run_id("sra_profile", STARTED)
    prefix, command, suffix = run_id.split("-")
    assert (prefix, command) == ("20261001T120501Z", "sra_profile")
    assert len(suffix) == 4 and int(suffix, 16) >= 0


def test_json_safe_args_drops_private_and_masks_secrets(tmp_path):
    args = _args(api_key="SECRET", func=print, _run_summary={}, folder=tmp_path, items={1, 2}, other=object())
    safe = run_log.json_safe_args(args)
    assert safe["api_key"] == "***"
    assert "func" not in safe and "_run_summary" not in safe
    assert safe["folder"] == str(tmp_path)
    assert safe["items"] == [1, 2]
    assert isinstance(safe["other"], str)
    json.dumps(safe)
    assert run_log.json_safe_args(_args(api_key=None))["api_key"] is None


def test_api_key_masked_in_argv_and_args(project, run_log_on):
    args = _args(api_key="SECRET", email="a@b.c")
    argv = ["download_metadata", "--api-key", "SECRET", "--api-k=SECRET", "--email", "a@b.c"]
    record = run_log.record_run(project, "download_metadata", argv, args, STARTED, 1.0, 0)
    assert record.argv == ["download_metadata", "--api-key", "***", "--api-k=***", "--email", "a@b.c"]
    assert record.args["api_key"] == "***"
    assert "SECRET" not in (run_log.runs_dir(project) / run_log.RUNS_FILE).read_text()


def test_setting_off_writes_nothing(project):
    assert _record(project) is None
    assert not (project / ".metaquest").exists()


# --- CLI hook ------------------------------------------------------------------------------


@pytest.fixture
def opting_blacklist(monkeypatch):
    """``blacklist`` opted into the run log, with ``execute`` replaced by a stub returning ``codes[0]``."""
    codes = [0]

    def execute(self, args):
        if isinstance(codes[0], BaseException):
            raise codes[0]
        run_log.note_run(args, summary={"stub": True})
        return codes[0]

    monkeypatch.setattr(BlacklistCommand, "records_run", lambda self, args: True)
    monkeypatch.setattr(BlacklistCommand, "execute", execute)
    return codes


def test_cli_records_an_opting_command(project, run_log_on, opting_blacklist, monkeypatch):
    monkeypatch.chdir(project)
    opting_blacklist[0] = 0
    assert main(["blacklist", "--list"]) == 0
    opting_blacklist[0] = ProcessingError("boom")
    assert main(["blacklist", "--list", "--registry", str(project / REGISTRY_FILENAME)]) == 1

    records = run_log.read_runs(project)
    assert [(r.command, r.exit_code) for r in records] == [("blacklist", 0), ("blacklist", 1)]
    assert records[0].argv == ["blacklist", "--list"]
    assert records[0].summary == {"stub": True}
    assert records[0].seconds >= 0
    assert "func" not in records[0].args and records[0].args["list"] is True


def test_cli_finds_registry_above_working_directory(project, run_log_on, opting_blacklist, monkeypatch):
    nested = project / "sub" / "dir"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert main(["blacklist", "--list"]) == 0
    assert len(run_log.read_runs(project)) == 1
    assert not (nested / ".metaquest").exists()


def test_cli_no_registry_means_no_folder(tmp_path, run_log_on, opting_blacklist, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["blacklist", "--list"]) == 0
    assert main(["blacklist", "--list", "--registry", str(tmp_path / "missing.json")]) == 0
    assert not (tmp_path / ".metaquest").exists()


def test_cli_non_opting_command_writes_nothing(project, run_log_on, monkeypatch):
    monkeypatch.setattr(BlacklistCommand, "execute", lambda self, args: 0)
    monkeypatch.chdir(project)
    assert main(["blacklist", "--list"]) == 0
    assert not (project / ".metaquest").exists()


def test_cli_setting_off_writes_nothing(project, opting_blacklist, monkeypatch):
    monkeypatch.chdir(project)
    assert main(["blacklist", "--list"]) == 0
    assert not (project / ".metaquest").exists()


@pytest.mark.parametrize("code", [0, 1, 4])
def test_cli_unwritable_log_keeps_exit_code(project, run_log_on, opting_blacklist, monkeypatch, caplog, code):
    (project / ".metaquest").write_text("a file where the folder should be")
    monkeypatch.chdir(project)
    opting_blacklist[0] = code
    with caplog.at_level(logging.WARNING):
        assert main(["blacklist", "--list"]) == code
    assert any("run log" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_cli_run_log_error_keeps_exit_code(project, run_log_on, opting_blacklist, monkeypatch):
    def broken(*args, **kwargs):
        raise DataAccessError("lock held")

    monkeypatch.setattr(run_log, "record_run", broken)
    monkeypatch.chdir(project)
    opting_blacklist[0] = 0
    assert main(["blacklist", "--list"]) == 0


def test_cli_masks_api_key(project, run_log_on, monkeypatch):
    monkeypatch.setattr(DownloadMetadataCommand, "records_run", lambda self, args: True)
    monkeypatch.setattr(DownloadMetadataCommand, "execute", lambda self, args: 0)
    monkeypatch.chdir(project)
    assert main(["download_metadata", "--api-key", "SECRET", "--email", "a@b.c", "--dry-run"]) == 0
    (record,) = run_log.read_runs(project)
    assert record.argv[1:3] == ["--api-key", "***"]
    assert record.args["api_key"] == "***"
    text = "".join(p.read_text() for p in Path(run_log.runs_dir(project)).iterdir() if p.is_file())
    assert "SECRET" not in text


def test_records_run_defaults_to_false():
    assert BlacklistCommand().records_run(_args()) is False
