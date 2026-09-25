"""Tests for the output helpers on `BaseCommand` and the one-stdout-channel rule.

stdout carries a command's result (tables, JSON); stderr carries logging. The JSON-mode tests
build a project state that makes the command log a warning and then parse the whole of stdout
as one JSON document, so a stray print or a log line on stdout would fail them.

Every store test runs under tmp_path and monkeypatches HOME/XDG_CONFIG_HOME/METAQUEST_DATA so
nothing here reads or writes the real user config.
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from metaquest.cli.base import BaseCommand, emit_error_json
from metaquest.cli.commands.status import StatusCommand
from metaquest.cli.commands.store import (
    StoreGcCommand,
    StoreInitCommand,
    StoreStatusCommand,
    StoreUsageCommand,
)
from metaquest.core.constants import STORE_ENV
from metaquest.store.catalog import catalog_write
from metaquest.store.layout import init_store, store_paths
from metaquest.store.sidecar import Sidecar

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv(STORE_ENV, raising=False)
    yield


class _EmitCommand(BaseCommand):
    """A minimal command that writes whatever it is given through the helpers."""

    def __init__(self, action):
        super().__init__()
        self._action = action

    @property
    def name(self) -> str:
        return "x"

    @property
    def help(self) -> str:
        return "x"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        self._action(self)
        return 0


class TestEmitHelpers:
    def test_emit_json_writes_exactly_one_document(self, capsys):
        _EmitCommand(lambda cmd: cmd.emit_json({"a": 1, "b": [1, 2]})).execute(argparse.Namespace())

        captured = capsys.readouterr()
        assert json.loads(captured.out) == {"a": 1, "b": [1, 2]}
        assert captured.out.endswith("}\n")
        assert captured.err == ""

    def test_emit_json_is_indented_by_two(self, capsys):
        _EmitCommand(lambda cmd: cmd.emit_json({"a": 1})).execute(argparse.Namespace())

        assert capsys.readouterr().out == '{\n  "a": 1\n}\n'

    def test_emit_writes_one_newline_terminated_line(self, capsys):
        _EmitCommand(lambda cmd: cmd.emit("hello")).execute(argparse.Namespace())

        assert capsys.readouterr().out == "hello\n"

    def test_emit_without_text_writes_an_empty_line(self, capsys):
        _EmitCommand(lambda cmd: cmd.emit()).execute(argparse.Namespace())

        assert capsys.readouterr().out == "\n"

    def test_emit_error_json_writes_one_error_document(self, capsys):
        emit_error_json("something went wrong")

        captured = capsys.readouterr()
        assert json.loads(captured.out) == {"error": "something went wrong"}
        assert captured.err == ""


# ---------------------------------------------------------------- JSON commands


def _corrupt_project_registry(tmp_path: Path) -> Path:
    """A store with one registered project whose registry no longer parses.

    ``stale_projects`` (used by ``store_status`` and ``store_gc``) logs a warning for such a
    project instead of failing, which is the warning these tests rely on.
    """
    root = tmp_path / "store"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    registry = project_dir / "metaquest_registry.json"
    args = argparse.Namespace(data_root=str(root), project_name=None, set_default=False, registry=str(registry))
    assert StoreInitCommand().execute(args) == 0
    registry.write_text("{ not json")
    return root


def _store_status_args(tmp_path: Path) -> argparse.Namespace:
    root = _corrupt_project_registry(tmp_path)
    return argparse.Namespace(data_root=str(root), registry=None, json=True, verbose=True)


def _store_gc_args(tmp_path: Path) -> argparse.Namespace:
    root = _corrupt_project_registry(tmp_path)
    return argparse.Namespace(
        data_root=str(root),
        registry=None,
        dry_run=True,
        yes=False,
        older_than=None,
        keep_partial=False,
        include_stale=False,
        accept_rebuilt=False,
        json=True,
    )


def _store_usage_args(tmp_path: Path) -> argparse.Namespace:
    root = tmp_path / "store"
    paths = init_store(root)
    with catalog_write(paths) as cat:
        cat.upsert_project("proja", "Wolbachia", str(tmp_path), str(tmp_path / "metaquest_registry.json"))
        cat.upsert_dataset(Sidecar(accession="SRR1", state="complete"))
        cat.record_usage("SRR1", "proja", "wMel", "downloaded")
    return argparse.Namespace(
        data_root=str(store_paths(root).root),
        registry=None,
        accession="SRR404",
        project=None,
        organism=None,
        unused=False,
        bytes_by_organism=False,
        json=True,
    )


def _status_args(tmp_path: Path) -> argparse.Namespace:
    # The store this project names is not there: status warns and still reports.
    for folder in ("fastq", "metadata", "genomes"):
        (tmp_path / folder).mkdir()
    return argparse.Namespace(
        fastq_folder=str(tmp_path / "fastq"),
        metadata_folder=str(tmp_path / "metadata"),
        genomes_folder=str(tmp_path / "genomes"),
        targeted_folder=str(tmp_path / "targeted"),
        matches_folder=str(tmp_path / "matches"),
        registry=str(tmp_path / "metaquest_registry.json"),
        data_root=str(tmp_path / "unmounted"),
        accessions_file=None,
        parsed_containment=None,
        stage=None,
        genome=None,
        init=False,
        reconcile=False,
        export_tsv=None,
        next=False,
        list_missing=False,
        json=True,
    )


def _warn_then(original):
    """Wrap a store_usage row lookup so it logs a warning first; an unknown accession alone
    produces no warning in store_usage."""

    def wrapper(catalog, accession):
        logging.getLogger("metaquest.cli.commands.store").warning("no usage recorded for %s", accession)
        return original(catalog, accession)

    return wrapper


JSON_COMMANDS = [
    pytest.param(StoreStatusCommand, _store_status_args, id="store_status"),
    pytest.param(StoreGcCommand, _store_gc_args, id="store_gc"),
    pytest.param(StoreUsageCommand, _store_usage_args, id="store_usage"),
    pytest.param(StatusCommand, _status_args, id="status"),
]


@pytest.mark.parametrize("command,argsbuilder", JSON_COMMANDS)
def test_json_mode_stdout_is_a_single_document_despite_warnings(
    command, argsbuilder, tmp_path, monkeypatch, capsys, caplog
):
    monkeypatch.chdir(tmp_path)
    args = argsbuilder(tmp_path)
    capsys.readouterr()
    caplog.clear()

    original = StoreUsageCommand._rows_for_accession
    with (
        caplog.at_level(logging.WARNING),
        patch.object(StoreUsageCommand, "_rows_for_accession", staticmethod(_warn_then(original))),
    ):
        rc = command().execute(args)

    out = capsys.readouterr().out
    assert rc == 0
    json.loads(out)  # the whole stream parses as one document
    assert any(record.levelno >= logging.WARNING for record in caplog.records), caplog.text


def test_no_library_module_prints():
    hits = subprocess.run(
        [
            "grep",
            "-rlnE",
            r"(^|[^.a-zA-Z_])print\(",
            "--include=*.py",
            "metaquest/data",
            "metaquest/store",
            "metaquest/processing",
            "metaquest/plugins",
            "metaquest/sra",
            "metaquest/visualization",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert hits.stdout.strip() == ""


def test_print_gate_script_passes_on_the_tree():
    result = subprocess.run(["bash", "scripts/check_no_print.sh"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_print_gate_script_fails_on_a_print_outside_base(tmp_path):
    (tmp_path / "metaquest" / "cli").mkdir(parents=True)
    (tmp_path / "metaquest" / "cli" / "base.py").write_text('print("allowed")\n')
    (tmp_path / "metaquest" / "library.py").write_text('def f():\n    print("not allowed")\n')
    (tmp_path / "metaquest" / "other.py").write_text("import pprint\npprint.pprint(1)\nself._print_table(1)\n")

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "check_no_print.sh"), str(tmp_path)], capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "metaquest/library.py:2:" in result.stdout
    assert "metaquest/cli/base.py:1:" not in result.stdout
    assert "metaquest/other.py:" not in result.stdout
