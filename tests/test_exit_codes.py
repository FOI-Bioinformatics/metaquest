"""Exit codes: 0 success, 1 failure, 2 usage, 3 configuration, 4 retryable (network, lock wait), 130 interrupt."""

import argparse
import logging
from unittest.mock import patch

import pytest

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import (
    ConfigurationError,
    DataAccessError,
    ExitCode,
    FormatError,
    LockTimeoutError,
    MetaQuestError,
    NetworkError,
    ProcessingError,
    SecurityError,
    TransientError,
    ValidationError,
    exit_code_for,
)
from metaquest.utils.lockfile import LockHeld, LockLost, LockPolicy, held_lock


@pytest.mark.parametrize(
    "error, code",
    [
        (MetaQuestError("x"), 1),
        (ValidationError("x"), 1),
        (FormatError("x"), 1),
        (DataAccessError("x"), 1),
        (ProcessingError("x"), 1),
        (SecurityError("x"), 1),
        (LockHeld("x"), 1),
        (LockLost("x"), 1),
        (ConfigurationError("x"), 3),
        (TransientError("x"), 4),
        (NetworkError("x"), 4),
        (LockTimeoutError("x"), 4),
        (KeyboardInterrupt(), 130),
        (ValueError("x"), 1),
        (OSError("x"), 1),
    ],
)
def test_exit_code_for(error, code):
    assert exit_code_for(error) == code


def test_exit_code_values():
    assert [int(c) for c in ExitCode] == [0, 1, 2, 3, 4, 130]


def test_transient_errors_are_data_access_errors():
    """Every existing ``except DataAccessError`` still catches a network or lock-wait failure."""
    assert issubclass(NetworkError, DataAccessError)
    assert issubclass(LockTimeoutError, DataAccessError)


def test_lock_wait_give_up_is_a_timeout_and_still_lock_held(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("1")
    policy = LockPolicy("Test lock", stale_seconds=600.0, wait_seconds=0.1, poll_seconds=0.02)
    with pytest.raises(LockTimeoutError) as excinfo:
        with held_lock(lock, policy):
            pass
    assert isinstance(excinfo.value, LockHeld)
    assert exit_code_for(excinfo.value) == 4


def test_non_blocking_lock_held_is_not_a_timeout(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("1")
    policy = LockPolicy("Test lock", stale_seconds=600.0, wait_seconds=0.1, poll_seconds=0.02)
    with pytest.raises(LockHeld) as excinfo:
        with held_lock(lock, policy, blocking=False):
            pass
    assert not isinstance(excinfo.value, LockTimeoutError)
    assert exit_code_for(excinfo.value) == 1


# --- BaseCommand.fail ---------------------------------------------------------


class _Failing(BaseCommand):
    def __init__(self, error):
        super().__init__()
        self.error = error

    @property
    def name(self) -> str:
        return "failing"

    @property
    def help(self) -> str:
        return "fails"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        try:
            raise self.error
        except MetaQuestError as e:
            return self.fail(e, "Doing the thing")


@pytest.mark.parametrize(
    "error, code", [(ValidationError("bad"), 1), (ConfigurationError("bad"), 3), (NetworkError("bad"), 4)]
)
def test_fail_logs_one_line_and_returns_the_code(error, code, caplog):
    with caplog.at_level(logging.ERROR):
        assert _Failing(error).execute(argparse.Namespace()) == code
    record = caplog.records[-1]
    assert record.getMessage() == "Doing the thing: bad"
    assert not record.exc_info


def test_fail_attaches_the_traceback_at_debug(caplog):
    error = NetworkError("bad")
    with caplog.at_level(logging.DEBUG):
        assert _Failing(error).execute(argparse.Namespace()) == 4
    record = caplog.records[-1]
    assert record.exc_info is not None and record.exc_info[1] is error


# --- main() -------------------------------------------------------------------


def _main(argv):
    from metaquest.cli.main import main

    with patch("metaquest.cli.main.setup_logging"):
        return main(argv)


@pytest.mark.parametrize(
    "error, code",
    [
        (ConfigurationError("missing"), 3),
        (LockTimeoutError("waited"), 4),
        (NetworkError("down"), 4),
        (MetaQuestError("plain"), 1),
        (RuntimeError("bug"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_main_maps_an_escaping_error_to_its_exit_code(error, code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("metaquest.cli.commands.blacklist.BlacklistCommand.execute", side_effect=error):
        assert _main(["blacklist", "--list"]) == code


def test_main_keyboard_interrupt_without_graceful_shutdown(tmp_path, monkeypatch):
    from metaquest.cli.commands.blacklist import BlacklistCommand

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BlacklistCommand, "graceful_shutdown", False)
    with patch.object(BlacklistCommand, "execute", side_effect=KeyboardInterrupt()):
        assert _main(["blacklist", "--list"]) == 130


def test_main_malformed_config_is_a_configuration_error(tmp_path, monkeypatch):
    monkeypatch.setenv("METAQUEST_PROGRESS_EVERY", "not-a-number")
    assert _main(["blacklist", "--list"]) == 3


# --- NCBI requests --------------------------------------------------------------


def _http_error(status):
    import requests

    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


def _request_errors():
    import requests

    return [
        (requests.ConnectionError("no route"), NetworkError),
        (requests.Timeout("read timed out"), NetworkError),
        (_http_error(429), NetworkError),
        (_http_error(503), NetworkError),
        (_http_error(400), DataAccessError),
        (_http_error(404), DataAccessError),
    ]


@pytest.mark.parametrize("error, expected", _request_errors())
def test_ncbi_request_failures_are_network_errors_only_when_retryable(error, expected):
    from metaquest.data.sra_metadata import SRAMetadataClient

    client = SRAMetadataClient("someone@example.org")
    with patch("metaquest.data.sra_metadata.requests.get", side_effect=error):
        with pytest.raises(DataAccessError) as excinfo:
            client._make_request("https://example.org/efetch", {})
    assert type(excinfo.value) is expected
