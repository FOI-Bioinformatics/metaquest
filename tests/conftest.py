"""
Shared test fixtures and configuration for MetaQuest tests.
"""

import matplotlib.pyplot as plt
import pytest

from metaquest.core.constants import STORE_ENV


@pytest.fixture(autouse=True)
def isolate_store_discovery(tmp_path, monkeypatch):
    """Keep every test away from the maintainer's own shared data store.

    A store root is discovered from ``METAQUEST_DATA``, a project registry, or the user
    config file at ``$XDG_CONFIG_HOME/metaquest/config.toml`` (``~/.config`` without it).
    The documented migration tells a user to run ``store_init --set-default``, after which an
    unprotected test suite would open that real store's catalogue, write to it, and read
    whatever it holds. Pointing HOME and XDG_CONFIG_HOME at this test's own tmp_path and
    dropping METAQUEST_DATA makes the discovery rules find nothing but what a test sets up.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv(STORE_ENV, raising=False)
    yield


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")
