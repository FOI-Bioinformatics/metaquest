"""
Shared test fixtures and configuration for MetaQuest tests.
"""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")
