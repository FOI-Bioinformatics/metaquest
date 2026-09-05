"""
Basic unit tests for MetaQuest.
"""

import metaquest


def test_version():
    """Test that version is defined."""
    assert metaquest.__version__ is not None
