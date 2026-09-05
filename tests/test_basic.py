"""
Basic unit tests for MetaQuest.
"""

import os
import tempfile

import pytest

import metaquest
from metaquest.core.exceptions import ValidationError
from metaquest.core.validation import detect_file_format


def test_version():
    """Test that version is defined."""
    assert metaquest.__version__ is not None


def test_detect_file_format_branchwater():
    """Test detection of Branchwater format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("acc,containment,cANI,biosample,bioproject\n")
        f.write("SRR123456,0.95,0.98,SAMN123456,PRJNA123456\n")
        f.flush()

        # Test format detection
        format_name = detect_file_format(f.name)
        assert format_name == "branchwater"

    # Clean up
    os.unlink(f.name)


def test_detect_file_format_unknown():
    """Test detection of unknown format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("column1,column2,column3\n")
        f.write("value1,value2,value3\n")
        f.flush()

        # Test format detection
        with pytest.raises(ValidationError):
            detect_file_format(f.name)

    # Clean up
    os.unlink(f.name)
