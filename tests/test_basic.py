"""
Basic unit tests for MetaQuest.
"""

import os
import tempfile

import pytest

import metaquest
from metaquest.core.exceptions import ValidationError
from metaquest.core.validation import detect_file_format, validate_csv_file


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


def test_detect_file_format_mastiff():
    """Test detection of Mastiff format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("SRA accession,containment,similarity,query_name,query_md5,status\n")
        f.write("SRR123456,0.95,0.98,test_genome,abcdef123456,completed\n")
        f.flush()

        # Test format detection
        format_name = detect_file_format(f.name)
        assert format_name == "mastiff"

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


def test_validate_csv_file_branchwater():
    """Test validation of Branchwater CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("acc,containment,cANI,biosample,bioproject\n")
        f.write("SRR123456,0.95,0.98,SAMN123456,PRJNA123456\n")
        f.flush()

        # Test validation
        format_name, headers = validate_csv_file(f.name)
        assert format_name == "branchwater"
        assert "acc" in headers
        assert "containment" in headers

    # Clean up
    os.unlink(f.name)


def test_validate_csv_file_missing_columns():
    """Test validation of CSV file with missing required columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("acc,someOtherColumn\n")
        f.write("SRR123456,value\n")
        f.flush()

        # Test validation
        with pytest.raises(ValidationError):
            validate_csv_file(f.name)

    # Clean up
    os.unlink(f.name)


def test_validate_csv_file_empty():
    """Test validation of empty CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("acc,containment\n")
        f.flush()

        # Test validation
        with pytest.raises(ValidationError):
            validate_csv_file(f.name)

    # Clean up
    os.unlink(f.name)
