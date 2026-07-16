"""
Tests for metaquest.data.metadata module.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

from metaquest.core.exceptions import DataAccessError, ValidationError
from metaquest.data.metadata import (
    _get_unique_accessions,
    _download_single_metadata,
    download_metadata,
    _download_accessions_metadata,
    _extract_metadata_fields,
    _extract_sample_attributes,
    parse_metadata,
    get_unique_sample_attributes,
    check_metadata_attributes,
)


class TestGetUniqueAccessions:
    """Test _get_unique_accessions function."""

    def test_get_unique_accessions_branchwater_format(self, tmp_path):
        """Test extracting unique accessions from branchwater format."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()

        # Create test CSV with branchwater format
        csv_content = "acc,containment,organism\nSRR123,0.85,E. coli\nSRR456,0.92,S. aureus\nSRR123,0.75,E. coli"
        (matches_dir / "genome1.csv").write_text(csv_content)

        result = _get_unique_accessions(matches_dir, threshold=0.8)
        assert result == {"SRR123", "SRR456"}

    def test_get_unique_accessions_sra_format(self, tmp_path):
        """Test extracting unique accessions from SRA format."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()

        # Create test CSV with SRA format
        csv_content = "SRA accession,containment,species\nSRR789,0.88,B. subtilis\nSRR101,0.95,P. aeruginosa"
        (matches_dir / "genome2.csv").write_text(csv_content)

        result = _get_unique_accessions(matches_dir, threshold=0.9)
        assert result == {"SRR101"}

    def test_get_unique_accessions_threshold_filtering(self, tmp_path):
        """Test threshold filtering works correctly."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()

        csv_content = "acc,containment,organism\nSRR123,0.85,E. coli\nSRR456,0.75,S. aureus\nSRR789,0.95,B. subtilis"
        (matches_dir / "genome1.csv").write_text(csv_content)

        result = _get_unique_accessions(matches_dir, threshold=0.8)
        assert result == {"SRR123", "SRR789"}

    def test_get_unique_accessions_no_csv_files(self, tmp_path):
        """Test handling when no CSV files are found."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()

        with patch("metaquest.data.metadata.logger") as mock_logger:
            result = _get_unique_accessions(matches_dir, threshold=0.0)

        assert result == set()
        mock_logger.warning.assert_called_once()

    def test_get_unique_accessions_unknown_format(self, tmp_path):
        """Test handling unknown file format."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()

        # Create CSV with unknown format
        csv_content = "unknown_col,other_col\nvalue1,value2"
        (matches_dir / "unknown.csv").write_text(csv_content)

        with patch("metaquest.data.metadata.logger") as mock_logger:
            result = _get_unique_accessions(matches_dir, threshold=0.0)

        assert result == set()
        mock_logger.warning.assert_called()

    def test_get_unique_accessions_read_error(self, tmp_path):
        """Test handling CSV read errors."""
        matches_dir = tmp_path / "matches"
        matches_dir.mkdir()

        # Create malformed CSV
        (matches_dir / "malformed.csv").write_text("invalid,csv\ncontent")

        with patch("pandas.read_csv", side_effect=Exception("Read error")):
            with patch("metaquest.data.metadata.logger") as mock_logger:
                result = _get_unique_accessions(matches_dir, threshold=0.0)

        assert result == set()
        mock_logger.warning.assert_called()


class TestDownloadSingleMetadata:
    """Test _download_single_metadata function."""

    def test_download_single_metadata_success(self, tmp_path):
        """Test successful metadata download."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        mock_response = b"<?xml version='1.0'?><root>test metadata</root>"  # Return bytes

        with patch("metaquest.data.metadata.Entrez.efetch") as mock_efetch:
            mock_handle = MagicMock()
            mock_handle.read.return_value = mock_response
            mock_efetch.return_value = mock_handle

            with patch("time.sleep"):  # Speed up test
                success, result = _download_single_metadata("SRR123", metadata_path, "test@example.com")

        assert success is True
        assert isinstance(result, Path)
        assert result.name == "SRR123_metadata.xml"
        assert result.read_text() == mock_response.decode()

    def test_download_single_metadata_http_error(self, tmp_path):
        """Test handling HTTP errors with retry."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        with patch("metaquest.data.metadata.Entrez.efetch", side_effect=Exception("HTTP 500")):
            with patch("time.sleep"):  # Speed up test
                success, result = _download_single_metadata("SRR123", metadata_path, "test@example.com")

        assert success is False
        assert "Failed after 3 retries" in result

    def test_download_single_metadata_partial_retry(self, tmp_path):
        """Test successful download after one retry."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        mock_response = b"<?xml version='1.0'?><root>test metadata</root>"  # Return bytes

        with patch("metaquest.data.metadata.Entrez.efetch") as mock_efetch:
            # First call fails, second succeeds
            mock_handle = MagicMock()
            mock_handle.read.return_value = mock_response
            mock_efetch.side_effect = [Exception("Network error"), mock_handle]

            with patch("time.sleep"):  # Speed up test
                success, result = _download_single_metadata("SRR123", metadata_path, "test@example.com")

        assert success is True
        assert isinstance(result, Path)


class TestDownloadMetadata:
    """Test download_metadata function."""

    def test_download_metadata_success(self, tmp_path):
        """Test successful metadata download workflow."""
        matches_dir = tmp_path / "matches"
        metadata_dir = tmp_path / "metadata"
        matches_dir.mkdir()

        # Create test match file
        csv_content = "acc,containment,organism\nSRR123,0.95,E. coli"
        (matches_dir / "genome1.csv").write_text(csv_content)

        with patch("metaquest.data.metadata._download_accessions_metadata") as mock_download:
            mock_download.return_value = {"SRR123": metadata_dir / "SRR123_metadata.xml"}

            result = download_metadata("test@example.com", matches_dir, metadata_dir, threshold=0.9)

        assert isinstance(result, dict)
        mock_download.assert_called_once()

    def test_download_metadata_dry_run(self, tmp_path):
        """Test dry run mode."""
        matches_dir = tmp_path / "matches"
        metadata_dir = tmp_path / "metadata"
        matches_dir.mkdir()

        csv_content = "acc,containment,organism\nSRR123,0.95,E. coli"
        (matches_dir / "genome1.csv").write_text(csv_content)

        with patch("metaquest.data.metadata.logger") as mock_logger:
            result = download_metadata("test@example.com", matches_dir, metadata_dir, dry_run=True)

        assert result == {}
        mock_logger.info.assert_called_with("Dry run, not downloading metadata")

    def test_download_metadata_skip_existing(self, tmp_path):
        """Test skipping existing metadata files."""
        matches_dir = tmp_path / "matches"
        metadata_dir = tmp_path / "metadata"
        matches_dir.mkdir()
        metadata_dir.mkdir()

        # Create existing metadata file
        (metadata_dir / "SRR123_metadata.xml").write_text("existing")

        csv_content = "acc,containment,organism\nSRR123,0.95,E. coli\nSRR456,0.85,S. aureus"
        (matches_dir / "genome1.csv").write_text(csv_content)

        with patch("metaquest.data.metadata._download_accessions_metadata") as mock_download:
            mock_download.return_value = {"SRR456": metadata_dir / "SRR456_metadata.xml"}

            download_metadata("test@example.com", matches_dir, metadata_dir)

        # Should only download SRR456, not SRR123
        download_args = mock_download.call_args[0]
        accessions_to_download = download_args[0]
        assert "SRR456" in accessions_to_download
        assert "SRR123" not in accessions_to_download

    def test_download_metadata_validation_error(self, tmp_path):
        """Test handling validation errors."""
        with patch("metaquest.data.metadata.validate_folder", side_effect=ValidationError("Invalid folder")):
            with pytest.raises(DataAccessError):
                download_metadata("test@example.com", "invalid", tmp_path)


class TestDownloadAccessionsMetadata:
    """Test _download_accessions_metadata function."""

    def test_download_accessions_metadata_success(self, tmp_path):
        """Test successful accessions download."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        accessions = ["SRR123", "SRR456"]

        with patch("metaquest.data.metadata._download_single_metadata") as mock_download:
            mock_download.side_effect = [
                (True, metadata_path / "SRR123_metadata.xml"),
                (True, metadata_path / "SRR456_metadata.xml"),
            ]

            result = _download_accessions_metadata(accessions, metadata_path, "test@example.com", 2)

        assert len(result) == 2
        assert "SRR123" in result
        assert "SRR456" in result

    def test_download_accessions_metadata_partial_failure(self, tmp_path):
        """Test handling partial download failures."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        accessions = ["SRR123", "SRR456"]

        with patch("metaquest.data.metadata._download_single_metadata") as mock_download:
            mock_download.side_effect = [(True, metadata_path / "SRR123_metadata.xml"), (False, "Download failed")]

            with patch("metaquest.data.metadata.logger") as mock_logger:
                result = _download_accessions_metadata(accessions, metadata_path, "test@example.com", 2)

        assert len(result) == 1
        assert "SRR123" in result
        assert "SRR456" not in result
        mock_logger.error.assert_called()


class TestExtractMetadataFields:
    """Test _extract_metadata_fields function."""

    def test_extract_metadata_fields_success(self):
        """Test successful metadata field extraction."""
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <EXPERIMENT>
                    <IDENTIFIERS>
                        <PRIMARY_ID>EXP123</PRIMARY_ID>
                    </IDENTIFIERS>
                    <TITLE>Test experiment</TITLE>
                    <LIBRARY_DESCRIPTOR>
                        <LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY>
                        <LIBRARY_SOURCE>GENOMIC</LIBRARY_SOURCE>
                    </LIBRARY_DESCRIPTOR>
                </EXPERIMENT>
                <SAMPLE>
                    <IDENTIFIERS>
                        <PRIMARY_ID>SAMN123</PRIMARY_ID>
                    </IDENTIFIERS>
                    <SAMPLE_NAME>
                        <SCIENTIFIC_NAME>Escherichia coli</SCIENTIFIC_NAME>
                    </SAMPLE_NAME>
                </SAMPLE>
                <RUN_SET>
                    <RUN>
                        <IDENTIFIERS>
                            <PRIMARY_ID>SRR123</PRIMARY_ID>
                        </IDENTIFIERS>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""

        tree = ET.fromstring(xml_content)

        result = _extract_metadata_fields(tree, "test.xml")

        assert result["Run_ID"] == "SRR123"
        assert result["Sample_ID"] == "SAMN123"
        assert result["Experiment_Title"] == "Test experiment"
        assert result["Experiment_ID"] == "EXP123"
        assert result["Sample_Scientific_Name"] == "Escherichia coli"
        assert result["Experiment_Library_Strategy"] == "WGS"

    def test_extract_metadata_fields_missing_elements(self):
        """Test handling missing XML elements."""
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""

        tree = ET.fromstring(xml_content)

        result = _extract_metadata_fields(tree, "test.xml")

        # Should have None values for missing elements
        assert result["Run_ID"] is None
        assert result["Sample_ID"] is None
        assert result["Experiment_Title"] is None

    def test_extract_metadata_fields_complex_structure(self):
        """Test extraction from complex XML structure."""
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <EXPERIMENT>
                    <IDENTIFIERS>
                        <PRIMARY_ID>EXP456</PRIMARY_ID>
                    </IDENTIFIERS>
                    <TITLE>Complex experiment</TITLE>
                    <LIBRARY_DESCRIPTOR>
                        <LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY>
                        <LIBRARY_SOURCE>GENOMIC</LIBRARY_SOURCE>
                        <LIBRARY_SELECTION>RANDOM</LIBRARY_SELECTION>
                    </LIBRARY_DESCRIPTOR>
                </EXPERIMENT>
                <SAMPLE>
                    <IDENTIFIERS>
                        <PRIMARY_ID>SAMN456</PRIMARY_ID>
                    </IDENTIFIERS>
                    <TITLE>Test sample</TITLE>
                </SAMPLE>
                <RUN_SET>
                    <RUN>
                        <IDENTIFIERS>
                            <PRIMARY_ID>SRR456</PRIMARY_ID>
                        </IDENTIFIERS>
                        <Total_spots>1000000</Total_spots>
                        <Total_bases>150000000</Total_bases>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""

        tree = ET.fromstring(xml_content)

        result = _extract_metadata_fields(tree, "test.xml")

        assert result["Run_ID"] == "SRR456"
        assert result["Sample_ID"] == "SAMN456"
        assert result["Experiment_Library_Strategy"] == "WGS"
        assert result["Experiment_Library_Source"] == "GENOMIC"
        assert result["Experiment_Library_Selection"] == "RANDOM"
        assert result["Sample_Title"] == "Test sample"
        assert result["Run_Total_Spots"] == "1000000"
        assert result["Run_Total_Bases"] == "150000000"


class TestExtractSampleAttributes:
    """Test _extract_sample_attributes function."""

    def test_extract_sample_attributes_success(self):
        """Test successful sample attributes extraction."""
        xml_content = """<?xml version="1.0"?>
        <root>
            <SAMPLE_ATTRIBUTES>
                <SAMPLE_ATTRIBUTE>
                    <TAG>organism</TAG>
                    <VALUE>Escherichia coli</VALUE>
                </SAMPLE_ATTRIBUTE>
                <SAMPLE_ATTRIBUTE>
                    <TAG>isolation_source</TAG>
                    <VALUE>clinical isolate</VALUE>
                </SAMPLE_ATTRIBUTE>
            </SAMPLE_ATTRIBUTES>
        </root>"""

        tree = ET.fromstring(xml_content)
        # Pre-populate unique_attributes since the function only extracts existing attributes
        unique_attributes = {"organism", "isolation_source"}

        result = _extract_sample_attributes(tree, unique_attributes)

        assert result["organism"] == "Escherichia coli"
        assert result["isolation_source"] == "clinical isolate"

    def test_extract_sample_attributes_no_attributes(self):
        """Test when no sample attributes are present."""
        xml_content = """<?xml version="1.0"?><root></root>"""

        tree = ET.fromstring(xml_content)
        unique_attributes = set()

        result = _extract_sample_attributes(tree, unique_attributes)

        assert result == {}
        assert len(unique_attributes) == 0

    def test_extract_sample_attributes_empty_values(self):
        """Test handling empty attribute values."""
        xml_content = """<?xml version="1.0"?>
        <root>
            <SAMPLE_ATTRIBUTES>
                <SAMPLE_ATTRIBUTE>
                    <TAG>organism</TAG>
                    <VALUE></VALUE>
                </SAMPLE_ATTRIBUTE>
                <SAMPLE_ATTRIBUTE>
                    <TAG>missing_value</TAG>
                </SAMPLE_ATTRIBUTE>
            </SAMPLE_ATTRIBUTES>
        </root>"""

        tree = ET.fromstring(xml_content)
        unique_attributes = set()

        result = _extract_sample_attributes(tree, unique_attributes)

        # Empty values should not be included
        assert "organism" not in result
        assert "missing_value" not in result


class TestParseMetadata:
    """Test parse_metadata function."""

    def test_parse_metadata_success(self, tmp_path):
        """Test successful metadata parsing."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        output_file = tmp_path / "parsed_metadata.tsv"

        # Create test XML files
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <SAMPLE>
                    <IDENTIFIERS>
                        <PRIMARY_ID>SAMN123</PRIMARY_ID>
                    </IDENTIFIERS>
                </SAMPLE>
                <RUN_SET>
                    <RUN>
                        <IDENTIFIERS>
                            <PRIMARY_ID>SRR123</PRIMARY_ID>
                        </IDENTIFIERS>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""

        (metadata_dir / "SRR123_metadata.xml").write_text(xml_content)

        result = parse_metadata(metadata_dir, output_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["Run_ID"] == "SRR123"
        assert result.iloc[0]["Sample_ID"] == "SAMN123"
        assert output_file.exists()

    def test_parse_metadata_no_xml_files(self, tmp_path):
        """Test when no XML files are found."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        output_file = tmp_path / "parsed_metadata.tsv"

        with patch("metaquest.data.metadata.logger") as mock_logger:
            result = parse_metadata(metadata_dir, output_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        mock_logger.warning.assert_called()

    def test_parse_metadata_invalid_xml(self, tmp_path):
        """Test handling invalid XML files."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        output_file = tmp_path / "parsed_metadata.tsv"

        # Create invalid XML
        (metadata_dir / "SRR123_metadata.xml").write_text("invalid xml content")

        with patch("metaquest.data.metadata.logger") as mock_logger:
            result = parse_metadata(metadata_dir, output_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        mock_logger.error.assert_called()

    def test_parse_metadata_nonexistent_folder(self):
        """Test handling nonexistent metadata folder."""
        with pytest.raises(ValidationError):
            parse_metadata("/nonexistent/folder", "/tmp/output.tsv")


class TestGetUniqueSampleAttributes:
    """Test get_unique_sample_attributes function."""

    def test_get_unique_sample_attributes_success(self, tmp_path):
        """Test successful unique attributes extraction."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        xml_content = """<?xml version="1.0"?>
        <root>
            <SAMPLE_ATTRIBUTES>
                <SAMPLE_ATTRIBUTE>
                    <TAG>organism</TAG>
                    <VALUE>E. coli</VALUE>
                </SAMPLE_ATTRIBUTE>
                <SAMPLE_ATTRIBUTE>
                    <TAG>isolation_source</TAG>
                    <VALUE>clinical</VALUE>
                </SAMPLE_ATTRIBUTE>
            </SAMPLE_ATTRIBUTES>
        </root>"""

        (metadata_dir / "SRR123_metadata.xml").write_text(xml_content)

        result = get_unique_sample_attributes(metadata_dir)

        assert isinstance(result, list)
        assert "organism" in result
        assert "isolation_source" in result

    def test_get_unique_sample_attributes_no_files(self, tmp_path):
        """Test when no XML files are found."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        result = get_unique_sample_attributes(metadata_dir)

        assert result == []


class TestCheckMetadataAttributes:
    """Test check_metadata_attributes function."""

    def test_check_metadata_attributes_success(self, tmp_path):
        """Test successful metadata attributes checking."""
        input_file = tmp_path / "metadata.tsv"
        output_file = tmp_path / "attributes.txt"

        # Create test metadata file
        df = pd.DataFrame(
            {
                "Run_ID": ["SRR123", "SRR456"],
                "organism": ["E. coli", "S. aureus"],
                "isolation_source": ["clinical", "environmental"],
            }
        )
        df.to_csv(input_file, sep="\t", index=False)

        result = check_metadata_attributes(input_file, output_file)

        assert isinstance(result, dict)
        assert "organism" in result
        assert "isolation_source" in result
        assert result["organism"] == 2
        assert result["isolation_source"] == 2
        assert output_file.exists()

    def test_check_metadata_attributes_empty_file(self, tmp_path):
        """Test handling empty metadata file."""
        input_file = tmp_path / "empty.tsv"
        output_file = tmp_path / "attributes.txt"

        # Create file with just headers
        input_file.write_text("Run_ID\tSample_ID\n")

        result = check_metadata_attributes(input_file, output_file)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_check_metadata_attributes_nonexistent_file(self):
        """Test handling nonexistent input file."""
        with pytest.raises(DataAccessError):
            check_metadata_attributes("/nonexistent/file.tsv", "/tmp/output.txt")


if __name__ == "__main__":
    pytest.main([__file__])
