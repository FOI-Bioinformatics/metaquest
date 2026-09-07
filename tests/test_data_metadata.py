"""
Tests for metaquest.data.metadata module.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET

from metaquest.core.exceptions import DataAccessError, ValidationError
from metaquest.data.metadata import (
    _get_unique_accessions,
    _download_single_metadata,
    download_metadata,
    _download_accessions_metadata,
    _download_batch_metadata,
    _pace_requests,
    _split_efetch_packages,
    _extract_metadata_fields,
    _extract_sample_attributes,
    parse_metadata,
    parse_metadata_xml,
    get_unique_sample_attributes,
    check_metadata_attributes,
)


def _http_error(code, reason="Error"):
    """Build a urllib HTTPError with the given status code, as Entrez.efetch would raise."""
    return HTTPError("https://eutils.ncbi.nlm.nih.gov/", code, reason, hdrs=None, fp=None)


def _package_xml(accessions_and_attrs):
    """Build a minimal efetch response with one EXPERIMENT_PACKAGE per accession.

    ``accessions_and_attrs`` is a list of (accession, {attr: value}) pairs.
    """
    root = ET.Element("EXPERIMENT_PACKAGE_SET")
    for accession, attrs in accessions_and_attrs:
        package = ET.SubElement(root, "EXPERIMENT_PACKAGE")
        run_set = ET.SubElement(package, "RUN_SET")
        run = ET.SubElement(run_set, "RUN", accession=accession, **attrs)
        identifiers = ET.SubElement(run, "IDENTIFIERS")
        ET.SubElement(identifiers, "PRIMARY_ID").text = accession
    return ET.tostring(root, encoding="unicode")


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

            with patch("metaquest.data.metadata._pace_requests"):
                success, result = _download_single_metadata("SRR123", metadata_path, "test@example.com")

        assert success is True
        assert isinstance(result, Path)
        assert result.name == "SRR123_metadata.xml"
        assert result.read_text() == mock_response.decode()

    def test_download_single_metadata_404_is_final(self, tmp_path):
        """A 404 on a single accession is a final failure, never retried."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        with patch("metaquest.data.metadata.Entrez.efetch", side_effect=_http_error(404)) as mock_efetch:
            with patch("metaquest.data.metadata._pace_requests"):
                with patch("time.sleep") as mock_sleep:
                    success, result = _download_single_metadata("SRR123", metadata_path, "test@example.com")

        assert success is False
        assert result == "HTTP 404: not found at NCBI"
        mock_efetch.assert_called_once()
        mock_sleep.assert_not_called()

    def test_download_single_metadata_503_retried_then_fails(self, tmp_path):
        """A 503 is retried up to 3 times with exponential backoff, then fails."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        with patch("metaquest.data.metadata.Entrez.efetch", side_effect=_http_error(503)):
            with patch("metaquest.data.metadata._pace_requests"):
                with patch("time.sleep") as mock_sleep:
                    success, result = _download_single_metadata("SRR123", metadata_path, "test@example.com")

        assert success is False
        assert result == "HTTP 503 after 3 attempts"
        assert mock_sleep.call_args_list == [call(2), call(4), call(8)]

    def test_download_single_metadata_partial_retry(self, tmp_path):
        """Test successful download after one retry."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        mock_response = b"<?xml version='1.0'?><root>test metadata</root>"  # Return bytes

        with patch("metaquest.data.metadata.Entrez.efetch") as mock_efetch:
            # First call fails, second succeeds
            mock_handle = MagicMock()
            mock_handle.read.return_value = mock_response
            mock_efetch.side_effect = [URLError("Network error"), mock_handle]

            with patch("metaquest.data.metadata._pace_requests"):
                with patch("time.sleep"):
                    success, result = _download_single_metadata("SRR123", metadata_path, "test@example.com")

        assert success is True
        assert isinstance(result, Path)

    def test_download_single_metadata_api_key_set_on_entrez(self, tmp_path):
        """Entrez.api_key equals the given key during the call; None leaves it None."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()
        mock_response = b"<?xml version='1.0'?><root>test metadata</root>"

        seen_keys = []

        def fake_efetch(**kwargs):
            from metaquest.data.metadata import Entrez

            seen_keys.append(Entrez.api_key)
            handle = MagicMock()
            handle.read.return_value = mock_response
            return handle

        with patch("metaquest.data.metadata.Entrez.efetch", side_effect=fake_efetch):
            with patch("metaquest.data.metadata._pace_requests"):
                _download_single_metadata("SRR123", metadata_path, "test@example.com", api_key="my-key")
                _download_single_metadata("SRR124", metadata_path, "test@example.com", api_key=None)

        assert seen_keys == ["my-key", None]


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

    def test_download_metadata_accessions_file_bypasses_matches_folder(self, tmp_path):
        """With --accessions-file the wanted set comes from the file; the matches folder is not read."""
        metadata_dir = tmp_path / "metadata"
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR1\nSRR2\n# comment\n\nSRR3\n")

        calls = []

        def fake_batch(batch, metadata_path, email, api_key):
            calls.extend(batch)
            successes = {}
            for accession in batch:
                xml_path = metadata_path / f"{accession}_metadata.xml"
                xml_path.write_text("<root/>")
                successes[accession] = xml_path
            return successes, {}

        with patch("metaquest.data.metadata.validate_folder") as mock_validate_folder:
            with patch("metaquest.data.metadata._download_batch_metadata", side_effect=fake_batch):
                result = download_metadata(
                    "test@example.com",
                    "does-not-exist",
                    metadata_dir,
                    accessions_file=accessions_file,
                )

        mock_validate_folder.assert_not_called()
        assert sorted(calls) == ["SRR1", "SRR2", "SRR3"]
        assert set(result) == {"SRR1", "SRR2", "SRR3"}

    def test_download_metadata_batch_size_out_of_range_raises(self, tmp_path):
        """batch_size outside 1-500 raises ValueError, not DataAccessError."""
        with pytest.raises(ValueError):
            download_metadata("test@example.com", tmp_path, tmp_path, batch_size=0)
        with pytest.raises(ValueError):
            download_metadata("test@example.com", tmp_path, tmp_path, batch_size=501)

    def test_download_metadata_passes_api_key_and_batch_size(self, tmp_path):
        """api_key and batch_size reach _download_accessions_metadata."""
        matches_dir = tmp_path / "matches"
        metadata_dir = tmp_path / "metadata"
        matches_dir.mkdir()
        (matches_dir / "genome1.csv").write_text("acc,containment,organism\nSRR123,0.95,E. coli")

        with patch("metaquest.data.metadata._download_accessions_metadata") as mock_download:
            mock_download.return_value = {}
            download_metadata("test@example.com", matches_dir, metadata_dir, api_key="my-key", batch_size=5)

        assert mock_download.call_args.kwargs["api_key"] == "my-key"
        assert mock_download.call_args.kwargs["batch_size"] == 5


class TestDownloadAccessionsMetadata:
    """Test _download_accessions_metadata function."""

    def test_download_accessions_metadata_success(self, tmp_path):
        """Test successful accessions download."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        accessions = ["SRR123", "SRR456"]

        with patch("metaquest.data.metadata._download_batch_metadata") as mock_download:
            mock_download.return_value = (
                {
                    "SRR123": metadata_path / "SRR123_metadata.xml",
                    "SRR456": metadata_path / "SRR456_metadata.xml",
                },
                {},
            )

            result = _download_accessions_metadata(accessions, metadata_path, "test@example.com", 2)

        assert len(result) == 2
        assert "SRR123" in result
        assert "SRR456" in result
        mock_download.assert_called_once_with(accessions, metadata_path, "test@example.com", None)

    def test_download_accessions_metadata_partial_failure(self, tmp_path):
        """Test handling partial download failures."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        accessions = ["SRR123", "SRR456"]

        with patch("metaquest.data.metadata._download_batch_metadata") as mock_download:
            mock_download.return_value = ({"SRR123": metadata_path / "SRR123_metadata.xml"}, {"SRR456": "not found"})

            with patch("metaquest.data.metadata.logger") as mock_logger:
                result = _download_accessions_metadata(accessions, metadata_path, "test@example.com", 2)

        assert len(result) == 1
        assert "SRR123" in result
        assert "SRR456" not in result
        mock_logger.error.assert_called()

    def test_download_accessions_metadata_sets_entrez_email_and_api_key(self, tmp_path):
        """Entrez.email and Entrez.api_key are set once before any batch is fetched."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        with patch("metaquest.data.metadata._download_batch_metadata", return_value=({}, {})):
            from metaquest.data.metadata import Entrez

            _download_accessions_metadata(["SRR1"], metadata_path, "test@example.com", 1, api_key="my-key")
            assert Entrez.email == "test@example.com"
            assert Entrez.api_key == "my-key"

            _download_accessions_metadata(["SRR1"], metadata_path, "test@example.com", 1, api_key=None)
            assert Entrez.api_key is None

    def test_download_accessions_metadata_batches_by_batch_size(self, tmp_path):
        """5 accessions with batch_size=2 make 3 efetch calls (batches of 2, 2, 1)."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()
        accessions = [f"SRR{i}" for i in range(5)]

        with patch("metaquest.data.metadata._download_batch_metadata", return_value=({}, {})) as mock_download:
            _download_accessions_metadata(accessions, metadata_path, "test@example.com", 5, batch_size=2)

        assert mock_download.call_count == 3
        called_batches = [c.args[0] for c in mock_download.call_args_list]
        assert called_batches == [accessions[0:2], accessions[2:4], accessions[4:5]]


class TestDownloadBatchMetadata:
    """Test _download_batch_metadata: one efetch call per batch, split into per-accession files."""

    def test_batch_of_three_with_shared_experiment_package(self, tmp_path):
        """Three accessions in one response, two sharing a package with two RUNs: three files."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        root = ET.Element("EXPERIMENT_PACKAGE_SET")
        shared_package = ET.SubElement(root, "EXPERIMENT_PACKAGE")
        run_set = ET.SubElement(shared_package, "RUN_SET")
        for accession, spots, size in (("SRR1", "100", "1000"), ("SRR2", "150", "1500")):
            run = ET.SubElement(run_set, "RUN", accession=accession, total_spots=spots, size=size)
            ET.SubElement(ET.SubElement(run, "IDENTIFIERS"), "PRIMARY_ID").text = accession

        solo_package = ET.SubElement(root, "EXPERIMENT_PACKAGE")
        solo_run_set = ET.SubElement(solo_package, "RUN_SET")
        solo_run = ET.SubElement(solo_run_set, "RUN", accession="SRR3", total_spots="300", size="3000")
        ET.SubElement(ET.SubElement(solo_run, "IDENTIFIERS"), "PRIMARY_ID").text = "SRR3"

        xml_text = ET.tostring(root, encoding="unicode")
        mock_handle = MagicMock()
        mock_handle.read.return_value = xml_text.encode()

        with patch("metaquest.data.metadata.Entrez.efetch", return_value=mock_handle) as mock_efetch:
            with patch("metaquest.data.metadata._pace_requests"):
                successes, failures = _download_batch_metadata(
                    ["SRR1", "SRR2", "SRR3"], metadata_path, "test@example.com", None
                )

        assert mock_efetch.call_count == 1
        assert failures == {}
        assert set(successes) == {"SRR1", "SRR2", "SRR3"}
        for accession, spots, size in (("SRR1", "100", "1000"), ("SRR2", "150", "1500"), ("SRR3", "300", "3000")):
            parsed = parse_metadata_xml(successes[accession])
            assert parsed["Run_ID"] == accession
            assert parsed["Run_Total_Spots"] == spots
            assert parsed["Run_Size"] == size

    def test_accession_missing_from_response_is_reported_failed(self, tmp_path):
        """An accession NCBI doesn't return is a failure, and efetch was called only once."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        xml_text = _package_xml([("SRR1", {"total_spots": "10"}), ("SRR2", {"total_spots": "20"})])
        mock_handle = MagicMock()
        mock_handle.read.return_value = xml_text.encode()

        with patch("metaquest.data.metadata.Entrez.efetch", return_value=mock_handle) as mock_efetch:
            with patch("metaquest.data.metadata._pace_requests"):
                successes, failures = _download_batch_metadata(
                    ["SRR1", "SRR2", "SRR4"], metadata_path, "test@example.com", None
                )

        assert mock_efetch.call_count == 1
        assert set(successes) == {"SRR1", "SRR2"}
        assert failures == {"SRR4": "not in the NCBI response"}

    def test_404_on_batch_falls_back_to_single_calls(self, tmp_path):
        """A 404 on a batch of more than one accession falls back to per-accession fetches."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()
        batch = ["SRR1", "SRR2", "SRR3"]

        def fake_efetch(**kwargs):
            if "," in kwargs["id"]:
                raise _http_error(404)
            accession = kwargs["id"]
            handle = MagicMock()
            handle.read.return_value = _package_xml([(accession, {})]).encode()
            return handle

        with patch("metaquest.data.metadata.Entrez.efetch", side_effect=fake_efetch) as mock_efetch:
            with patch("metaquest.data.metadata.Entrez.email", "test@example.com"):
                with patch("metaquest.data.metadata._pace_requests"):
                    successes, failures = _download_batch_metadata(batch, metadata_path, "test@example.com", None)

        assert failures == {}
        assert set(successes) == set(batch)
        # One batch call plus one call per accession in the fallback.
        assert mock_efetch.call_count == 1 + len(batch)

    def test_404_on_single_accession_batch_is_final(self, tmp_path):
        """A 404 on a batch of exactly one accession is a final failure, not retried."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        with patch("metaquest.data.metadata.Entrez.efetch", side_effect=_http_error(404)) as mock_efetch:
            with patch("metaquest.data.metadata._pace_requests"):
                with patch("time.sleep") as mock_sleep:
                    successes, failures = _download_batch_metadata(["SRR1"], metadata_path, "test@example.com", None)

        assert successes == {}
        assert failures == {"SRR1": "HTTP 404: not found at NCBI"}
        mock_efetch.assert_called_once()
        mock_sleep.assert_not_called()

    def test_503_retried_three_times_then_fails(self, tmp_path):
        """A 503 is retried with sleeps 2, 4, 8 and then fails the whole batch."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()
        batch = ["SRR1", "SRR2"]

        with patch("metaquest.data.metadata.Entrez.efetch", side_effect=_http_error(503)):
            with patch("metaquest.data.metadata._pace_requests"):
                with patch("time.sleep") as mock_sleep:
                    successes, failures = _download_batch_metadata(batch, metadata_path, "test@example.com", None)

        assert successes == {}
        assert failures == {accession: "HTTP 503 after 3 attempts" for accession in batch}
        assert mock_sleep.call_args_list == [call(2), call(4), call(8)]

    def test_success_path_sleeps_only_through_pace_requests(self, tmp_path):
        """The only sleep on the success path is the one _pace_requests issues."""
        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()

        xml_text = _package_xml([("SRR1", {"total_spots": "10"})])
        mock_handle = MagicMock()
        mock_handle.read.return_value = xml_text.encode()

        with patch("metaquest.data.metadata.Entrez.efetch", return_value=mock_handle):
            with patch("metaquest.data.metadata._pace_requests") as mock_pace:
                with patch("time.sleep") as mock_sleep:
                    successes, failures = _download_batch_metadata(
                        ["SRR1"], metadata_path, "test@example.com", "my-key"
                    )

        assert failures == {}
        mock_pace.assert_called_once_with("my-key")
        mock_sleep.assert_not_called()


class TestSplitEfetchPackages:
    """Test _split_efetch_packages directly."""

    def test_splits_shared_package_into_independent_documents(self):
        xml_text = _package_xml([("SRR1", {"total_spots": "10"}), ("SRR2", {"total_spots": "20"})])
        result = _split_efetch_packages(xml_text, {"SRR1", "SRR2"})

        assert set(result) == {"SRR1", "SRR2"}
        for accession in ("SRR1", "SRR2"):
            root = ET.fromstring(result[accession])
            assert root.tag == "EXPERIMENT_PACKAGE_SET"
            runs = root.findall(".//RUN")
            assert len(runs) == 1
            assert runs[0].get("accession") == accession

    def test_ignores_accessions_not_in_wanted_set(self):
        xml_text = _package_xml([("SRR1", {}), ("SRR2", {})])
        result = _split_efetch_packages(xml_text, {"SRR1"})
        assert set(result) == {"SRR1"}

    def test_missing_accession_absent_from_result(self):
        xml_text = _package_xml([("SRR1", {})])
        result = _split_efetch_packages(xml_text, {"SRR1", "SRR9"})
        assert set(result) == {"SRR1"}

    def test_includes_xml_declaration(self):
        xml_text = _package_xml([("SRR1", {})])
        result = _split_efetch_packages(xml_text, {"SRR1"})
        assert result["SRR1"].startswith("<?xml")


class TestPaceRequests:
    """Test _pace_requests: sleeps only enough to respect the per-key rate limit."""

    def test_sleeps_remaining_delay_without_api_key(self):
        import metaquest.data.metadata as metadata_module

        with patch("metaquest.data.metadata.time.monotonic", side_effect=[100.1, 100.34]):
            with patch("metaquest.data.metadata.time.sleep") as mock_sleep:
                metadata_module._last_request_time = 100.0
                _pace_requests(None)

        mock_sleep.assert_called_once()
        assert mock_sleep.call_args.args[0] == pytest.approx(0.24, abs=1e-6)

    def test_sleeps_remaining_delay_with_api_key(self):
        import metaquest.data.metadata as metadata_module

        with patch("metaquest.data.metadata.time.monotonic", side_effect=[100.05, 100.1]):
            with patch("metaquest.data.metadata.time.sleep") as mock_sleep:
                metadata_module._last_request_time = 100.0
                _pace_requests("my-key")

        mock_sleep.assert_called_once()
        assert mock_sleep.call_args.args[0] == pytest.approx(0.05, abs=1e-6)

    def test_no_sleep_when_delay_already_elapsed(self):
        import metaquest.data.metadata as metadata_module

        with patch("metaquest.data.metadata.time.monotonic", side_effect=[101.0, 101.0]):
            with patch("metaquest.data.metadata.time.sleep") as mock_sleep:
                metadata_module._last_request_time = 100.0
                _pace_requests(None)

        mock_sleep.assert_not_called()


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

    def test_extract_metadata_fields_ncbi_attributes(self):
        """Real NCBI efetch XML holds spots, bases, size and md5 as attributes, not child text."""
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <EXPERIMENT>
                    <IDENTIFIERS>
                        <PRIMARY_ID>EXP1</PRIMARY_ID>
                    </IDENTIFIERS>
                    <LIBRARY_DESCRIPTOR>
                        <LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY>
                        <LIBRARY_LAYOUT>
                            <PAIRED/>
                        </LIBRARY_LAYOUT>
                    </LIBRARY_DESCRIPTOR>
                    <PLATFORM>
                        <ILLUMINA>
                            <INSTRUMENT_MODEL>Illumina HiSeq 2500</INSTRUMENT_MODEL>
                        </ILLUMINA>
                    </PLATFORM>
                </EXPERIMENT>
                <RUN_SET>
                    <RUN accession="SRR1" total_spots="47964651" total_bases="14389395300" size="4744553813">
                        <IDENTIFIERS>
                            <PRIMARY_ID>SRR1</PRIMARY_ID>
                        </IDENTIFIERS>
                        <SRAFiles>
                            <SRAFile filename="SRR1" md5="abc" semantic_name="run"/>
                        </SRAFiles>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""

        tree = ET.fromstring(xml_content)

        result = _extract_metadata_fields(tree, "test.xml")

        assert result["Run_Total_Spots"] == "47964651"
        assert result["Run_Total_Bases"] == "14389395300"
        assert result["Run_Size"] == "4744553813"
        assert result["Run_MD5"] == "abc"
        assert result["Run_Filename"] == "SRR1"
        assert result["Experiment_Library_Layout"] == "PAIRED"
        assert result["Platform"] == "ILLUMINA"

    def test_extract_metadata_fields_srafile_prefers_run_semantic_name(self):
        """When multiple SRAFile entries exist, prefer the one marked semantic_name=run."""
        xml_content = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <RUN_SET>
                    <RUN accession="SRR2">
                        <SRAFiles>
                            <SRAFile filename="other" md5="wrong" semantic_name="other"/>
                            <SRAFile filename="SRR2" md5="right" semantic_name="run"/>
                        </SRAFiles>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""

        tree = ET.fromstring(xml_content)

        result = _extract_metadata_fields(tree, "test.xml")

        assert result["Run_MD5"] == "right"
        assert result["Run_Filename"] == "SRR2"


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


class TestParseMetadataXml:
    """Test parse_metadata_xml: single-file counterpart to parse_metadata used by the sidecar."""

    NCBI_XML = """<?xml version="1.0"?>
        <EXPERIMENT_PACKAGE_SET>
            <EXPERIMENT_PACKAGE>
                <EXPERIMENT>
                    <IDENTIFIERS>
                        <PRIMARY_ID>EXP1</PRIMARY_ID>
                    </IDENTIFIERS>
                    <LIBRARY_DESCRIPTOR>
                        <LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY>
                        <LIBRARY_LAYOUT>
                            <PAIRED/>
                        </LIBRARY_LAYOUT>
                    </LIBRARY_DESCRIPTOR>
                    <PLATFORM>
                        <ILLUMINA>
                            <INSTRUMENT_MODEL>Illumina HiSeq 2500</INSTRUMENT_MODEL>
                        </ILLUMINA>
                    </PLATFORM>
                </EXPERIMENT>
                <RUN_SET>
                    <RUN accession="SRR1" total_spots="47964651" total_bases="14389395300" size="4744553813">
                        <IDENTIFIERS>
                            <PRIMARY_ID>SRR1</PRIMARY_ID>
                        </IDENTIFIERS>
                        <SRAFiles>
                            <SRAFile filename="SRR1" md5="abc" semantic_name="run"/>
                        </SRAFiles>
                    </RUN>
                </RUN_SET>
            </EXPERIMENT_PACKAGE>
        </EXPERIMENT_PACKAGE_SET>"""

    def test_parse_metadata_xml_reads_ncbi_attributes(self, tmp_path):
        """A single metadata XML file parses to the same fields _extract_metadata_fields gives."""
        xml_file = tmp_path / "SRR1.xml"
        xml_file.write_text(self.NCBI_XML)

        result = parse_metadata_xml(xml_file)

        assert result["Run_Total_Spots"] == "47964651"
        assert result["Run_Total_Bases"] == "14389395300"
        assert result["Run_Size"] == "4744553813"
        assert result["Run_MD5"] == "abc"
        assert result["Run_Filename"] == "SRR1"
        assert result["Experiment_Library_Layout"] == "PAIRED"

    def test_parse_metadata_xml_missing_file_returns_empty_dict(self):
        """A missing metadata file is not an error: parse_metadata_xml returns {} with a warning."""
        result = parse_metadata_xml("/nonexistent/SRR1.xml")
        assert result == {}

    def test_parse_metadata_xml_unparsable_file_returns_empty_dict(self, tmp_path):
        """A malformed XML file also returns {} rather than raising."""
        xml_file = tmp_path / "bad.xml"
        xml_file.write_text("not xml at all <<<")

        result = parse_metadata_xml(xml_file)

        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__])
