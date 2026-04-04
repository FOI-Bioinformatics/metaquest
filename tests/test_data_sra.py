"""
Tests for metaquest.data.sra module.
"""

import pytest
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import ThreadPoolExecutor, Future

from metaquest.core.exceptions import DataAccessError, SecurityError
from metaquest.data.sra import (
    _read_blacklist_files,
    _prepare_temp_folder,
    _check_existing_download,
    _handle_download_output,
    download_accession,
    _check_existing_downloads,
    _process_download_results,
    _retry_failed_downloads,
    _handle_download_failure,
    download_sra,
    _find_paired_reads,
    assemble_datasets,
)


class TestReadBlacklistFiles:
    """Test _read_blacklist_files function."""

    def test_read_blacklist_files_success(self, tmp_path):
        """Test successful blacklist file reading."""
        blacklist1 = tmp_path / "blacklist1.txt"
        blacklist2 = tmp_path / "blacklist2.txt"
        
        blacklist1.write_text("SRR123\nSRR456\n\n")  # Include empty line
        blacklist2.write_text("SRR789\nSRR123\n")    # Include duplicate
        
        result = _read_blacklist_files([blacklist1, blacklist2])
        
        assert result == {"SRR123", "SRR456", "SRR789"}

    def test_read_blacklist_files_empty_list(self):
        """Test with empty blacklist files list."""
        result = _read_blacklist_files([])
        assert result == set()

    def test_read_blacklist_files_none(self):
        """Test with None blacklist files."""
        result = _read_blacklist_files(None)
        assert result == set()

    def test_read_blacklist_files_nonexistent_file(self, tmp_path):
        """Test handling nonexistent blacklist file."""
        nonexistent = tmp_path / "nonexistent.txt"
        
        with patch('metaquest.data.sra.logger') as mock_logger:
            result = _read_blacklist_files([nonexistent])
        
        assert result == set()
        mock_logger.warning.assert_called_once()

    def test_read_blacklist_files_empty_file(self, tmp_path):
        """Test handling empty blacklist file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        result = _read_blacklist_files([empty_file])
        assert result == set()

    def test_read_blacklist_files_whitespace_handling(self, tmp_path):
        """Test proper whitespace handling."""
        blacklist = tmp_path / "blacklist.txt"
        blacklist.write_text("  SRR123  \n\t SRR456 \t\n   \n")
        
        result = _read_blacklist_files([blacklist])
        assert result == {"SRR123", "SRR456"}


class TestPrepareTempFolder:
    """Test _prepare_temp_folder function."""

    def test_prepare_temp_folder_create_new(self, tmp_path):
        """Test creating new temp folder."""
        temp_base = tmp_path / "temp"

        result = _prepare_temp_folder(temp_base)

        assert result.exists()
        assert result.is_dir()

    def test_prepare_temp_folder_existing_folder(self, tmp_path):
        """Test with existing temp folder."""
        temp_base = tmp_path / "temp"
        temp_base.mkdir()

        result = _prepare_temp_folder(temp_base)

        assert result.exists()
        assert result.is_dir()

    def test_prepare_temp_folder_none_input(self):
        """Test with None temp folder."""
        import tempfile
        
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/test_temp"
            
            result = _prepare_temp_folder(None)
            
            mock_mkdtemp.assert_called_once()
            assert str(result) == "/tmp/test_temp"


class TestCheckExistingDownload:
    """Test _check_existing_download function."""

    def test_check_existing_download_no_files(self, tmp_path):
        """Test when no files exist."""
        output_path = tmp_path / "downloads" / "SRR123"
        
        result = _check_existing_download(output_path, force=False)
        assert result is False

    def test_check_existing_download_with_files_no_force(self, tmp_path):
        """Test when files exist and force=False."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("test data")
        
        result = _check_existing_download(output_path, force=False)
        assert result is True

    def test_check_existing_download_with_files_force(self, tmp_path):
        """Test when files exist and force=True."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("test data")
        
        with patch('shutil.rmtree') as mock_rmtree:
            result = _check_existing_download(output_path, force=True)
        
        assert result is False
        mock_rmtree.assert_called_once_with(output_path)

    def test_check_existing_download_empty_directory(self, tmp_path):
        """Test when directory exists but is empty."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)
        
        with patch('metaquest.data.sra.logger') as mock_logger:
            result = _check_existing_download(output_path, force=False)
        
        assert result is False
        # Directory should be removed
        assert not output_path.exists()

    def test_check_existing_download_rmdir_error(self, tmp_path):
        """Test handling rmdir error."""
        output_path = tmp_path / "downloads" / "SRR123"
        output_path.mkdir(parents=True)
        
        with patch.object(Path, 'rmdir', side_effect=OSError("Permission denied")):
            with patch('metaquest.data.sra.logger') as mock_logger:
                result = _check_existing_download(output_path, force=False)
        
        assert result is False
        mock_logger.warning.assert_called()


class TestHandleDownloadOutput:
    """Test _handle_download_output function."""

    def test_handle_download_output_success(self, tmp_path):
        """Test successful file moving."""
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output"
        temp_path.mkdir()
        
        # Create test FASTQ files
        (temp_path / "SRR123.fastq").write_text("@seq1\nACGT\n+\nIIII\n")
        (temp_path / "SRR123_2.fastq").write_text("@seq2\nTGCA\n+\nIIII\n")
        
        success, message = _handle_download_output(temp_path, output_path)
        
        assert success is True
        assert "2 files" in message
        assert (output_path / "SRR123.fastq").exists()
        assert (output_path / "SRR123_2.fastq").exists()
        assert not temp_path.exists()

    def test_handle_download_output_no_fastq_files(self, tmp_path):
        """Test when no FASTQ files are found."""
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output"
        temp_path.mkdir()
        
        # Create non-FASTQ file
        (temp_path / "other.txt").write_text("not a fastq")
        
        with patch('shutil.rmtree') as mock_rmtree:
            with patch('metaquest.data.sra.logger') as mock_logger:
                success, message = _handle_download_output(temp_path, output_path)
        
        assert success is False
        assert "No FASTQ files created" in message
        mock_logger.error.assert_called()
        mock_rmtree.assert_called_once_with(temp_path)

    def test_handle_download_output_rmdir_error(self, tmp_path):
        """Test handling rmtree error during cleanup."""
        temp_path = tmp_path / "temp"
        output_path = tmp_path / "output"
        temp_path.mkdir()

        (temp_path / "SRR123.fastq").write_text("@seq1\nACGT\n+\nIIII\n")

        with patch('shutil.rmtree', side_effect=OSError("Permission denied")):
            with patch('shutil.move'):
                with patch('metaquest.data.sra.logger') as mock_logger:
                    success, message = _handle_download_output(temp_path, output_path)

        assert success is True
        mock_logger.warning.assert_called()


class TestDownloadAccession:
    """Test download_accession function."""

    def test_download_accession_existing_skip(self, tmp_path):
        """Test skipping existing download."""
        output_folder = tmp_path / "downloads"
        output_path = output_folder / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("existing")
        
        success, message = download_accession("SRR123", output_folder, force=False)
        
        assert success is True
        assert "already exists" in message

    def test_download_accession_force_existing(self, tmp_path):
        """Test forcing download of existing accession."""
        output_folder = tmp_path / "downloads"
        output_path = output_folder / "SRR123"
        output_path.mkdir(parents=True)
        (output_path / "SRR123.fastq").write_text("existing")
        
        with patch('metaquest.data.sra._prepare_temp_folder') as mock_prep:
            with patch('metaquest.utils.security.SecureSubprocess.run_secure') as mock_run:
                with patch('metaquest.data.sra._handle_download_output') as mock_handle:
                    mock_prep.return_value = tmp_path / "temp"
                    mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")
                    mock_handle.return_value = (True, "Downloaded 1 files")
                    
                    success, message = download_accession("SRR123", output_folder, force=True)
        
        assert "Downloaded 1 files" in message

    def test_download_accession_security_error(self, tmp_path):
        """Test handling security error."""
        output_folder = tmp_path / "downloads"
        
        with patch('metaquest.data.sra._prepare_temp_folder') as mock_prep:
            with patch('metaquest.utils.security.SecureSubprocess.run_secure') as mock_run:
                mock_prep.return_value = tmp_path / "temp"
                mock_run.side_effect = SecurityError("Command blocked")
                
                success, message = download_accession("SRR123", output_folder)
        
        assert success is False
        assert "Security error" in message

    def test_download_accession_command_failure(self, tmp_path):
        """Test handling command failure."""
        import subprocess
        output_folder = tmp_path / "downloads"
        
        with patch('metaquest.data.sra._prepare_temp_folder') as mock_prep:
            with patch('metaquest.utils.security.SecureSubprocess.run_secure') as mock_run:
                mock_prep.return_value = tmp_path / "temp"
                mock_run.side_effect = subprocess.CalledProcessError(1, "fasterq-dump", stderr="Error occurred")
                
                success, message = download_accession("SRR123", output_folder)
        
        assert success is False
        assert "Download failed" in message

    def test_download_accession_success(self, tmp_path):
        """Test successful download."""
        output_folder = tmp_path / "downloads"
        temp_path = tmp_path / "temp"
        
        with patch('metaquest.data.sra._prepare_temp_folder') as mock_prep:
            with patch('metaquest.utils.security.SecureSubprocess.run_secure') as mock_run:
                with patch('metaquest.data.sra._handle_download_output') as mock_handle:
                    mock_prep.return_value = temp_path
                    mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")
                    mock_handle.return_value = (True, "Downloaded 2 files")
                    
                    success, message = download_accession("SRR123", output_folder, num_threads=8)
        
        assert success is True
        assert "Downloaded 2 files" in message
        # Verify command was called with correct arguments
        mock_run.assert_called_once()
        executable = mock_run.call_args[0][0]
        args = mock_run.call_args[0][1]
        assert executable == "fasterq-dump"
        assert "SRR123" in args
        assert "--threads" in args
        assert "8" in args


class TestCheckExistingDownloads:
    """Test _check_existing_downloads function."""

    def test_check_existing_downloads_none_exist(self, tmp_path):
        """Test when no downloads exist."""
        output_folder = tmp_path / "downloads"
        accessions = ["SRR123", "SRR456"]
        
        already_downloaded, to_download, blacklisted = _check_existing_downloads(accessions, output_folder, force=False)
        
        assert to_download == accessions
        assert already_downloaded == []

    def test_check_existing_downloads_some_exist(self, tmp_path):
        """Test when some downloads exist."""
        output_folder = tmp_path / "downloads"
        (output_folder / "SRR123").mkdir(parents=True)
        (output_folder / "SRR123" / "SRR123.fastq").write_text("data")
        
        accessions = ["SRR123", "SRR456"]
        
        already_downloaded, to_download, blacklisted = _check_existing_downloads(accessions, output_folder, force=False)
        
        assert to_download == ["SRR456"]
        assert already_downloaded == ["SRR123"]

    def test_check_existing_downloads_force_all(self, tmp_path):
        """Test with force=True."""
        output_folder = tmp_path / "downloads"
        (output_folder / "SRR123").mkdir(parents=True)
        (output_folder / "SRR123" / "SRR123.fastq").write_text("data")
        
        accessions = ["SRR123", "SRR456"]
        
        with patch('shutil.rmtree'):
            already_downloaded, to_download, blacklisted = _check_existing_downloads(accessions, output_folder, force=True)
        
        assert to_download == accessions
        assert already_downloaded == []


class TestProcessDownloadResults:
    """Test _process_download_results function."""

    def test_process_download_results_all_success(self):
        """Test processing all successful results."""
        # Create mock futures
        future1 = Mock()
        future1.result.return_value = (True, "Success message 1")
        future2 = Mock()
        future2.result.return_value = (True, "Success message 2")
        
        futures_results = [
            ("SRR123", (True, "Success message 1")),
            ("SRR456", (True, "Success message 2"))
        ]
        accessions_to_download = ["SRR123", "SRR456"]
        download_results = {}
        failed_accessions = []
        
        _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions)
        
        assert download_results == {"SRR123": "Success message 1", "SRR456": "Success message 2"}
        assert failed_accessions == []

    def test_process_download_results_some_failures(self):
        """Test processing mixed success/failure results."""
        future1 = Mock()
        future1.result.return_value = (True, "Success message")
        future2 = Mock()
        future2.result.return_value = (False, "Error message")
        
        futures_results = [
            ("SRR123", (True, "Success message")),
            ("SRR456", (False, "Error message"))
        ]
        accessions_to_download = ["SRR123", "SRR456"]
        download_results = {}
        failed_accessions = []
        
        _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions)
        
        assert download_results == {"SRR123": "Success message", "SRR456": "Error message"}
        assert failed_accessions == ["SRR456"]

    def test_process_download_results_exception(self):
        """Test handling exceptions in futures."""
        future1 = Mock()
        future1.result.return_value = (True, "Success message")
        future2 = Mock()
        future2.result.side_effect = Exception("Future failed")
        
        futures_results = [
            ("SRR123", (True, "Success message")),
            ("SRR456", Exception("Future failed"))
        ]
        accessions_to_download = ["SRR123", "SRR456"]
        download_results = {}
        failed_accessions = []
        
        with patch('metaquest.data.sra.logger') as mock_logger:
            _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions)
        
        assert download_results == {"SRR123": "Success message", "SRR456": "Error: cannot unpack non-iterable Exception object"}
        assert failed_accessions == ["SRR456"]
        mock_logger.error.assert_called()


class TestRetryFailedDownloads:
    """Test _retry_failed_downloads function."""

    def test_retry_failed_downloads_success(self, tmp_path):
        """Test successful retry of failed downloads."""
        failed_accessions = ["SRR123", "SRR456"]
        download_results = {}
        
        with patch('metaquest.data.sra.download_accession') as mock_download:
            mock_download.side_effect = [
                (True, "Retry success 1"),
                (True, "Retry success 2")
            ]
            
            retried_successful, updated_failed = _retry_failed_downloads(failed_accessions, max_retries=2, fastq_path=tmp_path, num_threads=4, temp_folder=None, download_results=download_results)
        
        assert download_results == {"SRR123": "Retry 1: Retry success 1", "SRR456": "Retry 1: Retry success 2"}
        assert len(updated_failed) == 0
        assert retried_successful == 2

    def test_retry_failed_downloads_partial_success(self, tmp_path):
        """Test partial success in retry."""
        failed_accessions = ["SRR123", "SRR456"]
        download_results = {}
        
        with patch('metaquest.data.sra.download_accession') as mock_download:
            mock_download.side_effect = [
                (True, "Retry success"),
                (False, "Retry failed")
            ]
            
            retried_successful, updated_failed = _retry_failed_downloads(failed_accessions, max_retries=2, fastq_path=tmp_path, num_threads=4, temp_folder=None, download_results=download_results)
        
        assert download_results == {"SRR123": "Retry 1: Retry success", "SRR456": "Retry 2 error: "}
        assert updated_failed == ["SRR456"]
        assert retried_successful == 1


class TestHandleDownloadFailure:
    """Test _handle_download_failure function."""

    def test_handle_download_failure_no_failures(self, tmp_path):
        """Test when no failures occurred."""
        fastq_path = tmp_path / "downloads"
        failed_accessions = []
        
        with patch('metaquest.data.sra.logger') as mock_logger:
            _handle_download_failure(fastq_path, failed_accessions)
        
        # No logging should occur when there are no failures
        mock_logger.info.assert_not_called()

    def test_handle_download_failure_with_failures(self, tmp_path):
        """Test when failures occurred."""
        fastq_path = tmp_path / "downloads"
        fastq_path.mkdir(parents=True, exist_ok=True)
        failed_accessions = ["SRR123", "SRR456"]
        
        with patch('metaquest.data.sra.logger') as mock_logger:
            _handle_download_failure(fastq_path, failed_accessions)
        
        mock_logger.info.assert_called()
        # Check that failed accessions file was created
        failed_file = fastq_path / "failed_accessions.txt"
        assert failed_file.exists()
        content = failed_file.read_text()
        assert "SRR123" in content
        assert "SRR456" in content


class TestDownloadSra:
    """Test download_sra function."""

    def test_download_sra_from_file(self, tmp_path):
        """Test downloading SRA data from accessions file."""
        accessions_file = tmp_path / "accessions.txt"
        output_folder = tmp_path / "downloads"
        accessions_file.write_text("SRR123\nSRR456\n")
        
        with patch('metaquest.data.sra._check_existing_downloads') as mock_check:
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                with patch('metaquest.data.sra._process_download_results') as mock_process:
                    with patch('metaquest.data.sra._handle_download_failure') as mock_handle:
                        mock_check.return_value = ([], ["SRR123", "SRR456"], [])
                        mock_executor_instance = Mock()
                        mock_executor.return_value.__enter__.return_value = mock_executor_instance
                        mock_executor_instance.submit.return_value = Mock()
                        mock_process.return_value = (2, 0)
                        
                        result = download_sra(output_folder, accessions_file, num_threads=4, max_workers=2)
        
        assert isinstance(result, dict)
        mock_check.assert_called_once()

    def test_download_sra_with_blacklist(self, tmp_path):
        """Test downloading with blacklist filtering."""
        accessions_file = tmp_path / "accessions.txt"
        blacklist_file = tmp_path / "blacklist.txt"
        output_folder = tmp_path / "downloads"
        
        accessions_file.write_text("SRR123\nSRR456\nSRR789\n")
        blacklist_file.write_text("SRR456\n")
        
        with patch('metaquest.data.sra._check_existing_downloads') as mock_check:
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                with patch('metaquest.data.sra._process_download_results') as mock_process:
                    with patch('metaquest.data.sra._handle_download_failure') as mock_handle:
                        mock_check.return_value = ([], ["SRR123", "SRR789"], [])
                        mock_executor_instance = Mock()
                        mock_executor.return_value.__enter__.return_value = mock_executor_instance
                        mock_process.return_value = (2, 0)
                        
                        result = download_sra(
                            output_folder, 
                            accessions_file, 
                            blacklist=[blacklist_file]
                        )
        
        # Verify that blacklist filtering works correctly
        check_call_args = mock_check.call_args
        all_accessions = check_call_args[0][0]  # First positional argument
        blacklisted_accessions = check_call_args[0][3]  # Fourth positional argument (set of blacklisted)
        
        # All accessions should be passed to the function
        assert "SRR123" in all_accessions
        assert "SRR456" in all_accessions  
        assert "SRR789" in all_accessions
        # But SRR456 should be in the blacklist set
        assert "SRR456" in blacklisted_accessions

    def test_download_sra_nonexistent_file(self, tmp_path):
        """Test error handling for nonexistent accessions file."""
        nonexistent_file = tmp_path / "nonexistent.txt"
        output_folder = tmp_path / "downloads"
        
        with pytest.raises(DataAccessError):
            download_sra(output_folder, nonexistent_file)

    def test_download_sra_dry_run(self, tmp_path):
        """Test dry run mode."""
        accessions_file = tmp_path / "accessions.txt"
        output_folder = tmp_path / "downloads"
        accessions_file.write_text("SRR123\nSRR456\n")
        
        with patch('metaquest.data.sra._check_existing_downloads') as mock_check:
            with patch('metaquest.data.sra.logger') as mock_logger:
                mock_check.return_value = ([], ["SRR123", "SRR456"], [])
                result = download_sra(output_folder, accessions_file, dry_run=True)
        
        expected_result = {
            "total": 2,
            "already_downloaded": 0,
            "blacklisted": 0,
            "to_download": 2,
            "successful": 0,
            "failed": 0,
        }
        assert result == expected_result
        mock_logger.info.assert_called_with("Dry run: would download 2 accessions")


class TestFindPairedReads:
    """Test _find_paired_reads function."""

    def test_find_paired_reads_standard_naming(self, tmp_path):
        """Test finding paired reads with standard naming."""
        files = [
            tmp_path / "SRR456_R1.fastq",
            tmp_path / "SRR456_R2.fastq", 
            tmp_path / "SRR789_R1.fastq"  # Single R1 without R2
        ]
        
        # Create all files
        for file in files:
            file.write_text("@seq1\nACGT\n+\nIIII\n")
        
        # Test with only the R1 file (function should find the R2 automatically)
        result = _find_paired_reads([files[0]])  # Only pass SRR456_R1.fastq
        
        # Should find 1 pair
        assert len(result) == 1
        r1_file, r2_file = result[0]
        assert r1_file == files[0]  # SRR456_R1.fastq
        assert r2_file == files[1]  # SRR456_R2.fastq

    def test_find_paired_reads_no_pairs(self, tmp_path):
        """Test when no paired reads are found."""
        files = [
            tmp_path / "SRR123.fastq",  # No R1/R2 naming
            tmp_path / "SRR456.fastq"
        ]
        
        for file in files:
            file.write_text("@seq1\nACGT\n+\nIIII\n")
        
        result = _find_paired_reads(files)
        
        # Should find no pairs since no files have R1 in their names
        assert len(result) == 0

    def test_find_paired_reads_empty_list(self):
        """Test with empty file list."""
        result = _find_paired_reads([])
        assert result == []


class TestAssembleDatasets:
    """Test assemble_datasets function."""

    def test_assemble_datasets_mixed_technologies(self, tmp_path):
        """Test assembling datasets with mixed technologies."""
        # Create test files with R1/R2 naming for Illumina
        illumina_files = [
            tmp_path / "SRR456_R1.fastq",
            tmp_path / "SRR456_R2.fastq"
        ]
        nanopore_files = [
            tmp_path / "SRR789.fastq"  # No R1/R2 in name = Nanopore
        ]
        
        for file in illumina_files + nanopore_files:
            file.write_text("@seq1\nACGT\n+\nIIII\n")
        
        # Mock args
        args = Mock()
        args.fastq_folder = tmp_path
        args.output_file = tmp_path / "datasets.json"
        
        result = assemble_datasets(args)
        
        # Should return empty list (no actual processing in updated function)
        assert isinstance(result, list)
        assert result == []
        # Output file should be created
        assert args.output_file.exists()

    def test_assemble_datasets_no_fastq_files(self, tmp_path):
        """Test when no FASTQ files are found."""
        args = Mock()
        args.fastq_folder = tmp_path
        args.output_file = tmp_path / "datasets.json"
        
        with patch('metaquest.data.sra.logger') as mock_logger:
            result = assemble_datasets(args)
        
        assert result == []
        mock_logger.warning.assert_called_with(f"No FASTQ files found in {tmp_path}")

    def test_assemble_datasets_nonexistent_folder(self, tmp_path):
        """Test with nonexistent FASTQ folder."""
        nonexistent = tmp_path / "nonexistent"
        
        args = Mock()
        args.fastq_folder = nonexistent
        args.output_file = tmp_path / "datasets.json"
        
        with pytest.raises(DataAccessError):
            assemble_datasets(args)


if __name__ == "__main__":
    pytest.main([__file__])