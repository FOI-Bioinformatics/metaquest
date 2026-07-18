"""
SRA data handling for MetaQuest.

This module provides functions for downloading and processing SRA data.
"""

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from metaquest.core.exceptions import DataAccessError, ProcessingError, SecurityError
from metaquest.data.file_io import ensure_directory
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree if present, logging on failure instead of raising."""
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"Could not remove directory {path}: {e}")


def accession_has_fastq(acc_dir: Union[str, Path]) -> bool:
    """Return True if the per-accession directory holds at least one FASTQ file.

    This is the single source of truth for "this accession is already
    downloaded" used across the download and status paths (a per-accession
    subdirectory containing any ``*.fastq*`` file).
    """
    acc_path = Path(acc_dir)
    return acc_path.is_dir() and any(acc_path.glob("*.fastq*"))


def _read_blacklist_files(blacklist_files):
    """
    Read accessions from blacklist files.

    Args:
        blacklist_files: List of blacklist file paths

    Returns:
        Set of blacklisted accessions
    """
    blacklisted_accessions: set = set()

    if not blacklist_files:
        return blacklisted_accessions

    for blacklist_file in blacklist_files:
        try:
            file_accessions = set()
            with open(blacklist_file, "r") as f:
                for line in f:
                    accession = line.strip()
                    if accession:
                        file_accessions.add(accession)
                        blacklisted_accessions.add(accession)
            logger.info(f"Read {len(file_accessions)} blacklisted accessions from {blacklist_file}")
        except Exception as e:
            logger.warning(f"Error reading blacklist file {blacklist_file}: {e}")

    return blacklisted_accessions


def _prepare_temp_folder(temp_folder):
    """
    Prepare the temporary folder for fasterq-dump.

    Args:
        temp_folder: Path to temporary folder

    Returns:
        Path object of the prepared temp folder, or None if not successful
    """
    import tempfile

    if not temp_folder:
        # Create a temporary directory
        try:
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary folder: {temp_dir}")
            return Path(temp_dir)
        except Exception as e:
            logger.warning(f"Could not create temporary folder: {e}")
            return None

    # Ensure temp folder exists
    temp_path_obj = Path(temp_folder)
    try:
        temp_path_obj.mkdir(parents=True, exist_ok=True)
        if not os.access(temp_path_obj, os.W_OK):
            logger.warning(f"Temp folder {temp_folder} exists but is not writable, " "using default temp location")
            return None
        else:
            logger.info(f"Using temp folder: {temp_path_obj.absolute()}")
            return temp_path_obj
    except Exception as e:
        logger.warning(f"Could not create or access temp folder {temp_folder}: {e}, " "using default temp location")
        return None


def _check_existing_download(output_path, force):
    """
    Check if the accession is already downloaded.

    Args:
        output_path: Path to output directory
        force: Whether to force redownload

    Returns:
        True if already downloaded, False otherwise
    """
    import shutil

    if force and output_path.exists():
        # Force redownload - remove existing directory
        try:
            shutil.rmtree(output_path)
            logger.info(f"Removed existing directory for force redownload: {output_path}")
        except Exception as e:
            logger.warning(f"Could not remove directory for force redownload {output_path}: {e}")
        return False

    if not force and output_path.exists():
        if accession_has_fastq(output_path):
            return True

        # Found empty directory, will redownload
        try:
            output_path.rmdir()
        except Exception as e:
            logger.warning(f"Could not remove empty directory {output_path}: {e}")

    return False


def _handle_download_output(temp_path, output_path):
    """
    Move downloaded files from temp path to output path.

    Args:
        temp_path: Path to temporary folder
        output_path: Path to output directory

    Returns:
        Tuple of (success, message)
    """
    # Check if files were actually created
    fastq_files = list(temp_path.glob("*.fastq*"))
    if not fastq_files:
        logger.error("No FASTQ files created despite successful command execution")
        # Clean up temp directory
        shutil.rmtree(temp_path)
        return False, "No FASTQ files created"

    # Move files to the final location
    # First ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    for file in fastq_files:
        shutil.move(str(file), str(output_path / file.name))

    # Remove the temporary directory
    try:
        shutil.rmtree(temp_path)
    except Exception as e:
        logger.warning(f"Could not remove temp directory {temp_path}: {e}")

    logger.info(f"Successfully downloaded: {len(fastq_files)} files")
    return True, f"Downloaded {len(fastq_files)} files"


def download_accession(
    accession: str,
    output_folder: Union[str, Path],
    num_threads: int = 4,
    force: bool = False,
    temp_folder: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """
    Download a single SRA accession using fasterq-dump.

    Args:
        accession: SRA accession to download
        output_folder: Folder to save the downloaded files
        num_threads: Number of threads to use for download
        force: If True, redownload even if files exist
        temp_folder: Directory for temporary files

    Returns:
        Tuple of (success, message)
    """
    output_path = Path(output_folder) / accession

    # Check if already downloaded
    if _check_existing_download(output_path, force):
        logger.info(f"Skipping {accession}, FASTQ files already exist")
        return True, "already exists"

    # Create a fresh temporary folder for download
    temp_path = Path(output_folder) / f"{accession}_temp"
    _safe_rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)

    temp_folder_path = None
    try:
        logger.info(f"Downloading SRA for {accession}")

        # Handle temp folder for fasterq-dump
        temp_folder_path = _prepare_temp_folder(temp_folder)

        # Build fasterq-dump arguments
        args = [
            "--threads",
            str(num_threads),
            "--progress",
            accession,
            "-O",
            str(temp_path),
        ]

        # Add temp folder if available
        if temp_folder_path:
            args.extend(["--temp", str(temp_folder_path.absolute())])

        # Run fasterq-dump command securely
        SecureSubprocess.run_secure("fasterq-dump", args)

        # Handle download output
        return _handle_download_output(temp_path, output_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading {accession}: {e.stderr}")
        _safe_rmtree(temp_path)
        return False, f"Download failed: {e.stderr}"

    except SecurityError as e:
        logger.error(f"Security error downloading {accession}: {e}")
        _safe_rmtree(temp_path)
        return False, f"Security error: {e}"

    except Exception as e:
        logger.error(f"Error downloading {accession}: {e}")
        _safe_rmtree(temp_path)
        return False, f"Download failed: {str(e)}"

    finally:
        # Clean up auto-created temp directory (from tempfile.mkdtemp)
        if temp_folder_path and not temp_folder:
            _safe_rmtree(temp_folder_path)


def _check_existing_downloads(
    accessions: List[str],
    fastq_path: Path,
    force: bool,
    blacklisted_accessions: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Check which accessions need downloading and which are already downloaded or blacklisted.

    Args:
        accessions: List of accessions
        fastq_path: Path to FASTQ directory
        force: Whether to force redownload
        blacklisted_accessions: Set of blacklisted accessions

    Returns:
        Tuple of (already_downloaded, to_download, blacklisted)
    """
    already_downloaded = []
    to_download = []
    blacklisted = []

    if blacklisted_accessions is None:
        blacklisted_accessions = set()

    for acc in accessions:
        if acc in blacklisted_accessions:
            blacklisted.append(acc)
            continue

        if not force and accession_has_fastq(fastq_path / acc):
            already_downloaded.append(acc)
        else:
            to_download.append(acc)

    return already_downloaded, to_download, blacklisted


def _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions):
    """
    Process download results from completed futures.

    Args:
        futures_results: List of (accession, future_result) tuples
        accessions_to_download: List of accessions that were attempted
        download_results: Dictionary to store results
        failed_accessions: List to store failed accessions

    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful_count = 0
    failed_count = 0

    for accession, result in futures_results:
        try:
            success, message = result
            download_results[accession] = message

            if success:
                successful_count += 1
                # Log progress periodically
                if successful_count % 5 == 0:
                    logger.info(
                        f"Downloaded {successful_count}/{len(accessions_to_download)} " f"({failed_count} failed)"
                    )
            else:
                failed_count += 1
                failed_accessions.append(accession)
                logger.warning(f"Failed to download {accession}: {message}")

        except Exception as e:
            failed_count += 1
            failed_accessions.append(accession)
            logger.error(f"Error processing download result for {accession}: {e}")
            download_results[accession] = f"Error: {str(e)}"

    return successful_count, failed_count


def _retry_failed_downloads(
    failed_accessions,
    max_retries,
    fastq_path,
    num_threads,
    temp_folder,
    download_results,
):
    """
    Retry failed downloads.

    Args:
        failed_accessions: List of accessions that failed
        max_retries: Maximum number of retry attempts
        fastq_path: Path to FASTQ directory
        num_threads: Number of threads to use
        temp_folder: Temporary folder path
        download_results: Dictionary to store results

    Returns:
        Tuple of (retried_successful, failed_accessions)
    """
    if max_retries <= 0 or not failed_accessions:
        return 0, failed_accessions

    logger.info(f"Retrying {len(failed_accessions)} failed downloads")
    retry_count = 0
    retried_successful = 0

    for retry in range(max_retries):
        if not failed_accessions:
            break

        logger.info(f"Retry attempt {retry + 1}/{max_retries}")
        retry_batch = failed_accessions.copy()
        failed_accessions = []

        for accession in retry_batch:
            retry_count += 1
            try:
                success, message = download_accession(
                    accession,
                    fastq_path,
                    num_threads,
                    force=True,
                    temp_folder=temp_folder,
                )
                download_results[accession] = f"Retry {retry + 1}: {message}"

                if success:
                    retried_successful += 1
                    logger.info(f"Successfully downloaded {accession} on retry {retry + 1}")
                else:
                    failed_accessions.append(accession)
                    logger.warning(f"Failed to download {accession} on retry {retry + 1}: {message}")

            except Exception as e:
                failed_accessions.append(accession)
                logger.error(f"Error retrying download for {accession}: {e}")
                download_results[accession] = f"Retry {retry + 1} error: {str(e)}"

    return retried_successful, failed_accessions


def _handle_download_failure(fastq_path, failed_accessions):
    """
    Handle failed downloads by writing failed accessions to a file.

    Args:
        fastq_path: Path to FASTQ directory
        failed_accessions: List of failed accessions
    """
    if not failed_accessions:
        return

    # Write failed accessions to file for easier retry
    failed_file = Path(fastq_path) / "failed_accessions.txt"
    with open(failed_file, "w") as f:
        for acc in failed_accessions:
            f.write(f"{acc}\n")

    logger.info(f"Failed accessions written to {failed_file}")
    logger.info(
        f"To retry only failed accessions: metaquest download_sra "
        f"--accessions-file {failed_file} "
        f"--fastq-folder {fastq_path}"
    )


def _execute_parallel_downloads(
    accessions, fastq_path, num_threads, max_workers, force, temp_folder, download_results, failed_accessions
):
    """Download accessions concurrently and tally results. Returns (successful, failed)."""
    futures_results: list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_accession, acc, fastq_path, num_threads, force, temp_folder): acc
            for acc in accessions
        }
        for future in as_completed(futures):
            acc = futures[future]
            try:
                futures_results.append((acc, future.result()))
            except Exception as e:
                logger.error(f"Download failed for {acc}: {e}")
                futures_results.append((acc, None))

    return _process_download_results(futures_results, accessions, download_results, failed_accessions)


def _resolve_fastq_path(fastq_folder: Union[str, Path], dry_run: bool) -> Path:
    """Return the FASTQ output path, creating it unless in dry-run mode.

    In dry-run mode the folder is not created, but an existing non-directory
    at that location is still rejected.
    """
    fastq_path = Path(fastq_folder)
    if dry_run:
        if fastq_path.exists() and not fastq_path.is_dir():
            raise DataAccessError(f"{fastq_folder} exists but is not a directory")
        logger.info(f"Dry run mode: Would use {fastq_path} for downloads")
        return fastq_path
    return ensure_directory(fastq_folder)


def _download_with_retries(
    accessions_to_download, fastq_path, num_threads, max_workers, force, temp_folder, max_retries
) -> Tuple[int, int, List[str], Dict[str, Any]]:
    """Run the parallel downloads and optional retry pass.

    Returns (successful_count, failed_count, failed_accessions, download_results).
    """
    failed_accessions: list = []
    download_results: dict = {}
    successful_count, failed_count = _execute_parallel_downloads(
        accessions_to_download,
        fastq_path,
        num_threads,
        max_workers,
        force,
        temp_folder,
        download_results,
        failed_accessions,
    )

    if max_retries > 0 and failed_accessions:
        retried_successful, failed_accessions = _retry_failed_downloads(
            failed_accessions,
            max_retries,
            fastq_path,
            num_threads,
            temp_folder,
            download_results,
        )
        successful_count += retried_successful
        failed_count -= retried_successful
        if retried_successful > 0:
            logger.info(f"Successfully downloaded {retried_successful} accessions on retry")

    return successful_count, failed_count, failed_accessions, download_results


def download_sra(
    fastq_folder: Union[str, Path],
    accessions_file: Union[str, Path],
    max_downloads: Optional[int] = None,
    dry_run: bool = False,
    num_threads: int = 4,
    max_workers: int = 4,
    force: bool = False,
    max_retries: int = 1,
    temp_folder: Optional[Union[str, Path]] = None,
    blacklist: Optional[List[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    """
    Download multiple SRA datasets.

    Args:
        fastq_folder: Folder to save downloaded FASTQ files
        accessions_file: File containing SRA accessions, one per line
        max_downloads: Maximum number of datasets to download
        dry_run: If True, only count accessions without downloading
        num_threads: Number of threads for each fasterq-dump
        max_workers: Number of parallel downloads
        force: If True, redownload even if files exist
        max_retries: Maximum number of retry attempts for failed downloads
        temp_folder: Directory to use for fasterq-dump temporary files
        blacklist: One or more files containing accessions to skip

    Returns:
        Dictionary with download statistics

    Raises:
        DataAccessError: If the download fails
    """
    try:
        # Handle the output folder based on dry run status
        fastq_path = _resolve_fastq_path(fastq_folder, dry_run)

        # Read accessions from file
        with open(accessions_file, "r") as f:
            all_accessions = [line.strip() for line in f if line.strip()]

        logger.info(f"Found {len(all_accessions)} accessions in file")

        # Read blacklisted accessions
        blacklisted_accessions = _read_blacklist_files(blacklist)
        if blacklisted_accessions:
            logger.info(f"Found total of {len(blacklisted_accessions)} blacklisted accessions")

        # Check which accessions need downloading
        already_downloaded, accessions_to_download, blacklisted = _check_existing_downloads(
            all_accessions, fastq_path, force, blacklisted_accessions
        )

        logger.info(f"{len(already_downloaded)} accessions already downloaded")
        logger.info(f"{len(blacklisted)} accessions blacklisted")
        logger.info(f"{len(accessions_to_download)} accessions need downloading")

        if dry_run:
            logger.info(f"Dry run: would download {len(accessions_to_download)} accessions")
            return {
                "total": len(all_accessions),
                "already_downloaded": len(already_downloaded),
                "blacklisted": len(blacklisted),
                "to_download": len(accessions_to_download),
                "successful": 0,
                "failed": 0,
            }

        # Limit number of downloads if specified
        if max_downloads is not None and max_downloads < len(accessions_to_download):
            logger.info(f"Limiting to {max_downloads} downloads")
            accessions_to_download = accessions_to_download[:max_downloads]

        # Download accessions in parallel, with an optional retry pass
        successful_count, failed_count, failed_accessions, download_results = _download_with_retries(
            accessions_to_download, fastq_path, num_threads, max_workers, force, temp_folder, max_retries
        )

        # Log final summary
        logger.info("Download summary:")
        logger.info(f"  Total accessions: {len(all_accessions)}")
        logger.info(f"  Already downloaded: {len(already_downloaded)}")
        logger.info(f"  Blacklisted: {len(blacklisted)}")
        logger.info(f"  Newly downloaded: {successful_count}")
        logger.info(f"  Failed downloads: {failed_count}")

        if failed_count > 0:
            logger.warning("Some downloads failed. Use --force to retry or --max-retries to enable " "automatic retry.")
            _handle_download_failure(fastq_path, failed_accessions)

        download_stats = {
            "total": len(all_accessions),
            "already_downloaded": len(already_downloaded),
            "blacklisted": len(blacklisted),
            "successful": successful_count,
            "failed": failed_count,
            "failed_accessions": failed_accessions,
            "results": download_results,
        }

        return download_stats

    except Exception as e:
        raise DataAccessError(f"Downloading SRA data: {e}")


def _find_paired_reads(illumina_files):
    """
    Find pairs of Illumina reads.

    Args:
        illumina_files: List of Illumina fastq files

    Returns:
        List of tuples (R1_file, R2_file)
    """
    read_pairs = []

    for fastq_file in illumina_files:
        if "R1" in fastq_file.name:
            # Find paired read file
            r2_file = fastq_file.with_name(fastq_file.name.replace("R1", "R2"))

            if r2_file.exists():
                read_pairs.append((fastq_file, r2_file))
            else:
                logger.warning(f"Could not find paired read file for {fastq_file}")

    return read_pairs


def assemble_datasets(args):
    """
    Assemble datasets from FASTQ files.

    De-novo assembly is not yet implemented. This function validates the input
    location and then raises ProcessingError, rather than silently producing an
    empty result and writing a misleading success file.

    Args:
        args: Command-line arguments with fastq_folder/data_files and output_file

    Raises:
        DataAccessError: If the input FASTQ folder does not exist
        ProcessingError: Always, because assembly is not implemented
    """
    # Determine source folder - prefer fastq_folder (tests) over data_files (CLI)
    if hasattr(args, "fastq_folder") and args.fastq_folder:
        fastq_folder = Path(args.fastq_folder)
    elif hasattr(args, "data_files") and args.data_files:
        # For CLI, assume the first data_files entry lives in the source folder
        fastq_folder = Path(args.data_files[0]).parent
    else:
        fastq_folder = Path("fastq")

    if not fastq_folder.exists():
        raise DataAccessError(f"Fastq folder {fastq_folder} does not exist")

    raise ProcessingError(
        "Dataset assembly is not implemented. Run a dedicated assembler "
        "(e.g. megahit for Illumina or flye for long reads) on the downloaded "
        "FASTQ files instead."
    )
