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
from typing import Any, Dict, Optional, Tuple, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.data.file_io import ensure_directory

logger = logging.getLogger(__name__)


def _prepare_temp_folder(temp_folder, temp_cmd):
    """
    Prepare the temporary folder for fasterq-dump.

    Args:
        temp_folder: Path to temporary folder
        temp_cmd: List to store temp folder command

    Returns:
        Updated temp_cmd list
    """
    if not temp_folder:
        return temp_cmd

    # Ensure temp folder exists
    temp_path_obj = Path(temp_folder)
    try:
        temp_path_obj.mkdir(parents=True, exist_ok=True)
        if not os.access(temp_path_obj, os.W_OK):
            logger.warning(
                f"Temp folder {temp_folder} exists but is not writable, "
                "using default temp location"
            )
        else:
            temp_cmd = ["--temp", str(temp_path_obj.absolute())]
            logger.info(f"Using temp folder: {temp_path_obj.absolute()}")
    except Exception as e:
        logger.warning(
            f"Could not create or access temp folder {temp_folder}: {e}, "
            "using default temp location"
        )

    return temp_cmd


def _check_existing_download(output_path, force):
    """
    Check if the accession is already downloaded.

    Args:
        output_path: Path to output directory
        force: Whether to force redownload

    Returns:
        True if already downloaded, False otherwise
    """
    if not force and output_path.exists():
        fastq_files = list(output_path.glob("*.fastq*"))
        if fastq_files:
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
        temp_path.rmdir()
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
        return True, "Already downloaded"

    # Create a temporary folder for download
    temp_path = Path(output_folder) / f"{accession}_temp"
    if temp_path.exists():
        # Clean up any existing temporary folder
        try:
            shutil.rmtree(temp_path)
        except Exception as e:
            logger.warning(f"Could not remove existing temp directory {temp_path}: {e}")

    # Create the temporary folder
    temp_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading SRA for {accession}")

        # Handle temp folder for fasterq-dump
        temp_cmd = []
        temp_cmd = _prepare_temp_folder(temp_folder, temp_cmd)

        # Build fasterq-dump command
        cmd = [
            "fasterq-dump",
            "--threads",
            str(num_threads),
            "--progress",
            accession,
            "-O",
            str(temp_path),
        ]

        # Add temp folder if available
        if temp_cmd:
            cmd.extend(temp_cmd)

        # Run fasterq-dump command
        logger.debug(f"Running command: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        # Handle download output
        return _handle_download_output(temp_path, output_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading {accession}: {e.stderr}")
        # Clean up temp directory
        shutil.rmtree(temp_path)
        return False, f"Download failed: {e.stderr}"

    except Exception as e:
        logger.error(f"Error downloading {accession}: {e}")
        # Clean up temp directory
        shutil.rmtree(temp_path)
        return False, f"Download failed: {str(e)}"


def _check_existing_downloads(accessions, fastq_path, force):
    """
    Check which accessions need downloading and which are already downloaded.

    Args:
        accessions: List of accessions
        fastq_path: Path to FASTQ directory
        force: Whether to force redownload

    Returns:
        Tuple of (already_downloaded, to_download)
    """
    already_downloaded = []
    to_download = []

    for acc in accessions:
        acc_path = fastq_path / acc
        if not force and acc_path.exists():
            fastq_files = list(acc_path.glob("*.fastq*"))
            if fastq_files:
                already_downloaded.append(acc)
            else:
                # Folder exists but no FASTQ files, needs download
                to_download.append(acc)
        else:
            to_download.append(acc)

    return already_downloaded, to_download


def _process_download_results(
    futures_results, accessions_to_download, download_results, failed_accessions
):
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
                        f"Downloaded {successful_count}/{len(accessions_to_download)} "
                        f"({failed_count} failed)"
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
                    logger.info(
                        f"Successfully downloaded {accession} on retry {retry + 1}"
                    )
                else:
                    failed_accessions.append(accession)
                    logger.warning(
                        f"Failed to download {accession} on retry {retry + 1}: {message}"
                    )

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

    Returns:
        Dictionary with download statistics

    Raises:
        DataAccessError: If the download fails
    """
    try:
        # Handle the output folder based on dry run status
        fastq_path = Path(fastq_folder)

        if dry_run:
            # In dry-run mode, don't create folders
            if fastq_path.exists() and not fastq_path.is_dir():
                raise DataAccessError(f"{fastq_folder} exists but is not a directory")
            logger.info(f"Dry run mode: Would use {fastq_path} for downloads")
        else:
            # Normal mode - ensure the directory exists
            fastq_path = ensure_directory(fastq_folder)

        # Read accessions from file
        with open(accessions_file, "r") as f:
            all_accessions = [line.strip() for line in f if line.strip()]

        logger.info(f"Found {len(all_accessions)} accessions in file")

        # Check which accessions need downloading
        already_downloaded, accessions_to_download = _check_existing_downloads(
            all_accessions, fastq_path, force
        )

        logger.info(f"{len(already_downloaded)} accessions already downloaded")
        logger.info(f"{len(accessions_to_download)} accessions need downloading")

        if dry_run:
            logger.info(
                f"Dry run: would download {len(accessions_to_download)} accessions"
            )
            return {
                "total": len(all_accessions),
                "already_downloaded": len(already_downloaded),
                "to_download": len(accessions_to_download),
                "successful": 0,
                "failed": 0,
            }

        # Limit number of downloads if specified
        if max_downloads is not None and max_downloads < len(accessions_to_download):
            logger.info(f"Limiting to {max_downloads} downloads")
            accessions_to_download = accessions_to_download[:max_downloads]

        # Download accessions in parallel
        successful_count = 0
        failed_count = 0
        failed_accessions = []
        download_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download jobs
            futures = {
                executor.submit(
                    download_accession, acc, fastq_path, num_threads, force, temp_folder
                ): acc
                for acc in accessions_to_download
            }

            # Process results as they complete
            futures_results = [
                (futures[future], future.result()) for future in as_completed(futures)
            ]

            successful_count, failed_count = _process_download_results(
                futures_results,
                accessions_to_download,
                download_results,
                failed_accessions,
            )

        # Retry failed downloads if requested
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
                logger.info(
                    f"Successfully downloaded {retried_successful} accessions on retry"
                )

        # Log final summary
        logger.info("Download summary:")
        logger.info(f"  Total accessions: {len(all_accessions)}")
        logger.info(f"  Already downloaded: {len(already_downloaded)}")
        logger.info(f"  Newly downloaded: {successful_count}")
        logger.info(f"  Failed downloads: {failed_count}")

        if failed_count > 0:
            logger.warning(
                "Some downloads failed. Use --force to retry or --max-retries to enable "
                "automatic retry."
            )
            _handle_download_failure(fastq_path, failed_accessions)

        download_stats = {
            "total": len(all_accessions),
            "already_downloaded": len(already_downloaded),
            "successful": successful_count,
            "failed": failed_count,
            "failed_accessions": failed_accessions,
            "results": download_results,
        }

        return download_stats

    except Exception as e:
        raise DataAccessError(f"Downloading SRA data: {e}")


def _process_illumina_dataset(fastq_file, r2_file):
    """
    Process a single Illumina paired-end dataset.

    Args:
        fastq_file: Path to R1 fastq file
        r2_file: Path to R2 fastq file
    """
    output_dir = f"{fastq_file.stem}_megahit"

    if Path(output_dir).exists():
        logger.info(f"Skipping {fastq_file.name}, output directory already exists")
        return

    logger.info(f"Assembling Illumina dataset for {fastq_file.name}")

    # Run megahit for Illumina paired-end data
    try:
        subprocess.run(
            [
                "megahit",
                "-1",
                str(fastq_file),
                "-2",
                str(r2_file),
                "-o",
                output_dir,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully assembled {fastq_file.name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error assembling {fastq_file.name}: {e.stderr}")

    except Exception as e:
        logger.error(f"Error assembling {fastq_file.name}: {e}")


def _process_nanopore_dataset(fastq_file):
    """
    Process a single Nanopore dataset.

    Args:
        fastq_file: Path to Nanopore fastq file
    """
    output_dir = f"{fastq_file.stem}_flye"

    if Path(output_dir).exists():
        logger.info(f"Skipping {fastq_file.name}, output directory already exists")
        return

    logger.info(f"Assembling Nanopore dataset for {fastq_file.name}")

    # Run flye for Nanopore data
    try:
        subprocess.run(
            [
                "flye",
                "--nano-raw",
                str(fastq_file),
                "--out-dir",
                output_dir,
                "--meta",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully assembled {fastq_file.name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error assembling {fastq_file.name}: {e.stderr}")

    except Exception as e:
        logger.error(f"Error assembling {fastq_file.name}: {e}")


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

    Args:
        args: Command-line arguments

    Raises:
        DataAccessError: If the assembly fails
    """
    try:
        fastq_folder = Path("fastq")

        if not fastq_folder.exists():
            raise DataAccessError(f"Fastq folder {fastq_folder} does not exist")

        # Identify Illumina and Nanopore files
        illumina_files = []
        nanopore_files = []

        for fastq_file in fastq_folder.glob("*.fastq.gz"):
            if "R1" in fastq_file.name or "R2" in fastq_file.name:
                illumina_files.append(fastq_file)
            else:
                nanopore_files.append(fastq_file)

        logger.info(
            f"Found {len(illumina_files)} Illumina files and {len(nanopore_files)} "
            "Nanopore files"
        )

        # Process Illumina datasets
        read_pairs = _find_paired_reads(illumina_files)

        for r1_file, r2_file in read_pairs:
            _process_illumina_dataset(r1_file, r2_file)

        # Process Nanopore datasets
        for fastq_file in nanopore_files:
            _process_nanopore_dataset(fastq_file)

    except Exception as e:
        raise DataAccessError(f"Error assembling datasets: {e}")
