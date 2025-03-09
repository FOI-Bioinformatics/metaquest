"""
SRA data handling for MetaQuest.

This module provides functions for downloading and processing SRA data.
"""

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.core.validation import validate_folder
from metaquest.data.file_io import ensure_directory

logger = logging.getLogger(__name__)


def download_accession(
    accession: str, output_folder: Union[str, Path], num_threads: int = 4
) -> bool:
    """
    Download a single SRA accession using fasterq-dump.

    Args:
        accession: SRA accession to download
        output_folder: Folder to save the downloaded files
        num_threads: Number of threads to use for download

    Returns:
        True if download successful, False otherwise
    """
    output_path = Path(output_folder) / accession

    # Skip if the output folder already exists
    if output_path.exists():
        logger.info(f"Skipping {accession}, folder already exists")
        return False

    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading SRA for {accession}")

        # Run fasterq-dump command
        subprocess.run(
            [
                "fasterq-dump",
                "--threads",
                str(num_threads),
                "--progress",
                accession,
                "-O",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(f"Successfully downloaded {accession}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading {accession}: {e.stderr}")
        return False

    except Exception as e:
        logger.error(f"Error downloading {accession}: {e}")
        return False


def download_sra(
    fastq_folder: Union[str, Path],
    accessions_file: Union[str, Path],
    max_downloads: Optional[int] = None,
    dry_run: bool = False,
    num_threads: int = 4,
    max_workers: int = 4,
) -> int:
    """
    Download multiple SRA datasets.

    Args:
        fastq_folder: Folder to save downloaded FASTQ files
        accessions_file: File containing SRA accessions, one per line
        max_downloads: Maximum number of datasets to download
        dry_run: If True, only count accessions without downloading
        num_threads: Number of threads for each fasterq-dump
        max_workers: Number of parallel downloads

    Returns:
        Number of successfully downloaded datasets

    Raises:
        DataAccessError: If the download fails
    """
    try:
        # Ensure output folder exists
        fastq_path = ensure_directory(fastq_folder)

        # Read accessions from file
        with open(accessions_file, "r") as f:
            all_accessions = [line.strip() for line in f if line.strip()]

        logger.info(f"Found {len(all_accessions)} accessions in file")

        # Filter out already downloaded accessions
        accessions_to_download = [
            acc for acc in all_accessions if not (fastq_path / acc).exists()
        ]

        logger.info(f"{len(accessions_to_download)} accessions need downloading")

        if dry_run:
            logger.info(
                f"Dry run: would download {len(accessions_to_download)} accessions"
            )
            return len(accessions_to_download)

        # Limit number of downloads if specified
        if max_downloads is not None and max_downloads < len(accessions_to_download):
            logger.info(f"Limiting to {max_downloads} downloads")
            accessions_to_download = accessions_to_download[:max_downloads]

        # Download accessions in parallel
        downloaded_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download jobs
            future_to_accession = {
                executor.submit(download_accession, acc, fastq_path, num_threads): acc
                for acc in accessions_to_download
            }

            # Process results as they complete
            for future in as_completed(future_to_accession):
                accession = future_to_accession[future]

                try:
                    success = future.result()
                    if success:
                        downloaded_count += 1

                        # Log progress periodically
                        if downloaded_count % 5 == 0:
                            logger.info(
                                f"Downloaded {downloaded_count}/{len(accessions_to_download)}"
                            )

                except Exception as e:
                    logger.error(
                        f"Error processing download result for {accession}: {e}"
                    )

        logger.info(f"Successfully downloaded {downloaded_count} datasets")
        return downloaded_count

    except Exception as e:
        raise DataAccessError(f"Error downloading SRA data: {e}")


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

        # Process Illumina and Nanopore datasets
        illumina_files = []
        nanopore_files = []

        for fastq_file in fastq_folder.glob("*.fastq.gz"):
            if "R1" in fastq_file.name or "R2" in fastq_file.name:
                illumina_files.append(fastq_file)
            else:
                nanopore_files.append(fastq_file)

        logger.info(
            f"Found {len(illumina_files)} Illumina files and {len(nanopore_files)} Nanopore files"
        )

        # Process Illumina datasets
        for fastq_file in fastq_folder.glob("*.fastq.gz"):
            if "R1" in fastq_file.name:
                # Find paired read file
                r2_file = fastq_file.with_name(fastq_file.name.replace("R1", "R2"))

                if not r2_file.exists():
                    logger.warning(f"Could not find paired read file for {fastq_file}")
                    continue

                output_dir = f"{fastq_file.stem}_megahit"

                if Path(output_dir).exists():
                    logger.info(
                        f"Skipping {fastq_file.name}, output directory already exists"
                    )
                    continue

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

        # Process Nanopore datasets
        for fastq_file in nanopore_files:
            output_dir = f"{fastq_file.stem}_flye"

            if Path(output_dir).exists():
                logger.info(
                    f"Skipping {fastq_file.name}, output directory already exists"
                )
                continue

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

    except Exception as e:
        raise DataAccessError(f"Error assembling datasets: {e}")
