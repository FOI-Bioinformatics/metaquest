"""Genome download functionality using NCBI datasets CLI.

This module provides functions for downloading genome assemblies from NCBI
using the datasets command-line tool, extracting FASTA files, and creating
manifests suitable for downstream analysis tools such as sourmash.
"""

import csv
import logging
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from metaquest.core.constants import GENOME_ACCESSION_PATTERN, GENOME_ACCESSION_PREFIXES
from metaquest.core.exceptions import DataAccessError
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


def _validate_genome_accession(accession: str) -> str:
    """Validate a genome assembly accession (GCF_/GCA_ format).

    Args:
        accession: The accession string to validate.

    Returns:
        The validated accession.

    Raises:
        DataAccessError: If the accession format is invalid.
    """
    if not accession.startswith(GENOME_ACCESSION_PREFIXES):
        raise DataAccessError(
            f"Invalid genome accession prefix: {accession}. "
            f"Expected GCF_ or GCA_ prefix."
        )
    if not re.match(GENOME_ACCESSION_PATTERN, accession):
        raise DataAccessError(
            f"Invalid genome accession format: {accession}. "
            f"Expected format: GCF_000000000.0 or GCA_000000000.0"
        )
    return accession


def check_datasets_available() -> bool:
    """Check if NCBI datasets CLI is installed and accessible.

    Returns:
        True if the datasets CLI is available, False otherwise.
    """
    try:
        SecureSubprocess.run_secure("datasets", ["--version"])
        return True
    except Exception:
        return False


def download_genomes(
    accessions: List[str],
    output_dir: Path,
    include: str = "genome",
    assembly_level: Optional[str] = None,
) -> Path:
    """Download genome FASTA files using NCBI datasets CLI.

    Args:
        accessions: List of genome accessions (GCF_/GCA_ format).
        output_dir: Directory for the downloaded zip file.
        include: Data type to include (default: genome).
        assembly_level: Optional assembly level filter (e.g., complete, chromosome).

    Returns:
        Path to the downloaded zip file.

    Raises:
        DataAccessError: If validation or download fails.
    """
    if not accessions:
        raise DataAccessError("No accessions provided for download.")

    for acc in accessions:
        _validate_genome_accession(acc)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "ncbi_dataset.zip"

    args = ["download", "genome", "accession"]
    args.extend(accessions)
    args.extend(["--include", include])
    args.extend(["--filename", str(zip_path)])

    if assembly_level:
        args.extend(["--assembly-level", assembly_level])

    logger.info(
        f"Downloading {len(accessions)} genome(s) from NCBI datasets"
    )

    try:
        SecureSubprocess.run_secure("datasets", args)
    except Exception as e:
        raise DataAccessError(f"Failed to download genomes: {e}")

    if not zip_path.exists():
        raise DataAccessError(
            "Download completed but zip file was not created."
        )

    logger.info(f"Downloaded genomes to {zip_path}")
    return zip_path


def download_from_file(
    accession_file: Path,
    output_dir: Path,
    include: str = "genome",
) -> Path:
    """Download genomes from a text file of accessions (one per line).

    Args:
        accession_file: Path to a file with one accession per line.
        output_dir: Directory for the downloaded zip file.
        include: Data type to include (default: genome).

    Returns:
        Path to the downloaded zip file.

    Raises:
        DataAccessError: If the file is missing or download fails.
    """
    accession_file = Path(accession_file)
    if not accession_file.exists():
        raise DataAccessError(
            f"Accession file not found: {accession_file}"
        )

    # Validate accessions in the file
    accessions = read_accession_file(accession_file)
    if not accessions:
        raise DataAccessError(
            f"No valid accessions found in {accession_file}"
        )
    for acc in accessions:
        _validate_genome_accession(acc)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "ncbi_dataset.zip"

    args = [
        "download", "genome", "accession",
        "--inputfile", str(accession_file),
        "--include", include,
        "--filename", str(zip_path),
    ]

    logger.info(
        f"Downloading genomes from accession file {accession_file}"
    )

    try:
        SecureSubprocess.run_secure("datasets", args)
    except Exception as e:
        raise DataAccessError(f"Failed to download genomes from file: {e}")

    if not zip_path.exists():
        raise DataAccessError(
            "Download completed but zip file was not created."
        )

    logger.info(f"Downloaded genomes to {zip_path}")
    return zip_path


def extract_and_organize(
    zip_path: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    """Extract downloaded zip and organize FASTA files by accession.

    The NCBI datasets zip typically contains files at:
    ncbi_dataset/data/{accession}/{accession}_{assembly}_genomic.fna

    Each FASTA is copied to output_dir/{accession}.fna for convenience.

    Args:
        zip_path: Path to the downloaded NCBI datasets zip file.
        output_dir: Directory to place organized FASTA files.

    Returns:
        Dict mapping accession to the organized FASTA file path.

    Raises:
        DataAccessError: If the zip is invalid or extraction fails.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    if not zip_path.exists():
        raise DataAccessError(f"Zip file not found: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise DataAccessError(f"Not a valid zip file: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    genome_paths: Dict[str, Path] = {}

    extract_dir = zip_path.parent / "ncbi_extract_tmp"

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Find FASTA files in the extracted data
        data_dir = extract_dir / "ncbi_dataset" / "data"
        if not data_dir.exists():
            raise DataAccessError(
                "Unexpected zip structure: ncbi_dataset/data/ not found."
            )

        for accession_dir in data_dir.iterdir():
            if not accession_dir.is_dir():
                continue
            accession = accession_dir.name
            # Skip non-accession directories (e.g., assembly_data_report)
            if not accession.startswith(GENOME_ACCESSION_PREFIXES):
                continue

            # Find FASTA files (.fna or .fasta)
            fasta_files = (
                list(accession_dir.glob("*.fna"))
                + list(accession_dir.glob("*.fasta"))
            )
            if not fasta_files:
                logger.warning(
                    f"No FASTA files found for {accession}"
                )
                continue

            # Use the first (typically only) FASTA file
            source_fasta = fasta_files[0]
            dest_fasta = output_dir / f"{accession}.fna"
            shutil.copy2(str(source_fasta), str(dest_fasta))
            genome_paths[accession] = dest_fasta
            logger.debug(f"Extracted {accession} -> {dest_fasta}")

    except zipfile.BadZipFile as e:
        raise DataAccessError(f"Corrupt zip file {zip_path}: {e}")
    except DataAccessError:
        raise
    except Exception as e:
        raise DataAccessError(f"Error extracting genomes: {e}")
    finally:
        # Clean up temporary extraction directory
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

    logger.info(
        f"Extracted and organized {len(genome_paths)} genome(s)"
    )
    return genome_paths


def read_accession_file(file_path: Path) -> List[str]:
    """Read accessions from a text file (one per line, skip empty/comments).

    Args:
        file_path: Path to the accession file.

    Returns:
        List of accession strings.

    Raises:
        DataAccessError: If the file cannot be read.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise DataAccessError(f"Accession file not found: {file_path}")

    accessions = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    accessions.append(line)
    except Exception as e:
        raise DataAccessError(f"Error reading accession file: {e}")

    return accessions


def create_genome_manifest(
    genome_paths: Dict[str, Path],
    manifest_file: Path,
) -> Path:
    """Create a CSV manifest suitable for sourmash manysketch.

    The output CSV has columns: name, genome_filename, protein_filename

    Args:
        genome_paths: Dict mapping accession to FASTA file path.
        manifest_file: Path to write the manifest CSV.

    Returns:
        Path to the created manifest file.

    Raises:
        DataAccessError: If the manifest cannot be written.
    """
    manifest_file = Path(manifest_file)

    if not genome_paths:
        raise DataAccessError("No genome paths provided for manifest.")

    try:
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "genome_filename", "protein_filename"])
            for accession, fasta_path in sorted(genome_paths.items()):
                writer.writerow([accession, str(fasta_path), ""])

    except Exception as e:
        raise DataAccessError(f"Error writing genome manifest: {e}")

    logger.info(
        f"Created genome manifest with {len(genome_paths)} entries: "
        f"{manifest_file}"
    )
    return manifest_file
