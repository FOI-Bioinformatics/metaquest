"""
Metadata handling for MetaQuest.

This module provides functions for downloading and processing metadata from NCBI.
"""

import logging
from pathlib import Path
import time
import pandas as pd
from typing import Dict, List, Optional, Set, Union

from Bio import Entrez
from lxml import etree
from urllib.error import HTTPError

from metaquest.core.exceptions import DataAccessError
from metaquest.core.validation import validate_folder
from metaquest.data.file_io import ensure_directory, list_files, write_csv

logger = logging.getLogger(__name__)

# Maximum number of retries for failed downloads
MAX_RETRIES = 3


def download_metadata(
    email: str,
    matches_folder: Union[str, Path],
    metadata_folder: Union[str, Path],
    threshold: float = 0.0,
    dry_run: bool = False,
) -> Dict[str, Path]:
    """
    Download metadata for SRA accessions found in match files.

    Args:
        email: Email address for NCBI API
        matches_folder: Folder containing match files
        metadata_folder: Folder to save metadata files
        threshold: Minimum containment threshold
        dry_run: If True, only count accessions without downloading

    Returns:
        Dictionary mapping accessions to metadata file paths

    Raises:
        DataAccessError: If the download fails
    """
    matches_path = validate_folder(matches_folder)
    metadata_path = ensure_directory(metadata_folder)

    # Set email for NCBI API
    Entrez.email = email

    # Get unique accessions from match files
    unique_accessions = set()

    try:
        # Find all CSV files in matches folder
        csv_files = list_files(matches_path, "*.csv")

        if not csv_files:
            logger.warning(f"No CSV files found in {matches_path}")
            return {}

        for csv_file in csv_files:
            # Read CSV file
            try:
                df = pd.read_csv(csv_file)

                # Determine accession column name based on format
                if "acc" in df.columns:
                    accession_col = "acc"
                    containment_col = "containment"
                elif "SRA accession" in df.columns:
                    accession_col = "SRA accession"
                    containment_col = "containment"
                else:
                    logger.warning(f"Unknown file format: {csv_file}")
                    continue

                # Filter by threshold
                if threshold > 0:
                    df = df[df[containment_col] > threshold]

                # Add accessions to set
                unique_accessions.update(df[accession_col].tolist())

            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")

        total_accessions = len(unique_accessions)
        logger.info(f"Found {total_accessions} unique accessions")

        # Check which accessions need downloading
        accessions_to_download = []
        for accession in unique_accessions:
            metadata_file = metadata_path / f"{accession}_metadata.xml"
            if not metadata_file.exists():
                accessions_to_download.append(accession)

        to_download_count = len(accessions_to_download)
        logger.info(f"Need to download {to_download_count} accessions")

        if dry_run:
            logger.info("Dry run, not downloading metadata")
            return {}

        # Download metadata for each accession
        result_files = {}
        downloaded_count = 0
        failed_count = 0

        for accession in accessions_to_download:
            metadata_file = metadata_path / f"{accession}_metadata.xml"

            retries = 0
            success = False

            while retries < MAX_RETRIES and not success:
                try:
                    logger.info(f"Downloading metadata for {accession}")

                    # Fetch metadata from NCBI
                    handle = Entrez.efetch(db="sra", id=accession, retmode="xml")
                    metadata = handle.read().decode()

                    # Save metadata to file
                    with open(metadata_file, "w") as f:
                        f.write(metadata)

                    result_files[accession] = metadata_file
                    downloaded_count += 1
                    success = True

                    # Log progress periodically
                    if downloaded_count % 10 == 0:
                        logger.info(
                            f"Downloaded {downloaded_count}/{to_download_count}"
                        )

                    # Be nice to NCBI servers
                    time.sleep(0.5)

                except HTTPError as e:
                    retries += 1
                    logger.warning(
                        f"Error downloading {accession}, retrying ({retries}/{MAX_RETRIES}): {e}"
                    )
                    time.sleep(2**retries)  # Exponential backoff

                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"Error downloading {accession}, retrying ({retries}/{MAX_RETRIES}): {e}"
                    )
                    time.sleep(2**retries)

            if not success:
                failed_count += 1
                logger.error(
                    f"Failed to download {accession} after {MAX_RETRIES} retries"
                )

        logger.info(
            f"Downloaded {downloaded_count} metadata files with {failed_count} failures"
        )
        return result_files

    except Exception as e:
        raise DataAccessError(f"Error downloading metadata: {e}")


def parse_metadata(
    metadata_folder: Union[str, Path], output_file: Union[str, Path]
) -> pd.DataFrame:
    """
    Parse metadata XML files and create a consolidated table.

    Args:
        metadata_folder: Folder containing metadata XML files
        output_file: Path to save the consolidated metadata table

    Returns:
        DataFrame containing the parsed metadata

    Raises:
        DataAccessError: If parsing fails
    """
    metadata_path = validate_folder(metadata_folder)

    metadata_records = []
    processed_count = 0
    error_count = 0

    try:
        # Get unique sample attributes
        unique_attributes = get_unique_sample_attributes(metadata_path)
        logger.info(f"Found {len(unique_attributes)} unique sample attributes")

        # Process each XML file
        xml_files = list_files(metadata_path, "*.xml")

        if not xml_files:
            logger.warning(f"No XML files found in {metadata_path}")
            return pd.DataFrame()

        logger.info(f"Processing {len(xml_files)} metadata files")

        for xml_file in xml_files:
            try:
                tree = etree.parse(str(xml_file))

                # Extract project information
                project_id = tree.findtext(".//STUDY/IDENTIFIERS/PRIMARY_ID")
                project_title = tree.findtext(".//STUDY/DESCRIPTOR/STUDY_TITLE")
                project_abstract = tree.findtext(".//STUDY/DESCRIPTOR/STUDY_ABSTRACT")

                # Extract sample information
                sample_id = tree.findtext(".//SAMPLE/IDENTIFIERS/PRIMARY_ID")
                sample_external_id = tree.findtext(".//SAMPLE/IDENTIFIERS/EXTERNAL_ID")
                sample_name = tree.findtext(".//SAMPLE/SAMPLE_NAME/TAXON_ID")
                sample_scientific_name = tree.findtext(
                    ".//SAMPLE/SAMPLE_NAME/SCIENTIFIC_NAME"
                )
                sample_title = tree.findtext(".//SAMPLE/TITLE")

                # Extract run information
                run_id = tree.findtext(".//RUN/IDENTIFIERS/PRIMARY_ID")
                run_total_spots = tree.findtext(".//RUN/Total_spots")
                run_total_bases = tree.findtext(".//RUN/Total_bases")
                run_size = tree.findtext(".//RUN/size")
                run_download_path = tree.findtext(".//RUN/download_path")
                run_md5 = tree.findtext(".//RUN/md5")
                run_filename = tree.findtext(".//RUN/filename")
                run_spot_length = tree.findtext(".//RUN/spot_length")
                run_reads = tree.findtext(".//RUN/reads")
                run_ftp = tree.findtext(".//RUN/ftp")
                run_aspera = tree.findtext(".//RUN/aspera")
                run_galaxy = tree.findtext(".//RUN/galaxy")

                # Extract experiment information
                experiment_id = tree.findtext(".//EXPERIMENT/IDENTIFIERS/PRIMARY_ID")
                experiment_title = tree.findtext(".//EXPERIMENT/TITLE")
                experiment_design = tree.findtext(
                    ".//EXPERIMENT/DESIGN/DESIGN_DESCRIPTION"
                )
                experiment_library_name = tree.findtext(
                    ".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_NAME"
                )
                experiment_library_strategy = tree.findtext(
                    ".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_STRATEGY"
                )
                experiment_library_source = tree.findtext(
                    ".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_SOURCE"
                )
                experiment_library_selection = tree.findtext(
                    ".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_SELECTION"
                )

                # Extract SRA URL
                srafile_elements = tree.findall(".//RUN/SRAFiles/SRAFile")
                sra_normalized_url = None
                if len(srafile_elements) > 1:
                    sra_normalized_url = srafile_elements[1].get("url")

                # Extract sample attributes
                sample_attributes = {}
                for attribute in tree.findall(".//SAMPLE_ATTRIBUTES/SAMPLE_ATTRIBUTE"):
                    tag = attribute.findtext("TAG")
                    value = attribute.findtext("VALUE")
                    if tag in unique_attributes:
                        sample_attributes[tag] = value

                # Create metadata record
                metadata_dict = {
                    "Run_ID": run_id,
                    "Run_Total_Spots": run_total_spots,
                    "Run_Total_Bases": run_total_bases,
                    "Run_Size": run_size,
                    "Run_Download_Path": run_download_path,
                    "Run_MD5": run_md5,
                    "Run_Filename": run_filename,
                    "Run_Spot_Length": run_spot_length,
                    "Run_Reads": run_reads,
                    "Run_FTP": run_ftp,
                    "Run_Aspera": run_aspera,
                    "Run_Galaxy": run_galaxy,
                    "Project_ID": project_id,
                    "Project_Title": project_title,
                    "Project_Abstract": project_abstract,
                    "Sample_ID": sample_id,
                    "Sample_External_ID": sample_external_id,
                    "Sample_Name": sample_name,
                    "Sample_Scientific_Name": sample_scientific_name,
                    "Sample_Title": sample_title,
                    "Experiment_ID": experiment_id,
                    "Experiment_Title": experiment_title,
                    "Experiment_Design": experiment_design,
                    "Experiment_Library_Name": experiment_library_name,
                    "Experiment_Library_Strategy": experiment_library_strategy,
                    "Experiment_Library_Source": experiment_library_source,
                    "Experiment_Library_Selection": experiment_library_selection,
                    "SRA_Normalized_URL": sra_normalized_url,
                }

                # Add sample attributes
                for attribute in unique_attributes:
                    metadata_dict[attribute] = sample_attributes.get(attribute, None)

                metadata_records.append(metadata_dict)
                processed_count += 1

                # Log progress periodically
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} metadata files")

            except Exception as e:
                error_count += 1
                logger.error(f"Error parsing {xml_file}: {e}")

        # Create DataFrame
        if not metadata_records:
            logger.warning("No metadata records created")
            return pd.DataFrame()

        metadata_df = pd.DataFrame(metadata_records)

        # Save to file
        write_csv(metadata_df, output_file, sep="\t", index=False)
        logger.info(
            f"Saved metadata table with {len(metadata_df)} records to {output_file}"
        )

        return metadata_df

    except Exception as e:
        raise DataAccessError(f"Error parsing metadata: {e}")


def get_unique_sample_attributes(metadata_folder: Union[str, Path]) -> List[str]:
    """
    Get unique sample attribute tags from all metadata files.

    Args:
        metadata_folder: Folder containing metadata XML files

    Returns:
        List of unique sample attribute tags
    """
    unique_attributes = set()

    try:
        # Find all XML files
        xml_files = list_files(Path(metadata_folder), "*.xml")

        for xml_file in xml_files:
            try:
                tree = etree.parse(str(xml_file))

                # Extract attribute tags
                for attribute in tree.findall(
                    ".//SAMPLE_ATTRIBUTES/SAMPLE_ATTRIBUTE/TAG"
                ):
                    unique_attributes.add(attribute.text)

            except Exception as e:
                logger.warning(f"Error reading attributes from {xml_file}: {e}")

        return sorted(list(unique_attributes))

    except Exception as e:
        logger.warning(f"Error getting unique sample attributes: {e}")
        return []


def check_metadata_attributes(
    file_path: Union[str, Path], output_file: Union[str, Path]
) -> Dict[str, int]:
    """
    Count occurrences of metadata attributes and save to file.

    Args:
        file_path: Path to the metadata table file
        output_file: Path to save the attribute counts

    Returns:
        Dictionary mapping attribute names to counts

    Raises:
        DataAccessError: If the operation fails
    """
    try:
        # Read metadata table
        df = pd.read_csv(file_path, sep="\t")

        # Filter columns to exclude standard categories
        excluded_prefixes = ("Run_", "Project_", "Sample_", "Experiment_")
        filtered_columns = [
            col for col in df.columns if not col.startswith(excluded_prefixes)
        ]

        # Count non-null values for each column
        counts = {col: df[col].count() for col in filtered_columns}

        # Sort by count
        sorted_counts = {
            k: v
            for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        }

        # Save to file
        with open(output_file, "w") as f:
            for key, value in sorted_counts.items():
                f.write(f"{key}\t{value}\n")

        logger.info(f"Saved attribute counts to {output_file}")
        return sorted_counts

    except Exception as e:
        raise DataAccessError(f"Error checking metadata attributes: {e}")
