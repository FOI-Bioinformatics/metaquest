
import pandas as pd
from Bio import Entrez
from pathlib import Path
import csv
import logging
from urllib.error import HTTPError
import time
from lxml import etree
from typing import Dict, Union, List, NoReturn
from time import sleep

MAX_RETRIES = 3  # Number of times to retry a failed download

def download_metadata(email: str, matches_folder: Union[str, Path], metadata_folder: Union[str, Path],
                      threshold: float, dry_run: bool = False) -> None:
    """
    Downloads metadata for each unique SRA accession in the given matches folder.

    Parameters:
    - email (str): Your email address for NCBI API access.
    - matches_folder (Union[str, Path]): The path to the folder containing match .csv files.
    - metadata_folder (Union[str, Path]): The path to the folder where metadata will be saved.
    - threshold (float): Threshold for containment values.
    - dry_run (bool): If True, does not download metadata but shows information about what would be downloaded.

    Returns:
    None

    This function scans all CSV files in the matches folder, identifies unique accessions
    based on a containment threshold, and downloads their metadata from NCBI.
    """
    logging.info("Starting metadata download.")

    matches_folder = Path(matches_folder)
    metadata_folder = Path(metadata_folder)
    metadata_folder.mkdir(exist_ok=True)  # Ensure the output directory exists
    Entrez.email = email

    total_unique_accessions = 0
    accessions_to_download = 0
    unique_accessions = set()

    logging.info(f"Scanning matches folder: {matches_folder}")

    for csv_file in matches_folder.glob('*.csv'):
        logging.info(f"Processing file: {csv_file}")
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                accession = row[0]
                containment = float(row[1])
                if containment > threshold:
                    unique_accessions.add(accession)

    total_unique_accessions = len(unique_accessions)

    logging.info(f"Total number of unique accessions: {total_unique_accessions}")

    for accession in unique_accessions:
        metadata_file = metadata_folder / f"{accession}_metadata.xml"
        if not metadata_file.exists():
            accessions_to_download += 1

    if dry_run:
        logging.info("Dry run enabled, not downloading metadata.")
        logging.info(f"Total number of unique accessions: {total_unique_accessions}")
        logging.info(f"Accessions to download: {accessions_to_download}")
        return

    downloaded_accessions = 0
    failed_downloads = 0

    for accession in unique_accessions:
        metadata_file = metadata_folder / f"{accession}_metadata.xml"
        if not metadata_file.exists():
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    logging.info(f"Downloading metadata for {accession}")
                    handle = Entrez.efetch(db="sra", id=accession, retmode="xml")
                    metadata = handle.read().decode()  # Decode bytes to string
                    with open(metadata_file, "w") as out_handle:
                        out_handle.write(metadata)
                    downloaded_accessions += 1
                    if downloaded_accessions % 100 == 0:
                        logging.info(f"Downloaded metadata for {downloaded_accessions} accessions.")
                    break  # Successful download, break the retry loop
                except HTTPError as e:
                    retries += 1
                    logging.warning(f"Failed to download metadata for {accession}. Retrying ({retries}/{MAX_RETRIES}). Error: {e}")
                    sleep(2 ** retries)  # Exponential backoff
            else:  # No break means all retries failed
                logging.error(f"Failed to download metadata for {accession} after {MAX_RETRIES} retries.")
                failed_downloads += 1

    logging.info(f"Total number of unique accessions: {total_unique_accessions}")
    logging.info(f"Accessions to download: {accessions_to_download}")
    logging.info(f"Failed downloads: {failed_downloads}")  # Print the number of failed downloads


def parse_metadata(args):
    """Parse metadata files to produce a consolidated metadata table.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the following attributes:
        * metadata_folder (str): Path to the folder containing metadata files.
        * metadata_table_file (str): Path to save the consolidated metadata table.
        
    This function reads individual metadata files, extracts relevant information,
    and produces a consolidated table. The resulting table is saved to the specified file.
    """
    metadata_folder = Path(args.metadata_folder)
    metadata_table = []
    start_time = time.time()  # Start time of parsing operation
    # Initialize a counter for parsed files
    parsed_files_count = 0
    for metadata_file in metadata_folder.glob('*.xml'):
        tree = etree.parse(str(metadata_file))

        # Project Information
        project_id = tree.findtext(".//STUDY/IDENTIFIERS/PRIMARY_ID")
        project_title = tree.findtext(".//STUDY/DESCRIPTOR/STUDY_TITLE")
        project_abstract = tree.findtext(".//STUDY/DESCRIPTOR/STUDY_ABSTRACT")

        # Sample Information
        sample_id = tree.findtext(".//SAMPLE/IDENTIFIERS/PRIMARY_ID")
        sample_external_id = tree.findtext(".//SAMPLE/IDENTIFIERS/EXTERNAL_ID")
        sample_name = tree.findtext(".//SAMPLE/SAMPLE_NAME/TAXON_ID")
        sample_scientific_name = tree.findtext(".//SAMPLE/SAMPLE_NAME/SCIENTIFIC_NAME")
        sample_title = tree.findtext(".//SAMPLE/TITLE")


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


        # Experiment Information
        experiment_id = tree.findtext(".//EXPERIMENT/IDENTIFIERS/PRIMARY_ID")
        experiment_title = tree.findtext(".//EXPERIMENT/TITLE")
        experiment_design = tree.findtext(".//EXPERIMENT/DESIGN/DESIGN_DESCRIPTION")
        experiment_library_name = tree.findtext(".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_NAME")
        experiment_library_strategy = tree.findtext(".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_STRATEGY")
        experiment_library_source = tree.findtext(".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_SOURCE")
        experiment_library_selection = tree.findtext(".//EXPERIMENT/LIBRARY_DESCRIPTOR/LIBRARY_SELECTION")

        # Extract the second SRAFile URL (SRA Normalized URL)
        srafile_elements = tree.findall(".//RUN/SRAFiles/SRAFile")
        sra_normalized_url = None
        if len(srafile_elements) > 1:
            sra_normalized_url = srafile_elements[1].get("url")


        # Add to metadata table
        metadata_table.append({
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
            "SRA_Normalized_URL": sra_normalized_url
        })

        parsed_files_count += 1  # Increment the counter each time a file is parsed

    # Convert to DataFrame and save as a .txt file
    metadata_df = pd.DataFrame(metadata_table)
    metadata_df.to_csv(args.metadata_table_file, sep="\t", index=False)

    end_time = time.time()  # End time of parsing operation
    elapsed_time = end_time - start_time  # Elapsed time
    print(f'Parsed {parsed_files_count} files in {elapsed_time} seconds.')



