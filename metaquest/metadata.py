
import os
import pandas as pd
from Bio import Entrez


def download_metadata(args):
    matches_folder = Path(args.matches_folder)
    metadata_folder = Path(args.metadata_folder)
    metadata_folder.mkdir(exist_ok=True)

    Entrez.email = args.email

    total_unique_accessions = 0
    accessions_to_download = 0
    unique_accessions = set()

    for csv_file in matches_folder.glob('*.csv'):
        print(f"Processing file: {csv_file}")
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                accession = row[0]
                containment = float(row[1])
                if containment > args.threshold:
                    unique_accessions.add(accession)

    total_unique_accessions = len(unique_accessions)

    for accession in unique_accessions:
        metadata_file = metadata_folder / f"{accession}_metadata.xml"
        if not metadata_file.exists():
            accessions_to_download += 1

    if args.dry_run:
        print("Dry run enabled, not downloading metadata.")
        print(f"Total number of unique accessions: {total_unique_accessions}")
        print(f"Accessions to download: {accessions_to_download}")
        return

    downloaded_accessions = 0 # Counter for downloaded accessions
    failed_downloads = 0  # Counter for failed downloads

    for accession in unique_accessions:
        metadata_file = metadata_folder / f"{accession}_metadata.xml"
        if not metadata_file.exists():
            try:
                logging.info(f'Downloading metadata for {accession}')
                handle = Entrez.efetch(db="sra", id=accession, retmode="xml")
                metadata = handle.read().decode()  # Decode bytes to string
                with open(metadata_file, "w") as out_handle:
                    out_handle.write(metadata)
                downloaded_accessions += 1
                if downloaded_accessions % 100 == 0:
                    print(f"Downloaded metadata for {downloaded_accessions} accessions.")
            except HTTPError:
                logging.error(f"Failed to download metadata for {accession}. Skipping to next accession.")
                failed_downloads += 1  # Increment the counter for each failed download

    print(f"Total number of unique accessions: {total_unique_accessions}")
    print(f"Accessions to download: {accessions_to_download}")
    print(f"Failed downloads: {failed_downloads}")  # Print the number of failed downloads





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
            "Experiment_Library_Selection": experiment_library_selection
        })

        parsed_files_count += 1  # Increment the counter each time a file is parsed

    # Convert to DataFrame and save as a .txt file
    metadata_df = pd.DataFrame(metadata_table)
    metadata_df.to_csv(args.metadata_table_file, sep="\t", index=False)

    end_time = time.time()  # End time of parsing operation
    elapsed_time = end_time - start_time  # Elapsed time
    print(f'Parsed {parsed_files_count} files in {elapsed_time} seconds.')



