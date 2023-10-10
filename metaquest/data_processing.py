import os
import pandas as pd
from pathlib import Path
import csv
import logging
import subprocess
import urllib.request
import gzip
from collections import defaultdict
from typing import Dict, Union, List, NoReturn, Set



def download_test_genome(output_folder: Union[str, Path]) -> None:
    logging.info("Starting the download of the test genome.")

    output_folder = Path(output_folder)
    output_path = output_folder / "GCF_000008985.1.fasta"

    # Check if the file already exists
    if output_path.exists():
        logging.info(f"Genome file already exists at {output_path}. Skipping download.")
        return

    output_folder.mkdir(exist_ok=True)  # Ensure the output directory exists
    logging.info(f"Output folder is set to {output_folder}")

    url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/008/985/GCF_000008985.1_ASM898v1/GCF_000008985.1_ASM898v1_genomic.fna.gz"

    # Define the local gz file path
    local_gz_file = output_folder / "temp_genome.fna.gz"

    # Download the compressed fasta file
    logging.info("Starting the download of the compressed fasta file.")
    urllib.request.urlretrieve(url, local_gz_file)
    logging.info("Successfully downloaded the compressed fasta file.")

    # Decompress the fasta file and save it to the output folder
    logging.info("Starting the decompression of the fasta file.")
    with gzip.open(local_gz_file, 'rt') as fh:  # Open the gzip file in text mode for reading
        genome_data = fh.read()
    logging.info("Successfully decompressed the fasta file.")

    with open(output_path, 'w') as out:
        out.write(genome_data)
    logging.info(f"Saved the decompressed fasta file to {output_path}")

    # Optionally, remove the temporary compressed file
    local_gz_file.unlink()
    logging.info("Removed the temporary compressed file.")

    logging.info(f"Downloaded and saved test genome to {output_path}")

# Function 'run_mastiff' runs the 'mastiff' command on each '.fasta' file in the 'genomes_folder'.

def run_mastiff(genomes_folder: Union[str, Path], matches_folder: Union[str, Path]) -> None:
    """
    Run mastiff on all fasta files in a given folder and save the matches in another folder.

    Parameters:
    - genomes_folder (Union[str, Path]): The path to the folder containing the genome files.
    - matches_folder (Union[str, Path]): The path to the folder where match results will be saved.

    Returns:
    None
    """
    logging.info("Starting mastiff on genomes.")

    genomes_folder = Path(genomes_folder)
    matches_folder = Path(matches_folder)
    matches_folder.mkdir(exist_ok=True)  # Ensure the output directory exists

    logging.info(f"Genomes folder is set to {genomes_folder}")
    logging.info(f"Matches folder is set to {matches_folder}")

    for fasta_file in genomes_folder.glob('*.fasta'):
        output_file = matches_folder / f'{fasta_file.stem}_matches.csv'

        # Check if the output file already exists
        if not output_file.exists():
            logging.info(f'Running mastiff on {fasta_file.name}')
            subprocess.run(['mastiff', str(fasta_file), '-o', str(output_file)], check=True)
            logging.info(f'Successfully ran mastiff on {fasta_file.name}')
        else:
            logging.info(f'Skipping mastiff on {fasta_file.name} because output file already exists')


def parse_containment(matches_folder: Union[str, Path], parsed_containment_file: Union[str, Path],
              summary_containment_file: Union[str, Path], step_size=0.1) -> None:
    """
    Generate summary and containment files from MASH matches.

    Parameters:
    - matches_folder (Union[str, Path]): The path to the folder containing _matches.csv files from MASH.
    - summary_file (Union[str, Path]): The path to save the output summary file.
    - containment_file (Union[str, Path]): The path to save the output containment file.

    Returns:
    None

    The function aggregates information from the MASH matches, computes the containment score,
    and saves the summary and containment data in the respective output files.
    """
    logging.info("Starting to summarize MASH matches.")

    matches_folder = Path(matches_folder)
    summary: Dict[str, Dict[str, float]] = defaultdict(dict)

    logging.info(f"Scanning matches folder: {matches_folder}")

    for csv_file in matches_folder.glob('*.csv'):
        file_id = csv_file.stem.replace("_matches", "")
        logging.info(f"Processing file: {file_id}")

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                accession = row['SRA accession']
                containment = float(row['containment'])

                if file_id not in summary[accession]:
                    summary[accession][file_id] = containment
                else:
                    summary[accession][file_id] = max(summary[accession][file_id], containment)

    logging.info("Completed processing of all match files.")

    # Create DataFrame from summary dict
    summary_df = pd.DataFrame.from_dict(summary, orient='index')

    # Fill NA/NaN values with 0
    summary_df.fillna(0, inplace=True)

    # Add 'max_containment' column by finding the max value across each row
    summary_df['max_containment'] = summary_df.max(axis=1)

    # Add 'max_containment_annotation' column with the csv file with highest containment
    summary_df['max_containment_annotation'] = summary_df.idxmax(axis=1)

    # Sort DataFrame by 'max_containment' column in descending order
    summary_df.sort_values(by='max_containment', ascending=False, inplace=True)

    # Save the summary DataFrame to a CSV file
    summary_df.to_csv(parsed_containment_file, sep="\t")
    logging.info(f"Summary saved to {parsed_containment_file}")

    # Open the containment-file in write mode
    with open(summary_containment_file, 'w') as f:
        # Write header
        f.write("Threshold\tCount\n")

        # Check different thresholds and write the count to the containment-file
        for i in range(int(1 / step_size), -1, -1):
            threshold = i * step_size
            rounded_threshold = round(threshold, 2)  # Round to two decimal places
            count = len(summary_df[summary_df['max_containment'] > threshold])
            f.write(f"{rounded_threshold}\t{count}\n")

    logging.info(f"Containment counts saved to {summary_containment_file}")


def count_single_sample(summary_file: str, metadata_file: str, summary_column: str, metadata_column: str,
                        threshold: float, top_n: int) -> Dict[str, int]:
    """
    Count occurrences for a single sample based on a containment threshold and match data between summary and metadata files.

    Parameters:
    - summary_file (str): Path to the summary file.
    - metadata_file (str): Path to the metadata file.
    - summary_column (str): Column name in the summary file to be considered.
    - metadata_column (str): Column name in the metadata file to be matched against the summary column.
    - threshold (float): Containment threshold for counting.
    - top_n (int): Number of top items to display.

    Returns:
    - Dict[str, int]: A dictionary of unique values in the metadata column with their counts.
    """

    logging.info(f"Starting the count_single_sample function with threshold: {threshold}")

    # Load the summary and metadata dataframes
    summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
    metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)

    # Join the dataframes on the index (SRA accession)
    df = summary_df.join(metadata_df, how='inner')

    # Select the accessions where the specified column in summary_df is greater than the threshold
    selected_accessions = df[df[summary_column] > threshold].index
    logging.info(f'Number of accessions with {summary_column} > {threshold}: {len(selected_accessions)}')

    # Select the rows from the metadata dataframe where the accession is in the selected accessions
    selected_metadata_df = metadata_df[metadata_df.index.isin(selected_accessions)]

    # Count the number of unique values in the specified column in metadata_df
    count_dict = selected_metadata_df[metadata_column].value_counts().to_dict()

    # Get the top n items
    top_n_items = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])

    # Log the top n
    for key, value in top_n_items.items():
        logging.info(f'{key}: {value}')

    return count_dict


def collect_genome_counts(
        summary_file: str,
        metadata_file: str,
        metadata_column: str,
        threshold: float,
        output_file: str,
        stat_file: str = None
    ) -> pd.DataFrame:
    """
    Count occurrences of genomes based on a threshold from summary and metadata files.

    Parameters:
    - summary_file (str): Path to the summary file.
    - metadata_file (str): Path to the metadata file.
    - metadata_column (str): Column name in the metadata file to be considered.
    - threshold (float): Containment threshold for counting.
    - output_file (str): Path to save the output file with genome counts.
    - stat_file (str): Path to save the statistics file.

    Returns:
    - pd.DataFrame: The resulting DataFrame containing genome counts.

    This function counts occurrences of genomes based on the specified threshold.
    It matches data between the summary and metadata files based on the provided column name.
    The results are saved to the specified output and statistics files.
    """
    logging.info("Starting collection of genome counts.")

    # If stat_file is not provided, generate it from output_file
    if stat_file is None:
        base_name, ext = os.path.splitext(output_file)
        stat_file = f"{base_name}_stats{ext}"
    print(stat_file)

    # Load the summary and metadata dataframes
    summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
    metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)

    # Lists and sets to store intermediate results
    df_list: List[pd.DataFrame] = []
    unique_accessions: Set[str] = set()

    # Iterate through all relevant columns in the summary DataFrame
    for sample_column in [col for col in summary_df.columns if "GCF" in col or "GCA" in col]:
        # Logic remains largely unchanged, optimized for clarity
        selected_accessions = summary_df.loc[summary_df[sample_column] > threshold].index
        unique_accessions.update(selected_accessions)

        selected_metadata_df = metadata_df.loc[metadata_df.index.isin(selected_accessions)]
        count_series = selected_metadata_df[metadata_column].value_counts()
        sample_df = pd.DataFrame({sample_column: count_series})
        df_list.append(sample_df)

    # Post-processing and output generation
    result_df = pd.concat(df_list, axis=1).fillna(0).astype(int)
    result_df.to_csv(output_file, sep="\t")

    logging.info(f"Table with sample files has been saved to {output_file}")

    total_counts = result_df.values.sum()
    logging.info(f"Total number of genome counts in the table: {total_counts}")

    logging.info(f"Total number of unique accessions after filtering: {len(unique_accessions)}")

    column_counts = result_df.sum().sort_values(ascending=False)
    column_counts.to_csv(stat_file, sep="\t", header=False)

    logging.info(f"Statistics have been saved to {stat_file}")

    return result_df





def assemble_datasets(args):
    """Assemble datasets based on input data files.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the following attributes:
        * data_files (List[str]): List of paths to data files.
        * output_file (str): Path to save the assembled dataset.

    The function assembles datasets by concatenating multiple input data files.
    The combined dataset is saved to the specified output file.
    """
    fastq_folder = Path('fastq')
    for fastq_file in fastq_folder.glob('*.fastq.gz'):
        if 'R1' in fastq_file.name or 'R2' in fastq_file.name:
            logging.info(f'Assembling Illumina dataset for {fastq_file.name}')
            subprocess.run(['megahit', '-1', str(fastq_file), '-2', str(fastq_file.replace('R1', 'R2')), '-o', f'{fastq_file.stem}_megahit'], check=True)
        else:
            logging.info(f'Assembling Nanopore dataset for {fastq_file.name}')
            subprocess.run(['flye', '--nano-raw', str(fastq_file), '--out-dir', f'{fastq_file.stem}_flye', '--meta'], check=True)

def download_accession(accession, fastq_folder):
    """
    Download the SRA dataset for the given accession.
    Saves the downloaded FASTQ file in the specified fastq_folder.
    Returns True if the download was successful, otherwise False.
    """
    output_file = fastq_folder / f'{accession}.fastq.gz'
    try:
        logging.info(f'Downloading SRA for {accession}')
        subprocess.run(['fastq-dump', '--gzip', '--split-files', accession, '-O', str(fastq_folder)], check=True)
        return True
    except subprocess.CalledProcessError:
        logging.error(f'Error downloading SRA for {accession}')
        return False

def download_sra(args):
    """
    Download SRA datasets based on the accessions in the summary.txt file.
    Only accessions with max_containment greater than the threshold will be considered.
    Limit the number of downloads with the max_downloads parameter.
    If dry_run is True, only log the number of datasets left to download without actually downloading them.
    """
    fastq_folder = Path(args.fastq_folder)
    fastq_folder.mkdir(exist_ok=True)

    # Load the summary.txt file without headers for the first column
    summary_df = pd.read_csv(args.summary_file, sep='\t', header=None, skiprows=1,
                             names=['accession', 'max_containment', 'max_containment_annotation1',
                                    'max_containment_annotation2'])
    accessions_to_download = summary_df[summary_df['max_containment'] > args.threshold]['accession'].tolist()

    if args.dry_run:
        logging.info(f"DRY RUN: {len(accessions_to_download)} datasets left to download.")
        print(len(accessions_to_download))
        return len(accessions_to_download)

    download_count = 0
    for accession in accessions_to_download:
        # Check if we've reached the maximum downloads for this call
        if args.max_downloads and download_count >= args.max_downloads:
            break

        if download_accession(accession, fastq_folder):
            download_count += 1

    return download_count