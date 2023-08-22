
import pandas as pd
from pathlib import Path
import csv
import logging
import subprocess
import urllib.request
import gzip
from collections import defaultdict
from typing import Dict


def download_test_genome(args):
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/008/985/GCF_000008985.1_ASM898v1/GCF_000008985.1_ASM898v1_genomic.fna.gz"
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)  # Ensure the output directory exists

    # Define the local gz file path
    local_gz_file = output_folder / "temp_genome.fna.gz"

    # Download the compressed fasta file
    urllib.request.urlretrieve(url, local_gz_file)

    # Decompress the fasta file and save it to the output folder
    with gzip.open(local_gz_file, 'rt') as fh:  # Open the gzip file in text mode for reading
        genome_data = fh.read()

    output_path = output_folder / "GCF_000008985.1.fasta"
    with open(output_path, 'w') as out:
        out.write(genome_data)

    # Optionally, remove the temporary compressed file
    local_gz_file.unlink()

    print(f"Downloaded and saved test genome to {output_path}")


# Function 'run_mastiff' runs the 'mastiff' command on each '.fasta' file in the 'genomes_folder'.

def run_mastiff(args):
    genomes_folder = Path(args.genomes_folder)
    matches_folder = Path(args.matches_folder)
    matches_folder.mkdir(exist_ok=True)

    for fasta_file in genomes_folder.glob('*.fasta'):
        output_file = matches_folder / f'{fasta_file.stem}_matches.csv'
        if not output_file.exists():
            logging.info(f'Running mastiff on {fasta_file.name}')
            subprocess.run(['mastiff', str(fasta_file), '-o', str(output_file)], check=True)
        else:
            logging.info(f'Skipping mastiff on {fasta_file.name} because output file already exists')


def summarize(args):
    """Generate summary and containment files from MASH matches.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the following attributes:
        * matches_folder (str): Path to the folder containing _matches.csv files from MASH.
        * summary_file (str): Path to save the output summary file.
        * containment_file (str): Path to save the output containment file.
        
    The function aggregates information from the MASH matches, computes the containment score,
    and saves the summary and containment data in the respective output files.
    """
    matches_folder = Path(args.matches_folder)
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for csv_file in matches_folder.glob('*.csv'):
        file_id = csv_file.stem.replace("_matches", "")
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                accession = row['SRA accession']
                containment = float(row['containment'])
                if file_id not in summary[accession]:
                    summary[accession][file_id] = containment
                else:
                    summary[accession][file_id] = max(summary[accession][file_id], containment)

    summary_df = pd.DataFrame.from_dict(summary, orient='index')

    # Fill NA/NaN values with 0
    summary_df.fillna(0, inplace=True)

    # Add 'max_containment' column by finding the max value across each row
    summary_df['max_containment'] = summary_df.max(axis=1)

    # Add 'max_containment_annotation' column with the csv file with highest containment
    summary_df['max_containment_annotation'] = summary_df.idxmax(axis=1)

    # Sort DataFrame by 'max_containment' column in descending order
    summary_df.sort_values(by='max_containment', ascending=False, inplace=True)

    summary_df.to_csv(args.summary_file, sep="\t")
    print(f"Summary saved to {args.summary_file}")

    # Open the containment-file in write mode
    with open(args.containment_file, 'w') as f:
        # Write header
        f.write("Threshold\tCount\n")

        # Check different thresholds and write the count to the containment-file
        for i in range(9, -1, -1):
            threshold = i / 10
            count = len(summary_df[summary_df['max_containment'] > threshold])
            f.write(f"{threshold}\t{count}\n")

    print(f"Containment counts saved to {args.containment_file}")





def count_single_sample(args):
    """Count occurrences for a single sample in the summary and metadata files.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the following attributes:
        * summary_file (str): Path to the summary file.
        * metadata_file (str): Path to the metadata file.
        * summary_column (str): Column name in the summary file to be considered.
        * metadata_column (str): Column name in the metadata file to be matched against the summary column.
        * threshold (float): Containment threshold for counting.

    The function counts occurrences for a single sample based on the specified threshold.
    It matches data between the summary and metadata files based on the provided column names.
    """
    # Load the summary and metadata dataframes
    summary_df = pd.read_csv(args.summary_file, sep="\t", index_col=0)
    metadata_df = pd.read_csv(args.metadata_file, sep="\t", index_col=0)

    # Join the dataframes on the index (SRA accession)
    df = summary_df.join(metadata_df)

    # Select the accessions where the specified column in summary_df is greater than the threshold
    selected_accessions = df[df[args.summary_column] > args.threshold].index
    print(f'Number of accessions with {args.summary_column} > {args.threshold}: {len(selected_accessions)}')

    # Select the rows from the metadata dataframe where the accession is in the selected accessions
    selected_metadata_df = metadata_df[metadata_df.index.isin(selected_accessions)]
    print(f'Number of rows in selected metadata dataframe: {len(selected_metadata_df)}')

    # Count the number of unique values in the specified column in metadata_df
    count_dict = selected_metadata_df[args.metadata_column].value_counts().to_dict()

    # Get the top n items
    top_n = dict(list(count_dict.items())[:args.top_n])

    # Print the top n
    for key, value in top_n.items():
        print(f'{key}: {value}')

    return count_dict



def collect_genome_counts(args):
    """Count occurrences of genomes based on a threshold from summary and metadata files.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the following attributes:
        * summary_file (str): Path to the summary file.
        * metadata_file (str): Path to the metadata file.
        * metadata_column (str): Column name in the metadata file to be considered.
        * threshold (float): Containment threshold for counting.
        * output_file (str): Path to save the output file with genome counts.

    The function counts occurrences of genomes based on the specified threshold.
    It matches data between the summary and metadata files based on the provided column name.
    The results are saved to the specified output file.
    """
    # Load the summary and metadata dataframes
    summary_df = pd.read_csv(args.summary_file, sep="\t", index_col=0)
    metadata_df = pd.read_csv(args.metadata_file, sep="\t", index_col=0)

    # Create a list to store the DataFrames for each sample column
    df_list = []

    # Create a set to store unique accessions after filtering
    unique_accessions = set()

    # Iterate through all sample columns in the summary DataFrame that contain "GCF" or "GCA"
    for sample_column in summary_df.columns:
        if "GCF" in sample_column or "GCA" in sample_column:
            # Select the accessions where the current sample column is greater than the threshold
            selected_accessions = summary_df[summary_df[sample_column] > args.threshold].index

            # Update the set of unique accessions
            unique_accessions.update(selected_accessions)

            # Select the rows from the metadata dataframe where the accession is in the selected accessions
            selected_metadata_df = metadata_df[metadata_df.index.isin(selected_accessions)]

            # Count the number of unique values in the specified column in metadata_df
            count_series = selected_metadata_df[args.metadata_column].value_counts()

            # Add the counts for the current sample column to a new DataFrame
            sample_df = pd.DataFrame({sample_column: count_series})

            # Add the DataFrame to the list
            df_list.append(sample_df)

    # Concatenate the DataFrames along the index (SRA accession) axis
    result_df = pd.concat(df_list, axis=1)

    # Fill missing values with 0 and convert the entire DataFrame to integers
    result_df = result_df.fillna(0).astype(int)

    # Output the result DataFrame to a tab-delimited text file
    output_path = args.output_file
    result_df.to_csv(output_path, sep="\t")

    print(f"Table with sample files has been saved to {output_path}")

    # Calculate and print the total number of counts in the table
    total_counts = result_df.sum().sum()
    print(f"Total number of genome counts in the table: {total_counts}")

    # Print the total number of unique accessions after filtering
    print(f"Total number of unique accessions after filtering: {len(unique_accessions)}")

    # Calculate and save the counts per column to a statistics file
    column_counts = result_df.sum().sort_values(ascending=False)
    column_counts.to_csv(args.stat_file, sep="\t", header=False)

    print(f"Statistics have been saved to {args.stat_file}")

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