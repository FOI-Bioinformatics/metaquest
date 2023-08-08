# METAQUEST: MEtagenomic TAble QUery and Extraction System for public daTa sEts.
# This tool is designed for querying and extracting information from tables of public metagenomic data sets.


import argparse
import os
import csv
import logging
import subprocess
from pathlib import Path
from collections import defaultdict
import pandas as pd
from Bio import Entrez
from urllib.error import HTTPError
from lxml import etree
from upsetplot import UpSet
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sets up the logging module to write the logging output to 'pipeline.log' file. The log messages are set to level INFO.
logging.basicConfig(filename='pipeline.log', level=logging.INFO)


def download_fasta(args):
    Entrez.email = args.email  # Set the email
    
    with open(args.accession_file, 'r') as f:
        accessions = [line.strip() for line in f.readlines()]

    for accession in accessions:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        filename = f"{args.output_folder}/{accession}.fasta"
        with open(filename, "w") as f:
            SeqIO.write(record, f, "fasta")
        print(f"{accession} downloaded and saved as {filename}")

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



# Function 'summarize' reads the '.csv' files in the 'matches_folder' and creates a summary dataframe.
def summarize(args):
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

    # Print out the number of accessions with max_containment greater than specified values
    for i in range(9, -1, -1):
        threshold = i / 10
        count = len(summary_df[summary_df['max_containment'] > threshold])
        print(f'Number of accessions with max containment > {threshold}: {count}')


def plot_heatmap(args):
    df = pd.read_csv(args.summary_file, sep='\t', index_col=0)
    # only keep columns containing GCF or GCA
    df = df[[col for col in df.columns if 'GCF' in col or 'GCA' in col]]
    # replace missing values with 0
    df.fillna(0, inplace=True)
    # set values less than the threshold to 0
    df[df < args.threshold] = 0
    # remove rows and columns that only contain 0
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.loc[(df != 0).any(axis=1), :]
    plt.figure(figsize=(10, 10))
    sns.clustermap(df, cmap='viridis')
    plt.savefig(args.heatmap_file)

    # PCA plot
    # Transpose the DataFrame: rows are columns and columns are rows
    df_T = df.T

    # Standardize the data
    X = StandardScaler().fit_transform(df_T)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)
    ax.scatter(principalDf['PC1'], principalDf['PC2'])
    plt.savefig(args.pca_file)

# Function 'plot_upset' reads the summary file and generates an UpSet plot.
def plot_upset(args):
    summary_df = pd.read_csv(args.summary_file, sep="\t", index_col=0)

    # Keep only the columns that contain "GCF" or "GCA"
    summary_df = summary_df[[col for col in summary_df.columns if "GCF" in col or "GCA" in col]]

    # Convert columns to numeric types (errors='coerce' turns non-numeric values into NaN)
    summary_df = summary_df.apply(pd.to_numeric, errors='coerce')

    # Binarize the containment values
    threshold = 0.1  # Define an appropriate threshold
    binary_df = (summary_df > threshold)

    # Replace NaN values with False
    binary_df = binary_df.fillna(False)

    print(binary_df.applymap(np.isreal).all())

    # Stack the DataFrame and then reset the index
    binary_df = binary_df.stack().reset_index()
    binary_df.columns = ['accession', 'file', 'value']

    # Only keep rows where value is True
    binary_df = binary_df[binary_df['value']]

    # Drop the 'value' column
    binary_df = binary_df.drop(columns='value')

    # Set the multi-index
    binary_df.set_index(['accession', 'file'], inplace=True)

    # Generate UpSet plot
    upset = UpSet(binary_df, subset_size='count', intersection_plot_elements=3)
    upset.plot()
    plt.savefig(args.upset_plot_file)

# Function 'download_metadata' downloads the metadata for each SRA accession in the '.csv' files in the 'matches_folder'.
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



# Function 'parse_metadata' parses the metadata for each '_metadata.xml' file in the 'metadata_folder'.
def parse_metadata(args):
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



def count_single_sample(args):
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



# Function 'download_sra' downloads the SRA data for each unique SRA accession in the '.csv' files in the 'matches_folder'.
def download_sra(args):
    matches_folder = Path(args.matches_folder)
    fastq_folder = Path(args.fastq_folder)
    fastq_folder.mkdir(exist_ok=True)

    unique_accessions = set()
    for csv_file in matches_folder.glob('*.csv'):
        print(csv_file)
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                unique_accessions.add(row[0])

    print(f"Number of unique accessions: {len(unique_accessions)}")

    with open('unique_accessions.txt', 'w') as file:
        file.writelines(accession + '\n' for accession in unique_accessions)

    for accession in unique_accessions:
        output_file = fastq_folder / f'{accession}.fastq.gz'
        if not output_file.exists():
            logging.info(f'Downloading SRA for {accession}')
            subprocess.run(['fastq-dump', '--gzip', '--split-files', accession, '-O', str(fastq_folder)], check=True)


# Function 'assemble_datasets' assembles datasets for each '.fastq.gz' file in the 'fastq_folder'.
def assemble_datasets(args):
    fastq_folder = Path('fastq')
    for fastq_file in fastq_folder.glob('*.fastq.gz'):
        if 'R1' in fastq_file.name or 'R2' in fastq_file.name:
            logging.info(f'Assembling Illumina dataset for {fastq_file.name}')
            subprocess.run(['megahit', '-1', str(fastq_file), '-2', str(fastq_file.replace('R1', 'R2')), '-o', f'{fastq_file.stem}_megahit'], check=True)
        else:
            logging.info(f'Assembling Nanopore dataset for {fastq_file.name}')
            subprocess.run(['flye', '--nano-raw', str(fastq_file), '--out-dir', f'{fastq_file.stem}_flye', '--meta'], check=True)

# The 'argparse.ArgumentParser' class holds all the information necessary to parse the command line into Python data types.
# The 'add_argument()' method is used to specify which command-line options the program is willing to accept.
parser = argparse.ArgumentParser()

# The add_subparsers() method is used to add subparsers to the main parser.
subparsers = parser.add_subparsers()

# Each subparser corresponds to a function that the script can perform.

parser_download_fasta = subparsers.add_parser('download-fasta', help='Download fasta files from NCBI based on accessions.')
parser_download_fasta.add_argument('--email', required=True, help='Your email address for NCBI API access.')
parser_download_fasta.add_argument('--accession-file', required=True, help='File containing accessions, one per line.')
parser_download_fasta.add_argument('--output-folder', default='genomes', help='Folder to save downloaded fasta files.')
parser_download_fasta.set_defaults(func=download_fasta)


parser_mastiff = subparsers.add_parser('mastiff', help='Runs mastiff on the genome data in the specified folder.')
parser_mastiff.add_argument('--genomes-folder', default='genomes', help='Folder containing the genomes to be analyzed.')
parser_mastiff.add_argument('--matches-folder', default='matches', help='Folder where the output matches will be stored.')
parser_mastiff.set_defaults(func=run_mastiff)

parser_summarize = subparsers.add_parser('summarize', help='Summarizes the data from the .csv files in the matches directory.')
parser_summarize.add_argument('--matches-folder', default='matches', help='Folder containing the matches to be summarized.')
parser_summarize.add_argument('--summary-file', default='SRA-summary.txt', help='File where the summary will be stored.')
parser_summarize.add_argument('--containment-file', default='top_containments.txt', help='File where the top containments will be stored.')
parser_summarize.set_defaults(func=summarize)


parser_plot_upset = subparsers.add_parser('plot-upset', help='Generate UpSet plot from the summary file.')
parser_plot_upset.add_argument('--summary-file', default='SRA-summary.txt', help='File containing the summary data.')
parser_plot_upset.add_argument('--upset-plot-file', default='upset_plot.png', help='File where the UpSet plot will be saved.')
parser_plot_upset.set_defaults(func=plot_upset)

# Add a new subparser for the 'plot_heatmap' function
parser_heatmap = subparsers.add_parser('plot_heatmap', help='Generate a heatmap from the summary file with a threshold.')
parser_heatmap.add_argument('--summary-file', default='SRA-summary.txt', help='Path to the summary file.')
parser_heatmap.add_argument('--heatmap-file', required=True, help='Path to save the output heatmap.')
parser_heatmap.add_argument("--pca-file", required=True, help="PCA plot file")
parser_heatmap.add_argument('--threshold', type=float, default=0.1, help='Threshold below which values will be set to 0.')
parser_heatmap.set_defaults(func=plot_heatmap)

parser_download_metadata = subparsers.add_parser('download-metadata', help='Download metadata for each SRA accession in the .csv files in the matches directory')
parser_download_metadata.add_argument('--email', required=True, help='Your email address for NCBI API access.')
parser_download_metadata.add_argument('--matches_folder', default='matches', help='Folder containing match .csv files')
parser_download_metadata.add_argument('--metadata_folder', default='metadata', help='Folder to save downloaded metadata')
parser_download_metadata.add_argument('--dry-run', action='store_true', help='If enabled, no downloads are performed. Only calculates the total number of accessions and the number of accessions to download.')
parser_download_metadata.set_defaults(func=download_metadata)


parser_parse_metadata = subparsers.add_parser('parse-metadata', help='Parse metadata for each *_metadata.xml file in the specified directory')
parser_parse_metadata.add_argument('--metadata-folder', default='metadata', help='Folder containing *_metadata.xml files to parse')
parser_parse_metadata.add_argument('--metadata-table-file', default='metadata_table.txt', help='File where the parsed metadata will be stored')
parser_parse_metadata.set_defaults(func=parse_metadata)


# Add a new subparser for the 'count_greater_than_threshold' function
parser_single_sample = subparsers.add_parser('single_sample', help='Counts the occurrences of unique values in a column for a single sample.')
parser_single_sample.add_argument('--summary-file', default='SRA-summary.txt', help='Path to the summary file.')
parser_single_sample.add_argument('--metadata-file', default='metadata_table.txt', help='Path to the metadata file.')
parser_single_sample.add_argument('--summary-column', required=True, help='Name of the column in the summary file to compare with the threshold.')
parser_single_sample.add_argument('--metadata-column', required=True, help='Name of the column in the metadata file to count the unique values of.')
parser_single_sample.add_argument('--threshold', type=float, default=0.1, help='Threshold for the column in the summary file.')
parser_single_sample.add_argument('--top-n', type=int, default=100, help='Number of top items to keep.')
parser_single_sample.set_defaults(func=count_single_sample)

parser_genome_count = subparsers.add_parser('genome_count', help='Collects genome counts and outputs a table with sample files.')
parser_genome_count.add_argument('--summary-file', default='SRA-summary.txt', help='Path to the summary file.')
parser_genome_count.add_argument('--metadata-file', default='metadata_table.txt', help='Path to the metadata file.')
parser_genome_count.add_argument('--metadata-column', required=True, help='Name of the column in th metadata fil.')
parser_genome_count.add_argument('--threshold', type=float, default=0.5, help='Threshold for the column in the summary file.')
parser_genome_count.add_argument('--output-file', default='collected_table.txt', help='Path to the output file.')
parser_genome_count.add_argument('--stat-file', default="collected_stats.txt", help='Path to the statistics file.')
parser_genome_count.set_defaults(func=collect_genome_counts)


parser_download_sra = subparsers.add_parser('download-sra', help='Downloads SRA data for each unique SRA accession in the .csv files in the matches directory.')
parser_download_sra.add_argument('--matches-folder', default='matches', help='Folder containing the matches for which to download SRA data.')
parser_download_sra.add_argument('--fastq-folder', default='fastq', help='Folder where the downloaded SRA data will be stored.')
parser_download_sra.set_defaults(func=download_sra)

parser_assemble = subparsers.add_parser('assemble', help='Assembles datasets for each .fastq.gz file in the fastq folder.')
parser_assemble.set_defaults(func=assemble_datasets)

# The args are parsed by calling parse_args() and returns some data (namespace object) that has the data attached as attributes.
args = parser.parse_args()

# If the namespace object has the 'func' attribute (which means a valid subcommand was provided), the corresponding function is called. If not, the usage message is printed.
if hasattr(args, 'func'):  # check if 'func' attribute exists
    args.func(args)
else:
    parser.print_help()  # print usage message if no subcommand was provided
