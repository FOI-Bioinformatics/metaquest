import os
import pandas as pd
from pathlib import Path
import csv
import logging
import subprocess
import urllib.request
import gzip
import shutil
from collections import defaultdict
from typing import Dict, Union, List, NoReturn, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# Required columns for different file formats
BRANCHWATER_REQUIRED_COLS = ['acc', 'containment']
MASTIFF_REQUIRED_COLS = ['SRA accession', 'containment']


class ValidationError(Exception):
    """Exception raised for validation errors in input files."""
    pass


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


def detect_file_format(file_path: Union[str, Path]) -> str:
    """
    Automatically detect the format of a CSV file (branchwater or mastiff).
    
    Parameters:
    - file_path (Union[str, Path]): Path to the CSV file to analyze
    
    Returns:
    - str: Detected format ('branchwater' or 'mastiff')
    
    Raises:
    - ValidationError: If the file format cannot be determined
    """
    try:
        with open(file_path, 'r') as f:
            # Read the header line
            header = f.readline().strip()
            
            # Check for branchwater format
            if 'acc' in header and 'containment' in header:
                return 'branchwater'
            
            # Check for mastiff format
            if 'SRA accession' in header and 'containment' in header:
                return 'mastiff'
            
            # If we can't determine the format
            raise ValidationError(f"Could not determine file format for {file_path}. Header: {header}")
    except Exception as e:
        raise ValidationError(f"Error reading file {file_path}: {str(e)}")


def validate_csv_file(file_path: Union[str, Path], file_format: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Validate a CSV file by checking required columns and format.
    
    Parameters:
    - file_path (Union[str, Path]): Path to the CSV file to validate
    - file_format (Optional[str]): The expected format ('branchwater' or 'mastiff')
                                  If None, format will be automatically detected
    
    Returns:
    - Tuple[str, List[str]]: Tuple of (detected format, list of column headers)
    
    Raises:
    - ValidationError: If the file does not meet validation requirements
    """
    try:
        # Auto-detect format if not specified
        detected_format = detect_file_format(file_path) if file_format is None else file_format
        
        # Determine required columns based on format
        if detected_format == 'branchwater':
            required_cols = BRANCHWATER_REQUIRED_COLS
        elif detected_format == 'mastiff':
            required_cols = MASTIFF_REQUIRED_COLS
        else:
            raise ValidationError(f"Unsupported file format: {detected_format}")
        
        # Read and validate headers
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                raise ValidationError(f"File {file_path} is empty")
            
            # Check for required columns
            missing_cols = [col for col in required_cols if col not in headers]
            if missing_cols:
                raise ValidationError(
                    f"Missing required columns in {file_path}: {', '.join(missing_cols)}\n"
                    f"Expected format: {detected_format}, Headers found: {', '.join(headers)}"
                )
            
            # Check for data rows
            try:
                next(reader)  # Try to read the first data row
            except StopIteration:
                raise ValidationError(f"File {file_path} contains headers but no data rows")
        
        return detected_format, headers
    
    except ValidationError:
        # Re-raise ValidationError as is
        raise
    except Exception as e:
        # Wrap other exceptions in ValidationError
        raise ValidationError(f"Error validating {file_path}: {str(e)}")


def use_branchwater_files(branchwater_folder: Union[str, Path], matches_folder: Union[str, Path]) -> None:
    """
    Process pre-downloaded Branchwater files and prepare them for MetaQuest pipeline.
    Validates files and handles errors for individual files.
    
    Parameters:
    - branchwater_folder (Union[str, Path]): Path to the folder containing downloaded Branchwater files.
    - matches_folder (Union[str, Path]): Path to the folder where processed match files will be saved.
    
    Returns:
    None
    """
    logging.info("Processing Branchwater containment files.")
    
    branchwater_folder = Path(branchwater_folder)
    matches_folder = Path(matches_folder)
    matches_folder.mkdir(exist_ok=True)  # Ensure the output directory exists
    
    logging.info(f"Branchwater files folder is set to {branchwater_folder}")
    logging.info(f"Matches folder is set to {matches_folder}")
    
    # Process all CSV files in the branchwater folder
    csv_files = list(branchwater_folder.glob('*.csv'))
    if not csv_files:
        logging.warning(f"No CSV files found in {branchwater_folder}")
        return
    
    processed_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        output_file = matches_folder / csv_file.name
        
        # Skip if the output file already exists
        if output_file.exists():
            logging.info(f'Skipping {csv_file.name} because output file already exists')
            processed_count += 1
            continue
        
        try:
            # Validate the file
            file_format, _ = validate_csv_file(csv_file)
            logging.info(f'Processing {file_format} file: {csv_file.name}')
            
            # Copy the file to the matches folder
            shutil.copy(csv_file, output_file)
            logging.info(f'Successfully processed {file_format} file: {csv_file.name}')
            processed_count += 1
            
        except ValidationError as e:
            error_count += 1
            logging.error(f"Validation error in {csv_file}: {str(e)}")
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing {csv_file}: {str(e)}")
    
    logging.info(f"Processed {processed_count} files successfully with {error_count} errors")


def parse_containment(matches_folder: Union[str, Path], parsed_containment_file: Union[str, Path],
                      summary_containment_file: Union[str, Path], step_size=0.1, file_format=None) -> None:
    """
    Generate summary and containment files from match files, with automatic format detection.

    Parameters:
    - matches_folder (Union[str, Path]): The path to the folder containing match files.
    - parsed_containment_file (Union[str, Path]): The path to save the parsed containment data.
    - summary_containment_file (Union[str, Path]): The path to save the containment summary.
    - step_size (float): Step size for threshold calculation in the summary.
    - file_format (str, optional): Format of the input files ('branchwater' or 'mastiff').
                                  If None, format will be automatically detected for each file.

    Returns:
    None
    """
    logging.info("Starting to summarize matches.")

    matches_folder = Path(matches_folder)
    summary: Dict[str, Dict[str, float]] = defaultdict(dict)

    logging.info(f"Scanning matches folder: {matches_folder}")
    
    processed_count = 0
    error_count = 0

    for csv_file in matches_folder.glob('*.csv'):
        # Extract file ID (genome identifier)
        file_id = csv_file.stem.replace("_matches", "")
        
        try:
            # Validate and detect format if not specified
            detected_format, _ = validate_csv_file(csv_file, file_format)
            logging.info(f"Processing file: {file_id} (format: {detected_format})")

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows_processed = 0
                
                for row in reader:
                    try:
                        if detected_format == 'branchwater':
                            accession = row.get('acc', '')
                            containment_str = row.get('containment', '')
                        else:  # mastiff
                            accession = row.get('SRA accession', '')
                            containment_str = row.get('containment', '')
                        
                        # Validate accession and containment
                        if not accession:
                            logging.warning(f"Skipping row with missing accession in {csv_file}")
                            continue
                        
                        # Handle missing or invalid containment values
                        try:
                            containment = float(containment_str)
                        except (ValueError, TypeError):
                            logging.warning(f"Invalid containment value '{containment_str}' in {csv_file} for accession {accession}")
                            continue

                        if file_id not in summary[accession]:
                            summary[accession][file_id] = containment
                        else:
                            summary[accession][file_id] = max(summary[accession][file_id], containment)
                        
                        rows_processed += 1
                    
                    except Exception as e:
                        logging.warning(f"Error processing row in {csv_file}: {str(e)}")
                
                logging.info(f"Processed {rows_processed} rows from {csv_file}")
            
            processed_count += 1
            
        except ValidationError as e:
            error_count += 1
            logging.error(f"Validation error in {csv_file}: {str(e)}")
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing {csv_file}: {str(e)}")

    logging.info(f"Completed processing of {processed_count} files with {error_count} errors.")
    
    if not summary:
        logging.warning("No valid data was found. Cannot create summary files.")
        return

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


def extract_branchwater_metadata(branchwater_folder: Union[str, Path], metadata_folder: Union[str, Path]) -> None:
    """
    Extract basic metadata from Branchwater CSV files and convert to a format compatible with MetaQuest.
    Validates files and handles errors for individual files.
    
    Parameters:
    - branchwater_folder (Union[str, Path]): Path to the folder containing Branchwater CSV files.
    - metadata_folder (Union[str, Path]): Path to the folder where extracted metadata will be saved.
    
    Returns:
    None
    """
    logging.info("Extracting metadata from Branchwater files.")
    
    branchwater_folder = Path(branchwater_folder)
    metadata_folder = Path(metadata_folder)
    metadata_folder.mkdir(exist_ok=True)
    
    metadata_records = []
    processed_count = 0
    error_count = 0
    
    for csv_file in branchwater_folder.glob('*.csv'):
        try:
            # Validate the file
            file_format, headers = validate_csv_file(csv_file)
            
            if file_format != 'branchwater':
                logging.warning(f"Skipping {csv_file} - not in Branchwater format")
                continue
                
            logging.info(f"Extracting metadata from {csv_file}")
            
            # Read the CSV file with pandas (more robust than csv module for handling various issues)
            df = pd.read_csv(csv_file, on_bad_lines='warn')
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Check if acc column exists and has a value
                    if 'acc' not in row or pd.isna(row['acc']):
                        continue
                        
                    # Create a record with all available columns
                    record = {'Run_ID': row['acc']}
                    
                    # Map known Branchwater columns to MetaQuest format
                    column_mapping = {
                        'biosample': 'Sample_ID',
                        'bioproject': 'Project_ID',
                        'organism': 'Sample_Scientific_Name',
                        'geo_loc_name_country_calc': 'geo_loc_name_country_calc',
                        'collection_date_sam': 'collection_date_sam',
                        'assay_type': 'assay_type',
                        'lat_lon': 'lat_lon'
                    }
                    
                    # Add all available mapped columns
                    for src_col, dest_col in column_mapping.items():
                        if src_col in row and not pd.isna(row[src_col]):
                            record[dest_col] = row[src_col]
                    
                    metadata_records.append(record)
                except Exception as e:
                    logging.warning(f"Error processing row in {csv_file}: {str(e)}")
            
            processed_count += 1
            
        except ValidationError as e:
            error_count += 1
            logging.error(f"Validation error in {csv_file}: {str(e)}")
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing {csv_file}: {str(e)}")
    
    logging.info(f"Processed {processed_count} files with {error_count} errors.")
    
    if not metadata_records:
        logging.warning("No metadata records were extracted.")
        return
    
    # Create a DataFrame from all records
    try:
        metadata_df = pd.DataFrame(metadata_records)
        
        # Remove duplicate records based on Run_ID
        metadata_df = metadata_df.drop_duplicates(subset=['Run_ID'])
        
        # Save to a tab-separated file
        output_file = metadata_folder / "branchwater_metadata.txt"
        metadata_df.to_csv(output_file, sep="\t", index=False)
        logging.info(f"Extracted metadata saved to {output_file} with {len(metadata_df)} records")
    except Exception as e:
        logging.error(f"Error creating metadata file: {str(e)}")


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

    try:
        # Load the summary and metadata dataframes
        summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
        metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)
        
        # Validate summary_column exists
        if summary_column not in summary_df.columns:
            raise ValueError(f"Column '{summary_column}' not found in summary file. Available columns: {', '.join(summary_df.columns)}")
        
        # Validate metadata_column exists
        if metadata_column not in metadata_df.columns:
            raise ValueError(f"Column '{metadata_column}' not found in metadata file. Available columns: {', '.join(metadata_df.columns)}")

        # Join the dataframes on the index (SRA accession)
        df = summary_df.join(metadata_df, how='inner')

        # Select the accessions where the specified column in summary_df is greater than the threshold
        selected_accessions = df[df[summary_column] > threshold].index
        logging.info(f'Number of accessions with {summary_column} > {threshold}: {len(selected_accessions)}')

        # Select the rows from the metadata dataframe where the accession is in the selected accessions
        selected_metadata_df = metadata_df[metadata_df.index.isin(selected_accessions)]

        # Check if we have matching data
        if selected_metadata_df.empty:
            logging.warning(f"No matching data found above threshold {threshold}")
            return {}

        # Count the number of unique values in the specified column in metadata_df
        count_dict = selected_metadata_df[metadata_column].value_counts().to_dict()

        # Get the top n items
        top_n_items = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])

        # Log the top n
        for key, value in top_n_items.items():
            logging.info(f'{key}: {value}')

        return count_dict
    
    except Exception as e:
        logging.error(f"Error in count_single_sample: {str(e)}")
        return {}


def count_metadata(
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

    try:
        # If stat_file is not provided, generate it from output_file
        if stat_file is None:
            base_name, ext = os.path.splitext(output_file)
            stat_file = f"{base_name}_stats{ext}"
        print(stat_file)

        # Load the summary and metadata dataframes
        summary_df = pd.read_csv(summary_file, sep="\t", index_col=0)
        metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)
        
        # Validate metadata_column exists
        if metadata_column not in metadata_df.columns:
            raise ValueError(f"Column '{metadata_column}' not found in metadata file. Available columns: {', '.join(metadata_df.columns)}")

        # Lists and sets to store intermediate results
        df_list: List[pd.DataFrame] = []
        unique_accessions: Set[str] = set()
        
        # Get relevant genome columns
        genome_columns = [col for col in summary_df.columns if "GCF" in col or "GCA" in col]
        
        if not genome_columns:
            logging.warning("No genome columns (GCF/GCA) found in summary file.")
            return pd.DataFrame()

        # Iterate through all relevant columns in the summary DataFrame
        for sample_column in genome_columns:
            try:
                # Select accessions above threshold
                selected_accessions = summary_df.loc[summary_df[sample_column] > threshold].index
                unique_accessions.update(selected_accessions)

                # Filter metadata by selected accessions
                selected_metadata_df = metadata_df.loc[metadata_df.index.isin(selected_accessions)]
                
                if selected_metadata_df.empty:
                    logging.warning(f"No matching metadata for {sample_column} above threshold {threshold}")
                    continue
                
                # Count occurrences of values in metadata column
                count_series = selected_metadata_df[metadata_column].value_counts()
                sample_df = pd.DataFrame({sample_column: count_series})
                df_list.append(sample_df)
            
            except Exception as e:
                logging.error(f"Error processing column {sample_column}: {str(e)}")

        # Check if we have any data
        if not df_list:
            logging.warning("No valid data collected. Cannot create output files.")
            return pd.DataFrame()

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
    
    except Exception as e:
        logging.error(f"Error in count_metadata: {str(e)}")
        return pd.DataFrame()


def download_accession(accession, fastq_folder, num_threads):
    """
    Download the SRA dataset for the given accession using fasterq-dump.
    Saves the downloaded FASTQ file in a subfolder named after the accession inside the specified fastq_folder.
    Returns True if the download was successful, otherwise False.
    """
    output_folder = fastq_folder / accession

    # Skip if the output folder already exists
    if output_folder.exists():
        logging.info(f"Skipping {accession}, folder already exists.")
        return False

    output_folder.mkdir(exist_ok=True)

    try:
        logging.info(f"Downloading SRA for {accession}")
        subprocess.run(
            ["fasterq-dump", "--threads", str(num_threads), "--progress", accession, "-O", str(output_folder)],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        logging.error(f"Error downloading SRA for {accession}")
        return False

def download_sra(fastq_folder, accessions_file, max_downloads=None, dry_run=False, num_threads=4, max_workers=4):
    """
    Download SRA datasets based on the accessions in the accessions_file.
    Utilizes multi-threading to speed up the download process.

    Parameters:
    - fastq_folder (str): Path where downloaded FASTQ files should be saved.
    - accessions_file (str): File containing a list of SRA accessions to download.
    - max_downloads (int, optional): Maximum number of datasets to download.
    - dry_run (bool): If True, only logs the total number of accessions without downloading.
    - num_threads (int): Number of threads per `fasterq-dump` subprocess.
    - max_workers (int): Number of threads to use for parallel downloads.

    Returns:
    - download_count (int): The number of successfully downloaded datasets.
    """
    fastq_folder = Path(fastq_folder)
    fastq_folder.mkdir(exist_ok=True)

    try:
        with open(accessions_file, "r") as f:
            all_accessions = [line.strip() for line in f if line.strip()]

        logging.info(f"Total number of accessions in accession file: {len(all_accessions)}")

        # Filter out accessions that already have a corresponding folder
        accessions_to_download = [acc for acc in all_accessions if not (fastq_folder / acc).exists()]

        logging.info(f"Total number of accessions to download: {len(accessions_to_download)}")

        if dry_run:
            logging.info(f"DRY RUN: {len(accessions_to_download)} datasets left to download.")
            return len(accessions_to_download)

        download_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_accession = {executor.submit(download_accession, accession, fastq_folder, num_threads): accession for accession in accessions_to_download}

            for future in as_completed(future_to_accession):
                accession = future_to_accession[future]
                try:
                    success = future.result()
                    if success:
                        download_count += 1

                    if max_downloads and download_count >= max_downloads:
                        logging.info("Reached maximum download limit.")
                        break
                except Exception as e:
                    logging.error(f"Error downloading {accession}: {e}")

        return download_count
    
    except Exception as e:
        logging.error(f"Error in download_sra: {str(e)}")
        return 0


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