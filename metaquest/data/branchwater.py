"""
Branchwater data handling module for MetaQuest.

This module provides functions for processing Branchwater containment files.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError, ValidationError
from metaquest.core.models import Containment, ContainmentSummary, SRAMetadata
from metaquest.core.validation import validate_folder
from metaquest.data.file_io import ensure_directory, list_files, copy_file, read_csv, write_csv
from metaquest.plugins.base import format_registry
from metaquest.plugins.formats.branchwater import BranchWaterFormatPlugin
from metaquest.plugins.formats.mastiff import MastiffFormatPlugin

logger = logging.getLogger(__name__)

# Register format plugins
format_registry.register(BranchWaterFormatPlugin)
format_registry.register(MastiffFormatPlugin)


def process_branchwater_files(
    source_folder: Union[str, Path],
    target_folder: Union[str, Path]
) -> Dict[str, Path]:
    """
    Process Branchwater files from source folder and copy to target folder.
    
    Args:
        source_folder: Folder containing Branchwater CSV files
        target_folder: Folder to save processed CSV files
        
    Returns:
        Dictionary mapping genome IDs to processed file paths
        
    Raises:
        DataAccessError: If the process fails
    """
    source_path = validate_folder(source_folder)
    target_path = ensure_directory(target_folder)
    
    result_files = {}
    processed_count = 0
    error_count = 0
    
    # Get all CSV files in the source folder
    csv_files = list_files(source_path, "*.csv")
    if not csv_files:
        logger.warning(f"No CSV files found in {source_path}")
        return result_files
    
    for csv_file in csv_files:
        output_file = target_path / csv_file.name
        genome_id = csv_file.stem
        
        # Skip if the output file already exists
        if output_file.exists():
            logger.info(f"Skipping {csv_file.name} as output file already exists")
            result_files[genome_id] = output_file
            processed_count += 1
            continue
        
        try:
            # Determine file format
            try:
                # Try to load as Branchwater format
                BranchWaterFormatPlugin.validate_header(
                    read_csv(csv_file, nrows=0).columns.tolist()
                )
                format_name = "branchwater"
            except (ValidationError, DataAccessError):
                try:
                    # Try to load as Mastiff format
                    MastiffFormatPlugin.validate_header(
                        read_csv(csv_file, nrows=0).columns.tolist()
                    )
                    format_name = "mastiff"
                except (ValidationError, DataAccessError):
                    raise ValidationError(f"Unknown file format for {csv_file}")
            
            # Copy the file
            result_file = copy_file(csv_file, output_file)
            logger.info(f"Processed {format_name} file: {csv_file.name}")
            result_files[genome_id] = result_file
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {csv_file}: {e}")
    
    logger.info(f"Processed {processed_count} files with {error_count} errors")
    return result_files


def extract_metadata_from_branchwater(
    source_folder: Union[str, Path],
    output_file: Union[str, Path]
) -> pd.DataFrame:
    """
    Extract metadata from Branchwater files and save to a file.
    
    Args:
        source_folder: Folder containing Branchwater CSV files
        output_file: Path to save metadata CSV file
        
    Returns:
        DataFrame containing extracted metadata
        
    Raises:
        DataAccessError: If the extraction fails
    """
    source_path = validate_folder(source_folder)
    metadata_records = []
    processed_count = 0
    error_count = 0
    
    # Get all CSV files in the source folder
    csv_files = list_files(source_path, "*.csv")
    if not csv_files:
        logger.warning(f"No CSV files found in {source_path}")
        return pd.DataFrame()
    
    for csv_file in csv_files:
        try:
            # Determine file format and use appropriate plugin
            try:
                df_headers = read_csv(csv_file, nrows=0)
                headers = df_headers.columns.tolist()
                
                if BranchWaterFormatPlugin.validate_header(headers):
                    format_plugin = BranchWaterFormatPlugin
                elif MastiffFormatPlugin.validate_header(headers):
                    format_plugin = MastiffFormatPlugin
                else:
                    raise ValidationError(f"Unknown file format for {csv_file}")
                
                genome_id = csv_file.stem
                
                # Parse file to get containments
                containments = format_plugin.parse_file(csv_file, genome_id)
                
                # Extract metadata from each containment
                for containment in containments:
                    metadata = format_plugin.extract_metadata(containment)
                    if metadata:
                        # Convert to dictionary for DataFrame
                        meta_dict = {
                            'Run_ID': metadata.accession,
                            'Sample_ID': metadata.biosample,
                            'Project_ID': metadata.bioproject,
                            'Sample_Scientific_Name': metadata.organism,
                            'geo_loc_name_country_calc': metadata.location,
                            'collection_date_sam': metadata.collection_date,
                            'assay_type': metadata.assay_type,
                            'lat_lon': metadata.latitude_longitude
                        }
                        
                        # Add all attributes
                        for key, value in metadata.attributes.items():
                            if key not in meta_dict:
                                meta_dict[key] = value
                        
                        metadata_records.append(meta_dict)
                
                processed_count += 1
                
            except Exception as e:
                raise DataAccessError(f"Error processing {csv_file}: {e}")
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error extracting metadata from {csv_file}: {e}")
    
    logger.info(f"Processed {processed_count} files with {error_count} errors")
    
    if not metadata_records:
        logger.warning("No metadata records extracted")
        return pd.DataFrame()
    
    try:
        # Create DataFrame and remove duplicates
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df = metadata_df.drop_duplicates(subset=['Run_ID'])
        
        # Save to file
        write_csv(metadata_df, output_file, index=False)
        logger.info(f"Extracted metadata saved to {output_file} with {len(metadata_df)} records")
        
        return metadata_df
        
    except Exception as e:
        raise DataAccessError(f"Error creating metadata file: {e}")


def parse_containment_data(
    matches_folder: Union[str, Path],
    output_file: Union[str, Path],
    summary_file: Union[str, Path],
    step_size: float = 0.1
) -> ContainmentSummary:
    """
    Parse containment data from match files and generate summary.
    
    Args:
        matches_folder: Folder containing match files
        output_file: Path to save parsed containment data
        summary_file: Path to save containment summary
        step_size: Step size for threshold calculation
        
    Returns:
        ContainmentSummary object
        
    Raises:
        DataAccessError: If the parsing fails
    """
    matches_path = validate_folder(matches_folder)
    
    # Dictionary to store containment data
    containment_data = defaultdict(dict)
    
    # Get all CSV files in the matches folder
    csv_files = list_files(matches_path, "*.csv")
    if not csv_files:
        logger.warning(f"No CSV files found in {matches_path}")
        return ContainmentSummary()
    
    processed_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        genome_id = csv_file.stem
        
        try:
            # Determine file format and use appropriate plugin
            try:
                df_headers = read_csv(csv_file, nrows=0)
                headers = df_headers.columns.tolist()
                
                if BranchWaterFormatPlugin.validate_header(headers):
                    format_plugin = BranchWaterFormatPlugin
                elif MastiffFormatPlugin.validate_header(headers):
                    format_plugin = MastiffFormatPlugin
                else:
                    raise ValidationError(f"Unknown file format for {csv_file}")
                
                # Parse file to get containments
                containments = format_plugin.parse_file(csv_file, genome_id)
                
                # Add to containment data dictionary
                for containment in containments:
                    if genome_id not in containment_data[containment.accession]:
                        containment_data[containment.accession][genome_id] = containment.value
                    else:
                        # Keep maximum containment value if multiple entries
                        containment_data[containment.accession][genome_id] = max(
                            containment_data[containment.accession][genome_id],
                            containment.value
                        )
                
                processed_count += 1
                
            except Exception as e:
                raise DataAccessError(f"Error processing {csv_file}: {e}")
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error parsing containment from {csv_file}: {e}")
    
    logger.info(f"Processed {processed_count} files with {error_count} errors")
    
    if not containment_data:
        logger.warning("No valid containment data found")
        return ContainmentSummary()
    
    try:
        # Create DataFrame from containment data
        df = pd.DataFrame.from_dict(containment_data, orient='index')
        
        # Fill NA values with 0
        df.fillna(0, inplace=True)
        
        # Add max_containment column
        df['max_containment'] = df.max(axis=1)
        
        # Add max_containment_annotation column
        df['max_containment_annotation'] = df.idxmax(axis=1)
        
        # Sort by max_containment
        df.sort_values(by='max_containment', ascending=False, inplace=True)
        
        # Save parsed containment data
        write_csv(df, output_file, sep="\t")
        logger.info(f"Parsed containment data saved to {output_file}")
        
        # Generate summary data
        thresholds = []
        counts = []
        
        for i in range(int(1 / step_size), -1, -1):
            threshold = i * step_size
            rounded_threshold = round(threshold, 2)
            count = len(df[df['max_containment'] > threshold])
            
            thresholds.append(rounded_threshold)
            counts.append(count)
        
        # Save summary data
        summary_df = pd.DataFrame({'Threshold': thresholds, 'Count': counts})
        write_csv(summary_df, summary_file, sep="\t", index=False)
        logger.info(f"Containment summary saved to {summary_file}")
        
        # Create and return ContainmentSummary object
        summary = ContainmentSummary(
            thresholds=thresholds,
            counts=counts,
            max_containment={acc: val for acc, val in df['max_containment'].items()}
        )
        
        # Populate genome_to_samples mapping
        for genome_id in df.columns:
            if genome_id not in ('max_containment', 'max_containment_annotation'):
                accessions = df[df[genome_id] > 0].index.tolist()
                summary.genome_to_samples[genome_id] = accessions
        
        # Populate sample_to_genomes mapping
        for accession, row in df.iterrows():
            genomes = [genome for genome in df.columns 
                      if genome not in ('max_containment', 'max_containment_annotation') 
                      and row[genome] > 0]
            summary.sample_to_genomes[accession] = genomes
        
        return summary
        
    except Exception as e:
        raise DataAccessError(f"Error generating containment summary: {e}")