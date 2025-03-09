# Using MetaQuest with Branchwater: Workflow Guide

This guide explains how to use the updated MetaQuest tool with Branchwater containment files.

## Transitioning from Mastiff to Branchwater

MetaQuest has been updated to work with Branchwater containment files instead of Mastiff. Here's how the workflow has changed:

### Previous Workflow (Mastiff)
1. Download test genomes
2. Run Mastiff on the genomes to get containment matches
3. Process matches and continue with analysis

### New Workflow (Branchwater)
1. Manually obtain containment files from Branchwater web interface
2. Process these files with MetaQuest
3. Continue with analysis

## Step-by-Step Guide

### 1. Get Containment Files from Branchwater

Visit [https://branchwater.jgi.doe.gov/](https://branchwater.jgi.doe.gov/) to search for your genomes of interest.

1. Upload or select your genome
2. Run the search against the SRA database
3. Download the results as CSV files
4. Save all downloaded CSV files to a single folder on your computer

Example CSV format from Branchwater:
```
acc,containment,cANI,biosample,bioproject,assay_type,collection_date_sam,geo_loc_name_country_calc,organism,lat_lon
SRR12345678,0.95,0.98,SAMN12345678,PRJNA123456,AMPLICON,2019-05-01,USA,Francisella adeliensis,35.7N 100.2W
```

### 2. Process Branchwater Files

Process the downloaded Branchwater files to prepare them for the MetaQuest pipeline:

```bash
metaquest use_branchwater --branchwater-folder /path/to/branchwater/files --matches-folder matches
```

This command:
- Takes Branchwater CSV files as input
- Processes them for compatibility with MetaQuest
- Stores the processed files in the matches folder

### 3. Decide on Metadata Approach

You have two options for metadata:

#### Option A: Use Metadata from Branchwater Files

Branchwater files already contain basic metadata (biosample, bioproject, organism, geographic location, etc.). You can extract this information without downloading from NCBI:

```bash
metaquest extract_branchwater_metadata --branchwater-folder /path/to/branchwater/files --metadata-folder metadata
```

This creates a simplified metadata file with the information available in the Branchwater CSVs.

#### Option B: Download Full Metadata from NCBI

For more comprehensive metadata:

```bash
metaquest download_metadata --matches_folder matches --metadata_folder metadata --threshold 0.95 --email your@email.com
metaquest parse_metadata --metadata_folder metadata --metadata_table_file parsed_metadata.txt
```

### 4. Continue with Standard MetaQuest Analysis

Once you have the containment data and metadata, you can continue with the standard MetaQuest analysis pipeline:

```bash
# Summarize containment results
metaquest parse_containment --file_format branchwater --matches_folder matches --parsed_containment_file parsed_containment.txt --summary_containment_file summary_containment.txt

# Analyze metadata attributes 
metaquest check_metadata_attributes --file-path parsed_metadata.txt --output-file metadata_counts.txt

# Count genome occurrences
metaquest count_metadata --summary-file parsed_containment.txt --metadata-file parsed_metadata.txt --metadata-column Sample_Scientific_Name --threshold 0.95 --output-file genome_counts.txt

# Visualize results
metaquest plot_containment --file_path parsed_containment.txt --column max_containment --plot_type rank --save_format png
metaquest plot_metadata_counts --file_path genome_counts.txt --plot_type bar --save_format png
```

### 5. Download Raw Data (Optional)

If you want to download the raw sequence data for specific accessions:

```bash
# Create a file with accessions you want to download
grep -v "^#" parsed_containment.txt | awk '$3 > 0.95 {print $1}' > high_containment_accessions.txt

# Download the data
metaquest download_sra --accessions_file high_containment_accessions.txt --fastq_folder fastq
```

## Example Workflow

```bash
# 1. Get Branchwater files (manual step)
# Download files from https://branchwater.jgi.doe.gov/

# 2. Process Branchwater files
metaquest use_branchwater --branchwater-folder branchwater_downloads --matches-folder matches

# 3. Extract metadata from Branchwater files
metaquest extract_branchwater_metadata --branchwater-folder branchwater_downloads --metadata-folder metadata

# 4. Parse containment data
metaquest parse_containment --file_format branchwater --matches_folder matches --parsed_containment_file parsed_containment.txt --summary_containment_file summary_containment.txt

# 5. Analyze and visualize
metaquest count_metadata --summary-file parsed_containment.txt --metadata-file metadata/branchwater_metadata.txt --metadata-column Sample_Scientific_Name --threshold 0.95 --output-file species_counts.txt
metaquest plot_metadata_counts --file_path species_counts.txt --plot_type bar --save_format png
```

## Branchwater Format vs. Mastiff Format

### Branchwater CSV Format
```
acc,containment,cANI,biosample,bioproject,assay_type,collection_date_sam,geo_loc_name_country_calc,organism,lat_lon
```

### Mastiff CSV Format
```
SRA accession,containment,similarity,query_name,query_md5,status
```

The MetaQuest code has been updated to automatically detect and handle both formats when using the `--file_format` parameter.