# MetaQuest

`metaquest` is a command-line tool designed to help users search through all SRA datasets to find containment of specified genomes. By analyzing the metadata information, it provides insights into where different species may be found.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FOI-Bioinformatics/MetaQuest.git
cd MetaQuest
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Install MetaQuest:
```bash
python setup.py install
```

## Usage with Branchwater

### 1. Getting Containment Files from Branchwater

First, visit [https://branchwater.jgi.doe.gov/](https://branchwater.jgi.doe.gov/) to search and download containment files for your genomes of interest. Save these CSV files to a designated folder.

### 2. Process Branchwater Files

Process the downloaded files to prepare them for the MetaQuest pipeline:

```bash
metaquest use_branchwater --branchwater-folder /path/to/branchwater/files --matches-folder matches
```

* `branchwater-folder`: The directory where Branchwater CSV files are located.
* `matches-folder`: The directory where the processed files will be saved.

### 3. Extract Basic Metadata from Branchwater Files (Optional)

You can extract basic metadata directly from Branchwater CSV files without downloading from NCBI:

```bash
metaquest extract_branchwater_metadata --branchwater-folder /path/to/branchwater/files --metadata-folder metadata
```

### 4. Summarizing Results

After processing the Branchwater files, you can summarize the results:

```bash
metaquest parse_containment --matches-folder matches --parsed-containment-file parsed_containment.txt --summary-containment-file summary_containment.txt --step-size 0.05 --file-format branchwater
```

*Example output:* summary.txt and containment.txt

### 5. Downloading Metadata from NCBI (Alternative to Step 3)

For more comprehensive metadata, you can download it from NCBI:

```bash
metaquest download_metadata --matches-folder matches --metadata-folder metadata --threshold 0.95 --email [EMAIL]
```

* `matches_folder`: Directory containing match files.
* `metadata_folder`: Directory where the metadata files will be saved.
* `threshold`: Only consider matches with containment above this threshold.

### 6. Parsing Metadata

Once the metadata is downloaded, you can parse it to generate a more concise and readable format:

```bash
metaquest parse_metadata --metadata-folder metadata --metadata-table-file parsed_metadata.txt
```

*Example output:* parsed_metadata.txt

### 7. Check Metadata Attributes

This step helps in understanding the distribution of metadata attributes:

```bash
metaquest check_metadata_attributes --file-path parsed_metadata.txt --output-file parsed_metadata_overview.txt
```

*Example output:* parsed_metadata_overview.txt

### 8. Genome Count

This step helps in understanding the distribution of genomes across different datasets:

```bash
metaquest count_metadata --summary-file parsed_containment.txt --metadata-file parsed_metadata.txt --metadata-column Sample_Scientific_Name --threshold 0.95 --output-file genome_counts.txt
```

*Example output:* genome_counts.txt

### 9. Single Sample Analysis

To analyze a single sample from the summary, you can use the `single_sample` command:

```bash
metaquest single_sample --summary-file parsed_containment.txt --metadata-file parsed_metadata.txt --summary-column GCF_000008985.1 --metadata-column Sample_Scientific_Name --threshold 0.95
```

### 10. Download SRA Data

To download the raw SRA data for accessions that match your criteria:

```bash
metaquest download_sra --accessions-file accessions.txt --fastq-folder fastq --num-threads 8 --max-workers 4
```

The accessions file should contain one SRA accession per line.

## Visualizing Results

### Plotting Containment Data

Plot the distribution of containment scores:

```bash
metaquest plot_containment --file-path parsed_containment.txt --column max_containment --plot-type rank --save-format png --threshold 0.05
```

Available plot types: rank, histogram, box, violin

### Plotting Metadata Counts

Visualize the distribution of metadata attributes:

```bash
metaquest plot_metadata_counts --file-path counts_Sample_Scientific_Name.txt --plot-type bar --save-format png
```

Available plot types: bar, pie, radar

## Contributing

We welcome contributions to `metaquest`! Whether you want to report a bug, suggest a feature, or contribute code, your input is valuable. Here's how to get started:

1. **Fork the Repository**: Create your own fork of the `metaquest` repository.
2. **Clone Your Fork**: Clone your fork to your local machine and set the upstream repository.
3. **Create a New Branch**: Make a new branch for your feature or bugfix.
4. **Make Your Changes**: Implement your feature or fix the bug and commit your changes.
5. **Push to Your Fork**: Push your changes to your fork on GitHub.
6. **Create a Pull Request**: From your fork, open a new pull request in the `metaquest` repository.