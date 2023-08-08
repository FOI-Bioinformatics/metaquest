# METAQUEST: Metagenomic Table Query and Extraction System for Public Datasets

## Introduction

METAQUEST is a command-line tool designed to fetch, analyze, and visualize data from metagenomic datasets. It can run various operations such as fetching metadata, summarizing matches, generating plots, downloading sequence read archives (SRA) data, and assembling datasets.

## Requirements

- Python 3.6 or later
- BioPython
- upsetplot
- pandas
- numpy
- lxml

These dependencies can be installed automatically when you install METAQUEST.

## Installation

First, download the METAQUEST software. Then navigate to the directory containing the `setup.py` file and run the following commands:

```bash
python setup.py sdist bdist_wheel
pip install .
```

## Usage

METAQUEST provides several commands, each corresponding to a different function of the software:

1. **mastiff**: Runs the 'mastiff' command on each '.fasta' file in the 'genomes_folder'.

    Example: `metaquest mastiff --genomes-folder genomes --matches-folder matches`

2. **summarize**: Reads the '.csv' files in the 'matches_folder' and creates a summary dataframe.

    Example: `metaquest summarize --matches-folder matches --summary-file SRA-summary.txt`

3. **plot-upset**: Reads the summary file and generates an UpSet plot.

    Example: `metaquest plot-upset --summary-file SRA-summary.txt --upset-plot-file upset_plot.png`

4. **download-metadata**: Downloads the metadata for each SRA accession in the '.csv' files in the 'matches_folder'.

    Example: `metaquest download-metadata --matches-folder matches --metadata-folder metadata --dry-run`

5. **parse-metadata**: Parses the metadata for each '_metadata.xml' file in the 'metadata_folder'.

    Example: `metaquest parse-metadata --metadata-folder metadata --metadata-table-file metadata_table.txt`

6. **download-sra**: Downloads the SRA data for each unique SRA accession in the '.csv' files in the 'matches_folder'.

    Example: `metaquest download-sra --matches-folder matches --fastq-folder fastq`

7. **assemble**: Assembles datasets for each '.fastq.gz' file in the 'fastq_folder'.

    Example: `metaquest assemble`

If you do not provide any command, METAQUEST will print a usage message.

Each command also accepts arguments to customize its behavior. To see a list of available arguments for a command, use the `-h` or `--help` option with the command, like so: `metaquest command --help`.

## Conclusion

METAQUEST is a powerful tool for working with metagenomic datasets. It provides a range of functionalities and is customizable via command-line arguments. This makes it a versatile tool for many bioinformatics workflows.
