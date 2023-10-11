# metaquest

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


## Usage

### 1. Downloading Test Genome
To get started, you can download a test genome using the `download-test-genome` command. This command fetches a sample genome from NCBI based on a predefined accession number.

```bash
metaquest download_test_genome
```

### 2. Running Mastiff

After acquiring the genome, you can run the `mastiff` command to search for matches in the SRA.

```bash
metaquest mastiff --genomes-folder genomes --matches-folder matches
```

* `genomes-folder`: The directory where genome files are located.
* `matches-folder`: The directory where the results will be saved.

### 3. Summarizing Results

After the `mastiff` run, you can summarize the results using the `summarize` command. This will generate a summary file and a containment file.

```bash
metaquest parse_containment --matches_folder matches --parsed_containment_file parsed_containment.txt --summary_containment_file summary_containment.txt --step_size 0.05
```

*Example output:* summary.txt and containment.txt

### 4. Downloading Metadata

To get additional information about each SRA dataset, you can download metadata using the `download-metadata` command.

```bash
metaquest download_metadata --matches_folder matches --metadata_folder metadata --threshold 0.95 --email [EMAIL]
```

* `matches_folder`: Directory containing match files.
* `metadata_folder`: Directory where the metadata files will be saved.
* `threshold`: Only consider matches with containment above this threshold.

### 5. Parsing Metadata

Once the metadata is downloaded, you can parse it to generate a more concise and readable format.

```bash
metaquest parse_metadata --metadata_folder metadata --metadata_table_file parsed_metadata.txt
```

*Example output:* parsed_metadata.txt

### 6. Check metadata attribures

This step helps in understanding the distribution of metadata attributes.

```bash
metaquest check_metadata_attributes --file-path parsed_metadata.txt --output-file  parsed_metadata_overview.txt
```

*Example output:* parsed_metadata_overview.txt



### 7. Genome Count

This step helps in understanding the distribution of genomes across different datasets.

```bash
metaquest genome_count --summary-file summary.txt --metadata-file parsed_metadata.txt --metadata-column Sample_Scientific_Name --threshold 0.95  --output-file genome_counts.txt
```

*Example output:* genome_counts.txt

### 7. Single Sample Analysis

To analyze a single sample from the summary, you can use the `single_sample` command.

```bash
metaquest single_sample --summary-file summary.txt --metadata-file parsed_metadata.txt --summary-column GCF_000008985.1 --metadata-column Sample_Scientific_Name --threshold 0.95
```

*Example output:* collected_stats.txt



## Contributing

We welcome contributions to `metaquest`! Whether you want to report a bug, suggest a feature, or contribute code, your input is valuable. Here's how to get started:

1. **Fork the Repository**: Create your own fork of the `metaquest` repository.
2. **Clone Your Fork**: Clone your fork to your local machine and set the upstream repository.
3. **Create a New Branch**: Make a new branch for your feature or bugfix.
4. **Make Your Changes**: Implement your feature or fix the bug and commit your changes.
5. **Push to Your Fork**: Push your changes to your fork on GitHub.
6. **Create a Pull Request**: From your fork, open a new pull request in the `metaquest` repository.


