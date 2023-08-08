# MetaQuest

**MetaQuest** is a unique tool designed to explore the Sequence Read Archive (SRA) datasets for containment of specified genomes. With the immense amount of data available in the SRA, it's crucial to discern where specific species may be found. MetaQuest simplifies this process by analyzing the metadata information, revealing the distribution and prevalence of particular genomes across multiple datasets.

## Features

- **Deep Dive into SRA**: Query the SRA database for genomes of interest efficiently.
- **Summarize Data**: Generate concise reports detailing the containment of genomes across datasets.
- **Advanced Visualizations**: Create UpSet plots, heatmaps, PCA, and UMAP to provide insights into data distribution.

## Installation

_(Installation steps or instructions.)_

## Usage

The primary command-line interface for MetaQuest:

```
metaquest.py <sub-command> [options]
```

### Sub-commands:

#### 1. `mastiff`
Runs mastiff on the genome data in the specified directory.
```
metaquest.py mastiff [options]
```

#### 2. `summarize`
Summarizes the data from the `.csv` files in the matches directory.
```
metaquest.py summarize [options]
```

#### 3. `plot-upset`
Generates an UpSet plot from the summary file.
```
metaquest.py plot-upset [options]
```

#### 4. `plot_heatmap`
Creates a heatmap from the summary file with a user-defined threshold.
```
metaquest.py plot_heatmap [options]
```

#### 5. `download-metadata`
Downloads metadata for each SRA accession in the `.csv` files in the matches directory.
```
metaquest.py download-metadata [options]
```

#### 6. `parse-metadata`
Parses metadata for each `*_metadata.xml` file in the designated directory.
```
metaquest.py parse-metadata [options]
```

#### 7. `single_sample`
Counts the occurrences of unique values in a column for an individual sample.
```
metaquest.py single_sample [options]
```

#### 8. `genome_count`
Collates genome counts and exports a table with sample files.
```
metaquest.py genome_count [options]
```

#### 9. `download-sra`
Downloads SRA data for each unique SRA accession in the `.csv` files in the matches directory.
```
metaquest.py download-sra [options]
```

#### 10. `assemble`
Assembles datasets for each `.fastq.gz` file in the fastq folder.
```
metaquest.py assemble [options]
```

_For each sub-command, use the `-h` or `--help` flag to get a detailed list of options._

## Contributing

We welcome contributions to `metaquest`! Here's how you can help:

1. **Fork** the repository on GitHub.
2. **Clone** your fork and create a new branch: `git checkout -b my-feature-branch`.
3. Make your changes, **commit** them, and **push** the branch.
4. Open a **pull request** against the `main` branch of the `metaquest` repository.
5. Ensure your changes pass all checks and address any feedback provided by the maintainers.
6. Once approved, your contribution will be merged. Thank you for contributing!
