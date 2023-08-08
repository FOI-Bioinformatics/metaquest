I understand now. Let's expand on the examples by explicitly stating the default values in the command examples. 

---

# metaquest

`metaquest` is a command-line tool designed to help users search through all SRA datasets to find containment of specified genomes. By analyzing the metadata information, it provides insights into where different species may be found.

## Installation

```bash
pip install metaquest
```

## Usage

### Running mastiff on genome data

```bash
metaquest mastiff --genome-folder genomes --output-folder matches
```

### Summarizing the data

```bash
metaquest summarize --matches-folder matches --output-file summary.txt
```

### Generating UpSet plot from summary

```bash
metaquest plot-upset --summary-file summary.txt --upset-plot-file upset.png
```

### Generating a heatmap

```bash
metaquest plot_heatmap --summary-file summary.txt --heatmap-file heatmap.png --threshold 0.1
```

### Downloading metadata for each SRA accession

```bash
metaquest download-metadata --matches-folder matches --output-folder metadata
```

### Parsing metadata

```bash
metaquest parse-metadata --metadata-folder metadata
```

### Counting occurrences for a single sample

```bash
metaquest single_sample --sample-file sample.csv
```

### Collecting genome counts

```bash
metaquest genome_count --matches-folder matches --output-file genome_count.txt
```

### Downloading SRA data

```bash
metaquest download-sra --matches-folder matches --output-folder fastq
```

### Assembling datasets

```bash
metaquest assemble --fastq-folder fastq --output-folder assemblies
```

## Contributing

We welcome contributions to `metaquest`! Whether you want to report a bug, suggest a feature, or contribute code, your input is valuable. Here's how to get started:

1. **Fork the Repository**: Create your own fork of the `metaquest` repository.
2. **Clone Your Fork**: Clone your fork to your local machine and set the upstream repository.
3. **Create a New Branch**: Make a new branch for your feature or bugfix.
4. **Make Your Changes**: Implement your feature or fix the bug and commit your changes.
5. **Push to Your Fork**: Push your changes to your fork on GitHub.
6. **Create a Pull Request**: From your fork, open a new pull request in the `metaquest` repository.

Please ensure your code adheres to our coding conventions and standards. Before submitting a pull request, make sure all tests pass.

---

This should provide a clearer picture for users who want to understand the default values used in each command.
