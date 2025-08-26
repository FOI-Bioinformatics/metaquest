# MetaQuest

[![CI](https://github.com/FOI-Bioinformatics/metaquest/actions/workflows/ci.yml/badge.svg)](https://github.com/FOI-Bioinformatics/metaquest/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> A comprehensive toolkit for analyzing metagenomic datasets based on genome containment

MetaQuest is a command-line tool designed to help researchers search through SRA (Sequence Read Archive) datasets to find containment of specified genomes. By analyzing metadata information, it provides insights into where different species may be found across environmental and clinical samples.

## 🚀 Key Features

- **Branchwater Integration**: Direct processing of containment files from the Branchwater platform
- **Metadata Analysis**: Comprehensive SRA metadata downloading and parsing from NCBI
- **Visualization Tools**: Built-in plotting capabilities for containment and metadata analysis
- **Scalable Processing**: Parallel download and processing capabilities
- **Flexible Formats**: Support for multiple file formats (Branchwater, Mastiff)
- **Security**: Input validation and secure subprocess handling

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install metaquest
```

### From Source
```bash
git clone https://github.com/FOI-Bioinformatics/metaquest.git
cd metaquest
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/FOI-Bioinformatics/metaquest.git
cd metaquest
pip install -e ".[dev]"
```

## 🔧 Requirements

- Python 3.8+
- Dependencies: pandas, matplotlib, seaborn, biopython, cartopy
- Optional tools: fasterq-dump (for SRA downloads), megahit/flye (for assembly)

## 📖 Quick Start

### 1. Download Test Data
```bash
metaquest download_test_genome --output-folder genomes
```

### 2. Process Branchwater Files
Visit [https://branchwater.jgi.doe.gov/](https://branchwater.jgi.doe.gov/) to download containment files, then:

```bash
metaquest use_branchwater --branchwater-folder branchwater_files --matches-folder matches
```

### 3. Parse Containment Data
```bash
metaquest parse_containment --matches-folder matches --step-size 0.1
```

### 4. Visualize Results
```bash
metaquest plot_containment --file-path parsed_containment.txt --plot-type rank
```

## 📚 Comprehensive Usage

### Core Workflow Commands

#### Branchwater Processing
```bash
# Process downloaded Branchwater files
metaquest use_branchwater --branchwater-folder /path/to/files --matches-folder matches

# Extract basic metadata from Branchwater files
metaquest extract_branchwater_metadata --branchwater-folder /path/to/files --metadata-folder metadata
```

#### Containment Analysis
```bash
# Parse containment data with custom thresholds
metaquest parse_containment --matches-folder matches \\
    --parsed-containment-file results.txt \\
    --summary-containment-file summary.txt \\
    --step-size 0.05 \\
    --file-format branchwater
```

#### Metadata Operations
```bash
# Download comprehensive metadata from NCBI
metaquest download_metadata --email your.email@domain.com \\
    --matches-folder matches \\
    --metadata-folder metadata \\
    --threshold 0.95

# Parse downloaded metadata
metaquest parse_metadata --metadata-folder metadata \\
    --metadata-table-file parsed_metadata.txt

# Count metadata by attributes
metaquest count_metadata --summary-file parsed_containment.txt \\
    --metadata-file parsed_metadata.txt \\
    --metadata-column Sample_Scientific_Name \\
    --threshold 0.95 \\
    --output-file counts.txt
```

#### Single Sample Analysis
```bash
metaquest single_sample --summary-file parsed_containment.txt \\
    --metadata-file parsed_metadata.txt \\
    --summary-column GCF_000008985.1 \\
    --metadata-column Sample_Scientific_Name \\
    --threshold 0.95 \\
    --top-n 100
```

#### SRA Data Download
```bash
# Download raw SRA data
metaquest download_sra --accessions-file accessions.txt \\
    --fastq-folder fastq \\
    --num-threads 8 \\
    --max-workers 4 \\
    --max-retries 3

# Assembly datasets (requires megahit/flye)
metaquest assemble_datasets --data-files fastq/*.fastq \\
    --output-file assembled_dataset.fasta
```

### Visualization Commands

#### Containment Plots
```bash
# Rank plot of containment scores
metaquest plot_containment --file-path parsed_containment.txt \\
    --column max_containment \\
    --plot-type rank \\
    --save-format png \\
    --threshold 0.05

# Histogram of containment distribution
metaquest plot_containment --file-path parsed_containment.txt \\
    --plot-type histogram \\
    --save-format pdf
```

**Available plot types**: `rank`, `histogram`, `box`, `violin`

#### Metadata Plots
```bash
# Bar chart of metadata counts
metaquest plot_metadata_counts --file-path counts.txt \\
    --plot-type bar \\
    --save-format png

# Pie chart for categorical data
metaquest plot_metadata_counts --file-path counts.txt \\
    --plot-type pie \\
    --title "Species Distribution"
```

**Available plot types**: `bar`, `pie`, `radar`

## 🏗️ Architecture

MetaQuest follows a layered, plugin-based architecture:

```
CLI Layer (commands, argument parsing)
    ↓
Core Logic Layer (models, validation, exceptions)
    ↓
Data Layer (file I/O, external APIs) ←→ Plugin System (formats, visualizers)
    ↓
External Systems (SRA, NCBI, Branchwater)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🔧 Development

### Setup Development Environment
```bash
# Clone and install in development mode
git clone https://github.com/FOI-Bioinformatics/metaquest.git
cd metaquest
pip install -e ".[dev]"
```

### Code Quality Tools
```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test

# Check all (format, lint, test)
make check
```

### Available Make Commands
- `make test` - Run tests with coverage
- `make lint` - Lint code with flake8
- `make format` - Format code with black
- `make check` - Check formatting, linting, and types
- `make build` - Build distribution packages
- `make clean` - Clean build artifacts

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=metaquest --cov-report=html

# Run specific test
pytest tests/test_basic.py::test_version
```

## 📊 Examples

Check out the `doc/` directory for detailed workflow examples:
- [Branchwater Workflow](doc/branchwater_workflow.md)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### How to Contribute
1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
4. **Make** your changes and add tests
5. **Run** code quality checks (`make check`)
6. **Commit** your changes (`git commit -m 'Add amazing feature'`)
7. **Push** to your branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to new functions
- Include docstrings for public methods
- Write tests for new functionality
- Ensure all CI checks pass

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/FOI-Bioinformatics/metaquest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FOI-Bioinformatics/metaquest/discussions)
- **Documentation**: [Architecture Guide](ARCHITECTURE.md)

## 🏷️ Citation

If you use MetaQuest in your research, please cite:

```bibtex
@software{metaquest,
  title={MetaQuest: A toolkit for analyzing metagenomic datasets based on genome containment},
  author={Andreas Sjödin and contributors},
  url={https://github.com/FOI-Bioinformatics/metaquest},
  version={0.3.1},
  year={2024}
}
```

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

---

**Made with ❤️ by the FOI Bioinformatics Team**