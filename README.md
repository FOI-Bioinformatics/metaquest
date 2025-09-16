# MetaQuest

**MetaQuest** is a comprehensive command-line bioinformatics toolkit for analyzing metagenomic datasets based on genome containment. The software processes Branchwater CSV files, downloads SRA metadata from NCBI, and provides advanced visualization and analysis capabilities including diversity analysis, interactive plotting, and taxonomic validation.

## Features

- **Branchwater Integration**: Process and analyze containment data from JGI Branchwater
- **Intelligent SRA Management**: Advanced downloading with resume capability, quality profiling, and statistical reporting
- **Diversity Analysis**: Calculate alpha/beta diversity metrics with statistical testing
- **Interactive Visualizations**: Create dynamic plots (PCA, heatmaps, diversity comparisons)
- **Taxonomic Validation**: Validate species names against NCBI taxonomy database
- **Plugin Architecture**: Extensible format handlers and visualization plugins
- **Robust Implementation**: Type hints, comprehensive testing (635 tests), numerical stability

## Installation

### Quick Start (Recommended)
```bash
git clone https://github.com/FOI-Bioinformatics/MetaQuest.git
cd MetaQuest
make dev-install  # Installs with all development dependencies
```

### Alternative Installation
```bash
# Traditional approach (still supported)
pip install -r requirements.txt
python setup.py install
```

### Development Commands
```bash
make help           # Show all available commands
make test          # Run tests with coverage (635 tests passing)
make lint          # Run code quality checks  
make check         # Full quality validation
make clean         # Clean build artifacts
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

## Advanced SRA Operations

### Intelligent SRA Downloading

Download SRA datasets with intelligent resume capability and bandwidth optimization:

```bash
# Intelligent download with resume capability
metaquest sra-download-intelligent \
    --accessions-file accessions.txt \
    --output-dir sra_downloads \
    --max-parallel-downloads 4 \
    --max-bandwidth-mbps 100 \
    --resume

# Dry run to estimate download time and requirements
metaquest sra-download-intelligent \
    --accessions-file accessions.txt \
    --dry-run
```

### SRA Quality Profiling

Generate comprehensive quality profiles for downloaded SRA datasets:

```bash
# Profile multiple datasets with detailed reports
metaquest sra-profile-quality \
    --accessions-file accessions.txt \
    --fastq-dir sra_downloads \
    --output-dir quality_profiles \
    --detailed-reports

# Profile single dataset
metaquest sra-profile-quality \
    --accession SRR123456 \
    --fastq-dir sra_downloads \
    --include-contamination
```

### Interactive SRA Dashboards

Generate interactive HTML dashboards for SRA analysis:

```bash
# Comprehensive dashboard
metaquest sra-dashboard \
    --accessions-file accessions.txt \
    --output-dir dashboards \
    --title "Project SRA Analysis" \
    --dashboard-type full

# Quality analysis dashboard only
metaquest sra-dashboard \
    --accessions-file accessions.txt \
    --dashboard-type quality
```

### Comparative SRA Analysis

Perform statistical comparisons between SRA dataset groups:

```bash
# Compare treatment vs control groups
metaquest sra-compare \
    --groups-file comparison_groups.json \
    --fastq-dir sra_downloads \
    --statistical-tests \
    --generate-report
```

Example groups file format:
```json
{
  "Treatment_Group": ["SRR123456", "SRR123457"],
  "Control_Group": ["SRR789012", "SRR789013"]
}
```

### Enhanced SRA Features

For additional SRA capabilities with technology detection:

```bash
# Get detailed dataset information before downloading
metaquest sra_info \
    --accessions-file accessions.txt \
    --email your.email@domain.com \
    --output-report sra_analysis.csv

# Enhanced download with technology detection
metaquest sra_download \
    --accessions-file accessions.txt \
    --fastq-folder fastq \
    --email your.email@domain.com \
    --num-threads 8 \
    --max-workers 4
```

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

## Advanced Analysis Features

### Diversity Analysis
Calculate comprehensive diversity metrics for your metagenomic datasets:

```bash
# Calculate alpha and beta diversity with PERMANOVA
metaquest diversity_analysis \
    --abundance-file abundance_matrix.csv \
    --metadata-file sample_metadata.csv \
    --alpha-metrics shannon simpson chao1 \
    --beta-metric bray_curtis \
    --permanova-formula "treatment + site"
```

### Interactive Visualizations
Create dynamic, browser-based plots for data exploration:

```bash
# Interactive PCA plot
metaquest interactive_plot \
    --data-file abundance_matrix.csv \
    --metadata-file metadata.csv \
    --plot-type pca \
    --color-by treatment \
    --output-file pca_plot.html

# Interactive heatmap
metaquest interactive_plot \
    --data-file abundance_matrix.csv \
    --plot-type heatmap \
    --title "Species Abundance Heatmap"

# Diversity comparison plots  
metaquest interactive_plot \
    --data-file abundance_matrix.csv \
    --metadata-file metadata.csv \
    --plot-type diversity \
    --color-by treatment_group
```

### Taxonomic Validation
Validate species names against NCBI taxonomy database:

```bash
# Validate species from text file
metaquest validate_taxonomy \
    --species-file species_list.txt \
    --email your.email@domain.com \
    --output-file validation_results.csv

# Validate from CSV with specific column
metaquest validate_taxonomy \
    --species-file data.csv \
    --species-column organism_name \
    --email your.email@domain.com
```

### Taxonomic Summary Analysis
Generate comprehensive taxonomic summaries at multiple levels:

```bash
metaquest taxonomic_summary \
    --abundance-file abundance_matrix.csv \
    --taxonomy-file validation_results.csv \
    --levels phylum class order family genus \
    --output-dir taxonomic_summaries
```

## Documentation

For comprehensive documentation including advanced features and technical details, see the [docs/](docs/) directory:

- **[Enhanced SRA Features](docs/SRA_ENHANCED_FEATURES.md)** - Advanced SRA downloading with technology detection and statistics
- **[Branchwater Workflow](docs/branchwater_workflow.md)** - Detailed workflow guide for branchwater functionality  
- **[Architecture](docs/ARCHITECTURE.md)** - Technical architecture and design decisions

## Development & Testing

MetaQuest follows modern Python development practices with comprehensive testing and quality assurance.

### Current Status
- **Test Coverage**: 53% overall (substantial improvement from previous versions)
- **CLI Commands**: 100% coverage ✅
- **Data Layer**: 96-98% coverage for core modules ✅
- **Core Processing**: 92-99% coverage ✅  
- **Code Quality**: All linting checks passing ✅

### Recent Enhancements
Significant improvements have been implemented across the codebase:

- **Intelligent SRA Package**: Complete implementation of next-generation SRA capabilities including intelligent downloads with resume functionality, comprehensive quality profiling, and interactive dashboard generation
- **Test Coverage Improvements**: Comprehensive testing infrastructure established with 53%+ overall coverage
- **Architecture Refinement**: Orphan code removal and clean separation of concerns between data layer and advanced SRA features
- **Quality Assurance**: All linting violations resolved and formatting standards enforced

### Development Workflow
```bash
# Set up development environment
make dev-install

# Run quality checks before committing
make check          # Format, lint, and type check
make test          # Run tests with coverage report
make pipeline      # Full integration test

# View available commands
make help
```

### Testing Structure
- **Comprehensive Test Suite**: 635 tests covering CLI, data processing, and core functionality
- **Numerical Stability**: Edge cases in statistical computing properly handled
- **Integration Tests**: End-to-end workflow validation via `local_test.sh`
- **Mock Architecture**: NCBI APIs, file operations, and system calls systematically mocked
- **Quality Assurance**: Automated formatting, linting, and type checking with zero warnings

### Architecture Highlights
- **Layered Architecture**: CLI, Core, Data, Processing, Visualization layers
- **Advanced SRA Package**: Specialized module for intelligent SRA operations (`metaquest.sra`)
- **Plugin System**: Extensible format handlers and visualizers  
- **Command Registry**: Modular CLI command architecture
- **Modern Packaging**: Uses `pyproject.toml` with backward compatibility

## Contributing

We welcome contributions to MetaQuest. Whether you want to report a bug, suggest a feature, or contribute code, your input is valuable.

### Quick Start for Contributors
```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/MetaQuest.git
cd MetaQuest

# 2. Set up development environment  
make dev-install

# 3. Create a feature branch
git checkout -b feature/my-new-feature

# 4. Make your changes and test
make check          # Ensure code quality
make test          # Run test suite
make pipeline      # Integration tests

# 5. Commit and push
git commit -m "Add new feature"
git push origin feature/my-new-feature

# 6. Create a Pull Request on GitHub
```

### Contribution Guidelines
- **Code Quality**: All contributions must pass `make check` (formatting, linting, type checking)
- **Testing**: New features should include tests with proper edge case handling
- **Documentation**: Update relevant documentation for new features
- **Architecture**: Follow existing patterns (see `CLAUDE.md` for detailed guidance)

### Current Priority Areas
Help us improve MetaQuest by contributing to these areas:
- **Visualization Testing**: Improve coverage in plotting and reporting modules
- **Processing Layer**: Enhance containment analysis and statistical processing tests
- **Plugin Development**: Add new format handlers or visualization plugins
- **Performance Optimization**: Large dataset handling and memory efficiency
- **Documentation**: Expand workflow examples and API documentation

For detailed development guidelines, see [CLAUDE.md](CLAUDE.md).