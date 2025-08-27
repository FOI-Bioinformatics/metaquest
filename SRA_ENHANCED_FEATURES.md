# Enhanced SRA Functionality in MetaQuest

MetaQuest now includes comprehensive, robust SRA (Sequence Read Archive) downloading capabilities with advanced features for handling all major sequencing technologies.

## 🚀 Key Features

### 📊 Pre-Download Information
- **Dataset Analysis**: Get detailed information about datasets before downloading
- **Technology Detection**: Automatic detection of Illumina, Nanopore, PacBio, and other platforms
- **Size Estimation**: Accurate storage requirements and download time estimation
- **Metadata Fetching**: Comprehensive dataset information from NCBI

### 🔬 Technology Support
- **Illumina**: Paired-end and single-end, all instruments (HiSeq, NovaSeq, MiSeq, etc.)
- **Oxford Nanopore**: MinION, GridION, PromethION long-read sequencing
- **PacBio**: Sequel, RS systems for long-read sequencing
- **Legacy Platforms**: 454, Ion Torrent, and other historical platforms

### 📈 Comprehensive Statistics
- **Read Statistics**: Count, length distribution, N50, GC content
- **Quality Metrics**: Average quality scores, quality distribution
- **File Validation**: Integrity checking and paired-end validation
- **Technology-Specific Stats**: Optimized calculations per platform

### 🛡️ Robust Error Handling
- **Retry Logic**: Automatic retry for failed downloads
- **Resume Capability**: Continue interrupted downloads
- **Blacklisting**: Skip problematic accessions
- **Validation**: Post-download integrity checks

## 📋 New CLI Commands

### `sra_info` - Dataset Information
Get detailed information about SRA datasets before downloading:

```bash
# Basic usage
metaquest sra_info --accessions-file accessions.txt --email your@email.com

# With API key for faster access
metaquest sra_info --accessions-file accessions.txt --email your@email.com --api-key YOUR_KEY

# Custom output report
metaquest sra_info --accessions-file accessions.txt --email your@email.com --output-report my_report.csv

# With bandwidth estimation for download time
metaquest sra_info --accessions-file accessions.txt --email your@email.com --bandwidth-mbps 50.0
```

**Output:**
- Total datasets and estimated size
- Technology distribution (Illumina, Nanopore, PacBio)
- Platform and instrument breakdown
- Size statistics and download time estimation
- Detailed CSV report with all metadata

### `sra_download` - Enhanced Downloading
Download SRA datasets with advanced features:

```bash
# Basic enhanced download
metaquest sra_download --accessions-file accessions.txt --email your@email.com --fastq-folder data/

# With technology detection and optimization
metaquest sra_download --accessions-file accessions.txt --email your@email.com --fastq-folder data/ --verify-tools

# Parallel downloads with retry
metaquest sra_download --accessions-file accessions.txt --email your@email.com --fastq-folder data/ --max-workers 8 --num-threads 6

# Dry run to preview
metaquest sra_download --accessions-file accessions.txt --email your@email.com --fastq-folder data/ --dry-run

# With blacklist to skip problematic accessions
metaquest sra_download --accessions-file accessions.txt --email your@email.com --fastq-folder data/ --blacklist failed.txt
```

**Features:**
- Technology-specific optimizations
- Intelligent file naming (R1/R2 for paired-end, _long for long reads)
- Parallel downloads with progress tracking
- Comprehensive error reporting
- Download resume capability

### `sra_stats` - Comprehensive Statistics
Calculate detailed statistics for downloaded datasets:

```bash
# Generate statistics for all datasets
metaquest sra_stats --fastq-folder data/

# Custom output file
metaquest sra_stats --fastq-folder data/ --output-report stats.csv

# Specific accessions only
metaquest sra_stats --fastq-folder data/ --accessions SRR123456 SRR789012
```

**Calculates:**
- Read counts and base counts
- Length distributions (min, max, average, N50)
- GC content analysis
- Quality score statistics
- Technology-specific metrics

### `sra_validate` - Dataset Validation
Validate integrity of downloaded datasets:

```bash
# Basic validation
metaquest sra_validate --fastq-folder data/

# Check paired-end consistency
metaquest sra_validate --fastq-folder data/ --check-pairs

# Validate specific datasets
metaquest sra_validate --fastq-folder data/ --accessions SRR123456
```

**Checks:**
- File existence and non-zero size
- FASTQ format validation
- Paired-end file consistency
- Read count matching
- Basic integrity verification

## 📁 File Organization

The enhanced SRA downloader organizes files in a standardized way:

```
fastq/
├── SRR123456/                 # Illumina paired-end
│   ├── SRR123456_R1.fastq.gz  # (or .fastq depending on source)
│   └── SRR123456_R2.fastq.gz
├── SRR789012/                 # Illumina single-end
│   └── SRR789012_R1.fastq.gz
├── SRR345678/                 # Nanopore long reads
│   └── SRR345678_long.fastq
└── SRR901234/                 # PacBio long reads
    └── SRR901234_long.fastq
```

**Note**: File extensions (.fastq or .fastq.gz) depend on the source data format.

## 🔧 Technology Detection

The system automatically detects sequencing technology based on:

- **Platform**: ILLUMINA, OXFORD_NANOPORE, PACBIO_SMRT
- **Instrument**: HiSeq, NovaSeq, MinION, GridION, Sequel, etc.
- **Strategy**: WGS, AMPLICON, RNA-Seq, etc.
- **Library Layout**: PAIRED, SINGLE

This enables:
- Technology-specific download optimizations
- Appropriate file naming conventions
- Optimized assembly recommendations
- Accurate statistics calculations

## 📊 Detailed Reports

### Metadata Report (CSV)
```csv
accession,title,organism,platform,instrument,strategy,layout,spots,bases,avg_length,size_mb,technology
SRR123456,Sample 1,E. coli,ILLUMINA,HiSeq 2500,WGS,PAIRED,1000000,150000000,150.0,100.5,illumina
SRR789012,Sample 2,E. coli,OXFORD_NANOPORE,MinION,WGS,SINGLE,50000,500000000,10000.0,250.8,nanopore
```

### Statistics Report (CSV)
```csv
accession,num_files,layout,total_reads,total_bases,avg_read_length,n50,gc_content,avg_quality
SRR123456,2,PAIRED,2000000,300000000,150.0,150,45.2,32.5
SRR789012,1,SINGLE,50000,500000000,10000.0,12500,42.8,28.1
```

### Download Report (CSV)
```csv
accession,status,technology,downloaded_reads,downloaded_bases,gc_content,message
SRR123456,Success,illumina,2000000,300000000,45.2,Downloaded 2 files
SRR789012,Success,nanopore,50000,500000000,42.8,Downloaded 1 file (nanopore)
SRR345678,Failed,unknown,0,0,0.0,Download command failed: Network error
```

## ⚡ Performance Optimizations

### Technology-Specific Optimizations
- **Illumina**: `--split-files` for proper R1/R2 separation
- **Long Reads**: `--include-technical` for metadata preservation
- **Parallel Downloads**: Configurable worker threads
- **Memory Management**: Efficient temp file handling

### Network Optimizations
- **Rate Limiting**: NCBI-compliant API access
- **Retry Logic**: Exponential backoff for failed requests
- **Batch Processing**: Efficient metadata fetching
- **Resume Downloads**: Continue interrupted transfers

## 🛠️ Prerequisites

### Required Tools
- **SRA Toolkit**: fasterq-dump, prefetch, vdb-validate
- **Python Dependencies**: biopython, pandas, requests

### Verification
```bash
# Verify tools are installed
metaquest sra_download --verify-tools
```

### NCBI API Access
- **Email**: Required for all NCBI API access
- **API Key**: Optional, but recommended for higher rate limits
- **Registration**: Get API key from https://www.ncbi.nlm.nih.gov/account/settings/

## 📈 Example Workflows

### Complete Analysis Workflow
```bash
# 1. Get dataset information
metaquest sra_info --accessions-file my_accessions.txt --email me@example.com

# 2. Download datasets with enhanced features  
metaquest sra_download --accessions-file my_accessions.txt --email me@example.com --verify-tools

# 3. Calculate comprehensive statistics
metaquest sra_stats --fastq-folder fastq/ --output-report download_stats.csv

# 4. Validate downloaded datasets
metaquest sra_validate --fastq-folder fastq/ --check-pairs

# 5. Continue with MetaQuest analysis...
```

### Large-Scale Download
```bash
# Preview large dataset
metaquest sra_info --accessions-file large_study.txt --email me@example.com

# Download with optimizations
metaquest sra_download \
    --accessions-file large_study.txt \
    --email me@example.com \
    --api-key YOUR_KEY \
    --fastq-folder fastq/ \
    --max-workers 8 \
    --num-threads 8 \
    --temp-folder /fast/tmp \
    --report-file large_study_report.csv
```

### Failed Download Recovery
```bash
# Downloads create failed_accessions.txt automatically
metaquest sra_download \
    --accessions-file fastq/failed_accessions.txt \
    --email me@example.com \
    --fastq-folder fastq/ \
    --force \
    --max-workers 4
```

## 🔍 Troubleshooting

### Common Issues

1. **SRA Tools Not Found**
   ```bash
   # Install SRA toolkit
   conda install -c bioconda sra-tools
   # or
   wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-linux64.tar.gz
   ```

2. **Network Timeouts**
   - Use NCBI API key for higher rate limits
   - Reduce `--max-workers` for slower connections
   - Check firewall settings

3. **Disk Space Issues**
   - Use `sra_info` to estimate space requirements
   - Configure `--temp-folder` on high-speed storage
   - Monitor available space during downloads

4. **Metadata Fetch Failures**
   - Verify email and API key
   - Check NCBI service status
   - Reduce batch sizes for problematic accessions

5. **Validation Failures**
   - Use `--check-pairs` for paired-end validation
   - Check FASTQ file integrity with `sra_validate`
   - Re-download failed files with `--force`

### Performance Tips

1. **Network Optimization**
   - Use API key for 10x faster metadata access
   - Increase `--max-workers` for fast connections
   - Use local SSD for `--temp-folder`

2. **Storage Optimization**
   - Estimate total size with `sra_info` first
   - Use compression-friendly filesystems
   - Consider download location for network storage

3. **Memory Management**
   - Reduce `--num-threads` if memory is limited
   - Monitor temp folder space usage
   - Use `--max-downloads` for testing

The enhanced SRA functionality in MetaQuest provides professional-grade sequencing data management with comprehensive technology support, robust error handling, and detailed reporting capabilities.