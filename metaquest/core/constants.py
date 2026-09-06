"""
Core constants for MetaQuest.

This module centralizes all magic numbers and hard-coded values used throughout the application.
"""

# SRA Accession Validation
SRA_VALID_PREFIXES = ("SRR", "ERR", "DRR")
SRA_ACCESSION_PATTERN = r"^[A-Z]{3}[0-9]+$"
GENOME_ACCESSION_PATTERN = r"^GC[AF]_\d{9}\.\d+$"
GENOME_ACCESSION_PREFIXES = ("GCF_", "GCA_")

# Default Thresholds
DEFAULT_CONTAINMENT_THRESHOLD = 0.1
DEFAULT_METADATA_THRESHOLD = 0.0
DEFAULT_SINGLE_SAMPLE_THRESHOLD = 0.1
DEFAULT_STEP_SIZE = 0.1

# Default File Names and Paths
DEFAULT_GENOME_FOLDER = "genomes"
DEFAULT_MATCHES_FOLDER = "matches"
DEFAULT_METADATA_FOLDER = "metadata"
DEFAULT_FASTQ_FOLDER = "fastq"
DEFAULT_PARSED_CONTAINMENT_FILE = "parsed_containment.txt"
DEFAULT_SUMMARY_CONTAINMENT_FILE = "top_containments.txt"
DEFAULT_METADATA_TABLE_FILE = "metadata_table.txt"
DEFAULT_METADATA_COUNTS_FILE = "metadata_counts.txt"
DEFAULT_BRANCHWATER_METADATA_FILE = "branchwater_metadata.txt"
FAILED_ACCESSIONS_FILE = "failed_accessions.txt"

# Threading and Performance Defaults
DEFAULT_NUM_THREADS = 4
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 1
DEFAULT_TOP_N = 100

# Visualization Defaults
DEFAULT_PLOT_COLUMN = "max_containment"
DEFAULT_PLOT_TYPE_CONTAINMENT = "rank"
DEFAULT_PLOT_TYPE_METADATA = "bar"
SUPPORTED_PLOT_FORMATS = ["png", "jpg", "pdf", "svg"]
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 300

# Plot Type Choices
CONTAINMENT_PLOT_TYPES = ["rank", "histogram", "box", "violin"]
METADATA_PLOT_TYPES = ["bar", "pie", "radar"]

# File Formats
SUPPORTED_FILE_FORMATS = ["branchwater"]

# Logging Configuration
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"

# Security Constraints
MAX_SUBPROCESS_TIMEOUT = 3600  # 1 hour
DANGEROUS_ENV_VARS = ["LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES"]

# Bioinformatics Tool Configuration
ALLOWED_BIOINFORMATICS_TOOLS = {
    "fasterq-dump": {
        "safe_params": {
            "--threads",
            "--progress",
            "-O",
            "--temp",
            "--split-files",
            "--split-3",
            "--skip-technical",
            "--include-technical",
            "--force",
            "--version",
        },
        "description": "NCBI SRA data download tool",
    },
    "prefetch": {
        "safe_params": {
            "-O",
            "--max-size",
            "--progress",
            "--resume",
            "--version",
        },
        "description": "NCBI SRA prefetch (downloads the .sra archive)",
    },
    "pigz": {
        "safe_params": {
            "-p",
            "-f",
            "-k",
            "--version",
        },
        "description": "Parallel gzip",
    },
    "megahit": {
        "safe_params": {
            "-1",
            "-2",
            "-r",
            "-o",
            "--num-cpu-threads",
            "--memory",
            "--min-contig-len",
            "--k-min",
            "--k-max",
            "--k-step",
            "--no-mercy",
            "--bubble-level",
            "--version",
        },
        "description": "Illumina assembly tool",
    },
    "flye": {
        "safe_params": {
            "--nano-raw",
            "--nano-corr",
            "--nano-hq",
            "--pacbio-raw",
            "--pacbio-corr",
            "--pacbio-hifi",
            "--out-dir",
            "--genome-size",
            "--iterations",
            "--meta",
            "--polish-target",
            "--threads",
            "--min-overlap",
            "--keep-haplotypes",
        },
        "description": "Long-read assembly tool",
    },
    "datasets": {
        "safe_params": {
            "download",
            "genome",
            "accession",
            "--include",
            "--filename",
            "--inputfile",
            "--assembly-level",
            "--version",
            "--no-progressbar",
        },
        "description": "NCBI datasets CLI for genome downloads",
    },
    "minimap2": {
        "safe_params": {
            "-a",
            "-x",
            "-t",
            "-o",
            "--secondary",
            "--version",
        },
        "description": "Read-to-reference aligner (targeted read extraction)",
    },
    "samtools": {
        "safe_params": {
            "view",
            "fastq",
            "-b",
            "-F",
            "-f",
            "-c",
            "-o",
            "-0",
            "-1",
            "-2",
            "-s",
            "-@",
            "--version",
        },
        "description": "SAM/BAM utilities (filter and export mapped reads)",
    },
}

# URL Constants
BRANCHWATER_BASE_URL = "https://branchwater.jgi.doe.gov/"

# Regular Expressions
UNSAFE_SHELL_CHARS = [
    "$",
    "`",
    ";",
    "|",
    "&",
    ">",
    "<",
    "*",
    "?",
    "[",
    "]",
    "(",
    ")",
    "{",
    "}",
]

# File Extensions
FASTQ_EXTENSIONS = [".fastq", ".fq", ".fastq.gz", ".fq.gz"]
ASSEMBLY_EXTENSIONS = [".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz"]

# FASTA file patterns accepted as genome inputs (plain and gzipped).
GENOME_FASTA_GLOBS = ("*.fna", "*.fna.gz", "*.fasta", "*.fasta.gz", "*.fa", "*.fa.gz")

# FASTQ file patterns accepted as read inputs (plain and gzipped).
FASTQ_GLOBS = ("*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz")

# Most screening entries the registry keeps per target genome; the lowest containments
# beyond this are dropped, so a broad search does not bloat the project journal.
DEFAULT_REGISTRY_MAX_SCREENED = 5000

# Shared data store: one folder holding a single copy of every downloaded
# metagenome, used across organism projects.
STORE_MARKER = "metaquest_store.json"
STORE_ENV = "METAQUEST_DATA"
CONFIG_DIRNAME = "metaquest"
CONFIG_FILENAME = "config.toml"
STORE_LAYOUT = "sra-v1"

# Per-accession dataset lock (metaquest.store.locks). A store download or adopt holds its
# accession's lock for as long as the transfer takes, which can be hours, so the holder
# refreshes the lock file's mtime every LOCK_HEARTBEAT_SECONDS and a waiter only reclaims a
# lock whose mtime is older than DATASET_LOCK_STALE_SECONDS, i.e. one whose holder has died.
LOCK_HEARTBEAT_SECONDS = 10.0
DATASET_LOCK_STALE_SECONDS = 600.0

# Memory and Resource Limits
DEFAULT_MEMORY_LIMIT_GB = 8
MAX_FILE_SIZE_MB = 1024  # 1GB max file size for uploads
MAX_CONCURRENT_DOWNLOADS = 10

# Error Messages
ERROR_MESSAGES = {
    "invalid_accession": "Invalid SRA accession format: {}",
    "security_violation": "Security violation detected: {}",
    "file_not_found": "Required file not found: {}",
    "permission_denied": "Permission denied accessing: {}",
    "timeout_exceeded": "Operation timed out after {} seconds",
    "invalid_format": "Unsupported file format: {}",
}

# Success Messages
SUCCESS_MESSAGES = {
    "download_complete": "Successfully downloaded: {}",
    "processing_complete": "Processing completed for: {}",
    "assembly_complete": "Assembly completed for: {}",
    "validation_passed": "Validation passed for: {}",
}

# Plugin System Constants
PLUGIN_REGISTRY_NAME = "metaquest_plugins"
DEFAULT_PLUGIN_TIMEOUT = 300  # 5 minutes
