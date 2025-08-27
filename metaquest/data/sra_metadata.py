"""
SRA metadata and statistics handling for MetaQuest.

This module provides comprehensive functionality for fetching SRA metadata,
detecting sequencing technologies, and calculating dataset statistics.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from Bio import SeqIO

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)


@dataclass
class SRADatasetInfo:
    """Information about an SRA dataset."""

    accession: str
    title: str
    organism: str
    platform: str
    instrument: str
    strategy: str
    layout: str
    spots: int
    bases: int
    avg_length: float
    size_mb: float
    release_date: str
    bioproject: str
    biosample: str
    library_selection: str
    library_source: str


@dataclass
class ReadStatistics:
    """Statistics for sequenced reads."""

    total_reads: int
    total_bases: int
    avg_read_length: float
    min_read_length: int
    max_read_length: int
    n50: int
    gc_content: float
    quality_scores: Optional[Dict[str, float]] = None


class SRAMetadataClient:
    """Client for fetching SRA metadata from NCBI."""

    def __init__(self, email: str, api_key: Optional[str] = None):
        """
        Initialize SRA metadata client.

        Args:
            email: Email address for NCBI API access
            api_key: Optional API key for increased rate limits
        """
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.last_request_time = 0
        self.request_delay = 0.34 if api_key else 0.5  # Conservative rate limiting

    def _make_request(self, url: str, params: Dict[str, str]) -> str:
        """Make rate-limited request to NCBI API."""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)

        # Add required parameters
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.text
        except requests.RequestException as e:
            logger.error(f"NCBI API request failed: {e}")
            raise DataAccessError(f"Failed to query NCBI: {e}")

    def get_sra_metadata(self, accessions: List[str]) -> Dict[str, SRADatasetInfo]:
        """
        Fetch metadata for SRA accessions.

        Args:
            accessions: List of SRA accessions

        Returns:
            Dictionary mapping accessions to SRADatasetInfo objects
        """
        if not accessions:
            return {}

        logger.info(f"Fetching metadata for {len(accessions)} SRA accessions")
        results = {}
        batch_size = 200  # Process in batches to avoid URL length limits

        for i in range(0, len(accessions), batch_size):
            batch = accessions[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(accessions) + batch_size - 1) // batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            try:
                batch_results = self._fetch_batch_metadata(batch)
                results.update(batch_results)
            except Exception as e:
                logger.error(f"Failed to fetch metadata for batch: {e}")
                # Continue with other batches
                continue

        logger.info(f"Successfully fetched metadata for {len(results)} accessions")
        return results

    def _fetch_batch_metadata(self, accessions: List[str]) -> Dict[str, SRADatasetInfo]:
        """Fetch metadata for a batch of accessions."""
        # Search for accessions
        search_url = f"{self.base_url}esearch.fcgi"
        search_query = " OR ".join(accessions)
        params = {
            "db": "sra",
            "term": search_query,
            "retmax": str(len(accessions)),
            "retmode": "json",
        }

        search_response = self._make_request(search_url, params)
        search_data = json.loads(search_response)

        if (
            "esearchresult" not in search_data
            or not search_data["esearchresult"]["idlist"]
        ):
            logger.warning(f"No results found for batch: {accessions[:3]}...")
            return {}

        # Fetch detailed information
        ids = search_data["esearchresult"]["idlist"]
        fetch_url = f"{self.base_url}efetch.fcgi"
        params = {
            "db": "sra",
            "id": ",".join(ids),
            "retmode": "xml",
        }

        fetch_response = self._make_request(fetch_url, params)
        return self._parse_sra_xml(fetch_response)

    def _parse_sra_xml(self, xml_content: str) -> Dict[str, SRADatasetInfo]:
        """Parse SRA XML response to extract metadata."""
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_content)
            results = {}

            for package in root.findall(".//EXPERIMENT_PACKAGE"):
                try:
                    dataset_info = self._extract_dataset_info(package)
                    if dataset_info:
                        results[dataset_info.accession] = dataset_info
                except Exception as e:
                    logger.warning(f"Failed to parse dataset package: {e}")
                    continue

            return results
        except Exception as e:
            logger.error(f"Failed to parse SRA XML: {e}")
            return {}

    def _extract_dataset_info(self, package) -> Optional[SRADatasetInfo]:
        """Extract dataset information from XML package."""
        try:
            # Get experiment info
            experiment = package.find(".//EXPERIMENT")
            if experiment is None:
                return None

            accession = experiment.get("accession", "")
            title = self._get_text(experiment, ".//TITLE", "")

            # Get platform info
            platform_elem = experiment.find(".//PLATFORM")
            platform = ""
            instrument = ""
            if platform_elem is not None:
                for child in platform_elem:
                    platform = child.tag
                    instrument_elem = child.find(".//INSTRUMENT_MODEL")
                    if instrument_elem is not None:
                        instrument = instrument_elem.text or ""
                    break

            # Get library info
            library_descriptor = experiment.find(".//LIBRARY_DESCRIPTOR")
            strategy = self._get_text(library_descriptor, ".//LIBRARY_STRATEGY", "")
            selection = self._get_text(library_descriptor, ".//LIBRARY_SELECTION", "")
            source = self._get_text(library_descriptor, ".//LIBRARY_SOURCE", "")
            layout_elem = (
                library_descriptor.find(".//LIBRARY_LAYOUT")
                if library_descriptor
                else None
            )
            layout = (
                "PAIRED"
                if layout_elem and layout_elem.find(".//PAIRED") is not None
                else "SINGLE"
            )

            # Get run info
            run_set = package.find(".//RUN_SET")
            spots = 0
            bases = 0
            size_mb = 0.0
            if run_set is not None:
                for run in run_set.findall(".//RUN"):
                    spots += int(self._get_text(run, ".//Statistics/@nspots", "0"))
                    bases += int(self._get_text(run, ".//Statistics/@nbases", "0"))
                    size_mb += float(
                        self._get_text(run, ".//Statistics/@size", "0")
                    ) / (1024 * 1024)

            avg_length = bases / spots if spots > 0 else 0.0

            # Get sample info
            sample = package.find(".//SAMPLE")
            organism = (
                self._get_text(sample, ".//SCIENTIFIC_NAME", "") if sample else ""
            )

            # Get study info
            study = package.find(".//STUDY")
            bioproject = (
                self._get_text(study, ".//EXTERNAL_ID[@namespace='BioProject']", "")
                if study
                else ""
            )

            # Get submission info
            submission = package.find(".//SUBMISSION")
            release_date = (
                self._get_text(submission, "./@received", "") if submission else ""
            )

            # Get biosample
            sample_attrs = package.findall(".//SAMPLE_ATTRIBUTE")
            biosample = ""
            for attr in sample_attrs:
                tag = self._get_text(attr, ".//TAG", "")
                if tag.lower() == "biosample":
                    biosample = self._get_text(attr, ".//VALUE", "")
                    break

            return SRADatasetInfo(
                accession=accession,
                title=title,
                organism=organism,
                platform=platform,
                instrument=instrument,
                strategy=strategy,
                layout=layout,
                spots=spots,
                bases=bases,
                avg_length=avg_length,
                size_mb=size_mb,
                release_date=release_date,
                bioproject=bioproject,
                biosample=biosample,
                library_selection=selection,
                library_source=source,
            )

        except Exception as e:
            logger.warning(f"Failed to extract dataset info: {e}")
            return None

    def _get_text(self, element, xpath: str, default: str = "") -> str:
        """Safely extract text from XML element."""
        if element is None:
            return default

        try:
            if xpath.startswith("./@"):
                # Attribute
                attr_name = xpath[3:]
                return element.get(attr_name, default)
            elif "/@" in xpath:
                # Nested attribute
                elem_path, attr_name = xpath.rsplit("/@", 1)
                elem = element.find(elem_path)
                return elem.get(attr_name, default) if elem is not None else default
            else:
                # Element text
                elem = element.find(xpath)
                return elem.text or default if elem is not None else default
        except Exception:
            return default


def detect_sequencing_technology(dataset_info: SRADatasetInfo) -> str:
    """
    Detect sequencing technology from SRA metadata.

    Args:
        dataset_info: SRA dataset information

    Returns:
        Technology type: 'illumina', 'nanopore', 'pacbio', or 'unknown'
    """
    platform = dataset_info.platform.lower()
    instrument = dataset_info.instrument.lower()

    # Illumina detection
    if platform == "illumina" or "illumina" in instrument:
        return "illumina"

    # Nanopore detection
    if (
        platform == "oxford_nanopore"
        or "nanopore" in instrument
        or "minion" in instrument
        or "gridion" in instrument
    ):
        return "nanopore"

    # PacBio detection
    if (
        platform == "pacbio_smrt"
        or "pacbio" in instrument
        or "sequel" in instrument
        or "rs" in instrument
    ):
        return "pacbio"

    # Check strategy for additional hints
    strategy = dataset_info.strategy.lower()
    if "nanopore" in strategy:
        return "nanopore"
    if "pacbio" in strategy:
        return "pacbio"

    logger.warning(
        f"Unknown sequencing technology for {dataset_info.accession}: "
        f"{platform}/{instrument}"
    )
    return "unknown"


def calculate_read_statistics(fastq_files: List[Path]) -> ReadStatistics:
    """
    Calculate comprehensive statistics for FASTQ files.

    Args:
        fastq_files: List of FASTQ file paths

    Returns:
        ReadStatistics object with computed statistics
    """
    total_reads = 0
    total_bases = 0
    read_lengths = []
    gc_count = 0
    quality_scores = []

    logger.info(f"Calculating statistics for {len(fastq_files)} FASTQ files")

    for fastq_file in fastq_files:
        logger.debug(f"Processing {fastq_file.name}")

        try:
            # Determine if file is gzipped
            open_func = _get_file_opener(fastq_file)

            with open_func(fastq_file, "rt") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    total_reads += 1
                    seq_len = len(record.seq)
                    total_bases += seq_len
                    read_lengths.append(seq_len)

                    # Calculate GC content
                    gc_count += record.seq.count("G") + record.seq.count("C")

                    # Calculate average quality score
                    if (
                        hasattr(record, "letter_annotations")
                        and "phred_quality" in record.letter_annotations
                    ):
                        avg_qual = sum(
                            record.letter_annotations["phred_quality"]
                        ) / len(record.letter_annotations["phred_quality"])
                        quality_scores.append(avg_qual)

        except Exception as e:
            logger.error(f"Error processing {fastq_file}: {e}")
            continue

    if total_reads == 0:
        logger.warning("No reads found in FASTQ files")
        return ReadStatistics(0, 0, 0.0, 0, 0, 0, 0.0)

    # Calculate statistics
    avg_read_length = total_bases / total_reads
    min_read_length = min(read_lengths) if read_lengths else 0
    max_read_length = max(read_lengths) if read_lengths else 0
    n50 = _calculate_n50(read_lengths)
    gc_content = (gc_count / total_bases) * 100 if total_bases > 0 else 0.0

    # Quality score statistics
    qual_stats = None
    if quality_scores:
        qual_stats = {
            "mean": sum(quality_scores) / len(quality_scores),
            "min": min(quality_scores),
            "max": max(quality_scores),
        }

    logger.info(
        f"Statistics calculated: {total_reads} reads, {total_bases} bases, "
        f"{avg_read_length:.1f} avg length"
    )

    return ReadStatistics(
        total_reads=total_reads,
        total_bases=total_bases,
        avg_read_length=avg_read_length,
        min_read_length=min_read_length,
        max_read_length=max_read_length,
        n50=n50,
        gc_content=gc_content,
        quality_scores=qual_stats,
    )


def _get_file_opener(file_path: Path):
    """Get appropriate file opener based on file extension."""
    import gzip

    if file_path.suffix.lower() in [".gz", ".gzip"]:
        return gzip.open
    else:
        return open


def _calculate_n50(read_lengths: List[int]) -> int:
    """Calculate N50 statistic for read lengths."""
    if not read_lengths:
        return 0

    sorted_lengths = sorted(read_lengths, reverse=True)
    total_length = sum(sorted_lengths)
    target = total_length / 2

    cumulative = 0
    for length in sorted_lengths:
        cumulative += length
        if cumulative >= target:
            return length

    return 0


def create_download_preview(
    accessions: List[str], metadata_client: SRAMetadataClient
) -> Tuple[Dict[str, SRADatasetInfo], Dict[str, int], float]:
    """
    Create a preview of what would be downloaded.

    Args:
        accessions: List of SRA accessions
        metadata_client: SRA metadata client

    Returns:
        Tuple of (metadata_dict, technology_counts, total_size_gb)
    """
    logger.info("Creating download preview...")

    # Fetch metadata
    metadata = metadata_client.get_sra_metadata(accessions)

    # Count technologies
    tech_counts = {}
    total_size_mb = 0.0

    for acc, info in metadata.items():
        tech = detect_sequencing_technology(info)
        tech_counts[tech] = tech_counts.get(tech, 0) + 1
        total_size_mb += info.size_mb

    total_size_gb = total_size_mb / 1024

    logger.info(f"Preview: {len(metadata)} datasets, {total_size_gb:.2f} GB total")
    return metadata, tech_counts, total_size_gb


def save_metadata_report(
    metadata: Dict[str, SRADatasetInfo], output_file: Union[str, Path]
) -> None:
    """
    Save metadata report to CSV file.

    Args:
        metadata: Dictionary of SRA metadata
        output_file: Output CSV file path
    """
    if not metadata:
        logger.warning("No metadata to save")
        return

    # Convert to DataFrame
    records = []
    for acc, info in metadata.items():
        records.append(
            {
                "accession": info.accession,
                "title": info.title,
                "organism": info.organism,
                "platform": info.platform,
                "instrument": info.instrument,
                "strategy": info.strategy,
                "layout": info.layout,
                "spots": info.spots,
                "bases": info.bases,
                "avg_length": info.avg_length,
                "size_mb": info.size_mb,
                "release_date": info.release_date,
                "bioproject": info.bioproject,
                "biosample": info.biosample,
                "library_selection": info.library_selection,
                "library_source": info.library_source,
                "technology": detect_sequencing_technology(info),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    logger.info(f"Metadata report saved to {output_file}")


def generate_statistics_report(
    fastq_folder: Union[str, Path], output_file: Union[str, Path]
) -> None:
    """
    Generate comprehensive statistics report for downloaded datasets.

    Args:
        fastq_folder: Folder containing FASTQ files
        output_file: Output report file path
    """
    fastq_path = Path(fastq_folder)
    if not fastq_path.exists():
        raise DataAccessError(f"FASTQ folder {fastq_folder} does not exist")

    logger.info("Generating statistics report for downloaded datasets")

    # Find all accession directories
    accession_dirs = [d for d in fastq_path.iterdir() if d.is_dir()]

    if not accession_dirs:
        logger.warning("No accession directories found")
        return

    # Calculate statistics for each dataset
    report_data = []

    for acc_dir in accession_dirs:
        logger.info(f"Processing {acc_dir.name}")

        # Find FASTQ files
        fastq_files = list(acc_dir.glob("*.fastq*"))

        if not fastq_files:
            logger.warning(f"No FASTQ files found in {acc_dir}")
            continue

        try:
            stats = calculate_read_statistics(fastq_files)

            # Detect layout and technology from file patterns
            layout = (
                "PAIRED"
                if any("_2" in f.name or "_R2" in f.name for f in fastq_files)
                else "SINGLE"
            )

            report_data.append(
                {
                    "accession": acc_dir.name,
                    "num_files": len(fastq_files),
                    "layout": layout,
                    "total_reads": stats.total_reads,
                    "total_bases": stats.total_bases,
                    "avg_read_length": stats.avg_read_length,
                    "min_read_length": stats.min_read_length,
                    "max_read_length": stats.max_read_length,
                    "n50": stats.n50,
                    "gc_content": stats.gc_content,
                    "avg_quality": (
                        stats.quality_scores["mean"] if stats.quality_scores else None
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Failed to calculate statistics for {acc_dir.name}: {e}")
            continue

    if report_data:
        # Save report
        df = pd.DataFrame(report_data)
        df.to_csv(output_file, index=False)
        logger.info(
            f"Statistics report saved to {output_file} ({len(report_data)} datasets)"
        )

        # Print summary
        print("\nDataset Statistics Summary:")
        print("==========================")
        print(f"Total datasets: {len(report_data)}")
        print(f"Total reads: {df['total_reads'].sum():,}")
        print(f"Total bases: {df['total_bases'].sum():,}")
        print(f"Average read length: {df['avg_read_length'].mean():.1f}")
        print(f"Average GC content: {df['gc_content'].mean():.1f}%")

        # Technology breakdown
        layout_counts = df["layout"].value_counts()
        print("\nLayout distribution:")
        for layout, count in layout_counts.items():
            print(f"  {layout}: {count}")
    else:
        logger.error("No statistics could be calculated")
