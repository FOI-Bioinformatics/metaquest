"""
Enhanced SRA data handling for MetaQuest.

This module provides enhanced SRA downloading with technology detection,
robust error handling, and comprehensive progress tracking.
"""

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from metaquest.core.exceptions import DataAccessError, SecurityError
from metaquest.data.file_io import ensure_directory
from metaquest.data.sra_metadata import (
    SRAMetadataClient,
    detect_sequencing_technology,
    create_download_preview,
    calculate_read_statistics,
)
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


class EnhancedSRADownloader:
    """Enhanced SRA downloader with technology detection and robust error handling."""

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        num_threads: int = 4,
        max_workers: int = 4,
        temp_folder: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize enhanced SRA downloader.

        Args:
            email: Email for NCBI API access
            api_key: Optional API key for increased rate limits
            num_threads: Threads per download
            max_workers: Parallel downloads
            temp_folder: Temporary directory for downloads
        """
        self.metadata_client = SRAMetadataClient(email, api_key)
        self.num_threads = num_threads
        self.max_workers = max_workers
        self.temp_folder = temp_folder

    def preview_downloads(
        self, accessions: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """
        Preview what will be downloaded without actually downloading.

        Args:
            accessions: List of SRA accessions

        Returns:
            Tuple of (metadata_dict, technology_counts, total_size_gb)
        """
        return create_download_preview(accessions, self.metadata_client)

    def download_accession_enhanced(
        self,
        accession: str,
        output_folder: Union[str, Path],
        force: bool = False,
        technology_hint: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Download a single SRA accession with enhanced error handling and metadata.

        Args:
            accession: SRA accession to download
            output_folder: Output folder for downloaded files
            force: Force redownload if files exist
            technology_hint: Optional technology hint to optimize download

        Returns:
            Tuple of (success, message, metadata_dict)
        """
        output_path = Path(output_folder) / accession
        metadata = {}

        try:
            # Get metadata for this accession
            logger.info(f"Fetching metadata for {accession}")
            sra_metadata = self.metadata_client.get_sra_metadata([accession])
            
            if accession in sra_metadata:
                dataset_info = sra_metadata[accession]
                technology = detect_sequencing_technology(dataset_info)
                
                metadata = {
                    "technology": technology,
                    "platform": dataset_info.platform,
                    "instrument": dataset_info.instrument,
                    "layout": dataset_info.layout,
                    "spots": dataset_info.spots,
                    "bases": dataset_info.bases,
                    "avg_length": dataset_info.avg_length,
                    "size_mb": dataset_info.size_mb,
                }
                
                logger.info(f"Detected {technology} technology for {accession}")
            else:
                logger.warning(f"Could not fetch metadata for {accession}")

            # Check if already downloaded
            if not force and self._check_existing_download(output_path):
                logger.info(f"Skipping {accession}, files already exist")
                return True, "Already downloaded", metadata

            # Download with technology-specific optimizations
            success, message = self._download_with_optimizations(
                accession, output_path, technology_hint or metadata.get("technology", "unknown")
            )

            # Calculate post-download statistics if successful
            if success:
                try:
                    fastq_files = list(output_path.glob("*.fastq*"))
                    if fastq_files:
                        stats = calculate_read_statistics(fastq_files)
                        metadata.update({
                            "downloaded_reads": stats.total_reads,
                            "downloaded_bases": stats.total_bases,
                            "actual_avg_length": stats.avg_read_length,
                            "gc_content": stats.gc_content,
                        })
                except Exception as e:
                    logger.warning(f"Could not calculate post-download statistics: {e}")

            return success, message, metadata

        except Exception as e:
            logger.error(f"Error in enhanced download for {accession}: {e}")
            return False, f"Enhanced download failed: {str(e)}", metadata

    def _download_with_optimizations(
        self, accession: str, output_path: Path, technology: str
    ) -> Tuple[bool, str]:
        """Download with technology-specific optimizations."""
        temp_path = output_path.parent / f"{accession}_temp"
        
        try:
            # Clean up any existing temp directory
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {accession} ({technology} technology)")

            # Build command with technology-specific optimizations
            args = self._build_download_command(accession, temp_path, technology)

            # Run fasterq-dump
            SecureSubprocess.run_secure("fasterq-dump", args)

            # Handle output
            return self._handle_download_output(temp_path, output_path, technology)

        except subprocess.CalledProcessError as e:
            logger.error(f"fasterq-dump failed for {accession}: {e.stderr}")
            self._cleanup_temp(temp_path)
            return False, f"Download command failed: {e.stderr}"

        except SecurityError as e:
            logger.error(f"Security error downloading {accession}: {e}")
            self._cleanup_temp(temp_path)
            return False, f"Security error: {e}"

        except Exception as e:
            logger.error(f"Unexpected error downloading {accession}: {e}")
            self._cleanup_temp(temp_path)
            return False, f"Download failed: {str(e)}"

    def _build_download_command(
        self, accession: str, temp_path: Path, technology: str
    ) -> List[str]:
        """Build fasterq-dump command with technology-specific optimizations."""
        args = [
            "--threads", str(self.num_threads),
            "--progress",
            accession,
            "-O", str(temp_path),
        ]

        # Add technology-specific optimizations
        if technology == "illumina":
            # For Illumina data, enable split-files for paired-end
            args.extend(["--split-files"])
        elif technology in ["nanopore", "pacbio"]:
            # For long reads, use different quality encoding
            args.extend(["--include-technical"])

        # Add temp folder if specified
        if self.temp_folder:
            temp_folder_path = Path(self.temp_folder)
            if temp_folder_path.exists() and os.access(temp_folder_path, os.W_OK):
                args.extend(["--temp", str(temp_folder_path.absolute())])

        return args

    def _handle_download_output(
        self, temp_path: Path, output_path: Path, technology: str
    ) -> Tuple[bool, str]:
        """Handle download output with technology-specific processing."""
        # Find downloaded files
        fastq_files = list(temp_path.glob("*.fastq*"))
        
        if not fastq_files:
            logger.error(f"No FASTQ files created for {output_path.name}")
            self._cleanup_temp(temp_path)
            return False, "No FASTQ files created"

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Rename files based on technology and layout
        renamed_files = self._rename_files_by_technology(fastq_files, technology, output_path)

        # Move files to final location
        for old_file, new_file in renamed_files.items():
            shutil.move(str(old_file), str(new_file))

        # Clean up temp directory
        self._cleanup_temp(temp_path)

        logger.info(f"Downloaded {len(renamed_files)} files for {output_path.name}")
        return True, f"Downloaded {len(renamed_files)} files ({technology})"

    def _rename_files_by_technology(
        self, fastq_files: List[Path], technology: str, output_path: Path
    ) -> Dict[Path, Path]:
        """Rename files based on technology and standard conventions."""
        renamed_files = {}
        accession = output_path.name

        if technology == "illumina":
            # Sort files to ensure consistent R1/R2 assignment
            fastq_files.sort()
            
            if len(fastq_files) == 2:
                # Paired-end
                renamed_files[fastq_files[0]] = output_path / f"{accession}_R1.fastq.gz"
                renamed_files[fastq_files[1]] = output_path / f"{accession}_R2.fastq.gz"
            elif len(fastq_files) == 1:
                # Single-end
                renamed_files[fastq_files[0]] = output_path / f"{accession}_R1.fastq.gz"
            else:
                # Multiple files - handle as numbered
                for i, file in enumerate(fastq_files, 1):
                    renamed_files[file] = output_path / f"{accession}_R{i}.fastq.gz"

        elif technology in ["nanopore", "pacbio"]:
            # Long reads - typically single file
            if len(fastq_files) == 1:
                ext = ".fastq.gz" if fastq_files[0].suffix == ".gz" else ".fastq"
                renamed_files[fastq_files[0]] = output_path / f"{accession}_long{ext}"
            else:
                # Multiple files
                for i, file in enumerate(fastq_files, 1):
                    ext = ".fastq.gz" if file.suffix == ".gz" else ".fastq"
                    renamed_files[file] = output_path / f"{accession}_long_{i}{ext}"

        else:
            # Unknown technology - use generic naming
            for i, file in enumerate(fastq_files, 1):
                ext = ".fastq.gz" if file.suffix == ".gz" else ".fastq"
                suffix = f"_{i}" if len(fastq_files) > 1 else ""
                renamed_files[file] = output_path / f"{accession}{suffix}{ext}"

        return renamed_files

    def _check_existing_download(self, output_path: Path) -> bool:
        """Check if files are already downloaded."""
        if not output_path.exists():
            return False
            
        fastq_files = list(output_path.glob("*.fastq*"))
        return len(fastq_files) > 0

    def _cleanup_temp(self, temp_path: Path) -> None:
        """Clean up temporary directory."""
        try:
            if temp_path.exists():
                shutil.rmtree(temp_path)
        except Exception as e:
            logger.warning(f"Could not remove temp directory {temp_path}: {e}")

    def download_batch_enhanced(
        self,
        accessions: List[str],
        output_folder: Union[str, Path],
        force: bool = False,
        max_downloads: Optional[int] = None,
        blacklist: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Download multiple SRA datasets with enhanced features.

        Args:
            accessions: List of SRA accessions
            output_folder: Output folder for downloads
            force: Force redownload
            max_downloads: Limit number of downloads
            blacklist: Set of accessions to skip

        Returns:
            Dictionary with download results and statistics
        """
        fastq_path = Path(output_folder)
        fastq_path.mkdir(parents=True, exist_ok=True)

        # Filter accessions
        if blacklist:
            accessions = [acc for acc in accessions if acc not in blacklist]

        # Limit downloads if requested
        if max_downloads and len(accessions) > max_downloads:
            accessions = accessions[:max_downloads]

        logger.info(f"Starting enhanced download of {len(accessions)} datasets")

        # Track results
        results = {
            "total": len(accessions),
            "successful": 0,
            "failed": 0,
            "failed_accessions": [],
            "download_results": {},
            "metadata": {},
            "technology_summary": {},
        }

        # Download in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit jobs
            futures = {
                executor.submit(
                    self.download_accession_enhanced, acc, fastq_path, force
                ): acc
                for acc in accessions
            }

            # Process results
            for future in as_completed(futures):
                accession = futures[future]
                try:
                    success, message, metadata = future.result()
                    
                    results["download_results"][accession] = message
                    results["metadata"][accession] = metadata
                    
                    if success:
                        results["successful"] += 1
                        
                        # Track technology
                        tech = metadata.get("technology", "unknown")
                        results["technology_summary"][tech] = (
                            results["technology_summary"].get(tech, 0) + 1
                        )
                        
                        logger.info(f"✓ {accession}: {message}")
                    else:
                        results["failed"] += 1
                        results["failed_accessions"].append(accession)
                        logger.error(f"✗ {accession}: {message}")

                except Exception as e:
                    results["failed"] += 1
                    results["failed_accessions"].append(accession)
                    results["download_results"][accession] = f"Exception: {str(e)}"
                    logger.error(f"✗ {accession}: Exception: {e}")

        # Log summary
        logger.info("Enhanced download complete:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        
        if results["technology_summary"]:
            logger.info("  Technology breakdown:")
            for tech, count in results["technology_summary"].items():
                logger.info(f"    {tech}: {count}")

        return results


def verify_sra_tools() -> bool:
    """
    Verify that SRA tools are installed and accessible.

    Returns:
        True if tools are available, False otherwise
    """
    required_tools = ["fasterq-dump", "prefetch", "vdb-validate"]
    
    for tool in required_tools:
        try:
            subprocess.run([tool, "--version"], 
                         capture_output=True, check=True, timeout=10)
            logger.debug(f"✓ {tool} is available")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.error(f"✗ {tool} is not available or not working properly")
            return False
    
    logger.info("All SRA tools are available")
    return True


def estimate_download_time(
    total_size_gb: float, 
    bandwidth_mbps: float = 100,
    num_parallel: int = 4
) -> float:
    """
    Estimate download time based on size and bandwidth.

    Args:
        total_size_gb: Total size in GB
        bandwidth_mbps: Bandwidth in Mbps
        num_parallel: Number of parallel downloads

    Returns:
        Estimated time in hours
    """
    # Convert GB to Mb
    total_size_mb = total_size_gb * 1024 * 8
    
    # Account for parallel downloads (with some efficiency loss)
    effective_bandwidth = bandwidth_mbps * num_parallel * 0.8
    
    # Calculate time in seconds, convert to hours
    time_seconds = total_size_mb / effective_bandwidth
    time_hours = time_seconds / 3600
    
    return time_hours


def create_download_report(
    results: Dict[str, Any], 
    output_file: Union[str, Path]
) -> None:
    """
    Create a comprehensive download report.

    Args:
        results: Download results from enhanced downloader
        output_file: Output report file
    """
    import pandas as pd
    
    if not results["metadata"]:
        logger.warning("No metadata available for report")
        return
    
    # Create detailed report
    report_data = []
    
    for accession, metadata in results["metadata"].items():
        status = "Success" if accession not in results["failed_accessions"] else "Failed"
        message = results["download_results"].get(accession, "")
        
        report_data.append({
            "accession": accession,
            "status": status,
            "message": message,
            "technology": metadata.get("technology", ""),
            "platform": metadata.get("platform", ""),
            "layout": metadata.get("layout", ""),
            "spots": metadata.get("spots", 0),
            "bases": metadata.get("bases", 0),
            "avg_length": metadata.get("avg_length", 0.0),
            "size_mb": metadata.get("size_mb", 0.0),
            "downloaded_reads": metadata.get("downloaded_reads", 0),
            "downloaded_bases": metadata.get("downloaded_bases", 0),
            "gc_content": metadata.get("gc_content", 0.0),
        })
    
    if report_data:
        df = pd.DataFrame(report_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Download report saved to {output_file}")
        
        # Print summary
        successful = df[df["status"] == "Success"]
        print(f"\nDownload Report Summary:")
        print(f"=======================")
        print(f"Total datasets: {len(df)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(df) - len(successful)}")
        
        if len(successful) > 0:
            print(f"Total downloaded reads: {successful['downloaded_reads'].sum():,}")
            print(f"Total downloaded bases: {successful['downloaded_bases'].sum():,}")
            print(f"Average GC content: {successful['gc_content'].mean():.1f}%")
            
            # Technology breakdown
            tech_counts = successful['technology'].value_counts()
            print(f"\nTechnology distribution:")
            for tech, count in tech_counts.items():
                print(f"  {tech}: {count}")