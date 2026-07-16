"""
Intelligent SRA Download Manager for MetaQuest.

This module provides advanced download management capabilities including:
- Intelligent resume/checkpoint functionality
- Bandwidth management and network adaptation
- Smart download scheduling and optimization
- Real-time progress monitoring with ETAs
- Advanced error recovery and retry strategies
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from metaquest.data.file_io import ensure_directory
from metaquest.data.sra import download_accession

# Conservative bandwidth assumption (Mbps) used only for rough time estimates.
# The download manager no longer probes the network to measure this.
_DEFAULT_BANDWIDTH_MBPS = 10.0

logger = logging.getLogger(__name__)


@dataclass
class NetworkConditions:
    """Network performance metrics."""

    bandwidth_mbps: float
    latency_ms: float
    packet_loss_pct: float
    connection_stability: float  # 0-1 score
    optimal_parallel_downloads: int
    last_measured: datetime


@dataclass
class DownloadCheckpoint:
    """Checkpoint data for resumable downloads."""

    accession: str
    total_size_bytes: Optional[int]
    downloaded_bytes: int
    chunk_checksums: Dict[int, str]  # chunk_index -> checksum
    download_start: datetime
    last_progress: datetime
    estimated_completion: Optional[datetime]
    retry_count: int
    failure_reasons: List[str]


@dataclass
class DownloadProgress:
    """Real-time download progress information."""

    accession: str
    status: str  # 'queued', 'downloading', 'completed', 'failed', 'paused'
    progress_pct: float
    downloaded_mb: float
    total_mb: Optional[float]
    speed_mbps: float
    eta_seconds: Optional[int]
    retry_count: int
    error_message: Optional[str]


@dataclass
class DownloadSession:
    """Complete download session information."""

    session_id: str
    accessions: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    total_size_mb: float
    downloaded_mb: float
    success_count: int
    failure_count: int
    average_speed_mbps: float
    network_conditions: NetworkConditions
    download_results: Dict[str, DownloadProgress]


class BandwidthManager:
    """Manages bandwidth allocation and network adaptation."""

    def __init__(self, max_bandwidth_mbps: Optional[float] = None):
        self.max_bandwidth_mbps = max_bandwidth_mbps
        self.current_usage_mbps = 0.0
        self.active_downloads = 0
        self.network_conditions: Optional[NetworkConditions] = None

    def measure_network_conditions(self) -> NetworkConditions:
        """Derive a parallelism recommendation from local CPU count.

        This deliberately performs no network I/O. A previous implementation
        shelled out to curl against an external host to measure bandwidth,
        which was unreliable and issued a request to a third party on every
        session. Parallelism is now based on the CPU count, and bandwidth is
        reported as a conservative default used only for rough time estimates.
        """
        cpu_count = os.cpu_count() or 4
        optimal_parallel = max(2, min(8, cpu_count))

        self.network_conditions = NetworkConditions(
            bandwidth_mbps=_DEFAULT_BANDWIDTH_MBPS,
            latency_ms=0.0,
            packet_loss_pct=0.0,
            connection_stability=1.0,
            optimal_parallel_downloads=optimal_parallel,
            last_measured=datetime.now(),
        )

        return self.network_conditions

    def allocate_bandwidth(self, requested_mbps: float) -> float:
        """Allocate bandwidth for a download."""
        if self.max_bandwidth_mbps is None:
            return requested_mbps

        available = self.max_bandwidth_mbps - self.current_usage_mbps
        allocated = min(requested_mbps, available)
        self.current_usage_mbps += allocated
        return allocated

    def release_bandwidth(self, allocated_mbps: float):
        """Release allocated bandwidth."""
        self.current_usage_mbps = max(0, self.current_usage_mbps - allocated_mbps)


class DownloadOptimizer:
    """Optimizes download order and batching strategies."""

    def __init__(self, metadata_client=None):
        self.metadata_client = metadata_client
        self.size_cache = {}

    def estimate_dataset_sizes(self, accessions: List[str]) -> Dict[str, Optional[int]]:
        """Estimate dataset sizes for download optimization."""
        sizes = {}

        for accession in accessions:
            if accession in self.size_cache:
                sizes[accession] = self.size_cache[accession]
                continue

            # Try to get size from metadata
            try:
                if self.metadata_client:
                    dataset_info = self.metadata_client.get_dataset_info(accession)
                    size_mb = getattr(dataset_info, "size_mb", None)
                    if size_mb:
                        size_bytes = int(size_mb * 1024 * 1024)
                        sizes[accession] = size_bytes
                        self.size_cache[accession] = size_bytes
                        continue
            except Exception as e:
                logger.debug(f"Could not get size for {accession}: {e}")

            # Default estimation based on accession patterns
            sizes[accession] = self._estimate_size_from_pattern(accession)

        return sizes

    def _estimate_size_from_pattern(self, accession: str) -> Optional[int]:
        """Estimate size based on accession patterns."""
        # Very rough estimates based on typical patterns
        if accession.startswith("ERR"):
            return 500 * 1024 * 1024  # 500MB average for EBI
        elif accession.startswith("SRR"):
            return 800 * 1024 * 1024  # 800MB average for NCBI
        else:
            return 1024 * 1024 * 1024  # 1GB default

    def optimize_download_order(self, accessions: List[str]) -> List[str]:
        """Optimize download order for better user experience."""
        sizes = self.estimate_dataset_sizes(accessions)

        # Separate into size categories
        small = []  # < 100MB
        medium = []  # 100MB - 1GB
        large = []  # > 1GB
        unknown = []  # No size info

        for accession in accessions:
            size = sizes.get(accession)
            if size is None:
                unknown.append(accession)
            elif size < 100 * 1024 * 1024:
                small.append(accession)
            elif size < 1024 * 1024 * 1024:
                medium.append(accession)
            else:
                large.append(accession)

        # Strategy: Start with small files for quick wins, then medium, then large
        # Mix in unknowns with medium files
        optimized_order = small + medium + unknown + large

        logger.info(
            f"Optimized order: {len(small)} small, {len(medium)} medium, "
            f"{len(large)} large, {len(unknown)} unknown size files"
        )

        return optimized_order


class CheckpointManager:
    """Manages download checkpoints for resume capability."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint: DownloadCheckpoint):
        """Save download checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint.accession}.checkpoint"
        try:
            data = asdict(checkpoint)
            data["download_start"] = checkpoint.download_start.isoformat()
            data["last_progress"] = checkpoint.last_progress.isoformat()
            data["estimated_completion"] = (
                checkpoint.estimated_completion.isoformat() if checkpoint.estimated_completion else None
            )
            with open(checkpoint_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {checkpoint.accession}: {e}")

    def load_checkpoint(self, accession: str) -> Optional[DownloadCheckpoint]:
        """Load download checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{accession}.checkpoint"
        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            data["download_start"] = datetime.fromisoformat(data["download_start"])
            data["last_progress"] = datetime.fromisoformat(data["last_progress"])
            data["estimated_completion"] = (
                datetime.fromisoformat(data["estimated_completion"]) if data["estimated_completion"] else None
            )
            data["chunk_checksums"] = {int(k): v for k, v in data["chunk_checksums"].items()}
            return DownloadCheckpoint(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {accession}: {e}")
            return None

    def remove_checkpoint(self, accession: str):
        """Remove checkpoint after successful download."""
        checkpoint_file = self.checkpoint_dir / f"{accession}.checkpoint"
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint for {accession}: {e}")

    def list_resumable_downloads(self) -> List[str]:
        """List accessions with resumable downloads."""
        resumable = []
        for checkpoint_file in self.checkpoint_dir.glob("*.checkpoint"):
            accession = checkpoint_file.stem
            resumable.append(accession)
        return resumable


class IntelligentDownloadManager:
    """Advanced SRA download manager with resume, optimization, and monitoring."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        temp_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        max_bandwidth_mbps: Optional[float] = None,
        max_parallel_downloads: Optional[int] = None,
        resume_enabled: bool = True,
    ):
        """
        Initialize intelligent download manager.

        Args:
            output_dir: Directory for downloaded files
            temp_dir: Temporary directory for partial downloads
            checkpoint_dir: Directory for checkpoint files
            max_bandwidth_mbps: Maximum bandwidth limit
            max_parallel_downloads: Maximum parallel downloads
            resume_enabled: Enable resume capability
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else self.output_dir / "temp"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"

        # Create directories
        for directory in [self.output_dir, self.temp_dir, self.checkpoint_dir]:
            ensure_directory(directory)

        # Initialize managers
        self.bandwidth_manager = BandwidthManager(max_bandwidth_mbps)
        self.optimizer = DownloadOptimizer()
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir) if resume_enabled else None

        # Configuration
        self.max_parallel_downloads = max_parallel_downloads
        self.resume_enabled = resume_enabled

        # State tracking
        self.active_downloads: Dict[str, DownloadProgress] = {}
        self.session_id = f"session_{int(time.time())}"
        self.is_paused = False

    def estimate_download_time(self, accessions: List[str]) -> Dict[str, Any]:
        """Estimate total download time and provide detailed breakdown."""
        # Measure current network conditions
        network_conditions = self.bandwidth_manager.measure_network_conditions()

        # Estimate sizes
        sizes = self.optimizer.estimate_dataset_sizes(accessions)
        total_size_mb = sum([s / (1024 * 1024) for s in sizes.values() if s])

        # Calculate estimated time
        effective_bandwidth = network_conditions.bandwidth_mbps * 0.8  # 80% efficiency
        parallel_factor = min(len(accessions), network_conditions.optimal_parallel_downloads)

        if parallel_factor > 1:
            # Parallel downloads provide diminishing returns
            parallel_efficiency = 1.0 - (0.1 * (parallel_factor - 1))
            effective_bandwidth *= parallel_factor * parallel_efficiency

        estimated_time_minutes = total_size_mb / effective_bandwidth if effective_bandwidth > 0 else 0

        return {
            "total_size_mb": total_size_mb,
            "estimated_time_minutes": estimated_time_minutes,
            "estimated_time_formatted": str(timedelta(minutes=estimated_time_minutes)),
            "network_bandwidth_mbps": network_conditions.bandwidth_mbps,
            "optimal_parallel_downloads": network_conditions.optimal_parallel_downloads,
            "individual_estimates": {
                acc: self._individual_estimate(sizes[acc], effective_bandwidth) for acc in accessions
            },
        }

    @staticmethod
    def _individual_estimate(size: Optional[int], effective_bandwidth: float) -> Dict[str, Any]:
        """Per-dataset size (MB) and time estimate; 'unknown' when size is missing."""
        if not size:
            return {"size_mb": "unknown", "estimated_minutes": "unknown"}
        size_mb = size / (1024 * 1024)
        estimated = size_mb / effective_bandwidth if effective_bandwidth > 0 else "unknown"
        return {"size_mb": size_mb, "estimated_minutes": estimated}

    def download_with_resume(self, accessions: List[str], force_restart: bool = False) -> DownloadSession:
        """
        Download SRA accessions, skipping any already present on disk.

        "Resume" here means accession-level: an accession whose FASTQ files
        already exist is skipped rather than re-downloaded, unless
        ``force_restart`` is set. fasterq-dump provides no byte-range resume,
        so partially downloaded accessions are restarted.

        Args:
            accessions: List of SRA accessions to download
            force_restart: Re-download every accession even if files exist

        Returns:
            DownloadSession with comprehensive results
        """
        session_start = datetime.now()
        logger.info(f"Starting download session {self.session_id} for {len(accessions)} accessions")

        # Choose parallelism from the CPU-based recommendation (no network probe).
        network_conditions = self.bandwidth_manager.measure_network_conditions()

        # Optimize download order
        optimized_accessions = self.optimizer.optimize_download_order(accessions)

        # Determine parallelism
        max_parallel = self.max_parallel_downloads or network_conditions.optimal_parallel_downloads
        logger.info(f"Downloading with up to {max_parallel} parallel workers")

        # Initialize session tracking
        session = DownloadSession(
            session_id=self.session_id,
            accessions=optimized_accessions,
            start_time=session_start,
            end_time=None,
            total_size_mb=0.0,
            downloaded_mb=0.0,
            success_count=0,
            failure_count=0,
            average_speed_mbps=0.0,
            network_conditions=network_conditions,
            download_results={},
        )

        # Execute downloads with parallelism
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_accession = {}

            for accession in optimized_accessions:
                if self.is_paused:
                    break

                future = executor.submit(self._download_single_accession, accession, force_restart)
                future_to_accession[future] = accession

            # Process results as they complete
            for future in as_completed(future_to_accession):
                if self.is_paused:
                    break

                accession = future_to_accession[future]
                try:
                    result = future.result()
                    session.download_results[accession] = result

                    if result.status == "completed":
                        session.success_count += 1
                    else:
                        session.failure_count += 1

                    session.downloaded_mb += result.downloaded_mb

                except Exception as e:
                    logger.error(f"Download failed for {accession}: {e}")
                    session.failure_count += 1

        # Finalize session
        session.end_time = datetime.now()
        session_duration = (session.end_time - session_start).total_seconds()
        session.average_speed_mbps = session.downloaded_mb / (session_duration / 60) if session_duration > 0 else 0

        logger.info(
            f"Download session completed: {session.success_count}/{len(accessions)} successful, "
            f"avg speed: {session.average_speed_mbps:.1f} MB/min"
        )

        return session

    def _download_single_accession(self, accession: str, force: bool = False) -> DownloadProgress:
        """Download a single SRA accession via the shared, validated path.

        Delegates to ``metaquest.data.sra.download_accession``, which runs
        fasterq-dump through ``SecureSubprocess`` and writes FASTQ files to
        ``output_dir/<accession>/``. That helper skips accessions whose FASTQ
        files already exist unless ``force`` is set, which is how a re-run
        resumes at accession granularity; fasterq-dump has no byte-range
        resume, so a partial download is always restarted.
        """
        progress = DownloadProgress(
            accession=accession,
            status="downloading",
            progress_pct=0.0,
            downloaded_mb=0.0,
            total_mb=None,
            speed_mbps=0.0,
            eta_seconds=None,
            retry_count=0,
            error_message=None,
        )

        # Track active download
        self.active_downloads[accession] = progress

        try:
            success, message = download_accession(
                accession,
                self.output_dir,
                force=force,
                temp_folder=self.temp_dir,
            )

            if success:
                progress.status = "completed"
                progress.progress_pct = 100.0
                acc_dir = self.output_dir / accession
                total_bytes = sum(f.stat().st_size for f in acc_dir.glob("*.fastq*"))
                progress.downloaded_mb = total_bytes / (1024 * 1024)
            else:
                progress.status = "failed"
                progress.error_message = message

        except Exception as e:
            logger.error(f"Error downloading {accession}: {e}")
            progress.status = "failed"
            progress.error_message = str(e)

        finally:
            # Clean up tracking
            self.active_downloads.pop(accession, None)

        return progress

    def pause_downloads(self):
        """Pause all active downloads."""
        self.is_paused = True
        logger.info("Download manager paused")

    def resume_downloads(self):
        """Resume paused downloads."""
        self.is_paused = False
        logger.info("Download manager resumed")

    def get_active_downloads(self) -> Dict[str, DownloadProgress]:
        """Get current status of active downloads."""
        return self.active_downloads.copy()

    def get_resumable_downloads(self) -> List[str]:
        """Get list of downloads that can be resumed."""
        if not self.checkpoint_manager:
            return []
        return self.checkpoint_manager.list_resumable_downloads()
