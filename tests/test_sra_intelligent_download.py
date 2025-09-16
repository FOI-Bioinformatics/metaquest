"""
Comprehensive tests for intelligent SRA download manager.

Tests cover:
- Network condition assessment and bandwidth management
- Download optimization and smart scheduling
- Checkpoint/resume functionality
- Progress monitoring and error recovery
- Integration with existing SRA tools
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from metaquest.sra.download_manager import (
    IntelligentDownloadManager,
    BandwidthManager, 
    DownloadOptimizer,
    CheckpointManager,
    NetworkConditions,
    DownloadCheckpoint,
    DownloadProgress,
    DownloadSession
)
from metaquest.core.exceptions import DataAccessError, SecurityError


class TestNetworkConditions:
    """Test NetworkConditions dataclass."""
    
    def test_network_conditions_creation(self):
        """Test creating NetworkConditions object."""
        conditions = NetworkConditions(
            bandwidth_mbps=25.5,
            latency_ms=50.0,
            packet_loss_pct=0.1,
            connection_stability=0.95,
            optimal_parallel_downloads=4,
            last_measured=datetime.now()
        )
        
        assert conditions.bandwidth_mbps == 25.5
        assert conditions.latency_ms == 50.0
        assert conditions.optimal_parallel_downloads == 4
        assert isinstance(conditions.last_measured, datetime)


class TestDownloadCheckpoint:
    """Test DownloadCheckpoint functionality."""
    
    def test_checkpoint_creation(self):
        """Test creating download checkpoint."""
        checkpoint = DownloadCheckpoint(
            accession="SRR123456",
            total_size_bytes=1024*1024*500,  # 500MB
            downloaded_bytes=1024*1024*250,  # 250MB
            chunk_checksums={0: "abc123", 1: "def456"},
            download_start=datetime.now(),
            last_progress=datetime.now(),
            estimated_completion=datetime.now() + timedelta(minutes=10),
            retry_count=1,
            failure_reasons=["Network timeout"]
        )
        
        assert checkpoint.accession == "SRR123456"
        assert checkpoint.total_size_bytes == 1024*1024*500
        assert checkpoint.downloaded_bytes == 1024*1024*250
        assert checkpoint.retry_count == 1
        assert len(checkpoint.failure_reasons) == 1


class TestBandwidthManager:
    """Test bandwidth management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = BandwidthManager(max_bandwidth_mbps=50.0)
    
    def test_bandwidth_manager_creation(self):
        """Test BandwidthManager initialization."""
        assert self.manager.max_bandwidth_mbps == 50.0
        assert self.manager.current_usage_mbps == 0.0
        assert self.manager.active_downloads == 0
    
    @patch('subprocess.run')
    def test_measure_network_conditions(self, mock_subprocess):
        """Test network condition measurement."""
        # Mock subprocess response for bandwidth test
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="10485760\n2.5\n"  # 10MB/s, 2.5s latency
        )
        
        conditions = self.manager.measure_network_conditions()
        
        assert isinstance(conditions, NetworkConditions)
        assert conditions.bandwidth_mbps == 10.0  # 10MB/s = 10 Mbps
        assert conditions.latency_ms == 2500  # 2.5s = 2500ms
        assert conditions.optimal_parallel_downloads > 0
        assert isinstance(conditions.last_measured, datetime)
    
    @patch('subprocess.run')
    def test_measure_network_conditions_failure(self, mock_subprocess):
        """Test network measurement with subprocess failure."""
        mock_subprocess.return_value = Mock(returncode=1, stdout="")
        
        conditions = self.manager.measure_network_conditions()
        
        # Should use conservative defaults
        assert conditions.bandwidth_mbps == 10.0
        assert conditions.latency_ms == 100.0
    
    def test_allocate_bandwidth(self):
        """Test bandwidth allocation."""
        # Test normal allocation
        allocated = self.manager.allocate_bandwidth(20.0)
        assert allocated == 20.0
        assert self.manager.current_usage_mbps == 20.0
        
        # Test allocation exceeding limit
        allocated2 = self.manager.allocate_bandwidth(40.0)
        assert allocated2 == 30.0  # Only 30 available
        assert self.manager.current_usage_mbps == 50.0
    
    def test_release_bandwidth(self):
        """Test bandwidth release."""
        self.manager.current_usage_mbps = 30.0
        
        self.manager.release_bandwidth(15.0)
        assert self.manager.current_usage_mbps == 15.0
        
        # Test releasing more than allocated
        self.manager.release_bandwidth(20.0)
        assert self.manager.current_usage_mbps == 0.0  # Should not go negative
    
    def test_unlimited_bandwidth(self):
        """Test manager with unlimited bandwidth."""
        unlimited_manager = BandwidthManager(max_bandwidth_mbps=None)
        
        allocated = unlimited_manager.allocate_bandwidth(100.0)
        assert allocated == 100.0


class TestDownloadOptimizer:
    """Test download optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = DownloadOptimizer()
    
    def test_estimate_dataset_sizes_from_cache(self):
        """Test size estimation with cached values."""
        self.optimizer.size_cache = {
            "SRR123456": 1024*1024*100,  # 100MB
            "SRR789012": 1024*1024*200   # 200MB
        }
        
        sizes = self.optimizer.estimate_dataset_sizes(["SRR123456", "SRR789012"])
        
        assert sizes["SRR123456"] == 1024*1024*100
        assert sizes["SRR789012"] == 1024*1024*200
    
    def test_estimate_dataset_sizes_from_patterns(self):
        """Test size estimation from accession patterns."""
        accessions = ["ERR123456", "SRR789012", "DRR999999"]
        sizes = self.optimizer.estimate_dataset_sizes(accessions)
        
        assert sizes["ERR123456"] == 500 * 1024 * 1024  # EBI pattern
        assert sizes["SRR789012"] == 800 * 1024 * 1024  # NCBI pattern  
        assert sizes["DRR999999"] == 1024 * 1024 * 1024  # Default
    
    def test_optimize_download_order(self):
        """Test download order optimization."""
        # Set up size cache with known sizes
        self.optimizer.size_cache = {
            "SMALL1": 50 * 1024 * 1024,     # 50MB - small
            "MEDIUM1": 300 * 1024 * 1024,   # 300MB - medium
            "LARGE1": 2048 * 1024 * 1024,   # 2GB - large
            "SMALL2": 80 * 1024 * 1024,     # 80MB - small
        }
        
        accessions = ["LARGE1", "MEDIUM1", "SMALL1", "SMALL2"]
        optimized = self.optimizer.optimize_download_order(accessions)
        
        # Should prioritize small files first
        assert optimized.index("SMALL1") < optimized.index("MEDIUM1")
        assert optimized.index("SMALL2") < optimized.index("MEDIUM1")
        assert optimized.index("MEDIUM1") < optimized.index("LARGE1")


class TestCheckpointManager:
    """Test checkpoint management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(Path(self.temp_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        checkpoint = DownloadCheckpoint(
            accession="SRR123456",
            total_size_bytes=1000000,
            downloaded_bytes=500000,
            chunk_checksums={0: "abc123"},
            download_start=datetime.now(),
            last_progress=datetime.now(),
            estimated_completion=None,
            retry_count=2,
            failure_reasons=["Timeout", "Network error"]
        )
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(checkpoint)
        
        # Load checkpoint
        loaded = self.checkpoint_manager.load_checkpoint("SRR123456")
        
        assert loaded is not None
        assert loaded.accession == "SRR123456"
        assert loaded.total_size_bytes == 1000000
        assert loaded.downloaded_bytes == 500000
        assert loaded.retry_count == 2
        assert len(loaded.failure_reasons) == 2
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading non-existent checkpoint."""
        loaded = self.checkpoint_manager.load_checkpoint("NONEXISTENT")
        assert loaded is None
    
    def test_remove_checkpoint(self):
        """Test removing checkpoint."""
        checkpoint = DownloadCheckpoint(
            accession="SRR123456",
            total_size_bytes=1000000,
            downloaded_bytes=500000,
            chunk_checksums={},
            download_start=datetime.now(),
            last_progress=datetime.now(),
            estimated_completion=None,
            retry_count=0,
            failure_reasons=[]
        )
        
        self.checkpoint_manager.save_checkpoint(checkpoint)
        assert self.checkpoint_manager.load_checkpoint("SRR123456") is not None
        
        self.checkpoint_manager.remove_checkpoint("SRR123456")
        assert self.checkpoint_manager.load_checkpoint("SRR123456") is None
    
    def test_list_resumable_downloads(self):
        """Test listing resumable downloads."""
        # Create multiple checkpoints
        for i in range(3):
            checkpoint = DownloadCheckpoint(
                accession=f"SRR12345{i}",
                total_size_bytes=1000000,
                downloaded_bytes=500000,
                chunk_checksums={},
                download_start=datetime.now(),
                last_progress=datetime.now(),
                estimated_completion=None,
                retry_count=0,
                failure_reasons=[]
            )
            self.checkpoint_manager.save_checkpoint(checkpoint)
        
        resumable = self.checkpoint_manager.list_resumable_downloads()
        assert len(resumable) == 3
        assert "SRR123450" in resumable
        assert "SRR123451" in resumable
        assert "SRR123452" in resumable


class TestIntelligentDownloadManager:
    """Test main IntelligentDownloadManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = IntelligentDownloadManager(
            output_dir=self.temp_dir,
            max_bandwidth_mbps=50.0,
            max_parallel_downloads=2,
            resume_enabled=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.output_dir == Path(self.temp_dir)
        assert self.manager.temp_dir == Path(self.temp_dir) / "temp"
        assert self.manager.checkpoint_dir == Path(self.temp_dir) / "checkpoints"
        assert self.manager.resume_enabled is True
        assert self.manager.max_parallel_downloads == 2
        
        # Check directories were created
        assert self.manager.temp_dir.exists()
        assert self.manager.checkpoint_dir.exists()
    
    @patch('metaquest.sra.download_manager.BandwidthManager.measure_network_conditions')
    def test_estimate_download_time(self, mock_network):
        """Test download time estimation."""
        # Mock network conditions
        mock_network.return_value = NetworkConditions(
            bandwidth_mbps=10.0,
            latency_ms=100,
            packet_loss_pct=0.0,
            connection_stability=1.0,
            optimal_parallel_downloads=4,
            last_measured=datetime.now()
        )
        
        # Mock size estimation
        self.manager.optimizer.size_cache = {
            "SRR123456": 100 * 1024 * 1024,  # 100MB
            "SRR789012": 200 * 1024 * 1024   # 200MB  
        }
        
        estimate = self.manager.estimate_download_time(["SRR123456", "SRR789012"])
        
        assert 'total_size_mb' in estimate
        assert 'estimated_time_minutes' in estimate
        assert 'network_bandwidth_mbps' in estimate
        assert estimate['total_size_mb'] == 300  # 100 + 200 MB
        assert estimate['network_bandwidth_mbps'] == 10.0
    
    def test_pause_resume_downloads(self):
        """Test pause and resume functionality."""
        assert self.manager.is_paused is False
        
        self.manager.pause_downloads()
        assert self.manager.is_paused is True
        
        self.manager.resume_downloads()
        assert self.manager.is_paused is False
    
    def test_get_resumable_downloads(self):
        """Test getting resumable downloads."""
        # Create a checkpoint
        checkpoint = DownloadCheckpoint(
            accession="SRR123456",
            total_size_bytes=1000000,
            downloaded_bytes=500000,
            chunk_checksums={},
            download_start=datetime.now(),
            last_progress=datetime.now(),
            estimated_completion=None,
            retry_count=0,
            failure_reasons=[]
        )
        self.manager.checkpoint_manager.save_checkpoint(checkpoint)
        
        resumable = self.manager.get_resumable_downloads()
        assert "SRR123456" in resumable
    
    def test_get_active_downloads(self):
        """Test getting active downloads."""
        # Add some active downloads
        progress1 = DownloadProgress(
            accession="SRR123456",
            status="downloading",
            progress_pct=50.0,
            downloaded_mb=100.0,
            total_mb=200.0,
            speed_mbps=5.0,
            eta_seconds=120,
            retry_count=0,
            error_message=None
        )
        
        self.manager.active_downloads["SRR123456"] = progress1
        
        active = self.manager.get_active_downloads()
        assert "SRR123456" in active
        assert active["SRR123456"].progress_pct == 50.0
    
    def test_update_progress(self):
        """Test progress update functionality."""
        # Add active download
        progress = DownloadProgress(
            accession="SRR123456",
            status="downloading",
            progress_pct=0.0,
            downloaded_mb=0.0,
            total_mb=None,
            speed_mbps=0.0,
            eta_seconds=None,
            retry_count=0,
            error_message=None
        )
        self.manager.active_downloads["SRR123456"] = progress
        
        # Update progress
        progress_info = {
            'percent': 75.0,
            'downloaded_mb': 150.0,
            'speed_mbps': 10.0,
            'eta_seconds': 30
        }
        
        self.manager._update_progress("SRR123456", progress_info)
        
        updated = self.manager.active_downloads["SRR123456"]
        assert updated.progress_pct == 75.0
        assert updated.downloaded_mb == 150.0
        assert updated.speed_mbps == 10.0
        assert updated.eta_seconds == 30
    
    @patch('metaquest.sra.download_manager.BandwidthManager.measure_network_conditions')
    def test_download_with_resume_basic_flow(self, mock_network):
        """Test basic download flow without actual downloads."""
        # Mock network conditions
        mock_network.return_value = NetworkConditions(
            bandwidth_mbps=10.0,
            latency_ms=100,
            packet_loss_pct=0.0,
            connection_stability=1.0,
            optimal_parallel_downloads=2,
            last_measured=datetime.now()
        )
        
        # Mock the download method to return expected results
        mock_result1 = DownloadProgress(
            accession="SRR123456",
            status="completed",
            progress_pct=100.0,
            downloaded_mb=100.0,
            total_mb=100.0,
            speed_mbps=5.0,
            eta_seconds=0,
            retry_count=0,
            error_message=None
        )
        
        mock_result2 = DownloadProgress(
            accession="SRR789012", 
            status="failed",
            progress_pct=0.0,
            downloaded_mb=0.0,
            total_mb=None,
            speed_mbps=0.0,
            eta_seconds=None,
            retry_count=1,
            error_message="Network error"
        )
        
        # Mock the internal download method
        with patch.object(self.manager, '_download_single_accession') as mock_download:
            mock_download.side_effect = [mock_result1, mock_result2]
            
            session = self.manager.download_with_resume(["SRR123456", "SRR789012"])
            
            assert isinstance(session, DownloadSession)
            assert len(session.accessions) == 2
            assert session.success_count == 1
            assert session.failure_count == 1
            assert "SRR123456" in session.download_results
            assert "SRR789012" in session.download_results
            assert session.download_results["SRR123456"].status == "completed"
            assert session.download_results["SRR789012"].status == "failed"


class TestDownloadProgressAndSession:
    """Test DownloadProgress and DownloadSession data structures."""
    
    def test_download_progress_creation(self):
        """Test DownloadProgress object creation."""
        progress = DownloadProgress(
            accession="SRR123456",
            status="downloading",
            progress_pct=45.5,
            downloaded_mb=227.5,
            total_mb=500.0,
            speed_mbps=12.3,
            eta_seconds=180,
            retry_count=1,
            error_message=None
        )
        
        assert progress.accession == "SRR123456"
        assert progress.status == "downloading"
        assert progress.progress_pct == 45.5
        assert progress.downloaded_mb == 227.5
        assert progress.total_mb == 500.0
        assert progress.speed_mbps == 12.3
        assert progress.eta_seconds == 180
        assert progress.retry_count == 1
        assert progress.error_message is None
    
    def test_download_session_creation(self):
        """Test DownloadSession object creation."""
        network_conditions = NetworkConditions(
            bandwidth_mbps=25.0,
            latency_ms=80,
            packet_loss_pct=0.5,
            connection_stability=0.95,
            optimal_parallel_downloads=6,
            last_measured=datetime.now()
        )
        
        session = DownloadSession(
            session_id="test_session_123",
            accessions=["SRR123456", "SRR789012"],
            start_time=datetime.now(),
            end_time=None,
            total_size_mb=1500.0,
            downloaded_mb=750.0,
            success_count=1,
            failure_count=1,
            average_speed_mbps=8.5,
            network_conditions=network_conditions,
            download_results={}
        )
        
        assert session.session_id == "test_session_123"
        assert len(session.accessions) == 2
        assert session.total_size_mb == 1500.0
        assert session.downloaded_mb == 750.0
        assert session.success_count == 1
        assert session.failure_count == 1
        assert session.average_speed_mbps == 8.5
        assert session.network_conditions.bandwidth_mbps == 25.0


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = IntelligentDownloadManager(
            output_dir=self.temp_dir,
            resume_enabled=True
        )
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_resume_workflow_with_existing_checkpoint(self):
        """Test resume workflow with existing checkpoint."""
        # Create existing checkpoint
        checkpoint = DownloadCheckpoint(
            accession="SRR123456",
            total_size_bytes=1000000,
            downloaded_bytes=500000,
            chunk_checksums={0: "abc123"},
            download_start=datetime.now() - timedelta(hours=1),
            last_progress=datetime.now() - timedelta(minutes=30),
            estimated_completion=None,
            retry_count=1,
            failure_reasons=["Network timeout"]
        )
        
        self.manager.checkpoint_manager.save_checkpoint(checkpoint)
        
        # Verify checkpoint exists
        resumable = self.manager.get_resumable_downloads()
        assert "SRR123456" in resumable
        
        # Verify checkpoint can be loaded
        loaded_checkpoint = self.manager.checkpoint_manager.load_checkpoint("SRR123456")
        assert loaded_checkpoint is not None
        assert loaded_checkpoint.downloaded_bytes == 500000
        assert loaded_checkpoint.retry_count == 1
    
    def test_bandwidth_optimization_workflow(self):
        """Test bandwidth optimization workflow."""
        # Test with limited bandwidth
        limited_manager = IntelligentDownloadManager(
            output_dir=self.temp_dir,
            max_bandwidth_mbps=10.0
        )
        
        # Allocate bandwidth
        allocated = limited_manager.bandwidth_manager.allocate_bandwidth(15.0)
        assert allocated == 10.0  # Should be limited to max
        
        # Test with unlimited bandwidth
        unlimited_manager = IntelligentDownloadManager(
            output_dir=self.temp_dir,
            max_bandwidth_mbps=None
        )
        
        allocated = unlimited_manager.bandwidth_manager.allocate_bandwidth(15.0) 
        assert allocated == 15.0  # Should get full requested amount
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid output directory
        with pytest.raises(Exception):
            invalid_manager = IntelligentDownloadManager(
                output_dir="/invalid/path/that/does/not/exist/and/cannot/be/created"
            )
        
        # Test checkpoint corruption handling
        checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        corrupt_checkpoint = checkpoint_dir / "SRR123456.checkpoint"
        
        # Create corrupt checkpoint file
        with open(corrupt_checkpoint, 'w') as f:
            f.write("corrupt data that is not a valid pickle")
        
        # Should handle corrupt checkpoint gracefully
        loaded = self.manager.checkpoint_manager.load_checkpoint("SRR123456")
        assert loaded is None
    
    def test_large_accession_list_optimization(self):
        """Test optimization with large accession lists."""
        # Create large list of accessions
        large_accession_list = [f"SRR{i:06d}" for i in range(100)]
        
        # Test order optimization
        optimized_order = self.manager.optimizer.optimize_download_order(large_accession_list)
        
        assert len(optimized_order) == 100
        assert set(optimized_order) == set(large_accession_list)  # All accessions preserved
        
        # Test time estimation
        estimate = self.manager.estimate_download_time(large_accession_list[:10])  # Test subset
        
        assert 'total_size_mb' in estimate
        assert 'estimated_time_minutes' in estimate
        assert estimate['total_size_mb'] > 0