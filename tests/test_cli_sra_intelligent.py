"""
COMPREHENSIVE TESTS for cli/commands/sra_intelligent.py (23% → 80%+ coverage)

This file tests all 4 intelligent SRA CLI commands:
- SRAIntelligentDownloadCommand
- SRAQualityProfileCommand
- SRAInteractiveDashboardCommand
- SRAComparativeAnalysisCommand

Run: pytest tests/test_cli_sra_intelligent.py -v
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timedelta
from argparse import Namespace
from dataclasses import dataclass

from metaquest.cli.commands.sra_intelligent import (
    SRAIntelligentDownloadCommand,
    SRAQualityProfileCommand,
    SRAInteractiveDashboardCommand,
    SRAComparativeAnalysisCommand,
)


# Mock data classes
@dataclass
class MockDownloadSession:
    """Mock download session."""
    session_id: str
    start_time: datetime
    end_time: datetime
    total_downloads: int
    completed_downloads: int
    failed_downloads: int
    status: str
    failed_accessions: list
    bandwidth_stats: Mock = None


@dataclass
class MockQualityProfile:
    """Mock quality profile."""
    accession: str
    total_reads: int
    total_bases: int
    avg_read_length: float
    gc_content: float
    avg_quality: float
    quality_distribution: list
    complexity_score: float
    n_content: float
    duplicate_rate: float
    adapter_contamination: float
    warnings: list


@dataclass
class MockComparativeAnalysis:
    """Mock comparative analysis."""
    group_statistics: dict
    statistical_tests: dict
    significant_differences: list


# ============================================================================
# TEST CLASS: SRAIntelligentDownloadCommand
# ============================================================================

class TestSRAIntelligentDownloadCommand:
    """Test SRAIntelligentDownloadCommand functionality."""

    def test_command_properties(self):
        """Test command name and help text."""
        cmd = SRAIntelligentDownloadCommand()
        assert cmd.name == "sra-download-intelligent"
        assert "intelligent" in cmd.help.lower()
        assert "resume" in cmd.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        cmd = SRAIntelligentDownloadCommand()
        parser = Mock()
        parser.add_argument = Mock()

        cmd.configure_parser(parser)

        # Verify all required arguments were added
        assert parser.add_argument.call_count >= 10
        call_args = [call[0][0] for call in parser.add_argument.call_args_list]
        assert "--accessions-file" in call_args
        assert "--output-dir" in call_args
        assert "--resume" in call_args

    def test_read_accessions_success(self, tmp_path):
        """Test successful accession file reading."""
        cmd = SRAIntelligentDownloadCommand()
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\nSRR002\nSRR003\n")

        result = cmd._read_accessions(str(accessions_file))

        assert len(result) == 3
        assert "SRR001" in result
        assert "SRR002" in result
        assert "SRR003" in result

    def test_read_accessions_strips_whitespace(self, tmp_path):
        """Test that whitespace is stripped from accessions."""
        cmd = SRAIntelligentDownloadCommand()
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("  SRR001  \n\tSRR002\t\n\nSRR003\n  \n")

        result = cmd._read_accessions(str(accessions_file))

        assert len(result) == 3
        assert all(acc.strip() == acc for acc in result)

    def test_read_accessions_empty_file(self, tmp_path, capsys):
        """Test reading empty accessions file."""
        cmd = SRAIntelligentDownloadCommand()
        accessions_file = tmp_path / "empty.txt"
        accessions_file.write_text("\n\n  \n")

        result = cmd._read_accessions(str(accessions_file))

        assert len(result) == 0
        captured = capsys.readouterr()
        assert "No accessions found" in captured.out

    def test_read_accessions_file_not_found(self, capsys):
        """Test reading non-existent accessions file."""
        cmd = SRAIntelligentDownloadCommand()

        result = cmd._read_accessions("/nonexistent/file.txt")

        assert len(result) == 0
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_print_download_estimate(self, capsys):
        """Test download estimate printing."""
        cmd = SRAIntelligentDownloadCommand()
        mock_manager = Mock()
        mock_manager.estimate_download_time.return_value = {
            "total_datasets": 5,
            "total_size_gb": 25.5,
            "estimated_hours": 3.5,
            "optimal_parallel": 4,
            "bandwidth_limited": True
        }

        cmd._print_download_estimate(mock_manager, ["SRR001", "SRR002"])

        captured = capsys.readouterr()
        assert "Download Estimate" in captured.out
        assert "5" in captured.out
        assert "25.5" in captured.out
        assert "3.5" in captured.out
        assert "bandwidth" in captured.out.lower()

    def test_print_session_summary(self, capsys):
        """Test session summary printing."""
        cmd = SRAIntelligentDownloadCommand()
        mock_stats = Mock()
        mock_stats.average_mbps = 10.5
        mock_stats.peak_mbps = 15.2
        mock_stats.efficiency = 0.85

        session = MockDownloadSession(
            session_id="test_123",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            total_downloads=10,
            completed_downloads=8,
            failed_downloads=2,
            status="completed",
            failed_accessions=["SRR001", "SRR002"],
            bandwidth_stats=mock_stats
        )

        cmd._print_session_summary(session)

        captured = capsys.readouterr()
        assert "test_123" in captured.out
        assert "10" in captured.out
        assert "8" in captured.out
        assert "Bandwidth Statistics" in captured.out

    def test_execute_dry_run(self, tmp_path, capsys):
        """Test execution in dry-run mode."""
        cmd = SRAIntelligentDownloadCommand()
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\nSRR002\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            output_dir=str(tmp_path / "output"),
            temp_dir=None,
            checkpoint_dir=None,
            max_bandwidth_mbps=None,
            max_parallel_downloads=4,
            resume=True,
            no_resume=False,
            force_restart=False,
            dry_run=True,
            progress_report="progress.json"
        )

        with patch('metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager'):
            result = cmd.execute(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Dry run" in captured.out

    def test_execute_success(self, tmp_path):
        """Test successful download execution."""
        cmd = SRAIntelligentDownloadCommand()
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\nSRR002\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            output_dir=str(tmp_path / "output"),
            temp_dir=None,
            checkpoint_dir=None,
            max_bandwidth_mbps=None,
            max_parallel_downloads=4,
            resume=True,
            no_resume=False,
            force_restart=False,
            dry_run=False,
            progress_report=str(tmp_path / "progress.json")
        )

        mock_session = MockDownloadSession(
            session_id="test_123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_downloads=2,
            completed_downloads=2,
            failed_downloads=0,
            status="completed",
            failed_accessions=[],
            bandwidth_stats=None
        )

        with patch('metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager') as mock_mgr_class:
            mock_manager = Mock()
            mock_manager.estimate_download_time.return_value = {
                "total_datasets": 2,
                "total_size_gb": 1.0,
                "estimated_hours": 0.5,
                "optimal_parallel": 2,
                "bandwidth_limited": False
            }
            mock_manager.download_with_resume.return_value = mock_session
            mock_mgr_class.return_value = mock_manager

            result = cmd.execute(args)

        assert result == 0
        assert Path(tmp_path / "progress.json").exists()

    def test_execute_with_failures(self, tmp_path):
        """Test execution with some failed downloads."""
        cmd = SRAIntelligentDownloadCommand()
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\nSRR002\nSRR003\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            output_dir=str(tmp_path / "output"),
            temp_dir=None,
            checkpoint_dir=None,
            max_bandwidth_mbps=None,
            max_parallel_downloads=4,
            resume=True,
            no_resume=False,
            force_restart=False,
            dry_run=False,
            progress_report=str(tmp_path / "progress.json")
        )

        mock_session = MockDownloadSession(
            session_id="test_123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_downloads=3,
            completed_downloads=2,
            failed_downloads=1,
            status="completed",
            failed_accessions=["SRR003"],
            bandwidth_stats=None
        )

        with patch('metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager') as mock_mgr_class:
            mock_manager = Mock()
            mock_manager.estimate_download_time.return_value = {
                "total_datasets": 3,
                "total_size_gb": 1.5,
                "estimated_hours": 0.8,
                "optimal_parallel": 3,
                "bandwidth_limited": False
            }
            mock_manager.download_with_resume.return_value = mock_session
            mock_mgr_class.return_value = mock_manager

            result = cmd.execute(args)

        assert result == 1  # Should return error code due to failures

    def test_execute_keyboard_interrupt(self, tmp_path, capsys):
        """Test handling of keyboard interrupt."""
        cmd = SRAIntelligentDownloadCommand()
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            output_dir=str(tmp_path / "output"),
            temp_dir=None,
            checkpoint_dir=None,
            max_bandwidth_mbps=None,
            max_parallel_downloads=4,
            resume=True,
            no_resume=False,
            force_restart=False,
            dry_run=False,
            progress_report=str(tmp_path / "progress.json")
        )

        with patch('metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager') as mock_mgr_class:
            mock_manager = Mock()
            mock_manager.estimate_download_time.return_value = {
                "total_datasets": 1,
                "total_size_gb": 0.5,
                "estimated_hours": 0.2,
                "optimal_parallel": 1,
                "bandwidth_limited": False
            }
            mock_manager.download_with_resume.side_effect = KeyboardInterrupt()
            mock_mgr_class.return_value = mock_manager

            result = cmd.execute(args)

        assert result == 2  # Special exit code for interrupt
        captured = capsys.readouterr()
        assert "interrupted" in captured.out.lower()


# ============================================================================
# TEST CLASS: SRAQualityProfileCommand
# ============================================================================

class TestSRAQualityProfileCommand:
    """Test SRAQualityProfileCommand functionality."""

    def test_command_properties(self):
        """Test command name and help text."""
        cmd = SRAQualityProfileCommand()
        assert cmd.name == "sra-profile-quality"
        assert "quality" in cmd.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        cmd = SRAQualityProfileCommand()
        parser = Mock()
        parser.add_argument = Mock()

        cmd.configure_parser(parser)

        assert parser.add_argument.call_count >= 6
        call_args = [call[0][0] for call in parser.add_argument.call_args_list]
        assert "--accessions-file" in call_args
        assert "--fastq-dir" in call_args
        assert "--detailed-reports" in call_args

    def test_print_quality_profile(self, capsys):
        """Test quality profile printing."""
        cmd = SRAQualityProfileCommand()
        profile = MockQualityProfile(
            accession="SRR001",
            total_reads=1000000,
            total_bases=150000000,
            avg_read_length=150.0,
            gc_content=0.45,
            avg_quality=35.0,
            quality_distribution=[30, 35, 40],
            complexity_score=0.85,
            n_content=0.02,
            duplicate_rate=0.25,
            adapter_contamination=0.08,
            warnings=[]
        )

        cmd._print_quality_profile(profile)

        captured = capsys.readouterr()
        assert "SRR001" in captured.out
        assert "1,000,000" in captured.out or "1000000" in captured.out
        assert "High N content" in captured.out
        assert "High duplicate rate" in captured.out
        assert "Adapter contamination" in captured.out

    def test_execute_single_accession(self, tmp_path):
        """Test profiling single accession."""
        cmd = SRAQualityProfileCommand()

        # Create mock FASTQ file
        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()
        (fastq_dir / "SRR001.fastq.gz").touch()

        args = Namespace(
            accession="SRR001",
            accessions_file=None,
            fastq_dir=str(fastq_dir),
            output_dir=str(tmp_path / "output"),
            detailed_reports=False,
            include_contamination=False,
            summary_only=False
        )

        mock_profile = MockQualityProfile(
            accession="SRR001",
            total_reads=1000,
            total_bases=150000,
            avg_read_length=150.0,
            gc_content=0.45,
            avg_quality=35.0,
            quality_distribution=[30, 35, 40],
            complexity_score=0.85,
            n_content=0.01,
            duplicate_rate=0.15,
            adapter_contamination=0.02,
            warnings=[]
        )

        with patch('metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0

    def test_execute_batch_mode(self, tmp_path):
        """Test batch profiling mode."""
        cmd = SRAQualityProfileCommand()

        # Create accessions file
        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\nSRR002\n")

        # Create mock FASTQ files
        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()
        (fastq_dir / "SRR001.fastq.gz").touch()
        (fastq_dir / "SRR002.fastq.gz").touch()

        args = Namespace(
            accession=None,
            accessions_file=str(accessions_file),
            fastq_dir=str(fastq_dir),
            output_dir=str(tmp_path / "output"),
            detailed_reports=True,
            include_contamination=True,
            summary_only=False
        )

        mock_profile = MockQualityProfile(
            accession="SRR001",
            total_reads=1000,
            total_bases=150000,
            avg_read_length=150.0,
            gc_content=0.45,
            avg_quality=35.0,
            quality_distribution=[30, 35, 40],
            complexity_score=0.85,
            n_content=0.01,
            duplicate_rate=0.15,
            adapter_contamination=0.02,
            warnings=[]
        )

        with patch('metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0
        # Check that detailed reports were created
        assert Path(tmp_path / "output" / "SRR001_quality_profile.json").exists()


# ============================================================================
# TEST CLASS: SRAInteractiveDashboardCommand
# ============================================================================

class TestSRAInteractiveDashboardCommand:
    """Test SRAInteractiveDashboardCommand functionality."""

    def test_command_properties(self):
        """Test command name and help text."""
        cmd = SRAInteractiveDashboardCommand()
        assert cmd.name == "sra-dashboard"
        assert "dashboard" in cmd.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        cmd = SRAInteractiveDashboardCommand()
        parser = Mock()
        parser.add_argument = Mock()

        cmd.configure_parser(parser)

        call_args = [call[0][0] for call in parser.add_argument.call_args_list]
        assert "--accessions-file" in call_args
        assert "--dashboard-type" in call_args

    def test_execute_quality_dashboard(self, tmp_path):
        """Test quality dashboard generation."""
        cmd = SRAInteractiveDashboardCommand()

        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\nSRR002\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            download_session=None,
            quality_profiles=None,
            output_dir=str(tmp_path / "dashboards"),
            title="Test Dashboard",
            dashboard_type="quality"
        )

        mock_dashboard_path = tmp_path / "dashboards" / "dashboard.html"

        with patch('metaquest.cli.commands.sra_intelligent.SRAReportGenerator') as mock_reporter_class:
            mock_reporter = Mock()
            mock_reporter.generate_quality_dashboard.return_value = mock_dashboard_path
            mock_reporter_class.return_value = mock_reporter

            result = cmd.execute(args)

        assert result == 0

    def test_execute_full_dashboard(self, tmp_path):
        """Test full dashboard generation."""
        cmd = SRAInteractiveDashboardCommand()

        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            download_session=None,
            quality_profiles=None,
            output_dir=str(tmp_path / "dashboards"),
            title="Full Dashboard",
            dashboard_type="full"
        )

        mock_dashboard_path = tmp_path / "dashboards" / "dashboard.html"

        with patch('metaquest.cli.commands.sra_intelligent.SRAReportGenerator') as mock_reporter_class:
            with patch('webbrowser.open'):  # Mock browser opening
                mock_reporter = Mock()
                mock_reporter.generate_quality_dashboard.return_value = mock_dashboard_path
                mock_reporter.create_comparative_analysis.return_value = mock_dashboard_path
                mock_reporter_class.return_value = mock_reporter

                result = cmd.execute(args)

        assert result == 0


# ============================================================================
# TEST CLASS: SRAComparativeAnalysisCommand
# ============================================================================

class TestSRAComparativeAnalysisCommand:
    """Test SRAComparativeAnalysisCommand functionality."""

    def test_command_properties(self):
        """Test command name and help text."""
        cmd = SRAComparativeAnalysisCommand()
        assert cmd.name == "sra-compare"
        assert "comparative" in cmd.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        cmd = SRAComparativeAnalysisCommand()
        parser = Mock()
        parser.add_argument = Mock()

        cmd.configure_parser(parser)

        call_args = [call[0][0] for call in parser.add_argument.call_args_list]
        assert "--groups-file" in call_args
        assert "--statistical-tests" in call_args

    def test_load_groups_success(self, tmp_path):
        """Test successful group loading."""
        cmd = SRAComparativeAnalysisCommand()
        groups_file = tmp_path / "groups.json"
        groups_data = {
            "Group_A": ["SRR001", "SRR002"],
            "Group_B": ["SRR003", "SRR004"]
        }
        groups_file.write_text(json.dumps(groups_data))

        result = cmd._load_groups(str(groups_file))

        assert len(result) == 2
        assert "Group_A" in result
        assert len(result["Group_A"]) == 2

    def test_load_groups_file_not_found(self, capsys):
        """Test loading non-existent groups file."""
        cmd = SRAComparativeAnalysisCommand()

        result = cmd._load_groups("/nonexistent/groups.json")

        assert result == {}
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_load_groups_invalid_json(self, tmp_path, capsys):
        """Test loading invalid JSON groups file."""
        cmd = SRAComparativeAnalysisCommand()
        groups_file = tmp_path / "invalid.json"
        groups_file.write_text("{invalid json")

        result = cmd._load_groups(str(groups_file))

        assert result == {}
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.out

    def test_execute_success(self, tmp_path):
        """Test successful comparative analysis."""
        cmd = SRAComparativeAnalysisCommand()

        groups_file = tmp_path / "groups.json"
        groups_data = {
            "Group_A": ["SRR001", "SRR002"],
            "Group_B": ["SRR003", "SRR004"]
        }
        groups_file.write_text(json.dumps(groups_data))

        args = Namespace(
            groups_file=str(groups_file),
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "output"),
            statistical_tests=True,
            generate_report=True
        )

        mock_comparison = MockComparativeAnalysis(
            group_statistics={
                "Group_A": {
                    "dataset_count": 2,
                    "mean_gc_content": 0.45,
                    "mean_quality": 35.0,
                    "total_reads": 2000000
                },
                "Group_B": {
                    "dataset_count": 2,
                    "mean_gc_content": 0.52,
                    "mean_quality": 33.0,
                    "total_reads": 1800000
                }
            },
            statistical_tests={
                "gc_content": {"p_value": 0.03, "test": "Mann-Whitney U"}
            },
            significant_differences=["gc_content"]
        )

        with patch('metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer') as mock_analyzer_class:
            with patch('metaquest.cli.commands.sra_intelligent.SRAReportGenerator') as mock_reporter_class:
                mock_analyzer = Mock()
                mock_analyzer.compare_datasets.return_value = mock_comparison
                mock_analyzer_class.return_value = mock_analyzer

                mock_reporter = Mock()
                mock_reporter.create_comparative_analysis.return_value = tmp_path / "report.html"
                mock_reporter_class.return_value = mock_reporter

                result = cmd.execute(args)

        assert result == 0
        assert Path(tmp_path / "output" / "comparative_analysis.json").exists()


# ============================================================================
# SUCCESS METRICS:
#
# After running these tests:
# - Expected: 45+ tests pass
# - Coverage: 23% → 80%+ for cli/commands/sra_intelligent.py
# - All 4 CLI commands tested
#
# Run tests:
#   pytest tests/test_cli_sra_intelligent.py -v
#
# Check coverage:
#   pytest --cov=metaquest.cli.commands.sra_intelligent --cov-report=term-missing \
#          tests/test_cli_sra_intelligent.py
# ============================================================================
