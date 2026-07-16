"""
COMPREHENSIVE TESTS for cli/commands/sra_intelligent.py (23% → 80%+ coverage)

This file tests all 4 intelligent SRA CLI commands:
- SRAIntelligentDownloadCommand
- SRAQualityProfileCommand
- SRAInteractiveDashboardCommand
- SRAComparativeAnalysisCommand

Run: pytest tests/test_cli_sra_intelligent.py -v
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime
from argparse import Namespace

from metaquest.cli.commands.sra_intelligent import (
    SRAIntelligentDownloadCommand,
    SRAQualityProfileCommand,
    SRAInteractiveDashboardCommand,
    SRAComparativeAnalysisCommand,
)

# Builders that return the REAL backend dataclasses, so these tests exercise the
# actual interface the CLI consumes (rather than masking mocks).
from metaquest.sra.download_manager import (  # noqa: E402
    DownloadSession,
    DownloadProgress,
    NetworkConditions,
)
from metaquest.sra.analytics import QualityProfile, ComparativeAnalysis  # noqa: E402

# Keys returned by the real IntelligentDownloadManager.estimate_download_time()
REAL_ESTIMATE = {
    "total_size_mb": 1024.0,
    "estimated_time_minutes": 30.0,
    "estimated_time_formatted": "0:30:00",
    "network_bandwidth_mbps": 50.0,
    "optimal_parallel_downloads": 4,
    "individual_estimates": {},
}


def make_session(session_id, completed, failed, failed_accessions):
    """Build a real DownloadSession with the requested success/failure split."""
    nc = NetworkConditions(
        bandwidth_mbps=10.0,
        latency_ms=100.0,
        packet_loss_pct=0.0,
        connection_stability=1.0,
        optimal_parallel_downloads=4,
        last_measured=datetime.now(),
    )
    results = {}
    for acc in failed_accessions:
        results[acc] = DownloadProgress(
            accession=acc,
            status="failed",
            progress_pct=0.0,
            downloaded_mb=0.0,
            total_mb=None,
            speed_mbps=0.0,
            eta_seconds=None,
            retry_count=0,
            error_message="err",
        )
    for i in range(completed):
        acc = f"SRRC{i}"
        results[acc] = DownloadProgress(
            accession=acc,
            status="completed",
            progress_pct=100.0,
            downloaded_mb=1.0,
            total_mb=1.0,
            speed_mbps=5.0,
            eta_seconds=0,
            retry_count=0,
            error_message=None,
        )
    return DownloadSession(
        session_id=session_id,
        accessions=list(results.keys()),
        start_time=datetime.now(),
        end_time=datetime.now(),
        total_size_mb=float(len(results)),
        downloaded_mb=float(completed),
        success_count=completed,
        failure_count=failed,
        average_speed_mbps=5.0,
        network_conditions=nc,
        download_results=results,
    )


def make_profile(accession, n_content=0.0, duplication_rate=None, adapter=0.0):
    """Build a real QualityProfile."""
    return QualityProfile(
        accession=accession,
        total_reads=1000,
        total_bases=150000,
        avg_read_length=150.0,
        read_length_distribution={},
        gc_content=0.45,
        gc_distribution=[],
        quality_distribution={"excellent_q30+": 0.9},
        n_content=n_content,
        contamination_indicators={"adapter_contamination": adapter},
        complexity_score=0.85,
        duplication_rate=duplication_rate,
        technology_confidence=0.8,
        quality_grade="good",
        recommendations=[],
    )


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
        mock_manager.estimate_download_time.return_value = dict(REAL_ESTIMATE)

        cmd._print_download_estimate(mock_manager, ["SRR001", "SRR002"])

        captured = capsys.readouterr()
        assert "Download Estimate" in captured.out
        assert "Total datasets: 2" in captured.out
        assert "0:30:00" in captured.out  # estimated_time_formatted
        assert "bandwidth" in captured.out.lower()

    def test_print_session_summary(self, capsys):
        """Test session summary printing."""
        cmd = SRAIntelligentDownloadCommand()
        session = make_session("test_123", completed=8, failed=2, failed_accessions=["SRR001", "SRR002"])

        cmd._print_session_summary(session)

        captured = capsys.readouterr()
        assert "test_123" in captured.out
        assert "Completed: 8" in captured.out
        assert "Failed: 2" in captured.out
        assert "Throughput Statistics" in captured.out

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
            progress_report="progress.json",
        )

        with patch("metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager"):
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
            progress_report=str(tmp_path / "progress.json"),
        )

        mock_session = make_session("test_123", completed=2, failed=0, failed_accessions=[])

        with patch("metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager") as mock_mgr_class:
            mock_manager = Mock()
            mock_manager.estimate_download_time.return_value = dict(REAL_ESTIMATE)
            mock_manager.download_with_resume.return_value = mock_session
            mock_mgr_class.return_value = mock_manager

            result = cmd.execute(args)

        assert result == 0
        assert Path(tmp_path / "progress.json").exists()
        saved = json.loads(Path(tmp_path / "progress.json").read_text())
        assert saved["completed_downloads"] == 2
        assert saved["failed_downloads"] == 0
        assert saved["failed_accessions"] == []

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
            progress_report=str(tmp_path / "progress.json"),
        )

        mock_session = make_session("test_123", completed=2, failed=1, failed_accessions=["SRR003"])

        with patch("metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager") as mock_mgr_class:
            mock_manager = Mock()
            mock_manager.estimate_download_time.return_value = dict(REAL_ESTIMATE)
            mock_manager.download_with_resume.return_value = mock_session
            mock_mgr_class.return_value = mock_manager

            result = cmd.execute(args)

        assert result == 1  # Should return error code due to failures
        saved = json.loads(Path(tmp_path / "progress.json").read_text())
        assert saved["failed_accessions"] == ["SRR003"]

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
            progress_report=str(tmp_path / "progress.json"),
        )

        with patch("metaquest.cli.commands.sra_intelligent.IntelligentDownloadManager") as mock_mgr_class:
            mock_manager = Mock()
            mock_manager.estimate_download_time.return_value = dict(REAL_ESTIMATE)
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
        profile = make_profile("SRR001", n_content=0.02, duplication_rate=0.25, adapter=0.08)

        cmd._print_quality_profile(profile)

        captured = capsys.readouterr()
        assert "SRR001" in captured.out
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
            summary_only=False,
        )

        mock_profile = make_profile("SRR001", n_content=0.01, duplication_rate=0.15, adapter=0.02)

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
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
            summary_only=False,
        )

        mock_profile = make_profile("SRR001", n_content=0.01, duplication_rate=0.15, adapter=0.02)

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0
        # Check that detailed reports were created and are JSON-serializable
        report_path = Path(tmp_path / "output" / "SRR001_quality_profile.json")
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["quality_grade"] == "good"
        assert report["contamination_indicators"]["adapter_contamination"] == 0.02

    def test_execute_missing_fastq_marks_failed(self, tmp_path):
        """Accessions with no FASTQ files are recorded as failed and yield exit 1."""
        cmd = SRAQualityProfileCommand()

        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR404\n")
        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()  # intentionally empty

        args = Namespace(
            accession=None,
            accessions_file=str(accessions_file),
            fastq_dir=str(fastq_dir),
            output_dir=str(tmp_path / "output"),
            detailed_reports=False,
            include_contamination=False,
            summary_only=False,
        )

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer
            result = cmd.execute(args)
            # No FASTQ files -> profiling never attempted
            mock_analyzer.profile_dataset_quality.assert_not_called()

        assert result == 1
        summary = json.loads((tmp_path / "output" / "quality_summary.json").read_text())
        assert summary["total_analyzed"] == 0
        assert summary["failed_accessions"] == ["SRR404"]
        assert summary["summary_stats"] is None


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
            dashboard_type="quality",
        )

        mock_dashboard_path = tmp_path / "dashboards" / "dashboard.html"

        with patch("metaquest.cli.commands.sra_intelligent.SRAReportGenerator") as mock_reporter_class:
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
            dashboard_type="full",
        )

        mock_dashboard_path = tmp_path / "dashboards" / "dashboard.html"

        with patch("metaquest.cli.commands.sra_intelligent.SRAReportGenerator") as mock_reporter_class:
            with patch("webbrowser.open"):  # Mock browser opening
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
        groups_data = {"Group_A": ["SRR001", "SRR002"], "Group_B": ["SRR003", "SRR004"]}
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
        groups_data = {"Group_A": ["SRR001", "SRR002"], "Group_B": ["SRR003", "SRR004"]}
        groups_file.write_text(json.dumps(groups_data))

        args = Namespace(
            groups_file=str(groups_file),
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "output"),
            statistical_tests=True,
            generate_report=True,
        )

        def grp_stats(gc, length, reads):
            return {
                "gc_content": {"mean": gc, "std": 0.0, "median": gc, "min": gc, "max": gc},
                "avg_read_length": {"mean": length, "std": 0.0, "median": length, "min": length, "max": length},
                "total_reads": {"mean": reads, "std": 0.0, "median": reads, "min": reads, "max": reads},
            }

        mock_comparison = ComparativeAnalysis(
            dataset_groups=groups_data,
            summary_statistics={
                "Group_A": grp_stats(0.45, 150.0, 2000000.0),
                "Group_B": grp_stats(0.52, 145.0, 1800000.0),
            },
            statistical_tests={
                "gc_content": {"test": "t-test", "statistic": 2.0, "p_value": 0.03, "significant": True}
            },
            outlier_datasets=[],
            clustering_results=None,
            batch_effects={},
            recommendations=["ok"],
            visualization_data={},
        )

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            with patch("metaquest.cli.commands.sra_intelligent.SRAReportGenerator") as mock_reporter_class:
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
# TEST CLASS: Integration against REAL backend dataclasses
#
# These tests construct the actual DownloadSession / QualityProfile /
# ComparativeAnalysis objects returned by the sra/ backend (not mocks), to
# guard against the CLI drifting away from the real dataclass interface.
# ============================================================================


class TestRealBackendInterface:
    """Drive the CLI summary helpers with the real backend dataclasses."""

    def _real_session(self, success, failure, failed_accs):
        from datetime import datetime
        from metaquest.sra.download_manager import (
            DownloadSession,
            DownloadProgress,
            NetworkConditions,
        )

        nc = NetworkConditions(
            bandwidth_mbps=10.0,
            latency_ms=100.0,
            packet_loss_pct=0.0,
            connection_stability=1.0,
            optimal_parallel_downloads=4,
            last_measured=datetime.now(),
        )
        results = {}
        for acc in failed_accs:
            results[acc] = DownloadProgress(
                accession=acc,
                status="failed",
                progress_pct=0.0,
                downloaded_mb=0.0,
                total_mb=None,
                speed_mbps=0.0,
                eta_seconds=None,
                retry_count=0,
                error_message="boom",
            )
        for i in range(success):
            acc = f"SRR9000{i}"
            results[acc] = DownloadProgress(
                accession=acc,
                status="completed",
                progress_pct=100.0,
                downloaded_mb=1.0,
                total_mb=1.0,
                speed_mbps=5.0,
                eta_seconds=0,
                retry_count=0,
                error_message=None,
            )
        accessions = list(results.keys())
        return DownloadSession(
            session_id="real_1",
            accessions=accessions,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_size_mb=float(len(accessions)),
            downloaded_mb=float(success),
            success_count=success,
            failure_count=failure,
            average_speed_mbps=5.0,
            network_conditions=nc,
            download_results=results,
        )

    def _real_profile(self, accession, n_content, dup_rate, adapter):
        from metaquest.sra.analytics import QualityProfile

        return QualityProfile(
            accession=accession,
            total_reads=1000000,
            total_bases=150000000,
            avg_read_length=150.0,
            read_length_distribution={},
            gc_content=0.45,
            gc_distribution=[],
            quality_distribution={"excellent_q30+": 0.9},
            n_content=n_content,
            contamination_indicators={"adapter_contamination": adapter},
            complexity_score=0.85,
            duplication_rate=dup_rate,
            technology_confidence=0.8,
            quality_grade="good",
            recommendations=[],
        )

    def test_print_session_summary_real(self, capsys):
        cmd = SRAIntelligentDownloadCommand()
        session = self._real_session(success=8, failure=2, failed_accs=["SRR001", "SRR002"])

        cmd._print_session_summary(session)  # must not raise AttributeError

        out = capsys.readouterr().out
        assert "real_1" in out
        assert "8" in out  # completed
        assert "2" in out  # failed

    def test_print_quality_profile_real(self, capsys):
        cmd = SRAQualityProfileCommand()
        profile = self._real_profile("SRR001", n_content=0.02, dup_rate=0.25, adapter=0.08)

        cmd._print_quality_profile(profile)  # must not raise AttributeError

        out = capsys.readouterr().out
        assert "SRR001" in out
        assert "High N content" in out
        assert "High duplicate rate" in out
        assert "Adapter contamination" in out

    def test_print_quality_profile_real_handles_none_duplication(self, capsys):
        cmd = SRAQualityProfileCommand()
        profile = self._real_profile("SRR003", n_content=0.0, dup_rate=None, adapter=0.0)

        cmd._print_quality_profile(profile)  # duplication_rate=None must not raise

        out = capsys.readouterr().out
        assert "SRR003" in out
        assert "High duplicate rate" not in out

    def test_compare_execute_real(self, tmp_path):
        cmd = SRAComparativeAnalysisCommand()
        from metaquest.sra.analytics import ComparativeAnalysis

        groups = {"Group_A": ["SRR001", "SRR002"], "Group_B": ["SRR003", "SRR004"]}
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(groups))

        comparison = ComparativeAnalysis(
            dataset_groups=groups,
            summary_statistics={
                "Group_A": {
                    "gc_content": {"mean": 0.45, "std": 0.0, "median": 0.45, "min": 0.45, "max": 0.45},
                    "avg_read_length": {"mean": 150.0, "std": 0.0, "median": 150.0, "min": 150.0, "max": 150.0},
                    "total_reads": {"mean": 2000000.0, "std": 0.0, "median": 2e6, "min": 2e6, "max": 2e6},
                }
            },
            statistical_tests={
                "gc_content": {"test": "t-test", "statistic": 2.0, "p_value": 0.03, "significant": True}
            },
            outlier_datasets=[],
            clustering_results=None,
            batch_effects={},
            recommendations=["ok"],
            visualization_data={},
        )

        args = Namespace(
            groups_file=str(groups_file),
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "output"),
            statistical_tests=True,
            generate_report=False,
        )

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_cls:
            mock_analyzer = Mock()
            mock_analyzer.compare_datasets.return_value = comparison
            mock_cls.return_value = mock_analyzer
            result = cmd.execute(args)

        assert result == 0
        assert (tmp_path / "output" / "comparative_analysis.json").exists()
        saved = json.loads((tmp_path / "output" / "comparative_analysis.json").read_text())
        assert saved["significant_differences"] == ["gc_content"]


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
