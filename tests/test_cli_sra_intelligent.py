"""
COMPREHENSIVE TESTS for cli/commands/sra_intelligent.py (23% → 80%+ coverage)

This file tests the intelligent SRA CLI commands:
- SRAQualityProfileCommand
- SRAInteractiveDashboardCommand
- SRAComparativeAnalysisCommand

Run: pytest tests/test_cli_sra_intelligent.py -v
"""

import argparse
import json
from pathlib import Path
from unittest.mock import Mock, patch
from argparse import Namespace

from metaquest.cli.commands.sra_intelligent import (
    SRAQualityProfileCommand,
    SRAInteractiveDashboardCommand,
    SRAComparativeAnalysisCommand,
)

# Builders that return the REAL backend dataclasses, so these tests exercise the
# actual interface the CLI consumes (rather than masking mocks).
from metaquest.sra.analytics import QualityProfile, ComparativeAnalysis  # noqa: E402
from metaquest.sra.analytics import SRADatasetAnalyzer as RealSRADatasetAnalyzer  # noqa: E402


def make_profile(accession, n_content=0.0, duplication_rate=None, adapter=0.0):
    """Build a real QualityProfile."""
    return QualityProfile(
        accession=accession,
        total_reads=1000,
        total_bases=150000,
        avg_read_length=150.0,
        read_length_distribution={},
        gc_content=0.45,
        gc_histogram={},
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
# TEST CLASS: SRAQualityProfileCommand
# ============================================================================


class TestSRAQualityProfileCommand:
    """Test SRAQualityProfileCommand functionality."""

    def test_command_properties(self):
        """Test command name and help text."""
        cmd = SRAQualityProfileCommand()
        assert cmd.name == "sra_profile_quality"
        assert "sra-profile-quality" in cmd.aliases
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
        assert "--sample-size" in call_args
        assert "--sampler" in call_args

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
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        mock_profile = make_profile("SRR001", n_content=0.01, duplication_rate=0.15, adapter=0.02)

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0

        registry = json.loads((tmp_path / "metaquest_registry.json").read_text())
        analysis = registry["datasets"]["SRR001"]["analyses"]["quality"]
        assert analysis["summary"] == {"grade": "good", "total_reads": 1000, "gc_content": 0.45}
        # output/ lives under the project root (the registry's own folder), so the registry
        # records it relative to it, which keeps the project movable.
        assert analysis["output"] == "output/quality_summary.json"

    def test_execute_passes_sample_size_and_sampler_to_analyzer(self, tmp_path):
        """--sample-size/--sampler are plumbed through to profile_dataset_quality."""
        cmd = SRAQualityProfileCommand()

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
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
            sample_size=500,
            sampler="head",
        )

        mock_profile = make_profile("SRR001")

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0
        _, kwargs = mock_analyzer.profile_dataset_quality.call_args
        assert kwargs["sample_size"] == 500
        assert kwargs["sampler"] == "head"

    def test_execute_defaults_sample_size_and_sampler_when_absent(self, tmp_path):
        """A Namespace without --sample-size/--sampler (e.g. an older caller) still works."""
        cmd = SRAQualityProfileCommand()

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
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
            # sample_size / sampler intentionally absent
        )

        mock_profile = make_profile("SRR001")

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0
        _, kwargs = mock_analyzer.profile_dataset_quality.call_args
        assert kwargs["sample_size"] == 10000
        assert kwargs["sampler"] == "uniform"

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
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        def profile_for(accession, fastq_path, **kwargs):
            return make_profile(accession, n_content=0.01, duplication_rate=0.15, adapter=0.02)

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.side_effect = profile_for
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0
        # Check that detailed reports were created and are JSON-serializable
        report_path = Path(tmp_path / "output" / "SRR001_quality_profile.json")
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["quality_grade"] == "good"
        assert report["contamination_indicators"]["adapter_contamination"] == 0.02
        # A profile written with --detailed-reports round-trips through load_quality_profiles.
        assert "read_length_distribution" in report
        assert "gc_histogram" in report
        assert "technology_confidence" in report

        registry = json.loads((tmp_path / "metaquest_registry.json").read_text())
        for acc in ("SRR001", "SRR002"):
            analysis = registry["datasets"][acc]["analyses"]["quality"]
            # Same project-relative recording as above.
            assert analysis["output"] == f"output/{acc}_quality_profile.json"

    def test_execute_records_usage_in_store_catalogue(self, tmp_path):
        """Each profiled accession is also recorded as 'analysed' usage in the store catalogue."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.catalog import Catalog
        from metaquest.store.layout import init_store, store_paths

        cmd = SRAQualityProfileCommand()

        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()
        (fastq_dir / "SRR001.fastq.gz").touch()
        registry_path = tmp_path / "metaquest_registry.json"
        store_root = tmp_path / "store"
        init_store(store_root)

        registry = _load(registry_path)
        registry.project = {"id": "proj1", "name": "demo", "path": str(tmp_path), "created": "now"}
        _save(registry)

        args = Namespace(
            accession="SRR001",
            accessions_file=None,
            fastq_dir=str(fastq_dir),
            output_dir=str(tmp_path / "output"),
            detailed_reports=False,
            include_contamination=False,
            summary_only=False,
            registry=str(registry_path),
            data_root=str(store_root),
        )

        mock_profile = make_profile("SRR001", n_content=0.01, duplication_rate=0.15, adapter=0.02)
        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer
            result = cmd.execute(args)

        assert result == 0
        with Catalog(store_paths(store_root)) as catalog:
            catalog.migrate()
            row = catalog.conn.execute(
                "SELECT stage FROM usage WHERE accession = ? AND project_id = ?", ("SRR001", "proj1")
            ).fetchone()
        assert row["stage"] == "analysed"

    def test_catalog_failure_leaves_quality_outcome_unchanged(self, tmp_path):
        """A broken catalogue write never changes the quality analysis's registry outcome or exit code."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.layout import init_store

        cmd = SRAQualityProfileCommand()

        fastq_dir = tmp_path / "fastq"
        fastq_dir.mkdir()
        (fastq_dir / "SRR001.fastq.gz").touch()
        registry_path = tmp_path / "metaquest_registry.json"
        store_root = tmp_path / "store"
        init_store(store_root)

        registry = _load(registry_path)
        registry.project = {"id": "proj1", "name": "demo", "path": str(tmp_path), "created": "now"}
        _save(registry)

        args = Namespace(
            accession="SRR001",
            accessions_file=None,
            fastq_dir=str(fastq_dir),
            output_dir=str(tmp_path / "output"),
            detailed_reports=False,
            include_contamination=False,
            summary_only=False,
            registry=str(registry_path),
            data_root=str(store_root),
        )

        mock_profile = make_profile("SRR001", n_content=0.01, duplication_rate=0.15, adapter=0.02)
        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.profile_dataset_quality.return_value = mock_profile
            mock_analyzer_class.return_value = mock_analyzer
            with patch("metaquest.store.usage.catalog_write", side_effect=RuntimeError("locked")):
                result = cmd.execute(args)

        assert result == 0
        registry_after = json.loads(registry_path.read_text())
        analysis = registry_after["datasets"]["SRR001"]["analyses"]["quality"]
        assert analysis["summary"] == {"grade": "good", "total_reads": 1000, "gc_content": 0.45}

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
            registry=str(tmp_path / "metaquest_registry.json"),
        )

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.find_fastq.return_value = None
            mock_analyzer_class.return_value = mock_analyzer
            result = cmd.execute(args)
            # No FASTQ files -> profiling never attempted
            mock_analyzer.profile_dataset_quality.assert_not_called()

        assert result == 1
        summary = json.loads((tmp_path / "output" / "quality_summary.json").read_text())
        assert summary["total_analyzed"] == 0
        assert summary["failed_accessions"] == ["SRR404"]
        assert summary["summary_stats"] is None

    def test_execute_locates_fastq_via_find_fastq_avoiding_prefix_collision(self, tmp_path):
        """SRR1 and SRR10 must not collide: each accession profiles its own FASTQ file.

        The old glob("**/{accession}*.fastq*") matched "SRR1" against SRR10's
        directory too (since "SRR10..." starts with "SRR1"), and did not sort its
        matches. The fix routes accession lookup through the analyzer's own
        find_fastq(), which uses exact accession boundaries and a sorted result.
        """
        cmd = SRAQualityProfileCommand()

        fastq_dir = tmp_path / "fastq"
        (fastq_dir / "SRR10").mkdir(parents=True)
        (fastq_dir / "SRR10" / "SRR10_1.fastq").write_text("x")
        (fastq_dir / "SRR10" / "SRR10_2.fastq").write_text("x")
        (fastq_dir / "SRR1_1.fastq").write_text("x")

        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR10\nSRR1\n")

        args = Namespace(
            accession=None,
            accessions_file=str(accessions_file),
            fastq_dir=str(fastq_dir),
            output_dir=str(tmp_path / "output"),
            detailed_reports=False,
            include_contamination=False,
            summary_only=True,
            registry=str(tmp_path / "metaquest_registry.json"),
            data_root=None,
        )

        mock_profile = make_profile("SRR", n_content=0.01, duplication_rate=0.15, adapter=0.02)
        original_find_fastq = RealSRADatasetAnalyzer.find_fastq

        with patch(
            "metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer.find_fastq",
            autospec=True,
            side_effect=lambda self, accession: original_find_fastq(self, accession),
        ) as mock_find_fastq:
            with patch(
                "metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer.profile_dataset_quality",
                return_value=mock_profile,
            ) as mock_profile_call:
                result = cmd.execute(args)

        assert result == 0
        # The command must locate FASTQ files through the analyzer's own find_fastq,
        # not a hand-rolled glob, so the SRR1/SRR10 collision it already guards
        # against is not reintroduced here.
        assert mock_find_fastq.call_count == 2
        called_paths = {call.args[0]: call.kwargs["fastq_path"] for call in mock_profile_call.call_args_list}
        assert called_paths["SRR10"] == str(fastq_dir / "SRR10" / "SRR10_1.fastq")
        assert called_paths["SRR1"] == str(fastq_dir / "SRR1_1.fastq")


# ============================================================================
# TEST CLASS: SRAInteractiveDashboardCommand
# ============================================================================


class TestSRAInteractiveDashboardCommand:
    """Test SRAInteractiveDashboardCommand functionality."""

    def test_command_properties(self):
        """Test command name and help text."""
        cmd = SRAInteractiveDashboardCommand()
        assert cmd.name == "sra_dashboard"
        assert "sra-dashboard" in cmd.aliases
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
            quality_profiles=None,
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "dashboards"),
            title="Test Dashboard",
            dashboard_type="quality",
            no_open=True,
        )

        mock_dashboard_path = tmp_path / "dashboards" / "dashboard.html"

        with patch("metaquest.cli.commands.sra_intelligent.SRAReportGenerator") as mock_reporter_class:
            mock_reporter = Mock()
            mock_reporter.generate_quality_dashboard.return_value = mock_dashboard_path
            mock_reporter_class.return_value = mock_reporter

            result = cmd.execute(args)

        assert result == 0

    def test_execute_uses_saved_quality_profiles(self, tmp_path):
        """A saved quality profile is loaded and reused instead of reprofiling from FASTQ."""
        cmd = SRAInteractiveDashboardCommand()

        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\n")

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        profile = make_profile("SRR001", n_content=0.01, duplication_rate=0.15, adapter=0.02)
        (profiles_dir / "SRR001_quality_profile.json").write_text(
            json.dumps(
                {
                    "accession": profile.accession,
                    "total_reads": profile.total_reads,
                    "total_bases": profile.total_bases,
                    "avg_read_length": profile.avg_read_length,
                    "read_length_distribution": profile.read_length_distribution,
                    "gc_content": profile.gc_content,
                    "gc_distribution": profile.gc_distribution,
                    "quality_distribution": profile.quality_distribution,
                    "n_content": profile.n_content,
                    "contamination_indicators": profile.contamination_indicators,
                    "complexity_score": profile.complexity_score,
                    "duplication_rate": profile.duplication_rate,
                    "technology_confidence": profile.technology_confidence,
                    "quality_grade": profile.quality_grade,
                    "recommendations": profile.recommendations,
                }
            )
        )

        args = Namespace(
            accessions_file=str(accessions_file),
            quality_profiles=str(profiles_dir),
            fastq_dir=str(tmp_path / "fastq"),  # never created: no FASTQ files exist on disk
            output_dir=str(tmp_path / "dashboards"),
            title="Test Dashboard",
            dashboard_type="quality",
            no_open=True,
        )

        mock_dashboard_path = tmp_path / "dashboards" / "dashboard.html"

        with patch("metaquest.cli.commands.sra_intelligent.SRAReportGenerator") as mock_reporter_class:
            mock_reporter = Mock()
            mock_reporter.analyzer.find_fastq.return_value = None
            mock_reporter.generate_quality_dashboard.return_value = mock_dashboard_path
            mock_reporter_class.return_value = mock_reporter

            result = cmd.execute(args)

        assert result == 0
        mock_reporter.analyzer.profile_dataset_quality.assert_not_called()
        _, kwargs = mock_reporter.generate_quality_dashboard.call_args
        assert kwargs["profiles"]["SRR001"].accession == "SRR001"
        assert kwargs["profiles"]["SRR001"].quality_grade == "good"

    def test_execute_opens_dashboard_when_not_suppressed(self, tmp_path):
        """Without --no-open the generated dashboard is passed to open_in_browser."""
        cmd = SRAInteractiveDashboardCommand()

        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            quality_profiles=None,
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "dashboards"),
            title="Test Dashboard",
            dashboard_type="quality",
            no_open=False,
        )

        mock_dashboard_path = tmp_path / "dashboards" / "dashboard.html"

        with patch("metaquest.cli.commands.sra_intelligent.SRAReportGenerator") as mock_reporter_class:
            with patch("metaquest.cli.commands.sra_intelligent.open_in_browser", return_value=True) as mock_open:
                mock_reporter = Mock()
                mock_reporter.generate_quality_dashboard.return_value = mock_dashboard_path
                mock_reporter_class.return_value = mock_reporter

                result = cmd.execute(args)

        assert result == 0
        mock_open.assert_called_once_with(mock_dashboard_path)

    def test_execute_full_dashboard(self, tmp_path):
        """Test full dashboard generation."""
        cmd = SRAInteractiveDashboardCommand()

        accessions_file = tmp_path / "accessions.txt"
        accessions_file.write_text("SRR001\n")

        args = Namespace(
            accessions_file=str(accessions_file),
            quality_profiles=None,
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "dashboards"),
            title="Full Dashboard",
            dashboard_type="full",
            no_open=True,
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
        assert cmd.name == "sra_compare"
        assert "sra-compare" in cmd.aliases
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
        assert "--quality-profiles" in call_args

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

    def test_execute_reuses_saved_quality_profiles(self, tmp_path):
        """--quality-profiles is loaded and reused instead of reprofiling every accession."""
        cmd = SRAComparativeAnalysisCommand()

        groups_file = tmp_path / "groups.json"
        groups_data = {"Group_A": ["SRR001"], "Group_B": ["SRR002"]}
        groups_file.write_text(json.dumps(groups_data))

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        for acc in ("SRR001", "SRR002"):
            profile = make_profile(acc)
            (profiles_dir / f"{acc}_quality_profile.json").write_text(
                json.dumps(
                    {
                        "accession": profile.accession,
                        "total_reads": profile.total_reads,
                        "total_bases": profile.total_bases,
                        "avg_read_length": profile.avg_read_length,
                        "read_length_distribution": profile.read_length_distribution,
                        "gc_content": profile.gc_content,
                        "gc_histogram": profile.gc_histogram,
                        "quality_distribution": profile.quality_distribution,
                        "n_content": profile.n_content,
                        "contamination_indicators": profile.contamination_indicators,
                        "complexity_score": profile.complexity_score,
                        "duplication_rate": profile.duplication_rate,
                        "technology_confidence": profile.technology_confidence,
                        "quality_grade": profile.quality_grade,
                        "recommendations": profile.recommendations,
                    }
                )
            )

        args = Namespace(
            groups_file=str(groups_file),
            quality_profiles=str(profiles_dir),
            fastq_dir=str(tmp_path / "fastq"),  # never created: no FASTQ files exist on disk
            output_dir=str(tmp_path / "output"),
            statistical_tests=False,
            generate_report=False,
        )

        mock_comparison = ComparativeAnalysis(
            dataset_groups=groups_data,
            summary_statistics={},
            statistical_tests={},
            outlier_datasets=[],
            clustering_results=None,
            batch_effects={},
            recommendations=[],
            visualization_data={},
        )

        with patch("metaquest.cli.commands.sra_intelligent.SRADatasetAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.find_fastq.return_value = None
            mock_analyzer.compare_datasets.return_value = mock_comparison
            mock_analyzer_class.return_value = mock_analyzer

            result = cmd.execute(args)

        assert result == 0
        mock_analyzer.profile_dataset_quality.assert_not_called()
        _, kwargs = mock_analyzer.compare_datasets.call_args
        assert set(kwargs["profiles"].keys()) == {"SRR001", "SRR002"}


# ============================================================================
# TEST CLASS: Integration against REAL backend dataclasses
#
# These tests construct the actual QualityProfile / ComparativeAnalysis objects
# returned by the sra/ backend (not mocks), to guard against the CLI drifting
# away from the real dataclass interface.
# ============================================================================


class TestRealBackendInterface:
    """Drive the CLI summary helpers with the real backend dataclasses."""

    def _real_profile(self, accession, n_content, dup_rate, adapter):
        from metaquest.sra.analytics import QualityProfile

        return QualityProfile(
            accession=accession,
            total_reads=1000000,
            total_bases=150000000,
            avg_read_length=150.0,
            read_length_distribution={},
            gc_content=0.45,
            gc_histogram={},
            quality_distribution={"excellent_q30+": 0.9},
            n_content=n_content,
            contamination_indicators={"adapter_contamination": adapter},
            complexity_score=0.85,
            duplication_rate=dup_rate,
            technology_confidence=0.8,
            quality_grade="good",
            recommendations=[],
        )

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
# TEST CLASS: Honest exit codes without real FASTQ data
# ============================================================================


class TestHonestExits:
    def test_compare_returns_1_without_fastq(self, tmp_path):
        from metaquest.cli.commands.sra_intelligent import SRAComparativeAnalysisCommand

        groups = tmp_path / "groups.json"
        groups.write_text('{"a": ["SRR000001"], "b": ["SRR000002"]}')
        args = argparse.Namespace(
            groups_file=str(groups),
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "out"),
            statistical_tests=False,
            generate_report=False,
        )
        assert SRAComparativeAnalysisCommand().execute(args) == 1

    def test_dashboard_returns_1_without_fastq(self, tmp_path):
        from metaquest.cli.commands.sra_intelligent import SRAInteractiveDashboardCommand

        acc_file = tmp_path / "acc.txt"
        acc_file.write_text("SRR000001\n")
        args = argparse.Namespace(
            accessions_file=str(acc_file),
            quality_profiles=None,
            fastq_dir=str(tmp_path / "fastq"),
            output_dir=str(tmp_path / "dash"),
            title="t",
            dashboard_type="quality",
            no_open=True,
        )
        assert SRAInteractiveDashboardCommand().execute(args) == 1


# ============================================================================
# SUCCESS METRICS:
#
# After running these tests:
# - Expected: 45+ tests pass
# - Coverage: 23% → 80%+ for cli/commands/sra_intelligent.py
# - All 3 CLI commands tested
#
# Run tests:
#   pytest tests/test_cli_sra_intelligent.py -v
#
# Check coverage:
#   pytest --cov=metaquest.cli.commands.sra_intelligent --cov-report=term-missing \
#          tests/test_cli_sra_intelligent.py
# ============================================================================
