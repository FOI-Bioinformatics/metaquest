"""
STARTER TESTS for sra/reporting.py (18% → 50%+ coverage)

This file provides foundational tests for the SRA reporting module.
Run: pytest tests/test_sra_reporting_starter.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from dataclasses import dataclass

from metaquest.sra.reporting import SRAReportGenerator


# Mock data classes since we're testing reporting, not the underlying data structures
@dataclass
class MockDownloadResult:
    """Mock download result for testing."""

    status: str
    downloaded_mb: float
    progress_pct: float
    speed_mbps: float
    retry_count: int = 0


@dataclass
class MockNetworkConditions:
    """Mock network conditions."""

    bandwidth_mbps: float = 100.0


@dataclass
class MockDownloadSession:
    """Mock download session for testing."""

    session_id: str
    accessions: list
    download_results: dict
    start_time: datetime
    end_time: datetime
    network_conditions: MockNetworkConditions


@dataclass
class MockQualityProfile:
    """Mock quality profile for testing."""

    total_reads: int
    total_bases: int
    avg_read_length: float
    gc_content: float
    quality_grade: str
    complexity_score: float
    n_content: float
    contamination_indicators: dict


@dataclass
class MockProcessingRecommendations:
    """Mock processing recommendations."""

    recommended_pipeline: str
    quality_trimming: dict
    adapter_removal: dict
    computational_requirements: dict
    estimated_processing_time: str


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "reports"
    return output_dir


@pytest.fixture
def mock_download_session():
    """Create mock download session with realistic data."""
    return MockDownloadSession(
        session_id="test_session_123",
        accessions=["SRR001", "SRR002", "SRR003"],
        download_results={
            "SRR001": MockDownloadResult("completed", 500.0, 100.0, 10.5),
            "SRR002": MockDownloadResult("completed", 750.0, 100.0, 12.3),
            "SRR003": MockDownloadResult("failed", 0.0, 0.0, 0.0),
        },
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now(),
        network_conditions=MockNetworkConditions(bandwidth_mbps=100.0),
    )


@pytest.fixture
def mock_quality_profile():
    """Create mock quality profile."""
    return MockQualityProfile(
        total_reads=1000000,
        total_bases=150000000,
        avg_read_length=150.0,
        gc_content=0.45,
        quality_grade="good",
        complexity_score=0.85,
        n_content=0.02,
        contamination_indicators={"adapter_contamination": 0.03},
    )


class TestSRAReportGeneratorInit:
    """Test SRAReportGenerator initialization."""

    def test_init_creates_output_dir(self, tmp_output_dir):
        """Test that initialization creates output directory."""
        generator = SRAReportGenerator(tmp_output_dir)

        assert generator.output_dir == tmp_output_dir
        assert tmp_output_dir.exists()
        assert tmp_output_dir.is_dir()

    def test_init_with_existing_dir(self, tmp_output_dir):
        """Test initialization with existing directory."""
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        generator = SRAReportGenerator(tmp_output_dir)

        assert generator.output_dir == tmp_output_dir
        assert tmp_output_dir.exists()

    def test_analyzer_is_initialized(self, tmp_output_dir):
        """Test that analyzer is initialized."""
        generator = SRAReportGenerator(tmp_output_dir)

        assert generator.analyzer is not None


class TestDownloadSummaryGeneration:
    """Test download summary report generation."""

    def test_create_download_summary_basic(self, tmp_output_dir, mock_download_session):
        """Test basic download summary creation."""
        generator = SRAReportGenerator(tmp_output_dir)

        with patch("metaquest.sra.reporting.PLOTLY_AVAILABLE", False):
            with patch("metaquest.sra.reporting.JINJA2_AVAILABLE", False):
                result_path = generator.create_download_summary(mock_download_session, include_plots=False)

        # Verify report was created
        assert result_path.exists()
        assert result_path.suffix == ".html"
        assert "download_summary" in result_path.name
        assert mock_download_session.session_id in result_path.name

    def test_create_download_summary_with_jinja2(self, tmp_output_dir, mock_download_session):
        """Test download summary with Jinja2 templating."""
        generator = SRAReportGenerator(tmp_output_dir)

        with patch("metaquest.sra.reporting.PLOTLY_AVAILABLE", False):
            with patch("metaquest.sra.reporting.JINJA2_AVAILABLE", True):
                with patch("metaquest.sra.reporting.Environment") as mock_env:
                    mock_template = Mock()
                    mock_template.render.return_value = "<html>Test Report</html>"
                    mock_env.return_value.from_string.return_value = mock_template

                    result_path = generator.create_download_summary(mock_download_session, include_plots=False)

        assert result_path.exists()
        # Verify template was rendered
        mock_template.render.assert_called_once()

    def test_download_summary_calculates_stats(self, tmp_output_dir, mock_download_session):
        """Test that download summary calculates statistics correctly."""
        generator = SRAReportGenerator(tmp_output_dir)

        with patch("metaquest.sra.reporting.PLOTLY_AVAILABLE", False):
            with patch("metaquest.sra.reporting.JINJA2_AVAILABLE", False):
                result_path = generator.create_download_summary(mock_download_session, include_plots=False)

        # Read the generated HTML and verify statistics are present
        html_content = result_path.read_text()

        # Should contain session information
        assert mock_download_session.session_id in html_content

        # Should contain some statistics (basic checks)
        assert "Summary" in html_content or "summary" in html_content


class TestExportMetadata:
    """Test metadata export functionality."""

    def test_export_unsupported_format_raises_error(self, tmp_output_dir):
        """Test that unsupported format raises ValueError."""
        generator = SRAReportGenerator(tmp_output_dir)

        with pytest.raises(ValueError, match="Unsupported format"):
            generator.export_metadata_enriched(["SRR001"], output_format="xml")

    def test_export_empty_accessions_list(self, tmp_output_dir):
        """Test export with empty accessions list."""
        generator = SRAReportGenerator(tmp_output_dir)

        # Mock the analyzer to avoid actual API calls
        with patch.object(generator.analyzer, "profile_dataset_quality"):
            with patch.object(generator.analyzer, "recommend_processing_params"):
                result_path = generator.export_metadata_enriched([], output_format="csv")

        # Should create an empty CSV file
        assert result_path.exists()
        assert result_path.suffix == ".csv"


class TestHelperMethods:
    """Test helper methods for report generation."""

    def test_calculate_quality_summary_empty_profiles(self, tmp_output_dir):
        """Test quality summary calculation with empty profiles."""
        generator = SRAReportGenerator(tmp_output_dir)

        result = generator._calculate_quality_summary({})

        assert result == {}

    def test_calculate_quality_summary_with_data(self, tmp_output_dir, mock_quality_profile):
        """Test quality summary calculation with data."""
        generator = SRAReportGenerator(tmp_output_dir)

        profiles = {
            "SRR001": mock_quality_profile,
            "SRR002": mock_quality_profile,
        }

        result = generator._calculate_quality_summary(profiles)

        # Verify structure
        assert "total_datasets" in result
        assert result["total_datasets"] == 2
        assert "total_reads" in result
        assert "average_gc_content" in result
        assert "quality_grade_distribution" in result


class TestPlotlyIntegration:
    """Test Plotly integration for interactive plots."""

    def test_create_download_plots_without_plotly(self, tmp_output_dir, mock_download_session):
        """Test plot creation when Plotly is not available."""
        generator = SRAReportGenerator(tmp_output_dir)

        with patch("metaquest.sra.reporting.PLOTLY_AVAILABLE", False):
            plots = generator._create_download_plots(mock_download_session)

        # Should return empty dict when Plotly unavailable
        assert plots == {}

    def test_create_download_plots_with_plotly(self, tmp_output_dir, mock_download_session):
        """Test plot creation when Plotly is available."""
        generator = SRAReportGenerator(tmp_output_dir)

        with patch("metaquest.sra.reporting.PLOTLY_AVAILABLE", True):
            with patch("metaquest.sra.reporting.go.Figure") as mock_fig:
                with patch("metaquest.sra.reporting.pyo.plot", return_value="<div>Plot HTML</div>"):
                    mock_figure = Mock()
                    mock_fig.return_value = mock_figure

                    plots = generator._create_download_plots(mock_download_session)

        # Should return dict with plot HTML
        assert isinstance(plots, dict)
        # Should have created some plots
        assert len(plots) > 0

    def test_create_quality_plots_without_plotly(self, tmp_output_dir):
        """Test quality plot creation without Plotly."""
        generator = SRAReportGenerator(tmp_output_dir)

        with patch("metaquest.sra.reporting.PLOTLY_AVAILABLE", False):
            plots = generator._create_quality_plots({})

        assert plots == {}


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_generate_quality_dashboard_no_profiles(self, tmp_output_dir):
        """Test quality dashboard generation with no valid profiles."""
        generator = SRAReportGenerator(tmp_output_dir)

        # Mock analyzer to fail for all accessions
        with patch.object(generator.analyzer, "profile_dataset_quality", side_effect=Exception("API Error")):
            with pytest.raises(ValueError, match="No datasets could be profiled"):
                generator.generate_quality_dashboard(["SRR001", "SRR002"])

    def test_export_with_failing_accessions(self, tmp_output_dir, mock_quality_profile):
        """Test export handles failing accessions gracefully."""
        generator = SRAReportGenerator(tmp_output_dir)

        # Mock to succeed for first, fail for second
        mock_recommendations = MockProcessingRecommendations(
            recommended_pipeline="standard",
            quality_trimming={"enabled": True},
            adapter_removal={"enabled": False},
            computational_requirements={"memory_gb": 8},
            estimated_processing_time="2 hours",
        )

        def profile_side_effect(accession):
            if accession == "SRR001":
                return mock_quality_profile
            else:
                raise Exception("Failed to profile")

        with patch.object(generator.analyzer, "profile_dataset_quality", side_effect=profile_side_effect):
            with patch.object(generator.analyzer, "recommend_processing_params", return_value=mock_recommendations):
                result_path = generator.export_metadata_enriched(["SRR001", "SRR002"], output_format="csv")

        # Should still create file with successful ones
        assert result_path.exists()


class TestHTMLGeneration:
    """Test HTML generation methods."""

    def test_generate_simple_download_html(self, tmp_output_dir, mock_download_session):
        """Test simple HTML generation without Jinja2."""
        generator = SRAReportGenerator(tmp_output_dir)

        report_data = {
            "session": mock_download_session,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary_stats": {
                "total_accessions": 3,
                "success_rate": 0.67,
                "total_size_mb": 1250.0,
                "average_speed_mbps": 11.4,
            },
        }

        html = generator._generate_simple_download_html(report_data)

        # Verify HTML structure
        assert "<html" in html
        assert "<title>" in html
        assert mock_download_session.session_id in html
        assert "Download summary" in html

    def test_generate_simple_quality_html(self, tmp_output_dir):
        """Test simple quality HTML generation."""
        generator = SRAReportGenerator(tmp_output_dir)

        dashboard_data = {
            "title": "Test Quality Dashboard",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_datasets": 5,
            "summary_stats": {"total_reads": 5000000, "average_gc_content": 0.45, "high_contamination_count": 2},
        }

        html = generator._generate_simple_quality_html(dashboard_data)

        # Verify HTML structure
        assert "<html" in html
        assert "Test Quality Dashboard" in html
        assert "5000000" in html or "5,000,000" in html

    def test_generate_simple_comparative_html(self, tmp_output_dir):
        """Test simple comparative HTML generation."""
        generator = SRAReportGenerator(tmp_output_dir)

        report_data = {
            "title": "Comparative Analysis",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "group_counts": {"Group A": 10, "Group B": 15},
        }

        html = generator._generate_simple_comparative_html(report_data)

        # Verify HTML structure
        assert "<html" in html
        assert "Comparative Analysis" in html
        assert "Group A" in html
        assert "datasets" in html


# ============================================================================
# SUCCESS METRICS:
#
# After running these starter tests:
# - Expected: 18 tests pass
# - Coverage: 18% → 50%+ for sra/reporting.py
# - Key methods tested: init, download summary, export, helpers
#
# Run tests:
#   pytest tests/test_sra_reporting_starter.py -v
#
# Check coverage:
#   pytest --cov=metaquest.sra.reporting --cov-report=term-missing \
#          tests/test_sra_reporting_starter.py
# ============================================================================
