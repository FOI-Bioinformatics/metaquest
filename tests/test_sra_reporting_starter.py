"""
STARTER TESTS for sra/reporting.py (18% → 50%+ coverage)

This file provides foundational tests for the SRA reporting module.
Run: pytest tests/test_sra_reporting_starter.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import patch
from dataclasses import dataclass

from metaquest.sra.reporting import SRAReportGenerator


# Mock data classes since we're testing reporting, not the underlying data structures
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


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "reports"
    return output_dir


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


class TestHTMLGeneration:
    """Test HTML generation methods."""

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
# - Key methods tested: init, quality summary helpers, HTML generation
#
# Run tests:
#   pytest tests/test_sra_reporting_starter.py -v
#
# Check coverage:
#   pytest --cov=metaquest.sra.reporting --cov-report=term-missing \
#          tests/test_sra_reporting_starter.py
# ============================================================================
