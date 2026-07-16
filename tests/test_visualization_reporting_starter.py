"""
STARTER TESTS for visualization/reporting.py (0% → 20% coverage)

This file provides a working foundation to start testing the reporting module.
Run: pytest tests/test_visualization_reporting_starter.py -v

After running these tests, coverage for reporting.py will increase from 0% to ~20%.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from metaquest.visualization.reporting import generate_report
from metaquest.core.exceptions import VisualizationError


@pytest.fixture
def sample_summary_data(tmp_path):
    """Create minimal containment summary data."""
    summary_file = tmp_path / "summary.tsv"
    data = pd.DataFrame(
        {
            "max_containment": [0.95, 0.87, 0.65, 0.23],
            "max_containment_annotation": [
                "Escherichia coli",
                "Salmonella enterica",
                "Klebsiella pneumoniae",
                "Pseudomonas aeruginosa",
            ],
            "GCF_000005825.2": [0.95, 0.12, 0.05, 0.02],
            "GCF_000006945.2": [0.23, 0.87, 0.15, 0.08],
            "GCF_000009605.1": [0.15, 0.25, 0.65, 0.12],
        },
        index=["SRR001", "SRR002", "SRR003", "SRR004"],
    )
    data.to_csv(summary_file, sep="\t")
    return summary_file


@pytest.fixture
def sample_metadata(tmp_path):
    """Create minimal metadata."""
    metadata_file = tmp_path / "metadata.tsv"
    data = pd.DataFrame(
        {
            "run_accession": ["SRR001", "SRR002", "SRR003", "SRR004"],
            "organism_name": ["E. coli", "Salmonella", "Klebsiella", "Pseudomonas"],
            "platform": ["ILLUMINA"] * 4,
            "library_source": ["GENOMIC"] * 4,
        }
    )
    data.to_csv(metadata_file, sep="\t", index=False)
    return metadata_file


class TestReportingErrorPaths:
    """Test error handling in report generation.

    These tests are easy wins - they test error paths with minimal mocking.
    Tests lines 48-54 of reporting.py
    """

    def test_unsupported_format_raises_error(self, sample_summary_data, tmp_path):
        """Test that unsupported format raises VisualizationError."""
        with pytest.raises(VisualizationError, match="Unsupported report format"):
            generate_report(
                title="Test Report",
                summary_file=str(sample_summary_data),
                output_file=str(tmp_path / "report.unknown"),
                format="unknown",
            )

    def test_html_without_jinja2_raises_error(self, sample_summary_data, tmp_path):
        """Test that HTML generation without jinja2 raises error."""
        with patch("metaquest.visualization.reporting.JINJA2_AVAILABLE", False):
            with pytest.raises(VisualizationError, match="requires jinja2"):
                generate_report(
                    title="Test Report",
                    summary_file=str(sample_summary_data),
                    output_file=str(tmp_path / "report.html"),
                    format="html",
                )


class TestPDFReportGeneration:
    """Test PDF report generation.

    These tests verify the PDF generation workflow with proper mocking.
    Tests lines 68-92 (generate_report → _generate_pdf_report)
    """

    def test_generate_pdf_report_minimal(self, sample_summary_data, tmp_path):
        """Test minimal PDF generation with full mocking.

        This tests the happy path: valid data, PDF format, no errors.
        """
        output_file = tmp_path / "report.pdf"

        # Mock the PDF writer
        mock_pdf_pages = MagicMock()

        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    # Mock the plotting functions that are called
                    with patch("metaquest.visualization.plots.plot_containment", return_value=mock_fig):
                        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=mock_fig):
                            result = generate_report(
                                title="Test Report",
                                summary_file=str(sample_summary_data),
                                output_file=str(output_file),
                                format="pdf",
                                threshold=0.1,
                                include_plots=True,
                                include_tables=True,
                            )

        # Verify the result
        assert result == output_file

        # Verify PdfPages was used as a context manager
        mock_pdf_pages.__enter__.assert_called_once()
        mock_pdf_pages.__exit__.assert_called_once()

    def test_generate_pdf_with_metadata(self, sample_summary_data, sample_metadata, tmp_path):
        """Test PDF generation with optional metadata file."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.plots.plot_containment", return_value=mock_fig):
                        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=mock_fig):
                            result = generate_report(
                                title="Test Report",
                                summary_file=str(sample_summary_data),
                                metadata_file=str(sample_metadata),
                                output_file=str(output_file),
                                format="pdf",
                            )

        assert result == output_file

    def test_generate_pdf_without_plots(self, sample_summary_data, tmp_path):
        """Test PDF generation with plots disabled."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    # Should NOT call plotting functions
                    with patch("metaquest.visualization.plots.plot_containment") as mock_plot:
                        result = generate_report(
                            title="Test Report",
                            summary_file=str(sample_summary_data),
                            output_file=str(output_file),
                            format="pdf",
                            include_plots=False,
                            include_tables=True,
                        )

        assert result == output_file
        # Verify plotting functions were NOT called
        mock_plot.assert_not_called()

    def test_generate_pdf_without_tables(self, sample_summary_data, tmp_path):
        """Test PDF generation with tables disabled."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.plots.plot_containment", return_value=mock_fig):
                        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=mock_fig):
                            result = generate_report(
                                title="Test Report",
                                summary_file=str(sample_summary_data),
                                output_file=str(output_file),
                                format="pdf",
                                include_plots=True,
                                include_tables=False,
                            )

        assert result == output_file


class TestHelperFunctions:
    """Test individual helper functions.

    These tests verify each helper function works correctly in isolation.
    This approach makes it easier to identify which specific function is broken.
    """

    def test_create_title_page(self, sample_summary_data):
        """Test _create_title_page creates figure with expected elements."""
        from metaquest.visualization.reporting import _create_title_page

        # Create minimal metadata
        metadata = pd.DataFrame({"run_accession": ["SRR001"]})

        mock_fig = Mock()
        mock_ax = Mock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            result = _create_title_page(
                title="Test Report", summary_data=str(sample_summary_data), metadata_data=metadata
            )

        # Verify result
        assert result == mock_fig

        # Verify axis was turned off
        mock_ax.axis.assert_called_once_with("off")

        # Verify text was added (multiple calls)
        assert mock_ax.text.called
        assert mock_ax.text.call_count >= 4  # Title, subtitle, date, metadata

        # Verify specific text content
        calls = mock_ax.text.call_args_list
        text_values = [str(call[0][2]) for call in calls]  # 3rd arg is text

        # Check for expected content
        assert any("Test Report" in t for t in text_values), "Title not found"
        assert any("MetaQuest" in t for t in text_values), "Subtitle not found"

    def test_create_containment_summary_page(self, sample_summary_data):
        """Test _create_containment_summary_page with threshold filtering."""
        from metaquest.visualization.reporting import _create_containment_summary_page

        # Load the data
        data = pd.read_csv(sample_summary_data, sep="\t", index_col=0)

        mock_fig = Mock()
        mock_ax = Mock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            fig, filtered = _create_containment_summary_page(summary_data=data, threshold=0.5)

        # Verify result
        assert fig == mock_fig

        # Verify threshold filtering worked
        assert len(filtered) == 3  # 3 samples above 0.5
        assert all(filtered["max_containment"] > 0.5)

        # Verify axis was turned off
        mock_ax.axis.assert_called_once_with("off")

        # Verify text and table were added
        assert mock_ax.text.called
        assert mock_ax.table.called


class TestDataLoading:
    """Test data loading and validation."""

    def test_loads_summary_data(self, sample_summary_data, tmp_path):
        """Test that summary data is loaded correctly."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(Mock(), Mock())):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.plots.plot_containment", return_value=Mock()):
                        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=Mock()):
                            # This should succeed if data loading works
                            result = generate_report(
                                title="Test",
                                summary_file=str(sample_summary_data),
                                output_file=str(output_file),
                                format="pdf",
                            )

        assert result == output_file

    def test_handles_missing_metadata_file(self, sample_summary_data, tmp_path):
        """Test that missing optional metadata file is handled gracefully."""
        output_file = tmp_path / "report.pdf"
        nonexistent_metadata = tmp_path / "nonexistent_metadata.tsv"

        mock_pdf_pages = MagicMock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(Mock(), Mock())):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.plots.plot_containment", return_value=Mock()):
                        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=Mock()):
                            # Should not raise error for missing optional metadata
                            result = generate_report(
                                title="Test",
                                summary_file=str(sample_summary_data),
                                metadata_file=str(nonexistent_metadata),
                                output_file=str(output_file),
                                format="pdf",
                            )

        assert result == output_file


# ============================================================================
# NEXT STEPS:
#
# 1. Run these tests:
#    pytest tests/test_visualization_reporting_starter.py -v
#
# 2. Check coverage:
#    pytest --cov=metaquest.visualization.reporting \
#           --cov-report=term-missing \
#           tests/test_visualization_reporting_starter.py
#
# 3. Expected: 10 tests pass, coverage increases from 0% to ~20%
#
# 4. Add more tests for HTML generation (follow same pattern)
#
# 5. Gradually increase coverage by testing:
#    - HTML generation
#    - Template rendering
#    - Plot generation helpers
#    - Error handling for edge cases
#
# See TESTING_QUICK_START.md for detailed guidance.
# ============================================================================
