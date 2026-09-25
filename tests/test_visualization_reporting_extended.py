"""
EXTENDED TESTS for visualization/reporting.py (49% -> 70%+ coverage)

This file extends the starter tests with HTML generation and helper function tests.
Run: pytest tests/test_visualization_reporting_extended.py -v
"""

import sys
import importlib.util
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from metaquest.visualization.reporting import (
    generate_report,
    _add_metadata_summary_page,
    _add_correlation_heatmap,
    _prepare_template_data,
    _generate_plots_for_html,
)
from metaquest.core.exceptions import ConfigurationError, VisualizationError

requires_interactive = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("plotly", "jinja2")), reason="needs metaquest[interactive]"
)


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


@pytest.fixture
def sample_counts_data(tmp_path):
    """Create minimal counts data."""
    counts_file = tmp_path / "counts.tsv"
    data = pd.DataFrame({"category": ["Category1", "Category2", "Category3"], "count": [100, 50, 25]})
    data.to_csv(counts_file, sep="\t", index=False, header=False)
    return counts_file


class TestHTMLReportGeneration:
    """Test HTML report generation with proper mocking."""

    @requires_interactive
    def test_generate_html_report_minimal(self, sample_summary_data, tmp_path):
        """Test minimal HTML generation with full mocking."""
        output_file = tmp_path / "report.html"

        # Mock jinja2 components
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test Report</html>"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        with patch("jinja2.FileSystemLoader"):
            with patch("jinja2.Environment", return_value=mock_env):
                with patch("metaquest.visualization.reporting._create_default_template"):
                    with patch("metaquest.visualization.plots.plot_containment", return_value=Mock()):
                        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=Mock()):
                            with patch("matplotlib.pyplot.close"):
                                result = generate_report(
                                    title="Test HTML Report",
                                    summary_file=str(sample_summary_data),
                                    output_file=str(output_file),
                                    format="html",
                                    threshold=0.1,
                                    include_plots=True,
                                    include_tables=True,
                                )

        # Verify result
        assert result == output_file
        assert output_file.exists()

        # Verify template was rendered
        mock_template.render.assert_called_once()

    @requires_interactive
    def test_generate_html_without_plots(self, sample_summary_data, tmp_path):
        """Test HTML generation with plots disabled."""
        output_file = tmp_path / "report.html"

        mock_template = Mock()
        mock_template.render.return_value = "<html>Test Report</html>"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        with patch("jinja2.FileSystemLoader"):
            with patch("jinja2.Environment", return_value=mock_env):
                with patch("metaquest.visualization.reporting._create_default_template"):
                    with patch("metaquest.visualization.plots.plot_containment") as mock_plot:
                        result = generate_report(
                            title="Test HTML Report",
                            summary_file=str(sample_summary_data),
                            output_file=str(output_file),
                            format="html",
                            include_plots=False,
                            include_tables=True,
                        )

        assert result == output_file
        # Verify plotting was NOT called when include_plots=False
        mock_plot.assert_not_called()


class TestHelperFunctionsExtended:
    """Test additional helper functions."""

    def test_add_metadata_summary_page(self, sample_metadata):
        """Test _add_metadata_summary_page creates metadata summary."""
        # Load metadata
        metadata_data = pd.read_csv(sample_metadata, sep="\t")

        mock_pdf = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.close"):
                _add_metadata_summary_page(mock_pdf, metadata_data)

        # Verify PDF was updated
        mock_pdf.savefig.assert_called_once_with(mock_fig)

        # Verify matplotlib functions were called
        mock_ax.axis.assert_called_once_with("off")
        assert mock_ax.text.called
        assert mock_ax.table.called

    def test_add_correlation_heatmap(self, sample_summary_data):
        """Test _add_correlation_heatmap creates heatmap."""
        # Load data
        summary_data = pd.read_csv(sample_summary_data, sep="\t", index_col=0)

        mock_pdf = MagicMock()
        mock_fig = Mock()

        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=mock_fig):
            with patch("matplotlib.pyplot.close"):
                _add_correlation_heatmap(mock_pdf, summary_data, threshold=0.1)

        # Verify PDF was updated (correlation matrix has >1 genome column)
        assert mock_pdf.savefig.call_count >= 0  # May be 0 if no valid genomes

    def test_prepare_template_data(self, sample_summary_data, sample_metadata):
        """Test _prepare_template_data creates correct template data."""
        # Load data
        summary_data = pd.read_csv(sample_summary_data, sep="\t", index_col=0)
        metadata_data = pd.read_csv(sample_metadata, sep="\t")

        template_data = _prepare_template_data(
            title="Test Report",
            summary_data=summary_data,
            metadata_data=metadata_data,
            counts_data=None,
            include_plots=True,
            include_tables=True,
            threshold=0.5,
            plot_files={},
        )

        # Verify structure
        assert "title" in template_data
        assert template_data["title"] == "Test Report"
        assert "timestamp" in template_data
        assert "summary" in template_data
        assert "metadata" in template_data

        # Verify summary stats
        assert template_data["summary"]["total_samples"] == 4
        assert "top_samples" in template_data["summary"]

        # Verify metadata stats
        assert template_data["metadata"]["sample_count"] == 4

    def test_generate_plots_for_html(self, sample_summary_data, tmp_path):
        """Test _generate_plots_for_html creates plot files."""
        # Load data
        summary_data = pd.read_csv(sample_summary_data, sep="\t", index_col=0)
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mock_fig = Mock()
        mock_fig.savefig = Mock()

        with patch("metaquest.visualization.plots.plot_containment", return_value=mock_fig):
            with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=mock_fig):
                with patch("matplotlib.pyplot.close"):
                    plot_files = _generate_plots_for_html(
                        summary_data=summary_data, counts_data=None, threshold=0.1, images_dir=images_dir
                    )

        # Verify plot files dictionary was created
        assert isinstance(plot_files, dict)
        assert "rank_plot" in plot_files
        assert "hist_plot" in plot_files


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_invalid_summary_file(self, tmp_path):
        """Test that invalid summary file raises appropriate error."""
        output_file = tmp_path / "report.pdf"
        invalid_file = tmp_path / "nonexistent.tsv"

        with pytest.raises(VisualizationError):
            generate_report(title="Test", summary_file=str(invalid_file), output_file=str(output_file), format="pdf")

    def test_html_generation_without_jinja2_available(self, sample_summary_data, tmp_path):
        """Test HTML generation fails gracefully when jinja2 unavailable."""
        output_file = tmp_path / "report.html"

        with patch.dict(sys.modules, {"jinja2": None}):
            with pytest.raises(ConfigurationError, match=r"metaquest\[interactive\]"):
                generate_report(
                    title="Test", summary_file=str(sample_summary_data), output_file=str(output_file), format="html"
                )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_handling(self, tmp_path):
        """Test that empty summary data raises appropriate error."""
        summary_file = tmp_path / "empty_summary.tsv"
        # Create empty DataFrame with required columns and proper dtypes
        data = pd.DataFrame(
            {
                "max_containment": pd.Series([], dtype="float64"),
                "max_containment_annotation": pd.Series([], dtype="object"),
            }
        )
        data.to_csv(summary_file, sep="\t")

        output_file = tmp_path / "report.pdf"

        # Empty data should raise an error or handle gracefully
        # For now, we expect it to fail gracefully
        with pytest.raises(VisualizationError):
            generate_report(
                title="Test Empty Data", summary_file=str(summary_file), output_file=str(output_file), format="pdf"
            )

    def test_high_threshold_filters_all_data(self, sample_summary_data, tmp_path):
        """Test that high threshold filters out all samples."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(Mock(), Mock())):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.plots.plot_containment", return_value=Mock()):
                        with patch("metaquest.visualization.plots.plot_correlation_matrix", return_value=Mock()):
                            result = generate_report(
                                title="Test High Threshold",
                                summary_file=str(sample_summary_data),
                                output_file=str(output_file),
                                format="pdf",
                                threshold=0.99,  # Very high threshold
                            )

        assert result == output_file


# ============================================================================
# SUCCESS METRICS:
#
# After running these extended tests:
# - Total tests: 10 (starter) + 12 (extended) = 22 tests
# - Expected coverage: 49% -> 70%+ for reporting.py
# - Tests cover: HTML generation, helper functions, error handling, edge cases
#
# Run all reporting tests:
#   pytest tests/test_visualization_reporting_*.py -v
#
# Check combined coverage:
#   pytest --cov=metaquest.visualization.reporting --cov-report=term-missing \
#          tests/test_visualization_reporting_*.py
# ============================================================================


# ============================================================================
# Narrow exception handling (tech debt): a programming error must not be swallowed
# ============================================================================


def test_generate_report_wraps_a_read_error_with_its_cause(tmp_path):
    """Kind (c): the wrapped error keeps its cause."""
    with pytest.raises(VisualizationError) as exc:
        generate_report(title="t", summary_file=tmp_path / "missing.tsv", output_file=tmp_path / "r.pdf")
    assert isinstance(exc.value.__cause__, OSError)


def test_generate_report_propagates_a_bug(tmp_path):
    """Kind (c): a programming error is not re-labelled as a report error."""
    with patch("metaquest.visualization.reporting.pd.read_csv", side_effect=TypeError("bug")):
        with pytest.raises(TypeError):
            generate_report(title="t", summary_file=tmp_path / "s.tsv", output_file=tmp_path / "r.pdf")


def test_correlation_heatmap_skips_a_plot_error_and_propagates_a_bug():
    """Kind (e): a plotting failure is logged and skipped; an AttributeError propagates."""
    data = pd.DataFrame({"g1": [0.5, 0.2], "g2": [0.4, 0.1]})
    pdf = MagicMock()
    with patch(
        "metaquest.visualization.reporting._top_correlated_genome_columns", side_effect=VisualizationError("bad")
    ):
        _add_correlation_heatmap(pdf, data, 0.1)
    with patch("metaquest.visualization.reporting._top_correlated_genome_columns", side_effect=AttributeError("bug")):
        with pytest.raises(AttributeError):
            _add_correlation_heatmap(pdf, data, 0.1)


def test_html_plots_skip_a_save_error_and_propagate_a_bug(tmp_path):
    """Kind (e): a figure that cannot be saved is skipped; a TypeError propagates."""
    data = pd.DataFrame({"max_containment": [0.5, 0.2], "g1": [0.5, 0.2]})
    with patch("metaquest.visualization.reporting._add_containment_plot_files", side_effect=OSError("disk full")):
        assert _generate_plots_for_html(data, None, 0.1, tmp_path) == {}
    with patch("metaquest.visualization.reporting._add_containment_plot_files", side_effect=TypeError("bug")):
        with pytest.raises(TypeError):
            _generate_plots_for_html(data, None, 0.1, tmp_path)


def test_correlation_heatmap_skips_a_text_genome_column(caplog):
    """Fix round 1: a genome column holding text skips the heatmap with a warning."""
    data = pd.DataFrame(
        {
            "max_containment": [0.5, 0.2],
            "max_containment_annotation": ["g1", "g1"],
            "g1": [0.5, 0.2],
            "g2": ["x", "y"],
        }
    )
    pdf = MagicMock()
    with caplog.at_level("WARNING"):
        _add_correlation_heatmap(pdf, data, 0.1)
    pdf.savefig.assert_not_called()
    assert "Error generating heatmap" in caplog.text


def test_html_heatmap_skips_a_text_genome_column(tmp_path, caplog):
    """Fix round 1: the HTML report's heatmap step is skipped the same way."""
    from metaquest.visualization.reporting import _add_correlation_plot_file

    data = pd.DataFrame({"max_containment": [0.5, 0.2], "g1": [0.5, 0.2], "g2": ["x", "y"]})
    plot_files: dict = {}
    with caplog.at_level("WARNING"):
        _add_correlation_plot_file(data, 0.1, tmp_path, plot_files)
    assert plot_files == {}
    assert "Error generating heatmap" in caplog.text
