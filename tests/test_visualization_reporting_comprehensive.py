"""
Comprehensive tests for metaquest.visualization.reporting module.

Salvaged and fixed from .broken test files, covering functions not yet
tested by starter/extended test files: _create_default_template,
_generate_html_report, _get_genome_columns, error handling paths,
and deeper coverage of _prepare_template_data and _generate_plots_for_html.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from metaquest.visualization.reporting import (
    generate_report,
    _create_title_page,
    _create_containment_summary_page,
    _add_metadata_summary_page,
    _add_correlation_heatmap,
    _generate_pdf_report,
    _prepare_template_data,
    _generate_plots_for_html,
    _create_default_template,
    _generate_html_report,
    _get_genome_columns,
)
from metaquest.core.exceptions import ProcessingError, VisualizationError


@pytest.fixture
def summary_df():
    """Standard summary DataFrame for tests."""
    return pd.DataFrame(
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


@pytest.fixture
def metadata_df():
    """Standard metadata DataFrame for tests."""
    return pd.DataFrame(
        {
            "run_accession": ["SRR001", "SRR002", "SRR003", "SRR004"],
            "organism_name": ["E. coli", "Salmonella", "Klebsiella", "Pseudomonas"],
            "platform": ["ILLUMINA"] * 4,
            "library_source": ["GENOMIC"] * 4,
        }
    )


@pytest.fixture
def counts_df():
    """Standard counts DataFrame for tests."""
    return pd.DataFrame(
        {
            "category": ["Category1", "Category2", "Category3"],
            "count": [100, 50, 25],
        }
    )


@pytest.fixture
def summary_file(tmp_path, summary_df):
    """Write summary data to a TSV file."""
    path = tmp_path / "summary.tsv"
    summary_df.to_csv(path, sep="\t")
    return path


@pytest.fixture
def metadata_file(tmp_path, metadata_df):
    """Write metadata to a TSV file."""
    path = tmp_path / "metadata.tsv"
    metadata_df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def counts_file(tmp_path, counts_df):
    """Write counts data to a TSV file."""
    path = tmp_path / "counts.tsv"
    counts_df.to_csv(path, sep="\t", index=False)
    return path


class TestGetGenomeColumns:
    """Test _get_genome_columns helper function."""

    def test_filters_known_metadata_columns(self, summary_df):
        result = _get_genome_columns(summary_df)
        assert "max_containment" not in result
        assert "max_containment_annotation" not in result

    def test_includes_gcf_columns(self, summary_df):
        result = _get_genome_columns(summary_df)
        assert "GCF_000005825.2" in result
        assert "GCF_000006945.2" in result
        assert "GCF_000009605.1" in result

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        with pytest.raises(ProcessingError, match="No genome columns found"):
            _get_genome_columns(df)

    def test_only_metadata_columns(self):
        df = pd.DataFrame({"max_containment": [0.5], "max_containment_annotation": ["Genome1"]})
        with pytest.raises(ProcessingError, match="No genome columns found"):
            _get_genome_columns(df)


class TestCreateDefaultTemplate:
    """Test _create_default_template function."""

    def test_returns_path(self, tmp_path):
        with patch("metaquest.visualization.reporting.Path") as mock_path_class:
            # Make __file__ resolve to tmp_path so template is written there
            template_path = tmp_path / "templates" / "report_template.html"
            mock_path_class.return_value = tmp_path
            mock_path_class.__truediv__ = Path.__truediv__

            # Just call the real function, it creates a file on disk
            result = _create_default_template()

        assert isinstance(result, Path)

    def test_template_contains_html_structure(self):
        """Test that the generated template has required HTML elements."""
        result = _create_default_template()
        # Read the template file
        content = result.read_text()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "{{ title }}" in content
        assert "{{ timestamp }}" in content

    def test_template_contains_summary_section(self):
        result = _create_default_template()
        content = result.read_text()

        assert "Containment Summary" in content
        assert "{{ summary.total_samples }}" in content
        assert "{{ summary.samples_above_threshold }}" in content

    def test_template_contains_metadata_section(self):
        result = _create_default_template()
        content = result.read_text()

        assert "Metadata Summary" in content
        assert "{% if metadata is defined %}" in content

    def test_template_contains_plot_conditionals(self):
        result = _create_default_template()
        content = result.read_text()

        assert "{% if include_plots" in content
        assert "rank_plot" in content
        assert "hist_plot" in content

    def test_returns_existing_template(self):
        """Test that existing template is returned without rewriting."""
        # First call creates the template
        path1 = _create_default_template()
        # Second call should return the same path
        path2 = _create_default_template()
        assert path1 == path2


class TestCreateTitlePageEdgeCases:
    """Additional edge cases for _create_title_page."""

    def test_title_page_with_string_summary_data(self):
        """Title page accepts a string path for summary_data (for len())."""
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            result = _create_title_page("Report Title", "path/to/summary.tsv", None)

        assert result == mock_fig
        # Should have at least title, subtitle, date, and sample count text calls
        assert mock_ax.text.call_count >= 4

    def test_title_page_with_empty_metadata(self):
        """Title page with empty metadata DataFrame."""
        mock_fig = Mock()
        mock_ax = Mock()
        empty_metadata = pd.DataFrame()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            result = _create_title_page("Empty Metadata Report", "summary.tsv", empty_metadata)

        assert result == mock_fig
        # Should include metadata line since metadata_data is not None
        assert mock_ax.text.call_count >= 5


class TestContainmentSummaryPageEdgeCases:
    """Additional edge cases for _create_containment_summary_page."""

    def test_all_samples_above_threshold(self, summary_df):
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            fig, filtered = _create_containment_summary_page(summary_df, threshold=0.01)

        assert len(filtered) == 4

    def test_single_sample(self):
        df = pd.DataFrame(
            {"max_containment": [0.95], "max_containment_annotation": ["Genome1"], "GCF_001": [0.95]},
            index=["Sample1"],
        )
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            fig, filtered = _create_containment_summary_page(df, threshold=0.5)

        assert len(filtered) == 1
        assert mock_ax.table.called


class TestAddCorrelationHeatmapEdgeCases:
    """Additional edge cases for _add_correlation_heatmap."""

    def test_no_genomes_above_threshold(self):
        """When no genomes have samples above threshold, no plot is created."""
        mock_pdf = Mock()
        # All containment values below threshold
        df = pd.DataFrame(
            {
                "max_containment": [0.01, 0.02],
                "max_containment_annotation": ["G1", "G2"],
                "GCF_001": [0.01, 0.02],
                "GCF_002": [0.01, 0.01],
            }
        )

        with patch("metaquest.visualization.reporting.plot_correlation_matrix") as mock_plot:
            _add_correlation_heatmap(mock_pdf, df, threshold=0.5)

        mock_plot.assert_not_called()

    def test_exception_is_caught(self):
        """Exceptions in heatmap generation are caught and logged."""
        mock_pdf = Mock()
        df = pd.DataFrame(
            {
                "max_containment": [0.95, 0.87],
                "max_containment_annotation": ["G1", "G2"],
                "GCF_001": [0.95, 0.12],
                "GCF_002": [0.23, 0.87],
            }
        )

        with patch(
            "metaquest.visualization.reporting.plot_correlation_matrix",
            side_effect=Exception("Plot error"),
        ):
            with patch("metaquest.visualization.reporting.logger") as mock_logger:
                _add_correlation_heatmap(mock_pdf, df, threshold=0.1)

        mock_logger.warning.assert_called_once()


class TestPrepareTemplateDataComprehensive:
    """Comprehensive tests for _prepare_template_data."""

    def test_with_all_data(self, summary_df, metadata_df, counts_df):
        result = _prepare_template_data(
            title="Full Report",
            summary_data=summary_df,
            metadata_data=metadata_df,
            counts_data=counts_df,
            include_plots=True,
            include_tables=True,
            threshold=0.5,
            plot_files={"rank_plot": "images/rank.png"},
        )

        assert result["title"] == "Full Report"
        assert result["include_plots"] is True
        assert result["include_tables"] is True
        assert result["threshold"] == 0.5
        assert result["plots"] == {"rank_plot": "images/rank.png"}
        assert "timestamp" in result
        assert result["summary"]["total_samples"] == 4
        assert result["summary"]["samples_above_threshold"] == 3  # 0.95, 0.87, 0.65 > 0.5
        assert result["summary"]["genome_count"] == 3
        assert len(result["summary"]["top_samples"]) == 4
        assert "metadata" in result
        assert result["metadata"]["sample_count"] == 4
        assert result["metadata"]["column_count"] == 4

    def test_without_metadata(self, summary_df):
        result = _prepare_template_data(
            title="No Metadata",
            summary_data=summary_df,
            metadata_data=None,
            counts_data=None,
            include_plots=False,
            include_tables=False,
            threshold=0.1,
            plot_files={},
        )

        assert "metadata" not in result
        assert result["summary"]["total_samples"] == 4
        assert result["summary"]["samples_above_threshold"] == 4  # all > 0.1

    def test_high_threshold_no_samples_above(self, summary_df):
        result = _prepare_template_data(
            title="High Threshold",
            summary_data=summary_df,
            metadata_data=None,
            counts_data=None,
            include_plots=False,
            include_tables=False,
            threshold=0.99,
            plot_files={},
        )

        assert result["summary"]["samples_above_threshold"] == 0

    def test_top_samples_contains_expected_fields(self, summary_df):
        result = _prepare_template_data(
            title="Test",
            summary_data=summary_df,
            metadata_data=None,
            counts_data=None,
            include_plots=False,
            include_tables=False,
            threshold=0.1,
            plot_files={},
        )

        sample = result["summary"]["top_samples"][0]
        assert "sample" in sample
        assert "containment" in sample
        assert "genome" in sample
        assert sample["sample"] == "SRR001"
        assert sample["containment"] == "0.9500"
        assert sample["genome"] == "Escherichia coli"

    def test_metadata_top_fields(self, summary_df, metadata_df):
        result = _prepare_template_data(
            title="Test",
            summary_data=summary_df,
            metadata_data=metadata_df,
            counts_data=None,
            include_plots=False,
            include_tables=False,
            threshold=0.1,
            plot_files={},
        )

        fields = result["metadata"]["top_fields"]
        assert len(fields) > 0
        field = fields[0]
        assert "field" in field
        assert "count" in field
        assert "percentage" in field


class TestGeneratePlotsForHtmlComprehensive:
    """Comprehensive tests for _generate_plots_for_html."""

    def test_with_counts_data(self, tmp_path, summary_df, counts_df):
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mock_fig = Mock()
        mock_fig.savefig = Mock()

        with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
            with patch("metaquest.visualization.reporting.plot_metadata_counts", return_value=mock_fig):
                with patch("metaquest.visualization.reporting.plot_correlation_matrix", return_value=mock_fig):
                    with patch("matplotlib.pyplot.close"):
                        result = _generate_plots_for_html(summary_df, counts_df, threshold=0.1, images_dir=images_dir)

        assert "rank_plot" in result
        assert "hist_plot" in result
        assert "counts_plot" in result
        assert "pie_plot" in result
        assert "heatmap_plot" in result

    def test_plot_error_caught(self, tmp_path, summary_df):
        """Errors during plot generation are caught gracefully."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        with patch(
            "metaquest.visualization.reporting.plot_containment",
            side_effect=Exception("Rendering error"),
        ):
            with patch("metaquest.visualization.reporting.logger") as mock_logger:
                result = _generate_plots_for_html(summary_df, None, threshold=0.1, images_dir=images_dir)

        assert isinstance(result, dict)
        mock_logger.warning.assert_called()

    def test_metadata_counts_error_caught(self, tmp_path, summary_df, counts_df):
        """Errors in metadata count plotting are caught independently."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mock_fig = Mock()
        mock_fig.savefig = Mock()

        with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
            with patch(
                "metaquest.visualization.reporting.plot_metadata_counts",
                side_effect=Exception("Counts error"),
            ):
                with patch("metaquest.visualization.reporting.plot_correlation_matrix", return_value=mock_fig):
                    with patch("matplotlib.pyplot.close"):
                        result = _generate_plots_for_html(summary_df, counts_df, threshold=0.1, images_dir=images_dir)

        # Containment plots should still be generated
        assert "rank_plot" in result
        assert "hist_plot" in result
        # Counts plots should be missing due to error
        assert "counts_plot" not in result

    def test_heatmap_error_caught(self, tmp_path, summary_df):
        """Errors in heatmap generation are caught independently."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mock_fig = Mock()
        mock_fig.savefig = Mock()

        with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
            with patch(
                "metaquest.visualization.reporting.plot_correlation_matrix",
                side_effect=Exception("Heatmap error"),
            ):
                with patch("matplotlib.pyplot.close"):
                    result = _generate_plots_for_html(summary_df, None, threshold=0.1, images_dir=images_dir)

        assert "rank_plot" in result
        assert "heatmap_plot" not in result

    def test_single_genome_no_heatmap(self, tmp_path):
        """With only one genome column, no heatmap is generated."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        df = pd.DataFrame(
            {
                "max_containment": [0.95, 0.87],
                "max_containment_annotation": ["G1", "G2"],
                "GCF_001": [0.95, 0.12],
            }
        )

        mock_fig = Mock()
        mock_fig.savefig = Mock()

        with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
            with patch("metaquest.visualization.reporting.plot_correlation_matrix") as mock_corr:
                with patch("matplotlib.pyplot.close"):
                    result = _generate_plots_for_html(df, None, threshold=0.1, images_dir=images_dir)

        mock_corr.assert_not_called()
        assert "heatmap_plot" not in result


class TestGeneratePdfReportComprehensive:
    """Comprehensive tests for _generate_pdf_report."""

    def test_with_counts_data_and_plots(self, tmp_path, summary_df, metadata_df, counts_df):
        """Test PDF generation with counts data and all features enabled."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
                        with patch("metaquest.visualization.reporting.plot_metadata_counts", return_value=mock_fig):
                            with patch("metaquest.visualization.reporting.plot_correlation_matrix", return_value=mock_fig):
                                result = _generate_pdf_report(
                                    title="Full Report",
                                    summary_data=summary_df,
                                    metadata_data=metadata_df,
                                    counts_data=counts_df,
                                    output_file=str(output_file),
                                    threshold=0.1,
                                    include_plots=True,
                                    include_tables=True,
                                )

        assert result == output_file

    def test_counts_plot_error_caught(self, tmp_path, summary_df, counts_df):
        """Errors in metadata count plotting are caught in PDF generation."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
                        with patch(
                            "metaquest.visualization.reporting.plot_metadata_counts",
                            side_effect=Exception("Counts error"),
                        ):
                            with patch("metaquest.visualization.reporting.plot_correlation_matrix", return_value=mock_fig):
                                # Should not raise
                                result = _generate_pdf_report(
                                    title="Report",
                                    summary_data=summary_df,
                                    metadata_data=None,
                                    counts_data=counts_df,
                                    output_file=str(output_file),
                                    threshold=0.1,
                                    include_plots=True,
                                    include_tables=True,
                                )

        assert result == output_file

    def test_no_plots_no_tables(self, tmp_path, summary_df):
        """Test PDF with both plots and tables disabled."""
        output_file = tmp_path / "report.pdf"

        mock_pdf_pages = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.reporting.plot_containment") as mock_plot:
                        result = _generate_pdf_report(
                            title="Minimal",
                            summary_data=summary_df,
                            metadata_data=None,
                            counts_data=None,
                            output_file=str(output_file),
                            threshold=0.1,
                            include_plots=False,
                            include_tables=False,
                        )

        assert result == output_file
        mock_plot.assert_not_called()


class TestGenerateHtmlReportComprehensive:
    """Comprehensive tests for _generate_html_report."""

    def test_full_html_report(self, tmp_path, summary_df, metadata_df, counts_df):
        """Test HTML report with all features."""
        output_file = tmp_path / "report.html"

        mock_template = Mock()
        mock_template.render.return_value = "<html>Full Report</html>"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        mock_fig = Mock()
        mock_fig.savefig = Mock()

        with patch("metaquest.visualization.reporting.JINJA2_AVAILABLE", True):
            with patch("metaquest.visualization.reporting.jinja2") as mock_jinja2:
                mock_jinja2.FileSystemLoader = Mock()
                mock_jinja2.Environment.return_value = mock_env
                with patch("metaquest.visualization.reporting._create_default_template"):
                    with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
                        with patch("metaquest.visualization.reporting.plot_correlation_matrix", return_value=mock_fig):
                            with patch("metaquest.visualization.reporting.plot_metadata_counts", return_value=mock_fig):
                                with patch("matplotlib.pyplot.close"):
                                    result = _generate_html_report(
                                        title="Full HTML Report",
                                        summary_data=summary_df,
                                        metadata_data=metadata_df,
                                        counts_data=counts_df,
                                        output_file=str(output_file),
                                        threshold=0.1,
                                        include_plots=True,
                                        include_tables=True,
                                    )

        assert result == output_file
        assert output_file.exists()
        mock_template.render.assert_called_once()

    def test_html_without_plots(self, tmp_path, summary_df):
        """Test HTML report with plots disabled."""
        output_file = tmp_path / "report.html"

        mock_template = Mock()
        mock_template.render.return_value = "<html>No Plots</html>"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        with patch("metaquest.visualization.reporting.JINJA2_AVAILABLE", True):
            with patch("metaquest.visualization.reporting.jinja2") as mock_jinja2:
                mock_jinja2.FileSystemLoader = Mock()
                mock_jinja2.Environment.return_value = mock_env
                with patch("metaquest.visualization.reporting._create_default_template"):
                    with patch("metaquest.visualization.reporting._generate_plots_for_html") as mock_plots:
                        result = _generate_html_report(
                            title="No Plots",
                            summary_data=summary_df,
                            metadata_data=None,
                            counts_data=None,
                            output_file=str(output_file),
                            threshold=0.1,
                            include_plots=False,
                            include_tables=True,
                        )

        assert result == output_file
        mock_plots.assert_not_called()

    def test_html_without_jinja2_raises(self, tmp_path, summary_df):
        """Test that HTML generation without jinja2 raises error."""
        output_file = tmp_path / "report.html"

        with patch("metaquest.visualization.reporting.JINJA2_AVAILABLE", False):
            with pytest.raises(VisualizationError, match="requires jinja2"):
                _generate_html_report(
                    title="Test",
                    summary_data=summary_df,
                    metadata_data=None,
                    counts_data=None,
                    output_file=str(output_file),
                    threshold=0.1,
                    include_plots=False,
                    include_tables=False,
                )

    def test_html_report_writes_rendered_content(self, tmp_path, summary_df):
        """Test that HTML report writes rendered template to file."""
        output_file = tmp_path / "report.html"
        expected_content = "<html><body>Rendered Content</body></html>"

        mock_template = Mock()
        mock_template.render.return_value = expected_content

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        with patch("metaquest.visualization.reporting.JINJA2_AVAILABLE", True):
            with patch("metaquest.visualization.reporting.jinja2") as mock_jinja2:
                mock_jinja2.FileSystemLoader = Mock()
                mock_jinja2.Environment.return_value = mock_env
                with patch("metaquest.visualization.reporting._create_default_template"):
                    _generate_html_report(
                        title="Content Test",
                        summary_data=summary_df,
                        metadata_data=None,
                        counts_data=None,
                        output_file=str(output_file),
                        threshold=0.1,
                        include_plots=False,
                        include_tables=True,
                    )

        assert output_file.read_text() == expected_content


class TestGenerateReportIntegration:
    """Integration tests for generate_report end-to-end paths."""

    def test_pdf_with_all_optional_files(self, summary_file, metadata_file, counts_file, tmp_path):
        """Test PDF generation with summary, metadata, and counts files."""
        output_file = tmp_path / "full_report.pdf"

        mock_pdf_pages = MagicMock()
        mock_fig = Mock()
        mock_ax = Mock()

        with patch("metaquest.visualization.reporting.PdfPages", return_value=mock_pdf_pages):
            with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
                with patch("matplotlib.pyplot.close"):
                    with patch("metaquest.visualization.reporting.plot_containment", return_value=mock_fig):
                        with patch("metaquest.visualization.reporting.plot_metadata_counts", return_value=mock_fig):
                            with patch("metaquest.visualization.reporting.plot_correlation_matrix", return_value=mock_fig):
                                result = generate_report(
                                    title="Full Integration Report",
                                    summary_file=str(summary_file),
                                    metadata_file=str(metadata_file),
                                    metadata_counts_file=str(counts_file),
                                    output_file=str(output_file),
                                    format="pdf",
                                    threshold=0.1,
                                    include_plots=True,
                                    include_tables=True,
                                )

        assert result == output_file

    def test_malformed_csv_raises_visualization_error(self, tmp_path):
        """Test that malformed CSV raises VisualizationError."""
        bad_file = tmp_path / "bad.tsv"
        bad_file.write_text("not\ta\tvalid\ncsv\tfile\twith\tbad\tdata")

        with pytest.raises(VisualizationError, match="Error generating report"):
            generate_report(
                title="Bad Data",
                summary_file=str(bad_file),
                output_file=str(tmp_path / "report.pdf"),
                format="pdf",
            )

    def test_html_report_end_to_end(self, summary_file, tmp_path):
        """Test HTML report generation end-to-end."""
        output_file = tmp_path / "report.html"

        mock_template = Mock()
        mock_template.render.return_value = "<html>Report</html>"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        with patch("metaquest.visualization.reporting.JINJA2_AVAILABLE", True):
            with patch("metaquest.visualization.reporting.jinja2") as mock_jinja2:
                mock_jinja2.FileSystemLoader = Mock()
                mock_jinja2.Environment.return_value = mock_env
                with patch("metaquest.visualization.reporting._create_default_template"):
                    result = generate_report(
                        title="HTML E2E",
                        summary_file=str(summary_file),
                        output_file=str(output_file),
                        format="html",
                        include_plots=False,
                    )

        assert result == output_file
        assert output_file.exists()


class TestAddMetadataSummaryPageEdgeCases:
    """Additional edge cases for _add_metadata_summary_page."""

    def test_metadata_with_nulls(self):
        """Test metadata summary with null values."""
        mock_pdf = Mock()
        mock_fig = Mock()
        mock_ax = Mock()

        metadata = pd.DataFrame(
            {
                "organism": ["E. coli", None, "K. pneumoniae"],
                "country": ["USA", "Denmark", None],
                "assay_type": [None, None, "WGS"],
            }
        )

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.close"):
                _add_metadata_summary_page(mock_pdf, metadata)

        mock_pdf.savefig.assert_called_once_with(mock_fig)
        mock_ax.table.assert_called_once()

    def test_metadata_many_columns(self):
        """Test metadata summary with more than 15 columns (tests nlargest(15))."""
        mock_pdf = Mock()
        mock_fig = Mock()
        mock_ax = Mock()

        # Create DataFrame with 20 columns
        data = {f"col_{i}": [f"val_{i}"] * 3 for i in range(20)}
        metadata = pd.DataFrame(data)

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.close"):
                _add_metadata_summary_page(mock_pdf, metadata)

        # Table should show top 15 columns
        call_args = mock_ax.table.call_args
        table_data = call_args[1]["cellText"] if "cellText" in call_args[1] else call_args[0][0]
        assert len(table_data) == 15
