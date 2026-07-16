"""
Advanced SRA Reporting and Dashboard Generation.

This module provides comprehensive reporting capabilities including:
- Interactive HTML dashboards with Plotly visualizations
- Download session summaries and performance analytics
- Quality control reports with recommendations
- Comparative analysis reports across datasets
- Export capabilities for metadata and statistics
"""

import logging
from datetime import datetime, timedelta
from html import escape as html_escape
from pathlib import Path
from typing import Dict, List, Union, Any

import pandas as pd
import numpy as np

# Conditional imports for enhanced features
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from jinja2 import Environment, BaseLoader

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from metaquest.sra.download_manager import DownloadSession
from metaquest.sra.analytics import (
    QualityProfile,
    ComparativeAnalysis,
    SRADatasetAnalyzer,
)
from metaquest.utils.html import plotly_js_script

logger = logging.getLogger(__name__)


class SRAReportGenerator:
    """Generate comprehensive reports for SRA download and analysis sessions."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = SRADatasetAnalyzer()

    def create_download_summary(self, session: DownloadSession, include_plots: bool = True) -> Path:
        """
        Create comprehensive download session summary report.

        Args:
            session: DownloadSession with results
            include_plots: Include interactive plots

        Returns:
            Path to generated HTML report
        """
        logger.info(f"Creating download summary for session {session.session_id}")

        report_data = {
            "session": session,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "include_plots": include_plots and PLOTLY_AVAILABLE,
        }

        # Calculate additional statistics
        session_duration = session.end_time - session.start_time if session.end_time else timedelta(0)
        successful_downloads = [r for r in session.download_results.values() if r.status == "completed"]
        failed_downloads = [r for r in session.download_results.values() if r.status == "failed"]

        # Performance metrics
        total_mb = sum(r.downloaded_mb for r in session.download_results.values())
        avg_speed = total_mb / (session_duration.total_seconds() / 60) if session_duration.total_seconds() > 0 else 0

        report_data.update(
            {
                "summary_stats": {
                    "total_accessions": len(session.accessions),
                    "successful_downloads": len(successful_downloads),
                    "failed_downloads": len(failed_downloads),
                    "success_rate": len(successful_downloads) / len(session.accessions) if session.accessions else 0,
                    "total_size_mb": total_mb,
                    "session_duration": str(session_duration),
                    "average_speed_mbps": avg_speed,
                    "network_bandwidth": session.network_conditions.bandwidth_mbps,
                }
            }
        )

        # Create visualizations if enabled
        if include_plots and PLOTLY_AVAILABLE:
            plots = self._create_download_plots(session)
            report_data["plots"] = plots

        # Generate HTML report
        html_content = self._generate_download_html(report_data)

        report_path = self.output_dir / f"download_summary_{session.session_id}.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Download summary saved to {report_path}")
        return report_path

    def generate_quality_dashboard(self, accessions: List[str], title: str = "SRA Quality Dashboard") -> Path:
        """
        Generate interactive quality control dashboard.

        Args:
            accessions: List of SRA accessions to analyze
            title: Dashboard title

        Returns:
            Path to generated HTML dashboard
        """
        logger.info(f"Generating quality dashboard for {len(accessions)} datasets")

        # Profile all datasets
        profiles = {}
        for accession in accessions:
            try:
                profile = self.analyzer.profile_dataset_quality(accession)
                profiles[accession] = profile
            except Exception as e:
                logger.warning(f"Failed to profile {accession}: {e}")

        if not profiles:
            raise ValueError("No datasets could be profiled")

        # Create dashboard data
        dashboard_data = {
            "title": title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_datasets": len(profiles),
            "profiles": profiles,
        }

        # Calculate summary statistics
        summary_stats = self._calculate_quality_summary(profiles)
        dashboard_data["summary_stats"] = summary_stats

        # Detect anomalies
        anomaly_report = self.analyzer.detect_dataset_anomalies(list(profiles.keys()))
        dashboard_data["anomaly_report"] = anomaly_report

        # Create visualizations
        if PLOTLY_AVAILABLE:
            plots = self._create_quality_plots(profiles)
            dashboard_data["plots"] = plots

        # Generate HTML dashboard
        html_content = self._generate_quality_html(dashboard_data)

        dashboard_path = self.output_dir / f"quality_dashboard_{int(datetime.now().timestamp())}.html"
        with open(dashboard_path, "w") as f:
            f.write(html_content)

        logger.info(f"Quality dashboard saved to {dashboard_path}")
        return dashboard_path

    def create_comparative_analysis(
        self, groups: Dict[str, List[str]], title: str = "Comparative Analysis Report"
    ) -> Path:
        """
        Create comparative analysis report between dataset groups.

        Args:
            groups: Dictionary mapping group names to accession lists
            title: Report title

        Returns:
            Path to generated HTML report
        """
        logger.info(f"Creating comparative analysis for {len(groups)} groups")

        # Perform comparative analysis
        comparison = self.analyzer.compare_datasets(groups)

        report_data = {
            "title": title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comparison": comparison,
            "group_counts": {name: len(accessions) for name, accessions in groups.items()},
        }

        # Create comparative visualizations
        if PLOTLY_AVAILABLE:
            plots = self._create_comparative_plots(comparison)
            report_data["plots"] = plots

        # Generate HTML report
        html_content = self._generate_comparative_html(report_data)

        report_path = self.output_dir / f"comparative_analysis_{int(datetime.now().timestamp())}.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Comparative analysis saved to {report_path}")
        return report_path

    def export_metadata_enriched(self, accessions: List[str], output_format: str = "csv") -> Path:
        """
        Export enriched metadata with quality metrics.

        Args:
            accessions: List of SRA accessions
            output_format: 'csv', 'json', or 'excel'

        Returns:
            Path to exported file
        """
        logger.info(f"Exporting enriched metadata for {len(accessions)} accessions")

        # Collect all data
        export_data = []
        for accession in accessions:
            try:
                profile = self.analyzer.profile_dataset_quality(accession)
                recommendations = self.analyzer.recommend_processing_params(accession, profile)

                data_row = {
                    "accession": accession,
                    "total_reads": profile.total_reads,
                    "total_bases": profile.total_bases,
                    "avg_read_length": profile.avg_read_length,
                    "gc_content": profile.gc_content,
                    "quality_grade": profile.quality_grade,
                    "complexity_score": profile.complexity_score,
                    "n_content": profile.n_content,
                    "adapter_contamination": profile.contamination_indicators.get("adapter_contamination", 0),
                    "recommended_pipeline": recommendations.recommended_pipeline,
                    "quality_trimming_needed": recommendations.quality_trimming["enabled"],
                    "adapter_removal_needed": recommendations.adapter_removal["enabled"],
                    "estimated_memory_gb": recommendations.computational_requirements["memory_gb"],
                    "estimated_processing_time": recommendations.estimated_processing_time,
                }

                export_data.append(data_row)

            except Exception as e:
                logger.error(f"Failed to process {accession}: {e}")

        # Create DataFrame and export
        df = pd.DataFrame(export_data)

        if output_format.lower() == "csv":
            output_path = self.output_dir / f"enriched_metadata_{int(datetime.now().timestamp())}.csv"
            df.to_csv(output_path, index=False)
        elif output_format.lower() == "json":
            output_path = self.output_dir / f"enriched_metadata_{int(datetime.now().timestamp())}.json"
            df.to_json(output_path, orient="records", indent=2)
        elif output_format.lower() == "excel":
            output_path = self.output_dir / f"enriched_metadata_{int(datetime.now().timestamp())}.xlsx"
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        logger.info(f"Enriched metadata exported to {output_path}")
        return output_path

    def _create_download_plots(self, session: DownloadSession) -> Dict[str, str]:
        """Create interactive plots for download session."""
        plots: dict = {}

        if not PLOTLY_AVAILABLE:
            return plots

        # Success rate pie chart
        success_count = sum(1 for r in session.download_results.values() if r.status == "completed")
        fail_count = len(session.download_results) - success_count

        fig_success = go.Figure(
            data=[
                go.Pie(
                    labels=["Successful", "Failed"],
                    values=[success_count, fail_count],
                    marker_colors=["#2ecc71", "#e74c3c"],
                )
            ]
        )
        fig_success.update_layout(title="Download Success Rate")
        plots["success_rate"] = pyo.plot(fig_success, output_type="div", include_plotlyjs=False)

        # Download speeds histogram
        speeds = [r.speed_mbps for r in session.download_results.values() if r.speed_mbps > 0]
        if speeds:
            fig_speeds = go.Figure(data=[go.Histogram(x=speeds, nbinsx=20)])
            fig_speeds.update_layout(
                title="Download Speed Distribution", xaxis_title="Speed (MB/s)", yaxis_title="Count"
            )
            plots["speed_distribution"] = pyo.plot(fig_speeds, output_type="div", include_plotlyjs=False)

        # File size vs download time scatter
        sizes = []
        times = []
        accessions = []

        for acc, result in session.download_results.items():
            if result.downloaded_mb > 0 and result.speed_mbps > 0:
                sizes.append(result.downloaded_mb)
                times.append(result.downloaded_mb / result.speed_mbps * 60)  # Convert to seconds
                accessions.append(acc)

        if sizes:
            fig_scatter = go.Figure(
                data=[
                    go.Scatter(
                        x=sizes,
                        y=times,
                        mode="markers",
                        text=accessions,
                        hovertemplate="<b>%{text}</b><br>Size: %{x:.1f} MB<br>Time: %{y:.1f} seconds",
                    )
                ]
            )
            fig_scatter.update_layout(
                title="File Size vs Download Time", xaxis_title="File Size (MB)", yaxis_title="Download Time (seconds)"
            )
            plots["size_vs_time"] = pyo.plot(fig_scatter, output_type="div", include_plotlyjs=False)

        return plots

    def _create_quality_plots(self, profiles: Dict[str, QualityProfile]) -> Dict[str, str]:
        """Create interactive plots for quality dashboard."""
        plots: dict = {}

        if not PLOTLY_AVAILABLE:
            return plots

        # Quality grade distribution
        grades = [p.quality_grade for p in profiles.values()]
        grade_counts = pd.Series(grades).value_counts()

        fig_grades = go.Figure(
            data=[
                go.Bar(
                    x=grade_counts.index,
                    y=grade_counts.values,
                    marker_color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
                )
            ]
        )
        fig_grades.update_layout(title="Quality Grade Distribution")
        plots["quality_grades"] = pyo.plot(fig_grades, output_type="div", include_plotlyjs=False)

        # GC content distribution
        gc_contents = [p.gc_content for p in profiles.values()]
        fig_gc = go.Figure(data=[go.Histogram(x=gc_contents, nbinsx=25)])
        fig_gc.update_layout(title="GC Content Distribution", xaxis_title="GC Content", yaxis_title="Count")
        plots["gc_distribution"] = pyo.plot(fig_gc, output_type="div", include_plotlyjs=False)

        # Read length vs complexity scatter
        read_lengths = [p.avg_read_length for p in profiles.values()]
        complexities = [p.complexity_score for p in profiles.values()]
        accessions = list(profiles.keys())

        fig_complexity = go.Figure(
            data=[
                go.Scatter(
                    x=read_lengths,
                    y=complexities,
                    mode="markers",
                    text=accessions,
                    hovertemplate="<b>%{text}</b><br>Length: %{x:.0f} bp<br>Complexity: %{y:.2f}",
                )
            ]
        )
        fig_complexity.update_layout(
            title="Read Length vs Sequence Complexity",
            xaxis_title="Average Read Length (bp)",
            yaxis_title="Complexity Score",
        )
        plots["complexity_scatter"] = pyo.plot(fig_complexity, output_type="div", include_plotlyjs=False)

        return plots

    def _create_comparative_plots(self, comparison: ComparativeAnalysis) -> Dict[str, str]:
        """Create interactive plots for comparative analysis."""
        plots: dict = {}

        if not PLOTLY_AVAILABLE or not comparison.visualization_data:
            return plots

        # Box plots for numeric variables
        boxplot_data = comparison.visualization_data.get("boxplot_data", [])
        if boxplot_data:
            df = pd.DataFrame(boxplot_data)

            numeric_cols = ["avg_read_length", "gc_content", "total_reads", "complexity_score"]
            for col in numeric_cols:
                if col in df.columns:
                    fig_box = go.Figure()

                    for group in df["group"].unique():
                        group_data = df[df["group"] == group][col]
                        fig_box.add_trace(go.Box(y=group_data, name=group, boxpoints="outliers"))

                    fig_box.update_layout(
                        title=f"{col.replace('_', ' ').title()} by Group", yaxis_title=col.replace("_", " ").title()
                    )
                    plots[f"{col}_boxplot"] = pyo.plot(fig_box, output_type="div", include_plotlyjs=False)

        return plots

    def _calculate_quality_summary(self, profiles: Dict[str, QualityProfile]) -> Dict[str, Any]:
        """Calculate summary statistics for quality profiles."""
        if not profiles:
            return {}

        # Aggregate statistics
        total_reads = sum(p.total_reads for p in profiles.values())
        total_bases = sum(p.total_bases for p in profiles.values())
        avg_gc = np.mean([p.gc_content for p in profiles.values()])
        avg_complexity = np.mean([p.complexity_score for p in profiles.values()])

        # Quality grade distribution
        grades = [p.quality_grade for p in profiles.values()]
        grade_dist = pd.Series(grades).value_counts().to_dict()

        # Contamination statistics
        adapter_contamination = [p.contamination_indicators.get("adapter_contamination", 0) for p in profiles.values()]
        avg_contamination = np.mean(adapter_contamination)
        high_contamination = sum(1 for c in adapter_contamination if c > 0.05)

        return {
            "total_datasets": len(profiles),
            "total_reads": total_reads,
            "total_bases": total_bases,
            "average_gc_content": avg_gc,
            "average_complexity": avg_complexity,
            "quality_grade_distribution": grade_dist,
            "average_contamination": avg_contamination,
            "high_contamination_count": high_contamination,
        }

    def _generate_download_html(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML content for download summary report."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_download_html(report_data)

        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>SRA Download Summary - {{ session.session_id }}</title>
    {{ plotly_js|safe }}
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #3498db; color: white;
            padding: 20px; border-radius: 5px; }
        .summary-grid { display: grid; gap: 20px; margin: 20px 0;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
        .summary-card { background-color: #f8f9fa; padding: 15px;
            border-radius: 5px; border-left: 4px solid #3498db; }
        .plot-container { margin: 20px 0; border: 1px solid #ddd;
            border-radius: 5px; }
        .results-table { width: 100%; border-collapse: collapse;
            margin: 20px 0; }
        .results-table th, .results-table td {
            border: 1px solid #ddd; padding: 8px; text-align: left; }
        .results-table th { background-color: #f2f2f2; }
        .status-completed { color: #27ae60; font-weight: bold; }
        .status-failed { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SRA Download Summary</h1>
        <p>Session: {{ session.session_id }} | Generated: {{ timestamp }}</p>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Downloads</h3>
            <p style="font-size: 2em; margin: 0;">{{ summary_stats.total_accessions }}</p>
        </div>
        <div class="summary-card">
            <h3>Success Rate</h3>
            <p style="font-size: 2em; margin: 0;">{{ "%.1f"|format(summary_stats.success_rate * 100) }}%</p>
        </div>
        <div class="summary-card">
            <h3>Total Size</h3>
            <p style="font-size: 2em; margin: 0;">{{ "%.1f"|format(summary_stats.total_size_mb / 1024) }} GB</p>
        </div>
        <div class="summary-card">
            <h3>Average Speed</h3>
            <p style="font-size: 2em; margin: 0;">{{ "%.1f"|format(summary_stats.average_speed_mbps) }} MB/min</p>
        </div>
    </div>

    {% if plots %}
    <h2>Download Analytics</h2>
    {% for plot_name, plot_html in plots.items() %}
    <div class="plot-container">
        {{ plot_html|safe }}
    </div>
    {% endfor %}
    {% endif %}

    <h2>Download Results</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>Accession</th>
                <th>Status</th>
                <th>Size (MB)</th>
                <th>Progress</th>
                <th>Speed (MB/s)</th>
                <th>Retries</th>
            </tr>
        </thead>
        <tbody>
            {% for accession, result in session.download_results.items() %}
            <tr>
                <td>{{ accession }}</td>
                <td class="status-{{ result.status }}">{{ result.status.title() }}</td>
                <td>{{ "%.1f"|format(result.downloaded_mb) }}</td>
                <td>{{ "%.1f"|format(result.progress_pct) }}%</td>
                <td>{{ "%.2f"|format(result.speed_mbps) }}</td>
                <td>{{ result.retry_count }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
        """

        template = Environment(loader=BaseLoader(), autoescape=True).from_string(template_str)
        return template.render(plotly_js=plotly_js_script(), **report_data)

    def _generate_quality_html(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML content for quality dashboard."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_quality_html(dashboard_data)

        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    {{ plotly_js|safe }}
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #2ecc71; color: white;
            padding: 20px; border-radius: 5px; }
        .summary-grid { display: grid; gap: 20px; margin: 20px 0;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
        .summary-card { background-color: #f8f9fa; padding: 15px;
            border-radius: 5px; border-left: 4px solid #2ecc71; }
        .plot-container { margin: 20px 0; border: 1px solid #ddd;
            border-radius: 5px; }
        .anomaly-section { background-color: #fff3cd; padding: 15px;
            border: 1px solid #ffeaa7; border-radius: 5px;
            margin: 20px 0; }
        .recommendations { background-color: #d4edda; padding: 15px;
            border: 1px solid #c3e6cb; border-radius: 5px;
            margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ timestamp }} | Total Datasets: {{ total_datasets }}</p>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Reads</h3>
            <p style="font-size: 1.5em; margin: 0;">
                {{ "{:,.0f}".format(summary_stats.total_reads) }}</p>
        </div>
        <div class="summary-card">
            <h3>Average GC Content</h3>
            <p style="font-size: 1.5em; margin: 0;">
                {{ "%.1f"|format(summary_stats.average_gc_content * 100) }}%</p>
        </div>
        <div class="summary-card">
            <h3>High Quality</h3>
            <p style="font-size: 1.5em; margin: 0;">
                {{ summary_stats.quality_grade_distribution.get('excellent', 0)
                + summary_stats.quality_grade_distribution.get('good', 0) }}</p>
        </div>
        <div class="summary-card">
            <h3>Contamination Issues</h3>
            <p style="font-size: 1.5em; margin: 0;">
                {{ summary_stats.high_contamination_count }}</p>
        </div>
    </div>

    {% if anomaly_report.anomalous_datasets %}
    <div class="anomaly-section">
        <h3>Anomalies Detected</h3>
        <p><strong>{{ anomaly_report.anomalous_datasets|length }}</strong>
            datasets flagged for review:</p>
        <ul>
            {% for dataset in anomaly_report.anomalous_datasets[:10] %}
            <li><strong>{{ dataset }}</strong>:
                {{ anomaly_report.explanations.get(dataset, 'Multiple issues') }}</li>
            {% endfor %}
            {% if anomaly_report.anomalous_datasets|length > 10 %}
            <li>... and {{ anomaly_report.anomalous_datasets|length - 10 }} more</li>
            {% endif %}
        </ul>
    </div>
    {% endif %}

    {% if plots %}
    <h2>Quality Metrics</h2>
    {% for plot_name, plot_html in plots.items() %}
    <div class="plot-container">
        {{ plot_html|safe }}
    </div>
    {% endfor %}
    {% endif %}

</body>
</html>
        """

        template = Environment(loader=BaseLoader(), autoescape=True).from_string(template_str)
        return template.render(plotly_js=plotly_js_script(), **dashboard_data)

    def _generate_comparative_html(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML content for comparative analysis report."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_comparative_html(report_data)

        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    {{ plotly_js|safe }}
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #9b59b6; color: white;
            padding: 20px; border-radius: 5px; }
        .group-grid { display: grid; gap: 20px; margin: 20px 0;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
        .group-card { background-color: #f8f9fa; padding: 15px;
            border-radius: 5px; border-left: 4px solid #9b59b6; }
        .plot-container { margin: 20px 0; border: 1px solid #ddd;
            border-radius: 5px; }
        .significant { background-color: #fff3cd; padding: 10px;
            border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ timestamp }}</p>
    </div>

    <h2>Group Summary</h2>
    <div class="group-grid">
        {% for group_name, count in group_counts.items() %}
        <div class="group-card">
            <h3>{{ group_name }}</h3>
            <p style="font-size: 1.5em; margin: 0;">{{ count }} datasets</p>
        </div>
        {% endfor %}
    </div>

    {% if comparison.statistical_tests %}
    <h2>Statistical Tests</h2>
    {% for test_name, test_result in comparison.statistical_tests.items() %}
    <div class="{% if test_result.significant %}significant{% endif %}">
        <h4>{{ test_name.replace('_', ' ').title() }}</h4>
        <p><strong>Test:</strong> {{ test_result.test }}</p>
        <p><strong>P-value:</strong> {{ "%.4f"|format(test_result.p_value) }}</p>
        <p><strong>Significant:</strong> {{ "Yes" if test_result.significant else "No" }}</p>
    </div>
    {% endfor %}
    {% endif %}

    {% if plots %}
    <h2>Comparative Visualizations</h2>
    {% for plot_name, plot_html in plots.items() %}
    <div class="plot-container">
        {{ plot_html|safe }}
    </div>
    {% endfor %}
    {% endif %}

    {% if comparison.recommendations %}
    <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
            {% for rec in comparison.recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

</body>
</html>
        """

        template = Environment(loader=BaseLoader(), autoescape=True).from_string(template_str)
        return template.render(plotly_js=plotly_js_script(), **report_data)

    def _generate_simple_download_html(self, report_data: Dict[str, Any]) -> str:
        """Generate simple HTML without Jinja2."""
        session = report_data["session"]
        stats = report_data["summary_stats"]

        session_id = html_escape(str(session.session_id))
        timestamp = html_escape(str(report_data["timestamp"]))

        html = f"""
<!DOCTYPE html>
<html>
<head><title>SRA Download Summary</title></head>
<body>
    <h1>SRA Download Summary</h1>
    <p>Session: {session_id}</p>
    <p>Generated: {timestamp}</p>

    <h2>Summary</h2>
    <ul>
        <li>Total Downloads: {stats['total_accessions']}</li>
        <li>Success Rate: {stats['success_rate']*100:.1f}%</li>
        <li>Total Size: {stats['total_size_mb']/1024:.1f} GB</li>
        <li>Average Speed: {stats['average_speed_mbps']:.1f} MB/min</li>
    </ul>

    <h2>Results</h2>
    <table border="1">
        <tr><th>Accession</th><th>Status</th><th>Size (MB)</th><th>Progress</th></tr>
        """

        for accession, result in session.download_results.items():
            acc_escaped = html_escape(str(accession))
            status_escaped = html_escape(str(result.status))
            html += f"""
        <tr>
            <td>{acc_escaped}</td>
            <td>{status_escaped}</td>
            <td>{result.downloaded_mb:.1f}</td>
            <td>{result.progress_pct:.1f}%</td>
        </tr>
            """

        html += """
    </table>
</body>
</html>
        """

        return html

    def _generate_simple_quality_html(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate simple quality HTML without Jinja2."""
        title = html_escape(str(dashboard_data["title"]))
        timestamp = html_escape(str(dashboard_data["timestamp"]))
        return f"""
<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
    <h1>{title}</h1>
    <p>Generated: {timestamp}</p>
    <p>Total Datasets: {dashboard_data['total_datasets']}</p>

    <h2>Summary Statistics</h2>
    <ul>
        <li>Total Reads: {dashboard_data['summary_stats']['total_reads']:,}</li>
        <li>Average GC Content: {dashboard_data['summary_stats']['average_gc_content']*100:.1f}%</li>
        <li>High Contamination Count: {dashboard_data['summary_stats']['high_contamination_count']}</li>
    </ul>
</body>
</html>
        """

    def _generate_simple_comparative_html(self, report_data: Dict[str, Any]) -> str:
        """Generate simple comparative HTML without Jinja2."""
        title = html_escape(str(report_data["title"]))
        timestamp = html_escape(str(report_data["timestamp"]))
        group_items = "".join(
            [
                f"<li>{html_escape(str(name))}: {count} datasets</li>"
                for name, count in report_data["group_counts"].items()
            ]
        )
        return f"""
<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
    <h1>{title}</h1>
    <p>Generated: {timestamp}</p>

    <h2>Group Sizes</h2>
    <ul>
        {group_items}
    </ul>
</body>
</html>
        """
