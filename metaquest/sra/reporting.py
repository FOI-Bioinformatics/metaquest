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
from metaquest.utils.html import CATEGORICAL_COLORS, REPORT_CSS, plotly_js_script, plotly_layout

# Quality-grade colours drawn from the validated categorical palette (ordinal:
# excellent -> poor).
_GRADE_COLORS = {"excellent": "#008300", "good": "#2a78d6", "fair": "#eda100", "poor": "#e34948"}

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

        # The overall success rate is already carried by the summary stat card and
        # the per-row green/red status in the results table, so it is not repeated
        # as a chart here.

        # Download speeds histogram
        speeds = [r.speed_mbps for r in session.download_results.values() if r.speed_mbps > 0]
        if speeds:
            fig_speeds = go.Figure(data=[go.Histogram(x=speeds, nbinsx=20)])
            fig_speeds.update_layout(**plotly_layout())
            fig_speeds.update_layout(
                title_text="Download speed distribution", xaxis_title="Speed (MB/s)", yaxis_title="Count"
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
            fig_scatter.update_layout(**plotly_layout())
            fig_scatter.update_layout(
                title_text="File size vs download time",
                xaxis_title="File size (MB)",
                yaxis_title="Download time (seconds)",
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
                    marker_color=[_GRADE_COLORS.get(str(g), CATEGORICAL_COLORS[0]) for g in grade_counts.index],
                )
            ]
        )
        fig_grades.update_layout(**plotly_layout())
        fig_grades.update_layout(title_text="Quality grade distribution")
        plots["quality_grades"] = pyo.plot(fig_grades, output_type="div", include_plotlyjs=False)

        # GC content distribution
        gc_contents = [p.gc_content for p in profiles.values()]
        fig_gc = go.Figure(data=[go.Histogram(x=gc_contents, nbinsx=25)])
        fig_gc.update_layout(**plotly_layout())
        fig_gc.update_layout(title_text="GC content distribution", xaxis_title="GC content", yaxis_title="Count")
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
        fig_complexity.update_layout(**plotly_layout())
        fig_complexity.update_layout(
            title_text="Read length vs sequence complexity",
            xaxis_title="Average read length (bp)",
            yaxis_title="Complexity score",
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

                    fig_box.update_layout(**plotly_layout())
                    fig_box.update_layout(
                        title_text=f"{col.replace('_', ' ').title()} by group",
                        yaxis_title=col.replace("_", " ").title(),
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
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SRA Download Summary - {{ session.session_id }}</title>
    {{ plotly_js|safe }}
    <style>{{ report_css|safe }}</style>
</head>
<body>
    <header class="mq-header"><div class="mq-wrap">
        <p class="mq-eyebrow">MetaQuest &middot; SRA download</p>
        <h1 class="mq-title">Download summary</h1>
        <p class="mq-readout"><span>session <b>{{ session.session_id }}</b></span>
        <span>generated <b>{{ timestamp }}</b></span></p>
    </div></header>
    <main class="mq-wrap">
        <section class="mq-stats" aria-label="Download summary">
            <div class="mq-stat"><p class="k">Total downloads</p>
                <div class="v">{{ summary_stats.total_accessions }}</div></div>
            <div class="mq-stat"><p class="k">Success rate</p>
                <div class="v">{{ "%.1f"|format(summary_stats.success_rate * 100) }}%</div></div>
            <div class="mq-stat"><p class="k">Total size</p>
                <div class="v">{{ "%.1f"|format(summary_stats.total_size_mb / 1024) }}<span
                    style="font-size:0.9rem"> GB</span></div></div>
            <div class="mq-stat"><p class="k">Average speed</p>
                <div class="v">{{ "%.1f"|format(summary_stats.average_speed_mbps) }}<span
                    style="font-size:0.9rem"> MB/min</span></div></div>
        </section>

        {% if plots %}
        <section class="mq-section">
            <h2>Download analytics</h2>
            <div class="mq-grid">
            {% for plot_name, plot_html in plots.items() %}
                <div class="mq-panel">{{ plot_html|safe }}</div>
            {% endfor %}
            </div>
        </section>
        {% endif %}

        <section class="mq-section">
            <h2>Download results</h2>
            <div class="mq-table-wrap">
                <table class="mq-table">
                    <thead><tr>
                        <th>Accession</th><th>Status</th><th>Size (MB)</th>
                        <th>Progress</th><th>Speed (MB/s)</th><th>Retries</th>
                    </tr></thead>
                    <tbody>
                    {% for accession, result in session.download_results.items() %}
                        <tr>
                            <td>{{ accession }}</td>
                            <td class="{{ 'mq-ok' if result.status == 'completed' else 'mq-bad' }}">{{
                                result.status.title() }}</td>
                            <td class="mq-num">{{ "%.1f"|format(result.downloaded_mb) }}</td>
                            <td class="mq-num">{{ "%.1f"|format(result.progress_pct) }}%</td>
                            <td class="mq-num">{{ "%.2f"|format(result.speed_mbps) }}</td>
                            <td class="mq-num">{{ result.retry_count }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>
        <footer class="mq-footer">Generated by MetaQuest</footer>
    </main>
</body>
</html>
        """

        template = Environment(loader=BaseLoader(), autoescape=True).from_string(template_str)
        return template.render(plotly_js=plotly_js_script(), report_css=REPORT_CSS, **report_data)

    def _generate_quality_html(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML content for quality dashboard."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_quality_html(dashboard_data)

        template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }}</title>
    {{ plotly_js|safe }}
    <style>{{ report_css|safe }}</style>
</head>
<body>
    <header class="mq-header"><div class="mq-wrap">
        <p class="mq-eyebrow">MetaQuest &middot; SRA quality</p>
        <h1 class="mq-title">{{ title }}</h1>
        <p class="mq-readout"><span>generated <b>{{ timestamp }}</b></span>
        <span><b>{{ total_datasets }}</b> datasets</span></p>
    </div></header>
    <main class="mq-wrap">
        <section class="mq-stats" aria-label="Quality summary">
            <div class="mq-stat"><p class="k">Total reads</p>
                <div class="v">{{ "{:,.0f}".format(summary_stats.total_reads) }}</div></div>
            <div class="mq-stat"><p class="k">Average GC content</p>
                <div class="v">{{ "%.1f"|format(summary_stats.average_gc_content * 100) }}%</div></div>
            <div class="mq-stat"><p class="k">High quality</p>
                <div class="v">{{ summary_stats.quality_grade_distribution.get('excellent', 0)
                    + summary_stats.quality_grade_distribution.get('good', 0) }}</div></div>
            <div class="mq-stat"><p class="k">Contamination issues</p>
                <div class="v">{{ summary_stats.high_contamination_count }}</div></div>
        </section>

        {% if anomaly_report.anomalous_datasets %}
        <div class="mq-note warn">
            <h3>Anomalies detected</h3>
            <p><b>{{ anomaly_report.anomalous_datasets|length }}</b> datasets flagged for review:</p>
            <ul>
                {% for dataset in anomaly_report.anomalous_datasets[:10] %}
                <li><b>{{ dataset }}</b>: {{ anomaly_report.explanations.get(dataset, 'Multiple issues') }}</li>
                {% endfor %}
                {% if anomaly_report.anomalous_datasets|length > 10 %}
                <li>&hellip; and {{ anomaly_report.anomalous_datasets|length - 10 }} more</li>
                {% endif %}
            </ul>
        </div>
        {% endif %}

        {% if plots %}
        <section class="mq-section">
            <h2>Quality metrics</h2>
            <div class="mq-grid">
            {% for plot_name, plot_html in plots.items() %}
                <div class="mq-panel">{{ plot_html|safe }}</div>
            {% endfor %}
            </div>
        </section>
        {% endif %}
        <footer class="mq-footer">Generated by MetaQuest</footer>
    </main>
</body>
</html>
        """

        template = Environment(loader=BaseLoader(), autoescape=True).from_string(template_str)
        return template.render(plotly_js=plotly_js_script(), report_css=REPORT_CSS, **dashboard_data)

    def _generate_comparative_html(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML content for comparative analysis report."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_comparative_html(report_data)

        template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }}</title>
    {{ plotly_js|safe }}
    <style>{{ report_css|safe }}</style>
</head>
<body>
    <header class="mq-header"><div class="mq-wrap">
        <p class="mq-eyebrow">MetaQuest &middot; SRA comparison</p>
        <h1 class="mq-title">{{ title }}</h1>
        <p class="mq-readout"><span>generated <b>{{ timestamp }}</b></span></p>
    </div></header>
    <main class="mq-wrap">
        <section class="mq-stats" aria-label="Group summary">
            {% for group_name, count in group_counts.items() %}
            <div class="mq-stat"><p class="k">{{ group_name }}</p>
                <div class="v">{{ count }}<span style="font-size:0.9rem"> datasets</span></div></div>
            {% endfor %}
        </section>

        {% if comparison.statistical_tests %}
        <section class="mq-section">
            <h2>Statistical tests</h2>
            {% for test_name, test_result in comparison.statistical_tests.items() %}
            <div class="mq-note{% if test_result.significant %} warn{% endif %}">
                <h3>{{ test_name.replace('_', ' ').title() }}</h3>
                <p>{{ test_result.test }} &middot; p-value
                    <b>{{ "%.4f"|format(test_result.p_value) }}</b> &middot;
                    {{ "significant" if test_result.significant else "not significant" }}</p>
            </div>
            {% endfor %}
        </section>
        {% endif %}

        {% if plots %}
        <section class="mq-section">
            <h2>Comparative visualizations</h2>
            <div class="mq-grid">
            {% for plot_name, plot_html in plots.items() %}
                <div class="mq-panel">{{ plot_html|safe }}</div>
            {% endfor %}
            </div>
        </section>
        {% endif %}

        {% if comparison.recommendations %}
        <div class="mq-note">
            <h3>Recommendations</h3>
            <ul>
                {% for rec in comparison.recommendations %}<li>{{ rec }}</li>{% endfor %}
            </ul>
        </div>
        {% endif %}
        <footer class="mq-footer">Generated by MetaQuest</footer>
    </main>
</body>
</html>
        """

        template = Environment(loader=BaseLoader(), autoescape=True).from_string(template_str)
        return template.render(plotly_js=plotly_js_script(), report_css=REPORT_CSS, **report_data)

    @staticmethod
    def _simple_shell(eyebrow: str, title: str, readout: str, body: str) -> str:
        """Wrap fallback (no-Jinja) content in the shared report shell."""
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{REPORT_CSS}</style></head><body>
<header class="mq-header"><div class="mq-wrap">
<p class="mq-eyebrow">{eyebrow}</p>
<h1 class="mq-title">{title}</h1>
<p class="mq-readout">{readout}</p></div></header>
<main class="mq-wrap">{body}
<footer class="mq-footer">Generated by MetaQuest</footer>
</main></body></html>"""

    def _generate_simple_download_html(self, report_data: Dict[str, Any]) -> str:
        """Generate simple HTML without Jinja2."""
        session = report_data["session"]
        stats = report_data["summary_stats"]
        session_id = html_escape(str(session.session_id))
        timestamp = html_escape(str(report_data["timestamp"]))

        rows = ""
        for accession, result in session.download_results.items():
            cls = "mq-ok" if str(result.status) == "completed" else "mq-bad"
            rows += (
                f"<tr><td>{html_escape(str(accession))}</td>"
                f'<td class="{cls}">{html_escape(str(result.status).title())}</td>'
                f'<td class="mq-num">{result.downloaded_mb:.1f}</td>'
                f'<td class="mq-num">{result.progress_pct:.1f}%</td></tr>'
            )

        body = f"""
<section class="mq-stats" aria-label="Download summary">
<div class="mq-stat"><p class="k">Total downloads</p><div class="v">{stats['total_accessions']}</div></div>
<div class="mq-stat"><p class="k">Success rate</p><div class="v">{stats['success_rate'] * 100:.1f}%</div></div>
<div class="mq-stat"><p class="k">Total size</p><div class="v">{stats['total_size_mb'] / 1024:.1f}\
<span style="font-size:0.9rem"> GB</span></div></div>
<div class="mq-stat"><p class="k">Average speed</p><div class="v">{stats['average_speed_mbps']:.1f}\
<span style="font-size:0.9rem"> MB/min</span></div></div>
</section>
<section class="mq-section"><h2>Download results</h2>
<div class="mq-table-wrap"><table class="mq-table">
<thead><tr><th>Accession</th><th>Status</th><th>Size (MB)</th><th>Progress</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>"""

        readout = f"<span>session <b>{session_id}</b></span> <span>generated <b>{timestamp}</b></span>"
        return self._simple_shell("MetaQuest &middot; SRA download", "Download summary", readout, body)

    def _generate_simple_quality_html(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate simple quality HTML without Jinja2."""
        title = html_escape(str(dashboard_data["title"]))
        timestamp = html_escape(str(dashboard_data["timestamp"]))
        s = dashboard_data["summary_stats"]

        body = f"""
<section class="mq-stats" aria-label="Quality summary">
<div class="mq-stat"><p class="k">Total reads</p><div class="v">{s['total_reads']:,}</div></div>
<div class="mq-stat"><p class="k">Average GC content</p>
<div class="v">{s['average_gc_content'] * 100:.1f}%</div></div>
<div class="mq-stat"><p class="k">High contamination</p><div class="v">{s['high_contamination_count']}</div></div>
</section>"""

        readout = (
            f"<span>generated <b>{timestamp}</b></span> <span><b>{dashboard_data['total_datasets']}</b> datasets</span>"
        )
        return self._simple_shell("MetaQuest &middot; SRA quality", title, readout, body)

    def _generate_simple_comparative_html(self, report_data: Dict[str, Any]) -> str:
        """Generate simple comparative HTML without Jinja2."""
        title = html_escape(str(report_data["title"]))
        timestamp = html_escape(str(report_data["timestamp"]))
        cards = "".join(
            f'<div class="mq-stat"><p class="k">{html_escape(str(name))}</p>'
            f'<div class="v">{count}<span style="font-size:0.9rem"> datasets</span></div></div>'
            for name, count in report_data["group_counts"].items()
        )
        body = f'<section class="mq-stats" aria-label="Group summary">{cards}</section>'
        readout = f"<span>generated <b>{timestamp}</b></span>"
        return self._simple_shell("MetaQuest &middot; SRA comparison", title, readout, body)
