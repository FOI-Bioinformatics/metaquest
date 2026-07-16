"""
Interactive HTML containment explorer for MetaQuest.

Generates a self-contained HTML file with interactive Plotly charts
for exploring containment results grouped by taxonomy. The explorer
provides summary statistics, a taxonomy sunburst, sample-by-taxon
heatmap, filterable results table, and distribution plots.
"""

import logging
from html import escape as html_escape
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from metaquest.core.models import TaxonomyInfo
from metaquest.core.utils import get_genome_columns
from metaquest.utils.html import (
    CONTAINMENT_SCALE,
    REPORT_CSS,
    TABLE_SCRIPT,
    plotly_js_script,
    plotly_layout,
)

try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from jinja2 import Environment, BaseLoader

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

logger = logging.getLogger(__name__)


def generate_containment_explorer(
    containment_df: pd.DataFrame,
    taxonomy: Dict[str, TaxonomyInfo],
    metadata: Optional[pd.DataFrame] = None,
    output_file: Union[str, Path] = "containment_explorer.html",
    title: str = "MetaQuest Containment Explorer",
    min_containment: float = 0.0,
) -> Path:
    """Generate interactive HTML containment explorer.

    Produces a single HTML file with interactive charts for exploring
    containment results grouped by taxonomy.

    Args:
        containment_df: Parsed containment DataFrame (samples x genomes).
        taxonomy: Dict mapping genome_id to TaxonomyInfo.
        metadata: Optional metadata DataFrame indexed by sample accession.
        output_file: Output HTML path.
        title: Report title.
        min_containment: Minimum containment threshold for inclusion.

    Returns:
        Path to generated HTML file.
    """
    output_path = Path(output_file)

    long_df = _build_long_dataframe(containment_df, taxonomy, metadata, min_containment)
    summary = _build_summary_data(long_df, containment_df, taxonomy)

    sunburst_html = ""
    heatmap_html = ""
    box_html = ""
    bar_html = ""

    if PLOTLY_AVAILABLE and not long_df.empty:
        sunburst_html = _build_sunburst(long_df)
        heatmap_html = _build_heatmap(long_df)
        box_html, bar_html = _build_distributions(long_df)

    table_html = _build_table_html(long_df)
    html_content = _assemble_html(title, summary, sunburst_html, heatmap_html, table_html, box_html, bar_html)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)

    logger.info(f"Containment explorer saved to {output_path}")
    return output_path


def _build_long_dataframe(
    containment_df: pd.DataFrame,
    taxonomy: Dict[str, TaxonomyInfo],
    metadata: Optional[pd.DataFrame],
    min_containment: float,
) -> pd.DataFrame:
    """Convert wide containment DataFrame to long format with taxonomy."""
    genome_cols = get_genome_columns(containment_df)
    rows = []
    for sample_id, row in containment_df.iterrows():
        for genome in genome_cols:
            val = row[genome]
            if val > min_containment:
                tax = taxonomy.get(genome)
                entry = {
                    "sample": str(sample_id),
                    "genome": genome,
                    "containment": val,
                    "family": tax.family if tax and tax.family else "Unknown",
                    "genus": tax.genus if tax and tax.genus else "Unknown",
                    "species": tax.species if tax and tax.species else "Unknown",
                }
                if metadata is not None and sample_id in metadata.index:
                    meta_row = metadata.loc[sample_id]
                    entry["location"] = str(meta_row.get("location", ""))
                    entry["date"] = str(meta_row.get("collection_date", ""))
                else:
                    entry["location"] = ""
                    entry["date"] = ""
                rows.append(entry)
    return pd.DataFrame(rows)


def _build_summary_data(
    long_df: pd.DataFrame,
    containment_df: pd.DataFrame,
    taxonomy: Dict[str, TaxonomyInfo],
) -> dict:
    """Compute summary statistics for the explorer header cards."""
    if long_df.empty:
        return {
            "total_samples": 0,
            "total_genomes": 0,
            "num_families": 0,
            "num_genera": 0,
            "containment_min": 0.0,
            "containment_max": 0.0,
            "top_families": [],
        }

    genome_cols = get_genome_columns(containment_df)
    samples_with_containment = int(long_df["sample"].nunique())
    total_genomes = len(genome_cols)
    families = long_df.loc[long_df["family"] != "Unknown", "family"].unique()
    genera = long_df.loc[long_df["genus"] != "Unknown", "genus"].unique()

    top_families = long_df.groupby("family")["sample"].nunique().sort_values(ascending=False).head(5)

    return {
        "total_samples": samples_with_containment,
        "total_genomes": total_genomes,
        "num_families": len(families),
        "num_genera": len(genera),
        "containment_min": float(long_df["containment"].min()),
        "containment_max": float(long_df["containment"].max()),
        "top_families": [{"name": name, "count": int(count)} for name, count in top_families.items()],
    }


def _build_sunburst(long_df: pd.DataFrame) -> str:
    """Build taxonomy sunburst chart. Returns Plotly HTML fragment."""
    sunburst_df = (
        long_df.groupby(["family", "genus", "species"])
        .agg(sample_count=("sample", "nunique"), mean_containment=("containment", "mean"))
        .reset_index()
    )
    fig = px.sunburst(
        sunburst_df,
        path=["family", "genus", "species"],
        values="sample_count",
        color="mean_containment",
        color_continuous_scale=CONTAINMENT_SCALE,
        title="Mean containment by taxon",
    )
    fig.update_layout(**plotly_layout(height=500, margin=dict(t=46, l=6, r=6, b=6)))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_heatmap(long_df: pd.DataFrame) -> str:
    """Build sample-by-taxon heatmap. Returns Plotly HTML fragment."""
    pivot = long_df.pivot_table(index="sample", columns="family", values="containment", aggfunc="max", fill_value=0)
    if pivot.empty:
        return ""
    fig = px.imshow(
        pivot,
        color_continuous_scale=CONTAINMENT_SCALE,
        labels=dict(x="Family", y="Sample", color="Containment"),
        title="Sample-by-family containment",
        aspect="auto",
    )
    fig.update_layout(**plotly_layout(height=max(320, 22 * len(pivot.index) + 110)))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_distributions(long_df: pd.DataFrame) -> tuple:
    """Build distribution plots. Returns (box_html, bar_html)."""
    box_fig = px.box(
        long_df,
        x="family",
        y="containment",
        color="family",
        title="Containment by family",
        labels={"family": "Family", "containment": "Containment"},
    )
    box_fig.update_layout(**plotly_layout(height=450, xaxis_tickangle=-30, showlegend=False))
    box_html = box_fig.to_html(full_html=False, include_plotlyjs=False)

    genus_counts = long_df.groupby("genus")["sample"].nunique().reset_index()
    genus_counts.columns = ["genus", "sample_count"]
    genus_counts = genus_counts.sort_values("sample_count", ascending=False).head(30)
    bar_fig = px.bar(
        genus_counts,
        x="genus",
        y="sample_count",
        title="Samples per genus",
        labels={"genus": "Genus", "sample_count": "Samples"},
    )
    bar_fig.update_layout(**plotly_layout(height=450, xaxis_tickangle=-30))
    bar_html = bar_fig.to_html(full_html=False, include_plotlyjs=False)

    return box_html, bar_html


def _build_table_html(long_df: pd.DataFrame) -> str:
    """Build HTML table rows from long-format data.

    The containment cell renders an inline heat-bar (width proportional to the
    value) beside the number, so magnitude is scannable down the column while
    the number remains the cell's text for numeric sorting.
    """
    if long_df.empty:
        return ""
    rows = []
    for _, r in long_df.iterrows():
        raw = r.get("containment", 0)
        val = 0.0 if pd.isna(raw) else max(0.0, min(1.0, float(raw)))
        before = "".join(f"<td>{html_escape(str(r.get(c, '')))}</td>" for c in ["sample", "genome"])
        containment = (
            f'<td class="mq-num"><span class="mq-cell">'
            f'<span class="mq-heatbar" style="--v:{val:.3f}"></span>'
            f"<span>{val:.3f}</span></span></td>"
        )
        after = "".join(
            f"<td>{html_escape(str(r.get(c, '')))}</td>" for c in ["species", "genus", "family", "location", "date"]
        )
        rows.append(f"<tr>{before}{containment}{after}</tr>")
    return "\n".join(rows)


_TABLE_HEADERS = ["Sample", "Genome", "Containment", "Species", "Genus", "Family", "Location", "Date"]


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }}</title>
    {{ plotly_js|safe }}
    <style>{{ report_css|safe }}</style>
</head>
<body>
    <a class="mq-skip" href="#results">Skip to results</a>
    <header class="mq-header">
        <div class="mq-wrap">
            <p class="mq-eyebrow">MetaQuest &middot; containment report</p>
            <h1 class="mq-title">{{ title }}</h1>
            <p class="mq-readout">
                <span><b>{{ summary.total_samples }}</b> samples</span>
                <span><b>{{ summary.total_genomes }}</b> genomes</span>
                <span><b>{{ summary.num_families }}</b> families</span>
                <span><b>{{ summary.num_genera }}</b> genera</span>
                <span>containment
                    <b>{{ "%.3f"|format(summary.containment_min) }}</b>&ndash;<b>{{
                        "%.3f"|format(summary.containment_max) }}</b></span>
            </p>
        </div>
    </header>
    <main class="mq-wrap">
        <section class="mq-stats" aria-label="Run summary">
            <div class="mq-stat"><p class="k">Samples</p><div class="v">{{ summary.total_samples }}</div></div>
            <div class="mq-stat"><p class="k">Genomes</p><div class="v">{{ summary.total_genomes }}</div></div>
            <div class="mq-stat"><p class="k">Families</p><div class="v">{{ summary.num_families }}</div></div>
            <div class="mq-stat"><p class="k">Genera</p><div class="v">{{ summary.num_genera }}</div></div>
            {% if summary.top_families %}
            <div class="mq-stat wide">
                <p class="k">Top families by sample count</p>
                <div class="sub">
                {%- for fam in summary.top_families -%}
                    {{ fam.name }} ({{ fam.count }}){% if not loop.last %} &middot; {% endif %}
                {%- endfor -%}
                </div>
            </div>
            {% endif %}
        </section>

        {% if sunburst_html or heatmap_html %}
        <section class="mq-section">
            <h2>Taxonomy overview</h2>
            <div class="mq-grid">
                {% if sunburst_html %}<div class="mq-panel">{{ sunburst_html|safe }}</div>{% endif %}
                {% if heatmap_html %}<div class="mq-panel">{{ heatmap_html|safe }}</div>{% endif %}
            </div>
        </section>
        {% endif %}

        <section class="mq-section" id="results">
            <h2>Results</h2>
            <div class="mq-toolbar">
                <input type="text" id="searchInput" onkeyup="filterTable()"
                       aria-label="Filter results" placeholder="Filter across all columns&hellip;">
                <button class="mq-btn" onclick="exportCSV()">Export CSV</button>
            </div>
            <div class="mq-table-wrap">
                <table class="mq-table" id="resultsTable">
                    <thead>
                        <tr>
                        {%- for h in headers %}
                            <th tabindex="0" role="button"
                                onclick="sortTable({{ loop.index0 }})"
                                onkeydown="sortKey(event,{{ loop.index0 }})">{{ h }}</th>
                        {%- endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {{ table_rows|safe }}
                    </tbody>
                </table>
            </div>
        </section>

        {% if box_html or bar_html %}
        <section class="mq-section">
            <h2>Distributions</h2>
            <div class="mq-grid">
                {% if box_html %}<div class="mq-panel">{{ box_html|safe }}</div>{% endif %}
                {% if bar_html %}<div class="mq-panel">{{ bar_html|safe }}</div>{% endif %}
            </div>
        </section>
        {% endif %}

        <footer class="mq-footer">Generated by MetaQuest</footer>
    </main>

    {{ table_script|safe }}
</body>
</html>"""


def _assemble_html(title, summary, sunburst_html, heatmap_html, table_rows, box_html, bar_html):
    """Assemble the final HTML from components."""
    if JINJA2_AVAILABLE:
        template = Environment(loader=BaseLoader(), autoescape=False).from_string(_HTML_TEMPLATE)
        return template.render(
            title=html_escape(title),
            summary=summary,
            sunburst_html=sunburst_html,
            heatmap_html=heatmap_html,
            table_rows=table_rows,
            box_html=box_html,
            bar_html=bar_html,
            plotly_js=plotly_js_script(),
            report_css=REPORT_CSS,
            table_script=TABLE_SCRIPT,
            headers=_TABLE_HEADERS,
        )
    return _assemble_html_simple(title, summary, table_rows, sunburst_html, heatmap_html, box_html, bar_html)


def _assemble_html_simple(title, summary, table_rows, sunburst_html, heatmap_html, box_html, bar_html):
    """Fallback HTML assembly without Jinja2 (same design system as the Jinja path)."""
    safe_title = html_escape(title)

    top_fam_html = ""
    if summary.get("top_families"):
        top_fam_str = " &middot; ".join(f"{f['name']} ({f['count']})" for f in summary["top_families"])
        top_fam_html = (
            '<div class="mq-stat wide"><p class="k">Top families by sample count</p>'
            f'<div class="sub">{top_fam_str}</div></div>'
        )

    def _panels(*fragments):
        return "".join(f'<div class="mq-panel">{h}</div>' for h in fragments if h)

    charts_section = ""
    if sunburst_html or heatmap_html:
        charts_section = (
            '<section class="mq-section"><h2>Taxonomy overview</h2>'
            f'<div class="mq-grid">{_panels(sunburst_html, heatmap_html)}</div></section>'
        )

    dist_section = ""
    if box_html or bar_html:
        dist_section = (
            '<section class="mq-section"><h2>Distributions</h2>'
            f'<div class="mq-grid">{_panels(box_html, bar_html)}</div></section>'
        )

    c_min = f"{summary['containment_min']:.3f}"
    c_max = f"{summary['containment_max']:.3f}"

    ths = "".join(
        f'<th tabindex="0" role="button" onclick="sortTable({i})" ' f'onkeydown="sortKey(event,{i})">{label}</th>'
        for i, label in enumerate(_TABLE_HEADERS)
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
{plotly_js_script()}
<style>{REPORT_CSS}</style></head><body>
<a class="mq-skip" href="#results">Skip to results</a>
<header class="mq-header"><div class="mq-wrap">
<p class="mq-eyebrow">MetaQuest &middot; containment report</p>
<h1 class="mq-title">{safe_title}</h1>
<p class="mq-readout"><span><b>{summary['total_samples']}</b> samples</span>
<span><b>{summary['total_genomes']}</b> genomes</span>
<span><b>{summary['num_families']}</b> families</span>
<span><b>{summary['num_genera']}</b> genera</span>
<span>containment <b>{c_min}</b>&ndash;<b>{c_max}</b></span></p>
</div></header>
<main class="mq-wrap">
<section class="mq-stats" aria-label="Run summary">
<div class="mq-stat"><p class="k">Samples</p><div class="v">{summary['total_samples']}</div></div>
<div class="mq-stat"><p class="k">Genomes</p><div class="v">{summary['total_genomes']}</div></div>
<div class="mq-stat"><p class="k">Families</p><div class="v">{summary['num_families']}</div></div>
<div class="mq-stat"><p class="k">Genera</p><div class="v">{summary['num_genera']}</div></div>
{top_fam_html}
</section>
{charts_section}
<section class="mq-section" id="results"><h2>Results</h2>
<div class="mq-toolbar">
<input type="text" id="searchInput" onkeyup="filterTable()" aria-label="Filter results"
    placeholder="Filter across all columns&hellip;">
<button class="mq-btn" onclick="exportCSV()">Export CSV</button></div>
<div class="mq-table-wrap"><table class="mq-table" id="resultsTable">
<thead><tr>{ths}</tr></thead><tbody>{table_rows}</tbody></table></div></section>
{dist_section}
<footer class="mq-footer">Generated by MetaQuest</footer>
</main>
{TABLE_SCRIPT}
</body></html>"""
