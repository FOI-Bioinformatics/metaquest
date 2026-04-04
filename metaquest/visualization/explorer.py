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

    top_families = (
        long_df.groupby("family")["sample"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
    )

    return {
        "total_samples": samples_with_containment,
        "total_genomes": total_genomes,
        "num_families": len(families),
        "num_genera": len(genera),
        "containment_min": float(long_df["containment"].min()),
        "containment_max": float(long_df["containment"].max()),
        "top_families": [
            {"name": name, "count": int(count)} for name, count in top_families.items()
        ],
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
        color_continuous_scale="Viridis",
        title="Taxonomy Sunburst",
    )
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0), height=500)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_heatmap(long_df: pd.DataFrame) -> str:
    """Build sample-by-taxon heatmap. Returns Plotly HTML fragment."""
    pivot = long_df.pivot_table(
        index="sample", columns="family", values="containment", aggfunc="max", fill_value=0
    )
    if pivot.empty:
        return ""
    fig = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        labels=dict(x="Family", y="Sample", color="Containment"),
        title="Sample x Family Containment Heatmap",
        aspect="auto",
    )
    fig.update_layout(height=max(300, 20 * len(pivot.index) + 100))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_distributions(long_df: pd.DataFrame) -> tuple:
    """Build distribution plots. Returns (box_html, bar_html)."""
    box_fig = px.box(
        long_df,
        x="family",
        y="containment",
        title="Containment Distribution per Family",
        labels={"family": "Family", "containment": "Containment"},
    )
    box_fig.update_layout(xaxis_tickangle=-45, height=450)
    box_html = box_fig.to_html(full_html=False, include_plotlyjs=False)

    genus_counts = long_df.groupby("genus")["sample"].nunique().reset_index()
    genus_counts.columns = ["genus", "sample_count"]
    genus_counts = genus_counts.sort_values("sample_count", ascending=False).head(30)
    bar_fig = px.bar(
        genus_counts,
        x="genus",
        y="sample_count",
        title="Sample Count per Genus",
        labels={"genus": "Genus", "sample_count": "Samples"},
    )
    bar_fig.update_layout(xaxis_tickangle=-45, height=450)
    bar_html = bar_fig.to_html(full_html=False, include_plotlyjs=False)

    return box_html, bar_html


def _build_table_html(long_df: pd.DataFrame) -> str:
    """Build HTML table rows from long-format data."""
    if long_df.empty:
        return ""
    rows = []
    for _, r in long_df.iterrows():
        cells = "".join(
            f"<td>{html_escape(str(r.get(c, '')))}</td>"
            for c in ["sample", "genome", "containment", "species", "genus", "family", "location", "date"]
        )
        rows.append(f"<tr>{cells}</tr>")
    return "\n".join(rows)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: "Helvetica Neue", Arial, sans-serif;
            margin: 0; padding: 0;
            background: #fafafa; color: #333;
        }
        .header {
            background: #2c3e50; color: #ecf0f1;
            padding: 24px 32px; margin-bottom: 24px;
        }
        .header h1 { margin: 0 0 4px 0; font-size: 1.6em; }
        .header p { margin: 0; opacity: 0.8; font-size: 0.9em; }
        .container { max-width: 1400px; margin: 0 auto; padding: 0 24px 48px; }
        .cards {
            display: flex; flex-wrap: wrap; gap: 16px;
            margin-bottom: 32px;
        }
        .card {
            flex: 1 1 180px; background: #fff;
            border-radius: 6px; padding: 16px 20px;
            border-left: 4px solid #2980b9;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .card h3 { margin: 0 0 6px; font-size: 0.85em; color: #7f8c8d; text-transform: uppercase; }
        .card .value { font-size: 1.8em; font-weight: 600; color: #2c3e50; }
        .top-families { margin-top: 8px; font-size: 0.85em; color: #555; }
        .section { margin-bottom: 32px; }
        .section h2 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }
        .chart-row {
            display: flex; flex-wrap: wrap; gap: 24px;
        }
        .chart-cell { flex: 1 1 48%; min-width: 300px; background: #fff;
            border-radius: 6px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .chart-full { background: #fff; border-radius: 6px; padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .search-bar { margin-bottom: 12px; }
        .search-bar input {
            padding: 8px 12px; width: 300px; border: 1px solid #ccc;
            border-radius: 4px; font-size: 0.95em;
        }
        .search-bar button {
            padding: 8px 16px; margin-left: 8px;
            background: #2980b9; color: #fff; border: none;
            border-radius: 4px; cursor: pointer; font-size: 0.95em;
        }
        .search-bar button:hover { background: #2471a3; }
        table.results {
            width: 100%; border-collapse: collapse;
            font-size: 0.9em;
        }
        table.results th {
            background: #2c3e50; color: #fff;
            padding: 10px 12px; text-align: left; cursor: pointer;
            user-select: none; position: sticky; top: 0;
        }
        table.results th:hover { background: #34495e; }
        table.results td {
            padding: 8px 12px; border-bottom: 1px solid #eee;
        }
        table.results tr:hover { background: #f5f9fc; }
        @media (max-width: 800px) {
            .chart-row { flex-direction: column; }
            .chart-cell { flex: 1 1 100%; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated by MetaQuest</p>
    </div>
    <div class="container">
        <div class="cards">
            <div class="card">
                <h3>Samples</h3>
                <div class="value">{{ summary.total_samples }}</div>
            </div>
            <div class="card">
                <h3>Genomes</h3>
                <div class="value">{{ summary.total_genomes }}</div>
            </div>
            <div class="card">
                <h3>Families</h3>
                <div class="value">{{ summary.num_families }}</div>
            </div>
            <div class="card">
                <h3>Genera</h3>
                <div class="value">{{ summary.num_genera }}</div>
            </div>
            <div class="card">
                <h3>Containment Range</h3>
                <div class="value">{{ "%.3f"|format(summary.containment_min) }} -
                {{ "%.3f"|format(summary.containment_max) }}</div>
            </div>
            {% if summary.top_families %}
            <div class="card" style="flex: 2 1 300px;">
                <h3>Top Families by Sample Count</h3>
                <div class="top-families">
                {% for fam in summary.top_families %}
                    {{ fam.name }} ({{ fam.count }}){% if not loop.last %}, {% endif %}
                {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>

        {% if sunburst_html or heatmap_html %}
        <div class="section">
            <h2>Taxonomy Overview</h2>
            <div class="chart-row">
                {% if sunburst_html %}
                <div class="chart-cell">{{ sunburst_html|safe }}</div>
                {% endif %}
                {% if heatmap_html %}
                <div class="chart-cell">{{ heatmap_html|safe }}</div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>Results Table</h2>
            <div class="search-bar">
                <input type="text" id="searchInput" onkeyup="filterTable()"
                       placeholder="Search across all columns...">
                <button onclick="exportCSV()">Export CSV</button>
            </div>
            <div style="max-height: 600px; overflow-y: auto;">
            <table class="results" id="resultsTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Sample</th>
                        <th onclick="sortTable(1)">Genome</th>
                        <th onclick="sortTable(2)">Containment</th>
                        <th onclick="sortTable(3)">Species</th>
                        <th onclick="sortTable(4)">Genus</th>
                        <th onclick="sortTable(5)">Family</th>
                        <th onclick="sortTable(6)">Location</th>
                        <th onclick="sortTable(7)">Date</th>
                    </tr>
                </thead>
                <tbody>
                    {{ table_rows|safe }}
                </tbody>
            </table>
            </div>
        </div>

        {% if box_html or bar_html %}
        <div class="section">
            <h2>Distribution Plots</h2>
            <div class="chart-row">
                {% if box_html %}
                <div class="chart-cell">{{ box_html|safe }}</div>
                {% endif %}
                {% if bar_html %}
                <div class="chart-cell">{{ bar_html|safe }}</div>
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>

    <script>
    function filterTable() {
        var input = document.getElementById("searchInput").value.toUpperCase();
        var table = document.getElementById("resultsTable");
        var tr = table.getElementsByTagName("tr");
        for (var i = 1; i < tr.length; i++) {
            var cells = tr[i].getElementsByTagName("td");
            var match = false;
            for (var j = 0; j < cells.length; j++) {
                if (cells[j].textContent.toUpperCase().indexOf(input) > -1) {
                    match = true; break;
                }
            }
            tr[i].style.display = match ? "" : "none";
        }
    }

    function sortTable(col) {
        var table = document.getElementById("resultsTable");
        var rows = Array.from(table.tBodies[0].rows);
        var asc = table.getAttribute("data-sort-col") == col
                  && table.getAttribute("data-sort-dir") != "asc";
        rows.sort(function(a, b) {
            var va = a.cells[col].textContent;
            var vb = b.cells[col].textContent;
            var na = parseFloat(va), nb = parseFloat(vb);
            if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
            return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        });
        rows.forEach(function(r) { table.tBodies[0].appendChild(r); });
        table.setAttribute("data-sort-col", col);
        table.setAttribute("data-sort-dir", asc ? "asc" : "desc");
    }

    function exportCSV() {
        var table = document.getElementById("resultsTable");
        var rows = table.querySelectorAll("tr");
        var csv = [];
        rows.forEach(function(row) {
            if (row.style.display === "none") return;
            var cols = row.querySelectorAll("td, th");
            var line = [];
            cols.forEach(function(col) {
                var val = col.textContent.replace(/"/g, '""');
                line.push('"' + val + '"');
            });
            csv.push(line.join(","));
        });
        var blob = new Blob([csv.join("\\n")], {type: "text/csv"});
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "containment_results.csv";
        a.click();
    }
    </script>
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
        )
    return _assemble_html_simple(title, summary, table_rows, sunburst_html, heatmap_html, box_html, bar_html)


def _assemble_html_simple(title, summary, table_rows, sunburst_html, heatmap_html, box_html, bar_html):
    """Fallback HTML assembly without Jinja2."""
    safe_title = html_escape(title)

    top_fam_str = ""
    if summary.get("top_families"):
        top_fam_str = ", ".join(f"{f['name']} ({f['count']})" for f in summary["top_families"])

    charts_section = ""
    if sunburst_html or heatmap_html:
        charts_section = f"""
        <div class="section"><h2>Taxonomy Overview</h2>
        <div class="chart-row">
        {"<div class='chart-cell'>" + sunburst_html + "</div>" if sunburst_html else ""}
        {"<div class='chart-cell'>" + heatmap_html + "</div>" if heatmap_html else ""}
        </div></div>"""

    dist_section = ""
    if box_html or bar_html:
        dist_section = f"""
        <div class="section"><h2>Distribution Plots</h2>
        <div class="chart-row">
        {"<div class='chart-cell'>" + box_html + "</div>" if box_html else ""}
        {"<div class='chart-cell'>" + bar_html + "</div>" if bar_html else ""}
        </div></div>"""

    c_min = f"{summary['containment_min']:.3f}"
    c_max = f"{summary['containment_max']:.3f}"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{safe_title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #fafafa; }}
.header {{ background: #2c3e50; color: #ecf0f1; padding: 24px 32px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 0 24px 48px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 24px 0; }}
.card {{ flex: 1 1 180px; background: #fff; border-radius: 6px; padding: 16px;
    border-left: 4px solid #2980b9; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.card h3 {{ margin: 0 0 6px; font-size: 0.85em; color: #7f8c8d; }}
.card .value {{ font-size: 1.8em; font-weight: 600; }}
.section {{ margin-bottom: 32px; }}
.chart-row {{ display: flex; flex-wrap: wrap; gap: 24px; }}
.chart-cell {{ flex: 1 1 48%; min-width: 300px; background: #fff; border-radius: 6px; padding: 12px; }}
table.results {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
table.results th {{ background: #2c3e50; color: #fff; padding: 10px; cursor: pointer; }}
table.results td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
</style></head><body>
<div class="header"><h1>{safe_title}</h1><p>Generated by MetaQuest</p></div>
<div class="container">
<div class="cards">
<div class="card"><h3>Samples</h3><div class="value">{summary['total_samples']}</div></div>
<div class="card"><h3>Genomes</h3><div class="value">{summary['total_genomes']}</div></div>
<div class="card"><h3>Families</h3><div class="value">{summary['num_families']}</div></div>
<div class="card"><h3>Genera</h3><div class="value">{summary['num_genera']}</div></div>
<div class="card"><h3>Range</h3><div class="value">{c_min} - {c_max}</div></div>
{"<div class='card'><h3>Top Families</h3><div>" + top_fam_str + "</div></div>" if top_fam_str else ""}
</div>
{charts_section}
<div class="section"><h2>Results Table</h2>
<input type="text" id="searchInput" onkeyup="filterTable()"
    placeholder="Search..." style="padding:8px;width:300px;margin-bottom:12px;">
<table class="results" id="resultsTable"><thead><tr>
<th onclick="sortTable(0)">Sample</th><th onclick="sortTable(1)">Genome</th>
<th onclick="sortTable(2)">Containment</th><th onclick="sortTable(3)">Species</th>
<th onclick="sortTable(4)">Genus</th><th onclick="sortTable(5)">Family</th>
<th onclick="sortTable(6)">Location</th><th onclick="sortTable(7)">Date</th>
</tr></thead><tbody>{table_rows}</tbody></table></div>
{dist_section}
</div>
<script>
function filterTable() {{
    var input = document.getElementById("searchInput").value.toUpperCase();
    var table = document.getElementById("resultsTable");
    var tr = table.getElementsByTagName("tr");
    for (var i = 1; i < tr.length; i++) {{
        var cells = tr[i].getElementsByTagName("td");
        var match = false;
        for (var j = 0; j < cells.length; j++) {{
            if (cells[j].textContent.toUpperCase().indexOf(input) > -1) {{
                match = true; break;
            }}
        }}
        tr[i].style.display = match ? "" : "none";
    }}
}}
function sortTable(col) {{
    var table = document.getElementById("resultsTable");
    var rows = Array.from(table.tBodies[0].rows);
    var asc = table.getAttribute("data-sort-col") == col && table.getAttribute("data-sort-dir") != "asc";
    rows.sort(function(a, b) {{
        var va = a.cells[col].textContent, vb = b.cells[col].textContent;
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(function(r) {{ table.tBodies[0].appendChild(r); }});
    table.setAttribute("data-sort-col", col);
    table.setAttribute("data-sort-dir", asc ? "asc" : "desc");
}}
function exportCSV() {{
    var table = document.getElementById("resultsTable");
    var rows = table.querySelectorAll("tr");
    var csv = [];
    rows.forEach(function(row) {{
        if (row.style.display === "none") return;
        var cols = row.querySelectorAll("td, th");
        var line = [];
        cols.forEach(function(col) {{
            var val = col.textContent.replace(/"/g, '""');
            line.push('"' + val + '"');
        }});
        csv.push(line.join(","));
    }});
    var blob = new Blob([csv.join("\\n")], {{type: "text/csv"}});
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "containment_results.csv";
    a.click();
}}
</script></body></html>"""
