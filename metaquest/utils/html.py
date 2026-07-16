"""Shared HTML assets for MetaQuest report and dashboard generators.

Centralizes the Plotly CDN ``<script>`` tag and the sortable/filterable table
JavaScript so the several HTML generators do not each carry their own copy.
"""

# Fallback plotly.js version used only if the installed plotly cannot report
# its bundled version. Kept close to a recent release.
_FALLBACK_PLOTLYJS_VERSION = "3.0.1"


def plotly_cdn_script() -> str:
    """Return a ``<script>`` tag loading the plotly.js the report expects.

    The version is taken from the installed plotly.py so the CDN runtime
    matches the figures it renders. This avoids the old
    ``plotly-latest.min.js`` alias, which is frozen at plotly.js v1 and cannot
    render output produced by modern plotly.py.
    """
    try:
        from plotly.offline import get_plotlyjs_version

        version = get_plotlyjs_version()
    except Exception:
        version = _FALLBACK_PLOTLYJS_VERSION
    return f'<script src="https://cdn.plot.ly/plotly-{version}.min.js"></script>'


# Client-side helpers for the interactive results table: full-text filtering,
# column sorting (numeric-aware), and CSV export of the visible rows. Uses
# single braces so it can be dropped into either a Jinja template (via a
# ``|safe`` variable) or an f-string (as a substituted value).
TABLE_SCRIPT = """<script>
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
</script>"""
