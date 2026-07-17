"""Tests for shared HTML report assets in metaquest.utils.html."""

from unittest.mock import patch

from metaquest.utils import html


class TestReportCss:
    """Tests for the shared design-system stylesheet."""

    def test_defines_the_shared_component_vocabulary(self):
        for token in ("--paper-bg", "--seq-fill", ".mq-header", ".mq-stat", ".mq-panel", ".mq-table", ".mq-heatbar"):
            assert token in html.REPORT_CSS

    def test_supports_light_and_dark_and_a11y(self):
        assert "prefers-color-scheme: dark" in html.REPORT_CSS
        assert "prefers-reduced-motion" in html.REPORT_CSS
        assert ":focus-visible" in html.REPORT_CSS

    def test_uses_no_remote_resources(self):
        # The reports are offline; the stylesheet must not fetch anything.
        assert "http://" not in html.REPORT_CSS and "https://" not in html.REPORT_CSS


class TestPlotlyLayout:
    """Tests for the shared Plotly styling helper."""

    def test_returns_transparent_themed_layout(self):
        layout = html.plotly_layout()
        assert layout["paper_bgcolor"] == "rgba(0,0,0,0)"
        assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"
        assert layout["colorway"] == html.CATEGORICAL_COLORS

    def test_overrides_win(self):
        layout = html.plotly_layout(height=360, showlegend=False)
        assert layout["height"] == 360
        assert layout["showlegend"] is False


class TestPlotlyJsScript:
    """Tests for the inline Plotly script tag."""

    def test_embeds_plotlyjs_inline(self):
        """The tag embeds the bundled plotly.js inline, not a network URL."""
        with patch("plotly.offline.get_plotlyjs", return_value="PLOTLY_JS_BODY"):
            tag = html.plotly_js_script()
        assert tag == '<script type="text/javascript">PLOTLY_JS_BODY</script>'

    def test_never_references_a_remote_cdn(self):
        """The opening <script> tag must be inline, never loading from a remote host."""
        tag = html.plotly_js_script()
        opening = tag[: tag.index(">") + 1]
        assert "src=" not in opening
        assert "cdn.plot.ly" not in opening

    def test_returns_empty_when_plotly_unavailable(self):
        """A failure to obtain plotly.js yields an empty string, not a crash."""
        with patch("plotly.offline.get_plotlyjs", side_effect=RuntimeError("boom")):
            assert html.plotly_js_script() == ""


class TestTableScript:
    """Tests for the shared sortable/filterable table JavaScript."""

    def test_defines_the_three_table_helpers(self):
        for fn in ("function filterTable", "function sortTable", "function exportCSV"):
            assert fn in html.TABLE_SCRIPT

    def test_is_wrapped_in_a_script_tag_with_single_braces(self):
        assert html.TABLE_SCRIPT.startswith("<script>")
        assert html.TABLE_SCRIPT.rstrip().endswith("</script>")
        # Single braces so it drops into both Jinja and f-string contexts.
        assert "{{" not in html.TABLE_SCRIPT and "}}" not in html.TABLE_SCRIPT
