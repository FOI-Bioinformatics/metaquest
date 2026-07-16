"""Tests for shared HTML report assets in metaquest.utils.html."""

from unittest.mock import patch

from metaquest.utils import html


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
