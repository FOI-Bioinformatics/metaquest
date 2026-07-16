"""Tests for shared HTML report assets in metaquest.utils.html."""

from unittest.mock import patch

from metaquest.utils import html


class TestPlotlyCdnScript:
    """Tests for the pinned Plotly CDN script tag."""

    def test_uses_installed_plotlyjs_version(self):
        """The tag pins the version reported by the installed plotly."""
        with patch("plotly.offline.get_plotlyjs_version", return_value="9.9.9"):
            tag = html.plotly_cdn_script()
        assert tag == '<script src="https://cdn.plot.ly/plotly-9.9.9.min.js"></script>'

    def test_never_emits_the_frozen_latest_alias(self):
        """The frozen 'plotly-latest' alias must never be produced."""
        assert "plotly-latest" not in html.plotly_cdn_script()

    def test_falls_back_when_version_lookup_fails(self):
        """A lookup failure yields the conservative fallback version, not a crash."""
        with patch("plotly.offline.get_plotlyjs_version", side_effect=RuntimeError("boom")):
            tag = html.plotly_cdn_script()
        assert html._FALLBACK_PLOTLYJS_VERSION in tag
        assert tag.startswith("<script") and tag.endswith("</script>")


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
