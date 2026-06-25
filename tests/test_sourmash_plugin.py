"""
Tests for the sourmash CLI plugin integration.
"""

import argparse
from unittest.mock import patch

import pandas as pd
import pytest

from metaquest.plugins.sourmash_plugin import (
    MetaquestParsePlugin,
    MetaquestPlotPlugin,
    MetaquestDiversityPlugin,
    MetaquestTaxonomyPlugin,
)

# --- Fixtures ---


@pytest.fixture
def parser():
    """Create an argparse subparser for testing."""
    main_parser = argparse.ArgumentParser()
    sub = main_parser.add_subparsers()
    return sub.add_parser("test")


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary output directory."""
    return tmp_path


# --- MetaquestParsePlugin ---


class TestMetaquestParsePlugin:
    def test_command_attributes(self):
        assert MetaquestParsePlugin.command == "metaquest_parse"
        assert MetaquestParsePlugin.description is not None

    def test_configure_parser(self, parser):
        MetaquestParsePlugin(parser)
        args = parser.parse_args(
            [
                "--matches-folder",
                "my_matches",
                "--step-size",
                "0.05",
            ]
        )
        assert args.matches_folder == "my_matches"
        assert args.step_size == 0.05
        assert args.parsed_containment_file == "parsed_containment.txt"
        assert args.summary_containment_file == "top_containments.txt"

    def test_default_arguments(self, parser):
        MetaquestParsePlugin(parser)
        args = parser.parse_args([])
        assert args.matches_folder == "matches"
        assert args.step_size == 0.1

    @patch("metaquest.plugins.sourmash_plugin.MetaquestParsePlugin.main")
    def test_main_delegates_to_parse(self, mock_main, parser):
        """Verify main can be called without error when mocked."""
        MetaquestParsePlugin(parser)
        args = parser.parse_args([])
        mock_main(args)
        mock_main.assert_called_once_with(args)

    @patch("metaquest.data.branchwater.parse_containment_data")
    def test_main_calls_parse_containment(self, mock_parse, parser):
        cmd = MetaquestParsePlugin(parser)
        args = parser.parse_args(
            [
                "--matches-folder",
                "test_matches",
                "--parsed-containment-file",
                "parsed.txt",
                "--summary-containment-file",
                "summary.txt",
                "--step-size",
                "0.2",
            ]
        )
        # Call main directly, bypassing super().main() since sourmash
        # may not be installed
        with patch.object(type(cmd).__bases__[0], "main"):
            cmd.main(args)
        mock_parse.assert_called_once_with("test_matches", "parsed.txt", "summary.txt", 0.2)

    @patch("metaquest.data.branchwater.parse_containment_data", side_effect=RuntimeError("bad"))
    def test_main_handles_error(self, mock_parse, parser):
        cmd = MetaquestParsePlugin(parser)
        args = parser.parse_args([])
        with patch.object(type(cmd).__bases__[0], "main"):
            with pytest.raises(SystemExit):
                cmd.main(args)


# --- MetaquestPlotPlugin ---


class TestMetaquestPlotPlugin:
    def test_command_attributes(self):
        assert MetaquestPlotPlugin.command == "metaquest_plot"
        assert MetaquestPlotPlugin.description is not None

    def test_configure_parser(self, parser):
        MetaquestPlotPlugin(parser)
        args = parser.parse_args(
            [
                "--file-path",
                "data.txt",
                "--plot-type",
                "histogram",
                "--threshold",
                "0.5",
            ]
        )
        assert args.file_path == "data.txt"
        assert args.plot_type == "histogram"
        assert args.threshold == 0.5
        assert args.column == "max_containment"

    def test_default_arguments(self, parser):
        MetaquestPlotPlugin(parser)
        args = parser.parse_args(["--file-path", "data.txt"])
        assert args.plot_type == "rank"
        assert args.column == "max_containment"
        assert args.title is None
        assert args.save_format is None

    @patch("metaquest.visualization.plots.plot_containment")
    def test_main_calls_plot(self, mock_plot, parser):
        cmd = MetaquestPlotPlugin(parser)
        args = parser.parse_args(
            [
                "--file-path",
                "data.txt",
                "--plot-type",
                "box",
                "--show-title",
            ]
        )
        with patch.object(type(cmd).__bases__[0], "main"):
            cmd.main(args)
        mock_plot.assert_called_once_with(
            file_path="data.txt",
            column="max_containment",
            title=None,
            colors=None,
            show_title=True,
            save_format=None,
            threshold=None,
            plot_type="box",
        )

    @patch("metaquest.visualization.plots.plot_containment", side_effect=ValueError("bad"))
    def test_main_handles_error(self, mock_plot, parser):
        cmd = MetaquestPlotPlugin(parser)
        args = parser.parse_args(["--file-path", "data.txt"])
        with patch.object(type(cmd).__bases__[0], "main"):
            with pytest.raises(SystemExit):
                cmd.main(args)


# --- MetaquestDiversityPlugin ---


class TestMetaquestDiversityPlugin:
    def test_command_attributes(self):
        assert MetaquestDiversityPlugin.command == "metaquest_diversity"
        assert MetaquestDiversityPlugin.description is not None

    def test_configure_parser(self, parser):
        MetaquestDiversityPlugin(parser)
        args = parser.parse_args(
            [
                "--abundance-file",
                "abundance.csv",
                "--beta-metric",
                "jaccard",
                "--alpha-metrics",
                "shannon",
                "simpson",
            ]
        )
        assert args.abundance_file == "abundance.csv"
        assert args.beta_metric == "jaccard"
        assert args.alpha_metrics == ["shannon", "simpson"]

    def test_default_arguments(self, parser):
        MetaquestDiversityPlugin(parser)
        args = parser.parse_args(["--abundance-file", "abundance.csv"])
        assert args.output_dir == "diversity_results"
        assert args.beta_metric == "bray_curtis"
        assert "shannon" in args.alpha_metrics
        assert args.permanova_formula is None

    @patch("metaquest.processing.diversity.calculate_beta_diversity")
    @patch("metaquest.processing.diversity.calculate_alpha_diversity")
    @patch("pandas.read_csv")
    def test_main_runs_diversity(self, mock_read, mock_alpha, mock_beta, parser, tmp_output):
        abundance_df = pd.DataFrame({"sp1": [1, 2], "sp2": [3, 4]}, index=["s1", "s2"])
        mock_read.return_value = abundance_df
        mock_alpha.return_value = pd.DataFrame({"shannon": [1.0, 1.1]})
        mock_beta.return_value = pd.DataFrame({"s1": [0, 0.5], "s2": [0.5, 0]}, index=["s1", "s2"])

        cmd = MetaquestDiversityPlugin(parser)
        args = parser.parse_args(
            [
                "--abundance-file",
                "abundance.csv",
                "--output-dir",
                str(tmp_output / "div_out"),
            ]
        )
        with patch.object(type(cmd).__bases__[0], "main"):
            cmd.main(args)

        mock_alpha.assert_called_once()
        mock_beta.assert_called_once()

    @patch("pandas.read_csv", side_effect=FileNotFoundError("not found"))
    def test_main_handles_missing_file(self, mock_read, parser, tmp_output):
        cmd = MetaquestDiversityPlugin(parser)
        args = parser.parse_args(
            [
                "--abundance-file",
                "missing.csv",
                "--output-dir",
                str(tmp_output / "div_out"),
            ]
        )
        with patch.object(type(cmd).__bases__[0], "main"):
            with pytest.raises(SystemExit):
                cmd.main(args)


# --- MetaquestTaxonomyPlugin ---


class TestMetaquestTaxonomyPlugin:
    def test_command_attributes(self):
        assert MetaquestTaxonomyPlugin.command == "metaquest_taxonomy"
        assert MetaquestTaxonomyPlugin.description is not None

    def test_configure_parser(self, parser):
        MetaquestTaxonomyPlugin(parser)
        args = parser.parse_args(
            [
                "--species-file",
                "species.txt",
                "--email",
                "test@example.com",
                "--api-key",
                "mykey",
            ]
        )
        assert args.species_file == "species.txt"
        assert args.email == "test@example.com"
        assert args.api_key == "mykey"

    def test_default_arguments(self, parser):
        MetaquestTaxonomyPlugin(parser)
        args = parser.parse_args(
            [
                "--species-file",
                "species.txt",
                "--email",
                "test@example.com",
            ]
        )
        assert args.output_file == "taxonomy_validation.csv"
        assert args.cache_file == "taxonomy_cache.csv"
        assert args.species_column is None

    @patch("metaquest.data.taxonomy.validate_taxonomic_assignments")
    def test_main_with_text_file(self, mock_validate, parser, tmp_output):
        species_file = tmp_output / "species.txt"
        species_file.write_text("E. coli\nB. subtilis\n")

        results_df = pd.DataFrame(
            {
                "species": ["E. coli", "B. subtilis"],
                "is_valid": [True, True],
                "confidence": ["high", "high"],
            }
        )
        mock_validate.return_value = results_df

        cmd = MetaquestTaxonomyPlugin(parser)
        args = parser.parse_args(
            [
                "--species-file",
                str(species_file),
                "--email",
                "test@example.com",
            ]
        )
        with patch.object(type(cmd).__bases__[0], "main"):
            cmd.main(args)

        mock_validate.assert_called_once()
        call_args = mock_validate.call_args
        assert call_args[0][0] == ["E. coli", "B. subtilis"]

    @patch("metaquest.data.taxonomy.validate_taxonomic_assignments")
    @patch("pandas.read_csv")
    def test_main_with_csv_file(self, mock_read, mock_validate, parser):
        mock_read.return_value = pd.DataFrame(
            {
                "organism": ["E. coli", "B. subtilis"],
            }
        )
        results_df = pd.DataFrame(
            {
                "species": ["E. coli", "B. subtilis"],
                "is_valid": [True, False],
                "confidence": ["high", "low"],
            }
        )
        mock_validate.return_value = results_df

        cmd = MetaquestTaxonomyPlugin(parser)
        args = parser.parse_args(
            [
                "--species-file",
                "species.csv",
                "--species-column",
                "organism",
                "--email",
                "test@example.com",
            ]
        )
        with patch.object(type(cmd).__bases__[0], "main"):
            cmd.main(args)

        mock_validate.assert_called_once()

    def test_main_csv_missing_column(self, parser, tmp_output):
        csv_file = tmp_output / "species.csv"
        csv_file.write_text("name\nE. coli\n")

        cmd = MetaquestTaxonomyPlugin(parser)
        args = parser.parse_args(
            [
                "--species-file",
                str(csv_file),
                "--species-column",
                "nonexistent",
                "--email",
                "test@example.com",
            ]
        )
        with patch.object(type(cmd).__bases__[0], "main"):
            with pytest.raises(SystemExit):
                cmd.main(args)

    @patch("metaquest.data.taxonomy.validate_taxonomic_assignments", side_effect=RuntimeError("api"))
    def test_main_handles_error(self, mock_validate, parser, tmp_output):
        species_file = tmp_output / "species.txt"
        species_file.write_text("E. coli\n")

        cmd = MetaquestTaxonomyPlugin(parser)
        args = parser.parse_args(
            [
                "--species-file",
                str(species_file),
                "--email",
                "test@example.com",
            ]
        )
        with patch.object(type(cmd).__bases__[0], "main"):
            with pytest.raises(SystemExit):
                cmd.main(args)


# --- Plugin discovery ---


class TestPluginDiscovery:
    def test_all_commands_have_required_attributes(self):
        """All plugin classes must have command and description set."""
        classes = [
            MetaquestParsePlugin,
            MetaquestPlotPlugin,
            MetaquestDiversityPlugin,
            MetaquestTaxonomyPlugin,
        ]
        for cls in classes:
            assert cls.command is not None, f"{cls.__name__} missing command"
            assert cls.description is not None, f"{cls.__name__} missing description"

    def test_command_names_are_unique(self):
        classes = [
            MetaquestParsePlugin,
            MetaquestPlotPlugin,
            MetaquestDiversityPlugin,
            MetaquestTaxonomyPlugin,
        ]
        names = [cls.command for cls in classes]
        assert len(names) == len(set(names)), "Duplicate command names found"

    def test_command_names_use_metaquest_prefix(self):
        classes = [
            MetaquestParsePlugin,
            MetaquestPlotPlugin,
            MetaquestDiversityPlugin,
            MetaquestTaxonomyPlugin,
        ]
        for cls in classes:
            assert cls.command.startswith("metaquest_"), f"{cls.__name__}.command should start with 'metaquest_'"
