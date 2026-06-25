"""
Test CLI advanced analysis commands functionality.

Tests for diversity analysis, interactive plotting, and taxonomy commands.
"""

import argparse
from unittest.mock import patch, mock_open
import pandas as pd
import pytest

from metaquest.cli.commands.advanced_analysis import (
    DiversityAnalysisCommand,
    InteractivePlotCommand,
    TaxonomyValidationCommand,
    TaxonomicSummaryCommand,
)


class TestDiversityAnalysisCommand:
    """Test DiversityAnalysisCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = DiversityAnalysisCommand()
        assert command.name == "diversity_analysis"
        assert "diversity" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = DiversityAnalysisCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test required arguments
        args = parser.parse_args(["--abundance-file", "abundance.csv"])
        assert args.abundance_file == "abundance.csv"
        assert args.output_dir == "diversity_results"
        assert args.alpha_metrics == ["shannon", "simpson", "chao1", "observed_species"]
        assert args.beta_metric == "bray_curtis"

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = DiversityAnalysisCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--abundance-file",
                "abundance.csv",
                "--metadata-file",
                "metadata.csv",
                "--output-dir",
                "custom_output",
                "--alpha-metrics",
                "shannon",
                "simpson",
                "--beta-metric",
                "jaccard",
                "--permanova-formula",
                "treatment + site",
            ]
        )

        assert args.metadata_file == "metadata.csv"
        assert args.output_dir == "custom_output"
        assert args.alpha_metrics == ["shannon", "simpson"]
        assert args.beta_metric == "jaccard"
        assert args.permanova_formula == "treatment + site"

    @patch("metaquest.cli.commands.advanced_analysis.calculate_alpha_diversity")
    @patch("metaquest.cli.commands.advanced_analysis.calculate_beta_diversity")
    @patch("metaquest.cli.commands.advanced_analysis.Path.mkdir")
    @patch("pandas.read_csv")
    def test_execute_success_basic(self, mock_read_csv, mock_mkdir, mock_beta_div, mock_alpha_div):
        """Test successful execution without PERMANOVA."""
        # Setup mocks
        mock_abundance_df = pd.DataFrame({"sample1": [1, 2], "sample2": [3, 4]})
        mock_read_csv.return_value = mock_abundance_df

        mock_alpha_result = pd.DataFrame({"shannon": [1.5, 2.0]})
        mock_alpha_div.return_value = mock_alpha_result

        mock_beta_result = pd.DataFrame([[0, 0.5], [0.5, 0]])
        mock_beta_div.return_value = mock_beta_result

        command = DiversityAnalysisCommand()
        args = argparse.Namespace(
            abundance_file="abundance.csv",
            metadata_file=None,
            output_dir="test_output",
            alpha_metrics=["shannon"],
            beta_metric="bray_curtis",
            permanova_formula=None,
        )

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            result = command.execute(args)

        assert result == 0
        mock_mkdir.assert_called_once()
        mock_alpha_div.assert_called_once_with(mock_abundance_df, ["shannon"])
        mock_beta_div.assert_called_once_with(mock_abundance_df, "bray_curtis")
        assert mock_to_csv.call_count == 2  # Alpha and beta results

    @patch("metaquest.cli.commands.advanced_analysis.calculate_alpha_diversity")
    @patch("metaquest.cli.commands.advanced_analysis.calculate_beta_diversity")
    @patch("metaquest.cli.commands.advanced_analysis.perform_permanova")
    @patch("metaquest.cli.commands.advanced_analysis.Path.mkdir")
    @patch("pandas.read_csv")
    def test_execute_success_with_permanova(
        self, mock_read_csv, mock_mkdir, mock_permanova, mock_beta_div, mock_alpha_div
    ):
        """Test successful execution with PERMANOVA."""
        # Setup mocks
        mock_abundance_df = pd.DataFrame({"sample1": [1, 2], "sample2": [3, 4]})
        mock_metadata_df = pd.DataFrame({"treatment": ["A", "B"]})
        mock_read_csv.side_effect = [mock_abundance_df, mock_metadata_df]

        mock_alpha_result = pd.DataFrame({"shannon": [1.5, 2.0]})
        mock_alpha_div.return_value = mock_alpha_result

        mock_beta_result = pd.DataFrame([[0, 0.5], [0.5, 0]])
        mock_beta_div.return_value = mock_beta_result

        mock_permanova_result = {"treatment": {"F_statistic": 2.5, "p_value": 0.03, "significant": True}}
        mock_permanova.return_value = mock_permanova_result

        command = DiversityAnalysisCommand()
        args = argparse.Namespace(
            abundance_file="abundance.csv",
            metadata_file="metadata.csv",
            output_dir="test_output",
            alpha_metrics=["shannon"],
            beta_metric="bray_curtis",
            permanova_formula="treatment",
        )

        with patch("pandas.DataFrame.to_csv"), patch("builtins.open", mock_open()) as mock_file:
            result = command.execute(args)

        assert result == 0
        mock_permanova.assert_called_once()
        mock_file.assert_called()

    @patch("pandas.read_csv")
    def test_execute_failure(self, mock_read_csv):
        """Test execution failure."""
        mock_read_csv.side_effect = Exception("File not found")

        command = DiversityAnalysisCommand()
        args = argparse.Namespace(
            abundance_file="nonexistent.csv",
            metadata_file=None,
            output_dir="test_output",
            alpha_metrics=["shannon"],
            beta_metric="bray_curtis",
            permanova_formula=None,
        )

        result = command.execute(args)
        assert result == 1


class TestInteractivePlotCommand:
    """Test InteractivePlotCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = InteractivePlotCommand()
        assert command.name == "interactive_plot"
        assert "interactive" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = InteractivePlotCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test required arguments
        args = parser.parse_args(["--data-file", "data.csv", "--plot-type", "pca"])
        assert args.data_file == "data.csv"
        assert args.plot_type == "pca"
        assert args.no_show is False

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = InteractivePlotCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--data-file",
                "data.csv",
                "--metadata-file",
                "metadata.csv",
                "--plot-type",
                "heatmap",
                "--color-by",
                "treatment",
                "--size-by",
                "weight",
                "--output-file",
                "plot.html",
                "--title",
                "My Plot",
                "--no-show",
            ]
        )

        assert args.metadata_file == "metadata.csv"
        assert args.color_by == "treatment"
        assert args.size_by == "weight"
        assert args.output_file == "plot.html"
        assert args.title == "My Plot"
        assert args.no_show is True

    @patch("pandas.read_csv")
    @patch("metaquest.cli.commands.advanced_analysis.create_interactive_pca")
    def test_execute_pca_plot(self, mock_create_pca, mock_read_csv):
        """Test PCA plot creation."""
        mock_data_df = pd.DataFrame({"gene1": [1, 2], "gene2": [3, 4]})
        mock_metadata_df = pd.DataFrame({"treatment": ["A", "B"]})
        mock_read_csv.side_effect = [mock_data_df, mock_metadata_df]

        command = InteractivePlotCommand()
        args = argparse.Namespace(
            data_file="data.csv",
            metadata_file="metadata.csv",
            plot_type="pca",
            color_by="treatment",
            size_by=None,
            output_file="pca.html",
            title="PCA Test",
            no_show=True,
        )

        result = command.execute(args)

        assert result == 0
        mock_create_pca.assert_called_once_with(
            mock_data_df,
            metadata=mock_metadata_df,
            color_by="treatment",
            size_by=None,
            title="PCA Test",
            output_file="pca.html",
            show_plot=False,
        )

    @patch("pandas.read_csv")
    @patch("metaquest.cli.commands.advanced_analysis.create_interactive_heatmap")
    def test_execute_heatmap_plot(self, mock_create_heatmap, mock_read_csv):
        """Test heatmap plot creation."""
        mock_data_df = pd.DataFrame({"gene1": [1, 2], "gene2": [3, 4]})
        mock_read_csv.return_value = mock_data_df

        command = InteractivePlotCommand()
        args = argparse.Namespace(
            data_file="data.csv",
            metadata_file=None,
            plot_type="heatmap",
            color_by=None,
            size_by=None,
            output_file=None,
            title=None,
            no_show=False,
        )

        result = command.execute(args)

        assert result == 0
        mock_create_heatmap.assert_called_once_with(
            mock_data_df, sample_metadata=None, title="Interactive Heatmap", output_file=None, show_plot=True
        )

    @patch("pandas.read_csv")
    @patch("metaquest.cli.commands.advanced_analysis.calculate_alpha_diversity")
    @patch("metaquest.cli.commands.advanced_analysis.create_diversity_comparison_plot")
    def test_execute_diversity_plot(self, mock_create_diversity, mock_calc_alpha, mock_read_csv):
        """Test diversity plot creation."""
        mock_data_df = pd.DataFrame({"gene1": [1, 2], "gene2": [3, 4]})
        mock_metadata_df = pd.DataFrame({"treatment": ["A", "B"]})
        mock_alpha_div = pd.DataFrame({"shannon": [1.5, 2.0]})

        mock_read_csv.side_effect = [mock_data_df, mock_metadata_df]
        mock_calc_alpha.return_value = mock_alpha_div

        command = InteractivePlotCommand()
        args = argparse.Namespace(
            data_file="data.csv",
            metadata_file="metadata.csv",
            plot_type="diversity",
            color_by="treatment",
            size_by=None,
            output_file=None,
            title=None,
            no_show=False,
        )

        result = command.execute(args)

        assert result == 0
        mock_calc_alpha.assert_called_once_with(mock_data_df)
        mock_create_diversity.assert_called_once_with(
            mock_alpha_div,
            mock_metadata_df,
            group_by="treatment",
            title="Diversity Comparison",
            output_file=None,
            show_plot=True,
        )

    @patch("pandas.read_csv")
    def test_execute_diversity_plot_missing_color_by(self, mock_read_csv):
        """Test diversity plot with missing color_by parameter."""
        mock_data_df = pd.DataFrame({"gene1": [1, 2], "gene2": [3, 4]})
        mock_read_csv.return_value = mock_data_df

        command = InteractivePlotCommand()
        args = argparse.Namespace(
            data_file="data.csv",
            metadata_file=None,
            plot_type="diversity",
            color_by=None,
            size_by=None,
            output_file=None,
            title=None,
            no_show=False,
        )

        result = command.execute(args)
        assert result == 1

    @patch("pandas.read_csv")
    def test_execute_unsupported_plot_type(self, mock_read_csv):
        """Test execution with unsupported plot type."""
        mock_data_df = pd.DataFrame({"gene1": [1, 2], "gene2": [3, 4]})
        mock_read_csv.return_value = mock_data_df

        command = InteractivePlotCommand()
        args = argparse.Namespace(
            data_file="data.csv",
            metadata_file=None,
            plot_type="tsne",  # Not yet implemented
            color_by=None,
            size_by=None,
            output_file=None,
            title=None,
            no_show=False,
        )

        result = command.execute(args)
        assert result == 1


class TestTaxonomyValidationCommand:
    """Test TaxonomyValidationCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = TaxonomyValidationCommand()
        assert command.name == "validate_taxonomy"
        assert "taxonomy" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = TaxonomyValidationCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test required arguments
        args = parser.parse_args(["--species-file", "species.txt", "--email", "test@example.com"])
        assert args.species_file == "species.txt"
        assert args.email == "test@example.com"
        assert args.output_file == "taxonomy_validation.csv"
        assert args.cache_file == "taxonomy_cache.csv"

    @patch("metaquest.cli.commands.advanced_analysis.validate_taxonomic_assignments")
    def test_execute_success_text_file(self, mock_validate):
        """Test successful execution with text file."""
        mock_results = pd.DataFrame(
            {
                "species": ["Escherichia coli", "Salmonella enterica"],
                "is_valid": [True, True],
                "confidence": ["high", "high"],
            }
        )
        mock_validate.return_value = mock_results

        command = TaxonomyValidationCommand()
        args = argparse.Namespace(
            species_file="species.txt",
            species_column=None,
            email="test@example.com",
            api_key=None,
            output_file="validation.csv",
            cache_file="cache.csv",
        )

        with patch("builtins.open", mock_open(read_data="Escherichia coli\nSalmonella enterica\n")):
            with patch("builtins.print") as mock_print:
                result = command.execute(args)

        assert result == 0
        mock_validate.assert_called_once()
        mock_print.assert_called()

    @patch("pandas.read_csv")
    @patch("metaquest.cli.commands.advanced_analysis.validate_taxonomic_assignments")
    def test_execute_success_csv_file_with_column(self, mock_validate, mock_read_csv):
        """Test successful execution with CSV file and specified column."""
        mock_df = pd.DataFrame({"species_name": ["Escherichia coli", "Salmonella enterica"], "abundance": [100, 50]})
        mock_read_csv.return_value = mock_df

        mock_results = pd.DataFrame(
            {
                "species": ["Escherichia coli", "Salmonella enterica"],
                "is_valid": [True, False],
                "confidence": ["high", "low"],
            }
        )
        mock_validate.return_value = mock_results

        command = TaxonomyValidationCommand()
        args = argparse.Namespace(
            species_file="species.csv",
            species_column="species_name",
            email="test@example.com",
            api_key="api123",
            output_file="validation.csv",
            cache_file="cache.csv",
        )

        with patch("builtins.print"):
            result = command.execute(args)

        assert result == 0
        mock_validate.assert_called_once_with(
            ["Escherichia coli", "Salmonella enterica"],
            email="test@example.com",
            api_key="api123",
            output_file="validation.csv",
            cache_file="cache.csv",
        )

    @patch("pandas.read_csv")
    def test_execute_csv_missing_column(self, mock_read_csv):
        """Test execution with CSV file missing specified column."""
        mock_df = pd.DataFrame({"other_column": ["data1", "data2"]})
        mock_read_csv.return_value = mock_df

        command = TaxonomyValidationCommand()
        args = argparse.Namespace(
            species_file="species.csv",
            species_column="species_name",
            email="test@example.com",
            api_key=None,
            output_file="validation.csv",
            cache_file="cache.csv",
        )

        result = command.execute(args)
        assert result == 1

    @patch("pandas.read_csv")
    @patch("metaquest.cli.commands.advanced_analysis.validate_taxonomic_assignments")
    def test_execute_success_csv_first_column(self, mock_validate, mock_read_csv):
        """Test successful execution with CSV file using first column."""
        mock_df = pd.DataFrame(
            {"first_col": ["Escherichia coli", "Salmonella enterica"], "other_col": ["data1", "data2"]}
        )
        mock_read_csv.return_value = mock_df

        mock_results = pd.DataFrame(
            {
                "species": ["Escherichia coli", "Salmonella enterica"],
                "is_valid": [True, True],
                "confidence": ["high", "high"],
            }
        )
        mock_validate.return_value = mock_results

        command = TaxonomyValidationCommand()
        args = argparse.Namespace(
            species_file="species.csv",
            species_column=None,  # Will use first column
            email="test@example.com",
            api_key=None,
            output_file="validation.csv",
            cache_file="cache.csv",
        )

        with patch("builtins.print"):
            result = command.execute(args)

        assert result == 0


class TestTaxonomicSummaryCommand:
    """Test TaxonomicSummaryCommand."""

    def test_command_properties(self):
        """Test command name and help."""
        command = TaxonomicSummaryCommand()
        assert command.name == "taxonomic_summary"
        assert "taxonomic" in command.help.lower()

    def test_configure_parser(self):
        """Test parser configuration."""
        command = TaxonomicSummaryCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test required arguments
        args = parser.parse_args(["--abundance-file", "abundance.csv", "--taxonomy-file", "taxonomy.csv"])
        assert args.abundance_file == "abundance.csv"
        assert args.taxonomy_file == "taxonomy.csv"
        assert args.output_dir == "taxonomic_summaries"
        assert args.levels == ["phylum", "class", "order", "family", "genus"]
        assert args.min_abundance == 0.001

    def test_configure_parser_with_options(self):
        """Test parser with optional arguments."""
        command = TaxonomicSummaryCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args(
            [
                "--abundance-file",
                "abundance.csv",
                "--taxonomy-file",
                "taxonomy.csv",
                "--output-dir",
                "custom_output",
                "--levels",
                "phylum",
                "genus",
                "--min-abundance",
                "0.01",
            ]
        )

        assert args.output_dir == "custom_output"
        assert args.levels == ["phylum", "genus"]
        assert args.min_abundance == 0.01

    @patch("pandas.read_csv")
    @patch("metaquest.cli.commands.advanced_analysis.analyze_taxonomic_composition")
    def test_execute_success(self, mock_analyze, mock_read_csv):
        """Test successful execution."""
        mock_abundance_df = pd.DataFrame({"sample1": [10, 20], "sample2": [15, 25]})
        mock_taxonomy_df = pd.DataFrame({"species": ["Species1", "Species2"], "phylum": ["Phylum1", "Phylum2"]})
        mock_read_csv.side_effect = [mock_abundance_df, mock_taxonomy_df]

        mock_summaries = {
            "phylum": pd.DataFrame({"Phylum1": [10, 15], "Phylum2": [20, 25]}),
            "genus": pd.DataFrame({"Genus1": [5, 8], "Genus2": [25, 32]}),
        }
        mock_analyze.return_value = mock_summaries

        command = TaxonomicSummaryCommand()
        args = argparse.Namespace(
            abundance_file="abundance.csv",
            taxonomy_file="taxonomy.csv",
            output_dir="summaries",
            levels=["phylum", "genus"],
            min_abundance=0.001,
        )

        with patch("builtins.print") as mock_print:
            result = command.execute(args)

        assert result == 0
        mock_analyze.assert_called_once_with(
            mock_abundance_df, mock_taxonomy_df, levels=["phylum", "genus"], output_dir="summaries"
        )
        mock_print.assert_called()

    @patch("pandas.read_csv")
    def test_execute_failure(self, mock_read_csv):
        """Test execution failure."""
        mock_read_csv.side_effect = Exception("File not found")

        command = TaxonomicSummaryCommand()
        args = argparse.Namespace(
            abundance_file="nonexistent.csv",
            taxonomy_file="taxonomy.csv",
            output_dir="summaries",
            levels=["phylum"],
            min_abundance=0.001,
        )

        result = command.execute(args)
        assert result == 1


class TestAdvancedAnalysisIntegration:
    """Integration tests for advanced analysis commands."""

    def test_all_commands_inherit_base_command(self):
        """Test that all command classes inherit from BaseCommand."""
        from metaquest.cli.base import BaseCommand

        commands = [
            DiversityAnalysisCommand(),
            InteractivePlotCommand(),
            TaxonomyValidationCommand(),
            TaxonomicSummaryCommand(),
        ]

        for command in commands:
            assert isinstance(command, BaseCommand)
            assert hasattr(command, "name")
            assert hasattr(command, "help")
            assert hasattr(command, "configure_parser")
            assert hasattr(command, "execute")

    def test_command_names_unique(self):
        """Test that all command names are unique."""
        commands = [
            DiversityAnalysisCommand(),
            InteractivePlotCommand(),
            TaxonomyValidationCommand(),
            TaxonomicSummaryCommand(),
        ]

        names = [cmd.name for cmd in commands]
        assert len(names) == len(set(names))

    def test_all_parsers_configured(self):
        """Test that all commands can configure parsers without error."""
        commands = [
            DiversityAnalysisCommand(),
            InteractivePlotCommand(),
            TaxonomyValidationCommand(),
            TaxonomicSummaryCommand(),
        ]

        for command in commands:
            parser = argparse.ArgumentParser()
            # Should not raise exception
            command.configure_parser(parser)

    @patch("metaquest.cli.commands.advanced_analysis.logger")
    def test_logging_behavior(self, mock_logger):
        """Test that commands use logging appropriately."""
        command = DiversityAnalysisCommand()

        # Mock the pandas import and other dependencies
        with patch("pandas.read_csv", side_effect=Exception("Test error")):
            args = argparse.Namespace(
                abundance_file="test.csv",
                metadata_file=None,
                output_dir="output",
                alpha_metrics=["shannon"],
                beta_metric="bray_curtis",
                permanova_formula=None,
            )

            result = command.execute(args)

            assert result == 1
            mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
