"""
Metadata-related CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.metadata import (
    check_metadata_attributes,
    download_metadata,
    parse_metadata,
)
from metaquest.processing.counts import count_metadata
from metaquest.visualization.plots import plot_metadata_counts


class DownloadMetadataCommand(BaseCommand):
    """Command for downloading metadata for SRA accessions."""

    @property
    def name(self) -> str:
        return "download_metadata"

    @property
    def help(self) -> str:
        return "Download metadata for SRA accessions"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--email", required=True, help="Your email address for NCBI API access")
        parser.add_argument("--matches-folder", default="matches", help="Folder containing match files")
        parser.add_argument(
            "--metadata-folder",
            default="metadata",
            help="Folder to save downloaded metadata",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.0,
            help="Threshold for containment values",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Calculate number of accessions without downloading",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            download_metadata(
                email=args.email,
                matches_folder=args.matches_folder,
                metadata_folder=args.metadata_folder,
                threshold=args.threshold,
                dry_run=args.dry_run,
            )
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error downloading metadata: {e}")
            return 1


class ParseMetadataCommand(BaseCommand):
    """Command for parsing downloaded metadata files."""

    @property
    def name(self) -> str:
        return "parse_metadata"

    @property
    def help(self) -> str:
        return "Parse downloaded metadata files"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--metadata-folder",
            default="metadata",
            help="Folder containing metadata files",
        )
        parser.add_argument(
            "--metadata-table-file",
            default="metadata_table.txt",
            help="File where the parsed metadata will be stored",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            parse_metadata(args.metadata_folder, args.metadata_table_file)
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error parsing metadata: {e}")
            return 1


class CheckMetadataAttributesCommand(BaseCommand):
    """Command for counting how often each sample-attribute column is populated."""

    @property
    def name(self) -> str:
        return "check_metadata_attributes"

    @property
    def help(self) -> str:
        return "Count how often each metadata attribute is populated"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--file-path",
            default="metadata_table.txt",
            help="Path to the parsed metadata table",
        )
        parser.add_argument(
            "--output-file",
            default="metadata_attribute_counts.txt",
            help="Path to save the attribute counts",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            check_metadata_attributes(args.file_path, args.output_file)
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error checking metadata attributes: {e}")
            return 1


class CountMetadataCommand(BaseCommand):
    """Command for counting metadata values by genome."""

    @property
    def name(self) -> str:
        return "count_metadata"

    @property
    def help(self) -> str:
        return "Count metadata values by genome"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--summary-file",
            default="parsed_containment.txt",
            help="Path to the summary file",
        )
        parser.add_argument(
            "--metadata-file",
            default="metadata_table.txt",
            help="Path to the metadata file",
        )
        parser.add_argument(
            "--metadata-column",
            required=True,
            help="Name of the column in the metadata file",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Threshold for containment values",
        )
        parser.add_argument(
            "--output-file",
            default="metadata_counts.txt",
            help="Path to the output file",
        )
        parser.add_argument("--stat-file", default=None, help="Path to the statistics file")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            count_metadata(
                summary_file=args.summary_file,
                metadata_file=args.metadata_file,
                metadata_column=args.metadata_column,
                threshold=args.threshold,
                output_file=args.output_file,
                stat_file=args.stat_file,
            )
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error counting metadata: {e}")
            return 1


class PlotMetadataCountsCommand(BaseCommand):
    """Command for plotting metadata counts."""

    @property
    def name(self) -> str:
        return "plot_metadata_counts"

    @property
    def help(self) -> str:
        return "Plot metadata counts"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--file-path", required=True, help="Path to the metadata counts file")
        parser.add_argument("--title", default=None, help="Title for the plot")
        parser.add_argument(
            "--plot-type",
            default="bar",
            choices=["bar", "pie", "radar"],
            help="Type of plot to generate",
        )
        parser.add_argument("--colors", default=None, help="Colors or colormap name")
        parser.add_argument("--show-title", action="store_true", help="Whether to display the title")
        parser.add_argument(
            "--save-format",
            default=None,
            choices=["png", "jpg", "pdf", "svg"],
            help="Format to save the figure",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            plot_metadata_counts(
                file_path=args.file_path,
                title=args.title,
                plot_type=args.plot_type,
                colors=args.colors,
                show_title=args.show_title,
                save_format=args.save_format,
            )
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error plotting metadata counts: {e}")
            return 1
