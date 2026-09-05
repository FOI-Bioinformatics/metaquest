"""
Metadata-related CLI commands.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.defaults import resolve_metadata_table
from metaquest.data.metadata import (
    check_metadata_attributes,
    download_metadata,
    parse_metadata,
)
from metaquest.data.registry import load_registry, nan_to_none, record_metadata, save_registry
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

    @property
    def group(self) -> str:
        return "Metadata"

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
            help="Threshold for containment values (inclusive)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Calculate number of accessions without downloading",
        )
        parser.add_argument(
            "--accessions-file",
            default=None,
            help="File of accessions to fetch metadata for; replaces the matches folder scan",
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            downloaded = download_metadata(
                email=args.email,
                matches_folder=args.matches_folder,
                metadata_folder=args.metadata_folder,
                threshold=args.threshold,
                dry_run=args.dry_run,
                accessions_file=args.accessions_file,
            )
            if not args.dry_run and downloaded:
                registry = load_registry(args.registry)
                for accession, xml_path in downloaded.items():
                    record_metadata(registry, accession, xml_path, {})
                save_registry(registry)
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

    @property
    def group(self) -> str:
        return "Metadata"

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
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")

    def _record_row(self, registry, metadata_folder: Path, row: "pd.Series") -> None:
        accession = row.get("Run_ID")
        if accession is None or pd.isna(accession):
            return
        fields: Dict[str, Any] = {}
        for field, column in (
            ("run_size", "Run_Size"),
            ("run_md5", "Run_MD5"),
            ("assay_type", "Experiment_Library_Strategy"),
            ("organism", "Sample_Scientific_Name"),
            ("collection_date", "collection_date"),
        ):
            if column in row.index:
                fields[field] = nan_to_none(row.get(column))
        record_metadata(registry, str(accession), metadata_folder / f"{accession}_metadata.xml", fields)

    def execute(self, args: argparse.Namespace) -> int:
        try:
            df = parse_metadata(args.metadata_folder, args.metadata_table_file)
            registry = load_registry(args.registry)
            metadata_folder = Path(args.metadata_folder)
            for _, row in df.iterrows():
                self._record_row(registry, metadata_folder, row)
            save_registry(registry)
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

    @property
    def group(self) -> str:
        return "Metadata"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--file-path",
            default=None,
            help="Parsed metadata table (default: metadata_table.txt, else metadata/branchwater_metadata.txt)",
        )
        parser.add_argument(
            "--output-file",
            default="metadata_attribute_counts.txt",
            help="Path to save the attribute counts",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            check_metadata_attributes(str(resolve_metadata_table(args.file_path)), args.output_file)
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

    @property
    def group(self) -> str:
        return "Metadata"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--summary-file",
            default="parsed_containment.txt",
            help="Summary file",
        )
        parser.add_argument(
            "--metadata-file",
            default=None,
            help="Parsed metadata table (default: metadata_table.txt, else metadata/branchwater_metadata.txt)",
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
            help="Threshold for containment values (inclusive)",
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
                metadata_file=str(resolve_metadata_table(args.metadata_file)),
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

    @property
    def group(self) -> str:
        return "Metadata"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--file-path", required=True, help="Counts table from count_metadata (or its _stats file)")
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
