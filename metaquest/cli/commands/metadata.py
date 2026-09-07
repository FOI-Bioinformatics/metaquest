"""
Metadata-related CLI commands.
"""

import argparse
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.defaults import resolve_metadata_table
from metaquest.data.metadata import (
    check_metadata_attributes,
    download_metadata,
    parse_metadata,
    parse_metadata_xml,
)
from metaquest.data.registry import load_registry, nan_to_none, record_metadata, save_registry
from metaquest.processing.counts import count_metadata
from metaquest.store.resolve import resolve_optional_store
from metaquest.visualization.plots import plot_metadata_counts

logger = logging.getLogger(__name__)


def _metadata_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract metadata fields from a parsed metadata dict or pandas row.

    Maps the parsed XML field names (used by parse_metadata_xml) to the registry
    field names (used by record_metadata). This mapping is shared by both
    DownloadMetadataCommand and ParseMetadataCommand to ensure consistency.

    Args:
        row: Mapping with keys like "Run_Total_Spots", "Run_MD5", etc.
            Can be a dict from parse_metadata_xml or a pandas Series.

    Returns:
        Dict with registry field names like "run_total_spots", "run_md5", etc.
    """
    fields: Dict[str, Any] = {}
    for field, column in (
        ("run_size", "Run_Size"),
        ("run_md5", "Run_MD5"),
        ("run_total_spots", "Run_Total_Spots"),
        ("run_total_bases", "Run_Total_Bases"),
        ("assay_type", "Experiment_Library_Strategy"),
        ("organism", "Sample_Scientific_Name"),
        ("collection_date", "collection_date"),
        ("library_layout", "Experiment_Library_Layout"),
        ("platform", "Platform"),
        ("library_strategy", "Experiment_Library_Strategy"),
    ):
        value = row.get(column)
        if value is not None:
            fields[field] = nan_to_none(value)
    return fields


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
        parser.add_argument(
            "--api-key",
            default=os.environ.get("NCBI_API_KEY"),
            help="NCBI API key for a higher rate limit (default: the NCBI_API_KEY environment variable)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=200,
            help="Accessions per NCBI request, from 1 to 500 (default: 200)",
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")

    def _share_with_store(self, args: argparse.Namespace, registry, downloaded: dict) -> None:
        """Copy each fetched XML into the store's ``metadata/`` folder, when a store resolves.

        NCBI's spot count for a run is what makes a store dataset's sidecar verifiable, and the
        store branch of ``download_sra`` already reads that folder. Copying is best effort: a
        store that cannot be written to costs the sharing, never the metadata this project just
        fetched.
        """
        paths = resolve_optional_store(getattr(args, "data_root", None), registry.store.get("root"))
        if paths is None:
            return
        try:
            paths.metadata.mkdir(parents=True, exist_ok=True)
            for xml_path in downloaded.values():
                shutil.copy2(xml_path, paths.metadata / Path(xml_path).name)
        except OSError as e:
            self.logger.warning("Could not copy metadata into the store at %s: %s", paths.metadata, e)
            return
        self.logger.info("Copied %d metadata file(s) into the store at %s", len(downloaded), paths.metadata)

    def execute(self, args: argparse.Namespace) -> int:
        try:
            downloaded = download_metadata(
                email=args.email,
                matches_folder=args.matches_folder,
                metadata_folder=args.metadata_folder,
                threshold=args.threshold,
                dry_run=args.dry_run,
                accessions_file=args.accessions_file,
                api_key=args.api_key,
                batch_size=args.batch_size,
            )
            if not args.dry_run and downloaded:
                registry = load_registry(args.registry)
                for accession, xml_path in downloaded.items():
                    # Parse the metadata XML and record parsed fields right away
                    try:
                        parsed_dict = parse_metadata_xml(xml_path)
                        fields = _metadata_fields(parsed_dict)
                    except (MetaQuestError, ValueError, OSError, ET.ParseError) as e:
                        self.logger.warning(
                            f"Could not parse metadata for {accession}: {e}; recorded the file path only"
                        )
                        fields = {}
                    record_metadata(registry, accession, xml_path, fields)
                save_registry(registry)
                self._share_with_store(args, registry, downloaded)
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
        fields = _metadata_fields(row)
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
