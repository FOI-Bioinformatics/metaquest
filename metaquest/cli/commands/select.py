"""CLI command that turns containment results into an accession list for downloads."""

import argparse
from pathlib import Path

from metaquest.cli.base import BaseCommand
from metaquest.core.constants import DEFAULT_CONTAINMENT_THRESHOLD
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.defaults import resolve_metadata_table
from metaquest.processing.selection import select_accessions


class SelectDatasetsCommand(BaseCommand):
    """Write the accessions that meet a containment threshold (and optional metadata filter)."""

    @property
    def name(self) -> str:
        return "select_datasets"

    @property
    def help(self) -> str:
        return "Write an accessions file from parsed containment, for download_sra"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--parsed-containment", default="parsed_containment.txt", help="Table from parse_containment"
        )
        parser.add_argument("--genome-id", default=None, help="Genome column to rank on (default: max_containment)")
        parser.add_argument(
            "--threshold", type=float, default=DEFAULT_CONTAINMENT_THRESHOLD, help="Minimum containment, inclusive"
        )
        parser.add_argument(
            "--metadata-file",
            default=None,
            help="Metadata table for filtering (default: metadata_table.txt, else metadata/branchwater_metadata.txt)",
        )
        parser.add_argument("--metadata-column", default=None, help="Metadata column to filter on")
        parser.add_argument("--metadata-value", default=None, help="Required value in that column")
        parser.add_argument("--output", default="accessions.txt", help="Output file, one accession per line")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            metadata_file = None
            if args.metadata_column:
                metadata_file = resolve_metadata_table(args.metadata_file)

            accessions = select_accessions(
                parsed_containment=args.parsed_containment,
                genome_id=args.genome_id,
                threshold=args.threshold,
                metadata_file=metadata_file,
                metadata_column=args.metadata_column,
                metadata_value=args.metadata_value,
            )

            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(f"{acc}\n" for acc in accessions))
            self.logger.info("Wrote %d accession(s) to %s", len(accessions), output)
            if not accessions:
                self.logger.warning("No accessions met the criteria; %s is empty", output)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error selecting datasets: %s", e)
            return 1
