"""CLI command that turns containment results into an accession list for downloads."""

import argparse
from pathlib import Path

from metaquest.cli.base import BaseCommand
from metaquest.core.constants import DEFAULT_CONTAINMENT_THRESHOLD, DEFAULT_TOP_N
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.defaults import resolve_metadata_table
from metaquest.data.registry import load_registry, query, record_selection, save_registry
from metaquest.processing.selection import select_accessions_ranked


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
        genome_group = parser.add_mutually_exclusive_group()
        genome_group.add_argument(
            "--genome-id", default=None, help="Genome column to rank on (default: max_containment)"
        )
        genome_group.add_argument(
            "--genome-ids", nargs="+", default=None, help="Multiple genome columns to rank on together"
        )
        parser.add_argument(
            "--require",
            choices=["any", "all"],
            default="any",
            help="With --genome-ids, require any or all columns to meet the threshold (default: any)",
        )
        parser.add_argument(
            "--threshold", type=float, default=DEFAULT_CONTAINMENT_THRESHOLD, help="Minimum containment, inclusive"
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=None,
            help=f"Keep only the top N accessions after filtering (suggested default: {DEFAULT_TOP_N})",
        )
        parser.add_argument(
            "--metadata-file",
            default=None,
            help="Metadata table for filtering (default: metadata_table.txt, else metadata/branchwater_metadata.txt)",
        )
        parser.add_argument("--metadata-column", default=None, help="Metadata column to filter on")
        parser.add_argument("--metadata-value", default=None, help="Required value in that column")
        parser.add_argument("--output", default="accessions.txt", help="Output file, one accession per line")
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")
        parser.add_argument(
            "--skip-excluded",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Drop accessions already marked excluded in the registry (default: on)",
        )
        parser.add_argument(
            "--skip-downloaded",
            action="store_true",
            help="Drop accessions already marked downloaded in the registry",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            metadata_file = None
            if args.metadata_column:
                metadata_file = resolve_metadata_table(args.metadata_file)

            registry = load_registry(args.registry)
            exclude = set(query(registry, "excluded")) if args.skip_excluded else set()
            excluded_count = len(exclude)
            downloaded = set(query(registry, "downloaded")) if args.skip_downloaded else set()
            downloaded_count = len(downloaded)
            exclude |= downloaded

            ranked = select_accessions_ranked(
                parsed_containment=args.parsed_containment,
                genome_id=args.genome_id,
                threshold=args.threshold,
                metadata_file=metadata_file,
                metadata_column=args.metadata_column,
                metadata_value=args.metadata_value,
                top_n=args.top_n,
                exclude=exclude,
                genome_ids=args.genome_ids,
                require=args.require,
            )
            accessions = [accession for accession, _, _ in ranked]

            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(f"{acc}\n" for acc in accessions))
            self.logger.info("Wrote %d accession(s) to %s", len(accessions), output)
            if not accessions:
                self.logger.warning("No accessions met the criteria; %s is empty", output)
            self.logger.info(
                "%d already downloaded, %d excluded, %d selected",
                downloaded_count,
                excluded_count,
                len(accessions),
            )

            ranked_records = [
                {"accession": accession, "rank": i + 1, "column": column, "value": value}
                for i, (accession, column, value) in enumerate(ranked)
            ]
            record_selection(
                registry,
                accessions,
                {
                    "column": args.genome_id or "max_containment",
                    "threshold": args.threshold,
                    "metadata_column": args.metadata_column,
                    "metadata_value": args.metadata_value,
                    "table": str(args.parsed_containment),
                    "top_n": args.top_n,
                    "require": args.require,
                    "genome_ids": args.genome_ids,
                    "skip_excluded": args.skip_excluded,
                    "skip_downloaded": args.skip_downloaded,
                },
                output,
                ranked=ranked_records,
            )
            save_registry(registry)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error selecting datasets: %s", e)
            return 1
