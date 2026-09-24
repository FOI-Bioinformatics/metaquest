"""CLI command that writes the consolidated per-accession, per-genome results table."""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from metaquest.cli.base import BaseCommand
from metaquest.core.constants import DEFAULT_PARSED_CONTAINMENT_FILE
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.file_io import write_csv
from metaquest.data.registry import load_registry, record_export, registry_transaction
from metaquest.processing.results import results_dataframe, results_rows

EXPORT_NAME = "results_table"


class ResultsTableCommand(BaseCommand):
    """Write one row per screened (accession, genome) pair from the registry and the parsed containment table."""

    @property
    def name(self) -> str:
        return "results_table"

    @property
    def help(self) -> str:
        return "Write the consolidated results table, one row per accession and genome"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output", default="results.tsv", help="Output table (tab-separated)")
        parser.add_argument("--genome-id", default=None, help="Only rows for this genome")
        parser.add_argument(
            "--parsed-containment",
            default=DEFAULT_PARSED_CONTAINMENT_FILE,
            help="Table from parse_containment; unrounded containment values and pairs the registry "
            "did not keep. When missing, only the registry is used",
        )
        parser.add_argument(
            "--min-containment",
            type=float,
            default=0.0,
            help="Only pairs with at least this containment; above 0 this also drops pairs never screened",
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")
        parser.add_argument(
            "--no-record",
            dest="no_record",
            action="store_true",
            help="Write the table but do not record the export in the registry",
        )

    def _load_parsed_table(self, path: str) -> Optional[pd.DataFrame]:
        table_path = Path(path)
        if not table_path.exists():
            self.logger.info("Parsed containment table %s not found; using the registry only", table_path)
            return None
        return pd.read_csv(table_path, sep="\t", index_col=0)

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            parsed_table = self._load_parsed_table(args.parsed_containment)
            rows = results_rows(
                registry,
                parsed_table=parsed_table,
                genome_id=args.genome_id,
                min_containment=args.min_containment,
            )
            output = Path(args.output)
            write_csv(results_dataframe(rows), output, sep="\t", index=False)

            accessions = len({row["accession"] for row in rows})
            genomes = len({row["genome_id"] for row in rows})
            log = self.logger.warning if not rows else self.logger.info
            log(
                "Wrote %d row(s) covering %d accession(s) and %d genome(s) to %s",
                len(rows),
                accessions,
                genomes,
                output,
            )

            if args.no_record:
                self.logger.info("Not recording this export in the registry (--no-record)")
            elif registry.path is None or not registry.path.exists():
                # A reporting command does not create a project registry of its own.
                self.logger.info("no project registry at %s; export not recorded", registry.path)
            else:
                summary = {
                    "rows": len(rows),
                    "accessions": accessions,
                    "genomes": genomes,
                    "genome_id": args.genome_id,
                    "min_containment": args.min_containment,
                    "parsed_containment": str(args.parsed_containment) if parsed_table is not None else None,
                }
                with registry_transaction(args.registry) as locked:
                    record_export(locked, EXPORT_NAME, output, summary)
            self.logger.info("Next: open %s in a spreadsheet, or read it with pandas.read_csv(sep='\\t')", output)
            return 0
        except (MetaQuestError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error writing the results table: %s", e)
            return 1
