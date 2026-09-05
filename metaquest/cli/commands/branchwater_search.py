"""CLI command that searches the Branchwater index with a genome and writes a Branchwater CSV."""

import argparse
from pathlib import Path

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.branchwater_search import (
    DEFAULT_SERVER,
    load_signature,
    search_index,
    sketch_fasta,
    write_branchwater_csv,
)


class BranchwaterSearchCommand(BaseCommand):
    """Search Branchwater with a genome FASTA or a sourmash signature."""

    @property
    def name(self) -> str:
        return "branchwater_search"

    @property
    def help(self) -> str:
        return "Search the Branchwater index of SRA metagenomes with a genome and write a Branchwater CSV"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--genome-fasta", help="Genome FASTA to sketch (needs the sourmash extra)")
        source.add_argument("--signature", help="Existing sourmash signature file (k=21, scaled=1000)")
        parser.add_argument("--threshold", type=float, default=0.1, help="Minimum containment reported by Branchwater")
        parser.add_argument("--branchwater-folder", default="branchwater", help="Folder for the Branchwater CSV")
        parser.add_argument(
            "--output", default=None, help="Output CSV (default: <branchwater-folder>/<input stem>.csv)"
        )
        parser.add_argument("--server", default=DEFAULT_SERVER, help="Branchwater search API base URL")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if args.genome_fasta:
                signature = sketch_fasta(args.genome_fasta)
                source = Path(args.genome_fasta)
            else:
                signature = load_signature(args.signature)
                source = Path(args.signature)

            output = Path(args.output) if args.output else Path(args.branchwater_folder) / f"{source.stem}.csv"
            matches = search_index(signature, args.threshold, server=args.server)
            write_branchwater_csv(matches, output)

            if matches:
                self.logger.info(
                    "%d metagenome(s) contain %s at >= %.2f; best containment %.4f (%s)",
                    len(matches),
                    source.stem,
                    args.threshold,
                    matches[0][1],
                    matches[0][0],
                )
            else:
                self.logger.warning(
                    "No metagenome reached containment %.2f. The public index has returned nothing for known "
                    "positives before; try a control genome (for example Salmonella LT2, GCF_000006945.2) to "
                    "check that the index is answering.",
                    args.threshold,
                )
            self.logger.info("Next: metaquest use_branchwater --branchwater-folder %s", output.parent)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error searching Branchwater: %s", e)
            return 1
