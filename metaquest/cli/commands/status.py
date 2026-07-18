"""Local inventory / status CLI command.

Reports what MetaQuest has already downloaded locally (SRA reads, NCBI metadata,
genome assemblies) so a user can see what is available without re-downloading,
and optionally reconciles that against a wanted list of accessions.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.sra import accession_has_fastq


class StatusCommand(BaseCommand):
    """Command to report locally available data and reconcile it against a wanted list."""

    @property
    def name(self) -> str:
        return "status"

    @property
    def help(self) -> str:
        return "Report which SRA reads, metadata, and genomes are already available locally"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument("--metadata-folder", default="metadata", help="Folder holding NCBI metadata XML")
        parser.add_argument("--genomes-folder", default="genomes", help="Folder holding genome FASTA files")
        parser.add_argument(
            "--accessions-file",
            help="Optional file of SRA accessions (one per line) to reconcile against local FASTQ/metadata",
        )
        parser.add_argument(
            "--parsed-containment",
            help="Optional parsed containment table; its sample accessions are the wanted list",
        )
        parser.add_argument("--list-missing", action="store_true", help="Also print the accessions that are missing")
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON")

    def _wanted_accessions(self, args: argparse.Namespace) -> List[str]:
        """Load the wanted accession list from a file and/or a parsed containment table."""
        wanted: List[str] = []
        if args.accessions_file:
            path = Path(args.accessions_file)
            if not path.exists():
                raise MetaQuestError(f"Accessions file not found: {path}")
            wanted += [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        if args.parsed_containment:
            import pandas as pd

            path = Path(args.parsed_containment)
            if not path.exists():
                raise MetaQuestError(f"Parsed containment file not found: {path}")
            wanted += [str(i) for i in pd.read_csv(path, sep="\t", index_col=0).index]
        # de-duplicate, preserve order
        return list(dict.fromkeys(wanted))

    @staticmethod
    def _reconcile(wanted: List[str], present_fn) -> Tuple[List[str], List[str]]:
        """Split a wanted list into (present, missing) using a predicate."""
        present = [a for a in wanted if present_fn(a)]
        missing = [a for a in wanted if a not in present]
        return present, missing

    def _build_report(self, args: argparse.Namespace) -> dict:
        fastq_dir = Path(args.fastq_folder)
        meta_dir = Path(args.metadata_folder)
        genomes_dir = Path(args.genomes_folder)

        if fastq_dir.is_dir():
            on_disk_fastq = sorted(d.name for d in fastq_dir.iterdir() if accession_has_fastq(d))
        else:
            on_disk_fastq = []
        on_disk_meta = sorted(p.name[: -len("_metadata.xml")] for p in meta_dir.glob("*_metadata.xml"))
        genome_globs = ("*.fna", "*.fna.gz", "*.fasta", "*.fasta.gz", "*.fa", "*.fa.gz")
        on_disk_genomes = sorted({p.name for g in genome_globs for p in genomes_dir.glob(g)})

        report: dict = {
            "on_disk": {
                "fastq_accessions": len(on_disk_fastq),
                "metadata_xml": len(on_disk_meta),
                "genome_fasta": len(on_disk_genomes),
            }
        }

        wanted = self._wanted_accessions(args)
        if wanted:
            fastq_present, fastq_missing = self._reconcile(wanted, lambda a: accession_has_fastq(fastq_dir / a))
            meta_present, meta_missing = self._reconcile(wanted, lambda a: (meta_dir / f"{a}_metadata.xml").exists())
            report["wanted"] = {
                "total": len(wanted),
                "fastq_present": len(fastq_present),
                "fastq_missing": fastq_missing,
                "metadata_present": len(meta_present),
                "metadata_missing": meta_missing,
            }
        return report

    def _print_report(self, report: dict, list_missing: bool) -> None:
        od = report["on_disk"]
        print("Local inventory")
        print("===============")
        print(f"  FASTQ accessions on disk : {od['fastq_accessions']}")
        print(f"  Metadata XML on disk     : {od['metadata_xml']}")
        print(f"  Genome FASTA on disk     : {od['genome_fasta']}")

        w = report.get("wanted")
        if w:
            print(f"\nReconciled against {w['total']} wanted accession(s)")
            print(f"  FASTQ    : {w['fastq_present']} present, {len(w['fastq_missing'])} missing")
            print(f"  Metadata : {w['metadata_present']} present, {len(w['metadata_missing'])} missing")
            if list_missing:
                if w["fastq_missing"]:
                    print("  Missing FASTQ    : " + ", ".join(w["fastq_missing"]))
                if w["metadata_missing"]:
                    print("  Missing metadata : " + ", ".join(w["metadata_missing"]))

    def execute(self, args: argparse.Namespace) -> int:
        try:
            report = self._build_report(args)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                self._print_report(report, args.list_missing)
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error building status report: {e}")
            return 1
