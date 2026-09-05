"""Local inventory / status CLI command.

Reports what MetaQuest has already downloaded locally (SRA reads, NCBI metadata,
genome assemblies) so a user can see what is available without re-downloading,
and reports where every accession sits in the registry (screened, selected,
excluded, downloaded, analysed, extracted, assembled). When no registry file
exists yet, the report is reconstructed in memory from what is on disk.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metaquest.cli.base import BaseCommand
from metaquest.core.constants import GENOME_FASTA_GLOBS
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.file_io import write_csv
from metaquest.data.read_extraction import summarise_contigs
from metaquest.data.registry import (
    ProjectPaths,
    ReconcileReport,
    Registry,
    STAGES,
    bootstrap_from_disk,
    known_genome_ids,
    load_registry,
    query,
    reconcile,
    registry_path,
    save_registry,
    scan_assemblies,
    stage_counts,
    to_dataframes,
)
from metaquest.data.sra import accession_has_fastq

_CONTIGS_NAME = "final.contigs.fa"


class StatusCommand(BaseCommand):
    """Command to report locally available data and the registry's per-accession stages."""

    @property
    def name(self) -> str:
        return "status"

    @property
    def help(self) -> str:
        return "Report which SRA reads, metadata, and genomes are already available locally"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument("--metadata-folder", default="metadata", help="Folder holding NCBI metadata XML")
        parser.add_argument("--genomes-folder", default="genomes", help="Folder holding genome FASTA files")
        parser.add_argument(
            "--targeted-folder", default="targeted", help="Root folder of extracted reads and assemblies"
        )
        parser.add_argument("--matches-folder", default="matches", help="Folder of Branchwater match CSVs")
        parser.add_argument(
            "--accessions-file",
            help="Optional file of SRA accessions (one per line) to reconcile against local FASTQ/metadata",
        )
        parser.add_argument(
            "--parsed-containment",
            help="Optional parsed containment table; its sample accessions are the wanted list",
        )
        parser.add_argument(
            "--registry",
            default=None,
            help="Registry file (default: metaquest_registry.json found upwards from here)",
        )
        parser.add_argument("--stage", choices=list(STAGES), default=None, help="List the accessions in one stage")
        parser.add_argument(
            "--genome",
            action="append",
            default=None,
            help="Restrict extraction and assembly stages to a genome id (repeatable)",
        )
        parser.add_argument("--init", action="store_true", help="Create the registry from what is on disk")
        parser.add_argument(
            "--reconcile",
            action="store_true",
            help="Compare the registry with the disk and record missing downloads",
        )
        parser.add_argument(
            "--export-tsv", default=None, help="Write <PREFIX>_datasets.tsv and <PREFIX>_extractions.tsv"
        )
        parser.add_argument("--next", action="store_true", help="Suggest the commands that advance the most accessions")
        parser.add_argument("--list-missing", action="store_true", help="Also print the accessions that are missing")
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON")

    # ------------------------------------------------------------------ wanted

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

    def _resolve_wanted(self, args: argparse.Namespace, registry: Registry) -> List[str]:
        """Explicit accessions/containment file win; otherwise fall back to the registry's selection."""
        if args.accessions_file or args.parsed_containment:
            return self._wanted_accessions(args)
        return query(registry, "selected")

    @staticmethod
    def _reconcile_present_missing(wanted: List[str], present_fn) -> Tuple[List[str], List[str]]:
        """Split a wanted list into (present, missing) using a predicate."""
        present = [a for a in wanted if present_fn(a)]
        missing = [a for a in wanted if a not in present]
        return present, missing

    # -------------------------------------------------------------- reporting

    def _inventory_report(self, args: argparse.Namespace, registry: Registry) -> Dict[str, Any]:
        fastq_dir = Path(args.fastq_folder)
        meta_dir = Path(args.metadata_folder)
        genomes_dir = Path(args.genomes_folder)

        if fastq_dir.is_dir():
            on_disk_fastq = sorted(d.name for d in fastq_dir.iterdir() if accession_has_fastq(d))
        else:
            on_disk_fastq = []
        on_disk_meta = sorted(p.name[: -len("_metadata.xml")] for p in meta_dir.glob("*_metadata.xml"))
        on_disk_genomes = sorted({p.name for g in GENOME_FASTA_GLOBS for p in genomes_dir.glob(g)})

        report: Dict[str, Any] = {
            "on_disk": {
                "fastq_accessions": len(on_disk_fastq),
                "metadata_xml": len(on_disk_meta),
                "genome_fasta": len(on_disk_genomes),
            }
        }

        wanted = self._resolve_wanted(args, registry)
        if wanted:
            fastq_present, fastq_missing = self._reconcile_present_missing(
                wanted, lambda a: accession_has_fastq(fastq_dir / a)
            )
            meta_present, meta_missing = self._reconcile_present_missing(
                wanted, lambda a: (meta_dir / f"{a}_metadata.xml").exists()
            )
            report["wanted"] = {
                "total": len(wanted),
                "fastq_present": len(fastq_present),
                "fastq_missing": fastq_missing,
                "metadata_present": len(meta_present),
                "metadata_missing": meta_missing,
            }
        return report

    @staticmethod
    def _stage_filter_accessions(registry: Registry, stage: str, genomes: Optional[List[str]]) -> List[str]:
        if not genomes:
            return query(registry, stage)
        seen: List[str] = []
        for genome_id in genomes:
            for acc in query(registry, stage, genome_id):
                if acc not in seen:
                    seen.append(acc)
        return seen

    def _genome_report(
        self,
        registry: Registry,
        paths: ProjectPaths,
        genome_filter: Optional[List[str]],
        counts: Dict[str, Any],
    ) -> Dict[str, Any]:
        genome_ids = sorted(known_genome_ids(registry))
        if genome_filter:
            wanted = set(genome_filter)
            genome_ids = [g for g in genome_ids if g in wanted]

        empty_by_genome: Dict[str, List[str]] = {}
        for acc, per_genome in scan_assemblies(paths.targeted, genome_ids).items():
            for genome_id, asm_dir in per_genome.items():
                if summarise_contigs(asm_dir / _CONTIGS_NAME)["contigs"] == 0:
                    empty_by_genome.setdefault(genome_id, []).append(acc)

        report: Dict[str, Any] = {}
        for genome_id in genome_ids:
            info = counts["genomes"].get(genome_id, {"extracted": 0, "assembled": 0, "zero_mapped": []})
            report[genome_id] = {
                "extracted": info.get("extracted", 0),
                "assembled": info.get("assembled", 0),
                "zero_mapped": info.get("zero_mapped", []),
                "empty_assembly_dirs": sorted(empty_by_genome.get(genome_id, [])),
            }
        return report

    @staticmethod
    def _drift_report(drift: ReconcileReport) -> Dict[str, Any]:
        return {
            "recorded_missing": list(drift.recorded_missing),
            "untracked_fastq": list(drift.untracked_fastq),
            "untracked_extractions": [[acc, genome_id] for acc, genome_id in drift.untracked_extractions],
            "empty_assembly_dirs": [[acc, genome_id] for acc, genome_id in drift.empty_assembly_dirs],
        }

    @staticmethod
    def _download_next_steps(registry: Registry) -> List[Dict[str, Any]]:
        to_download = [
            acc
            for acc, record in registry.datasets.items()
            if record.get("selection", {}).get("selected")
            and not record.get("exclusion", {}).get("excluded")
            and record.get("download", {}).get("state") != "downloaded"
        ]
        if not to_download:
            return []
        by_output: Dict[str, List[str]] = {}
        for acc in to_download:
            output = registry.datasets[acc].get("selection", {}).get("output") or "accessions.txt"
            by_output.setdefault(output, []).append(acc)
        return [
            {"command": f"metaquest download_sra --accessions-file {output}", "accessions": accs}
            for output, accs in by_output.items()
        ]

    @staticmethod
    def _extraction_next_steps(registry: Registry, paths: ProjectPaths) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        downloaded = query(registry, "downloaded")
        for genome_id in sorted(known_genome_ids(registry)):
            genome_fasta = paths.genomes / f"{genome_id}.fna"
            extracted = set(query(registry, "extracted", genome_id))
            to_extract = [acc for acc in downloaded if acc not in extracted]
            if to_extract:
                steps.append(
                    {
                        "command": f"metaquest extract_target_reads --genome-id {genome_id} "
                        f"--genome-fasta {genome_fasta}",
                        "accessions": to_extract,
                    }
                )
            assembled = set(query(registry, "assembled", genome_id))
            to_assemble = [acc for acc in query(registry, "extracted", genome_id) if acc not in assembled]
            if to_assemble:
                steps.append(
                    {
                        "command": f"metaquest extract_target_reads --genome-id {genome_id} "
                        f"--genome-fasta {genome_fasta} --assemble",
                        "accessions": to_assemble,
                    }
                )
        return steps

    def _next_steps(self, registry: Registry, paths: ProjectPaths) -> List[Dict[str, Any]]:
        return self._download_next_steps(registry) + self._extraction_next_steps(registry, paths)

    def _export_tsv(self, registry: Registry, prefix: str) -> None:
        datasets_df, extractions_df = to_dataframes(registry)
        write_csv(datasets_df, f"{prefix}_datasets.tsv", sep="\t")
        write_csv(extractions_df, f"{prefix}_extractions.tsv", sep="\t")

    # ---------------------------------------------------------------- printing

    @staticmethod
    def _print_inventory(report: Dict[str, Any], list_missing: bool) -> None:
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

    @staticmethod
    def _print_stages(stages: Dict[str, Any]) -> None:
        print("\nStages")
        print("======")
        for stage in STAGES:
            info = stages[stage]
            print(f"  {stage:<10s} : {info['count']}")

    @staticmethod
    def _print_genomes(genomes: Dict[str, Any]) -> None:
        if not genomes:
            return
        print("\nGenomes")
        print("=======")
        for genome_id, info in genomes.items():
            zero_mapped = len(info["zero_mapped"])
            empty_dirs = len(info["empty_assembly_dirs"])
            print(
                f"  {genome_id}: extracted {info['extracted']} ({zero_mapped} with 0 mapped reads), "
                f"assembled {info['assembled']}, {empty_dirs} empty assembly dir(s)"
            )

    @staticmethod
    def _print_stage_filter(registry: Registry, stage: Optional[str], genomes: Optional[List[str]]) -> None:
        if not stage:
            return
        accs = StatusCommand._stage_filter_accessions(registry, stage, genomes)
        detail = f" (genome {', '.join(genomes)})" if genomes else ""
        print(f"\nStage '{stage}'{detail}: " + (", ".join(accs) if accs else "(none)"))

    @staticmethod
    def _print_gaps(registry: Registry) -> None:
        selected = set(query(registry, "selected"))
        excluded = set(query(registry, "excluded"))
        downloaded_set = set(query(registry, "downloaded"))
        not_downloaded = sorted(selected - excluded - downloaded_set)
        print("\nGaps")
        print("====")
        print("  Selected but not downloaded : " + (", ".join(not_downloaded) if not_downloaded else "(none)"))
        for genome_id in sorted(known_genome_ids(registry)):
            gap = sorted(downloaded_set - set(query(registry, "extracted", genome_id)))
            if gap:
                print(f"  Downloaded but not extracted for {genome_id} : " + ", ".join(gap))

    @staticmethod
    def _print_drift(drift: Dict[str, Any]) -> None:
        if not drift:
            return
        print("\nDrift against disk")
        print("===================")
        print(
            "  Recorded downloaded but missing on disk : "
            + (", ".join(drift["recorded_missing"]) if drift["recorded_missing"] else "(none)")
        )
        print(
            "  On disk but not tracked as downloaded   : "
            + (", ".join(drift["untracked_fastq"]) if drift["untracked_fastq"] else "(none)")
        )
        if drift["untracked_extractions"]:
            pairs = ", ".join(f"{acc}/{genome_id}" for acc, genome_id in drift["untracked_extractions"])
            print(f"  Untracked extractions                   : {pairs}")
        if drift["empty_assembly_dirs"]:
            pairs = ", ".join(f"{acc}/{genome_id}" for acc, genome_id in drift["empty_assembly_dirs"])
            print(f"  Empty assembly directories               : {pairs}")

    @staticmethod
    def _print_next(steps: List[Dict[str, Any]]) -> None:
        if not steps:
            return
        print("\nSuggested next steps")
        print("=====================")
        for step in steps:
            print(f"  {step['command']}")
            print("    accessions: " + ", ".join(step["accessions"]))

    def _print_report(self, args: argparse.Namespace, report: Dict[str, Any], registry: Registry) -> None:
        self._print_inventory(report, args.list_missing)
        self._print_stages(report["stages"])
        self._print_genomes(report["genomes"])
        self._print_stage_filter(registry, args.stage, args.genome)
        if args.list_missing:
            self._print_gaps(registry)
        self._print_drift(report["drift"])
        if args.next:
            self._print_next(report.get("next", []))

    def _emit(self, args: argparse.Namespace, report: Dict[str, Any], registry: Registry) -> None:
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            self._print_report(args, report, registry)

    # ------------------------------------------------------------------ execute

    def execute(self, args: argparse.Namespace) -> int:
        try:
            paths = ProjectPaths(
                Path(args.fastq_folder),
                Path(args.metadata_folder),
                Path(args.genomes_folder),
                Path(args.targeted_folder),
                Path(args.matches_folder),
            )
            registry_file = registry_path(args.registry)
            existed = registry_file.exists()
            if existed and not args.init:
                registry = load_registry(registry_file)
            else:
                registry = bootstrap_from_disk(paths, args.accessions_file, args.parsed_containment)
                registry.path = registry_file
                if args.init:
                    save_registry(registry)
                    self.logger.info("Registry written to %s", registry_file)

            drift = None
            if args.reconcile:
                drift = reconcile(registry, paths)
                save_registry(registry)

            report = self._inventory_report(args, registry)
            report["registry"] = {
                "path": str(registry_file),
                "exists": existed or args.init,
                "updated": registry.updated,
            }
            counts = stage_counts(registry)
            report["stages"] = {s: {"count": counts["stages"][s], "accessions": query(registry, s)} for s in STAGES}
            report["genomes"] = self._genome_report(registry, paths, args.genome, counts)
            report["drift"] = self._drift_report(drift) if drift else {}
            if args.next:
                report["next"] = self._next_steps(registry, paths)
            if args.export_tsv:
                self._export_tsv(registry, args.export_tsv)

            if not existed and not args.init:
                self.logger.info(
                    "No registry yet; the stages above were reconstructed from disk. Run: metaquest status --init"
                )

            self._emit(args, report, registry)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error building status report: %s", e)
            return 1
