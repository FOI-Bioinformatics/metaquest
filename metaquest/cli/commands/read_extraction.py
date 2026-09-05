"""CLI command for targeted read extraction before assembly."""

import argparse
from pathlib import Path
from typing import Dict, List

from metaquest.cli.base import BaseCommand
from metaquest.core.constants import DEFAULT_CONTAINMENT_THRESHOLD
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.read_extraction import (
    MINIMAP2_PRESETS,
    ExtractionResult,
    assemble_extracted_reads,
    extract_target_reads,
    megahit_version,
    resolve_assembly_threads,
    selected_samples,
    summarise_contigs,
)
from metaquest.data.registry import (
    extraction_record,
    load_registry,
    record_assembly,
    record_extraction,
    registry_transaction,
)


class ExtractTargetReadsCommand(BaseCommand):
    """Map each sample's reads to a target genome and keep only the mapped reads."""

    @property
    def name(self) -> str:
        return "extract_target_reads"

    @property
    def help(self) -> str:
        return "Filter reads that map to a target genome for a small, targeted assembly"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--parsed-containment",
            required=True,
            help="Parsed containment table (samples x genomes) from parse_containment",
        )
        parser.add_argument("--genome-id", required=True, help="Target genome column to extract against")
        parser.add_argument("--genome-fasta", required=True, help="FASTA file for the target genome")
        parser.add_argument("--fastq-folder", default="fastq", help="Root folder of per-accession FASTQ files")
        parser.add_argument("--output-folder", default="targeted", help="Root folder for the extracted reads")
        parser.add_argument(
            "--threshold",
            type=float,
            default=DEFAULT_CONTAINMENT_THRESHOLD,
            help="Minimum containment for a sample to be included",
        )
        parser.add_argument(
            "--preset", choices=sorted(MINIMAP2_PRESETS), default="sr", help="minimap2 preset for the read type"
        )
        parser.add_argument("--threads", type=int, default=4, help="Threads for minimap2 and samtools")
        parser.add_argument(
            "--assemble", action="store_true", help="Assemble each sample's extracted reads with megahit"
        )
        parser.add_argument(
            "--assembly-threads",
            type=int,
            default=None,
            help="Threads for the megahit assembly (defaults to 1 on macOS, --threads elsewhere)",
        )
        parser.add_argument("--min-contig-len", type=int, default=None, help="megahit minimum contig length")
        parser.add_argument(
            "--dry-run", action="store_true", help="List the qualifying samples without running any tool"
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Redo extraction and assembly even when the registry says they are done",
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")

    def _record_result(self, args: argparse.Namespace, accession: str, outcome: ExtractionResult) -> None:
        """Checkpoint one extraction result; skipped samples are already recorded."""
        if outcome.skipped:
            return
        with registry_transaction(args.registry) as reg:
            record_extraction(
                reg,
                accession,
                args.genome_id,
                outcome.files,
                outcome.mapped_records,
                outcome.unequal_mates,
                {
                    "genome_fasta": str(Path(args.genome_fasta)),
                    "preset": args.preset,
                    "threshold": args.threshold,
                },
            )

    @staticmethod
    def _has_assembly_record(args: argparse.Namespace, accession: str) -> bool:
        """True when the registry already holds an assembly block for this sample and genome."""
        record = extraction_record(load_registry(args.registry), accession, args.genome_id) or {}
        return record.get("assembly") is not None

    def _report_dry_run(self, args: argparse.Namespace, results: Dict[str, ExtractionResult]) -> None:
        """List the samples a real run would extract, and those it would skip."""
        would_skip = [acc for acc, r in results.items() if r.skipped]
        self.logger.info(
            "Dry run: %d sample(s) would be extracted for %s", len(results) - len(would_skip), args.genome_id
        )
        if would_skip:
            self.logger.info("  %d already extracted, would be skipped: %s", len(would_skip), ", ".join(would_skip))
        for accession, outcome in results.items():
            if not outcome.skipped:
                self.logger.info("  %s", accession)

    def _report_no_reads(self, args: argparse.Namespace, results: Dict[str, ExtractionResult]) -> None:
        """Say which of the three reasons left the run without a single mapped read."""
        selected = selected_samples(args.parsed_containment, args.genome_id, args.threshold)
        if not selected:
            self.logger.error("No sample meets containment >= %s for %s", args.threshold, args.genome_id)
        elif not results:
            self.logger.error(
                "No FASTQ files found for the %d selected sample(s) under %s", len(selected), args.fastq_folder
            )
        else:
            self.logger.error("No reads mapped to %s in any sample; check the FASTQ files and --preset", args.genome_id)

    def _assemble(self, args: argparse.Namespace, with_reads: Dict[str, List[Path]]) -> None:
        """Assemble every sample that has mapped reads, recording each assembly as it lands."""
        asm_threads = resolve_assembly_threads(args.assembly_threads, args.threads)
        if args.assembly_threads is None and asm_threads < args.threads:
            self.logger.info(
                "Running megahit single-threaded on macOS (its parallel sort is unstable here); "
                "override with --assembly-threads"
            )
        version = megahit_version()
        for accession, reads in with_reads.items():
            out_dir = Path(args.output_folder) / accession / f"{args.genome_id}_assembly"
            _, ran = assemble_extracted_reads(
                reads, out_dir, threads=asm_threads, min_contig_len=args.min_contig_len, force=args.force
            )
            if not ran and self._has_assembly_record(args, accession):
                # megahit did not run, so the recorded version and parameters still describe
                # the assembly on disk; leave them alone.
                continue
            with registry_transaction(args.registry) as reg:
                record_assembly(
                    reg,
                    accession,
                    args.genome_id,
                    out_dir,
                    summarise_contigs(out_dir / "final.contigs.fa"),
                    version,
                    {"threads": asm_threads, "min_contig_len": args.min_contig_len},
                )
        self.logger.info("Assembled %d sample(s)", len(with_reads))

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            already_done = {
                acc: rec
                for acc in registry.datasets
                if (rec := extraction_record(registry, acc, args.genome_id)) is not None
            }

            results = extract_target_reads(
                parsed_containment=args.parsed_containment,
                genome_id=args.genome_id,
                genome_fasta=args.genome_fasta,
                fastq_folder=args.fastq_folder,
                output_folder=args.output_folder,
                threshold=args.threshold,
                preset=args.preset,
                threads=args.threads,
                dry_run=args.dry_run,
                force=args.force,
                already_done=already_done,
                on_result=lambda accession, outcome: self._record_result(args, accession, outcome),
            )

            if args.dry_run:
                self._report_dry_run(args, results)
                return 0

            # A zero-mapped sample can still have leftover files on disk from an earlier run;
            # they are not reads that mapped, and must never reach the assembler.
            with_reads = {acc: r.files for acc, r in results.items() if r.files and r.mapped_records > 0}
            self.logger.info("Extracted reads for %d of %d sample(s)", len(with_reads), len(results))
            if not with_reads:
                self._report_no_reads(args, results)
                return 1

            if args.assemble:
                self._assemble(args, with_reads)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error extracting target reads: %s", e)
            return 1
