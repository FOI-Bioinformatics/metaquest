"""CLI command for targeted read extraction before assembly."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metaquest.cli.base import BaseCommand
from metaquest.core.constants import DEFAULT_CONTAINMENT_THRESHOLD
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.read_extraction import (
    MINIMAP2_PRESETS,
    ExtractionResult,
    _record_matches,
    _sample_reads,
    assemble_extracted_reads,
    assembly_coverage,
    extract_target_reads,
    fasta_length,
    megahit_version,
    resolve_assembly_threads,
    resolve_index_path,
    selected_samples,
    summarise_contigs,
)
from metaquest.data.registry import (
    Registry,
    extraction_record,
    load_registry,
    record_assembly,
    record_extraction,
    registry_transaction,
    resolve_project_path,
    upsert_dataset,
)
from metaquest.data.sra import count_fastq_reads
from metaquest.data.sra_metadata import _resolved_sidecar_path
from metaquest.store.layout import StorePaths
from metaquest.store.resolve import resolve_optional_store
from metaquest.store.stats import cached_stats
from metaquest.store.usage import record_usage_safe
from metaquest.utils.security import missing_tools


def _non_negative_int(value: str) -> int:
    """argparse type for --min-mapq: a mapping quality is never negative.

    Without this, ``--min-mapq -5`` reaches samtools as ``-q -5``, where the security layer
    rejects ``-5`` as an unknown flag only after minimap2 has already run.
    """
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"--min-mapq must be zero or a positive integer, got {value!r}")
    return parsed


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
            "--min-mapq",
            type=_non_negative_int,
            default=0,
            help=(
                "Minimum mapping quality kept, in addition to dropping secondary/supplementary "
                "alignments (default: 0, keep every mapped record). Try 20 for a close relative of "
                "the target genome; a divergent strain can genuinely map with a low MAPQ, so raising "
                "this can discard real matches."
            ),
        )
        parser.add_argument(
            "--temp-folder",
            default=None,
            help="Where the intermediate SAM alignment(s) are written (default: alongside each sample's output)",
        )
        parser.add_argument(
            "--allow-truncated",
            action="store_true",
            help="Extract even for a sample whose registry download verdict is 'truncated'",
        )
        parser.add_argument(
            "--debug-keep-sam",
            action="store_true",
            help="Keep the intermediate SAM alignment(s) instead of removing them once the BAM exists",
        )
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
            "--assembly-preset",
            choices=["default", "meta-sensitive", "meta-large"],
            default="meta-sensitive",
            help="megahit --presets value ('default' omits the flag)",
        )
        parser.add_argument(
            "--keep-intermediate",
            action="store_true",
            help="Keep megahit's intermediate_contigs/ folder instead of removing it after a successful assembly",
        )
        parser.add_argument(
            "--no-coverage",
            action="store_true",
            help="Skip mapping the extracted reads back onto the assembled contigs for coverage stats",
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="List the qualifying samples without running any tool"
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Redo extraction and assembly even when the registry says they are done",
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")

    @staticmethod
    def _resolve_store(args: argparse.Namespace, registry: Registry) -> Optional[StorePaths]:
        """Resolve the shared data store (if any); returns None without one.

        Extraction reads the project's own ``fastq/`` folder either way, so a store that
        cannot be reached is a warning, not a reason to stop."""
        return resolve_optional_store(getattr(args, "data_root", None), registry.store.get("root"))

    @staticmethod
    def _mate_signature(reads: List[Path]) -> List[List[Any]]:
        """``[[name, size bytes, mtime], ...]`` for the mate files a count was taken from."""
        signature = []
        for path in reads:
            stat = path.stat()
            signature.append([path.name, stat.st_size, stat.st_mtime])
        return signature

    def _counts_from_store(self, accession: str, reads: List[Path]) -> Optional[Tuple[int, int]]:
        """The mate counts from the dataset's shared statistics record, when it has one.

        The record is size/mtime invalidated against the files on disk, so unlike the
        registry's own copy it cannot describe a download that has since been replaced.
        """
        acc_dir = reads[0].parent
        cached = cached_stats(acc_dir, _resolved_sidecar_path(acc_dir))
        reads_per_file = (cached or {}).get("reads_per_file") or {}
        first, second = (reads_per_file.get(path.name) for path in reads)
        if first is None or second is None:
            return None
        return int(first), int(second)

    def _cached_registry_counts(
        self, registry: Registry, accession: str, reads: List[Path]
    ) -> Optional[Tuple[int, int]]:
        """The registry's recorded mate pair, but only while it still describes these files.

        The pair is stored with a signature of the files it was counted from. A sample
        re-downloaded since (``--resume-partial`` completing a truncated pair) no longer
        matches, and the stale pair is ignored: an unequal pair would otherwise force
        single-end mapping of a now-complete paired run.
        """
        download = registry.datasets.get(accession, {}).get("download") or {}
        cached = download.get("mate_reads")
        if cached is None or len(cached) != 2:
            return None
        if download.get("mate_reads_signature") != self._mate_signature(reads):
            self.logger.debug("%s: the recorded mate counts no longer match the files on disk", accession)
            return None
        return int(cached[0]), int(cached[1])

    def _mate_counts(
        self, args: argparse.Namespace, registry: Registry, selected: List[str]
    ) -> Dict[str, Tuple[int, int]]:
        """Accession -> (mate 1 reads, mate 2 reads) for every selected sample with both mate
        files present on disk.

        The counts come from the dataset's shared statistics record when it has one, then
        from the registry's own recorded pair while its file signature still matches, and
        otherwise from counting the files now, which is recorded with a signature so a later
        run does not recount them. Counting is best-effort: a file that cannot be read (e.g.
        corrupted, or not really gzip despite its name) is logged and skipped rather than
        stopping the extraction.
        """
        counts: Dict[str, Tuple[int, int]] = {}
        for accession in selected:
            reads = _sample_reads(Path(args.fastq_folder), accession)
            if len(reads) != 2:
                continue
            try:
                pair = self._counts_from_store(accession, reads) or self._cached_registry_counts(
                    registry, accession, reads
                )
                if pair is not None:
                    counts[accession] = pair
                    continue
                signature = self._mate_signature(reads)
                pair = (count_fastq_reads(reads[0]), count_fastq_reads(reads[1]))
            except Exception as e:
                self.logger.warning(
                    "Could not count reads in %s's mate files (%s); skipping the pre-count", accession, e
                )
                continue
            counts[accession] = pair
            with registry_transaction(args.registry) as reg:
                download = upsert_dataset(reg, accession).setdefault("download", {"attempts": 0})
                download["mate_reads"] = [pair[0], pair[1]]
                download["mate_reads_signature"] = signature
        return counts

    def _samples_needing_mate_counts(
        self, selected: List[str], already_done: Dict[str, Any], truncated: Dict[str, Any], args: argparse.Namespace
    ) -> List[str]:
        """The selected samples that will really be mapped, so only their mates are counted.

        Counting reads both mate files of a sample, which is a streaming pass over every
        byte; doing it for a sample that is about to be skipped as already extracted or as a
        truncated download is pure cost.
        """
        genome_path = Path(args.genome_fasta)

        def will_be_skipped(accession: str) -> bool:
            record = already_done.get(accession)
            done = (
                record is not None
                and not args.force
                and _record_matches(record, genome_path, args.preset, args.threshold)
            )
            return done or (accession in truncated and not args.allow_truncated)

        return [accession for accession in selected if not will_be_skipped(accession)]

    @staticmethod
    def _truncated_downloads(registry: Registry) -> Dict[str, Dict[str, Any]]:
        """Accession -> download completeness verdict, for every accession whose verdict is
        ``"truncated"``."""
        truncated = {}
        for accession, record in registry.datasets.items():
            verdict = (record.get("download") or {}).get("complete") or {}
            if verdict.get("verdict") == "truncated":
                truncated[accession] = verdict
        return truncated

    def _record_result(
        self,
        args: argparse.Namespace,
        accession: str,
        outcome: ExtractionResult,
        store: Optional[StorePaths] = None,
    ) -> None:
        """Checkpoint one extraction result; skipped samples are already recorded."""
        if outcome.skipped:
            return
        index_dir = Path(args.output_folder) / ".index"
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
                    "filter_flags": "0x904",
                    "min_mapq": args.min_mapq,
                    "index": str(resolve_index_path(args.genome_fasta, args.preset, index_dir)),
                },
                mapped_total=outcome.mapped_total,
            )
            record_usage_safe(
                store, reg, accession, args.genome_id, "extracted", detail=f"{outcome.mapped_records} mapped reads"
            )

    @staticmethod
    def _resolved_extraction_record(registry: Registry, accession: str, genome_id: str) -> Optional[Dict[str, Any]]:
        """The recorded extraction for one sample, with its ``genome_fasta`` and ``files``
        resolved against the project root.

        The registry stores these paths relative to its own location so the project can be
        moved; ``extract_target_reads`` and its ``_record_matches`` helper know nothing about
        the registry or its project root, so the paths must already be absolute (or otherwise
        directly usable) by the time they reach it.
        """
        record = extraction_record(registry, accession, genome_id)
        if record is None:
            return None
        resolved = dict(record)
        genome_fasta = record.get("genome_fasta")
        if genome_fasta is not None:
            resolved["genome_fasta"] = str(resolve_project_path(registry, genome_fasta))
        resolved["files"] = [str(resolve_project_path(registry, p)) for p in record.get("files", [])]
        return resolved

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

    def _assemble(
        self,
        args: argparse.Namespace,
        with_reads: Dict[str, List[Path]],
        results: Dict[str, ExtractionResult],
        store: Optional[StorePaths] = None,
    ) -> None:
        """Assemble every sample that has mapped reads, recording each assembly as it lands."""
        asm_threads = resolve_assembly_threads(args.assembly_threads, args.threads)
        if args.assembly_threads is None and asm_threads < args.threads:
            self.logger.info(
                "Running megahit single-threaded on macOS (its parallel sort is unstable here); "
                "override with --assembly-threads"
            )
        version = megahit_version()
        genome_length = fasta_length(args.genome_fasta)
        for accession, reads in with_reads.items():
            out_dir = Path(args.output_folder) / accession / f"{args.genome_id}_assembly"
            _, ran = assemble_extracted_reads(
                reads,
                out_dir,
                threads=asm_threads,
                min_contig_len=args.min_contig_len,
                force=args.force,
                preset=args.assembly_preset,
                keep_intermediate=args.keep_intermediate,
            )
            if not ran and self._has_assembly_record(args, accession):
                # megahit did not run, so the recorded version and parameters still describe
                # the assembly on disk; leave them alone.
                continue
            contigs_path = out_dir / "final.contigs.fa"
            stats: Dict[str, Any] = dict(summarise_contigs(contigs_path))
            stats["genome_fraction_estimate"] = (stats["total_bp"] / genome_length) if genome_length else None
            if not args.no_coverage:
                coverage = assembly_coverage(
                    contigs_path,
                    reads,
                    args.preset,
                    asm_threads,
                    out_dir,
                    mapped_reads=results[accession].mapped_records,
                )
                stats.update(coverage)
            with registry_transaction(args.registry) as reg:
                record_assembly(
                    reg,
                    accession,
                    args.genome_id,
                    out_dir,
                    stats,
                    version,
                    {
                        "threads": asm_threads,
                        "min_contig_len": args.min_contig_len,
                        "preset": args.assembly_preset,
                    },
                )
                record_usage_safe(
                    store, reg, accession, args.genome_id, "assembled", detail=f"{stats.get('contigs', 0)} contigs"
                )
        self.logger.info("Assembled %d sample(s)", len(with_reads))

    def _check_required_tools(self, args: argparse.Namespace) -> bool:
        """Report any of minimap2/samtools (and megahit when --assemble) missing on PATH.

        Checked once before any work starts, rather than surfacing as a raw subprocess
        error partway through extraction; skipped entirely for --dry-run, which never runs
        an external tool. Returns True when every required tool is present.
        """
        if args.dry_run:
            return True
        required = ["minimap2", "samtools"] + (["megahit"] if args.assemble else [])
        missing = missing_tools(required)
        for tool in missing:
            self.logger.error("%s not found on PATH. Install it, for example: conda install -c bioconda %s", tool, tool)
        return not missing

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if not self._check_required_tools(args):
                return 1
            registry = load_registry(args.registry)
            store = self._resolve_store(args, registry)
            already_done = {
                acc: rec
                for acc in registry.datasets
                if (rec := self._resolved_extraction_record(registry, acc, args.genome_id)) is not None
            }

            truncated_downloads = self._truncated_downloads(registry)

            mate_counts: Dict[str, Any] = {}
            if not args.dry_run:
                selected = selected_samples(args.parsed_containment, args.genome_id, args.threshold)
                to_count = self._samples_needing_mate_counts(selected, already_done, truncated_downloads, args)
                mate_counts = self._mate_counts(args, registry, to_count)

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
                on_result=lambda accession, outcome: self._record_result(args, accession, outcome, store),
                min_mapq=args.min_mapq,
                temp_folder=args.temp_folder,
                allow_truncated=args.allow_truncated,
                mate_counts=mate_counts,
                truncated_downloads=truncated_downloads,
                keep_sam=args.debug_keep_sam,
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
                self._assemble(args, with_reads, results, store)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error extracting target reads: %s", e)
            return 1
