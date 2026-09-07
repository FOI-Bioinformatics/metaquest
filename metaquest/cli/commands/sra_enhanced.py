"""
Enhanced SRA CLI commands for MetaQuest.

This module provides the sra_info, sra_stats, and sra_validate commands for
previewing NCBI metadata, computing statistics, and validating downloaded datasets.
"""

import argparse
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from metaquest.cli.base import BaseCommand
from metaquest.data.defaults import read_records
from metaquest.data.registry import Registry, load_registry, nan_to_none, record_analysis, save_registry
from metaquest.data.sra import (
    MATE1_SUFFIXES,
    MATE_SUFFIXES,
    count_fastq_reads,
    fastq_files,
    fastq_stem,
)
from metaquest.data.sra_metadata import (
    SRAMetadataClient,
    _resolved_sidecar_path,
    create_download_preview,
    estimate_download_time,
    save_metadata_report,
    generate_statistics_report,
)
from metaquest.store.layout import StorePaths
from metaquest.store.resolve import resolve_optional_store
from metaquest.store.sidecar import md5_file, read_sidecar
from metaquest.store.stats import DEFAULT_SAMPLE_SIZE, cached_stats
from metaquest.store.usage import record_usage_safe

logger = logging.getLogger(__name__)

# Suffixes marking the second mate of a pair, i.e. MATE_SUFFIXES minus MATE1_SUFFIXES.
_MATE2_SUFFIXES = tuple(suffix for suffix in MATE_SUFFIXES if suffix not in MATE1_SUFFIXES)


def _positive_int(value: str) -> int:
    """argparse type for --sample-size: rejects zero and negative values with a clear message."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--sample-size must be a positive integer, got {value!r}")
    return parsed


def _resolve_command_store(args, registry: Registry) -> Optional[StorePaths]:
    """Resolve the shared data store (if any) for a command's ``--data-root``/registry.

    ``getattr`` guards ``args.data_root`` so a namespace built without that attribute (an
    older test, or a caller that never reaches this code path) is never broken by it. A store
    that cannot be reached only costs the usage record, so it is logged and skipped rather
    than failing an analysis the project can run on its own files.
    """
    return resolve_optional_store(getattr(args, "data_root", None), registry.store.get("root"))


class SRAInfoCommand(BaseCommand):
    """Command for getting SRA dataset information before downloading."""

    @property
    def name(self) -> str:
        return "sra_info"

    @property
    def help(self) -> str:
        return "Get detailed information about SRA datasets before downloading"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser):
        parser.add_argument(
            "--accessions-file",
            required=True,
            help="File containing SRA accessions, one per line",
        )
        parser.add_argument(
            "--email",
            required=True,
            help="Email address for NCBI API access (required by NCBI)",
        )
        parser.add_argument(
            "--api-key",
            help="NCBI API key for increased rate limits (optional)",
        )
        parser.add_argument(
            "--output-report",
            default="sra_info_report.csv",
            help="Output file for detailed report",
        )
        parser.add_argument(
            "--bandwidth-mbps",
            type=float,
            default=100.0,
            help="Estimated bandwidth in Mbps for download time estimation",
        )

    @staticmethod
    def _print_analysis_summary(accessions, metadata, tech_counts, total_size_gb, bandwidth_mbps):
        """Print the SRA dataset analysis summary (counts, distributions, size, ETA)."""
        print("\nSRA Dataset Analysis:")
        print("===================")
        print(f"Total accessions: {len(accessions)}")
        print(f"Metadata fetched: {len(metadata)}")
        print(f"Total estimated size: {total_size_gb:.2f} GB")

        if tech_counts:
            print("\nTechnology distribution:")
            for tech, count in tech_counts.items():
                print(f"  {tech}: {count} datasets")

        platforms: dict = {}
        layouts: dict = {}
        for info in metadata.values():
            platforms[info.platform] = platforms.get(info.platform, 0) + 1
            layouts[info.layout] = layouts.get(info.layout, 0) + 1

        if platforms:
            print("\nPlatform distribution:")
            for platform, count in platforms.items():
                print(f"  {platform}: {count}")
        if layouts:
            print("\nLayout distribution:")
            for layout, count in layouts.items():
                print(f"  {layout}: {count}")

        sizes = [info.size_mb / 1024 for info in metadata.values()]  # Convert to GB
        if sizes:
            print("\nSize statistics:")
            print(f"  Average size per dataset: {sum(sizes)/len(sizes):.2f} GB")
            print(f"  Largest dataset: {max(sizes):.2f} GB")
            print(f"  Smallest dataset: {min(sizes):.2f} GB")

        estimated_hours = estimate_download_time(total_size_gb, bandwidth_mbps, 4)
        if estimated_hours < 1:
            print(f"  Estimated download time: {estimated_hours*60:.0f} minutes")
        else:
            print(f"  Estimated download time: {estimated_hours:.1f} hours")

    def execute(self, args):
        try:
            with open(args.accessions_file, "r") as f:
                accessions = [line.strip() for line in f if line.strip()]

            if not accessions:
                print("No accessions found in file")
                return 1

            print(f"Analyzing {len(accessions)} SRA accessions...")

            client = SRAMetadataClient(args.email, args.api_key)
            metadata, tech_counts, total_size_gb = create_download_preview(accessions, client)

            if not metadata:
                print("Could not fetch metadata for any accessions")
                return 1

            self._print_analysis_summary(accessions, metadata, tech_counts, total_size_gb, args.bandwidth_mbps)

            save_metadata_report(metadata, args.output_report)
            print(f"\nDetailed report saved to: {args.output_report}")

            return 0

        except Exception as e:
            logger.error(f"SRA info command failed: {e}")
            return 1


class SRAStatsCommand(BaseCommand):
    """Command for calculating comprehensive statistics on downloaded SRA data."""

    @property
    def name(self) -> str:
        return "sra_stats"

    @property
    def help(self) -> str:
        return "Calculate comprehensive statistics for downloaded SRA datasets"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser):
        parser.add_argument(
            "--fastq-folder",
            default="fastq",
            help="Folder containing downloaded FASTQ files",
        )
        parser.add_argument(
            "--output-report",
            default="sra_statistics.csv",
            help="Output file for statistics report",
        )
        parser.add_argument(
            "--accessions",
            nargs="*",
            help="Specific accessions to analyze (default: all)",
        )
        parser.add_argument(
            "--sample-size",
            type=_positive_int,
            default=DEFAULT_SAMPLE_SIZE,
            help="Records sampled per dataset for the per-read metrics such as GC content, "
            f"quality and read length; read totals stay exact (default: {DEFAULT_SAMPLE_SIZE})",
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")

    def _record_statistics(self, args, report_path: Path) -> None:
        """Record an sra_stats analysis for every accession in the statistics report.

        If the report is missing, empty, or lacks an accession column (e.g. the
        generator was mocked in a test without writing a real file), this logs at
        debug level and does nothing.
        """
        if not report_path.exists():
            logger.debug("Statistics report %s not found; skipping registry recording", report_path)
            return
        try:
            df = read_records(report_path)
        except Exception as e:
            logger.debug("Could not read statistics report %s: %s", report_path, e)
            return
        if df.empty or "accession" not in df.columns:
            logger.debug("Statistics report %s is empty; skipping registry recording", report_path)
            return

        registry = load_registry(args.registry)
        store = _resolve_command_store(args, registry)
        for _, row in df.iterrows():
            accession = str(row["accession"])
            sampled = nan_to_none(row.get("sampled"))
            summary = {
                "total_reads": nan_to_none(row.get("total_reads")),
                "gc_content": nan_to_none(row.get("gc_content")),
                "avg_read_length": nan_to_none(row.get("avg_read_length")),
                # True when the per-read metrics came from a sample of the records; the
                # read total itself is exact either way.
                "sampled": None if sampled is None else bool(sampled),
            }
            record_analysis(registry, accession, "sra_stats", report_path, summary)
            record_usage_safe(store, registry, accession, "", "analysed", detail="sra_stats")
        save_registry(registry)

    def execute(self, args):
        try:
            fastq_folder = Path(args.fastq_folder)

            if not fastq_folder.exists():
                print(f"FASTQ folder {fastq_folder} does not exist")
                return 1

            print("Calculating comprehensive statistics for downloaded datasets...")

            # Generate statistics report
            generate_statistics_report(
                fastq_folder, args.output_report, sample_size=getattr(args, "sample_size", DEFAULT_SAMPLE_SIZE)
            )

            print(f"\nStatistics report saved to: {args.output_report}")

            self._record_statistics(args, Path(args.output_report))

            return 0

        except Exception as e:
            logger.error(f"SRA stats command failed: {e}")
            return 1


class SRAValidateCommand(BaseCommand):
    """Command for validating downloaded SRA datasets."""

    @property
    def name(self) -> str:
        return "sra_validate"

    @property
    def help(self) -> str:
        return "Validate integrity of downloaded SRA datasets"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser):
        parser.add_argument(
            "--fastq-folder",
            default="fastq",
            help="Folder containing downloaded FASTQ files",
        )
        parser.add_argument(
            "--accessions",
            nargs="*",
            help="Specific accessions to validate (default: all)",
        )
        parser.add_argument(
            "--check-pairs",
            action="store_true",
            help="Check that paired-end files have matching read counts",
        )
        parser.add_argument(
            "--md5",
            action="store_true",
            help="Re-hash each FASTQ file and compare it with the md5 the store recorded for "
            "that file when it was stored, which detects a file changed or corrupted since "
            "(no-op for a dataset without a store sidecar)",
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")

    def _find_accession_dirs(self, fastq_folder, specific_accessions=None):
        """Find accession directories to validate."""
        accession_dirs = [d for d in fastq_folder.iterdir() if d.is_dir()]
        if specific_accessions:
            accession_dirs = [d for d in accession_dirs if d.name in specific_accessions]
        return accession_dirs

    @staticmethod
    def _empty_file_issues(fastq_files) -> list:
        """Issues for any zero-byte FASTQ files."""
        return [f"Empty file: {f.name}" for f in fastq_files if f.stat().st_size == 0]

    @staticmethod
    def _first_record_issue(path: Path) -> Optional[str]:
        """The problem with ``path``'s first FASTQ record, or None when it looks right.

        Opens the file once (gzip aware) and reads four lines: the '@' header, the sequence,
        the '+' separator and a quality string of the same length. A large corrupted file is
        never parsed beyond that.
        """
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            header = handle.readline()
            if not header:
                return f"No valid FASTQ records in {path.name}"
            if not header.startswith("@"):
                return f"FASTQ format error in {path.name}: header does not start with '@'"
            seq, plus, qual = handle.readline(), handle.readline(), handle.readline()
        if not seq or not plus or not qual:
            return f"FASTQ format error in {path.name}: the first record is incomplete"
        if not plus.startswith("+"):
            return f"FASTQ format error in {path.name}: the third line does not start with '+'"
        if len(seq.rstrip("\r\n")) != len(qual.rstrip("\r\n")):
            return f"FASTQ format error in {path.name}: sequence/quality length mismatch"
        return None

    @staticmethod
    def _fastq_format_issues(acc_dir: Path) -> list:
        """Issue for any FASTQ file in ``acc_dir`` whose first record is malformed.

        Every file is checked (gzip aware, via ``metaquest.data.sra.fastq_files``), not only
        the first, since a download can leave one good mate and one broken one.
        """
        issues = []
        for f in fastq_files(acc_dir):
            try:
                issue = SRAValidateCommand._first_record_issue(f)
            except (ValueError, OSError) as e:
                issues.append(f"FASTQ format error in {f.name}: {e}")
                continue
            if issue is not None:
                issues.append(issue)
        return issues

    @staticmethod
    def _mate_count_issues(acc_dir: Path, cached: Optional[Dict[str, Any]]) -> list:
        """Issue when a paired-end dataset's two mate files have different read counts.

        Read counts come from a cached stats record's ``reads_per_file`` (already computed,
        no file I/O) when available, else a fresh ``count_fastq_reads`` per mate file. A
        dataset with no complete mate-1/mate-2 pair (single-end, or an incomplete pair) is
        not flagged here.
        """
        files = fastq_files(acc_dir)
        mate1 = next((f for f in files if fastq_stem(f).endswith(MATE1_SUFFIXES)), None)
        mate2 = next((f for f in files if fastq_stem(f).endswith(_MATE2_SUFFIXES)), None)
        if mate1 is None or mate2 is None:
            return []
        reads_per_file = (cached or {}).get("reads_per_file") or {}
        n1 = reads_per_file.get(mate1.name)
        if n1 is None:
            n1 = count_fastq_reads(mate1)
        n2 = reads_per_file.get(mate2.name)
        if n2 is None:
            n2 = count_fastq_reads(mate2)
        if n1 != n2:
            return [f"mate files differ ({n1} vs {n2})"]
        return []

    @staticmethod
    def _completeness_issues(acc_dir: Path, record: Optional[Dict[str, Any]]) -> list:
        """Issue when this accession's download did not complete against NCBI's spot count.

        Prefers the store sidecar (freshest, when ``acc_dir`` is a store link) over the
        registry's own download verdict (``"truncated"`` in its own vocabulary). Silent when
        neither source has ever verified this accession against NCBI.

        Three sidecar states are reported: ``"partial"`` (fewer reads on disk than NCBI's
        spot count), ``"failed"`` (the download did not finish) and ``"downloading"`` (a
        download is in progress, so the files on disk are not the finished dataset). Only
        ``"complete"`` and ``"adopted"`` datasets pass.
        """
        sidecar_path = _resolved_sidecar_path(acc_dir)
        if sidecar_path is not None:
            sidecar = read_sidecar(sidecar_path)
            if sidecar is None:
                return []
            if sidecar.state == "partial":
                reads = sidecar.reads_per_mate
                spots = sidecar.ncbi.get("spots")
                return [f"partial: {reads} reads on disk vs {spots} spots at NCBI"]
            if sidecar.state == "failed":
                return ["failed at NCBI download"]
            if sidecar.state == "downloading":
                return ["download in progress elsewhere"]
            return []

        verdict = ((record or {}).get("download") or {}).get("complete") or {}
        if verdict.get("verdict") == "truncated":
            reads = verdict.get("reads_r1")
            spots = verdict.get("expected_spots")
            return [f"partial: {reads} reads on disk vs {spots} spots at NCBI"]
        return []

    @staticmethod
    def _md5_issues(acc_dir: Path) -> list:
        """Issue for any FASTQ file whose md5 no longer matches the one the store recorded.

        The comparison is against the sidecar's own ``files[].md5``, computed over the stored
        FASTQ file when it was downloaded or adopted, so it detects a file changed or
        corrupted since. It is not NCBI's md5, which covers the ``.sra`` archive rather than
        the FASTQ files extracted from it and so can never match one. A no-op when there is
        no sidecar to compare against."""
        sidecar_path = _resolved_sidecar_path(acc_dir)
        if sidecar_path is None:
            return []
        sidecar = read_sidecar(sidecar_path)
        if sidecar is None:
            return []
        recorded = {entry.get("name"): entry.get("md5") for entry in sidecar.files}
        issues = []
        for f in fastq_files(acc_dir):
            expected = recorded.get(f.name)
            if expected is None:
                continue
            if md5_file(f) != expected:
                issues.append(f"md5 mismatch: {f.name}")
        return issues

    def _validate_directory(
        self,
        acc_dir,
        registry: Optional[Registry] = None,
        check_pairs: bool = False,
        check_md5: bool = False,
    ):
        """Validate a single accession directory."""
        print(f"Validating {acc_dir.name}...")

        raw_files = list(acc_dir.glob("*.fastq*"))
        if not raw_files:
            return {
                "accession": acc_dir.name,
                "status": "FAILED",
                "issues": "No FASTQ files found",
                "issues_list": ["No FASTQ files found"],
                "num_files": 0,
                "checks": [],
            }

        checks = ["empty_files", "format"]
        issues = self._empty_file_issues(raw_files)
        issues += self._fastq_format_issues(acc_dir)

        if check_pairs:
            # Only the mate-count check reads the statistics record, so a run without
            # --check-pairs does not stat the files or read the sidecar for nothing.
            checks.append("mate_counts")
            issues += self._mate_count_issues(acc_dir, cached_stats(acc_dir, _resolved_sidecar_path(acc_dir)))

        checks.append("completeness")
        record = registry.datasets.get(acc_dir.name) if registry is not None else None
        issues += self._completeness_issues(acc_dir, record)

        if check_md5:
            checks.append("md5")
            issues += self._md5_issues(acc_dir)

        return {
            "accession": acc_dir.name,
            "status": "PASSED" if not issues else "FAILED",
            "issues": "; ".join(issues) if issues else "None",
            "issues_list": issues,
            "num_files": len(raw_files),
            "checks": checks,
        }

    def _print_validation_results(self, validation_results):
        """Print validation results summary."""
        print("\nValidation Results:")
        print("=================")

        passed = [r for r in validation_results if r["status"] == "PASSED"]
        failed = [r for r in validation_results if r["status"] == "FAILED"]

        print(f"Total validated: {len(validation_results)}")
        print(f"Passed: {len(passed)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed validations:")
            for result in failed:
                print(f"  {result['accession']}: {result['issues']}")

        return len(failed) == 0

    def execute(self, args):
        try:
            fastq_folder = Path(args.fastq_folder)
            if not fastq_folder.exists():
                print(f"FASTQ folder {fastq_folder} does not exist")
                return 1

            print("Validating downloaded SRA datasets...")

            accession_dirs = self._find_accession_dirs(fastq_folder, args.accessions)
            if not accession_dirs:
                print("No accession directories found")
                return 1

            registry = load_registry(args.registry)
            store = _resolve_command_store(args, registry)
            check_md5 = getattr(args, "md5", False)
            validation_results = []
            for acc_dir in accession_dirs:
                result = self._validate_directory(acc_dir, registry, args.check_pairs, check_md5)
                validation_results.append(result)
                record_analysis(
                    registry,
                    result["accession"],
                    "validate",
                    "",
                    {
                        "passed": result["status"] == "PASSED",
                        "files": result.get("num_files", 0),
                        "issues": result.get("issues_list", []),
                    },
                )
                record_usage_safe(store, registry, result["accession"], "", "analysed", detail="validate")
            save_registry(registry)

            success = self._print_validation_results(validation_results)
            return 0 if success else 1

        except Exception as e:
            logger.error(f"SRA validation failed: {e}")
            return 1
