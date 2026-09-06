"""
Enhanced SRA CLI commands for MetaQuest.

This module provides the sra_info, sra_stats, and sra_validate commands for
previewing NCBI metadata, computing statistics, and validating downloaded datasets.
"""

import logging
from pathlib import Path
from typing import Optional

from metaquest.cli.base import BaseCommand
from metaquest.data.defaults import read_records
from metaquest.data.registry import Registry, load_registry, nan_to_none, record_analysis, save_registry
from metaquest.data.sra_metadata import (
    SRAMetadataClient,
    create_download_preview,
    estimate_download_time,
    save_metadata_report,
    generate_statistics_report,
)
from metaquest.store.layout import StorePaths, store_paths
from metaquest.store.resolve import resolve_store_root
from metaquest.store.usage import record_usage_safe

logger = logging.getLogger(__name__)


def _resolve_command_store(args, registry: Registry) -> Optional[StorePaths]:
    """Resolve the shared data store (if any) for a command's ``--data-root``/registry.

    ``getattr`` guards ``args.data_root`` so a namespace built without that attribute (an
    older test, or a caller that never reaches this code path) is never broken by it.
    """
    store_root = resolve_store_root(getattr(args, "data_root", None), registry.store.get("root"))
    if store_root is None:
        return None
    return store_paths(store_root)


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
            summary = {
                "total_reads": nan_to_none(row.get("total_reads")),
                "gc_content": nan_to_none(row.get("gc_content")),
                "avg_read_length": nan_to_none(row.get("avg_read_length")),
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
            generate_statistics_report(fastq_folder, args.output_report)

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
    def _paired_end_issues(fastq_files) -> list:
        """Issue if R1/R2 counts are mismatched."""
        r1_files = [f for f in fastq_files if "_R1" in f.name or "_1" in f.name]
        r2_files = [f for f in fastq_files if "_R2" in f.name or "_2" in f.name]
        if len(r1_files) != len(r2_files) and len(r2_files) > 0:
            return ["Mismatched paired-end files"]
        return []

    @staticmethod
    def _fastq_format_issues(fastq_files) -> list:
        """Issue if the first FASTQ file has no parseable records or fails to parse."""
        try:
            from Bio import SeqIO

            for f in fastq_files[:1]:  # Check first file only for speed
                with open(f, "rt") as handle:
                    records = list(SeqIO.parse(handle, "fastq"))
                    if len(records) == 0:
                        return [f"No valid FASTQ records in {f.name}"]
        except Exception as e:
            return [f"FASTQ format error: {e}"]
        return []

    def _validate_directory(self, acc_dir, check_pairs=False):
        """Validate a single accession directory."""
        print(f"Validating {acc_dir.name}...")

        fastq_files = list(acc_dir.glob("*.fastq*"))
        if not fastq_files:
            return {
                "accession": acc_dir.name,
                "status": "FAILED",
                "issues": "No FASTQ files found",
                "num_files": 0,
            }

        issues = self._empty_file_issues(fastq_files)
        if check_pairs:
            issues += self._paired_end_issues(fastq_files)
        issues += self._fastq_format_issues(fastq_files)

        return {
            "accession": acc_dir.name,
            "status": "PASSED" if not issues else "FAILED",
            "issues": "; ".join(issues) if issues else "None",
            "num_files": len(fastq_files),
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
            validation_results = []
            for acc_dir in accession_dirs:
                result = self._validate_directory(acc_dir, args.check_pairs)
                validation_results.append(result)
                record_analysis(
                    registry,
                    result["accession"],
                    "validate",
                    "",
                    {"passed": result["status"] == "PASSED", "files": result.get("num_files", 0)},
                )
                record_usage_safe(store, registry, result["accession"], "", "analysed", detail="validate")
            save_registry(registry)

            success = self._print_validation_results(validation_results)
            return 0 if success else 1

        except Exception as e:
            logger.error(f"SRA validation failed: {e}")
            return 1
