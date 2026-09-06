"""
SRA-related CLI commands.
"""

import argparse
import csv
import os
import shutil

from metaquest.cli.base import BaseCommand
from pathlib import Path

from metaquest.core.constants import FAILED_ACCESSIONS_FILE
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.registry import Registry, load_registry, query, record_download, registry_transaction
from metaquest.data.sra import default_max_workers, download_sra, parse_verdict_message


class DownloadSraCommand(BaseCommand):
    """Command for downloading SRA datasets."""

    @property
    def name(self) -> str:
        return "download_sra"

    @property
    def help(self) -> str:
        return "Download SRA datasets"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--fastq-folder",
            default="fastq",
            help="Folder to save downloaded FASTQ files",
        )
        parser.add_argument(
            "--accessions-file",
            required=True,
            help="File containing SRA accessions, one per line",
        )
        parser.add_argument(
            "--max-downloads",
            type=int,
            default=None,
            help="Maximum number of datasets to download",
        )
        parser.add_argument(
            "--num-threads",
            type=int,
            default=4,
            help="Number of threads for each fasterq-dump",
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=None,
            help="Number of parallel downloads (default: computed from the CPU count)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Calculate number of accessions without downloading",
        )
        parser.add_argument("--force", action="store_true", help="Force redownload even if files exist")
        parser.add_argument(
            "--max-retries",
            type=int,
            default=1,
            help="Maximum number of retry attempts for failed downloads",
        )
        parser.add_argument(
            "--temp-folder",
            help="Directory to use for fasterq-dump temporary files (must be writable)",
        )
        parser.add_argument(
            "--blacklist",
            nargs="+",
            help="One or more files containing blacklisted accessions, one per line",
        )
        parser.add_argument(
            "--report-file",
            default=None,
            help=(
                "Write a CSV of accession,status,message after the run "
                "(statuses: downloaded, failed, already_present, blacklisted, skipped); "
                "accessions skipped by --max-downloads get a skipped row"
            ),
        )
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        parser.add_argument(
            "--verify-downloads",
            dest="verify_downloads",
            action="store_true",
            default=True,
            help="Verify each download's read count against NCBI's recorded total spots (default: on)",
        )
        parser.add_argument(
            "--no-verify-downloads",
            dest="verify_downloads",
            action="store_false",
            help="Skip completeness verification against NCBI spot counts",
        )
        parser.add_argument(
            "--redownload-truncated",
            action="store_true",
            help="Redownload accessions whose registry verdict is 'truncated' rather than skipping them",
        )

    def _log_dry_run_summary(self, args: argparse.Namespace, stats: dict) -> None:
        """Log the summary for a dry run."""
        self.logger.info(f"Dry run: would download {stats['to_download']} of {stats['total']} datasets")
        self.logger.info(f"  {stats['already_downloaded']} datasets would be skipped (already downloaded)")
        if stats.get("blacklisted", 0) > 0:
            self.logger.info(f"  {stats['blacklisted']} datasets would be skipped (blacklisted)")
        if stats.get("to_download", 0) > 0:
            self.logger.info(f"  Output folder would be: {args.fastq_folder}")
            if args.max_downloads:
                self.logger.info(f"  Limited to {args.max_downloads} downloads")

    def _log_download_summary(self, stats: dict) -> None:
        """Log the summary for a completed download run."""
        self.logger.info("Download summary:")
        self.logger.info(f"  Successfully downloaded: {stats['successful']} datasets")
        self.logger.info(f"  Failed downloads: {stats['failed']} datasets")
        self.logger.info(f"  Already downloaded: {stats['already_downloaded']} datasets")
        if stats.get("blacklisted", 0) > 0:
            self.logger.info(f"  Blacklisted: {stats['blacklisted']} datasets")
        self.logger.info(f"  Total processed: {stats['total']} datasets")

    def _report_failed_downloads(self, args: argparse.Namespace, stats: dict) -> None:
        """Warn about failures and point at the retry file the data layer already wrote."""
        self.logger.warning("Some downloads failed. Use --force to retry or --max-retries to enable automatic retry.")
        if not stats.get("failed_accessions"):
            return
        failed_file = Path(args.fastq_folder) / FAILED_ACCESSIONS_FILE
        self.logger.info(
            f"To retry only failed accessions: metaquest download_sra "
            f"--accessions-file {failed_file} "
            f"--fastq-folder {args.fastq_folder}"
        )

    @staticmethod
    def _write_report(report_file: str, stats: dict) -> None:
        """Write one row per accession with its outcome."""
        failed = set(stats.get("failed_accessions", []))
        rows = []
        for accession, message in stats.get("results", {}).items():
            rows.append((accession, "failed" if accession in failed else "downloaded", message))
        rows.extend((acc, "already_present", "") for acc in stats.get("already_downloaded_accessions", []))
        rows.extend((acc, "blacklisted", "") for acc in stats.get("blacklisted_accessions", []))
        rows.extend((acc, "skipped", "--max-downloads") for acc in stats.get("skipped_accessions", []))
        path = Path(report_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["accession", "status", "message"])
            writer.writerows(sorted(rows))

    @staticmethod
    def _record_skip(reg: Registry, acc: str, message: str, fastq_dir: Path) -> None:
        """Record a skipped accession, unless its record already says the reads are downloaded.

        A blacklist entry or a `--max-downloads` cut-off must never erase the files and
        sizes of an accession that was downloaded on an earlier run.
        """
        if reg.datasets.get(acc, {}).get("download", {}).get("state") == "downloaded":
            return
        record_download(reg, acc, "skipped", fastq_dir, message)

    def _record_run_outcomes(self, args: argparse.Namespace, stats: dict, fastq_dir: Path) -> None:
        """Record the outcomes the download loop could not report, one transaction per accession."""
        for acc in stats.get("already_downloaded_accessions", []):
            with registry_transaction(args.registry) as reg:
                if reg.datasets.get(acc, {}).get("download", {}).get("state") != "downloaded":
                    record_download(reg, acc, "downloaded", fastq_dir, attempt=False)
        for acc in stats.get("blacklisted_accessions", []):
            with registry_transaction(args.registry) as reg:
                self._record_skip(reg, acc, "blacklisted", fastq_dir)
        for acc in stats.get("skipped_accessions", []):
            with registry_transaction(args.registry) as reg:
                self._record_skip(reg, acc, "--max-downloads", fastq_dir)

    def _resolve_max_workers(self, args: argparse.Namespace) -> int:
        """Resolve --max-workers, falling back to a CPU-derived default, and warn on oversubscription."""
        max_workers = args.max_workers if args.max_workers is not None else default_max_workers(args.num_threads)
        cpu_count = os.cpu_count() or 4
        if max_workers * args.num_threads > cpu_count:
            self.logger.warning(
                "--max-workers %d x --num-threads %d = %d threads requested, which exceeds "
                "the %d CPUs detected on this machine; downloads may be slower than expected",
                max_workers,
                args.num_threads,
                max_workers * args.num_threads,
                cpu_count,
            )
        return max_workers

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if not args.dry_run and shutil.which("fasterq-dump") is None:
                self.logger.error(
                    "fasterq-dump not found on PATH. Install sra-tools, "
                    "for example: conda install -c bioconda sra-tools"
                )
                return 1

            verify_downloads = getattr(args, "verify_downloads", True)
            redownload_truncated = getattr(args, "redownload_truncated", False)
            max_workers = self._resolve_max_workers(args)

            excluded: set = set()
            expected_spots: dict = {}
            truncated: set = set()
            on_result = None
            fastq_dir = Path(args.fastq_folder)

            if not args.dry_run:
                project_registry = load_registry(args.registry)
                excluded = set(query(project_registry, "excluded"))

                if verify_downloads:
                    for acc, record in project_registry.datasets.items():
                        spots = (record.get("metadata") or {}).get("run_total_spots")
                        if spots is not None:
                            expected_spots[acc] = spots

                if redownload_truncated:
                    truncated = {
                        acc
                        for acc, record in project_registry.datasets.items()
                        if (record.get("download") or {}).get("complete", {}).get("verdict") == "truncated"
                    }

                def _record_result(accession: str, success: bool, message: str) -> None:
                    with registry_transaction(args.registry) as reg:
                        complete = parse_verdict_message(message) if success else None
                        record_download(
                            reg,
                            accession,
                            "downloaded" if success else "failed",
                            fastq_dir,
                            message,
                            complete=complete,
                        )

                on_result = _record_result

            download_stats = download_sra(
                fastq_folder=args.fastq_folder,
                accessions_file=args.accessions_file,
                max_downloads=args.max_downloads,
                dry_run=args.dry_run,
                num_threads=args.num_threads,
                max_workers=max_workers,
                force=args.force,
                max_retries=args.max_retries,
                temp_folder=args.temp_folder,
                blacklist=args.blacklist,
                blacklist_accessions=excluded,
                on_result=on_result,
                expected_spots=expected_spots if verify_downloads else None,
                redownload_truncated=redownload_truncated,
                truncated_accessions=truncated,
            )

            if args.dry_run:
                self._log_dry_run_summary(args, download_stats)
            else:
                self._log_download_summary(download_stats)
                self._record_run_outcomes(args, download_stats, fastq_dir)

                if args.report_file:
                    self._write_report(args.report_file, download_stats)
                    self.logger.info("Download report written to %s", args.report_file)

            if not args.dry_run and download_stats["failed"] > 0:
                self._report_failed_downloads(args, download_stats)
                return 1

            return 0

        except MetaQuestError as e:
            self.logger.error(f"Error downloading SRA data: {e}")
            return 1
