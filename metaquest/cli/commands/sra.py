"""
SRA-related CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from pathlib import Path

from metaquest.core.exceptions import MetaQuestError
from metaquest.data.sra import (
    download_sra,
    assemble_datasets,
)


class DownloadSraCommand(BaseCommand):
    """Command for downloading SRA datasets."""

    @property
    def name(self) -> str:
        return "download_sra"

    @property
    def help(self) -> str:
        return "Download SRA datasets"

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
            default=4,
            help="Number of threads for parallel downloads",
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

    def execute(self, args: argparse.Namespace) -> int:
        try:
            download_stats = download_sra(
                fastq_folder=args.fastq_folder,
                accessions_file=args.accessions_file,
                max_downloads=args.max_downloads,
                dry_run=args.dry_run,
                num_threads=args.num_threads,
                max_workers=args.max_workers,
                force=args.force,
                max_retries=args.max_retries,
                temp_folder=args.temp_folder,
                blacklist=args.blacklist,
            )

            if args.dry_run:
                self.logger.info(
                    f"Dry run: would download {download_stats['to_download']} of {download_stats['total']} datasets"
                )
                self.logger.info(
                    f"  {download_stats['already_downloaded']} datasets would be skipped (already downloaded)"
                )
                if "blacklisted" in download_stats and download_stats["blacklisted"] > 0:
                    self.logger.info(f"  {download_stats['blacklisted']} datasets would be skipped (blacklisted)")
                if "to_download" in download_stats and download_stats["to_download"] > 0:
                    self.logger.info(f"  Output folder would be: {args.fastq_folder}")
                    if args.max_downloads:
                        self.logger.info(f"  Limited to {args.max_downloads} downloads")
            else:
                self.logger.info("Download summary:")
                self.logger.info(f"  Successfully downloaded: {download_stats['successful']} datasets")
                self.logger.info(f"  Failed downloads: {download_stats['failed']} datasets")
                self.logger.info(f"  Already downloaded: {download_stats['already_downloaded']} datasets")
                if "blacklisted" in download_stats and download_stats["blacklisted"] > 0:
                    self.logger.info(f"  Blacklisted: {download_stats['blacklisted']} datasets")
                self.logger.info(f"  Total processed: {download_stats['total']} datasets")

            # Return error if there were failed downloads
            if not args.dry_run and download_stats["failed"] > 0:
                self.logger.warning(
                    "Some downloads failed. Use --force to retry or --max-retries to enable automatic retry."
                )
                if "failed_accessions" in download_stats and download_stats["failed_accessions"]:
                    # Write failed accessions to file for easier retry
                    failed_file = Path(args.fastq_folder) / "failed_accessions.txt"
                    with open(failed_file, "w") as f:
                        for acc in download_stats["failed_accessions"]:
                            f.write(f"{acc}\n")
                    self.logger.info(f"Failed accessions written to {failed_file}")
                    self.logger.info(
                        f"To retry only failed accessions: metaquest download_sra "
                        f"--accessions-file {failed_file} "
                        f"--fastq-folder {args.fastq_folder}"
                    )
                return 1  # Return error code

            return 0

        except MetaQuestError as e:
            self.logger.error(f"Error downloading SRA data: {e}")
            return 1


class AssembleDatasetsCommand(BaseCommand):
    """Command for assembling datasets from fastq files."""

    @property
    def name(self) -> str:
        return "assemble_datasets"

    @property
    def help(self) -> str:
        return "Assemble datasets from fastq files"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data-files", required=True, nargs="+", help="List of paths to data files")
        parser.add_argument("--output-file", required=True, help="Path to save the assembled dataset")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            assemble_datasets(args)
            return 0
        except MetaQuestError as e:
            self.logger.error(f"Error assembling datasets: {e}")
            return 1
