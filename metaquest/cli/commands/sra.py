"""
SRA-related CLI commands.
"""

import argparse

from metaquest.cli.base import BaseCommand
from metaquest.cli.commands_legacy import (
    download_sra_command,
    assemble_datasets_command,
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
        parser.add_argument(
            "--force", action="store_true", help="Force redownload even if files exist"
        )
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
        return download_sra_command(args)


class AssembleDatasetsCommand(BaseCommand):
    """Command for assembling datasets from fastq files."""

    @property
    def name(self) -> str:
        return "assemble_datasets"

    @property
    def help(self) -> str:
        return "Assemble datasets from fastq files"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data-files", required=True, nargs="+", help="List of paths to data files"
        )
        parser.add_argument(
            "--output-file", required=True, help="Path to save the assembled dataset"
        )

    def execute(self, args: argparse.Namespace) -> int:
        return assemble_datasets_command(args)
