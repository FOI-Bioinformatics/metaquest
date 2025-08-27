"""
Enhanced SRA CLI commands for MetaQuest.

This module provides comprehensive SRA downloading commands with technology detection,
metadata fetching, and detailed statistics reporting.
"""

import logging
from pathlib import Path

from metaquest.cli.base import BaseCommand
from metaquest.data.sra_enhanced import (
    EnhancedSRADownloader,
    verify_sra_tools,
    estimate_download_time,
    create_download_report,
)
from metaquest.data.sra_metadata import (
    save_metadata_report,
    generate_statistics_report,
)

logger = logging.getLogger(__name__)


class SRAInfoCommand(BaseCommand):
    """Command for getting SRA dataset information before downloading."""

    @property
    def name(self) -> str:
        return "sra_info"

    @property
    def help(self) -> str:
        return "Get detailed information about SRA datasets before downloading"

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

    def execute(self, args):
        try:
            # Read accessions
            with open(args.accessions_file, "r") as f:
                accessions = [line.strip() for line in f if line.strip()]

            if not accessions:
                print("No accessions found in file")
                return 1

            print(f"Analyzing {len(accessions)} SRA accessions...")

            # Initialize downloader
            downloader = EnhancedSRADownloader(args.email, args.api_key)

            # Get preview
            metadata, tech_counts, total_size_gb = downloader.preview_downloads(
                accessions
            )

            if not metadata:
                print("Could not fetch metadata for any accessions")
                return 1

            # Print summary
            print("\nSRA Dataset Analysis:")
            print("===================")
            print(f"Total accessions: {len(accessions)}")
            print(f"Metadata fetched: {len(metadata)}")
            print(f"Total estimated size: {total_size_gb:.2f} GB")

            # Technology breakdown
            if tech_counts:
                print("\nTechnology distribution:")
                for tech, count in tech_counts.items():
                    print(f"  {tech}: {count} datasets")

            # Platform breakdown
            platforms = {}
            layouts = {}
            for acc, info in metadata.items():
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

            # Size statistics
            sizes = [info.size_mb / 1024 for info in metadata.values()]  # Convert to GB
            if sizes:
                print("\nSize statistics:")
                print(f"  Average size per dataset: {sum(sizes)/len(sizes):.2f} GB")
                print(f"  Largest dataset: {max(sizes):.2f} GB")
                print(f"  Smallest dataset: {min(sizes):.2f} GB")

            # Download time estimation
            estimated_hours = estimate_download_time(
                total_size_gb, args.bandwidth_mbps, 4
            )
            if estimated_hours < 1:
                print(f"  Estimated download time: {estimated_hours*60:.0f} minutes")
            else:
                print(f"  Estimated download time: {estimated_hours:.1f} hours")

            # Save detailed report
            save_metadata_report(metadata, args.output_report)
            print(f"\nDetailed report saved to: {args.output_report}")

            return 0

        except Exception as e:
            logger.error(f"SRA info command failed: {e}")
            return 1


class SRADownloadEnhancedCommand(BaseCommand):
    """Command for enhanced SRA downloading with technology detection."""

    @property
    def name(self) -> str:
        return "sra_download"

    @property
    def help(self) -> str:
        return "Download SRA datasets with enhanced features and technology detection"

    def configure_parser(self, parser):
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
            "--email",
            required=True,
            help="Email address for NCBI API access (required by NCBI)",
        )
        parser.add_argument(
            "--api-key",
            help="NCBI API key for increased rate limits (optional)",
        )
        parser.add_argument(
            "--max-downloads",
            type=int,
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
            help="Number of parallel downloads",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be downloaded without downloading",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force redownload even if files exist",
        )
        parser.add_argument(
            "--temp-folder",
            help="Directory for temporary files (must be writable)",
        )
        parser.add_argument(
            "--blacklist",
            nargs="+",
            help="Files containing blacklisted accessions",
        )
        parser.add_argument(
            "--verify-tools",
            action="store_true",
            help="Verify SRA tools are installed before starting",
        )
        parser.add_argument(
            "--report-file",
            default="download_report.csv",
            help="Output file for download report",
        )

    def _verify_tools(self, args):
        """Verify SRA tools if requested."""
        if args.verify_tools:
            print("Verifying SRA tools...")
            if not verify_sra_tools():
                print("SRA tools verification failed. Please install SRA toolkit.")
                return False
            print("✓ SRA tools verified")
        return True

    def _read_accessions(self, filename):
        """Read accessions from file."""
        with open(filename, "r") as f:
            accessions = [line.strip() for line in f if line.strip()]
        if not accessions:
            print("No accessions found in file")
            return None
        return accessions

    def _read_blacklist(self, blacklist_files):
        """Read blacklisted accessions from files."""
        blacklisted = set()
        if blacklist_files:
            for blacklist_file in blacklist_files:
                try:
                    with open(blacklist_file, "r") as f:
                        file_accessions = {line.strip() for line in f if line.strip()}
                        blacklisted.update(file_accessions)
                    logger.info(
                        f"Read {len(file_accessions)} blacklisted accessions "
                        f"from {blacklist_file}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error reading blacklist file {blacklist_file}: {e}"
                    )
        return blacklisted

    def _handle_dry_run(self, downloader, accessions):
        """Handle dry run mode."""
        print("Dry run mode: analyzing datasets...")
        metadata, tech_counts, total_size_gb = downloader.preview_downloads(accessions)

        print("\nWould download:")
        print(f"  Accessions: {len(accessions)}")
        print(f"  Total size: {total_size_gb:.2f} GB")

        if tech_counts:
            print("  Technologies:")
            for tech, count in tech_counts.items():
                print(f"    {tech}: {count}")

    def _print_results(self, results):
        """Print download results summary."""
        print("\nDownload Summary:")
        print("================")
        print(f"Total: {results['total']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")

        if results["technology_summary"]:
            print("\nTechnology breakdown:")
            for tech, count in results["technology_summary"].items():
                print(f"  {tech}: {count}")

    def execute(self, args):
        try:
            if not self._verify_tools(args):
                return 1

            accessions = self._read_accessions(args.accessions_file)
            if accessions is None:
                return 1

            blacklisted = self._read_blacklist(args.blacklist)

            downloader = EnhancedSRADownloader(
                args.email,
                args.api_key,
                args.num_threads,
                args.max_workers,
                args.temp_folder,
            )

            if args.dry_run:
                self._handle_dry_run(downloader, accessions)
                return 0

            print(f"Starting enhanced download of {len(accessions)} datasets...")
            if blacklisted:
                print(f"Blacklisted accessions: {len(blacklisted)}")

            results = downloader.download_batch_enhanced(
                accessions,
                args.fastq_folder,
                args.force,
                args.max_downloads,
                blacklisted,
            )

            self._print_results(results)

            create_download_report(results, args.report_file)
            print(f"\nDetailed report saved to: {args.report_file}")

            if results["failed_accessions"]:
                failed_file = Path(args.fastq_folder) / "failed_accessions.txt"
                with open(failed_file, "w") as f:
                    for acc in results["failed_accessions"]:
                        f.write(f"{acc}\n")
                print(f"Failed accessions saved to: {failed_file}")

            return 0 if results["failed"] == 0 else 1

        except Exception as e:
            logger.error(f"Enhanced SRA download failed: {e}")
            return 1


class SRAStatsCommand(BaseCommand):
    """Command for calculating comprehensive statistics on downloaded SRA data."""

    @property
    def name(self) -> str:
        return "sra_stats"

    @property
    def help(self) -> str:
        return "Calculate comprehensive statistics for downloaded SRA datasets"

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

    def _find_accession_dirs(self, fastq_folder, specific_accessions=None):
        """Find accession directories to validate."""
        accession_dirs = [d for d in fastq_folder.iterdir() if d.is_dir()]
        if specific_accessions:
            accession_dirs = [
                d for d in accession_dirs if d.name in specific_accessions
            ]
        return accession_dirs

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

        issues = []

        # Check file sizes
        for f in fastq_files:
            if f.stat().st_size == 0:
                issues.append(f"Empty file: {f.name}")

        # Check paired-end consistency (simplified)
        if check_pairs and len(fastq_files) == 2:
            # Basic check - could be enhanced
            pass

        # Basic FASTQ format validation
        try:
            from Bio import SeqIO

            for f in fastq_files[:1]:  # Check first file only for speed
                with open(f, "rt") as handle:
                    records = list(SeqIO.parse(handle, "fastq"))
                    if len(records) == 0:
                        issues.append(f"No valid FASTQ records in {f.name}")
                    break
        except Exception as e:
            issues.append(f"FASTQ format error: {e}")

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

            # Find accession directories
            accession_dirs = [d for d in fastq_folder.iterdir() if d.is_dir()]

            if args.accessions:
                accession_dirs = [
                    d for d in accession_dirs if d.name in args.accessions
                ]

            if not accession_dirs:
                print("No accession directories found")
                return 1

            validation_results = []

            for acc_dir in accession_dirs:
                print(f"Validating {acc_dir.name}...")

                # Find FASTQ files
                fastq_files = list(acc_dir.glob("*.fastq*"))

                if not fastq_files:
                    validation_results.append(
                        {
                            "accession": acc_dir.name,
                            "status": "FAILED",
                            "issue": "No FASTQ files found",
                        }
                    )
                    continue

                # Basic validation
                issues = []

                # Check file sizes
                for f in fastq_files:
                    if f.stat().st_size == 0:
                        issues.append(f"Empty file: {f.name}")

                # Check paired-end consistency
                if args.check_pairs:
                    r1_files = [
                        f for f in fastq_files if "_R1" in f.name or "_1" in f.name
                    ]
                    r2_files = [
                        f for f in fastq_files if "_R2" in f.name or "_2" in f.name
                    ]

                    if len(r1_files) != len(r2_files) and len(r2_files) > 0:
                        issues.append("Mismatched paired-end files")

                # Basic FASTQ format validation
                try:
                    from Bio import SeqIO

                    for f in fastq_files[:1]:  # Check first file only for speed
                        with open(f, "rt") as handle:
                            records = list(SeqIO.parse(handle, "fastq"))
                            if len(records) == 0:
                                issues.append(f"No valid FASTQ records in {f.name}")
                            break
                except Exception as e:
                    issues.append(f"FASTQ format error: {e}")

                validation_results.append(
                    {
                        "accession": acc_dir.name,
                        "status": "PASSED" if not issues else "FAILED",
                        "issues": "; ".join(issues) if issues else "None",
                        "num_files": len(fastq_files),
                    }
                )

            # Print results
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

            return 0 if len(failed) == 0 else 1

        except Exception as e:
            logger.error(f"SRA validation failed: {e}")
            return 1
