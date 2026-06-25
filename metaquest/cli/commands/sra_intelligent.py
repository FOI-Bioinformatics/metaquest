"""
Intelligent SRA CLI commands for MetaQuest.

This module provides next-generation SRA capabilities including intelligent download
management with resume capability, comprehensive quality analysis, and interactive
reporting dashboards.
"""

import logging
import json
from pathlib import Path
from typing import List

from metaquest.cli.base import BaseCommand
from metaquest.sra import (
    IntelligentDownloadManager,
    SRADatasetAnalyzer,
    SRAReportGenerator,
    DownloadSession,
    QualityProfile,
)

logger = logging.getLogger(__name__)


class SRAIntelligentDownloadCommand(BaseCommand):
    """Command for intelligent SRA downloading with resume capability."""

    @property
    def name(self) -> str:
        return "sra-download-intelligent"

    @property
    def help(self) -> str:
        return "Download SRA datasets with intelligent resume capability and optimization"

    def configure_parser(self, parser):
        parser.add_argument(
            "--accessions-file",
            required=True,
            help="File containing SRA accessions, one per line",
        )
        parser.add_argument(
            "--output-dir",
            default="sra_downloads",
            help="Directory for downloaded files",
        )
        parser.add_argument(
            "--temp-dir",
            help="Directory for temporary files",
        )
        parser.add_argument(
            "--checkpoint-dir",
            help="Directory for download checkpoints",
        )
        parser.add_argument(
            "--max-bandwidth-mbps",
            type=float,
            help="Maximum bandwidth limit in Mbps",
        )
        parser.add_argument(
            "--max-parallel-downloads",
            type=int,
            default=4,
            help="Maximum number of parallel downloads",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            default=True,
            help="Enable resume capability (default: enabled)",
        )
        parser.add_argument(
            "--no-resume",
            action="store_true",
            help="Disable resume capability",
        )
        parser.add_argument(
            "--force-restart",
            action="store_true",
            help="Force restart from beginning (ignore existing checkpoints)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show download plan without executing",
        )
        parser.add_argument(
            "--progress-report",
            default="download_progress.json",
            help="File to save download progress report",
        )

    def _read_accessions(self, filename: str) -> List[str]:
        """Read accessions from file."""
        try:
            with open(filename, "r") as f:
                accessions = [line.strip() for line in f if line.strip()]
            if not accessions:
                print(f"No accessions found in {filename}")
                return []
            return accessions
        except FileNotFoundError:
            print(f"Accessions file not found: {filename}")
            return []

    def _print_download_estimate(self, manager: IntelligentDownloadManager, accessions: List[str]):
        """Print download time estimates."""
        try:
            estimate = manager.estimate_download_time(accessions)
            print("\nDownload Estimate:")
            print("=================")
            print(f"Total datasets: {len(accessions)}")
            print(f"Total size: {estimate['total_size_mb'] / 1024:.2f} GB")
            print(f"Estimated time: {estimate['estimated_time_formatted']}")
            print(f"Network bandwidth: {estimate['network_bandwidth_mbps']:.1f} Mbps")
            print(f"Optimal parallelization: {estimate['optimal_parallel_downloads']}")
        except Exception as e:
            logger.warning(f"Could not generate estimate: {e}")

    def _print_session_summary(self, session: DownloadSession):
        """Print download session summary."""
        print("\nDownload Session Summary:")
        print("========================")
        print(f"Session ID: {session.session_id}")
        print(f"Started: {session.start_time}")
        print(f"Total datasets: {len(session.accessions)}")
        print(f"Completed: {session.success_count}")
        print(f"Failed: {session.failure_count}")

        if session.end_time:
            print(f"Finished: {session.end_time}")
            duration = session.end_time - session.start_time
            print(f"Duration: {duration}")

        if session.network_conditions:
            print("\nThroughput Statistics:")
            print(f"  Measured bandwidth: {session.network_conditions.bandwidth_mbps:.1f} Mbps")
            print(f"  Average speed: {session.average_speed_mbps:.1f} MB/min")

    def execute(self, args):
        try:
            accessions = self._read_accessions(args.accessions_file)
            if not accessions:
                return 1

            # Configure resume setting
            resume_enabled = args.resume and not args.no_resume

            print("Initializing intelligent download manager...")
            print(f"Resume capability: {'enabled' if resume_enabled else 'disabled'}")

            manager = IntelligentDownloadManager(
                output_dir=args.output_dir,
                temp_dir=args.temp_dir,
                checkpoint_dir=args.checkpoint_dir,
                max_bandwidth_mbps=args.max_bandwidth_mbps,
                max_parallel_downloads=args.max_parallel_downloads,
                resume_enabled=resume_enabled,
            )

            # Show download estimate
            self._print_download_estimate(manager, accessions)

            if args.dry_run:
                print("\n🔍 Dry run mode - no files will be downloaded")
                return 0

            print(f"\nStarting intelligent download of {len(accessions)} datasets...")
            print("Press Ctrl+C to safely interrupt and save progress")

            # Start download with resume capability
            session = manager.download_with_resume(accessions, args.force_restart)

            # Print results
            self._print_session_summary(session)

            # Save progress report
            failed_accessions = [
                acc for acc, result in session.download_results.items() if result.status == "failed"
            ]
            progress_file = Path(args.progress_report)
            with open(progress_file, "w") as progress_f:
                json.dump(
                    {
                        "session_id": session.session_id,
                        "start_time": session.start_time.isoformat(),
                        "end_time": session.end_time.isoformat() if session.end_time else None,
                        "total_datasets": len(session.accessions),
                        "completed_downloads": session.success_count,
                        "failed_downloads": session.failure_count,
                        "failed_accessions": failed_accessions,
                        "average_speed_mbps": session.average_speed_mbps,
                        "network_bandwidth_mbps": (
                            session.network_conditions.bandwidth_mbps if session.network_conditions else None
                        ),
                    },
                    progress_f,
                    indent=2,
                )

            print(f"\nProgress report saved to: {progress_file}")

            if session.failure_count > 0:
                print(f"\n⚠️  {session.failure_count} downloads failed")
                return 1

            print("\n✅ All downloads completed successfully!")
            return 0

        except KeyboardInterrupt:
            print("\n\n⏸️  Download interrupted by user")
            print("Progress has been saved - use --resume to continue")
            return 2
        except Exception as e:
            logger.error(f"Intelligent download failed: {e}")
            return 1


class SRAQualityProfileCommand(BaseCommand):
    """Command for comprehensive SRA dataset quality profiling."""

    @property
    def name(self) -> str:
        return "sra-profile-quality"

    @property
    def help(self) -> str:
        return "Generate comprehensive quality profiles for SRA datasets"

    def configure_parser(self, parser):
        parser.add_argument(
            "--accessions-file",
            required=True,
            help="File containing SRA accessions, one per line",
        )
        parser.add_argument(
            "--fastq-dir",
            default="sra_downloads",
            help="Directory containing downloaded FASTQ files",
        )
        parser.add_argument(
            "--output-dir",
            default="sra_quality_profiles",
            help="Directory for quality profile outputs",
        )
        parser.add_argument(
            "--accession",
            help="Profile single accession instead of batch",
        )
        parser.add_argument(
            "--detailed-reports",
            action="store_true",
            help="Generate detailed per-accession reports",
        )
        parser.add_argument(
            "--include-contamination",
            action="store_true",
            help="Include contamination detection analysis",
        )
        parser.add_argument(
            "--summary-only",
            action="store_true",
            help="Generate only summary statistics",
        )

    def _read_accessions(self, filename: str) -> List[str]:
        """Read accessions from file."""
        try:
            with open(filename, "r") as f:
                accessions = [line.strip() for line in f if line.strip()]
            return accessions
        except FileNotFoundError:
            print(f"Accessions file not found: {filename}")
            return []

    def _print_quality_profile(self, profile: QualityProfile):
        """Print quality profile summary."""
        print(f"\nQuality Profile: {profile.accession}")
        print("=" * 50)
        print(f"Total reads: {profile.total_reads:,}")
        print(f"Total bases: {profile.total_bases:,}")
        print(f"Average read length: {profile.avg_read_length:.1f}")
        print(f"GC content: {profile.gc_content:.1%}")
        print(f"Quality grade: {profile.quality_grade}")
        print(f"Sequence complexity: {profile.complexity_score:.3f}")

        if profile.n_content > 0.01:  # > 1%
            print(f"⚠️  High N content: {profile.n_content:.1%}")

        if profile.duplication_rate is not None and profile.duplication_rate > 0.20:  # > 20%
            print(f"⚠️  High duplicate rate: {profile.duplication_rate:.1%}")

        adapter_contamination = profile.contamination_indicators.get("adapter_contamination", 0)
        if adapter_contamination > 0.05:  # > 5%
            print(f"⚠️  Adapter contamination: {adapter_contamination:.1%}")

    def execute(self, args):
        try:
            # Setup
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            analyzer = SRADatasetAnalyzer()

            if args.accession:
                # Single accession mode
                accessions = [args.accession]
                print(f"Profiling single accession: {args.accession}")
            else:
                # Batch mode
                accessions = self._read_accessions(args.accessions_file)
                if not accessions:
                    return 1
                print(f"Profiling {len(accessions)} accessions...")

            profiles = []
            failed_accessions = []

            for i, accession in enumerate(accessions, 1):
                print(f"[{i}/{len(accessions)}] Analyzing {accession}...")

                try:
                    # Find FASTQ files for this accession
                    fastq_dir = Path(args.fastq_dir)
                    accession_files = list(fastq_dir.glob(f"**/{accession}*.fastq*"))

                    if not accession_files:
                        print(f"⚠️  No FASTQ files found for {accession}")
                        failed_accessions.append(accession)
                        continue

                    # Profile the dataset
                    profile = analyzer.profile_dataset_quality(accession, fastq_path=str(accession_files[0]))

                    profiles.append(profile)

                    if not args.summary_only:
                        self._print_quality_profile(profile)

                    # Save detailed report if requested
                    if args.detailed_reports:
                        profile_file = output_dir / f"{accession}_quality_profile.json"
                        with open(profile_file, "w") as profile_f:
                            json.dump(
                                {
                                    "accession": profile.accession,
                                    "total_reads": profile.total_reads,
                                    "total_bases": profile.total_bases,
                                    "avg_read_length": profile.avg_read_length,
                                    "gc_content": profile.gc_content,
                                    "quality_grade": profile.quality_grade,
                                    "quality_distribution": profile.quality_distribution,
                                    "complexity_score": profile.complexity_score,
                                    "n_content": profile.n_content,
                                    "duplication_rate": profile.duplication_rate,
                                    "contamination_indicators": profile.contamination_indicators,
                                    "recommendations": profile.recommendations,
                                },
                                profile_f,
                                indent=2,
                            )
                        print(f"  Detailed report saved: {profile_file}")

                except Exception as e:
                    logger.warning(f"Failed to profile {accession}: {e}")
                    failed_accessions.append(accession)
                    continue

            # Generate summary statistics
            if profiles:
                print(f"\nSummary Statistics ({len(profiles)} datasets):")
                print("=" * 50)

                total_reads = sum(p.total_reads for p in profiles)
                total_bases = sum(p.total_bases for p in profiles)
                avg_gc = sum(p.gc_content for p in profiles) / len(profiles)

                print(f"Total reads across all datasets: {total_reads:,}")
                print(f"Total bases across all datasets: {total_bases:,}")
                print(f"Average GC content: {avg_gc:.1%}")

                # Quality flags
                high_n_content = len([p for p in profiles if p.n_content > 0.01])
                high_duplicates = len(
                    [p for p in profiles if p.duplication_rate is not None and p.duplication_rate > 0.20]
                )
                contaminated = len(
                    [p for p in profiles if p.contamination_indicators.get("adapter_contamination", 0) > 0.05]
                )

                if high_n_content:
                    print(f"⚠️  {high_n_content} datasets with high N content")
                if high_duplicates:
                    print(f"⚠️  {high_duplicates} datasets with high duplicate rates")
                if contaminated:
                    print(f"⚠️  {contaminated} datasets with adapter contamination")

            # Save batch summary
            summary_file = output_dir / "quality_summary.json"
            with open(summary_file, "w") as summary_f:
                json.dump(
                    {
                        "total_analyzed": len(profiles),
                        "total_failed": len(failed_accessions),
                        "failed_accessions": failed_accessions,
                        "summary_stats": (
                            {
                                "total_reads": total_reads,
                                "total_bases": total_bases,
                                "avg_gc_content": avg_gc,
                            }
                            if profiles
                            else None
                        ),
                    },
                    summary_f,
                    indent=2,
                )

            print("\nQuality analysis complete!")
            print(f"Summary saved to: {summary_file}")

            if failed_accessions:
                print(f"⚠️  {len(failed_accessions)} accessions failed analysis")
                return 1

            return 0

        except Exception as e:
            logger.error(f"Quality profiling failed: {e}")
            return 1


class SRAInteractiveDashboardCommand(BaseCommand):
    """Command for generating interactive SRA analysis dashboards."""

    @property
    def name(self) -> str:
        return "sra-dashboard"

    @property
    def help(self) -> str:
        return "Generate interactive HTML dashboards for SRA analysis"

    def configure_parser(self, parser):
        parser.add_argument(
            "--accessions-file",
            required=True,
            help="File containing SRA accessions, one per line",
        )
        parser.add_argument(
            "--download-session",
            help="JSON file from intelligent download session",
        )
        parser.add_argument(
            "--quality-profiles",
            help="Directory containing quality profile JSONs",
        )
        parser.add_argument(
            "--output-dir",
            default="sra_dashboards",
            help="Directory for dashboard outputs",
        )
        parser.add_argument(
            "--title",
            default="SRA Analysis Dashboard",
            help="Dashboard title",
        )
        parser.add_argument(
            "--dashboard-type",
            choices=["download", "quality", "comparative", "full"],
            default="full",
            help="Type of dashboard to generate",
        )

    def _read_accessions(self, filename: str) -> List[str]:
        """Read accessions from file."""
        try:
            with open(filename, "r") as f:
                accessions = [line.strip() for line in f if line.strip()]
            return accessions
        except FileNotFoundError:
            print(f"Accessions file not found: {filename}")
            return []

    def execute(self, args):
        try:
            # Setup
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            accessions = self._read_accessions(args.accessions_file)
            if not accessions:
                return 1

            reporter = SRAReportGenerator(output_dir=str(output_dir))

            print(f"Generating {args.dashboard_type} dashboard for {len(accessions)} accessions...")

            dashboard_path = None

            if args.dashboard_type in ["download", "full"]:
                # Download summary dashboard
                if args.download_session:
                    try:
                        with open(args.download_session, "r") as session_f:
                            json.load(session_f)  # Load but don't store (not used yet)
                        print("Creating download summary dashboard...")
                        # Note: This would need to reconstruct DownloadSession object
                        # For now, we'll create a quality dashboard
                        dashboard_path = reporter.generate_quality_dashboard(
                            accessions, title=f"{args.title} - Download Summary"
                        )
                    except Exception as e:
                        logger.warning(f"Could not load download session: {e}")

            if args.dashboard_type in ["quality", "full"]:
                # Quality analysis dashboard
                print("Creating quality analysis dashboard...")
                dashboard_path = reporter.generate_quality_dashboard(
                    accessions, title=f"{args.title} - Quality Analysis"
                )

            if args.dashboard_type in ["comparative", "full"]:
                # Comparative analysis dashboard
                print("Creating comparative analysis dashboard...")
                # Group accessions by some criteria (could be enhanced later)
                groups = {"All Datasets": accessions}
                dashboard_path = reporter.create_comparative_analysis(
                    groups, title=f"{args.title} - Comparative Analysis"
                )

            if dashboard_path:
                print("\n✅ Dashboard generated successfully!")
                print(f"Open in browser: {dashboard_path}")

                # Try to open in browser
                try:
                    import webbrowser

                    webbrowser.open(f"file://{dashboard_path.absolute()}")
                    print("Dashboard opened in browser")
                except Exception:
                    pass

                return 0
            else:
                print("No dashboard was generated")
                return 1

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return 1


class SRAComparativeAnalysisCommand(BaseCommand):
    """Command for comparative analysis of SRA datasets."""

    @property
    def name(self) -> str:
        return "sra-compare"

    @property
    def help(self) -> str:
        return "Perform comparative analysis between SRA dataset groups"

    def configure_parser(self, parser):
        parser.add_argument(
            "--groups-file",
            required=True,
            help="JSON file defining accession groups for comparison",
        )
        parser.add_argument(
            "--fastq-dir",
            default="sra_downloads",
            help="Directory containing downloaded FASTQ files",
        )
        parser.add_argument(
            "--output-dir",
            default="sra_comparative_analysis",
            help="Directory for analysis outputs",
        )
        parser.add_argument(
            "--statistical-tests",
            action="store_true",
            help="Include statistical significance tests",
        )
        parser.add_argument(
            "--generate-report",
            action="store_true",
            default=True,
            help="Generate HTML comparative analysis report",
        )

    def _load_groups(self, filename: str) -> dict:
        """Load accession groups from JSON file."""
        try:
            with open(filename, "r") as f:
                groups = json.load(f)
            return groups
        except FileNotFoundError:
            print(f"Groups file not found: {filename}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in groups file: {e}")
            return {}

    def execute(self, args):
        try:
            # Load groups
            groups = self._load_groups(args.groups_file)
            if not groups:
                print("Example groups file format:")
                print(
                    json.dumps({"Group_A": ["SRR123456", "SRR123457"], "Group_B": ["SRR789012", "SRR789013"]}, indent=2)
                )
                return 1

            print(f"Loaded {len(groups)} groups for comparison:")
            for name, accessions in groups.items():
                print(f"  {name}: {len(accessions)} accessions")

            # Setup
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            analyzer = SRADatasetAnalyzer()

            print("\nPerforming comparative analysis...")

            # Run comparative analysis
            comparison = analyzer.compare_datasets(groups)

            # Print summary results
            print("\nComparative Analysis Results:")
            print("=" * 40)

            for group_name, col_stats in comparison.summary_statistics.items():
                print(f"\n{group_name}:")
                print(f"  Datasets: {len(groups.get(group_name, []))}")
                gc_stats = col_stats.get("gc_content")
                if gc_stats:
                    print(f"  Avg GC content: {gc_stats['mean']:.1%}")
                length_stats = col_stats.get("avg_read_length")
                if length_stats:
                    print(f"  Avg read length: {length_stats['mean']:.1f}")
                reads_stats = col_stats.get("total_reads")
                if reads_stats:
                    print(f"  Mean total reads: {reads_stats['mean']:,.0f}")

            # Statistical tests results
            if args.statistical_tests and comparison.statistical_tests:
                print("\nStatistical Tests:")
                print("-" * 20)
                for test_name, result in comparison.statistical_tests.items():
                    p_value = result.get("p_value", 1.0)
                    significant = p_value < 0.05
                    print(f"{test_name}: p={p_value:.4f} {'*' if significant else ''}")

            # Save detailed results
            significant_differences = [
                col for col, test in comparison.statistical_tests.items() if test.get("significant")
            ]
            results_file = output_dir / "comparative_analysis.json"
            with open(results_file, "w") as results_f:
                json.dump(
                    {
                        "groups": groups,
                        "summary_statistics": comparison.summary_statistics,
                        "statistical_tests": comparison.statistical_tests,
                        "significant_differences": significant_differences,
                        "outlier_datasets": comparison.outlier_datasets,
                    },
                    results_f,
                    indent=2,
                )

            print(f"\nDetailed results saved to: {results_file}")

            # Generate HTML report
            if args.generate_report:
                reporter = SRAReportGenerator(output_dir=str(output_dir))
                report_path = reporter.create_comparative_analysis(groups, title="SRA Comparative Analysis Report")
                print(f"Interactive report generated: {report_path}")

            return 0

        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            return 1
