"""
Advanced SRA Download and Analytics Package for MetaQuest.

This package provides next-generation capabilities for SRA data handling including:
- Intelligent download management with resume capability
- Comprehensive dataset quality analysis and profiling
- Advanced statistical reporting and interactive dashboards
- ML-powered optimization and anomaly detection

Main Components:
- download_manager: Intelligent SRA download with resume/optimization
- analytics: Quality profiling and comparative dataset analysis
- reporting: Interactive HTML reports and dashboards

Usage:
    from metaquest.sra import IntelligentDownloadManager, SRADatasetAnalyzer, SRAReportGenerator

    # Download with intelligence
    manager = IntelligentDownloadManager(output_dir="downloads")
    session = manager.download_with_resume(["SRR123456", "SRR789012"])

    # Analyze quality
    analyzer = SRADatasetAnalyzer()
    profile = analyzer.profile_dataset_quality("SRR123456")

    # Generate reports
    reporter = SRAReportGenerator(output_dir="reports")
    report_path = reporter.create_download_summary(session)
"""

from .download_manager import (
    IntelligentDownloadManager,
    BandwidthManager,
    DownloadOptimizer,
    CheckpointManager,
    NetworkConditions,
    DownloadCheckpoint,
    DownloadProgress,
    DownloadSession,
)

from .analytics import (
    SRADatasetAnalyzer,
    SequenceQualityAnalyzer,
    QualityProfile,
    ComparativeAnalysis,
    AnomalyReport,
    ProcessingRecommendations,
)

from .reporting import SRAReportGenerator

__version__ = "1.0.0"
__author__ = "MetaQuest Development Team"

__all__ = [
    # Download Management
    "IntelligentDownloadManager",
    "BandwidthManager",
    "DownloadOptimizer",
    "CheckpointManager",
    "NetworkConditions",
    "DownloadCheckpoint",
    "DownloadProgress",
    "DownloadSession",
    # Analytics
    "SRADatasetAnalyzer",
    "SequenceQualityAnalyzer",
    "QualityProfile",
    "ComparativeAnalysis",
    "AnomalyReport",
    "ProcessingRecommendations",
    # Reporting
    "SRAReportGenerator",
]
