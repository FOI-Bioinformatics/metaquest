"""
Advanced SRA Analytics and Reporting Package for MetaQuest.

This package provides comprehensive dataset quality analysis, statistical
comparison, and interactive HTML reporting for SRA-derived sequencing data.

Main Components:
- analytics: Quality profiling and comparative dataset analysis
- reporting: Interactive HTML reports and dashboards
"""

from .analytics import (
    SRADatasetAnalyzer,
    SequenceQualityAnalyzer,
    QualityProfile,
    ComparativeAnalysis,
    AnomalyReport,
    ProcessingRecommendations,
    load_quality_profiles,
)

from .reporting import SRAReportGenerator

__version__ = "1.0.0"
__author__ = "MetaQuest Development Team"

__all__ = [
    # Analytics
    "SRADatasetAnalyzer",
    "SequenceQualityAnalyzer",
    "QualityProfile",
    "ComparativeAnalysis",
    "AnomalyReport",
    "ProcessingRecommendations",
    "load_quality_profiles",
    # Reporting
    "SRAReportGenerator",
]
