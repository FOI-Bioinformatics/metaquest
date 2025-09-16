"""
Advanced SRA Dataset Analytics and Statistical Reporting.

This module provides comprehensive analysis capabilities for SRA datasets including:
- Quality profiling and contamination detection
- Comparative analysis across multiple datasets
- Technology and temporal trend analysis
- Advanced statistical reporting with visualizations
- Processing parameter recommendations
"""

import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from metaquest.core.exceptions import DataAccessError
from metaquest.data.sra_metadata import SRADatasetInfo, ReadStatistics

logger = logging.getLogger(__name__)


@dataclass
class QualityProfile:
    """Comprehensive quality profile for SRA dataset."""

    accession: str
    total_reads: int
    total_bases: int
    avg_read_length: float
    read_length_distribution: Dict[str, int]  # length_range -> count
    gc_content: float
    gc_distribution: List[float]  # per-read GC content
    quality_distribution: Dict[str, float]  # quality_range -> percentage
    n_content: float
    contamination_indicators: Dict[str, float]
    complexity_score: float
    duplication_rate: Optional[float]
    technology_confidence: float
    quality_grade: str  # 'excellent', 'good', 'fair', 'poor'
    recommendations: List[str]


@dataclass
class ComparativeAnalysis:
    """Results from comparative analysis across datasets."""

    dataset_groups: Dict[str, List[str]]
    summary_statistics: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, Any]]
    outlier_datasets: List[str]
    clustering_results: Optional[Dict[str, Any]]
    batch_effects: Dict[str, float]
    recommendations: List[str]
    visualization_data: Dict[str, Any]


@dataclass
class AnomalyReport:
    """Report of detected anomalies in datasets."""

    anomalous_datasets: List[str]
    anomaly_types: Dict[str, List[str]]  # anomaly_type -> affected_datasets
    severity_scores: Dict[str, float]  # dataset -> severity (0-1)
    explanations: Dict[str, str]  # dataset -> explanation
    recommended_actions: Dict[str, List[str]]


@dataclass
class ProcessingRecommendations:
    """Recommended processing parameters based on dataset characteristics."""

    accession: str
    recommended_pipeline: str
    quality_trimming: Dict[str, Any]
    adapter_removal: Dict[str, Any]
    contamination_filtering: Dict[str, Any]
    assembly_parameters: Dict[str, Any]
    expected_coverage: Optional[float]
    computational_requirements: Dict[str, Any]
    estimated_processing_time: str


class SequenceQualityAnalyzer:
    """Analyzes sequence quality metrics from FASTQ files."""

    def __init__(self):
        self.quality_encodings = {"sanger": 33, "illumina_1.3": 64, "illumina_1.5": 64, "solexa": 64}

    def analyze_fastq_quality(self, fastq_path: Union[str, Path], sample_size: int = 10000) -> Dict[str, Any]:
        """
        Analyze quality metrics from FASTQ file.

        Args:
            fastq_path: Path to FASTQ file
            sample_size: Number of reads to sample for analysis

        Returns:
            Dictionary of quality metrics
        """
        fastq_path = Path(fastq_path)
        if not fastq_path.exists():
            raise DataAccessError(f"FASTQ file not found: {fastq_path}")

        read_lengths = []
        gc_contents = []
        quality_scores = []
        n_contents = []
        sequences = []

        try:
            # Sample reads for analysis
            with open(fastq_path, "r") as handle:
                if fastq_path.suffix.endswith(".gz"):
                    import gzip

                    handle = gzip.open(fastq_path, "rt")

                read_count = 0
                for record in SeqIO.parse(handle, "fastq"):
                    if read_count >= sample_size:
                        break

                    sequence = str(record.seq)
                    qualities = record.letter_annotations["phred_quality"]

                    # Basic metrics
                    read_lengths.append(len(sequence))
                    gc_contents.append(self._calculate_gc_content(sequence))
                    quality_scores.extend(qualities)
                    n_contents.append(sequence.count("N") / len(sequence))
                    sequences.append(sequence)

                    read_count += 1

        except Exception as e:
            logger.error(f"Error analyzing FASTQ file {fastq_path}: {e}")
            raise DataAccessError(f"Failed to analyze FASTQ file: {e}")

        if not read_lengths:
            raise DataAccessError("No valid reads found in FASTQ file")

        # Calculate comprehensive statistics
        return {
            "total_reads_sampled": len(read_lengths),
            "read_length_stats": {
                "mean": statistics.mean(read_lengths),
                "median": statistics.median(read_lengths),
                "std": statistics.stdev(read_lengths) if len(read_lengths) > 1 else 0,
                "min": min(read_lengths),
                "max": max(read_lengths),
                "distribution": self._get_length_distribution(read_lengths),
            },
            "gc_content_stats": {
                "mean": statistics.mean(gc_contents),
                "median": statistics.median(gc_contents),
                "std": statistics.stdev(gc_contents) if len(gc_contents) > 1 else 0,
                "distribution": gc_contents,
            },
            "quality_stats": {
                "mean": statistics.mean(quality_scores),
                "median": statistics.median(quality_scores),
                "q25": np.percentile(quality_scores, 25),
                "q75": np.percentile(quality_scores, 75),
                "distribution": self._get_quality_distribution(quality_scores),
            },
            "n_content_stats": {
                "mean": statistics.mean(n_contents),
                "max": max(n_contents),
                "high_n_reads": sum(1 for n in n_contents if n > 0.05),
            },
            "complexity_metrics": self._analyze_sequence_complexity(sequences),
            "contamination_indicators": self._detect_contamination_indicators(sequences),
        }

    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of sequence."""
        gc_count = sequence.count("G") + sequence.count("C")
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0

    def _get_length_distribution(self, lengths: List[int]) -> Dict[str, int]:
        """Get read length distribution in ranges."""
        distribution = {"0-50": 0, "51-100": 0, "101-150": 0, "151-250": 0, "251-500": 0, "501-1000": 0, "1000+": 0}

        for length in lengths:
            if length <= 50:
                distribution["0-50"] += 1
            elif length <= 100:
                distribution["51-100"] += 1
            elif length <= 150:
                distribution["101-150"] += 1
            elif length <= 250:
                distribution["151-250"] += 1
            elif length <= 500:
                distribution["251-500"] += 1
            elif length <= 1000:
                distribution["501-1000"] += 1
            else:
                distribution["1000+"] += 1

        return distribution

    def _get_quality_distribution(self, quality_scores: List[int]) -> Dict[str, float]:
        """Get quality score distribution."""
        total = len(quality_scores)
        if total == 0:
            return {}

        distribution = {
            "excellent_q30+": sum(1 for q in quality_scores if q >= 30) / total,
            "good_q20-29": sum(1 for q in quality_scores if 20 <= q < 30) / total,
            "fair_q10-19": sum(1 for q in quality_scores if 10 <= q < 20) / total,
            "poor_q0-9": sum(1 for q in quality_scores if q < 10) / total,
        }

        return distribution

    def _analyze_sequence_complexity(self, sequences: List[str]) -> Dict[str, float]:
        """Analyze sequence complexity indicators."""
        if not sequences:
            return {}

        # Calculate various complexity metrics
        kmer_diversities = []
        homopolymer_rates = []

        for seq in sequences[:1000]:  # Sample first 1000 sequences
            # K-mer diversity (using 3-mers)
            kmers = [seq[i : i + 3] for i in range(len(seq) - 2)]
            unique_kmers = len(set(kmers))
            total_kmers = len(kmers)
            kmer_diversities.append(unique_kmers / total_kmers if total_kmers > 0 else 0)

            # Homopolymer run detection
            homopolymer_count = 0
            current_run = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i - 1]:
                    current_run += 1
                else:
                    if current_run >= 4:  # Runs of 4+ same base
                        homopolymer_count += 1
                    current_run = 1
            homopolymer_rates.append(homopolymer_count / len(seq) if len(seq) > 0 else 0)

        return {
            "kmer_diversity_mean": statistics.mean(kmer_diversities) if kmer_diversities else 0,
            "homopolymer_rate_mean": statistics.mean(homopolymer_rates) if homopolymer_rates else 0,
            "complexity_score": statistics.mean(kmer_diversities) if kmer_diversities else 0,
        }

    def _detect_contamination_indicators(self, sequences: List[str]) -> Dict[str, float]:
        """Detect potential contamination indicators."""
        # Simple contamination indicators
        indicators = {"adapter_contamination": 0.0, "vector_contamination": 0.0, "overrepresented_sequences": 0.0}

        if not sequences:
            return indicators

        # Common adapter sequences
        adapters = [
            "AGATCGGAAGAGC",  # Illumina universal adapter
            "CTGTCTCTTATACACATCT",  # Illumina adapter
            "AATGATACGGCGACCACCGAGATCTACAC",  # Illumina P5
        ]

        adapter_hits = 0
        for seq in sequences:
            for adapter in adapters:
                if adapter in seq:
                    adapter_hits += 1
                    break

        indicators["adapter_contamination"] = adapter_hits / len(sequences)

        # Check for overrepresented sequences
        sequence_counts = Counter(sequences)
        most_common = sequence_counts.most_common(10)
        if most_common:
            max_count = most_common[0][1]
            indicators["overrepresented_sequences"] = max_count / len(sequences)

        return indicators


class SRADatasetAnalyzer:
    """Main analyzer for comprehensive SRA dataset analysis."""

    def __init__(self):
        self.quality_analyzer = SequenceQualityAnalyzer()

    def profile_dataset_quality(
        self, accession: str, fastq_path: Optional[Union[str, Path]] = None, metadata: Optional[SRADatasetInfo] = None
    ) -> QualityProfile:
        """
        Generate comprehensive quality profile for SRA dataset.

        Args:
            accession: SRA accession
            fastq_path: Path to FASTQ file (optional, will attempt to locate)
            metadata: Dataset metadata (optional)

        Returns:
            QualityProfile with comprehensive analysis
        """
        logger.info(f"Profiling dataset quality for {accession}")

        # Locate FASTQ file if not provided
        if fastq_path is None:
            fastq_path = self._locate_fastq_file(accession)

        if fastq_path and Path(fastq_path).exists():
            # Analyze FASTQ file
            quality_metrics = self.quality_analyzer.analyze_fastq_quality(fastq_path)
        else:
            logger.warning(f"FASTQ file not found for {accession}, using metadata only")
            quality_metrics = {}

        # Extract metrics
        total_reads = quality_metrics.get("total_reads_sampled", 0)
        read_length_stats = quality_metrics.get("read_length_stats", {})
        gc_stats = quality_metrics.get("gc_content_stats", {})
        quality_stats = quality_metrics.get("quality_stats", {})
        complexity_metrics = quality_metrics.get("complexity_metrics", {})
        contamination = quality_metrics.get("contamination_indicators", {})

        # Calculate quality grade
        quality_grade = self._calculate_quality_grade(quality_stats, contamination, complexity_metrics)

        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            quality_stats, contamination, complexity_metrics, metadata
        )

        return QualityProfile(
            accession=accession,
            total_reads=total_reads,
            total_bases=int(total_reads * read_length_stats.get("mean", 0)),
            avg_read_length=read_length_stats.get("mean", 0),
            read_length_distribution=read_length_stats.get("distribution", {}),
            gc_content=gc_stats.get("mean", 0),
            gc_distribution=gc_stats.get("distribution", []),
            quality_distribution=quality_stats.get("distribution", {}),
            n_content=quality_metrics.get("n_content_stats", {}).get("mean", 0),
            contamination_indicators=contamination,
            complexity_score=complexity_metrics.get("complexity_score", 0),
            duplication_rate=None,  # Would need additional analysis
            technology_confidence=self._estimate_technology_confidence(metadata),
            quality_grade=quality_grade,
            recommendations=recommendations,
        )

    def compare_datasets(
        self, groups: Dict[str, List[str]], metadata_df: Optional[pd.DataFrame] = None
    ) -> ComparativeAnalysis:
        """
        Perform comparative analysis across dataset groups.

        Args:
            groups: Dictionary mapping group names to lists of accessions
            metadata_df: DataFrame with metadata for all datasets

        Returns:
            ComparativeAnalysis with statistical comparisons
        """
        logger.info(f"Comparing {len(groups)} dataset groups")

        # Collect quality profiles for all datasets
        all_profiles = {}
        for group_name, accessions in groups.items():
            for accession in accessions:
                try:
                    profile = self.profile_dataset_quality(accession)
                    all_profiles[accession] = profile
                except Exception as e:
                    logger.warning(f"Failed to profile {accession}: {e}")

        # Create comparison DataFrame
        comparison_data = []
        for accession, profile in all_profiles.items():
            # Find which group this accession belongs to
            group = None
            for group_name, acc_list in groups.items():
                if accession in acc_list:
                    group = group_name
                    break

            if group:
                comparison_data.append(
                    {
                        "accession": accession,
                        "group": group,
                        "avg_read_length": profile.avg_read_length,
                        "gc_content": profile.gc_content,
                        "total_reads": profile.total_reads,
                        "complexity_score": profile.complexity_score,
                        "quality_grade": profile.quality_grade,
                        "adapter_contamination": profile.contamination_indicators.get("adapter_contamination", 0),
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)

        if comparison_df.empty:
            logger.warning("No data available for comparison")
            return ComparativeAnalysis(
                dataset_groups=groups,
                summary_statistics={},
                statistical_tests={},
                outlier_datasets=[],
                clustering_results=None,
                batch_effects={},
                recommendations=["No data available for analysis"],
                visualization_data={},
            )

        # Calculate summary statistics
        summary_stats = {}
        numeric_cols = ["avg_read_length", "gc_content", "total_reads", "complexity_score"]

        for group in groups.keys():
            group_data = comparison_df[comparison_df["group"] == group]
            if not group_data.empty:
                summary_stats[group] = {}
                for col in numeric_cols:
                    if col in group_data.columns:
                        values = group_data[col].dropna()
                        if not values.empty:
                            summary_stats[group][col] = {
                                "mean": float(values.mean()),
                                "std": float(values.std()),
                                "median": float(values.median()),
                                "min": float(values.min()),
                                "max": float(values.max()),
                            }

        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(comparison_df, groups)

        # Detect outliers
        outliers = self._detect_outliers(comparison_df)

        # Generate recommendations
        recommendations = self._generate_comparative_recommendations(summary_stats, statistical_tests, outliers)

        return ComparativeAnalysis(
            dataset_groups=groups,
            summary_statistics=summary_stats,
            statistical_tests=statistical_tests,
            outlier_datasets=outliers,
            clustering_results=None,  # Would implement clustering if needed
            batch_effects={},  # Would implement batch effect detection
            recommendations=recommendations,
            visualization_data=self._prepare_visualization_data(comparison_df),
        )

    def detect_dataset_anomalies(
        self, accessions: List[str], metadata_df: Optional[pd.DataFrame] = None
    ) -> AnomalyReport:
        """
        Detect anomalies in SRA datasets.

        Args:
            accessions: List of SRA accessions to analyze
            metadata_df: DataFrame with metadata

        Returns:
            AnomalyReport with detected anomalies
        """
        logger.info(f"Detecting anomalies in {len(accessions)} datasets")

        anomalous_datasets = []
        anomaly_types = defaultdict(list)
        severity_scores = {}
        explanations = {}
        recommended_actions = defaultdict(list)

        # Profile each dataset and look for anomalies
        for accession in accessions:
            try:
                profile = self.profile_dataset_quality(accession)

                # Check for various anomaly types
                severity = 0.0
                anomaly_reasons = []

                # Low complexity
                if profile.complexity_score < 0.1:
                    anomaly_types["low_complexity"].append(accession)
                    anomaly_reasons.append("Very low sequence complexity")
                    severity += 0.3

                # High contamination
                adapter_contamination = profile.contamination_indicators.get("adapter_contamination", 0)
                if adapter_contamination > 0.1:
                    anomaly_types["high_contamination"].append(accession)
                    anomaly_reasons.append(f"High adapter contamination: {adapter_contamination:.1%}")
                    severity += 0.4

                # Unusual GC content
                if profile.gc_content < 0.2 or profile.gc_content > 0.8:
                    anomaly_types["unusual_gc"].append(accession)
                    anomaly_reasons.append(f"Unusual GC content: {profile.gc_content:.1%}")
                    severity += 0.2

                # High N content
                if profile.n_content > 0.05:
                    anomaly_types["high_n_content"].append(accession)
                    anomaly_reasons.append(f"High N content: {profile.n_content:.1%}")
                    severity += 0.3

                # Poor quality grade
                if profile.quality_grade == "poor":
                    anomaly_types["poor_quality"].append(accession)
                    anomaly_reasons.append("Poor overall quality")
                    severity += 0.5

                if severity > 0.2:  # Threshold for anomaly
                    anomalous_datasets.append(accession)
                    severity_scores[accession] = min(severity, 1.0)
                    explanations[accession] = "; ".join(anomaly_reasons)

                    # Generate recommendations
                    actions = []
                    if "high_contamination" in [t for t in anomaly_types if accession in anomaly_types[t]]:
                        actions.append("Perform adapter trimming")
                        actions.append("Check for vector contamination")
                    if "poor_quality" in [t for t in anomaly_types if accession in anomaly_types[t]]:
                        actions.append("Apply quality filtering")
                        actions.append("Consider excluding from analysis")
                    if "low_complexity" in [t for t in anomaly_types if accession in anomaly_types[t]]:
                        actions.append("Check for PCR artifacts")
                        actions.append("Verify sample preparation")

                    recommended_actions[accession] = actions

            except Exception as e:
                logger.error(f"Error analyzing {accession} for anomalies: {e}")
                anomaly_types["analysis_failed"].append(accession)
                severity_scores[accession] = 0.1
                explanations[accession] = f"Analysis failed: {str(e)}"

        return AnomalyReport(
            anomalous_datasets=anomalous_datasets,
            anomaly_types=dict(anomaly_types),
            severity_scores=severity_scores,
            explanations=explanations,
            recommended_actions=dict(recommended_actions),
        )

    def recommend_processing_params(
        self, accession: str, profile: Optional[QualityProfile] = None
    ) -> ProcessingRecommendations:
        """
        Generate processing parameter recommendations based on dataset characteristics.

        Args:
            accession: SRA accession
            profile: Quality profile (will generate if not provided)

        Returns:
            ProcessingRecommendations with optimized parameters
        """
        if profile is None:
            profile = self.profile_dataset_quality(accession)

        # Determine recommended pipeline
        if profile.avg_read_length > 1000:
            pipeline = "long_read"
        elif profile.avg_read_length > 250:
            pipeline = "paired_end"
        else:
            pipeline = "short_read"

        # Quality trimming parameters
        quality_trimming = {
            "enabled": profile.quality_grade in ["poor", "fair"],
            "quality_threshold": 20 if profile.quality_grade == "poor" else 15,
            "min_length": max(50, int(profile.avg_read_length * 0.7)),
        }

        # Adapter removal
        adapter_contamination = profile.contamination_indicators.get("adapter_contamination", 0)
        adapter_removal = {
            "enabled": adapter_contamination > 0.01,
            "stringency": "high" if adapter_contamination > 0.1 else "medium",
        }

        # Contamination filtering
        contamination_filtering = {
            "enabled": any(v > 0.05 for v in profile.contamination_indicators.values()),
            "check_vector": True,
            "check_adapters": adapter_contamination > 0.01,
            "complexity_filter": profile.complexity_score < 0.3,
        }

        # Assembly parameters (basic recommendations)
        assembly_parameters = {
            "kmer_size": "auto" if profile.avg_read_length < 150 else 31,
            "coverage_cutoff": 5 if profile.quality_grade in ["good", "excellent"] else 10,
            "error_correction": profile.quality_grade != "excellent",
        }

        # Computational requirements estimation
        computational_requirements = {
            "memory_gb": max(8, int(profile.total_reads / 10000000) * 2),
            "cpu_cores": 4 if profile.total_reads < 50000000 else 8,
            "storage_gb": max(10, int(profile.total_bases / 1000000000) * 5),
        }

        # Processing time estimation
        base_time_hours = profile.total_reads / 5000000  # Rough estimate
        if quality_trimming["enabled"]:
            base_time_hours *= 1.5
        if contamination_filtering["enabled"]:
            base_time_hours *= 1.3

        processing_time = f"{base_time_hours:.1f} hours"

        return ProcessingRecommendations(
            accession=accession,
            recommended_pipeline=pipeline,
            quality_trimming=quality_trimming,
            adapter_removal=adapter_removal,
            contamination_filtering=contamination_filtering,
            assembly_parameters=assembly_parameters,
            expected_coverage=None,  # Would need genome size estimate
            computational_requirements=computational_requirements,
            estimated_processing_time=processing_time,
        )

    def _locate_fastq_file(self, accession: str) -> Optional[Path]:
        """Attempt to locate FASTQ file for accession."""
        # Common locations to check
        possible_paths = [
            Path(f"{accession}.fastq"),
            Path(f"{accession}.fastq.gz"),
            Path(f"downloads/{accession}.fastq.gz"),
            Path(f"data/{accession}.fastq.gz"),
            Path(f"fastq/{accession}.fastq.gz"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _calculate_quality_grade(self, quality_stats: Dict, contamination: Dict, complexity: Dict) -> str:
        """Calculate overall quality grade."""
        score = 0.0

        # Quality score component
        if quality_stats.get("mean", 0) >= 30:
            score += 0.4
        elif quality_stats.get("mean", 0) >= 20:
            score += 0.2

        # Contamination component
        adapter_cont = contamination.get("adapter_contamination", 0)
        if adapter_cont < 0.01:
            score += 0.3
        elif adapter_cont < 0.05:
            score += 0.1

        # Complexity component
        complexity_score = complexity.get("complexity_score", 0)
        if complexity_score > 0.7:
            score += 0.3
        elif complexity_score > 0.4:
            score += 0.15

        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _generate_quality_recommendations(
        self, quality_stats: Dict, contamination: Dict, complexity: Dict, metadata: Optional[SRADatasetInfo]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        if quality_stats.get("mean", 0) < 20:
            recommendations.append("Apply quality trimming with Q20 threshold")

        if contamination.get("adapter_contamination", 0) > 0.05:
            recommendations.append("Remove adapter sequences")

        if complexity.get("complexity_score", 0) < 0.3:
            recommendations.append("Check for PCR duplicates and low-complexity regions")

        if not recommendations:
            recommendations.append("Dataset appears to be high quality")

        return recommendations

    def _estimate_technology_confidence(self, metadata: Optional[SRADatasetInfo]) -> float:
        """Estimate confidence in technology detection."""
        if not metadata:
            return 0.5

        # Simple heuristic based on metadata completeness
        confidence = 0.5
        if hasattr(metadata, "platform") and metadata.platform:
            confidence += 0.3
        if hasattr(metadata, "instrument") and metadata.instrument:
            confidence += 0.2

        return min(confidence, 1.0)

    def _perform_statistical_tests(
        self, comparison_df: pd.DataFrame, groups: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical tests between groups."""
        tests = {}

        numeric_cols = ["avg_read_length", "gc_content", "total_reads", "complexity_score"]
        group_names = list(groups.keys())

        if len(group_names) < 2:
            return tests

        for col in numeric_cols:
            if col not in comparison_df.columns:
                continue

            tests[col] = {}

            # Get data for each group
            group_data = []
            for group in group_names:
                data = comparison_df[comparison_df["group"] == group][col].dropna()
                if not data.empty:
                    group_data.append(data.values)

            if len(group_data) >= 2:
                # Check if data is nearly identical (would cause precision loss)
                all_values = np.concatenate(group_data)
                variance = np.var(all_values)
                
                if variance < 1e-10:  # Extremely low variance, skip statistical tests
                    tests[col]["test"] = "skipped (identical values)"
                    tests[col]["statistic"] = np.nan
                    tests[col]["p_value"] = 1.0  # No significant difference for identical values
                    tests[col]["significant"] = False
                    tests[col]["note"] = "Values too similar for meaningful statistical comparison"
                else:
                    # Perform t-test if 2 groups, ANOVA if more
                    try:
                        if len(group_data) == 2:
                            statistic, p_value = stats.ttest_ind(group_data[0], group_data[1])
                            tests[col]["test"] = "t-test"
                        else:
                            statistic, p_value = stats.f_oneway(*group_data)
                            tests[col]["test"] = "ANOVA"

                        tests[col]["statistic"] = float(statistic)
                        tests[col]["p_value"] = float(p_value)
                        tests[col]["significant"] = p_value < 0.05
                    except RuntimeWarning:
                        # Handle precision loss warnings gracefully
                        tests[col]["test"] = "failed (precision loss)"
                        tests[col]["statistic"] = np.nan
                        tests[col]["p_value"] = np.nan
                        tests[col]["significant"] = False

        return tests

    def _detect_outliers(self, comparison_df: pd.DataFrame) -> List[str]:
        """Detect outlier datasets using statistical methods."""
        outliers = []

        numeric_cols = ["avg_read_length", "gc_content", "total_reads", "complexity_score"]

        for col in numeric_cols:
            if col not in comparison_df.columns:
                continue

            values = comparison_df[col].dropna()
            if len(values) < 4:  # Need sufficient data
                continue

            # Use IQR method
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outlier_mask = (values < lower_bound) | (values > upper_bound)
            outlier_accessions = comparison_df.loc[values[outlier_mask].index, "accession"].tolist()
            outliers.extend(outlier_accessions)

        return list(set(outliers))  # Remove duplicates

    def _generate_comparative_recommendations(
        self, summary_stats: Dict, statistical_tests: Dict, outliers: List[str]
    ) -> List[str]:
        """Generate recommendations based on comparative analysis."""
        recommendations = []

        if statistical_tests:
            significant_tests = [col for col, test in statistical_tests.items() if test.get("significant", False)]
            if significant_tests:
                recommendations.append(f"Significant differences detected in: {', '.join(significant_tests)}")
                recommendations.append("Consider including group as covariate in downstream analysis")

        if outliers:
            recommendations.append(
                f"Outlier datasets detected: {', '.join(outliers[:5])}{'...' if len(outliers) > 5 else ''}"
            )
            recommendations.append("Review outlier datasets for quality issues")

        if not recommendations:
            recommendations.append("No major issues detected in comparative analysis")

        return recommendations

    def _prepare_visualization_data(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for visualization."""
        return {
            "boxplot_data": comparison_df.to_dict("records"),
            "summary_table": comparison_df.groupby("group").describe().to_dict(),
            "correlation_matrix": comparison_df.select_dtypes(include=[np.number]).corr().to_dict(),
        }
