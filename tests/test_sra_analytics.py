"""
Comprehensive tests for SRA dataset analytics and quality profiling.

Tests cover:
- Sequence quality analysis from FASTQ files
- Dataset quality profiling and grading
- Comparative analysis across dataset groups
- Anomaly detection in datasets
- Processing parameter recommendations
- Statistical testing and visualization data preparation
"""

from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pytest
import pandas as pd

# Mock Bio.SeqIO to avoid Biopython dependency in tests
with patch.dict("sys.modules", {"Bio": Mock(), "Bio.SeqIO": Mock()}):
    from metaquest.sra.analytics import (
        SRADatasetAnalyzer,
        SequenceQualityAnalyzer,
        QualityProfile,
        ComparativeAnalysis,
        AnomalyReport,
        ProcessingRecommendations,
    )

from metaquest.core.exceptions import DataAccessError


class TestSequenceQualityAnalyzer:
    """Test sequence quality analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceQualityAnalyzer()

    def test_analyzer_initialization(self):
        """Test SequenceQualityAnalyzer initialization."""
        assert hasattr(self.analyzer, "quality_encodings")
        assert "sanger" in self.analyzer.quality_encodings
        assert self.analyzer.quality_encodings["sanger"] == 33

    def test_calculate_gc_content(self):
        """Test GC content calculation."""
        # Test normal sequence
        gc_content = self.analyzer._calculate_gc_content("ATGCGCAT")
        assert gc_content == 0.5  # 4 G/C out of 8 bases

        # Test all GC
        gc_content_all = self.analyzer._calculate_gc_content("GGGGCCCC")
        assert gc_content_all == 1.0

        # Test no GC
        gc_content_none = self.analyzer._calculate_gc_content("ATATATAT")
        assert gc_content_none == 0.0

        # Test empty sequence
        gc_content_empty = self.analyzer._calculate_gc_content("")
        assert gc_content_empty == 0.0

    def test_get_length_distribution(self):
        """Test read length distribution calculation."""
        lengths = [25, 75, 125, 175, 300, 750, 1500]
        distribution = self.analyzer._get_length_distribution(lengths)

        assert distribution["0-50"] == 1  # 25
        assert distribution["51-100"] == 1  # 75
        assert distribution["101-150"] == 1  # 125
        assert distribution["151-250"] == 1  # 175
        assert distribution["251-500"] == 1  # 300
        assert distribution["501-1000"] == 1  # 750
        assert distribution["1000+"] == 1  # 1500

    def test_get_quality_distribution(self):
        """Test quality score distribution calculation."""
        quality_scores = [5, 15, 25, 35, 8, 22, 32, 12]
        distribution = self.analyzer._get_quality_distribution(quality_scores)

        assert distribution["excellent_q30+"] == 2 / 8  # 35, 32
        assert distribution["good_q20-29"] == 2 / 8  # 25, 22
        assert distribution["fair_q10-19"] == 2 / 8  # 15, 12
        assert distribution["poor_q0-9"] == 2 / 8  # 5, 8

    def test_analyze_sequence_complexity(self):
        """Test sequence complexity analysis."""
        # High complexity sequence
        high_complexity = ["ATGCGTACGTAGC", "CGTACGTAGCTAG", "TACGTAGCTAGCT"]
        complexity_high = self.analyzer._analyze_sequence_complexity(high_complexity)

        assert "kmer_diversity_mean" in complexity_high
        assert "homopolymer_rate_mean" in complexity_high
        assert "complexity_score" in complexity_high

        # Low complexity sequence (homopolymer runs)
        low_complexity = ["AAAAATTTTTGGGG", "CCCCCCAAAAAAA", "TTTTTTTTGGGGGG"]
        complexity_low = self.analyzer._analyze_sequence_complexity(low_complexity)

        assert complexity_low["homopolymer_rate_mean"] > 0

    def test_detect_contamination_indicators(self):
        """Test contamination detection."""
        # Sequences with adapter contamination
        contaminated_sequences = [
            "ATGCAGATCGGAAGAGCGTACG",  # Contains Illumina adapter
            "CGTACGTAGCTAG",
            "TACGTAGCTAGCT",
        ]

        indicators = self.analyzer._detect_contamination_indicators(contaminated_sequences)

        assert "adapter_contamination" in indicators
        assert "vector_contamination" in indicators
        assert "overrepresented_sequences" in indicators
        assert indicators["adapter_contamination"] > 0

    @patch("builtins.open", new_callable=mock_open, read_data="@seq1\nATGCGTACGT\n+\nIIIIIIIIII\n")
    @patch("metaquest.sra.analytics.SeqIO")
    def test_analyze_fastq_quality_success(self, mock_seqio, mock_file):
        """Test successful FASTQ quality analysis."""
        # Mock SeqIO.parse to return mock records
        mock_record1 = Mock()
        mock_record1.seq = "ATGCGTACGT"
        mock_record1.letter_annotations = {"phred_quality": [30, 35, 40, 35, 30, 25, 30, 35, 40, 35]}

        mock_record2 = Mock()
        mock_record2.seq = "CGTACGTAGC"
        mock_record2.letter_annotations = {"phred_quality": [25, 30, 35, 30, 25, 20, 25, 30, 35, 30]}

        mock_seqio.parse.return_value = [mock_record1, mock_record2]

        with patch("pathlib.Path.exists", return_value=True):
            result = self.analyzer.analyze_fastq_quality("test.fastq", sample_size=100)

        assert "total_reads_sampled" in result
        assert "read_length_stats" in result
        assert "gc_content_stats" in result
        assert "quality_stats" in result
        assert "complexity_metrics" in result
        assert result["total_reads_sampled"] == 2

    def test_analyze_fastq_quality_file_not_found(self):
        """Test FASTQ analysis with missing file."""
        with pytest.raises(DataAccessError, match="FASTQ file not found"):
            self.analyzer.analyze_fastq_quality("/nonexistent/file.fastq")

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_analyze_fastq_quality_io_error(self, mock_file):
        """Test FASTQ analysis with IO error."""
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(DataAccessError, match="Failed to analyze FASTQ file"):
                self.analyzer.analyze_fastq_quality("test.fastq")


class TestSRADatasetAnalyzer:
    """Test main SRA dataset analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SRADatasetAnalyzer()

    def test_analyzer_initialization(self):
        """Test SRADatasetAnalyzer initialization."""
        assert hasattr(self.analyzer, "quality_analyzer")
        assert isinstance(self.analyzer.quality_analyzer, SequenceQualityAnalyzer)

    def test_locate_fastq_file(self):
        """Test FASTQ file location."""
        with patch("pathlib.Path.exists") as mock_exists:
            # Mock first path exists, others don't
            call_count = 0

            def exists_side_effect():
                nonlocal call_count
                call_count += 1
                return call_count == 1  # True only for first call

            mock_exists.side_effect = exists_side_effect

            result = self.analyzer._locate_fastq_file("SRR123456")
            assert result == Path("SRR123456.fastq")

    def test_locate_fastq_file_not_found(self):
        """Test FASTQ file location when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            result = self.analyzer._locate_fastq_file("NONEXISTENT")
            assert result is None

    def test_calculate_quality_grade(self):
        """Test quality grade calculation."""
        # Excellent quality
        excellent_grade = self.analyzer._calculate_quality_grade(
            quality_stats={"mean": 35},
            contamination={"adapter_contamination": 0.005},
            complexity={"complexity_score": 0.8},
        )
        assert excellent_grade == "excellent"

        # Poor quality
        poor_grade = self.analyzer._calculate_quality_grade(
            quality_stats={"mean": 15},
            contamination={"adapter_contamination": 0.1},
            complexity={"complexity_score": 0.2},
        )
        assert poor_grade == "poor"

        # Fair quality (score = 0.2 + 0.1 + 0.15 = 0.45)
        fair_grade = self.analyzer._calculate_quality_grade(
            quality_stats={"mean": 25},
            contamination={"adapter_contamination": 0.02},
            complexity={"complexity_score": 0.6},
        )
        assert fair_grade == "fair"

    def test_generate_quality_recommendations(self):
        """Test quality recommendation generation."""
        # Low quality scenario
        recommendations = self.analyzer._generate_quality_recommendations(
            quality_stats={"mean": 15},
            contamination={"adapter_contamination": 0.1},
            complexity={"complexity_score": 0.2},
            metadata=None,
        )

        assert len(recommendations) > 0
        assert any("quality trimming" in rec.lower() for rec in recommendations)
        assert any("adapter" in rec.lower() for rec in recommendations)
        assert any("duplicates" in rec.lower() for rec in recommendations)

        # High quality scenario
        good_recommendations = self.analyzer._generate_quality_recommendations(
            quality_stats={"mean": 35},
            contamination={"adapter_contamination": 0.01},
            complexity={"complexity_score": 0.8},
            metadata=None,
        )

        assert any("high quality" in rec.lower() for rec in good_recommendations)

    def test_estimate_technology_confidence(self):
        """Test technology detection confidence estimation."""
        # Complete metadata
        complete_metadata = Mock()
        complete_metadata.platform = "ILLUMINA"
        complete_metadata.instrument = "HiSeq 2500"

        confidence = self.analyzer._estimate_technology_confidence(complete_metadata)
        assert confidence == 1.0

        # Partial metadata
        partial_metadata = Mock()
        partial_metadata.platform = "ILLUMINA"
        partial_metadata.instrument = None

        confidence_partial = self.analyzer._estimate_technology_confidence(partial_metadata)
        assert 0.5 < confidence_partial < 1.0

        # No metadata
        confidence_none = self.analyzer._estimate_technology_confidence(None)
        assert confidence_none == 0.5

    @patch.object(SequenceQualityAnalyzer, "analyze_fastq_quality")
    def test_profile_dataset_quality(self, mock_analyze):
        """Test dataset quality profiling."""
        # Mock FASTQ analysis results
        mock_analyze.return_value = {
            "total_reads_sampled": 10000,
            "read_length_stats": {"mean": 150, "distribution": {"101-150": 10000}},
            "gc_content_stats": {"mean": 0.45, "distribution": [0.45] * 10000},
            "quality_stats": {"mean": 30, "distribution": {"excellent_q30+": 0.8}},
            "n_content_stats": {"mean": 0.01},
            "complexity_metrics": {"complexity_score": 0.7},
            "contamination_indicators": {"adapter_contamination": 0.02},
        }

        with patch.object(self.analyzer, "_locate_fastq_file", return_value=Path("test.fastq")):
            with patch("pathlib.Path.exists", return_value=True):
                profile = self.analyzer.profile_dataset_quality("SRR123456")

        assert isinstance(profile, QualityProfile)
        assert profile.accession == "SRR123456"
        assert profile.total_reads == 10000
        assert profile.avg_read_length == 150
        assert profile.gc_content == 0.45
        assert profile.complexity_score == 0.7
        assert profile.quality_grade in ["excellent", "good", "fair", "poor"]
        assert len(profile.recommendations) > 0

    def test_compare_datasets(self):
        """Test dataset comparison functionality."""
        groups = {"group1": ["SRR123456", "SRR123457"], "group2": ["SRR789012", "SRR789013"]}

        # Mock profile generation
        mock_profiles = {
            "SRR123456": QualityProfile(
                accession="SRR123456",
                total_reads=10000,
                total_bases=1500000,
                avg_read_length=150,
                read_length_distribution={},
                gc_content=0.45,
                gc_distribution=[],
                quality_distribution={},
                n_content=0.01,
                contamination_indicators={"adapter_contamination": 0.02},
                complexity_score=0.7,
                duplication_rate=None,
                technology_confidence=0.9,
                quality_grade="good",
                recommendations=[],
            ),
            "SRR123457": QualityProfile(
                accession="SRR123457",
                total_reads=12000,
                total_bases=1800000,
                avg_read_length=150,
                read_length_distribution={},
                gc_content=0.48,
                gc_distribution=[],
                quality_distribution={},
                n_content=0.015,
                contamination_indicators={"adapter_contamination": 0.025},
                complexity_score=0.75,
                duplication_rate=None,
                technology_confidence=0.85,
                quality_grade="good",
                recommendations=[],
            ),
            "SRR789012": QualityProfile(
                accession="SRR789012",
                total_reads=8000,
                total_bases=1200000,
                avg_read_length=150,
                read_length_distribution={},
                gc_content=0.42,
                gc_distribution=[],
                quality_distribution={},
                n_content=0.008,
                contamination_indicators={"adapter_contamination": 0.03},
                complexity_score=0.65,
                duplication_rate=None,
                technology_confidence=0.88,
                quality_grade="fair",
                recommendations=[],
            ),
            "SRR789013": QualityProfile(
                accession="SRR789013",
                total_reads=9000,
                total_bases=1350000,
                avg_read_length=150,
                read_length_distribution={},
                gc_content=0.40,
                gc_distribution=[],
                quality_distribution={},
                n_content=0.012,
                contamination_indicators={"adapter_contamination": 0.035},
                complexity_score=0.68,
                duplication_rate=None,
                technology_confidence=0.82,
                quality_grade="fair",
                recommendations=[],
            ),
        }

        with patch.object(self.analyzer, "profile_dataset_quality") as mock_profile:

            def profile_side_effect(accession):
                return mock_profiles.get(accession)

            mock_profile.side_effect = profile_side_effect

            comparison = self.analyzer.compare_datasets(groups)

        assert isinstance(comparison, ComparativeAnalysis)
        assert comparison.dataset_groups == groups
        assert "group1" in comparison.summary_statistics
        assert len(comparison.recommendations) > 0

    def test_detect_dataset_anomalies(self):
        """Test anomaly detection functionality."""
        accessions = ["SRR123456", "SRR789012", "SRR555555"]

        # Mock profiles with different quality characteristics
        mock_profiles = {
            "SRR123456": QualityProfile(  # Normal dataset
                accession="SRR123456",
                total_reads=10000,
                total_bases=1500000,
                avg_read_length=150,
                read_length_distribution={},
                gc_content=0.45,
                gc_distribution=[],
                quality_distribution={},
                n_content=0.01,
                contamination_indicators={"adapter_contamination": 0.02},
                complexity_score=0.7,
                duplication_rate=None,
                technology_confidence=0.9,
                quality_grade="good",
                recommendations=[],
            ),
            "SRR789012": QualityProfile(  # Anomalous dataset - high contamination
                accession="SRR789012",
                total_reads=8000,
                total_bases=1200000,
                avg_read_length=150,
                read_length_distribution={},
                gc_content=0.15,  # Unusual GC
                gc_distribution=[],
                quality_distribution={},
                n_content=0.08,  # High N
                contamination_indicators={"adapter_contamination": 0.15},
                complexity_score=0.05,  # Low complexity
                duplication_rate=None,
                technology_confidence=0.6,
                quality_grade="poor",
                recommendations=[],
            ),
            "SRR555555": QualityProfile(  # Normal dataset
                accession="SRR555555",
                total_reads=11000,
                total_bases=1650000,
                avg_read_length=150,
                read_length_distribution={},
                gc_content=0.42,
                gc_distribution=[],
                quality_distribution={},
                n_content=0.012,
                contamination_indicators={"adapter_contamination": 0.018},
                complexity_score=0.72,
                duplication_rate=None,
                technology_confidence=0.88,
                quality_grade="good",
                recommendations=[],
            ),
        }

        with patch.object(self.analyzer, "profile_dataset_quality") as mock_profile:

            def profile_side_effect(accession):
                return mock_profiles.get(accession)

            mock_profile.side_effect = profile_side_effect

            anomaly_report = self.analyzer.detect_dataset_anomalies(accessions)

        assert isinstance(anomaly_report, AnomalyReport)
        assert "SRR789012" in anomaly_report.anomalous_datasets  # Should be flagged
        assert "SRR123456" not in anomaly_report.anomalous_datasets  # Should be normal
        assert len(anomaly_report.anomaly_types) > 0
        assert "SRR789012" in anomaly_report.severity_scores
        assert anomaly_report.severity_scores["SRR789012"] > 0.2

    def test_recommend_processing_params(self):
        """Test processing parameter recommendations."""
        # Create test profile
        test_profile = QualityProfile(
            accession="SRR123456",
            total_reads=10000,
            total_bases=1500000,
            avg_read_length=150,
            read_length_distribution={},
            gc_content=0.45,
            gc_distribution=[],
            quality_distribution={},
            n_content=0.01,
            contamination_indicators={"adapter_contamination": 0.08},
            complexity_score=0.7,
            duplication_rate=None,
            technology_confidence=0.9,
            quality_grade="fair",
            recommendations=[],
        )

        recommendations = self.analyzer.recommend_processing_params("SRR123456", test_profile)

        assert isinstance(recommendations, ProcessingRecommendations)
        assert recommendations.accession == "SRR123456"
        assert recommendations.recommended_pipeline in ["short_read", "paired_end", "long_read"]
        assert "enabled" in recommendations.quality_trimming
        assert "enabled" in recommendations.adapter_removal
        assert recommendations.adapter_removal["enabled"] is True  # High contamination
        assert "memory_gb" in recommendations.computational_requirements
        assert recommendations.estimated_processing_time is not None


class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SRADatasetAnalyzer()

    def test_perform_statistical_tests(self):
        """Test statistical test performance."""
        # Create test comparison data
        comparison_df = pd.DataFrame(
            {
                "accession": ["SRR1", "SRR2", "SRR3", "SRR4"],
                "group": ["A", "A", "B", "B"],
                "avg_read_length": [150, 155, 250, 245],
                "gc_content": [0.45, 0.47, 0.52, 0.50],
                "total_reads": [10000, 12000, 8000, 9000],
                "complexity_score": [0.7, 0.75, 0.6, 0.65],
            }
        )

        groups = {"A": ["SRR1", "SRR2"], "B": ["SRR3", "SRR4"]}

        tests = self.analyzer._perform_statistical_tests(comparison_df, groups)

        assert "avg_read_length" in tests
        assert "gc_content" in tests
        assert tests["avg_read_length"]["test"] == "t-test"  # Two groups
        assert "statistic" in tests["avg_read_length"]
        assert "p_value" in tests["avg_read_length"]
        assert "significant" in tests["avg_read_length"]

    def test_detect_outliers(self):
        """Test outlier detection."""
        # Create test data with obvious outlier
        comparison_df = pd.DataFrame(
            {
                "accession": ["SRR1", "SRR2", "SRR3", "SRR4", "SRR5"],
                "avg_read_length": [150, 155, 148, 152, 500],  # SRR5 is outlier
                "gc_content": [0.45, 0.47, 0.44, 0.46, 0.48],
                "total_reads": [10000, 12000, 11000, 10500, 9500],
            }
        )

        outliers = self.analyzer._detect_outliers(comparison_df)

        assert "SRR5" in outliers  # Should detect read length outlier

    def test_generate_comparative_recommendations(self):
        """Test comparative analysis recommendations."""
        summary_stats = {"group_A": {"avg_read_length": {"mean": 150}}, "group_B": {"avg_read_length": {"mean": 250}}}

        statistical_tests = {"avg_read_length": {"significant": True, "p_value": 0.001}}

        outliers = ["SRR123456"]

        recommendations = self.analyzer._generate_comparative_recommendations(
            summary_stats, statistical_tests, outliers
        )

        assert len(recommendations) > 0
        assert any("significant" in rec.lower() for rec in recommendations)
        assert any("outlier" in rec.lower() for rec in recommendations)

    def test_prepare_visualization_data(self):
        """Test visualization data preparation."""
        comparison_df = pd.DataFrame(
            {
                "accession": ["SRR1", "SRR2", "SRR3"],
                "group": ["A", "A", "B"],
                "avg_read_length": [150, 155, 250],
                "gc_content": [0.45, 0.47, 0.52],
            }
        )

        viz_data = self.analyzer._prepare_visualization_data(comparison_df)

        assert "boxplot_data" in viz_data
        assert "summary_table" in viz_data
        assert "correlation_matrix" in viz_data
        assert len(viz_data["boxplot_data"]) == 3


class TestDataStructures:
    """Test data structure creation and validation."""

    def test_quality_profile_creation(self):
        """Test QualityProfile creation."""
        profile = QualityProfile(
            accession="SRR123456",
            total_reads=10000,
            total_bases=1500000,
            avg_read_length=150.0,
            read_length_distribution={"101-150": 8000, "151-250": 2000},
            gc_content=0.45,
            gc_distribution=[0.45] * 100,
            quality_distribution={"excellent_q30+": 0.8},
            n_content=0.01,
            contamination_indicators={"adapter_contamination": 0.02},
            complexity_score=0.7,
            duplication_rate=0.15,
            technology_confidence=0.9,
            quality_grade="good",
            recommendations=["High quality dataset"],
        )

        assert profile.accession == "SRR123456"
        assert profile.total_reads == 10000
        assert profile.quality_grade == "good"
        assert len(profile.recommendations) == 1

    def test_comparative_analysis_creation(self):
        """Test ComparativeAnalysis creation."""
        analysis = ComparativeAnalysis(
            dataset_groups={"A": ["SRR1", "SRR2"], "B": ["SRR3", "SRR4"]},
            summary_statistics={"A": {"mean_gc": 0.45}, "B": {"mean_gc": 0.52}},
            statistical_tests={"gc_content": {"p_value": 0.001, "significant": True}},
            outlier_datasets=["SRR5"],
            clustering_results=None,
            batch_effects={"batch1": 0.1},
            recommendations=["Significant differences detected"],
            visualization_data={},
        )

        assert len(analysis.dataset_groups) == 2
        assert "A" in analysis.dataset_groups
        assert analysis.statistical_tests["gc_content"]["significant"] is True
        assert "SRR5" in analysis.outlier_datasets

    def test_anomaly_report_creation(self):
        """Test AnomalyReport creation."""
        report = AnomalyReport(
            anomalous_datasets=["SRR123456", "SRR789012"],
            anomaly_types={"high_contamination": ["SRR123456"], "low_complexity": ["SRR789012"]},
            severity_scores={"SRR123456": 0.8, "SRR789012": 0.6},
            explanations={"SRR123456": "High adapter contamination", "SRR789012": "Low complexity"},
            recommended_actions={"SRR123456": ["Trim adapters"], "SRR789012": ["Check PCR artifacts"]},
        )

        assert len(report.anomalous_datasets) == 2
        assert "high_contamination" in report.anomaly_types
        assert report.severity_scores["SRR123456"] > report.severity_scores["SRR789012"]
        assert len(report.recommended_actions["SRR123456"]) > 0

    def test_processing_recommendations_creation(self):
        """Test ProcessingRecommendations creation."""
        recommendations = ProcessingRecommendations(
            accession="SRR123456",
            recommended_pipeline="paired_end",
            quality_trimming={"enabled": True, "quality_threshold": 20},
            adapter_removal={"enabled": True, "stringency": "high"},
            contamination_filtering={"enabled": True, "check_vector": True},
            assembly_parameters={"kmer_size": 31, "coverage_cutoff": 5},
            expected_coverage=50.0,
            computational_requirements={"memory_gb": 16, "cpu_cores": 4},
            estimated_processing_time="2.5 hours",
        )

        assert recommendations.accession == "SRR123456"
        assert recommendations.recommended_pipeline == "paired_end"
        assert recommendations.quality_trimming["enabled"] is True
        assert recommendations.computational_requirements["memory_gb"] == 16
