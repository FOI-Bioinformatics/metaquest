"""Per-read quality figures from a sample of a dataset's FASTQ files.

``SequenceQualityAnalyzer.analyze_fastq_quality`` samples reads from every mate file of a
dataset (uniformly over the whole of each file by default) and returns their length, GC
(in percent), base quality, N content, complexity, duplication and adapter figures. The
dataset totals come from the shared statistics record instead; see
``metaquest.sra.analytics.SRADatasetAnalyzer.profile_dataset_quality``.
"""

import itertools
import logging
import random
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from Bio import SeqIO

from metaquest.core.exceptions import DataAccessError
from metaquest.data.sra import iter_fastq_records

logger = logging.getLogger(__name__)

# What reading a FASTQ file raises for a file that is unreadable, truncated or malformed (a
# corrupt gzip stream raises zlib.error, which is not an OSError); anything else is a bug.
_FASTQ_READ_ERRORS = (OSError, EOFError, ValueError, UnicodeDecodeError, zlib.error)


def _gc_histogram(gc_contents: List[float]) -> Dict[str, int]:
    """Bucket per-read GC fractions into 5-percent-wide bins, e.g. ``{"40-45": 12, ...}``."""
    histogram: Dict[str, int] = defaultdict(int)
    for gc in gc_contents:
        start = min(95, max(0, int((gc * 100) // 5) * 5))
        histogram[f"{start}-{start + 5}"] += 1
    return dict(histogram)


class SequenceQualityAnalyzer:
    """Analyzes sequence quality metrics from FASTQ files."""

    def __init__(self):
        self.quality_encodings = {"sanger": 33, "illumina_1.3": 64, "illumina_1.5": 64, "solexa": 64}

    def analyze_fastq_quality(
        self,
        fastq_path: Union[str, Path, Sequence[Union[str, Path]]],
        sample_size: int = 10000,
        sampler: str = "uniform",
    ) -> Dict[str, Any]:
        """
        Analyze quality metrics from the FASTQ file(s) of one dataset.

        Args:
            fastq_path: One FASTQ file, or every mate file of a dataset; the sample is then
                drawn from all of them, so mate 2 is represented as well as mate 1
            sample_size: Number of reads to sample for analysis, across all files
            sampler: ``"uniform"`` (default) reservoir-samples reads across the whole of
                every file, so reads past the first ``sample_size`` are represented too, not
                just the head. ``"head"`` takes the first reads of each file only (an equal
                share of ``sample_size`` per file), for a caller that wants the cheapest
                possible read of files it already knows to be homogeneous.

        Returns:
            Dictionary of quality metrics; GC figures are in percent (0-100)
        """
        paths = [Path(fastq_path)] if isinstance(fastq_path, (str, Path)) else [Path(p) for p in fastq_path]
        for path in paths:
            if not path.exists():
                raise DataAccessError(f"FASTQ file not found: {path}")

        try:
            sample = self._sample(paths, sample_size, sampler)
        except _FASTQ_READ_ERRORS as e:
            logger.error(f"Error analyzing FASTQ file {paths[0] if paths else ''}: {e}")
            raise DataAccessError(f"Failed to analyze FASTQ file: {e}")

        if not sample[0]:
            raise DataAccessError("No valid reads found in FASTQ file")
        return self._summarise(*sample)

    def _sample(
        self, paths: List[Path], sample_size: int, sampler: str
    ) -> Tuple[List[int], List[float], List[int], List[float], List[str]]:
        """Per-read lengths, GC fractions, base qualities, N fractions and sequences of the sample."""
        if sampler != "head":
            return self._sample_uniform(paths, sample_size)
        share = -(-sample_size // max(len(paths), 1))
        merged: Tuple[List[int], List[float], List[int], List[float], List[str]] = ([], [], [], [], [])
        for path in paths:
            lengths, gc_fractions, qualities, n_fractions, sequences = self._sample_head(path, share)
            merged[0].extend(lengths)
            merged[1].extend(gc_fractions)
            merged[2].extend(qualities)
            merged[3].extend(n_fractions)
            merged[4].extend(sequences)
        return merged

    def _summarise(
        self,
        read_lengths: List[int],
        gc_contents: List[float],
        quality_scores: List[int],
        n_contents: List[float],
        sequences: List[str],
    ) -> Dict[str, Any]:
        """The quality metrics of one sample; GC figures in percent."""
        gc_percents = [gc * 100 for gc in gc_contents]
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
                "mean": statistics.mean(gc_percents),
                "median": statistics.median(gc_percents),
                "std": statistics.stdev(gc_percents) if len(gc_percents) > 1 else 0,
                # A histogram, not the raw per-read list: the list used to make every
                # profile JSON grow with the dataset instead of staying a fixed size.
                "histogram": _gc_histogram(gc_contents),
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
            "duplication_rate": self._calculate_duplication_rate(sequences),
        }

    def _sample_head(
        self, fastq_path: Path, sample_size: int
    ) -> Tuple[List[int], List[float], List[int], List[float], List[str]]:
        """First ``sample_size`` reads of ``fastq_path`` (the original, head-only sampling)."""
        import gzip

        read_lengths: List[int] = []
        gc_contents: List[float] = []
        quality_scores: List[int] = []
        n_contents: List[float] = []
        sequences: List[str] = []

        opener = gzip.open if fastq_path.suffix.endswith(".gz") else open
        with opener(fastq_path, "rt") as handle:
            read_count = 0
            for record in SeqIO.parse(handle, "fastq"):
                if read_count >= sample_size:
                    break

                sequence = str(record.seq)
                qualities = record.letter_annotations["phred_quality"]

                read_lengths.append(len(sequence))
                gc_contents.append(self._calculate_gc_content(sequence))
                quality_scores.extend(qualities)
                n_contents.append(sequence.count("N") / len(sequence))
                sequences.append(sequence)

                read_count += 1

        return read_lengths, gc_contents, quality_scores, n_contents, sequences

    def _sample_uniform(
        self, fastq_path: Union[Path, Sequence[Path]], sample_size: int
    ) -> Tuple[List[int], List[float], List[int], List[float], List[str]]:
        """Reservoir-sample ``sample_size`` reads uniformly over the whole of every file.

        Unlike ``_sample_head``, a read from anywhere in the files has an equal chance of
        being included, so a dataset whose later reads (or second mates) differ from its
        first ones is still represented in the quality metrics. Streams with
        ``iter_fastq_records`` (no Biopython) so a large file is never fully parsed
        record-by-record.
        """
        paths = [fastq_path] if isinstance(fastq_path, (str, Path)) else list(fastq_path)
        reservoir: List[Tuple[str, str]] = []
        seen = 0
        rng = random.Random(0)
        for seq, qual in itertools.chain.from_iterable(iter_fastq_records(p) for p in paths):
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append((seq, qual))
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_size:
                    reservoir[j] = (seq, qual)

        read_lengths: List[int] = []
        gc_contents: List[float] = []
        quality_scores: List[int] = []
        n_contents: List[float] = []
        sequences: List[str] = []
        for seq, qual in reservoir:
            read_lengths.append(len(seq))
            gc_contents.append(self._calculate_gc_content(seq))
            quality_scores.extend(ord(c) - 33 for c in qual)
            n_contents.append((seq.count("N") / len(seq)) if seq else 0.0)
            sequences.append(seq)

        return read_lengths, gc_contents, quality_scores, n_contents, sequences

    def _calculate_duplication_rate(self, sequences: List[str]) -> float:
        """Fraction of sampled reads that are exact duplicates of another read.

        Computed over the sampled sequences as ``1 - unique / total``; 0.0 when
        every read is distinct and rising toward 1.0 as duplicates dominate.
        """
        if not sequences:
            return 0.0
        return 1.0 - len(set(sequences)) / len(sequences)

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
