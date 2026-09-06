"""
One cached FASTQ statistics record per dataset.

Today three commands read the same FASTQ files three different ways: ``sra_stats`` parses
every read in pure Python, ``sra_profile_quality`` samples the first 10,000 reads, and
``sra_compare``/anomaly detection reprofile from scratch every time. ``compute_dataset_stats``
computes one record covering what all three need; ``cached_stats`` and ``store_stats`` are
the sidecar-backed cache around it (see ``metaquest.store.sidecar.Sidecar.stats``), so a
dataset already profiled once is not re-read for the next command.

Read counts and (when available) bases/min/max/avg length come from an exact pass: a
streaming newline count (``metaquest.data.sra.count_fastq_reads``), or ``seqkit stats -T``
when the tool is installed, which is both exact and faster. Everything that needs per-read
content (GC, quality, complexity, length distribution, N50) comes from a uniform reservoir
sample over the first file, so a large dataset is never fully parsed in Python.
"""

import logging
import random
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from metaquest.core.constants import DEFAULT_NUM_THREADS
from metaquest.data.sra import count_fastq_reads, fastq_files, iter_fastq_records
from metaquest.store.sidecar import read_sidecar, write_sidecar
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)

# Default reservoir sample size for the quality/GC/complexity part of the stats record.
DEFAULT_SAMPLE_SIZE = 10000

# Read-length histogram buckets (inclusive bounds), matching the convention already used by
# metaquest.sra.analytics's per-read length distribution; a length past the last bound falls
# into "1000+".
_LENGTH_BUCKETS = [(0, 50), (51, 100), (101, 150), (151, 250), (251, 500), (501, 1000)]

_REQUIRED_SEQKIT_COLUMNS = ("num_seqs", "sum_len", "min_len", "avg_len", "max_len")


def _file_signature(files: List[Path]) -> Dict[str, List[Union[int, float]]]:
    """``{file name: [size bytes, mtime]}`` snapshot used to detect a changed dataset."""
    signature: Dict[str, List[Union[int, float]]] = {}
    for f in files:
        st = f.stat()
        signature[f.name] = [st.st_size, st.st_mtime]
    return signature


def _reservoir_sample(path: Path, sample_size: int) -> Tuple[List[str], List[str], int]:
    """Reservoir-sample up to ``sample_size`` (sequence, quality) pairs uniformly from ``path``.

    Returns ``(sequences, qualities, reads_seen)``. Streams the file once with
    ``iter_fastq_records``; every read past ``sample_size`` has an equal chance of replacing
    an already-sampled read, so the sample is not biased toward the file's head. Seeded for
    reproducibility across repeated runs on the same file.
    """
    sequences: List[str] = []
    qualities: List[str] = []
    seen = 0
    rng = random.Random(0)
    for seq, qual in iter_fastq_records(path):
        seen += 1
        if len(sequences) < sample_size:
            sequences.append(seq)
            qualities.append(qual)
        else:
            j = rng.randint(0, seen - 1)
            if j < sample_size:
                sequences[j] = seq
                qualities[j] = qual
    return sequences, qualities, seen


def _gc_fraction(seq: str) -> float:
    """Fraction of ``seq`` that is G or C; 0.0 for an empty sequence."""
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _n50(lengths: List[int]) -> int:
    """N50 of ``lengths``: the length at which the cumulative sum first reaches half the total."""
    if not lengths:
        return 0
    sorted_lengths = sorted(lengths, reverse=True)
    target = sum(sorted_lengths) / 2
    cumulative = 0
    for length in sorted_lengths:
        cumulative += length
        if cumulative >= target:
            return length
    return 0


def _length_histogram(lengths: List[int]) -> Dict[str, int]:
    """Bucket read lengths into named ranges, e.g. ``{"101-150": 42, ...}``."""
    histogram = {f"{lo}-{hi}": 0 for lo, hi in _LENGTH_BUCKETS}
    histogram["1000+"] = 0
    for length in lengths:
        for lo, hi in _LENGTH_BUCKETS:
            if lo <= length <= hi:
                histogram[f"{lo}-{hi}"] += 1
                break
        else:
            histogram["1000+"] += 1
    return histogram


def _percentile(ordered: List[int], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list; 0.0 for an empty list."""
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * (pct / 100)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - k) + ordered[hi] * (k - lo))


def _quality_summary(scores: List[int]) -> Dict[str, float]:
    """Mean/median/q25/q75 of flattened per-base Phred quality scores."""
    if not scores:
        return {"mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    ordered = sorted(scores)
    return {
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "q25": _percentile(ordered, 25),
        "q75": _percentile(ordered, 75),
    }


def _parse_seqkit_table(stdout: str, files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Parse ``seqkit stats -T`` tab-delimited output into ``{file name: {column: value}}``.

    Columns are located by name in the header row rather than by fixed position, since a
    seqkit version may add columns; a comma in a numeric field (some builds format large
    counts that way even in ``-T`` mode) is stripped before conversion. Raises ``ValueError``
    when the expected columns or row count are not found, which the caller treats as a
    seqkit failure and falls back to the streaming count.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("seqkit stats produced no data rows")

    header = lines[0].split("\t")
    index = {name: i for i, name in enumerate(header)}
    if not all(name in index for name in _REQUIRED_SEQKIT_COLUMNS):
        raise ValueError(f"seqkit stats output missing expected columns: {header}")

    data_lines = lines[1:]
    if len(data_lines) != len(files):
        raise ValueError(f"seqkit stats row count ({len(data_lines)}) did not match input file count ({len(files)})")

    rows: Dict[str, Dict[str, Any]] = {}
    for file_path, line in zip(files, data_lines):
        cols = line.split("\t")
        rows[file_path.name] = {
            "num_seqs": int(cols[index["num_seqs"]].replace(",", "")),
            "sum_len": int(cols[index["sum_len"]].replace(",", "")),
            "min_len": int(cols[index["min_len"]].replace(",", "")),
            "avg_len": float(cols[index["avg_len"]].replace(",", "")),
            "max_len": int(cols[index["max_len"]].replace(",", "")),
        }
    return rows


def _run_seqkit_stats(files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Run ``seqkit stats -T -j <threads> <files...>`` and parse its output table."""
    args = ["stats", "-T", "-j", str(DEFAULT_NUM_THREADS), *(str(f) for f in files)]
    result = SecureSubprocess.run_secure("seqkit", args)
    return _parse_seqkit_table(result.stdout or "", files)


def _try_seqkit_stats(file_paths: List[Path], use_seqkit: bool) -> Optional[Dict[str, Dict[str, Any]]]:
    """``_run_seqkit_stats`` when requested and installed, else None; a failure logs and falls back."""
    if not (use_seqkit and shutil.which("seqkit")):
        return None
    try:
        return _run_seqkit_stats(file_paths)
    except Exception as e:
        logger.warning("seqkit stats failed for %s, falling back to streaming counts: %s", file_paths, e)
        return None


def _counts_from_seqkit(seqkit_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Exact reads/bases/length figures straight from a parsed seqkit stats table."""
    reads_per_file = {name: row["num_seqs"] for name, row in seqkit_rows.items()}
    reads_total = sum(reads_per_file.values())
    bases_total = sum(row["sum_len"] for row in seqkit_rows.values())
    return {
        "reads_per_file": reads_per_file,
        "reads_total": reads_total,
        "bases_total": bases_total,
        "min_read_length": min(row["min_len"] for row in seqkit_rows.values()),
        "max_read_length": max(row["max_len"] for row in seqkit_rows.values()),
        "avg_read_length": (bases_total / reads_total) if reads_total else 0.0,
    }


def _counts_from_streaming(file_paths: List[Path]) -> Dict[str, Any]:
    """Exact read counts from a streaming newline count; length/base figures filled in later
    from the reservoir sample, since a full pass just to measure lengths would defeat the
    point of not using seqkit."""
    reads_per_file = {f.name: count_fastq_reads(f) for f in file_paths}
    return {
        "reads_per_file": reads_per_file,
        "reads_total": sum(reads_per_file.values()),
        "bases_total": None,
        "min_read_length": None,
        "max_read_length": None,
        "avg_read_length": None,
    }


def _approximate_lengths_from_sample(counts: Dict[str, Any], sample_lengths: List[int]) -> Dict[str, Any]:
    """Fill in length/base figures from the reservoir sample when there is no exact count."""
    avg_read_length = _mean(sample_lengths)
    counts["avg_read_length"] = avg_read_length
    counts["min_read_length"] = min(sample_lengths) if sample_lengths else 0
    counts["max_read_length"] = max(sample_lengths) if sample_lengths else 0
    counts["bases_total"] = int(round(avg_read_length * counts["reads_total"]))
    return counts


def compute_dataset_stats(
    files: List[Union[str, Path]], sample_size: int = DEFAULT_SAMPLE_SIZE, use_seqkit: bool = True
) -> Dict[str, Any]:
    """Compute the shared statistics record for one dataset's FASTQ files.

    ``reads_per_file``/``reads_total`` (and, when available, ``bases_total`` and
    ``min``/``avg``/``max_read_length``) are exact: from ``seqkit stats -T`` when
    ``use_seqkit`` and the tool is installed (a seqkit failure logs a warning and falls back
    rather than raising), otherwise from a streaming newline count, with the length figures
    then approximated from the sample below. Everything that needs per-read content
    (``gc_content``, ``n_content``, ``length_histogram``, ``n50``, ``quality_summary``,
    ``duplication_rate``) comes from a uniform reservoir sample of ``sample_size`` reads over
    the first file; ``sampled`` is True when that file holds more reads than the sample.

    ``signature`` records each file's size and mtime so ``cached_stats`` can tell whether the
    dataset on disk still matches a cached copy of this result.
    """
    if not files:
        raise ValueError("compute_dataset_stats requires at least one FASTQ file")
    file_paths = [Path(f) for f in files]
    signature = _file_signature(file_paths)

    seqkit_rows = _try_seqkit_stats(file_paths, use_seqkit)
    counts = _counts_from_seqkit(seqkit_rows) if seqkit_rows is not None else _counts_from_streaming(file_paths)

    sequences, qualities, seen_in_sample_file = _reservoir_sample(file_paths[0], sample_size)
    sampled = seen_in_sample_file > sample_size
    sample_lengths = [len(s) for s in sequences]

    if seqkit_rows is None:
        counts = _approximate_lengths_from_sample(counts, sample_lengths)

    return {
        **counts,
        "n50": _n50(sample_lengths),
        "gc_content": _mean([_gc_fraction(s) for s in sequences]),
        "length_histogram": _length_histogram(sample_lengths),
        "quality_summary": _quality_summary([ord(c) - 33 for q in qualities for c in q]),
        "duplication_rate": (1.0 - len(set(sequences)) / len(sequences)) if sequences else 0.0,
        "n_content": _mean([(s.count("N") / len(s)) if s else 0.0 for s in sequences]),
        "sample_size": sample_size,
        "sampled": sampled,
        "computed": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
    }


def cached_stats(acc_dir: Union[str, Path], sidecar_path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    """The sidecar's cached stats when its recorded signature matches the files in ``acc_dir`` now.

    Returns None when there is no sidecar path, no readable sidecar, no stats recorded yet,
    or any file's size/mtime has changed since the stats were computed (a file added,
    removed, resized, or rewritten in place).
    """
    if sidecar_path is None:
        return None
    sidecar = read_sidecar(sidecar_path)
    if sidecar is None or not sidecar.stats:
        return None
    signature = sidecar.stats.get("signature")
    if not signature:
        return None
    current = _file_signature(fastq_files(acc_dir))
    if current != signature:
        return None
    return sidecar.stats


def store_stats(sidecar_path: Optional[Union[str, Path]], stats: Dict[str, Any]) -> None:
    """Write ``stats`` into the sidecar at ``sidecar_path``, with a fresh ``stats_computed``.

    A no-op when there is no sidecar to update (``sidecar_path`` is None, or names a sidecar
    that does not exist) -- a project without a shared store then keeps the stats only in its
    own analysis record rather than in a sidecar that was never created.
    """
    if sidecar_path is None:
        return
    sidecar = read_sidecar(sidecar_path)
    if sidecar is None:
        return
    sidecar.stats = stats
    sidecar.stats_computed = datetime.now(timezone.utc).isoformat()
    write_sidecar(sidecar_path, sidecar)
