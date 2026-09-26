"""Synthetic FASTQ writers for the sampler tests and performance checks.

Kept apart from ``tests/perf_fixtures.py`` for now; the two are meant to be merged later.
"""

import gzip
from pathlib import Path
from typing import Callable, Optional


def write_fastq(
    path: Path,
    n: int,
    seq_for: Optional[Callable[[int], str]] = None,
    qual_for: Optional[Callable[[int], str]] = None,
) -> Path:
    """Write ``n`` four-line records to ``path`` (gzip-compressed when it ends in ``.gz``).

    Record ``i`` is named ``r<i>``; its sequence is ``seq_for(i)`` (default ``f"A{i}"``) and
    its quality ``qual_for(i)`` (default ``"I"`` repeated to the sequence length).
    """
    seq_for = seq_for or (lambda i: f"A{i}")
    parts = []
    for i in range(n):
        seq = seq_for(i)
        qual = qual_for(i) if qual_for else "I" * len(seq)
        parts.append(f"@r{i}\n{seq}\n+\n{qual}\n")
    data = "".join(parts).encode("ascii")
    if str(path).endswith(".gz"):
        with gzip.open(path, "wb", compresslevel=1) as handle:
            handle.write(data)
    else:
        path.write_bytes(data)
    return path


def write_illumina_like_fastq_gz(path: Path, n: int, read_length: int = 150) -> Path:
    """Write ``n`` gzip-compressed records of ``read_length`` bases with varied bases and qualities.

    Sequences and qualities are shifted windows over fixed patterns, which is quick to
    generate; the content does not matter for timing.
    """
    bases = "ACGTTGCAAGCTTCGA" * (read_length // 16 + 2)
    quals = "".join(chr(33 + (i * 7) % 41) for i in range(read_length + 16))
    parts = []
    for i in range(n):
        offset = i % 16
        parts.append(f"@SRR0.{i} {i} length={read_length}\n")
        parts.append(bases[offset : offset + read_length] + "\n+\n")
        parts.append(quals[offset : offset + read_length] + "\n")
    with gzip.open(path, "wb", compresslevel=1) as handle:
        handle.write("".join(parts).encode("ascii"))
    return path
