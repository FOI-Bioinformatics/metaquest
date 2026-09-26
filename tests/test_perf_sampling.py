"""Performance check for the index-based FASTQ sampler (metaquest.data.sra.sampling)."""

import time

import pytest

from metaquest.data.sra import sample_records
from tests.perf_fastq import write_illumina_like_fastq_gz


@pytest.fixture(scope="module")
def fastq_100k(tmp_path_factory):
    """A 100,000-record gzip FASTQ file of 150 bp reads."""
    return write_illumina_like_fastq_gz(tmp_path_factory.mktemp("perf") / "SRR0_1.fastq.gz", 100_000)


def test_sample_records_100k_to_10k_is_fast(fastq_100k):
    """Sampling 10,000 of 100,000 gzip records, record count included.

    Measured at 0.07 s (0.08 s under coverage) on an Apple-silicon laptop on 2026-09-26; the
    reservoir sampler this replaced took 0.11 s here and scaled worse on a real run (25 s for
    ``sra_profile --sample-size 10000`` on an 11.3M-record mate file). The bound is three times
    the measured time.
    """
    start = time.perf_counter()
    got = sample_records([fastq_100k], 10_000, seed=0)
    elapsed = time.perf_counter() - start
    assert len(got) == 10_000
    assert elapsed < 0.24, f"sample_records took {elapsed:.2f} s"
