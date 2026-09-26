"""Time bound for summarising a parsed containment table and recording it in the registry.

The bound is about three times the time measured after the numpy change, under ``make test``
(coverage on), on the development machine (Apple silicon laptop, 2026-09-26). A failure reports the
measured time, so a slower machine can be told apart from a regression by the figure below.
"""

import time

import pytest

from metaquest.data import registry as reg
from metaquest.data.branchwater import _generate_containment_summary
from tests.perf_containment import containment_data

BOUND_SECONDS = 1.2


@pytest.fixture(scope="module")
def containment_20000x5():
    """20,000 accessions by 5 genomes, about half the cells positive, the last genome all zeros."""
    return containment_data(20000, 5)


def test_summary_and_screening_on_20000_rows_is_fast(containment_20000x5, tmp_path):
    """Summary at step 0.001 plus screening into an empty registry, as ``parse_containment`` runs them.

    Measured 2026-09-26 under coverage: 0.39 s after the change (4.34 s before it, with one frame
    copy per threshold, one ``iterrows`` pass, a re-read of the written table and one screening
    block conversion and timestamp per positive cell). Bound: 1.2 s, the best of two runs.
    """
    timings = []
    for attempt in range(2):
        registry = reg.load_registry(tmp_path / f"registry_{attempt}.json")
        start = time.perf_counter()
        summary = _generate_containment_summary(
            containment_20000x5, tmp_path / "parsed.txt", tmp_path / "summary.txt", 0.001
        )
        recorded = reg.record_screening_from_table(registry, summary.table, tmp_path / "matches")
        timings.append(time.perf_counter() - start)
    best = min(timings)
    assert len(summary.sample_to_genomes) == 20000
    assert recorded > 0
    assert best < BOUND_SECONDS, f"summary and screening took {best:.2f} s on 20,000 x 5"
