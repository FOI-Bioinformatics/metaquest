"""Time bound for parsing a folder of metadata XML files into the metadata table.

The bound is about three times the time measured after the single-pass change, under ``make test``
(coverage on), on the development machine (Apple silicon laptop, 2026-09-26). A failure reports the
measured time, so a slower machine can be told apart from a regression by the figure below.
"""

import time

import pytest

from metaquest.data.metadata import parse_metadata
from tests.perf_metadata import write_metadata_folder


@pytest.fixture(scope="module")
def metadata_folder_300(tmp_path_factory):
    """300 synthetic metadata XML files, 40 sample attributes each, from overlapping tag sets."""
    folder = tmp_path_factory.mktemp("perf_metadata") / "metadata"
    write_metadata_folder(folder, count=300, per_file=40)
    return folder


def test_parse_metadata_on_300_files_is_fast(metadata_folder_300, tmp_path):
    """Parsing 300 files into a 300-row table with every distinct attribute as a column.

    Measured 2026-09-26 under coverage: 0.12 s after the change (0.28 s before it, with two parses
    per file, list-based tag lookups, dense rows and pandas' default CSV chunking). Bound: 0.37 s,
    the best of three runs. The bound alone does not tell the two apart at this size; the one-parse-
    per-file spy test in test_data_metadata.py guards the main change.
    """
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        table = parse_metadata(metadata_folder_300, tmp_path / "metadata_table.txt")
        timings.append(time.perf_counter() - start)
    best = min(timings)
    assert table.shape[0] == 300
    assert table.shape[1] > 30 + 40
    assert best < 0.37, f"parse_metadata took {best:.2f} s on 300 files"
