"""Performance regression tests with time bounds, on a synthetic 20,000-dataset registry.

Each bound is about three times the time measured after the corresponding fix, under
``make test`` (coverage on), on the development machine (Apple silicon laptop, 2026-09-26).
A test that fails reports the measured time, so a slower machine can be told apart from a
regression by comparing against the figure in the docstring.
"""

import argparse
import time

import pytest

from metaquest.cli.commands.sra import DownloadSraCommand
from metaquest.data import registry as registry_mod
from metaquest.data import registry_blocks as rb
from perf_fixtures import build_registry


@pytest.fixture(scope="module")
def registry_20k(tmp_path_factory):
    """A 20,000-dataset registry file and the accession lists ``build_registry`` wrote."""
    root = tmp_path_factory.mktemp("perf_registry")
    path = root / registry_mod.REGISTRY_FILENAME
    names = build_registry(path)
    return path, names


def test_record_run_outcomes_with_5000_skipped_is_one_fast_transaction(registry_20k, monkeypatch):
    """5,000 skipped accessions are recorded in one registry write.

    Measured 2026-09-26 under coverage: 0.37 s after the fix (about 0.4 s per accession, over 30 minutes in
    total, before it). Bound: 1.1 s.
    """
    path, names = registry_20k
    skipped = names["all"][-5000:]
    args = argparse.Namespace(registry=str(path))
    writes = []
    real_write = registry_mod._write_registry

    def counting_write(registry, target):
        writes.append(target)
        return real_write(registry, target)

    monkeypatch.setattr(registry_mod, "_write_registry", counting_write)
    start = time.perf_counter()
    DownloadSraCommand()._record_run_outcomes(args, {"skipped_accessions": skipped}, path.parent / "fastq")
    elapsed = time.perf_counter() - start
    monkeypatch.undo()
    assert len(writes) == 1
    assert rb.download_block(registry_mod.load_registry(path), skipped[-1]).state == "skipped"
    assert elapsed < 1.1, f"_record_run_outcomes took {elapsed:.2f} s for 5,000 skipped accessions"


def test_write_registry_on_20000_datasets_is_fast(registry_20k):
    """Serialising and writing the 20,000-dataset registry.

    Measured 2026-09-26 under coverage: 0.09 s compact (0.33 s with the earlier indented output). Bound: 0.3 s,
    the best of three writes so a single slow disk flush does not fail the test.
    """
    path, _ = registry_20k
    registry = registry_mod.load_registry(path)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        registry_mod._write_registry(registry, path)
        timings.append(time.perf_counter() - start)
    best = min(timings)
    assert "\n  " not in path.read_text()[:10000]
    assert best < 0.3, f"_write_registry took {best:.2f} s on 20,000 datasets"
