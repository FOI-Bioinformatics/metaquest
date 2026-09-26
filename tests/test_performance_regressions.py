"""Performance regression tests with time bounds, on a synthetic 20,000-dataset registry.

Each bound is about three times the time measured after the corresponding fix, under
``make test`` (coverage on), on the development machine (Apple silicon laptop, 2026-09-26).
A test that fails reports the measured time, so a slower machine can be told apart from a
regression by comparing against the figure in the docstring.
"""

import argparse
import shutil
import time
from pathlib import Path

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


def test_write_registry_on_20000_datasets_is_fast(registry_20k, tmp_path):
    """Serialising and writing the 20,000-dataset registry.

    Measured 2026-09-26 under coverage: 0.09 s compact (0.33 s with the earlier indented output). Bound: 0.3 s,
    the best of three writes so a single slow disk flush does not fail the test.
    """
    # A copy of its own: the module-scoped registry is changed by the other test in this module.
    path = Path(shutil.copy(registry_20k[0], tmp_path / registry_mod.REGISTRY_FILENAME))
    registry = registry_mod.load_registry(path)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        registry_mod._write_registry(registry, path)
        timings.append(time.perf_counter() - start)
    best = min(timings)
    assert "\n  " not in path.read_text()[:10000]
    assert best < 0.3, f"_write_registry took {best:.2f} s on 20,000 datasets"


def _best_of_three(call):
    """The shortest of three timed runs of ``call``, with the result of the last run."""
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        result = call()
        timings.append(time.perf_counter() - start)
    return min(timings), result


def test_status_report_on_20000_datasets_is_fast(registry_20k):
    """``build_report``, the dict ``status --json`` prints, on the 20,000-dataset registry.

    Measured 2026-09-26 under coverage: 0.19 s after the fix (0.55 s before it, when the stages
    were counted with one pass over the registry per stage and genome and then listed again). Bound: 0.6 s,
    the best of three runs.
    """
    from metaquest.processing.status_report import build_report

    path, names = registry_20k
    root = path.parent
    registry = registry_mod.load_registry(path)
    args = argparse.Namespace(
        fastq_folder=str(root / "fastq"),
        metadata_folder=str(root / "metadata"),
        genomes_folder=str(root / "genomes"),
        accessions_file=None,
        parsed_containment=None,
        data_root=None,
        genome=None,
        init=False,
    )
    paths = registry_mod.ProjectPaths(fastq=root / "fastq", targeted=root / "targeted")
    best, report = _best_of_three(lambda: build_report(registry, args, paths, path, True))
    assert report["wanted"]["total"] == len(names["selected"])
    assert report["stages"]["screened"]["count"] == len(names["all"])
    assert best < 0.6, f"build_report took {best:.2f} s on 20,000 datasets"


def test_results_rows_on_20000_datasets_by_3_genomes_is_fast(registry_20k):
    """``results_rows`` for 60,000 (accession, genome) pairs, with a parsed containment table of 20,000 rows.

    Measured 2026-09-26 under coverage: 0.36 s after the fix (1.80 s before it, when every pair
    converted about six registry blocks and every table row became a pandas Series). Bound: 1.1 s, the
    best of three runs.
    """
    import numpy as np
    import pandas as pd

    from metaquest.processing.results import results_rows

    path, names = registry_20k
    registry = registry_mod.load_registry(path)
    values = np.random.default_rng(0).random((len(names["all"]), 3))
    table = pd.DataFrame(values, index=names["all"], columns=["G1", "G2", "G3"])
    table["max_containment"] = table.max(axis=1)
    best, rows = _best_of_three(lambda: results_rows(registry, parsed_table=table))
    assert len(rows) == 3 * len(names["all"])
    assert best < 1.1, f"results_rows took {best:.2f} s for 60,000 pairs"
