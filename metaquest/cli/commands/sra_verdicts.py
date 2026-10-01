"""The registry inputs and completeness verdicts ``download_sra`` records.

``registry_inputs`` collects what the download loop needs from the project registry, with each
accession's expected spot count falling back to the metadata XML (the project's metadata folder,
then the store's) and then to the count of the verdict on file. ``present_verdicts`` verifies an
accession that is on disk without a ``downloaded`` record, as a run killed after its files were in
place leaves it; it reads FASTQ files, so it runs before the registry lock is taken.
``recorded_verdict`` is the verdict to record, under the lock, for a download, a present accession
or a dataset linked from the shared store (``linked_verdict``): recomputed when a read count and a
spot count are both known, and never an ``unverified`` one in place of a recorded ``complete`` or
``truncated`` one.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from metaquest.cli.base import read_accessions_file
from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry_blocks as rb
from metaquest.data.registry import Registry, project_root, query
from metaquest.data.sra.fastq import verify_download
from metaquest.data.sra.spots import merged_verdict, positive_int, spots_from_xml, verdict_for_count
from metaquest.store.layout import StorePaths

logger = logging.getLogger(__name__)

# The keys of a registry download verdict; ``verify_download`` also returns ``bytes_total``.
_VERDICT_KEYS = ("verdict", "ratio", "expected_spots", "reads_r1")


def _run_accessions(args: argparse.Namespace) -> List[str]:
    """The accessions this run names, or none when the accessions file cannot be read.

    ``download_sra`` reads the same file and reports a missing or unreadable one itself, so the
    spot count fallback only skips it here.
    """
    path = getattr(args, "accessions_file", None)
    if not path:
        return []
    try:
        return read_accessions_file(path)
    except DataAccessError as e:
        logger.debug("No spot counts from metadata XML: %s", e)
        return []


def _fallback_spots(
    accessions: Iterable[str], registry: Registry, known: Mapping[str, int], folders: List[Path], excluded: set
) -> Dict[str, int]:
    """Spot counts for the accessions ``known`` lacks: ``<ACC>_metadata.xml`` in ``folders``, then the verdict on file.

    The order follows ``metaquest.data.sra.spots.expected_spots`` after its registry metadata step:
    the XML holds NCBI's count, the recorded verdict only a copy an earlier run made.
    """
    found: Dict[str, int] = {}
    for acc in accessions:
        if acc in known or acc in excluded or acc in found:
            continue
        spots = spots_from_xml(acc, folders)
        if spots is None:
            previous = rb.download_verdict(registry, acc)
            spots = positive_int(previous.expected_spots) if previous is not None else None
        if spots is not None:
            found[acc] = spots
    return found


def registry_inputs(
    args: argparse.Namespace, project_registry: Registry, store: Optional[StorePaths]
) -> Tuple[set, dict, set, dict]:
    """The excluded accessions, expected spot counts, truncated accessions and run sizes in the registry.

    All four are empty for a dry run. Expected spot counts are only collected with
    ``--verify-downloads`` (the default): the registry's ``run_total_spots`` first, then, for an
    accession of this run without one, ``Run_Total_Spots`` in ``<ACC>_metadata.xml`` in the
    project's ``metadata`` folder and then in the store's, and last the ``expected_spots`` of the
    verdict on file (so a ``--redownload-truncated`` run judges the new files against the count the
    old ones were judged by). Truncated accessions are only collected
    with ``--redownload-truncated``. Run sizes (NCBI's ``.sra`` size in bytes) feed the free-space
    guard; an accession without one needs ``--min-free-gb`` instead.
    """
    excluded: set = set()
    expected_spots: dict = {}
    truncated: set = set()
    run_sizes: dict = {}
    if args.dry_run:
        return excluded, expected_spots, truncated, run_sizes
    excluded = set(query(project_registry, "excluded"))
    verify = getattr(args, "verify_downloads", True)
    for acc in project_registry.datasets:
        metadata = rb.metadata_block(project_registry, acc) or rb.MetadataBlock()
        spots = positive_int(metadata.run_total_spots)
        if verify and spots is not None:
            expected_spots[acc] = spots
        if metadata.run_size is not None:
            run_sizes[acc] = metadata.run_size
    if verify:
        folders = [project_root(project_registry) / "metadata"] + ([store.metadata] if store is not None else [])
        run = _run_accessions(args)
        expected_spots.update(_fallback_spots(run, project_registry, expected_spots, folders, excluded))
    if getattr(args, "redownload_truncated", False):
        truncated = {
            acc
            for acc in project_registry.datasets
            if (verdict := rb.download_verdict(project_registry, acc)) is not None and verdict.verdict == "truncated"
        }
    return excluded, expected_spots, truncated, run_sizes


def _unverified(spots: Optional[int] = None) -> dict:
    """The ``unverified`` verdict, in the four registry verdict keys, with ``spots`` as its count."""
    verdict = verdict_for_count(None, spots)
    return {key: verdict[key] for key in _VERDICT_KEYS}


def _needs_verdict(fastq_dir: Path, acc: str, registry: Optional[Registry]) -> bool:
    """True when ``acc`` is a real folder in ``fastq_dir`` and has no ``downloaded`` record."""
    acc_dir = fastq_dir / acc
    if acc_dir.is_symlink() or not acc_dir.is_dir():
        return False
    block = rb.download_block(registry, acc) if registry is not None else None
    return block is None or block.state != "downloaded"


def present_verdicts(
    fastq_dir: Union[str, Path],
    accessions: Iterable[str],
    registry: Optional[Registry],
    expected: Mapping[str, int],
) -> Dict[str, dict]:
    """The completeness verdict of each present accession that has no ``downloaded`` record.

    An accession gets a verdict when its folder in ``fastq_dir`` is a real folder (a store link
    takes the sidecar's verdict instead) and ``registry`` (a snapshot, read without the lock) has
    no ``downloaded`` record for it. With a spot count in ``expected``, ``verify_download`` counts
    the mate-1 (or single-end) reads plus the unpaired ones. Without one the verdict could only be
    ``unverified``, so no file is read and the ``unverified`` verdict is returned directly; a file
    that cannot be read is logged and gets the ``unverified`` verdict too. The caller merges each
    with the verdict on file (``recorded_verdict``), so a recorded ``truncated`` one is kept.
    Accessions that need no verdict are left out.
    """
    root = Path(fastq_dir)
    verdicts: Dict[str, dict] = {}
    for acc in accessions:
        if acc in verdicts or not _needs_verdict(root, acc, registry):
            continue
        spots = positive_int(expected.get(acc))
        if spots is None:
            verdicts[acc] = _unverified()
            continue
        try:
            result = verify_download(acc, root / acc, spots)
        except (OSError, EOFError, ValueError) as e:
            logger.warning("Could not verify the reads of %s found on disk, recording it unverified: %s", acc, e)
            verdicts[acc] = _unverified(spots)
            continue
        verdicts[acc] = {key: result[key] for key in _VERDICT_KEYS}
        if result["verdict"] == "truncated":
            logger.warning(
                "%s was found on disk without a download record and holds %s of %s spots (truncated); "
                "rerun with --redownload-truncated to download it again",
                acc,
                result["reads_r1"],
                spots,
            )
    return verdicts


def linked_verdict(
    previous: Optional[Union[Mapping[str, Any], rb.Verdict]],
    sidecar: Optional[Mapping[str, Any]],
    expected: Optional[int],
) -> Optional[Dict[str, Any]]:
    """The verdict to record for a dataset linked from the store, given the ``previous`` one.

    ``sidecar`` is the store sidecar's completeness verdict as ``sidecar_completeness`` shapes it,
    whose ``reads_r1`` is the sidecar's ``reads_per_mate``; it is read before the registry lock is
    taken, so nothing is read here. The spot count is ``expected`` when known, else the sidecar's
    own. With both counts known the verdict is recomputed; otherwise ``merged_verdict`` keeps a
    recorded ``complete`` or ``truncated`` verdict over an ``unverified`` one. Returns None when
    neither verdict is known.
    """
    spots = positive_int(expected)
    if spots is None and sidecar is not None:
        spots = positive_int(sidecar.get("expected_spots"))
    return merged_verdict(previous, sidecar, _read_count(sidecar), spots)


def _read_count(verdict: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The ``reads_r1`` of ``verdict`` when it is a whole number of zero or more, else None."""
    reads = verdict.get("reads_r1") if verdict is not None else None
    if isinstance(reads, bool) or not isinstance(reads, int) or reads < 0:
        return None
    return reads


def recorded_verdict(
    registry: Registry, accession: str, new: Optional[Mapping[str, Any]], expected: Optional[int], linked: bool
) -> Optional[Dict[str, Any]]:
    """The verdict to record for ``accession``, merged with the one ``registry`` holds; call under the lock.

    ``new`` is the store sidecar's verdict when ``linked`` (``linked_verdict``), otherwise the
    verdict of a download's result message or of ``present_verdicts``. With the read count it
    carries and ``expected`` both known the verdict is recomputed (a complete redownload replaces
    ``truncated``); otherwise a recorded ``complete`` or ``truncated`` verdict is kept over an
    ``unverified`` one. None (no ``new``) leaves the verdict on file as it is. Reads no file.
    """
    if new is None:
        return None
    previous = rb.download_verdict(registry, accession)
    if linked:
        return linked_verdict(previous, new, expected)
    return merged_verdict(previous, new, _read_count(new), positive_int(expected))
