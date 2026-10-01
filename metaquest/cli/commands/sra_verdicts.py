"""The registry inputs and completeness verdicts ``download_sra`` records.

``registry_inputs`` collects what the download loop needs from the project registry, with each
accession's expected spot count falling back to the metadata XML (the project's metadata folder,
then the store's) when the registry records none. ``present_verdicts`` verifies an accession that
is on disk without a ``downloaded`` record, as a run killed after its files were in place leaves
it; it reads FASTQ files, so it runs before the registry lock is taken. ``linked_verdict`` is the
verdict to record for a dataset linked from the shared store: the sidecar's, recomputed when its
read count and a spot count are both known, and never an ``unverified`` one in place of a
recorded ``complete`` or ``truncated`` one.
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
from metaquest.data.sra.spots import merged_verdict, positive_int, spots_from_xml
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


def _xml_spots(
    accessions: Iterable[str], known: Mapping[str, int], folders: List[Path], excluded: set
) -> Dict[str, int]:
    """Spot counts from ``<ACC>_metadata.xml`` in ``folders`` for the accessions ``known`` lacks."""
    found: Dict[str, int] = {}
    for acc in accessions:
        if acc in known or acc in excluded or acc in found:
            continue
        spots = spots_from_xml(acc, folders)
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
    project's ``metadata`` folder and then in the store's. Truncated accessions are only collected
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
        expected_spots.update(_xml_spots(_run_accessions(args), expected_spots, folders, excluded))
    if getattr(args, "redownload_truncated", False):
        truncated = {
            acc
            for acc in project_registry.datasets
            if (verdict := rb.download_verdict(project_registry, acc)) is not None and verdict.verdict == "truncated"
        }
    return excluded, expected_spots, truncated, run_sizes


def _needs_verdict(fastq_dir: Path, acc: str, registry: Optional[Registry], expected: Mapping[str, int]) -> bool:
    """True when ``acc`` is a real folder in ``fastq_dir`` with a spot count and no ``downloaded`` record."""
    acc_dir = fastq_dir / acc
    if acc_dir.is_symlink() or not acc_dir.is_dir() or positive_int(expected.get(acc)) is None:
        return False
    block = rb.download_block(registry, acc) if registry is not None else None
    return block is None or block.state != "downloaded"


def present_verdicts(
    fastq_dir: Union[str, Path],
    accessions: Iterable[str],
    registry: Optional[Registry],
    expected: Mapping[str, int],
) -> Dict[str, Optional[dict]]:
    """The completeness verdict of each present accession that has no ``downloaded`` record.

    An accession is verified when its folder in ``fastq_dir`` is a real folder (a store link
    takes the sidecar's verdict instead), ``registry`` (a snapshot, read without the lock) has
    no ``downloaded`` record for it, and ``expected`` holds its spot count: without a count the
    verdict could only be ``unverified``, which is not worth reading every read of the file for.
    ``verify_download`` counts the mate-1 (or single-end) reads plus the unpaired ones. A file
    that cannot be read is logged and maps to None. Accessions not verified are left out.
    """
    root = Path(fastq_dir)
    verdicts: Dict[str, Optional[dict]] = {}
    for acc in accessions:
        if acc in verdicts or not _needs_verdict(root, acc, registry, expected):
            continue
        spots = positive_int(expected.get(acc))
        try:
            result = verify_download(acc, root / acc, spots)
        except (OSError, EOFError, ValueError) as e:
            logger.warning("Could not verify the reads of %s found on disk: %s", acc, e)
            verdicts[acc] = None
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
    reads = sidecar.get("reads_r1") if sidecar is not None else None
    if isinstance(reads, bool) or not isinstance(reads, int) or reads < 0:
        reads = None
    spots = positive_int(expected)
    if spots is None and sidecar is not None:
        spots = positive_int(sidecar.get("expected_spots"))
    return merged_verdict(previous, sidecar, reads, spots)
