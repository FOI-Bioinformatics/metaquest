"""Expected spot counts of an SRA accession and the completeness verdict a read count gives against them.

A download's completeness verdict compares the reads on disk with NCBI's recorded spot count
(``Run_Total_Spots``). That count can be known in several places: the project registry's metadata
block, an earlier download verdict, the store sidecar's ``ncbi`` block, or a metadata XML file on
disk. ``expected_spots`` is the one lookup that consults them in a fixed order, so every caller
gets the same count for the same accession, and ``merged_verdict`` is the one rule for combining a
new verdict with the recorded one: a ``truncated`` verdict is never replaced by ``unverified``,
since the absence of a spot count is not evidence that the reads are now complete.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry_blocks as rb
from metaquest.data.sra.fastq import COMPLETE_RATIO_THRESHOLD

if TYPE_CHECKING:
    from metaquest.data.registry import Registry
    from metaquest.store.layout import StorePaths

logger = logging.getLogger(__name__)

_UNVERIFIED = "unverified"


def _positive_int(value: Any) -> Optional[int]:
    """``value`` as an int when it is a positive whole number (or its string form), else None.

    A spot count of zero is treated as unknown, as ``verify_download`` has always done.
    """
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def verdict_for_count(reads_r1: Optional[int], expected_spots: Optional[int]) -> Dict[str, Any]:
    """The completeness verdict of ``reads_r1`` reads against ``expected_spots`` NCBI spots.

    ``"complete"`` when the ratio of reads to spots is at least ``COMPLETE_RATIO_THRESHOLD``,
    ``"truncated"`` below it, and ``"unverified"`` when either count is unknown (None) or the spot
    count is zero. The ratio is rounded to four decimals. Returns a dict with the keys
    ``method``, ``ratio``, ``verdict``, ``expected_spots`` and ``reads_r1``.
    """
    ratio: Optional[float]
    if expected_spots and reads_r1 is not None:
        ratio = round(reads_r1 / expected_spots, 4)
        verdict = "complete" if ratio >= COMPLETE_RATIO_THRESHOLD else "truncated"
        method = "spots"
    else:
        ratio = None
        verdict = _UNVERIFIED
        method = _UNVERIFIED
    return {
        "method": method,
        "ratio": ratio,
        "verdict": verdict,
        "expected_spots": expected_spots,
        "reads_r1": reads_r1,
    }


def spots_from_xml(accession: str, folders: Iterable[Union[str, Path]]) -> Optional[int]:
    """``Run_Total_Spots`` from the first ``<accession>_metadata.xml`` in ``folders`` that records one.

    A folder without the file is skipped. A file that cannot be read or parsed (``OSError``,
    ``ValueError``, ``xml.etree.ElementTree.ParseError``) is logged and skipped. Returns None when
    no folder yields a positive spot count.
    """
    # Imported here: metaquest.data.metadata pulls in pandas, Biopython and lxml, which the
    # download path does not otherwise need at import time.
    from metaquest.data.metadata import parse_metadata_xml

    for folder in folders:
        xml_path = Path(folder) / f"{accession}_metadata.xml"
        if not xml_path.is_file():
            continue
        try:
            parsed = parse_metadata_xml(xml_path)
        except (OSError, ValueError, ET.ParseError) as e:
            logger.warning("Could not read the spot count of %s from %s: %s", accession, xml_path, e)
            continue
        spots = _positive_int((parsed or {}).get("Run_Total_Spots"))
        if spots is not None:
            return spots
    return None


def _sidecar_spots(accession: str, store: Union["StorePaths", str, Path]) -> Optional[int]:
    """``ncbi.spots`` from ``accession``'s store sidecar, or None when there is no readable sidecar."""
    # Imported here: metaquest.store imports metaquest.data.sra, so a module-level import would
    # form a cycle.
    from metaquest.store.layout import StorePaths, sidecar_path, store_paths
    from metaquest.store.sidecar import read_sidecar

    paths = store if isinstance(store, StorePaths) else store_paths(Path(store))
    path = sidecar_path(paths, accession)
    if not path.is_file():
        return None
    try:
        sidecar = read_sidecar(path)
    except DataAccessError as e:
        logger.warning("Ignoring the sidecar of %s: %s", accession, e)
        return None
    if sidecar is None or not isinstance(sidecar.ncbi, dict):
        return None
    return _positive_int(sidecar.ncbi.get("spots"))


def expected_spots(
    registry: Optional["Registry"],
    accession: str,
    store: Optional[Union["StorePaths", str, Path]] = None,
    xml_folders: Iterable[Union[str, Path]] = (),
) -> Optional[int]:
    """NCBI's recorded spot count for ``accession``, or None when no source records one.

    The sources are consulted in this order, and the first positive count wins:

    1. the registry's metadata block (``run_total_spots``);
    2. the ``expected_spots`` of the download verdict the registry already records;
    3. ``ncbi.spots`` in the store sidecar, when ``store`` (a ``StorePaths`` or a store root) is given;
    4. ``Run_Total_Spots`` in ``<accession>_metadata.xml`` in each of ``xml_folders``, in order.
    """
    if registry is not None:
        metadata = rb.metadata_block(registry, accession)
        if metadata is not None:
            spots = _positive_int(metadata.run_total_spots)
            if spots is not None:
                return spots
        previous = rb.download_verdict(registry, accession)
        if previous is not None:
            spots = _positive_int(previous.expected_spots)
            if spots is not None:
                return spots
    if store is not None:
        spots = _sidecar_spots(accession, store)
        if spots is not None:
            return spots
    return spots_from_xml(accession, xml_folders)


def _as_dict(verdict: Any) -> Optional[Dict[str, Any]]:
    """A plain-dict copy of ``verdict`` (a mapping or a registry ``Verdict``), or None."""
    if verdict is None:
        return None
    if isinstance(verdict, rb.RegistryBlock):
        return verdict.to_dict()
    if isinstance(verdict, Mapping):
        return dict(verdict)
    raise TypeError(f"Not a verdict: {verdict!r}")


def merged_verdict(
    previous: Optional[Union[Mapping[str, Any], "rb.Verdict"]],
    new: Optional[Union[Mapping[str, Any], "rb.Verdict"]],
    reads_r1: Optional[int],
    expected: Optional[int],
) -> Optional[Dict[str, Any]]:
    """The completeness verdict to record, given the ``previous`` one and a ``new`` one.

    When both ``reads_r1`` and a spot count ``expected`` are known, the verdict is recomputed from
    them with ``verdict_for_count``, which may turn a recorded ``truncated`` into ``complete``.
    Otherwise ``new`` is taken, except that a ``previous`` ``truncated`` verdict is never replaced by
    an ``unverified`` (or missing) one: an unknown spot count says nothing about whether the reads
    are now complete. A missing ``new`` keeps ``previous``. Returns None when neither is known. The
    result is always a new dict; the arguments are not modified.
    """
    if reads_r1 is not None and _positive_int(expected) is not None:
        return verdict_for_count(reads_r1, _positive_int(expected))
    before = _as_dict(previous)
    after = _as_dict(new)
    if after is None:
        return before
    if before is not None and before.get("verdict") == "truncated" and after.get("verdict") in (None, _UNVERIFIED):
        return before
    return after
