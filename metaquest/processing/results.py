"""The consolidated results table: one row per screened (accession, genome) pair.

Joins what the project registry records for each pair (screening, selection, exclusion,
download, run metadata, read extraction, reference coverage and assembly) with the
containment values of the parsed containment table, which are unrounded and not limited
by ``cap_screening``.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from metaquest.data.registry import Registry, to_int_or_none

RESULTS_COLUMNS = [
    "accession",
    "genome_id",
    "containment",
    "selected",
    "excluded",
    "exclusion_reason",
    "download_state",
    "run_total_spots",
    "run_size",
    "mapped_reads",
    "mapping_rate_to_reference",
    "breadth",
    "mean_depth",
    "contigs",
    "total_bp",
    "n50",
    "genome_fraction_estimate",
    "assembly_mapping_rate",
]

_SUMMARY_COLUMNS = ("max_containment", "max_containment_annotation")

Pair = Tuple[str, str]


def _positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or number <= 0:
        return None
    return number


def screened_pairs(registry: Registry, parsed_table: Optional[pd.DataFrame] = None) -> Dict[Pair, Optional[float]]:
    """Every (accession, genome) pair with its containment, or None when it was never screened.

    Pairs come from three sources: the registry's ``screening.genomes`` entries; every genome
    column of ``parsed_table`` (all columns except ``max_containment`` and
    ``max_containment_annotation``) with a value above 0; and extraction records with no
    screening entry (containment None). Where the registry and the table both hold a value,
    the table's is used, since the registry rounds it to four decimals.
    """
    pairs: Dict[Pair, Optional[float]] = {}
    for accession, record in registry.datasets.items():
        for genome_id, entry in ((record.get("screening") or {}).get("genomes") or {}).items():
            value = (entry or {}).get("containment")
            pairs[(accession, genome_id)] = float(value) if value is not None else None
    if parsed_table is not None:
        genome_columns = [c for c in parsed_table.columns if c not in _SUMMARY_COLUMNS]
        for accession, row in parsed_table.iterrows():
            for genome_id in genome_columns:
                value = _positive_float(row[genome_id])
                if value is not None:
                    pairs[(str(accession), str(genome_id))] = value
    for accession, record in registry.datasets.items():
        for genome_id, entry in (record.get("extractions") or {}).items():
            if entry is not None:
                pairs.setdefault((accession, genome_id), None)
    return pairs


def _mapping_rate(mapped_reads: Optional[int], spots: Optional[int]) -> Optional[float]:
    """Mapped reads over the run's spot count, or None when either is missing or zero.

    ``mapped_reads`` counts BAM records that passed the extraction filters, while a spot is one
    read or one read pair, so paired-end data can give a value above 1. The ratio is reported
    as is, without halving, since whether both mates mapped is not known here.
    """
    if not mapped_reads or not spots or mapped_reads <= 0 or spots <= 0:
        return None
    return mapped_reads / spots


def _row(registry: Registry, accession: str, genome_id: str, containment: Optional[float]) -> Dict[str, Any]:
    record = registry.datasets.get(accession) or {}
    selection = record.get("selection") or {}
    exclusion = record.get("exclusion") or {}
    excluded = bool(exclusion.get("excluded"))
    metadata = record.get("metadata") or {}
    extraction = (record.get("extractions") or {}).get(genome_id) or {}
    assembly = extraction.get("assembly") or {}
    spots = to_int_or_none(metadata.get("run_total_spots"))
    mapped_reads = to_int_or_none(extraction.get("mapped_reads"))
    return {
        "accession": accession,
        "genome_id": genome_id,
        "containment": containment,
        "selected": bool(selection.get("selected")),
        "excluded": excluded,
        "exclusion_reason": (exclusion.get("reason") or None) if excluded else None,
        "download_state": (record.get("download") or {}).get("state"),
        "run_total_spots": spots,
        "run_size": to_int_or_none(metadata.get("run_size")),
        "mapped_reads": mapped_reads,
        "mapping_rate_to_reference": _mapping_rate(mapped_reads, spots),
        "breadth": extraction.get("breadth"),
        "mean_depth": extraction.get("mean_depth"),
        "contigs": assembly.get("contigs"),
        "total_bp": assembly.get("total_bp"),
        "n50": assembly.get("n50"),
        "genome_fraction_estimate": assembly.get("genome_fraction_estimate"),
        "assembly_mapping_rate": assembly.get("mapping_rate"),
    }


def results_rows(
    registry: Registry,
    parsed_table: Optional[pd.DataFrame] = None,
    genome_id: Optional[str] = None,
    min_containment: float = 0.0,
) -> List[Dict[str, Any]]:
    """One dict per (accession, genome) pair, keyed by ``RESULTS_COLUMNS`` in that order.

    ``genome_id`` keeps only that genome's pairs. ``min_containment`` keeps pairs whose
    containment is at least that value; above 0 it also drops pairs with no containment
    (extracted but never screened), since those cannot be shown to meet it. Rows are sorted
    by decreasing containment, with unknown containment last, then by accession and genome.
    """
    rows = []
    for (accession, genome), containment in screened_pairs(registry, parsed_table).items():
        if genome_id is not None and genome != genome_id:
            continue
        if min_containment > 0 and (containment is None or containment < min_containment):
            continue
        rows.append(_row(registry, accession, genome, containment))
    rows.sort(
        key=lambda row: (
            row["containment"] is None,
            -(row["containment"] or 0.0),
            row["accession"],
            row["genome_id"],
        )
    )
    return rows


def results_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """The rows as a DataFrame with exactly ``RESULTS_COLUMNS`` (header only when ``rows`` is empty).

    Built with object dtype so integer columns with missing values stay integers in the
    written table instead of turning into floats.
    """
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS, dtype=object)
