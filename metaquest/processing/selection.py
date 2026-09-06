"""Select SRA accessions from a parsed containment table, optionally filtered by metadata."""

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError, ProcessingError

logger = logging.getLogger(__name__)

DEFAULT_COLUMN = "max_containment"


def _check_columns_exist(containment: pd.DataFrame, columns: List[str], table_name: str) -> None:
    missing = [col for col in columns if col not in containment.columns]
    if missing:
        raise ProcessingError(
            f"Column '{missing[0]}' not found in {table_name}. Available columns: "
            f"{', '.join(str(c) for c in containment.columns)}"
        )


def _rank_single_column(containment: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(containment[column], errors="coerce").fillna(0.0)
    return values[values >= threshold].sort_values(ascending=False)


def _rank_multi_column(
    containment: pd.DataFrame, genome_ids: List[str], threshold: float, require: str
) -> Tuple[str, pd.Series]:
    numeric = containment[genome_ids].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if require == "all":
        values = numeric.min(axis=1)
        keep = numeric.ge(threshold).all(axis=1)
    else:
        values = numeric.max(axis=1)
        keep = values >= threshold
    column = "+".join(genome_ids)
    return column, values[keep].sort_values(ascending=False)


def _filter_by_metadata(
    ranked: List[Tuple[str, str, float]],
    metadata_file: Optional[Union[str, Path]],
    metadata_column: str,
    metadata_value: str,
) -> List[Tuple[str, str, float]]:
    if metadata_file is None:
        raise ProcessingError("metadata_file is required when filtering on metadata")
    meta_path = Path(metadata_file)
    if not meta_path.exists():
        raise DataAccessError(f"Metadata table not found: {meta_path}")
    metadata = pd.read_csv(meta_path, sep="\t", index_col=0, dtype=str)
    if metadata_column not in metadata.columns:
        raise ProcessingError(
            f"Column '{metadata_column}' not found in {meta_path.name}. Available columns: "
            f"{', '.join(str(c) for c in metadata.columns)}"
        )
    wanted = metadata_value.strip().lower()
    matching = {
        str(idx).strip() for idx, val in metadata[metadata_column].items() if str(val).strip().lower() == wanted
    }
    filtered = [entry for entry in ranked if entry[0] in matching]
    logger.info("%d accession(s) remain after %s == %r", len(filtered), metadata_column, metadata_value)
    return filtered


def select_accessions_ranked(
    parsed_containment: Union[str, Path],
    genome_id: Optional[str] = None,
    threshold: float = 0.1,
    metadata_file: Optional[Union[str, Path]] = None,
    metadata_column: Optional[str] = None,
    metadata_value: Optional[str] = None,
    top_n: Optional[int] = None,
    exclude: Optional[Set[str]] = None,
    genome_ids: Optional[List[str]] = None,
    require: str = "any",
) -> List[Tuple[str, str, float]]:
    """Return (accession, column, value) triples meeting the threshold, best first.

    Args:
        parsed_containment: Table from parse_containment (samples x genomes, tab-separated).
        genome_id: Genome column to rank on; defaults to ``max_containment``. Mutually
            exclusive with ``genome_ids``.
        threshold: Minimum containment (inclusive).
        metadata_file: Optional metadata table keyed by Run_ID in its first column.
        metadata_column: Column in the metadata table to filter on.
        metadata_value: Required value for that column (case-insensitive match).
        top_n: Keep only the top N accessions, applied after exclusions and the metadata
            filter so the N returned are all usable.
        exclude: Accessions to drop before ranking and truncation.
        genome_ids: Multiple genome columns to rank on together; mutually exclusive with
            ``genome_id``. With ``require="any"`` accessions rank on the row-wise max over
            these columns; with ``require="all"`` every column must be at or above the
            threshold, ranked on the row-wise min.
        require: ``"any"`` or ``"all"``, how ``genome_ids`` combine (ignored otherwise).

    Raises:
        DataAccessError: If a table is missing.
        ProcessingError: If a column is unknown, ``genome_id``/``genome_ids`` are both
            given, ``require`` is invalid, or the metadata filter is incomplete.
    """
    if genome_id is not None and genome_ids is not None:
        raise ProcessingError("genome_id and genome_ids are mutually exclusive")
    if require not in ("any", "all"):
        raise ProcessingError(f"Unknown require '{require}'. Choose one of: any, all")

    table_path = Path(parsed_containment)
    if not table_path.exists():
        raise DataAccessError(f"Parsed containment table not found: {table_path}")
    if bool(metadata_column) != bool(metadata_value):
        raise ProcessingError("metadata_column and metadata_value must be given together")

    containment = pd.read_csv(table_path, sep="\t", index_col=0)

    if genome_ids is not None:
        _check_columns_exist(containment, genome_ids, table_path.name)
        column, selected = _rank_multi_column(containment, genome_ids, threshold, require)
    else:
        column = genome_id or DEFAULT_COLUMN
        _check_columns_exist(containment, [column], table_path.name)
        selected = _rank_single_column(containment, column, threshold)

    ranked = [(str(acc).strip(), column, float(val)) for acc, val in selected.items()]
    logger.info("%d accession(s) meet %s >= %.3f", len(ranked), column, threshold)

    if exclude:
        ranked = [entry for entry in ranked if entry[0] not in exclude]
        logger.info("%d accession(s) remain after excluding %d accession(s)", len(ranked), len(exclude))

    if metadata_column and metadata_value:
        ranked = _filter_by_metadata(ranked, metadata_file, metadata_column, metadata_value)

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked


def select_accessions(
    parsed_containment: Union[str, Path],
    genome_id: Optional[str] = None,
    threshold: float = 0.1,
    metadata_file: Optional[Union[str, Path]] = None,
    metadata_column: Optional[str] = None,
    metadata_value: Optional[str] = None,
    top_n: Optional[int] = None,
    exclude: Optional[Set[str]] = None,
    genome_ids: Optional[List[str]] = None,
    require: str = "any",
) -> List[str]:
    """Return accessions whose containment meets the threshold, best first.

    A thin wrapper over ``select_accessions_ranked`` that drops the ranking column and
    value. See that function for the full parameter documentation.
    """
    ranked = select_accessions_ranked(
        parsed_containment,
        genome_id=genome_id,
        threshold=threshold,
        metadata_file=metadata_file,
        metadata_column=metadata_column,
        metadata_value=metadata_value,
        top_n=top_n,
        exclude=exclude,
        genome_ids=genome_ids,
        require=require,
    )
    return [accession for accession, _, _ in ranked]
