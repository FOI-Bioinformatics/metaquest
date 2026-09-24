"""Select SRA accessions from a parsed containment table, optionally filtered by metadata."""

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError, ProcessingError

logger = logging.getLogger(__name__)

DEFAULT_COLUMN = "max_containment"

# Column names written by metaquest.data.metadata._extract_metadata_fields (the NCBI table).
RUN_SIZE_COLUMN = "Run_Size"
SPOTS_COLUMN = "Run_Total_Spots"
PLATFORM_COLUMN = "Platform"

# Decimal multipliers: SRA reports run sizes in bytes, and 1G here means 10^9 bytes.
_SIZE_MULTIPLIERS = {"": 1, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}
_SIZE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?|\.\d+)\s*([KMGT]?)B?$", re.IGNORECASE)

Entry = Tuple[str, str, float]


def parse_size(value: Union[str, int]) -> int:
    """Parse a byte count such as ``1024``, ``500M``, ``1.5G`` or ``2GB`` into bytes.

    Suffixes K, M, G and T use decimal multipliers (10^3 to 10^12), an optional trailing ``B``
    is accepted and case is ignored.

    Raises:
        ProcessingError: If the value is not a non-negative number with an optional suffix.
    """
    text = str(value).strip()
    match = _SIZE_PATTERN.match(text)
    if not match:
        raise ProcessingError(f"Cannot parse size {value!r}; expected a byte count such as 1024, 500M or 1.5G")
    number = float(match.group(1)) * _SIZE_MULTIPLIERS[match.group(2).upper()]
    if not math.isfinite(number):
        raise ProcessingError(f"Cannot parse size {value!r}; the value is not finite")
    return int(round(number))


@dataclass
class RunFilters:
    """Per-run filters on the metadata table: a size ceiling, spot count bounds and a platform."""

    max_run_size: Optional[int] = None
    min_spots: Optional[int] = None
    max_spots: Optional[int] = None
    platform: Optional[str] = None

    def active(self) -> bool:
        """True when at least one filter is set."""
        return any(v is not None for v in (self.max_run_size, self.min_spots, self.max_spots, self.platform))


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


def _load_metadata(metadata_file: Union[str, Path]) -> pd.DataFrame:
    """Read a metadata table keyed by Run_ID in its first column, all values as strings.

    Run IDs are stripped of surrounding whitespace and only the first row of a repeated Run ID
    is kept, so every lookup returns a single value.
    """
    meta_path = Path(metadata_file)
    if not meta_path.exists():
        raise DataAccessError(f"Metadata table not found: {meta_path}")
    metadata = pd.read_csv(meta_path, sep="\t", index_col=0, dtype=str)
    metadata.index = pd.Index([str(idx).strip() for idx in metadata.index])
    return metadata[~metadata.index.duplicated(keep="first")]


def _filter_by_metadata(
    ranked: List[Entry],
    metadata: pd.DataFrame,
    table_name: str,
    metadata_column: str,
    metadata_value: str,
) -> List[Entry]:
    if metadata_column not in metadata.columns:
        raise ProcessingError(
            f"Column '{metadata_column}' not found in {table_name}. Available columns: "
            f"{', '.join(str(c) for c in metadata.columns)}"
        )
    wanted = metadata_value.strip().lower()
    matching = {idx for idx, val in metadata[metadata_column].items() if str(val).strip().lower() == wanted}
    filtered = [entry for entry in ranked if entry[0] in matching]
    logger.info("%d accession(s) remain after %s == %r", len(filtered), metadata_column, metadata_value)
    return filtered


def _has_column(metadata: pd.DataFrame, column: str, flag: str, table_name: str) -> bool:
    """True when ``column`` exists; otherwise log a warning that ``flag`` is not applied."""
    if column in metadata.columns:
        return True
    logger.warning(
        "%s has no %s column, so %s is not applied (Branchwater-derived tables carry none; "
        "run download_metadata and parse_metadata for the NCBI metadata table)",
        table_name,
        column,
        flag,
    )
    return False


def _apply_run_test(
    ranked: List[Entry],
    values: pd.Series,
    keep: Callable[[Any], bool],
    column: str,
    description: str,
) -> List[Entry]:
    """Keep entries whose value passes ``keep``; a run absent from ``values`` or without a value is dropped.

    An unknown value is dropped rather than kept because the requested bound cannot be checked
    for it (an unknown size may lie far above a ceiling); the number of such runs is logged so
    the loss stays visible.
    """
    kept: List[Entry] = []
    unknown = 0
    for entry in ranked:
        value = values.get(entry[0])
        if value is None or pd.isna(value) or value == "":
            unknown += 1
        elif keep(value):
            kept.append(entry)
    logger.info(
        "%d accession(s) dropped by %s (%d with no %s value)", len(ranked) - len(kept), description, unknown, column
    )
    return kept


def _filter_by_run(ranked: List[Entry], metadata: pd.DataFrame, filters: RunFilters, table_name: str) -> List[Entry]:
    """Apply the run size, spot count and platform filters, in that order."""
    if filters.max_run_size is not None and _has_column(metadata, RUN_SIZE_COLUMN, "--max-run-size", table_name):
        sizes = pd.to_numeric(metadata[RUN_SIZE_COLUMN], errors="coerce")
        ceiling = filters.max_run_size
        ranked = _apply_run_test(
            ranked, sizes, lambda v: v <= ceiling, RUN_SIZE_COLUMN, f"--max-run-size > {ceiling} bytes"
        )
    if filters.min_spots is not None and _has_column(metadata, SPOTS_COLUMN, "--min-spots", table_name):
        spots = pd.to_numeric(metadata[SPOTS_COLUMN], errors="coerce")
        floor = filters.min_spots
        ranked = _apply_run_test(ranked, spots, lambda v: v >= floor, SPOTS_COLUMN, f"--min-spots < {floor}")
    if filters.max_spots is not None and _has_column(metadata, SPOTS_COLUMN, "--max-spots", table_name):
        spots = pd.to_numeric(metadata[SPOTS_COLUMN], errors="coerce")
        cap = filters.max_spots
        ranked = _apply_run_test(ranked, spots, lambda v: v <= cap, SPOTS_COLUMN, f"--max-spots > {cap}")
    if filters.platform is not None and _has_column(metadata, PLATFORM_COLUMN, "--platform", table_name):
        wanted = filters.platform.strip().lower()
        platforms = metadata[PLATFORM_COLUMN].map(lambda v: v.strip().lower() if isinstance(v, str) else v)
        ranked = _apply_run_test(
            ranked, platforms, lambda v: v == wanted, PLATFORM_COLUMN, f"--platform != {filters.platform!r}"
        )
    logger.info("%d accession(s) remain after the run filters", len(ranked))
    return ranked


def _log_selected_volume(ranked: List[Entry], metadata: Optional[pd.DataFrame]) -> None:
    """Log the summed Run_Size of the selection, when the metadata table carries that column."""
    if metadata is None or RUN_SIZE_COLUMN not in metadata.columns:
        return
    sizes = pd.to_numeric(metadata[RUN_SIZE_COLUMN], errors="coerce")
    values = [sizes.get(acc) for acc, _, _ in ranked]
    known = [float(v) for v in values if v is not None and not pd.isna(v)]
    logger.info(
        "Selected volume: %.2f GB across %d run(s) (%d with unknown size)",
        sum(known) / 1e9,
        len(ranked),
        len(ranked) - len(known),
    )


def _validate_arguments(
    genome_id: Optional[str],
    genome_ids: Optional[List[str]],
    require: str,
    top_n: Optional[int],
    run_filters: RunFilters,
) -> None:
    if genome_id is not None and genome_ids is not None:
        raise ProcessingError("genome_id and genome_ids are mutually exclusive")
    if require not in ("any", "all"):
        raise ProcessingError(f"Unknown require '{require}'. Choose one of: any, all")
    if top_n is not None and top_n < 1:
        raise ProcessingError("--top-n must be a positive integer")
    min_spots, max_spots = run_filters.min_spots, run_filters.max_spots
    if min_spots is not None and max_spots is not None and min_spots > max_spots:
        raise ProcessingError(f"min_spots ({min_spots}) is greater than max_spots ({max_spots})")


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
    run_filters: Optional[RunFilters] = None,
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
        run_filters: Optional run size, spot count and platform filters, read from the
            ``Run_Size``, ``Run_Total_Spots`` and ``Platform`` columns of ``metadata_file``.
            They apply after the metadata equality filter and before ``top_n``. A run absent
            from the table or without a value is dropped by an active filter on that column;
            a column missing from the table logs a warning and that filter is skipped.

    Raises:
        DataAccessError: If a table is missing.
        ProcessingError: If a column is unknown, ``genome_id``/``genome_ids`` are both
            given, ``require`` is invalid, the metadata filter is incomplete, run filters are
            given without a metadata file, or ``min_spots`` exceeds ``max_spots``.
    """
    run_filters = run_filters or RunFilters()
    _validate_arguments(genome_id, genome_ids, require, top_n, run_filters)

    table_path = Path(parsed_containment)
    if not table_path.exists():
        raise DataAccessError(f"Parsed containment table not found: {table_path}")
    if bool(metadata_column) != bool(metadata_value):
        raise ProcessingError("metadata_column and metadata_value must be given together")
    needs_metadata = bool(metadata_column) or run_filters.active()
    if needs_metadata and metadata_file is None:
        raise ProcessingError("metadata_file is required when filtering on metadata, run size, spot count or platform")

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

    metadata: Optional[pd.DataFrame] = None
    if needs_metadata and metadata_file is not None:
        metadata = _load_metadata(metadata_file)
        table_name = Path(metadata_file).name
        if metadata_column and metadata_value:
            ranked = _filter_by_metadata(ranked, metadata, table_name, metadata_column, metadata_value)
        if run_filters.active():
            ranked = _filter_by_run(ranked, metadata, run_filters, table_name)

    if top_n is not None:
        ranked = ranked[:top_n]

    _log_selected_volume(ranked, metadata)
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
    run_filters: Optional[RunFilters] = None,
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
        run_filters=run_filters,
    )
    return [accession for accession, _, _ in ranked]
