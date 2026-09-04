"""Select SRA accessions from a parsed containment table, optionally filtered by metadata."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError, ProcessingError

logger = logging.getLogger(__name__)

DEFAULT_COLUMN = "max_containment"


def select_accessions(
    parsed_containment: Union[str, Path],
    genome_id: Optional[str] = None,
    threshold: float = 0.1,
    metadata_file: Optional[Union[str, Path]] = None,
    metadata_column: Optional[str] = None,
    metadata_value: Optional[str] = None,
) -> List[str]:
    """Return accessions whose containment meets the threshold, best first.

    Args:
        parsed_containment: Table from parse_containment (samples x genomes, tab-separated).
        genome_id: Genome column to rank on; defaults to ``max_containment``.
        threshold: Minimum containment (inclusive).
        metadata_file: Optional metadata table keyed by Run_ID in its first column.
        metadata_column: Column in the metadata table to filter on.
        metadata_value: Required value for that column (case-insensitive match).

    Raises:
        DataAccessError: If a table is missing.
        ProcessingError: If a column is unknown or the metadata filter is incomplete.
    """
    table_path = Path(parsed_containment)
    if not table_path.exists():
        raise DataAccessError(f"Parsed containment table not found: {table_path}")
    if bool(metadata_column) != bool(metadata_value):
        raise ProcessingError("metadata_column and metadata_value must be given together")

    containment = pd.read_csv(table_path, sep="\t", index_col=0)
    column = genome_id or DEFAULT_COLUMN
    if column not in containment.columns:
        raise ProcessingError(
            f"Column '{column}' not found in {table_path.name}. Available columns: "
            f"{', '.join(str(c) for c in containment.columns)}"
        )

    values = pd.to_numeric(containment[column], errors="coerce").fillna(0.0)
    selected = values[values >= threshold].sort_values(ascending=False)
    accessions = [str(acc).strip() for acc in selected.index]
    logger.info("%d accession(s) meet %s >= %.3f", len(accessions), column, threshold)

    if metadata_column and metadata_value:
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
        accessions = [acc for acc in accessions if acc in matching]
        logger.info("%d accession(s) remain after %s == %r", len(accessions), metadata_column, metadata_value)

    return accessions
