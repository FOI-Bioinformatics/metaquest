"""Resolution of default input files so pipeline steps chain without extra flags."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)

# Order matters: the NCBI table (parse_metadata) is richer than the Branchwater one.
METADATA_TABLE_CANDIDATES: List[Path] = [
    Path("metadata_table.txt"),
    Path("metadata/metadata_table.txt"),
    Path("metadata/branchwater_metadata.txt"),
]


def resolve_metadata_table(explicit: Optional[str]) -> Path:
    """Return the metadata table to use.

    An explicit path must exist. Otherwise the first existing candidate in
    ``METADATA_TABLE_CANDIDATES`` is used, so a project that only ran
    ``extract_branchwater_metadata`` still works with the defaults.

    Raises:
        DataAccessError: If nothing usable exists.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise DataAccessError(f"Metadata table not found: {explicit}")
        return path

    for candidate in METADATA_TABLE_CANDIDATES:
        if candidate.exists():
            logger.info("Using metadata table %s", candidate)
            return candidate

    tried = ", ".join(str(c) for c in METADATA_TABLE_CANDIDATES)
    raise DataAccessError(
        f"No metadata table found (tried {tried}). Run parse_metadata or extract_branchwater_metadata, "
        "or pass --metadata-file."
    )


def read_matrix(path: Union[str, Path]) -> pd.DataFrame:
    """Read a samples x features table as numeric data.

    Tab-separated ``.txt``/``.tsv`` files (as written by parse_containment) and
    comma-separated files are both accepted. Non-numeric columns such as
    ``max_containment_annotation`` are dropped with a log line.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    numeric = df.select_dtypes(include="number")
    dropped = [str(c) for c in df.columns if c not in numeric.columns]
    if dropped:
        logger.info("Ignoring non-numeric column(s) in %s: %s", path.name, ", ".join(dropped))
    return numeric
