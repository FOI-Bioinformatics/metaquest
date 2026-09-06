"""
Safe usage recording into the shared store's catalogue.

Every pipeline stage that touches a dataset (a fresh download, a link from the store, an
analysis, a targeted extraction, an assembly) can record that fact in the catalogue's
``usage`` table, so the catalogue answers "which projects used this accession, for which
genome, at which stage" across every project sharing the store. That history is a
convenience, never a dependency: a project must keep working exactly as before when there
is no store, when the project has never been bound to one (no ``store_init``), or when the
catalogue write itself fails (a lock timeout, a corrupt database, a disk error). The two
functions here are the only way any CLI command touches the catalogue for this purpose, and
neither ever raises.
"""

import logging
from typing import Iterable, Optional, Tuple

from metaquest.data.registry import Registry
from metaquest.store.catalog import catalog_write
from metaquest.store.layout import StorePaths

logger = logging.getLogger(__name__)


def _upsert_project_row(catalog, registry: Registry, project_id: str) -> None:
    project = registry.project or {}
    catalog.upsert_project(
        project_id,
        project.get("name", ""),
        project.get("path", ""),
        str(registry.path),
    )


def record_usage_safe(
    paths: Optional[StorePaths],
    registry: Registry,
    accession: str,
    genome_id: str,
    stage: str,
    detail: str = "",
) -> bool:
    """Record one dataset's use in the store catalogue; never raises.

    Returns False, doing nothing else, when there is no store (``paths`` is None) or this
    project has never been bound to one (``registry.project`` carries no ``id``, i.e.
    ``store_init`` was never run for it). Otherwise upserts the project's row (so a project
    that has moved, or one seen for the first time, is always current) and records the
    usage row, both inside one ``catalog_write`` session. Every exception the write can
    raise (``DataAccessError`` from a corrupt or locked catalogue, ``OSError`` from the
    filesystem, or anything else) is caught, logged at warning naming the accession and
    stage, and turned into False: the pipeline stage that triggered this call must never
    fail because the catalogue could not be written.
    """
    project = registry.project or {}
    project_id = project.get("id")
    if paths is None or not project_id:
        return False

    try:
        with catalog_write(paths) as catalog:
            _upsert_project_row(catalog, registry, project_id)
            catalog.record_usage(accession, project_id, genome_id, stage, detail)
        return True
    except Exception as e:
        logger.warning("Could not record usage for %s (stage=%s): %s", accession, stage, e)
        return False


def record_usage_many(
    paths: Optional[StorePaths],
    registry: Registry,
    rows: Iterable[Tuple[str, str, str, str]],
) -> bool:
    """Like ``record_usage_safe``, for several ``(accession, genome_id, stage, detail)`` rows.

    All rows are written inside one ``catalog_write`` session, so a batch (e.g. every
    already-downloaded accession a download run found linked from the store) takes one
    lock instead of one per accession. Same safety as ``record_usage_safe``: no store or no
    bound project returns False without writing anything; any exception during the write
    is caught, logged at warning, and turned into False.
    """
    project = registry.project or {}
    project_id = project.get("id")
    if paths is None or not project_id:
        return False

    rows = list(rows)
    if not rows:
        return True

    try:
        with catalog_write(paths) as catalog:
            _upsert_project_row(catalog, registry, project_id)
            for accession, genome_id, stage, detail in rows:
                catalog.record_usage(accession, project_id, genome_id, stage, detail)
        return True
    except Exception as e:
        logger.warning("Could not record usage batch (%d row(s)): %s", len(rows), e)
        return False
