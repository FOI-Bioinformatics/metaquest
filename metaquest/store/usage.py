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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from metaquest.core.exceptions import DataAccessError
from metaquest.data.registry import Registry, load_registry
from metaquest.store.catalog import Catalog, catalog_write
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


def stale_projects(catalog: Catalog) -> List[Dict[str, Any]]:
    """Projects in the catalogue whose registry no longer keeps them alive.

    A project row is stale when the pipeline run that recorded it will never write to this
    store again: its registry file (the ``registry`` column, written by ``store_init``) has
    been removed, or the registry now found at that path belongs to a different project (its
    ``project.id`` no longer matches this row's ``project_id``, e.g. the project folder was
    reused for a fresh ``store_init``). A registry that exists but fails to parse, or whose
    path cannot even be checked (e.g. a permissions error on a shared multi-user store), is
    treated the same as missing: it cannot vouch for this project either, and the check must
    never crash the caller over one unreadable project.

    Shared by ``store_status`` (which only reports stale projects) and ``store_gc`` (which
    also uses this to decide that a dataset's only usage rows no longer keep it alive).
    Returns each stale row as a dict with ``project_id``, ``name``, ``path`` and ``registry``,
    sorted by ``project_id``.
    """
    rows = catalog.conn.execute("SELECT project_id, name, path, registry FROM projects ORDER BY project_id").fetchall()
    stale: List[Dict[str, Any]] = []
    for row in rows:
        registry_path = row["registry"]
        try:
            if not registry_path or not Path(registry_path).is_file():
                stale.append(dict(row))
                continue
            registry = load_registry(registry_path)
        except (OSError, DataAccessError, ValueError) as e:
            logger.warning("Could not check registry %s while checking staleness: %s", registry_path, e)
            stale.append(dict(row))
            continue
        if (registry.project or {}).get("id") != row["project_id"]:
            stale.append(dict(row))
    return stale


def linked_by(paths: StorePaths, catalog: Catalog, accession: str) -> List[str]:
    """Names of every non-stale project that still symlinks ``accession`` to this store.

    Usage rows are the normal record of "a project uses this dataset", but a project can end
    up linking an accession without one: usage tracking predates that project's ``store_link``
    call, or a hook that would have recorded it failed silently. Before ``store_gc`` removes a
    dataset with no (live) usage rows, it must also check the filesystem directly: for every
    project ``stale_projects`` does not consider gone, does its default ``fastq/<accession>``
    folder (relative to the project's recorded ``path``) exist as a symlink resolving to this
    store's copy of the dataset? A project whose path or link cannot be checked (missing,
    unreadable, broken symlink) is silently treated as not linking it, never as an error.
    Returns the matching project names, sorted, or an empty list when none link it.
    """
    stale_ids = {row["project_id"] for row in stale_projects(catalog)}
    target = (paths.sra / accession).resolve()

    rows = catalog.conn.execute("SELECT project_id, name, path FROM projects ORDER BY project_id").fetchall()
    linked: List[str] = []
    for row in rows:
        if row["project_id"] in stale_ids:
            continue
        project_path = row["path"]
        if not project_path:
            continue
        link_path = Path(project_path) / "fastq" / accession
        try:
            if link_path.is_symlink() and link_path.resolve() == target:
                linked.append(row["name"] or row["project_id"])
        except OSError:
            continue
    return sorted(linked)
