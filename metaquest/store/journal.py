"""Append-only journal of the store's project and usage records.

The SQLite catalogue is the working copy of "which project used which dataset", but it is the
only copy: sidecars know nothing about projects. ``store_reindex`` rebuilds the catalogue from
sidecars and would otherwise come back with no projects and no usage, after which ``store_gc``
sees every dataset as unused. Every ``upsert_project`` and ``record_usage`` therefore also
appends one JSON line here, and ``replay`` feeds those lines back into a rebuilt catalogue.
Appends happen inside ``catalog_write``, which holds the store-wide write lock.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from metaquest.store.layout import StorePaths

logger = logging.getLogger(__name__)

PROJECTS_FILE = "projects.jsonl"
USAGE_FILE = "usage.jsonl"


def _append(paths: StorePaths, name: str, record: Dict[str, Any]) -> None:
    paths.journal.mkdir(parents=True, exist_ok=True)
    record = dict(record, at=datetime.now(timezone.utc).isoformat())
    with open(paths.journal / name, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_project(paths: StorePaths, project_id: str, name: str, path: str, registry: str) -> None:
    """Append one project record (mirroring ``Catalog.upsert_project``'s arguments)."""
    _append(paths, PROJECTS_FILE, {"project_id": project_id, "name": name, "path": path, "registry": registry})


def append_usage(paths: StorePaths, accession: str, project_id: str, genome_id: str, stage: str, detail: str) -> None:
    """Append one usage record (mirroring ``Catalog.record_usage``'s arguments)."""
    _append(
        paths,
        USAGE_FILE,
        {"accession": accession, "project_id": project_id, "genome_id": genome_id, "stage": stage, "detail": detail},
    )


def _lines(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable journal line %s:%d", path, number)


def replay(paths: StorePaths, catalog: Any) -> Tuple[int, int]:
    """Feed every journaled project and usage record into ``catalog``; returns the two counts.

    The catalogue calls back into ``append_*`` when it writes, so replay runs with journaling
    suspended (``catalog.journal_enabled = False``) to avoid duplicating the file.
    """
    projects = 0
    usage = 0
    previous = getattr(catalog, "journal_enabled", True)
    catalog.journal_enabled = False
    try:
        for record in _lines(paths.journal / PROJECTS_FILE):
            catalog.upsert_project(
                record["project_id"], record.get("name", ""), record.get("path", ""), record.get("registry", "")
            )
            projects += 1
        for record in _lines(paths.journal / USAGE_FILE):
            catalog.record_usage(
                record["accession"],
                record["project_id"],
                record.get("genome_id", ""),
                record.get("stage", ""),
                record.get("detail", ""),
            )
            usage += 1
    finally:
        catalog.journal_enabled = previous
    return projects, usage
