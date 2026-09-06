"""
SQLite catalogue over the shared data store: ``<root>/catalog.sqlite``.

The catalogue is a rebuildable index over every sidecar in the store, plus two
cross-project tables (``projects`` and ``usage``) that record which project used
which dataset, for which target genome, at which pipeline stage. It answers
questions no single sidecar or project registry can: which projects used one
accession, which downloaded datasets no project references any more, how many
bytes are held per target genome, and which datasets were analysed for one
genome across every project. Because it is rebuildable from the sidecars
(``reindex``), losing or corrupting ``catalog.sqlite`` never loses data; it
only loses the cross-project ``usage`` history, which lives only here.

Writes are serialised across processes with the same lock protocol the
per-project registry uses (``metaquest.data.registry._acquire_lock``), on a
``catalog.sqlite.lock`` file next to the database, so two projects on one
machine or a shared network volume never interleave writes. Reads may open the
database directly without taking the lock.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from metaquest.data.registry import _acquire_lock
from metaquest.store.layout import StorePaths
from metaquest.store.sidecar import Sidecar

logger = logging.getLogger(__name__)

# Bump when the catalogue's schema changes shape.
SCHEMA_VERSION = 1

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS store_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets (
        accession TEXT PRIMARY KEY,
        state TEXT,
        layout TEXT,
        compression TEXT,
        reads_per_mate INTEGER,
        bases_total INTEGER,
        bytes_total INTEGER,
        ncbi_spots INTEGER,
        ncbi_bases INTEGER,
        completeness TEXT,
        downloaded TEXT,
        tool_version TEXT,
        updated TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS files (
        accession TEXT NOT NULL REFERENCES datasets(accession) ON DELETE CASCADE,
        name TEXT NOT NULL,
        bytes INTEGER,
        md5 TEXT,
        reads INTEGER,
        PRIMARY KEY (accession, name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        name TEXT,
        path TEXT,
        registry TEXT,
        created TEXT,
        last_seen TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS usage (
        accession TEXT NOT NULL REFERENCES datasets(accession) ON DELETE CASCADE,
        project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        genome_id TEXT NOT NULL DEFAULT '',
        stage TEXT NOT NULL,
        first_used TEXT,
        last_used TEXT,
        detail TEXT,
        PRIMARY KEY (accession, project_id, genome_id, stage)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_project ON usage(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_usage_genome ON usage(genome_id)",
    """
    CREATE VIEW IF NOT EXISTS unused_datasets AS
    SELECT d.accession AS accession
    FROM datasets d
    LEFT JOIN usage u ON u.accession = d.accession
    WHERE u.accession IS NULL
    """,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    """One open connection to ``catalog.sqlite``.

    Use as a context manager. Opening never migrates the schema and never takes
    the write lock; both are the caller's job (``catalog_write`` does both for a
    write session). A plain ``with Catalog(paths) as catalog:`` is for reads
    against a catalogue that already exists.
    """

    def __init__(self, paths: StorePaths):
        self.paths = paths
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        """The open connection; raises if used outside a ``with Catalog(...) as catalog:`` block."""
        if self._conn is None:
            raise RuntimeError("Catalog is not open; use it as a context manager")
        return self._conn

    def __enter__(self) -> "Catalog":
        self.paths.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.paths.catalog))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.execute("PRAGMA journal_mode = WAL")
        mode = cursor.fetchone()[0]
        if str(mode).lower() != "wal":
            logger.info(
                "Catalog at %s did not enable WAL journalling (got %r); continuing with the "
                "default journal mode. This is expected on some network filesystems.",
                self.paths.catalog,
                mode,
            )
        self._conn = conn
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------ schema

    def migrate(self) -> None:
        """Create the catalogue schema if it does not already exist. Safe to call repeatedly."""
        for statement in _SCHEMA_STATEMENTS:
            self.conn.execute(statement)
        self.conn.execute(
            "INSERT OR IGNORE INTO store_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    # ---------------------------------------------------------------- datasets

    def upsert_dataset(self, sidecar: Sidecar) -> None:
        """Insert or update the ``datasets``/``files`` rows for one sidecar.

        ``bytes_total`` is the sum of the sidecar's file sizes; ``completeness``
        stores the verdict string only (the method and ratio are not indexed).
        The file rows for this accession are replaced wholesale, since a sidecar
        always lists every file currently on disk for the accession.
        """
        files = sidecar.files or []
        bytes_total = sum(int(f.get("bytes") or 0) for f in files)
        now = _now()

        self.conn.execute(
            """
            INSERT INTO datasets (
                accession, state, layout, compression, reads_per_mate, bases_total,
                bytes_total, ncbi_spots, ncbi_bases, completeness, downloaded,
                tool_version, updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(accession) DO UPDATE SET
                state=excluded.state,
                layout=excluded.layout,
                compression=excluded.compression,
                reads_per_mate=excluded.reads_per_mate,
                bases_total=excluded.bases_total,
                bytes_total=excluded.bytes_total,
                ncbi_spots=excluded.ncbi_spots,
                ncbi_bases=excluded.ncbi_bases,
                completeness=excluded.completeness,
                downloaded=excluded.downloaded,
                tool_version=excluded.tool_version,
                updated=excluded.updated
            """,
            (
                sidecar.accession,
                sidecar.state,
                sidecar.layout,
                sidecar.compression,
                sidecar.reads_per_mate,
                sidecar.bases_total,
                bytes_total,
                sidecar.ncbi.get("spots"),
                sidecar.ncbi.get("bases"),
                sidecar.completeness.get("verdict"),
                sidecar.downloaded,
                sidecar.tool_version,
                now,
            ),
        )
        self.conn.execute("DELETE FROM files WHERE accession = ?", (sidecar.accession,))
        self.conn.executemany(
            "INSERT INTO files (accession, name, bytes, md5, reads) VALUES (?, ?, ?, ?, ?)",
            [(sidecar.accession, f.get("name"), f.get("bytes"), f.get("md5"), f.get("reads")) for f in files],
        )

    def get_dataset(self, accession: str) -> Optional[Dict[str, Any]]:
        """The ``datasets`` row for ``accession`` as a plain dict (with its ``files``), or None."""
        row = self.conn.execute("SELECT * FROM datasets WHERE accession = ?", (accession,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        file_rows = self.conn.execute(
            "SELECT name, bytes, md5, reads FROM files WHERE accession = ? ORDER BY name",
            (accession,),
        ).fetchall()
        result["files"] = [dict(f) for f in file_rows]
        return result

    # ---------------------------------------------------------------- projects

    def upsert_project(self, project_id: str, name: str, path: str, registry: str) -> None:
        """Insert or update one project's row, keeping its original ``created`` timestamp."""
        now = _now()
        self.conn.execute(
            """
            INSERT INTO projects (project_id, name, path, registry, created, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                name=excluded.name,
                path=excluded.path,
                registry=excluded.registry,
                last_seen=excluded.last_seen
            """,
            (project_id, name, path, registry, now, now),
        )

    # ------------------------------------------------------------------- usage

    def record_usage(self, accession: str, project_id: str, genome_id: str, stage: str, detail: str = "") -> None:
        """Record that ``project_id`` used ``accession`` (for ``genome_id``, at ``stage``).

        ``first_used`` is set once and kept on every later call for the same
        (accession, project_id, genome_id, stage) key; ``last_used`` and ``detail``
        are refreshed each time. If ``accession`` is not yet in ``datasets`` (usage
        can be recorded before a dataset is catalogued, e.g. a link recorded ahead
        of the next reindex), a minimal placeholder row with ``state="unknown"`` is
        inserted first so the foreign key from ``usage`` to ``datasets`` is
        satisfied; a later ``upsert_dataset`` or ``reindex`` fills it in properly.
        """
        now = _now()
        genome_id = genome_id or ""

        self.conn.execute(
            "INSERT OR IGNORE INTO datasets (accession, state, updated) VALUES (?, ?, ?)",
            (accession, "unknown", now),
        )
        self.conn.execute(
            """
            INSERT INTO usage (accession, project_id, genome_id, stage, first_used, last_used, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(accession, project_id, genome_id, stage) DO UPDATE SET
                last_used=excluded.last_used,
                detail=excluded.detail
            """,
            (accession, project_id, genome_id, stage, now, now, detail),
        )

    # ----------------------------------------------------------------- queries

    def projects_for(self, accession: str) -> List[Dict[str, Any]]:
        """Every project that has recorded usage of ``accession``, sorted by project id."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT p.*
            FROM projects p
            JOIN usage u ON u.project_id = p.project_id
            WHERE u.accession = ?
            ORDER BY p.project_id
            """,
            (accession,),
        ).fetchall()
        return [dict(row) for row in rows]

    def datasets_for_project(self, project_id: str) -> List[Dict[str, Any]]:
        """Every dataset ``project_id`` has recorded usage of, sorted by accession."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT d.*
            FROM datasets d
            JOIN usage u ON u.accession = d.accession
            WHERE u.project_id = ?
            ORDER BY d.accession
            """,
            (project_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def unused(self) -> List[str]:
        """Accessions in ``datasets`` with no usage row at all, sorted."""
        rows = self.conn.execute("SELECT accession FROM unused_datasets ORDER BY accession").fetchall()
        return [row["accession"] for row in rows]

    def bytes_by_genome(self) -> Dict[str, int]:
        """Total ``bytes_total`` per genome, summed over distinct (accession, genome_id) usage pairs.

        A dataset used by several projects for the same genome is counted once,
        since it occupies the bytes only once on disk.
        """
        rows = self.conn.execute("""
            SELECT pair.genome_id AS genome_id, SUM(d.bytes_total) AS total
            FROM (SELECT DISTINCT genome_id, accession FROM usage) pair
            JOIN datasets d ON d.accession = pair.accession
            GROUP BY pair.genome_id
            ORDER BY pair.genome_id
            """).fetchall()
        return {row["genome_id"]: (row["total"] or 0) for row in rows}

    def datasets_for_genome(self, genome_id: str) -> List[Tuple[str, str]]:
        """Distinct (accession, project_id) pairs with recorded usage for ``genome_id``."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT accession, project_id
            FROM usage
            WHERE genome_id = ?
            ORDER BY accession, project_id
            """,
            (genome_id,),
        ).fetchall()
        return [(row["accession"], row["project_id"]) for row in rows]

    # ----------------------------------------------------------------- rebuild

    def reindex(self, sidecars: Iterable[Sidecar]) -> int:
        """Rebuild ``datasets``/``files`` from ``sidecars``; keep ``projects`` and ``usage``.

        Every accession not present in ``sidecars`` is removed from ``datasets``;
        the foreign key cascade then removes its ``files`` and ``usage`` rows too.
        Every accession that is present is upserted, so its ``usage`` history
        (referencing the same, still-present, accession) is left untouched.
        Returns how many sidecars were indexed.
        """
        sidecar_list = list(sidecars)
        keep = {sidecar.accession for sidecar in sidecar_list}

        existing = {row["accession"] for row in self.conn.execute("SELECT accession FROM datasets").fetchall()}
        for accession in existing - keep:
            self.conn.execute("DELETE FROM datasets WHERE accession = ?", (accession,))

        for sidecar in sidecar_list:
            self.upsert_dataset(sidecar)

        return len(sidecar_list)


@contextmanager
def catalog_write(paths: StorePaths) -> Iterator[Catalog]:
    """Open the catalogue for a write session, serialised across processes.

    Acquires ``paths.catalog_lock`` (blocking, with the same wait/stale-lock
    protocol as the per-project registry's ``_acquire_lock``), opens the
    catalogue, migrates its schema, yields it for the caller to write through,
    commits on a clean exit, and always releases the lock. If the block raises,
    the connection is closed without committing (uncommitted changes are
    discarded) and the lock is still released.
    """
    paths.root.mkdir(parents=True, exist_ok=True)
    _acquire_lock(paths.catalog_lock)
    try:
        with Catalog(paths) as catalog:
            catalog.migrate()
            yield catalog
            catalog.conn.commit()
    finally:
        paths.catalog_lock.unlink(missing_ok=True)
