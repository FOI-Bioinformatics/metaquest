"""
`store_gc`: report, or with `--yes` remove, datasets nothing references any more.
"""

import argparse
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metaquest.cli.base import BaseCommand, emit_error_json
from metaquest.cli.commands.store._shared import _no_store_hint, _stale_project_row
from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry_blocks as rb
from metaquest.data.file_io import is_hidden_name
from metaquest.data.registry import load_registry
from metaquest.data.sra import is_transient_folder
from metaquest.store.catalog import REBUILT_WITHOUT_PROJECTS, Catalog, catalog_write
from metaquest.store.layout import StorePaths, sra_dir, store_paths
from metaquest.store.locks import lock_holder, lock_is_held
from metaquest.store.resolve import resolve_store_root
from metaquest.store.usage import linked_by, stale_projects

logger = logging.getLogger(__name__)


def _remove_path(path: Path) -> None:
    """Remove a file or directory tree at ``path``, logging (never raising) on failure."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError as e:
        logger.warning("Could not remove %s: %s", path, e)


def _path_bytes(path: Path) -> int:
    """Total bytes held by ``path``: its own size for a file, or the recursive sum for a directory."""
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    if path.is_dir():
        for sub in path.rglob("*"):
            if sub.is_file():
                try:
                    total += sub.stat().st_size
                except OSError:
                    continue
    return total


class StoreGcCommand(BaseCommand):
    """Command to report, and optionally remove, unused datasets and leftover temp files.

    A dataset is a removal candidate when it has no usage rows at all and no live project
    symlinks it. Three things keep a dataset out of the candidate list and into the report's
    ``still_linked``, ``in_use`` and ``kept_stale`` sections: a live project's symlink, a held
    accession lock (another run is downloading or adopting it right now), and usage rows that
    belong only to projects ``stale_projects`` (see ``metaquest.store.usage``) cannot see from
    this machine, which on a shared store is every project on another workstation. That last
    case needs ``--include-stale`` before it is removed. Placeholder rows (``state="unknown"``,
    a usage row recorded ahead of its dataset) stand for no files and are never candidates.

    Leftover temp artifacts (``<store>/tmp/*_temp`` from an interrupted download,
    ``<store>/tmp/*_adopt`` from an interrupted adopt, ``<store>/tmp/*_old`` from a publish,
    the ``.sra-cache`` archive cache under ``tmp`` or ``sra``) are reported and removed
    independently of the dataset check, minus anything whose accession lock is held. Nothing
    is removed unless ``--yes`` is given; the default is a dry-run report only.
    """

    @property
    def name(self) -> str:
        """Return the command name."""
        return "store_gc"

    @property
    def help(self) -> str:
        """Return the command's help text."""
        return (
            "Report, and with --yes remove, unused datasets and leftover temp files from the store "
            "(datasets a project links, or another run is working on, are always kept)"
        )

    @property
    def group(self) -> str:
        """Return the pipeline-step group this command is listed under."""
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add the command's arguments."""
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        parser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=False,
            help="Report candidates without removing anything (this is the default when --yes is not given)",
        )
        parser.add_argument(
            "--yes",
            action="store_true",
            default=False,
            help="Remove the reported candidates (cannot be combined with --dry-run)",
        )
        parser.add_argument(
            "--older-than",
            dest="older_than",
            type=int,
            default=None,
            help="Only consider datasets whose sidecar was downloaded at least this many days ago",
        )
        parser.add_argument(
            "--keep-partial", action="store_true", help="Never remove a dataset whose state is 'partial'"
        )
        parser.add_argument(
            "--include-stale",
            dest="include_stale",
            action="store_true",
            default=False,
            help=(
                "Also remove datasets whose only users are projects that look stale from this "
                "machine; every project on another workstation of a shared store looks stale here"
            ),
        )
        parser.add_argument(
            "--accept-rebuilt",
            dest="accept_rebuilt",
            action="store_true",
            default=False,
            help=(
                "Confirm that every project using this store has run store_init or store_link since "
                "store_reindex rebuilt the catalogue without any project records; clears that "
                "catalogue flag so store_gc can run"
            ),
        )
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON")

    def _refuse_before_candidates(
        self, catalog: Catalog, rebuilt: Optional[str], accept_rebuilt: bool
    ) -> Optional[str]:
        """The refusal message (after logging it) when gc must not look for candidates at all,
        or None to proceed: the catalogue carries the flag ``store_reindex`` sets when it
        restored no project and the user has not passed ``--accept-rebuilt``, or it records no
        project while it holds datasets (then ``--accept-rebuilt`` is refused too, since no
        project has registered again yet). Reads only; the flag is cleared by the caller once
        the report has been built. The caller also prints the returned message as JSON when
        ``--json`` is given, since a refusal must be visible to a script parsing stdout, not
        only to the log."""
        if rebuilt is not None and not accept_rebuilt:
            message = (
                f"store_reindex rebuilt the catalogue on {rebuilt} without any project records, so the "
                "datasets of every project that has not registered again since would look unused. Run "
                "store_init (or store_link) from every project that uses this store, then run store_gc "
                "--accept-rebuilt."
            )
            self.logger.error(message)
            return message
        project_count = catalog.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        if project_count == 0 and any(True for _ in catalog.conn.execute("SELECT 1 FROM datasets LIMIT 1")):
            if rebuilt is not None:
                message = (
                    "--accept-rebuilt refused: the catalogue still records no project at all. Run store_init "
                    "(or store_link) from every project that uses this store first."
                )
            else:
                message = (
                    "The catalogue records no project at all, so nothing can be told apart from unused data. "
                    "Run store_reindex (which replays the journal) or store_init from each project first."
                )
            self.logger.error(message)
            return message
        return None

    # ------------------------------------------------------------- candidates

    @staticmethod
    def _downloaded_before_cutoff(downloaded: Optional[str], older_than_days: Optional[int]) -> bool:
        """True when ``downloaded`` (a sidecar's ISO timestamp) is at least ``older_than_days``
        old; True unconditionally when ``older_than_days`` is None (no filter requested). A
        dataset with no recorded ``downloaded`` date, or an unparsable one, is treated as not
        old enough to remove under a threshold, since its age cannot be confirmed."""
        if older_than_days is None:
            return True
        if not downloaded:
            return False
        try:
            when = datetime.fromisoformat(downloaded)
        except ValueError:
            return False
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - when).total_seconds() / 86400.0
        return age_days >= older_than_days

    @staticmethod
    def _usage_by_accession(catalog: Catalog) -> Dict[str, set]:
        """The set of project ids that has recorded usage of each accession."""
        usage_by_accession: Dict[str, set] = {}
        for row in catalog.conn.execute("SELECT DISTINCT accession, project_id FROM usage").fetchall():
            usage_by_accession.setdefault(row["accession"], set()).add(row["project_id"])
        return usage_by_accession

    @classmethod
    def _classify_dataset(
        cls,
        row: Any,
        paths: StorePaths,
        catalog: Catalog,
        project_ids: set,
        stale: Dict[str, str],
        include_stale: bool,
    ) -> Tuple[str, str]:
        """Why this dataset is, or is not, a removal candidate: ``(bucket, reason)``.

        The buckets are ``candidate``, ``in_use`` (another run holds the accession's lock right
        now), ``linked`` (a live project still symlinks it despite carrying no usage row),
        ``stale`` (its only users look stale from this machine, kept unless ``--include-stale``)
        and ``keep`` (a live project uses it).
        """
        accession = row["accession"]
        if lock_is_held(paths, accession):
            return "in_use", f"in use: {lock_holder(paths, accession)}"
        if project_ids and not project_ids <= set(stale):
            return "keep", "in use by a live project"

        linked_names = linked_by(paths, catalog, accession)
        if linked_names:
            return "linked", "still linked by " + ", ".join(linked_names)

        if not project_ids:
            return "candidate", "unused"
        names = sorted(stale.get(pid, pid) for pid in project_ids)
        reason = "stale projects: " + ", ".join(names)
        return ("candidate" if include_stale else "stale"), reason

    @classmethod
    def _dataset_candidates(
        cls,
        catalog: Catalog,
        paths: StorePaths,
        stale: List[Dict[str, Any]],
        older_than_days: Optional[int],
        keep_partial: bool,
        include_stale: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Sort every catalogued dataset into removal candidates and the reasons for keeping.

        Returns a dict with ``candidates``, ``still_linked``, ``in_use`` and ``kept_stale``.
        Placeholder rows (``state = "unknown"``, inserted so a usage row can reference an
        accession the catalogue has not indexed yet) stand for no files at all and are never
        offered for removal.
        """
        stale_names = {row["project_id"]: (row.get("name") or row["project_id"]) for row in stale}

        rows = catalog.conn.execute(
            "SELECT accession, state, COALESCE(bytes_total, 0) AS bytes, downloaded FROM datasets "
            "WHERE state IS NOT 'unknown' ORDER BY accession"
        ).fetchall()
        usage_by_accession = cls._usage_by_accession(catalog)

        buckets: Dict[str, List[Dict[str, Any]]] = {
            "candidates": [],
            "still_linked": [],
            "in_use": [],
            "kept_stale": [],
        }
        bucket_names = {"candidate": "candidates", "linked": "still_linked", "in_use": "in_use", "stale": "kept_stale"}
        for row in rows:
            if keep_partial and row["state"] == "partial":
                continue
            if not cls._downloaded_before_cutoff(row["downloaded"], older_than_days):
                continue
            project_ids = usage_by_accession.get(row["accession"], set())
            bucket, reason = cls._classify_dataset(row, paths, catalog, project_ids, stale_names, include_stale)
            if bucket == "keep":
                continue
            buckets[bucket_names[bucket]].append(
                {"accession": row["accession"], "bytes": row["bytes"], "reason": reason}
            )
        return buckets

    @staticmethod
    def _accession_of_leftover(name: str) -> str:
        """The accession a leftover folder belongs to, e.g. ``SRR1`` for ``SRR1_temp`` or ``SRR1_fqtmp``."""
        for suffix in ("_temp", "_fqtmp", "_adopt", "_old"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @classmethod
    def _leftover_candidates(cls, paths: StorePaths) -> List[Dict[str, Any]]:
        """Leftover build folders and cached archives, minus anything a live run is using.

        A ``<ACC>_temp`` build folder, an ``<ACC>_fqtmp`` fasterq-dump scratch folder or a cached
        ``.sra`` archive whose accession lock is held is a download in progress, not a leftover:
        removing it would pull the files out from under a running fasterq-dump.
        """
        candidates: List[Dict[str, Any]] = []
        tmp = paths.tmp
        if tmp.is_dir():
            for entry in sorted(tmp.iterdir()):
                if entry.name == ".sra-cache" and entry.is_dir():
                    for sub in sorted(entry.iterdir()):
                        if lock_is_held(paths, cls._accession_of_leftover(sub.name)):
                            continue
                        candidates.append({"path": sub, "bytes": _path_bytes(sub), "reason": "leftover"})
                    continue
                if is_hidden_name(entry.name):
                    continue
                if not entry.is_dir():
                    continue
                if not (is_transient_folder(entry.name) or entry.name.endswith(("_adopt", "_old"))):
                    continue
                if lock_is_held(paths, cls._accession_of_leftover(entry.name)):
                    continue
                candidates.append({"path": entry, "bytes": _path_bytes(entry), "reason": "leftover"})
        sra_cache = paths.sra / ".sra-cache"
        if sra_cache.exists():
            candidates.append({"path": sra_cache, "bytes": _path_bytes(sra_cache), "reason": "leftover"})
        return candidates

    # ----------------------------------------------------------------- print

    def _print_report(self, report: Dict[str, Any], performed: bool) -> None:
        verb = "Removed" if performed else "Would remove"
        self.emit(
            f"{verb} {len(report['datasets'])} dataset(s), {len(report['leftovers'])} leftover(s), "
            f"{report['total_bytes']} bytes total"
        )
        for entry in report["datasets"]:
            self.emit(f"  dataset   {entry['accession']:<15s} {entry['bytes']:>12} bytes  {entry['reason']}")
        for entry in report["leftovers"]:
            self.emit(f"  leftover  {entry['path']:<40s} {entry['bytes']:>12} bytes  {entry['reason']}")
        for key in ("still_linked", "in_use", "kept_stale"):
            for entry in report.get(key, []):
                self.emit(f"  kept      {entry['accession']:<15s} {entry['bytes']:>12} bytes  {entry['reason']}")
        if report["stale_projects"]:
            self.emit("Stale projects:")
            for entry in report["stale_projects"]:
                self.emit(f"  {entry['name']} on {entry['hostname']} ({entry['reason']}: {entry['registry']})")

    # --------------------------------------------------------------- execute

    def execute(self, args: argparse.Namespace) -> int:
        """Run the command; return the exit code."""
        if args.dry_run and args.yes:
            self.logger.error("--dry-run and --yes cannot be combined")
            return 1

        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, rb.store_block(registry).root)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint(getattr(args, "json", False))
            return 1

        paths = store_paths(root)
        try:
            with Catalog(paths) as catalog:
                rebuilt = catalog.get_meta(REBUILT_WITHOUT_PROJECTS)
                refusal = self._refuse_before_candidates(catalog, rebuilt, getattr(args, "accept_rebuilt", False))
                if refusal is not None:
                    if args.json:
                        emit_error_json(refusal)
                    return 1
                stale = stale_projects(catalog)
                buckets = self._dataset_candidates(
                    catalog, paths, stale, args.older_than, args.keep_partial, getattr(args, "include_stale", False)
                )
            if rebuilt is not None:
                # Only now that at least one project is recorded and the report has been built.
                with catalog_write(paths) as catalog:
                    catalog.delete_meta(REBUILT_WITHOUT_PROJECTS)
                self.logger.warning(
                    "Cleared the flag set when the catalogue was rebuilt without projects on %s", rebuilt
                )
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        dataset_candidates = buckets["candidates"]
        leftover_candidates = self._leftover_candidates(paths)
        stale_project_rows = sorted(
            (_stale_project_row(row) for row in stale),
            key=lambda r: r["project_id"],
        )

        report: Dict[str, Any] = {
            "root": str(root),
            "datasets": dataset_candidates,
            "leftovers": [
                {"path": str(c["path"]), "bytes": c["bytes"], "reason": c["reason"]} for c in leftover_candidates
            ],
            "still_linked": buckets["still_linked"],
            "in_use": buckets["in_use"],
            "kept_stale": buckets["kept_stale"],
            "total_bytes": sum(c["bytes"] for c in dataset_candidates) + sum(c["bytes"] for c in leftover_candidates),
            "stale_projects": stale_project_rows,
            "removed_datasets": [],
            "removed_leftovers": [],
        }

        if not args.yes:
            self.logger.info("Dry run: nothing removed; pass --yes to remove")

        if args.yes and self._remove_candidates(paths, dataset_candidates, leftover_candidates, report) != 0:
            return 1

        if args.json:
            self.emit_json(report)
        else:
            self._print_report(report, args.yes)
        return 0

    def _remove_candidates(
        self,
        paths: StorePaths,
        dataset_candidates: List[Dict[str, Any]],
        leftover_candidates: List[Dict[str, Any]],
        report: Dict[str, Any],
    ) -> int:
        """Remove the candidate datasets and leftovers, recording what went in ``report``."""
        removed_datasets: List[str] = []
        for candidate in dataset_candidates:
            accession = candidate["accession"]
            _remove_path(sra_dir(paths, accession))
            removed_datasets.append(accession)
        if removed_datasets:
            try:
                with catalog_write(paths) as catalog:
                    for accession in removed_datasets:
                        catalog.delete_dataset(accession)
            except DataAccessError as e:
                self.logger.error(str(e))
                return 1

        removed_leftovers: List[str] = []
        for candidate in leftover_candidates:
            _remove_path(candidate["path"])
            removed_leftovers.append(str(candidate["path"]))

        report["removed_datasets"] = removed_datasets
        report["removed_leftovers"] = removed_leftovers
        return 0
