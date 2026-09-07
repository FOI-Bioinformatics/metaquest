"""
Shared data store CLI commands.

`store_init` creates (or reuses) a shared data store folder and records it, along with this
project's identity, in the project registry. `store_status` and `store_reindex` operate
against whichever store root resolves for the current project (an explicit `--data-root`, the
`METAQUEST_DATA` environment variable, the registry's recorded `store.root`, or the user's
default config), via `metaquest.store.resolve.resolve_store_root`.

Every command here exists to operate on the store, so an unreachable one is an error with
exit 1, not something to work around; the analysis and reporting commands degrade instead
(see `metaquest.store.resolve.resolve_optional_store`). A project that links from the store
without ever running `store_init` has its identity minted on the spot
(`metaquest.store.usage.ensure_project_identity`), so `store_gc` can always see who uses what.
"""

import argparse
import json
import logging
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import DataAccessError, MetaQuestError
from metaquest.data.registry import load_registry, record_download, registry_transaction
from metaquest.data.sra import is_transient_folder, verify_download
from metaquest.store.adopt import adopt
from metaquest.store.catalog import Catalog, catalog_write
from metaquest.store.layout import StorePaths, init_store, read_marker, sidecar_path, sra_dir, store_paths
from metaquest.store.link import LINK_MODES, link_dataset, unlink_dataset
from metaquest.store.locks import lock_holder, lock_is_held
from metaquest.store.resolve import resolve_store_root, write_config_data_root
from metaquest.store.sidecar import Sidecar, md5_file, read_sidecar, sidecar_completeness, write_sidecar
from metaquest.store.usage import ensure_project_identity, linked_by, record_usage_many, stale_projects

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _no_store_hint() -> None:
    print("No store configured; run: metaquest store_init --data-root PATH")


def _stale_project_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """One stale project as reported by ``store_status`` and ``store_gc``.

    Carries the reason staleness was decided ("registry missing" reads very differently from
    "project id differs") and the host that wrote the row, since a project on another
    workstation of a shared store always looks registry-missing from here.
    """
    return {
        "project_id": row["project_id"],
        "name": row.get("name") or row["project_id"],
        "registry": row.get("registry"),
        "hostname": row.get("hostname") or "an unrecorded host",
        "reason": row.get("reason") or "registry missing",
    }


def _sidecar_completeness(paths: StorePaths, accession: str) -> Optional[Dict[str, Any]]:
    """The completeness verdict recorded in the store's sidecar for ``accession``, or None."""
    return sidecar_completeness(sidecar_path(paths, accession))


def _gitignore_guard(cwd: Path, log: logging.Logger) -> None:
    """Keep `fastq/` out of git for a project that has just adopted the shared store.

    Only ever reads git state (`git ls-files`) to decide whether to warn; never runs a
    command that changes the git index or working tree. Run from both `store_init` and
    `store_adopt`, since either can be the moment a project's reads become links.
    """
    if not (cwd / ".git").is_dir():
        return

    gitignore = cwd / ".gitignore"
    existing_lines = gitignore.read_text().splitlines() if gitignore.exists() else []
    if not any(line.strip() in ("fastq/", "fastq") for line in existing_lines):
        with gitignore.open("a") as handle:
            if existing_lines and existing_lines[-1] != "":
                handle.write("\n")
            handle.write("fastq/\n")
        log.info("Added fastq/ to %s", gitignore)

    try:
        result = subprocess.run(["git", "ls-files", "fastq"], cwd=cwd, capture_output=True, text=True, check=False)
    except OSError as e:
        log.warning("Could not check git tracking of fastq/: %s", e)
        return

    if result.stdout.strip():
        log.warning("fastq/ is tracked by git; remove it from version control, for example: git rm -r --cached fastq")


def _refuse_unusable_root(root: Path) -> None:
    """Raise unless ``root`` is either an existing store or a directory safe to make one in.

    ``store_init --data-root ~`` (or any folder already holding unrelated work) would
    otherwise scatter ``sra/``, ``tmp/``, ``locks/``, ``metadata/`` and a marker through it.
    A folder that does not exist yet, an empty one, or one that already carries a store
    marker are all fine.
    """
    if not root.exists() or read_marker(root) is not None:
        return
    if not root.is_dir():
        raise DataAccessError(f"Store root '{root}' is not a directory")
    if any(root.iterdir()):
        raise DataAccessError(
            f"Store root '{root}' is not empty and holds no store marker; "
            "point --data-root at an empty folder or an existing store"
        )


class StoreInitCommand(BaseCommand):
    """Command to initialize a shared data store and bind this project to it."""

    @property
    def name(self) -> str:
        return "store_init"

    @property
    def help(self) -> str:
        return "Initialize the shared data store and record this project's use of it"

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data-root",
            required=True,
            help="Folder to use as the shared data store root (must be empty, or an existing store)",
        )
        parser.add_argument(
            "--project-name",
            default=None,
            help="Name to record for this project (default: the working directory's name)",
        )
        parser.add_argument(
            "--set-default",
            action="store_true",
            help="Also record this root as the user's default store, in the user config file",
        )
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )

    # --------------------------------------------------------------- execute

    def _warn_on_rebinding(self, registry_store: Dict[str, Any], root: Path) -> None:
        """Say so when this project was already bound to a different store root.

        Rebinding leaves the datasets the project links from the old store exactly where they
        are; the links now point outside the store this project records, which is worth one
        line rather than silence.
        """
        previous = (registry_store or {}).get("root")
        if previous and previous != str(root):
            self.logger.warning(
                "This project was bound to the store at %s and is now bound to %s; "
                "datasets it links from the old store are untouched and still linked there",
                previous,
                root,
            )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            root = Path(args.data_root)
            _refuse_unusable_root(root)
            paths = init_store(root)
            # catalog_write migrates the schema itself; opening (and closing) it here is
            # enough to make sure catalog.sqlite exists before anything else touches it.
            with catalog_write(paths):
                pass

            cwd = Path.cwd()
            with registry_transaction(args.registry) as registry:
                existing_project = dict(registry.project) if registry.project else {}
                project_id = existing_project.get("id") or str(uuid.uuid4())
                created = existing_project.get("created") or _now()
                name = args.project_name or cwd.name

                registry.project = {
                    "id": project_id,
                    "name": name,
                    "path": str(cwd.resolve()),
                    "created": created,
                }
                self._warn_on_rebinding(registry.store, root.resolve())
                registry.store = {
                    "root": str(root.resolve()),
                    "mode": "symlink",
                    # Preserved: store_init cannot rebuild the list of datasets this project
                    # links, and resetting it would lose that record silently.
                    "linked": sorted(registry.store.get("linked") or []),
                }
                project_snapshot = dict(registry.project)
                registry_path_str = str(registry.path)

            with catalog_write(paths) as catalog:
                catalog.upsert_project(
                    project_snapshot["id"], project_snapshot["name"], project_snapshot["path"], registry_path_str
                )

            if args.set_default:
                write_config_data_root(root.resolve())
                self.logger.info("Recorded %s as the default store in the user config", root.resolve())

            _gitignore_guard(cwd, self.logger)

            self.logger.info("Store root: %s", root.resolve())
            self.logger.info("Project id: %s (%s)", project_snapshot["id"], project_snapshot["name"])
            return 0
        except MetaQuestError as e:
            self.logger.error("Error initializing store: %s", e)
            return 1


class StoreStatusCommand(BaseCommand):
    """Command to report on the shared data store: dataset counts, bytes, and stale projects."""

    @property
    def name(self) -> str:
        return "store_status"

    @property
    def help(self) -> str:
        return "Report dataset counts, bytes and stale projects for the shared data store"

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
        parser.add_argument("--verbose", action="store_true", help="Also list every dataset in the store")

    @staticmethod
    def _dataset_counts_and_bytes(catalog: Catalog) -> Any:
        """Dataset counts by state and total bytes, without the placeholder rows.

        A usage row recorded before its dataset was catalogued inserts a ``state="unknown"``
        row that stands for no files at all; counting it as a dataset would overstate what
        the store holds.
        """
        rows = catalog.conn.execute(
            "SELECT state, COUNT(*) AS n, COALESCE(SUM(bytes_total), 0) AS bytes FROM datasets "
            "WHERE state IS NOT 'unknown' GROUP BY state"
        ).fetchall()
        counts = {row["state"]: row["n"] for row in rows}
        bytes_total = sum(row["bytes"] for row in rows)
        return counts, bytes_total

    @staticmethod
    def _datasets_list(catalog: Catalog) -> List[Dict[str, Any]]:
        rows = catalog.conn.execute(
            "SELECT accession, state, bytes_total FROM datasets WHERE state IS NOT 'unknown' ORDER BY accession"
        ).fetchall()
        result = []
        for row in rows:
            project_count = len(catalog.projects_for(row["accession"]))
            result.append(
                {
                    "accession": row["accession"],
                    "state": row["state"],
                    "bytes": row["bytes_total"] or 0,
                    "projects": project_count,
                }
            )
        return result

    @staticmethod
    def _stale_project_rows(catalog: Catalog) -> List[Dict[str, Any]]:
        return [_stale_project_row(row) for row in stale_projects(catalog)]

    def _build_report(self, root: Path, args: argparse.Namespace) -> Dict[str, Any]:
        paths = store_paths(root)
        marker = read_marker(root) or {}
        with Catalog(paths) as catalog:
            counts, bytes_total = self._dataset_counts_and_bytes(catalog)
            project_count = catalog.conn.execute("SELECT COUNT(*) AS n FROM projects").fetchone()["n"]
            stale = self._stale_project_rows(catalog)
            report: Dict[str, Any] = {
                "root": str(root),
                "id": marker.get("id"),
                "datasets": counts,
                "bytes_total": bytes_total,
                "projects": project_count,
                "stale_projects": stale,
            }
            if args.verbose:
                report["datasets_list"] = self._datasets_list(catalog)
        return report

    @staticmethod
    def _print_report(report: Dict[str, Any], verbose: bool) -> None:
        print("Store")
        print("=====")
        print(f"  Root         : {report['root']}")
        print(f"  Id           : {report['id']}")
        print(f"  Bytes total  : {report['bytes_total']}")
        print(f"  Projects     : {report['projects']}")
        for state, count in sorted(report["datasets"].items()):
            print(f"  {state:<12s}: {count}")
        if report["stale_projects"]:
            print("  Stale projects:")
            for entry in report["stale_projects"]:
                print(f"    {entry['name']} on {entry['hostname']} ({entry['reason']}: {entry['registry']})")
        if verbose:
            print("\nDatasets")
            print("========")
            for entry in report.get("datasets_list", []):
                print(
                    f"  {entry['accession']:<15s} {entry['state']:<10s} "
                    f"{entry['bytes']:>12} bytes  {entry['projects']} project(s)"
                )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, registry.store.get("root"))
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        try:
            report = self._build_report(root, args)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            self._print_report(report, args.verbose)
        return 0


class StoreReindexCommand(BaseCommand):
    """Command to rebuild the store catalogue from every dataset's sidecar file."""

    @property
    def name(self) -> str:
        return "store_reindex"

    @property
    def help(self) -> str:
        return (
            "Rebuild the store catalogue from every dataset's sidecar file "
            "(refuses to run when any sidecar cannot be read)"
        )

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )

    @staticmethod
    def _read_all_sidecars(paths: StorePaths) -> Tuple[List[Sidecar], List[str]]:
        """Every dataset's sidecar, plus the accessions whose sidecar could not be read.

        A reindex rebuilds ``datasets`` from exactly what it reads, so a sidecar missed here
        would look like a dataset that no longer exists. The caller stops rather than acting
        on a partial reading.
        """
        sidecars: List[Sidecar] = []
        unreadable: List[str] = []
        if not paths.sra.is_dir():
            return sidecars, unreadable
        for acc_dir in sorted(paths.sra.iterdir()):
            if not acc_dir.is_dir() or acc_dir.name == ".sra-cache":
                continue
            sidecar = read_sidecar(sidecar_path(paths, acc_dir.name))
            if sidecar is None:
                unreadable.append(acc_dir.name)
                continue
            sidecars.append(sidecar)
        return sidecars, unreadable

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, registry.store.get("root"))
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        try:
            paths = store_paths(root)
            sidecars, unreadable = self._read_all_sidecars(paths)
            if unreadable:
                self.logger.error(
                    "Not reindexing: %d sidecar(s) could not be read, and rebuilding from a "
                    "partial reading would drop those datasets and their usage history: %s",
                    len(unreadable),
                    ", ".join(unreadable),
                )
                return 1
            with catalog_write(paths) as catalog:
                count = catalog.reindex(sidecars)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        print(f"Reindexed {count} dataset(s)")
        return 0


class StoreAdoptCommand(BaseCommand):
    """Command to fold a project's own downloaded FASTQ folders into the shared store.

    Each accession is staged and moved under its own per-accession lock (see
    ``metaquest.store.adopt``), so running this command concurrently against the same store from
    two projects is safe: a shared accession simply serialises rather than racing, and only the
    project's own ``fastq/<ACC>`` folders are ever claimed. Store folders belonging to other
    projects, accessions another run is publishing right now, and accessions the store's
    filesystem has no room to stage are reported and left alone.
    """

    @property
    def name(self) -> str:
        return "store_adopt"

    @property
    def help(self) -> str:
        return (
            "Move or copy project-owned FASTQ folders into the shared store, then link them back "
            "(each accession is locked, so this is safe to run concurrently from several projects)"
        )

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "--move",
            dest="move",
            action="store_true",
            default=True,
            help="Remove each accession's project folder and link it to the store's copy (default)",
        )
        mode.add_argument(
            "--copy",
            dest="move",
            action="store_false",
            help="Leave each accession's project folder as is; the store keeps its own copy, unlinked",
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Report what would be adopted without changing anything"
        )
        parser.add_argument(
            "--compress",
            dest="compress",
            action="store_true",
            default=True,
            help="Gzip-compress plain FASTQ files while adopting them (default: on)",
        )
        parser.add_argument(
            "--no-compress", dest="compress", action="store_false", help="Leave FASTQ files uncompressed"
        )
        parser.add_argument(
            "--metadata-folder",
            default="metadata",
            help="Folder holding NCBI metadata XML, consulted for each accession's recorded spot count",
        )
        parser.add_argument(
            "--lock-wait",
            dest="lock_wait",
            type=float,
            default=0.0,
            help=(
                "Seconds to wait for another project's work on the same accession before giving "
                "up on it (default: 0, wait for as long as the other project keeps working)"
            ),
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, registry.store.get("root"))
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        try:
            paths = store_paths(root)
            report = adopt(
                args.fastq_folder,
                paths,
                move=args.move,
                dry_run=args.dry_run,
                compress=args.compress,
                metadata_folders=[Path(args.metadata_folder), paths.metadata],
                lock_wait=getattr(args, "lock_wait", 0.0),
            )
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if args.dry_run:
            print(f"Would adopt {len(report.planned)} dataset(s)")
            if report.planned:
                print("  " + ", ".join(sorted(report.planned)))
            if report.conflicts:
                print(f"Conflicts (left in place): {', '.join(sorted(report.conflicts))}")
            return 0

        # Only an accession the project now links to (adopted via --move, or deduplicated,
        # which always links) needs its download record pointed at the store; a --copy
        # accession keeps its project folder exactly as it was, unlinked.
        newly_linked = sorted(set(report.adopted) | set(report.deduplicated))
        if newly_linked:
            self._record_linked(args, paths, newly_linked)
        self._print_report(report)
        return 0

    @staticmethod
    def _record_linked(args: argparse.Namespace, paths: StorePaths, newly_linked: List[str]) -> None:
        """Point the registry's download records at the store and record the linked usage."""
        with registry_transaction(args.registry) as reg:
            ensure_project_identity(reg)
            for acc in newly_linked:
                complete = _sidecar_completeness(paths, acc)
                record_download(
                    reg,
                    acc,
                    "downloaded",
                    args.fastq_folder,
                    attempt=False,
                    complete=complete,
                    source="store",
                    store_name=acc,
                )
            linked = set(reg.store.get("linked") or [])
            linked.update(newly_linked)
            reg.store["linked"] = sorted(linked)
            usage_registry = reg
        # Recorded outside the transaction: the catalogue lock is a separate wait, and
        # holding the project's registry lock while queueing for it can time the registry
        # write out.
        record_usage_many(paths, usage_registry, [(acc, "", "linked", "store_adopt") for acc in newly_linked])
        _gitignore_guard(Path.cwd(), logger)

    def _print_report(self, report: Any) -> None:
        print(
            f"Adopted {len(report.adopted)}, copied {len(report.copied)}, "
            f"deduplicated {len(report.deduplicated)}, conflicts {len(report.conflicts)}, "
            f"skipped {len(report.skipped)}"
        )
        for label, accessions in (
            ("Resumed after an interrupted run", report.resumed),
            ("Left to their own project (no sidecar, not ours)", report.foreign),
            ("In progress elsewhere", report.in_progress),
            ("Refused for lack of free space", report.refused),
        ):
            if accessions:
                print(f"{label}: {', '.join(sorted(accessions))}")
        if report.conflicts:
            self.logger.warning("Conflicting accessions left in place: %s", ", ".join(sorted(report.conflicts)))


class StoreVerifyCommand(BaseCommand):
    """Command to verify store datasets against their sidecars, optionally by md5 or spot count."""

    @property
    def name(self) -> str:
        return "store_verify"

    @property
    def help(self) -> str:
        return "Verify store datasets against their sidecars (bytes, optionally md5 and NCBI spot counts)"

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("accessions", nargs="*", help="Accessions to verify (default: every dataset in the store)")
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        parser.add_argument("--md5", action="store_true", help="Recompute and compare each file's md5 checksum")
        parser.add_argument(
            "--spots", action="store_true", help="Compare the read count against NCBI's recorded spot count"
        )
        parser.add_argument(
            "--fix-state",
            action="store_true",
            help="Rewrite the sidecar state and catalogue entry when a check finds a mismatch",
        )

    @staticmethod
    def _accessions_to_check(args: argparse.Namespace, paths: StorePaths) -> List[str]:
        if args.accessions:
            return list(args.accessions)
        if not paths.sra.is_dir():
            return []
        return sorted(p.name for p in paths.sra.iterdir() if p.is_dir())

    @staticmethod
    def _check_bytes_and_md5(accession: str, store_dir: Path, sidecar: Sidecar, check_md5: bool):
        """Returns ``(bytes_ok, md5_ok, detail)``; ``detail`` names the first file and reason
        for any bytes or md5 mismatch (or missing/unreadable file), else None."""
        bytes_ok = True
        md5_ok: Optional[bool] = True if check_md5 else None
        detail: Optional[str] = None
        for entry in sidecar.files:
            name = str(entry.get("name"))
            file_path = store_dir / name
            if not file_path.is_file():
                bytes_ok = False
                if check_md5:
                    md5_ok = False
                detail = detail or f"{name}: missing"
                continue
            try:
                actual_bytes = file_path.stat().st_size
                if actual_bytes != entry.get("bytes"):
                    bytes_ok = False
                    detail = detail or f"{name}: size mismatch ({actual_bytes} vs {entry.get('bytes')} bytes)"
                if check_md5:
                    actual_md5 = md5_file(file_path)
                    if actual_md5 != entry.get("md5"):
                        md5_ok = False
                        detail = detail or f"{name}: md5 mismatch"
            except OSError as e:
                logger.warning("%s: could not read %s: %s", accession, file_path, e)
                bytes_ok = False
                if check_md5:
                    md5_ok = False
                detail = detail or f"{name}: {e}"
        return bytes_ok, md5_ok, detail

    def _verify_one(self, accession: str, paths: StorePaths, check_md5: bool, check_spots: bool) -> Dict[str, Any]:
        store_dir = sra_dir(paths, accession)
        sc_path = sidecar_path(paths, accession)
        sidecar = read_sidecar(sc_path)
        if sidecar is None:
            return {
                "accession": accession,
                "state": "missing",
                "bytes_ok": False,
                "md5_ok": None,
                "verdict": "missing",
                "sidecar": None,
                "spots_verdict": None,
                "spots_ratio": None,
                "mismatch_detail": None,
            }

        bytes_ok, md5_ok, detail = self._check_bytes_and_md5(accession, store_dir, sidecar, check_md5)

        spots_verdict = None
        spots_ratio = None
        if check_spots:
            try:
                verify = verify_download(accession, store_dir, sidecar.ncbi.get("spots"))
                spots_verdict = verify["verdict"]
                spots_ratio = verify["ratio"]
            except (EOFError, OSError) as e:
                self.logger.warning("%s: could not verify read counts: %s", accession, e)
                bytes_ok = False
                spots_verdict = "corrupt"
                detail = detail or f"read count verification failed: {e}"

        if not bytes_ok or (check_md5 and md5_ok is False):
            verdict = "corrupt"
        elif check_spots and spots_verdict == "truncated":
            verdict = "truncated"
        elif check_spots and spots_verdict == "unverified":
            verdict = "unverified"
        else:
            verdict = "ok"

        return {
            "accession": accession,
            "state": sidecar.state,
            "bytes_ok": bytes_ok,
            "md5_ok": md5_ok,
            "verdict": verdict,
            "sidecar": sidecar,
            "spots_verdict": spots_verdict,
            "spots_ratio": spots_ratio,
            "mismatch_detail": detail,
        }

    def _fix_state(self, result: Dict[str, Any], paths: StorePaths) -> None:
        """Rewrite the sidecar's state (and, for a spots-only mismatch, its completeness) to
        match what this check found.

        A bytes or md5 mismatch always wins: the file itself is wrong, so the dataset is
        ``"failed"`` with an error naming the first mismatch, regardless of what the spots
        check says (a corrupt file can still happen to contain the right number of reads). Only
        when bytes and md5 (if checked) both check out does the spots verdict decide
        ``complete``/``partial``.
        """
        sidecar = result.get("sidecar")
        if sidecar is None:
            return

        if not result.get("bytes_ok") or result.get("md5_ok") is False:
            error = result.get("mismatch_detail") or "verify: bytes or md5 mismatch"
            if sidecar.state == "failed" and sidecar.error == error:
                return
            sidecar.state = "failed"
            sidecar.error = error
            write_sidecar(sidecar_path(paths, result["accession"]), sidecar)
            with catalog_write(paths) as catalog:
                catalog.upsert_dataset(sidecar)
            result["state"] = "failed"
            return

        spots_verdict = result.get("spots_verdict")
        state_for_verdict = {"complete": "complete", "truncated": "partial"}
        if spots_verdict not in state_for_verdict:
            # "unverified" (no recorded spot count) or "corrupt" (the files could not be read
            # for the spots check specifically, bytes/md5 having passed): neither is a state
            # this check can confidently rewrite.
            return
        new_state = state_for_verdict[spots_verdict]
        if new_state == sidecar.state and sidecar.completeness.get("verdict") == spots_verdict:
            return
        sidecar.state = new_state
        sidecar.completeness = {"method": "spots", "ratio": result.get("spots_ratio"), "verdict": spots_verdict}
        write_sidecar(sidecar_path(paths, result["accession"]), sidecar)
        with catalog_write(paths) as catalog:
            catalog.upsert_dataset(sidecar)
        result["state"] = new_state

    @staticmethod
    def _print_table(results: List[Dict[str, Any]]) -> None:
        print(f"{'accession':<15s} {'state':<10s} {'bytes_ok':<9s} {'md5_ok':<7s} verdict")
        for r in results:
            md5_col = "-" if r["md5_ok"] is None else str(r["md5_ok"])
            print(f"{r['accession']:<15s} {r['state']:<10s} {str(r['bytes_ok']):<9s} {md5_col:<7s} {r['verdict']}")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, registry.store.get("root"))
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        try:
            paths = store_paths(root)
            accessions = self._accessions_to_check(args, paths)
            results = []
            for accession in accessions:
                result = self._verify_one(accession, paths, args.md5, args.spots)
                if args.fix_state:
                    self._fix_state(result, paths)
                results.append(result)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        self._print_table(results)
        failed = any(r["verdict"] in ("corrupt", "truncated", "missing") for r in results)
        return 1 if failed else 0


class StoreLinkCommand(BaseCommand):
    """Command to link project accessions to the shared store's copies."""

    @property
    def name(self) -> str:
        return "store_link"

    @property
    def help(self) -> str:
        return "Link project accessions to the shared store's copies (complete datasets only, unless --accept-partial)"

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("accessions", nargs="+", help="Accessions to link from the store")
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--link-mode",
            choices=list(LINK_MODES),
            default="auto",
            help=(
                "How this project points at the store's copy: a relative or absolute symlink, "
                "a copy of the folder, or auto (relative when the store and the project share a "
                "parent folder)"
            ),
        )
        parser.add_argument(
            "--accept-partial",
            dest="accept_partial",
            action="store_true",
            default=False,
            help="Link a dataset whose store copy is incomplete, failed or undescribed",
        )

    def _refuse_incomplete(self, paths: StorePaths, accession: str, accept_partial: bool) -> bool:
        """True when ``accession`` must not be linked as it stands.

        The store's sidecar is the record of whether a dataset is usable. A ``partial`` or
        ``failed`` one, or one with no sidecar at all (an interrupted run, or a folder put
        there by hand), reads as complete once it is linked into ``fastq/``, so linking it
        silently would feed an unfinished dataset into every later analysis.
        """
        sidecar = read_sidecar(sidecar_path(paths, accession))
        if sidecar is None:
            state = "no sidecar"
        elif sidecar.state in ("partial", "failed", "downloading"):
            state = sidecar.state
        else:
            return False

        if accept_partial:
            self.logger.warning("%s: linking a dataset the store records as %s", accession, state)
            return False
        self.logger.error(
            "%s: the store records this dataset as %s; rerun with --accept-partial to link it anyway",
            accession,
            state,
        )
        return True

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, registry.store.get("root"))
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        paths = store_paths(root)
        accept_partial = getattr(args, "accept_partial", False)
        linked: List[str] = []
        failed: List[str] = []
        for accession in args.accessions:
            if self._refuse_incomplete(paths, accession, accept_partial):
                failed.append(accession)
                continue
            try:
                link_dataset(args.fastq_folder, accession, paths, mode=args.link_mode)
                linked.append(accession)
            except DataAccessError as e:
                self.logger.error("%s: %s", accession, e)
                failed.append(accession)

        if linked:
            with registry_transaction(args.registry) as reg:
                ensure_project_identity(reg)
                for accession in linked:
                    complete = _sidecar_completeness(paths, accession)
                    record_download(
                        reg,
                        accession,
                        "downloaded",
                        args.fastq_folder,
                        attempt=False,
                        complete=complete,
                        source="store",
                        store_name=accession,
                    )
                reg_linked = set(reg.store.get("linked") or [])
                reg_linked.update(linked)
                reg.store["linked"] = sorted(reg_linked)
                usage_registry = reg
            # Outside the transaction: see the note in store_adopt.
            record_usage_many(paths, usage_registry, [(acc, "", "linked", "store_link") for acc in linked])

        print(f"Linked {len(linked)} of {len(args.accessions)} accession(s)")
        return 1 if failed else 0


class StoreUnlinkCommand(BaseCommand):
    """Command to remove a project's store link, without touching the store's copy."""

    @property
    def name(self) -> str:
        return "store_unlink"

    @property
    def help(self) -> str:
        return "Remove a project's link to the store (never removes a real directory)"

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("accessions", nargs="+", help="Accessions to unlink")
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        removed: List[str] = []
        refused: List[str] = []
        for accession in args.accessions:
            link_path = Path(args.fastq_folder) / accession
            if link_path.exists() and not link_path.is_symlink():
                self.logger.error("%s is a real directory, not a store link; refusing to remove it", accession)
                refused.append(accession)
                continue
            try:
                was_removed = unlink_dataset(args.fastq_folder, accession)
            except DataAccessError as e:
                self.logger.error(str(e))
                refused.append(accession)
                continue
            if was_removed:
                removed.append(accession)

        if removed:
            with registry_transaction(args.registry) as reg:
                for accession in removed:
                    record_download(reg, accession, "missing", args.fastq_folder, attempt=False)
                reg_linked = set(reg.store.get("linked") or [])
                reg_linked.difference_update(removed)
                reg.store["linked"] = sorted(reg_linked)

        print(f"Unlinked {len(removed)} of {len(args.accessions)} accession(s)")
        return 1 if refused else 0


class StoreUsageCommand(BaseCommand):
    """Command to report catalogue usage: by accession, project, organism, or store-wide."""

    _COLUMNS: Dict[str, List[Tuple[str, str]]] = {
        "accession": [
            ("project_name", "project"),
            ("project_id", "id"),
            ("genome_id", "genome_id"),
            ("stage", "stage"),
            ("first_used", "first_used"),
            ("last_used", "last_used"),
        ],
        "project": [
            ("accession", "accession"),
            ("genome_id", "genome_id"),
            ("stage", "stage"),
            ("last_used", "last_used"),
        ],
        "organism": [("accession", "accession"), ("project_name", "project"), ("stage", "stage")],
        "unused": [("accession", "accession"), ("state", "state"), ("bytes", "bytes")],
        "bytes-by-organism": [("genome_id", "genome_id"), ("datasets", "datasets"), ("bytes", "bytes")],
    }

    @property
    def name(self) -> str:
        return "store_usage"

    @property
    def help(self) -> str:
        return "Report catalogue usage by accession, project, organism, or store-wide"

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        selector = parser.add_mutually_exclusive_group(required=True)
        selector.add_argument("--accession", default=None, help="Every project that has used this accession")
        selector.add_argument("--project", default=None, help="Every dataset this project (by name or id) has used")
        selector.add_argument("--organism", default=None, help="Every dataset used for this target genome id")
        selector.add_argument(
            "--unused", action="store_true", help="Datasets in the store with no recorded usage at all"
        )
        selector.add_argument(
            "--bytes-by-organism", action="store_true", help="Total bytes and dataset counts, grouped by genome id"
        )
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON")

    # --------------------------------------------------------------- queries

    @staticmethod
    def _resolve_project_id(catalog: Catalog, value: str) -> Tuple[str, List[str]]:
        """Resolve ``value`` (a project id or name) to the id to query.

        Returns ``(project_id, [])`` when ``value`` is itself a known project id, or matches
        exactly one project's name. Returns ``(value, [])`` unchanged when it matches no
        project at all (the caller's query then simply returns no rows). Returns
        ``(None-ish, ambiguous_ids)`` when ``value`` matches more than one project's name; the
        caller must treat a non-empty second element as an error.
        """
        row = catalog.conn.execute("SELECT project_id FROM projects WHERE project_id = ?", (value,)).fetchone()
        if row is not None:
            return row["project_id"], []
        rows = catalog.conn.execute(
            "SELECT project_id FROM projects WHERE name = ? ORDER BY project_id", (value,)
        ).fetchall()
        ids = [r["project_id"] for r in rows]
        if len(ids) == 1:
            return ids[0], []
        if len(ids) > 1:
            return value, ids
        return value, []

    @staticmethod
    def _rows_for_accession(catalog: Catalog, accession: str) -> List[Dict[str, Any]]:
        rows = catalog.conn.execute(
            """
            SELECT p.name AS project_name, p.project_id AS project_id, u.genome_id AS genome_id,
                   u.stage AS stage, u.first_used AS first_used, u.last_used AS last_used
            FROM usage u
            JOIN projects p ON p.project_id = u.project_id
            WHERE u.accession = ?
            ORDER BY p.project_id, u.genome_id, u.stage
            """,
            (accession,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _rows_for_project(catalog: Catalog, project_id: str) -> List[Dict[str, Any]]:
        rows = catalog.conn.execute(
            """
            SELECT DISTINCT u.accession AS accession, u.genome_id AS genome_id, u.stage AS stage,
                   u.last_used AS last_used
            FROM usage u
            WHERE u.project_id = ?
            ORDER BY u.accession, u.genome_id, u.stage
            """,
            (project_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _rows_for_organism(catalog: Catalog, genome_id: str) -> List[Dict[str, Any]]:
        rows = catalog.conn.execute(
            """
            SELECT DISTINCT u.accession AS accession, p.name AS project_name, u.stage AS stage
            FROM usage u
            JOIN projects p ON p.project_id = u.project_id
            WHERE u.genome_id = ?
            ORDER BY u.accession, p.name, u.stage
            """,
            (genome_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _rows_unused(catalog: Catalog) -> List[Dict[str, Any]]:
        rows = catalog.conn.execute("""
            SELECT d.accession AS accession, d.state AS state, COALESCE(d.bytes_total, 0) AS bytes
            FROM datasets d
            LEFT JOIN usage u ON u.accession = d.accession
            WHERE u.accession IS NULL
            ORDER BY d.accession
            """).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _rows_bytes_by_organism(catalog: Catalog) -> List[Dict[str, Any]]:
        rows = catalog.conn.execute("""
            SELECT pair.genome_id AS genome_id, COUNT(DISTINCT pair.accession) AS datasets,
                   COALESCE(SUM(d.bytes_total), 0) AS bytes
            FROM (SELECT DISTINCT genome_id, accession FROM usage) pair
            JOIN datasets d ON d.accession = pair.accession
            GROUP BY pair.genome_id
            ORDER BY pair.genome_id
            """).fetchall()
        return [dict(row) for row in rows]

    # ---------------------------------------------------------------- print

    @classmethod
    def _print_rows(cls, selector: str, rows: List[Dict[str, Any]]) -> None:
        columns = cls._COLUMNS[selector]
        print("  ".join(f"{label:<15s}" for _, label in columns))
        for row in rows:
            print("  ".join(f"{str(row.get(key, '')):<15s}" for key, _ in columns))

    # --------------------------------------------------------------- execute

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, registry.store.get("root"))
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        try:
            paths = store_paths(root)
            with Catalog(paths) as catalog:
                if args.accession:
                    selector = "accession"
                    rows = self._rows_for_accession(catalog, args.accession)
                elif args.project:
                    selector = "project"
                    project_id, ambiguous = self._resolve_project_id(catalog, args.project)
                    if ambiguous:
                        self.logger.error(
                            "Project name %r is ambiguous: %s", args.project, ", ".join(sorted(ambiguous))
                        )
                        return 1
                    rows = self._rows_for_project(catalog, project_id)
                elif args.organism:
                    selector = "organism"
                    rows = self._rows_for_organism(catalog, args.organism)
                elif args.unused:
                    selector = "unused"
                    rows = self._rows_unused(catalog)
                else:
                    selector = "bytes-by-organism"
                    rows = self._rows_bytes_by_organism(catalog)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if args.json:
            print(json.dumps({"selector": selector, "rows": rows}, indent=2))
        else:
            self._print_rows(selector, rows)
        return 0


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
        return "store_gc"

    @property
    def help(self) -> str:
        return (
            "Report, and with --yes remove, unused datasets and leftover temp files from the store "
            "(datasets a project links, or another run is working on, are always kept)"
        )

    @property
    def group(self) -> str:
        return "Store"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
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
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON")

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
        """The accession a leftover folder belongs to, e.g. ``SRR1`` for ``SRR1_temp``."""
        for suffix in ("_temp", "_adopt", "_old"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @classmethod
    def _leftover_candidates(cls, paths: StorePaths) -> List[Dict[str, Any]]:
        """Leftover build folders and cached archives, minus anything a live run is using.

        A ``<ACC>_temp`` folder or a cached ``.sra`` archive whose accession lock is held is a
        download in progress, not a leftover: removing it would pull the files out from under
        a running fasterq-dump.
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

    @staticmethod
    def _print_report(report: Dict[str, Any], performed: bool) -> None:
        verb = "Removed" if performed else "Would remove"
        print(
            f"{verb} {len(report['datasets'])} dataset(s), {len(report['leftovers'])} leftover(s), "
            f"{report['total_bytes']} bytes total"
        )
        for entry in report["datasets"]:
            print(f"  dataset   {entry['accession']:<15s} {entry['bytes']:>12} bytes  {entry['reason']}")
        for entry in report["leftovers"]:
            print(f"  leftover  {entry['path']:<40s} {entry['bytes']:>12} bytes  {entry['reason']}")
        for key in ("still_linked", "in_use", "kept_stale"):
            for entry in report.get(key, []):
                print(f"  kept      {entry['accession']:<15s} {entry['bytes']:>12} bytes  {entry['reason']}")
        if report["stale_projects"]:
            print("Stale projects:")
            for entry in report["stale_projects"]:
                print(f"  {entry['name']} on {entry['hostname']} ({entry['reason']}: {entry['registry']})")

    # --------------------------------------------------------------- execute

    def execute(self, args: argparse.Namespace) -> int:
        if args.dry_run and args.yes:
            self.logger.error("--dry-run and --yes cannot be combined")
            return 1

        try:
            registry = load_registry(args.registry)
            root = resolve_store_root(args.data_root, registry.store.get("root"))
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        if root is None:
            _no_store_hint()
            return 1

        paths = store_paths(root)
        try:
            with Catalog(paths) as catalog:
                stale = stale_projects(catalog)
                buckets = self._dataset_candidates(
                    catalog, paths, stale, args.older_than, args.keep_partial, getattr(args, "include_stale", False)
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
            print(json.dumps(report, indent=2))
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
