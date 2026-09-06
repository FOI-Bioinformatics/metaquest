"""
Shared data store CLI commands.

`store_init` creates (or reuses) a shared data store folder and records it, along with this
project's identity, in the project registry. `store_status` and `store_reindex` operate
against whichever store root resolves for the current project (an explicit `--data-root`, the
`METAQUEST_DATA` environment variable, the registry's recorded `store.root`, or the user's
default config), via `metaquest.store.resolve.resolve_store_root`.
"""

import argparse
import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import DataAccessError, MetaQuestError
from metaquest.data.registry import load_registry, registry_transaction
from metaquest.store.catalog import Catalog, catalog_write
from metaquest.store.layout import StorePaths, init_store, read_marker, sidecar_path, store_paths
from metaquest.store.resolve import resolve_store_root, write_config_data_root
from metaquest.store.sidecar import Sidecar, read_sidecar

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _no_store_hint() -> None:
    print("No store configured; run: metaquest store_init --data-root PATH")


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
        parser.add_argument("--data-root", required=True, help="Folder to use as the shared data store root")
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

    # ------------------------------------------------------------------- git

    def _gitignore_guard(self, cwd: Path) -> None:
        """Keep `fastq/` out of git for a project that has just adopted the shared store.

        Only ever reads git state (`git ls-files`) to decide whether to warn; never runs a
        command that changes the git index or working tree.
        """
        git_dir = cwd / ".git"
        if not git_dir.is_dir():
            return

        gitignore = cwd / ".gitignore"
        existing_lines = gitignore.read_text().splitlines() if gitignore.exists() else []
        if not any(line.strip() in ("fastq/", "fastq") for line in existing_lines):
            with gitignore.open("a") as handle:
                if existing_lines and existing_lines[-1] != "":
                    handle.write("\n")
                handle.write("fastq/\n")
            self.logger.info("Added fastq/ to %s", gitignore)

        try:
            result = subprocess.run(
                ["git", "ls-files", "fastq"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as e:
            self.logger.warning("Could not check git tracking of fastq/: %s", e)
            return

        if result.stdout.strip():
            self.logger.warning(
                "fastq/ is tracked by git; remove it from version control, for example: " "git rm -r --cached fastq"
            )

    # --------------------------------------------------------------- execute

    def execute(self, args: argparse.Namespace) -> int:
        try:
            root = Path(args.data_root)
            paths = init_store(root)
            with catalog_write(paths) as catalog:
                catalog.migrate()

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
                registry.store = {
                    "root": str(root.resolve()),
                    "mode": "symlink",
                    "linked": [],
                }
                project_snapshot = dict(registry.project)

            if args.set_default:
                write_config_data_root(root.resolve())
                self.logger.info("Recorded %s as the default store in the user config", root.resolve())

            self._gitignore_guard(cwd)

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
        rows = catalog.conn.execute(
            "SELECT state, COUNT(*) AS n, COALESCE(SUM(bytes_total), 0) AS bytes FROM datasets GROUP BY state"
        ).fetchall()
        counts = {row["state"]: row["n"] for row in rows}
        bytes_total = sum(row["bytes"] for row in rows)
        return counts, bytes_total

    @staticmethod
    def _datasets_list(catalog: Catalog) -> List[Dict[str, Any]]:
        rows = catalog.conn.execute("SELECT accession, state, bytes_total FROM datasets ORDER BY accession").fetchall()
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
    def _stale_project_ids(catalog: Catalog) -> List[str]:
        rows = catalog.conn.execute("SELECT project_id, path FROM projects ORDER BY project_id").fetchall()
        return [row["project_id"] for row in rows if not Path(row["path"]).exists()]

    def _build_report(self, root: Path, args: argparse.Namespace) -> Dict[str, Any]:
        paths = store_paths(root)
        marker = read_marker(root) or {}
        with Catalog(paths) as catalog:
            catalog.migrate()
            counts, bytes_total = self._dataset_counts_and_bytes(catalog)
            project_count = catalog.conn.execute("SELECT COUNT(*) AS n FROM projects").fetchone()["n"]
            stale = self._stale_project_ids(catalog)
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
            print("  Stale projects: " + ", ".join(report["stale_projects"]))
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
        return "Rebuild the store catalogue from every dataset's sidecar file"

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
    def _read_all_sidecars(paths: StorePaths) -> List[Sidecar]:
        sidecars: List[Sidecar] = []
        if not paths.sra.is_dir():
            return sidecars
        for acc_dir in sorted(paths.sra.iterdir()):
            if not acc_dir.is_dir():
                continue
            sidecar = read_sidecar(sidecar_path(paths, acc_dir.name))
            if sidecar is not None:
                sidecars.append(sidecar)
        return sidecars

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
            sidecars = self._read_all_sidecars(paths)
            with catalog_write(paths) as catalog:
                count = catalog.reindex(sidecars)
        except DataAccessError as e:
            self.logger.error(str(e))
            return 1

        print(f"Reindexed {count} dataset(s)")
        return 0
