"""
SRA-related CLI commands.
"""

import argparse
import csv
import os
import shutil
from typing import Callable, Optional, Set

from metaquest.cli.base import BaseCommand
from pathlib import Path

from metaquest.core.constants import FAILED_ACCESSIONS_FILE
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.registry import (
    Registry,
    load_registry,
    project_root,
    query,
    record_download,
    registry_transaction,
)
from metaquest.data.sra import default_max_workers, download_sra, parse_verdict_message, transient_bytes
from metaquest.store.layout import StorePaths, sidecar_path, store_paths
from metaquest.store.link import LINK_MODES, is_store_link
from metaquest.store.resolve import resolve_store_root
from metaquest.store.sidecar import read_sidecar
from metaquest.store.usage import record_usage_many, record_usage_safe

# Markers the data layer puts in a result message for a dataset the shared store provided
# (linked from a copy already there) or received (downloaded into it by this run).
STORE_LINKED_PREFIX = "linked from store"
STORE_SAVED_SUFFIX = "; stored"

# Kept .sra-cache archives and <ACC>_temp build folders bigger than this, summed across a
# run's candidate folders, are worth a warning: they are easy to forget about and can
# quietly use up a lot of disk.
TRANSIENT_BYTES_WARN_THRESHOLD = 1024**3


class DownloadSraCommand(BaseCommand):
    """Command for downloading SRA datasets."""

    @property
    def name(self) -> str:
        return "download_sra"

    @property
    def help(self) -> str:
        return "Download SRA datasets"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--fastq-folder",
            default="fastq",
            help="Folder to save downloaded FASTQ files",
        )
        parser.add_argument(
            "--accessions-file",
            required=True,
            help="File containing SRA accessions, one per line",
        )
        parser.add_argument(
            "--max-downloads",
            type=int,
            default=None,
            help="Maximum number of datasets to download",
        )
        parser.add_argument(
            "--num-threads",
            type=int,
            default=4,
            help="Number of threads for each fasterq-dump",
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=None,
            help="Number of parallel downloads (default: computed from the CPU count)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Calculate number of accessions without downloading",
        )
        parser.add_argument("--force", action="store_true", help="Force redownload even if files exist")
        parser.add_argument(
            "--max-retries",
            type=int,
            default=1,
            help="Maximum number of retry attempts for failed downloads",
        )
        parser.add_argument(
            "--temp-folder",
            help="Directory to use for fasterq-dump temporary files (must be writable)",
        )
        parser.add_argument(
            "--blacklist",
            nargs="+",
            help="One or more files containing blacklisted accessions, one per line",
        )
        parser.add_argument(
            "--report-file",
            default=None,
            help=(
                "Write a CSV of accession,status,message after the run "
                "(statuses: downloaded, failed, already_present, blacklisted, skipped); "
                "accessions skipped by --max-downloads get a skipped row"
            ),
        )
        parser.add_argument(
            "--registry",
            default=None,
            help="Path to the project registry file (defaults to the nearest metaquest_registry.json)",
        )
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument(
            "--verify-downloads",
            dest="verify_downloads",
            action="store_true",
            default=True,
            help="Verify each download's read count against NCBI's recorded total spots (default: on)",
        )
        parser.add_argument(
            "--no-verify-downloads",
            dest="verify_downloads",
            action="store_false",
            help="Skip completeness verification against NCBI spot counts",
        )
        parser.add_argument(
            "--redownload-truncated",
            action="store_true",
            help="Redownload accessions whose registry verdict is 'truncated' rather than skipping them",
        )
        parser.add_argument(
            "--sra-cache",
            default=None,
            help="Directory for prefetch's downloaded .sra archives (default: <fastq-folder>/.sra-cache)",
        )
        parser.add_argument(
            "--no-prefetch",
            dest="use_prefetch",
            action="store_false",
            default=True,
            help="Run fasterq-dump directly against the accession instead of prefetch then fasterq-dump",
        )
        parser.add_argument(
            "--keep-sra",
            action="store_true",
            help="Keep the downloaded .sra archive after a successful, verified download",
        )
        parser.add_argument(
            "--compress",
            dest="compress",
            action="store_true",
            default=True,
            help="Gzip each downloaded FASTQ file (default: on)",
        )
        parser.add_argument(
            "--no-compress",
            dest="compress",
            action="store_false",
            help="Leave downloaded FASTQ files uncompressed",
        )
        parser.add_argument(
            "--link-mode",
            choices=list(LINK_MODES),
            default="auto",
            help=(
                "How this project points at a dataset in the shared store: a relative or "
                "absolute symlink, a copy of the folder, or auto (relative when the store "
                "and the project share a parent folder)"
            ),
        )
        parser.add_argument(
            "--accept-partial",
            action="store_true",
            help="Use a store dataset whose download is incomplete instead of refusing it",
        )
        parser.add_argument(
            "--no-resume-partial",
            dest="resume_partial",
            action="store_false",
            default=True,
            help="Do not download an incomplete store dataset again",
        )

    def _log_dry_run_summary(self, args: argparse.Namespace, stats: dict) -> None:
        """Log the summary for a dry run."""
        self.logger.info(f"Dry run: would download {stats['to_download']} of {stats['total']} datasets")
        self.logger.info(f"  {stats['already_downloaded']} datasets would be skipped (already downloaded)")
        if stats.get("blacklisted", 0) > 0:
            self.logger.info(f"  {stats['blacklisted']} datasets would be skipped (blacklisted)")
        if stats.get("to_download", 0) > 0:
            self.logger.info(f"  Output folder would be: {args.fastq_folder}")
            if args.max_downloads:
                self.logger.info(f"  Limited to {args.max_downloads} downloads")

    def _log_download_summary(self, stats: dict) -> None:
        """Log the summary for a completed download run."""
        self.logger.info("Download summary:")
        self.logger.info(f"  Successfully downloaded: {stats['successful']} datasets")
        self.logger.info(f"  Failed downloads: {stats['failed']} datasets")
        self.logger.info(f"  Already downloaded: {stats['already_downloaded']} datasets")
        if stats.get("blacklisted", 0) > 0:
            self.logger.info(f"  Blacklisted: {stats['blacklisted']} datasets")
        self.logger.info(f"  Total processed: {stats['total']} datasets")

    def _report_failed_downloads(self, args: argparse.Namespace, stats: dict) -> None:
        """Warn about failures and point at the retry file the data layer already wrote."""
        self.logger.warning("Some downloads failed. Use --force to retry or --max-retries to enable automatic retry.")
        if not stats.get("failed_accessions"):
            return
        failed_file = Path(args.fastq_folder) / FAILED_ACCESSIONS_FILE
        self.logger.info(
            f"To retry only failed accessions: metaquest download_sra "
            f"--accessions-file {failed_file} "
            f"--fastq-folder {args.fastq_folder}"
        )

    @staticmethod
    def _write_report(report_file: str, stats: dict) -> None:
        """Write one row per accession with its outcome."""
        failed = set(stats.get("failed_accessions", []))
        rows = []
        for accession, message in stats.get("results", {}).items():
            rows.append((accession, "failed" if accession in failed else "downloaded", message))
        rows.extend((acc, "already_present", "") for acc in stats.get("already_downloaded_accessions", []))
        rows.extend((acc, "blacklisted", "") for acc in stats.get("blacklisted_accessions", []))
        rows.extend((acc, "skipped", "--max-downloads") for acc in stats.get("skipped_accessions", []))
        path = Path(report_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["accession", "status", "message"])
            writer.writerows(sorted(rows))

    @staticmethod
    def _record_skip(reg: Registry, acc: str, message: str, fastq_dir: Path) -> None:
        """Record a skipped accession, unless its record already says the reads are downloaded.

        A blacklist entry or a `--max-downloads` cut-off must never erase the files and
        sizes of an accession that was downloaded on an earlier run.
        """
        if reg.datasets.get(acc, {}).get("download", {}).get("state") == "downloaded":
            return
        record_download(reg, acc, "skipped", fastq_dir, message)

    def _record_run_outcomes(
        self, args: argparse.Namespace, stats: dict, fastq_dir: Path, store: Optional[StorePaths] = None
    ) -> None:
        """Record the outcomes the download loop could not report, one transaction per accession.

        Every already-downloaded accession the shared store backs is also recorded as
        ``"linked"`` usage in the store catalogue, in one batched write after the loop
        (``record_usage_many``) rather than one lock per accession.
        """
        usage_rows = []
        for acc in stats.get("already_downloaded_accessions", []):
            from_store = store is not None and is_store_link(fastq_dir / acc, store)
            with registry_transaction(args.registry) as reg:
                if reg.datasets.get(acc, {}).get("download", {}).get("state") != "downloaded":
                    complete = self._sidecar_completeness(store, acc) if from_store else None
                    record_download(
                        reg,
                        acc,
                        "downloaded",
                        fastq_dir,
                        attempt=False,
                        complete=complete,
                        source="store" if from_store else None,
                        store_name=acc if from_store else None,
                    )
                    if from_store:
                        self._mark_linked(reg, acc)
            if from_store:
                usage_rows.append((acc, "", "linked", "already downloaded"))
        if usage_rows:
            usage_registry = load_registry(args.registry)
            record_usage_many(store, usage_registry, usage_rows)
        for acc in stats.get("blacklisted_accessions", []):
            with registry_transaction(args.registry) as reg:
                self._record_skip(reg, acc, "blacklisted", fastq_dir)
        for acc in stats.get("skipped_accessions", []):
            with registry_transaction(args.registry) as reg:
                self._record_skip(reg, acc, "--max-downloads", fastq_dir)

    def _resolve_max_workers(self, args: argparse.Namespace) -> int:
        """Resolve --max-workers, falling back to a CPU-derived default.

        Oversubscription is only worth warning about when the user chose the worker count:
        the derived default is already floored at one worker, so a single many-threaded
        download on a small machine is nothing the user could have set differently.
        """
        explicit = args.max_workers is not None
        max_workers = args.max_workers if explicit else default_max_workers(args.num_threads)
        cpu_count = os.cpu_count() or 4
        if explicit and max_workers * args.num_threads > cpu_count:
            self.logger.warning(
                "--max-workers %d x --num-threads %d = %d threads requested, which exceeds "
                "the %d CPUs detected on this machine; downloads may be slower than expected",
                max_workers,
                args.num_threads,
                max_workers * args.num_threads,
                cpu_count,
            )
        return max_workers

    def _resolve_store(self, args: argparse.Namespace, project_registry: Registry) -> Optional[StorePaths]:
        """Resolve the shared data store (if any), log it, and return its on-disk layout."""
        store_root = resolve_store_root(args.data_root, project_registry.store.get("root"))
        if store_root is None:
            return None
        self.logger.info("Using shared data store at %s", store_root)
        return store_paths(store_root)

    def _result_recorder(
        self, args: argparse.Namespace, fastq_dir: Path, store: Optional[StorePaths] = None
    ) -> Callable[[str, bool, str], None]:
        """The callback the download loop uses to record each accession's outcome.

        A dataset linked from the shared store is recorded as downloaded without counting an
        attempt against it, since no download ran; one this run downloaded into the store
        counts as an attempt like any other. Both are added to the project's list of linked
        datasets, and recorded as store catalogue usage: ``"linked"`` for a dataset the store
        already held, ``"downloaded"`` for one this run saved into it.
        """

        def _record_result(accession: str, success: bool, message: str) -> None:
            linked = bool(success) and message.startswith(STORE_LINKED_PREFIX)
            from_store = linked or (bool(success) and message.endswith(STORE_SAVED_SUFFIX))
            with registry_transaction(args.registry) as reg:
                complete = parse_verdict_message(message) if success else None
                if complete is None and from_store:
                    # "linked from store" carries no verify-download message of its own; the
                    # store's sidecar already has the completeness verdict from when the
                    # dataset was originally downloaded.
                    complete = self._sidecar_completeness(store, accession)
                record_download(
                    reg,
                    accession,
                    "downloaded" if success else "failed",
                    fastq_dir,
                    message,
                    attempt=not linked,
                    complete=complete,
                    source="store" if from_store else None,
                    store_name=accession if from_store else None,
                )
                if from_store:
                    self._mark_linked(reg, accession)
                    stage = "linked" if linked else "downloaded"
                    record_usage_safe(store, reg, accession, "", stage, detail=message)

        return _record_result

    @staticmethod
    def _sidecar_completeness(store: Optional[StorePaths], accession: str) -> Optional[dict]:
        """The completeness verdict recorded in the store's sidecar for ``accession``, or None.

        None when there is no store, or no sidecar (not yet catalogued, or unreadable);
        ``read_sidecar`` already logs a warning for the latter case.
        """
        if store is None:
            return None
        sidecar = read_sidecar(sidecar_path(store, accession))
        if sidecar is None:
            return None
        return {
            "verdict": sidecar.completeness.get("verdict"),
            "ratio": sidecar.completeness.get("ratio"),
            "expected_spots": sidecar.ncbi.get("spots"),
            "reads_r1": sidecar.reads_per_mate,
        }

    @staticmethod
    def _mark_linked(reg: Registry, accession: str) -> None:
        """Add ``accession`` to the registry's list of datasets this project links from the store."""
        linked = set(reg.store.get("linked") or [])
        linked.add(accession)
        reg.store["linked"] = sorted(linked)

    def _transient_folders(self, args: argparse.Namespace, fastq_dir: Path, store: Optional[StorePaths]) -> Set[Path]:
        """Folders where ``download_accession`` can leave ``.sra-cache`` archives or
        ``<ACC>_temp`` build directories behind: the FASTQ output folder (always, since
        that is where ``<ACC>_temp`` lands and, without a store, ``.sra-cache`` too), any
        explicit ``--temp-folder`` or ``--sra-cache``, and, with a shared store, the
        store's own ``tmp`` folder (the default home for ``.sra-cache`` when downloading
        through a store)."""
        folders = {fastq_dir}
        temp_folder = getattr(args, "temp_folder", None)
        if temp_folder:
            folders.add(Path(temp_folder))
        sra_cache = getattr(args, "sra_cache", None)
        if sra_cache:
            folders.add(Path(sra_cache).parent)
        if store is not None:
            folders.add(store.tmp)
        return folders

    def _warn_if_transient_bytes_large(
        self, args: argparse.Namespace, fastq_dir: Path, store: Optional[StorePaths]
    ) -> None:
        """Warn, naming each folder and its size, when kept transient artifacts add up."""
        sized = [(folder, transient_bytes(folder)) for folder in self._transient_folders(args, fastq_dir, store)]
        sized = [(folder, size) for folder, size in sized if size > 0]
        total = sum(size for _, size in sized)
        if total <= TRANSIENT_BYTES_WARN_THRESHOLD:
            return
        detail = ", ".join(f"{folder} ({size} bytes)" for folder, size in sorted(sized, key=lambda item: str(item[0])))
        self.logger.warning(
            "Kept .sra-cache archives and <ACC>_temp build folders total %d bytes, over the "
            "%d byte warning threshold: %s",
            total,
            TRANSIENT_BYTES_WARN_THRESHOLD,
            detail,
        )

    def _store_options(self, args: argparse.Namespace, store: Optional[StorePaths], registry: Registry) -> dict:
        """The store-related keyword arguments for ``download_sra``, empty without a store.

        NCBI's spot count for an accession is read from the project's own metadata folder
        when it has one, since that is where ``download_metadata`` writes; the store's
        metadata folder is the fallback.
        """
        if store is None:
            return {}
        return {
            "store": store,
            "link_mode": getattr(args, "link_mode", "auto"),
            "accept_partial": getattr(args, "accept_partial", False),
            "resume_partial": getattr(args, "resume_partial", True),
            "store_metadata": [project_root(registry) / "metadata", store.metadata],
        }

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if not args.dry_run and shutil.which("fasterq-dump") is None:
                self.logger.error(
                    "fasterq-dump not found on PATH. Install sra-tools, "
                    "for example: conda install -c bioconda sra-tools"
                )
                return 1

            verify_downloads = getattr(args, "verify_downloads", True)
            redownload_truncated = getattr(args, "redownload_truncated", False)
            sra_cache = getattr(args, "sra_cache", None)
            use_prefetch = getattr(args, "use_prefetch", True)
            keep_sra = getattr(args, "keep_sra", False)
            compress = getattr(args, "compress", True)
            max_workers = self._resolve_max_workers(args)

            excluded: set = set()
            expected_spots: dict = {}
            truncated: set = set()
            on_result = None
            fastq_dir = Path(args.fastq_folder)

            project_registry = load_registry(args.registry)
            store = self._resolve_store(args, project_registry)

            if not args.dry_run:
                excluded = set(query(project_registry, "excluded"))

                if verify_downloads:
                    for acc, record in project_registry.datasets.items():
                        spots = (record.get("metadata") or {}).get("run_total_spots")
                        if spots is not None:
                            expected_spots[acc] = spots

                if redownload_truncated:
                    truncated = {
                        acc
                        for acc, record in project_registry.datasets.items()
                        if (record.get("download") or {}).get("complete", {}).get("verdict") == "truncated"
                    }

                on_result = self._result_recorder(args, fastq_dir, store)

            download_stats = download_sra(
                fastq_folder=args.fastq_folder,
                accessions_file=args.accessions_file,
                max_downloads=args.max_downloads,
                dry_run=args.dry_run,
                num_threads=args.num_threads,
                max_workers=max_workers,
                force=args.force,
                max_retries=args.max_retries,
                temp_folder=args.temp_folder,
                blacklist=args.blacklist,
                blacklist_accessions=excluded,
                on_result=on_result,
                expected_spots=expected_spots if verify_downloads else None,
                redownload_truncated=redownload_truncated,
                truncated_accessions=truncated,
                sra_cache=sra_cache,
                use_prefetch=use_prefetch,
                keep_sra=keep_sra,
                compress=compress,
                **self._store_options(args, store, project_registry),
            )

            if args.dry_run:
                self._log_dry_run_summary(args, download_stats)
            else:
                self._log_download_summary(download_stats)
                self._record_run_outcomes(args, download_stats, fastq_dir, store)
                self._warn_if_transient_bytes_large(args, fastq_dir, store)

                if args.report_file:
                    self._write_report(args.report_file, download_stats)
                    self.logger.info("Download report written to %s", args.report_file)

            if not args.dry_run and download_stats.get("aborted"):
                self.logger.error(
                    "Download run aborted (%s); accessions attempted before the abort were " "still recorded above",
                    download_stats["aborted"],
                )
                return 1

            if not args.dry_run and download_stats["failed"] > 0:
                self._report_failed_downloads(args, download_stats)
                return 1

            return 0

        except MetaQuestError as e:
            self.logger.error(f"Error downloading SRA data: {e}")
            return 1
