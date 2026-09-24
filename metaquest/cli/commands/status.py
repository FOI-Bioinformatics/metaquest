"""Local inventory / status CLI command.

Reports what MetaQuest has already downloaded locally (SRA reads, NCBI metadata,
genome assemblies) so a user can see what is available without re-downloading,
and reports where every accession sits in the registry (screened, selected,
excluded, downloaded, analysed, extracted, assembled). When no registry file
exists yet, the report is reconstructed in memory from what is on disk.
"""

import argparse
import json
import logging
import math
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metaquest.cli.base import BaseCommand
from metaquest.core.constants import DEFAULT_CONTAINMENT_THRESHOLD, DEFAULT_PARSED_CONTAINMENT_FILE, GENOME_FASTA_GLOBS
from metaquest.core.exceptions import DataAccessError, MetaQuestError
from metaquest.data.file_io import visible_files, write_csv
from metaquest.data.registry import (
    ProjectPaths,
    ReconcileReport,
    Registry,
    STAGES,
    bootstrap_from_disk,
    empty_assembly_dirs,
    extraction_record,
    known_genome_ids,
    load_registry,
    query,
    reconcile,
    registry_path,
    resolve_project_path,
    save_registry,
    stage_counts,
    to_dataframes,
)
from metaquest.data.sra import STORE_READY_STATES, accession_has_fastq, is_transient_folder
from metaquest.store.catalog import Catalog
from metaquest.store.layout import StorePaths, sidecar_path, store_paths
from metaquest.store.link import is_store_link
from metaquest.store.resolve import resolve_store_root
from metaquest.store.sidecar import read_sidecar

logger = logging.getLogger(__name__)


def _as_float(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None when it is not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_positive_int(value: Any) -> Optional[int]:
    """``value`` as a positive int, or None when it is not one."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _as_non_negative_int(value: Any) -> Optional[int]:
    """``value`` as an int of zero or more, or None when it is not one."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _warn_malformed(name: str, value: Any, reason: str) -> None:
    """Warn that a recorded criteria value could not be used to build a reselect command."""
    logger.warning("Recorded %s %r %s", name, value, reason)


def _run_filter_flags(criteria: Dict[str, Any]) -> str:
    """The --max-run-size, --min-spots, --max-spots and --platform flags a selection recorded.

    The run size is a byte count above zero and the spot bounds are zero or more; a recorded value
    that fails that check is left out with a warning rather than pasted into the command. The
    platform is shell-quoted.
    """
    part = ""
    numeric = [
        ("max_run_size", "--max-run-size", _as_positive_int),
        ("min_spots", "--min-spots", _as_non_negative_int),
        ("max_spots", "--max-spots", _as_non_negative_int),
    ]
    for key, flag, coerce in numeric:
        raw = criteria.get(key)
        if raw is None:
            continue
        number = coerce(raw)
        if number is None:
            _warn_malformed(key, raw, f"is not a valid count; the suggested command omits {flag}")
        else:
            part += f" {flag} {number}"
    platform = criteria.get("platform")
    if platform is not None:
        part += f" --platform {shlex.quote(str(platform))}"
    return part


def _format_bytes(count: int) -> str:
    """A byte count in decimal units, as the run size flags read them (e.g. 600 MB, 1.5 GB)."""
    for factor, unit in ((10**12, "TB"), (10**9, "GB"), (10**6, "MB"), (10**3, "KB")):
        if count >= factor:
            return f"{count / factor:g} {unit}"
    return f"{count} bytes"


def _run_filter_detail(criteria: Dict[str, Any]) -> List[str]:
    """The run filters a selection recorded, for the selected stage row.

    A malformed value is left out; ``status --next`` warns about it when it builds the
    reselect command.
    """
    parts = []
    size = _as_positive_int(criteria.get("max_run_size"))
    if size is not None:
        parts.append(f"max size {_format_bytes(size)}")
    for key, symbol in (("min_spots", ">="), ("max_spots", "<=")):
        spots = _as_non_negative_int(criteria.get(key))
        if spots is not None:
            parts.append(f"spots {symbol} {spots}")
    if criteria.get("platform") is not None:
        parts.append(f"platform {criteria['platform']}")
    return parts


class StatusCommand(BaseCommand):
    """Command to report locally available data and the registry's per-accession stages."""

    @property
    def name(self) -> str:
        return "status"

    @property
    def help(self) -> str:
        return "Report which SRA reads, metadata, and genomes are already available locally"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--fastq-folder", default="fastq", help="Folder holding per-accession FASTQ downloads")
        parser.add_argument("--metadata-folder", default="metadata", help="Folder holding NCBI metadata XML")
        parser.add_argument("--genomes-folder", default="genomes", help="Folder holding genome FASTA files")
        parser.add_argument(
            "--targeted-folder", default="targeted", help="Root folder of extracted reads and assemblies"
        )
        parser.add_argument("--matches-folder", default="matches", help="Folder of Branchwater match CSVs")
        parser.add_argument(
            "--accessions-file",
            help="Optional file of SRA accessions (one per line) to reconcile against local FASTQ/metadata",
        )
        parser.add_argument(
            "--parsed-containment",
            help="Optional parsed containment table; its sample accessions are the wanted list",
        )
        parser.add_argument(
            "--registry",
            default=None,
            help="Registry file (default: metaquest_registry.json found upwards from here)",
        )
        parser.add_argument("--data-root", default=None, help="Shared data store root (overrides discovery)")
        parser.add_argument("--stage", choices=list(STAGES), default=None, help="List the accessions in one stage")
        parser.add_argument(
            "--genome",
            action="append",
            default=None,
            help="Restrict extraction and assembly stages to a genome id (repeatable)",
        )
        parser.add_argument("--init", action="store_true", help="Create the registry from what is on disk")
        parser.add_argument(
            "--reconcile",
            action="store_true",
            help="Compare the registry with the disk and record missing downloads",
        )
        parser.add_argument(
            "--export-tsv", default=None, help="Write <PREFIX>_datasets.tsv and <PREFIX>_extractions.tsv"
        )
        parser.add_argument("--next", action="store_true", help="Suggest the commands that advance the most accessions")
        parser.add_argument("--list-missing", action="store_true", help="Also print the accessions that are missing")
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON")

    # ------------------------------------------------------------------ wanted

    def _wanted_accessions(self, args: argparse.Namespace) -> List[str]:
        """Load the wanted accession list from a file and/or a parsed containment table."""
        wanted: List[str] = []
        if args.accessions_file:
            path = Path(args.accessions_file)
            if not path.exists():
                raise MetaQuestError(f"Accessions file not found: {path}")
            wanted += [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        if args.parsed_containment:
            import pandas as pd

            path = Path(args.parsed_containment)
            if not path.exists():
                raise MetaQuestError(f"Parsed containment file not found: {path}")
            wanted += [str(i) for i in pd.read_csv(path, sep="\t", index_col=0).index]
        # de-duplicate, preserve order
        return list(dict.fromkeys(wanted))

    def _resolve_wanted(self, args: argparse.Namespace, registry: Registry) -> List[str]:
        """Explicit accessions/containment file win; otherwise fall back to the registry's selection."""
        if args.accessions_file or args.parsed_containment:
            return self._wanted_accessions(args)
        return query(registry, "selected")

    @staticmethod
    def _reconcile_present_missing(wanted: List[str], present_fn) -> Tuple[List[str], List[str]]:
        """Split a wanted list into (present, missing) using a predicate."""
        present = [a for a in wanted if present_fn(a)]
        missing = [a for a in wanted if a not in present]
        return present, missing

    # -------------------------------------------------------------- reporting

    def _inventory_report(
        self, args: argparse.Namespace, registry: Registry, store: Optional[StorePaths] = None
    ) -> Dict[str, Any]:
        fastq_dir = Path(args.fastq_folder)
        meta_dir = Path(args.metadata_folder)
        genomes_dir = Path(args.genomes_folder)

        on_disk_fastq = sorted(
            d.name
            for d in visible_files(fastq_dir, dirs=True)
            if not is_transient_folder(d.name) and accession_has_fastq(d)
        )
        on_disk_meta = sorted(p.name[: -len("_metadata.xml")] for p in visible_files(meta_dir, "*_metadata.xml"))
        on_disk_genomes = sorted(p.name for p in visible_files(genomes_dir, *GENOME_FASTA_GLOBS))

        report: Dict[str, Any] = {
            "on_disk": {
                "fastq_accessions": len(on_disk_fastq),
                "metadata_xml": len(on_disk_meta),
                "genome_fasta": len(on_disk_genomes),
            }
        }

        wanted = self._resolve_wanted(args, registry)
        if wanted:
            fastq_present, fastq_missing = self._reconcile_present_missing(
                wanted, lambda a: accession_has_fastq(fastq_dir / a)
            )
            meta_present, meta_missing = self._reconcile_present_missing(
                wanted, lambda a: (meta_dir / f"{a}_metadata.xml").exists()
            )
            report["wanted"] = {
                "total": len(wanted),
                "fastq_present": len(fastq_present),
                "fastq_missing": fastq_missing,
                "fastq_incomplete_store_links": self._incomplete_store_links(fastq_dir, fastq_missing, store),
                "metadata_present": len(meta_present),
                "metadata_missing": meta_missing,
            }
        return report

    @staticmethod
    def _incomplete_store_links(fastq_dir: Path, missing: List[str], store: Optional[StorePaths]) -> List[str]:
        """Missing accessions whose ``fastq/<ACC>`` is a symlink into the store, but the store's
        recorded sidecar state for that dataset falls outside ``STORE_READY_STATES``.

        Such a link is not simply absent: a download into the store was attempted (and left a
        ``failed`` or ``partial`` sidecar, or one is still ``downloading``), so the report says
        why the accession reads as missing rather than leaving that to be rediscovered by hand.
        """
        if store is None:
            return []
        incomplete = []
        for acc in missing:
            if not is_store_link(fastq_dir / acc, store):
                continue
            sidecar = read_sidecar(sidecar_path(store, acc))
            if sidecar is not None and sidecar.state not in STORE_READY_STATES:
                incomplete.append(acc)
        return sorted(incomplete)

    @staticmethod
    def _stage_filter_accessions(registry: Registry, stage: str, genomes: Optional[List[str]]) -> List[str]:
        if not genomes:
            return query(registry, stage)
        seen: List[str] = []
        for genome_id in genomes:
            for acc in query(registry, stage, genome_id):
                if acc not in seen:
                    seen.append(acc)
        return seen

    def _genome_report(
        self,
        registry: Registry,
        paths: ProjectPaths,
        genome_filter: Optional[List[str]],
        counts: Dict[str, Any],
    ) -> Dict[str, Any]:
        genome_ids = sorted(known_genome_ids(registry))
        if genome_filter:
            wanted = set(genome_filter)
            genome_ids = [g for g in genome_ids if g in wanted]

        empty_by_genome: Dict[str, List[str]] = {}
        for acc, genome_id in empty_assembly_dirs(paths.targeted, genome_ids):
            empty_by_genome.setdefault(genome_id, []).append(acc)

        report: Dict[str, Any] = {}
        for genome_id in genome_ids:
            info = counts["genomes"].get(genome_id, {"extracted": 0, "assembled": 0, "zero_mapped": []})
            report[genome_id] = {
                "extracted": info.get("extracted", 0),
                "assembled": info.get("assembled", 0),
                "zero_mapped": info.get("zero_mapped", []),
                "empty_assembly_dirs": sorted(empty_by_genome.get(genome_id, [])),
            }
        return report

    @staticmethod
    def _drift_report(drift: ReconcileReport) -> Dict[str, Any]:
        return {
            "recorded_missing": list(drift.recorded_missing),
            "untracked_fastq": list(drift.untracked_fastq),
            "untracked_extractions": [[acc, genome_id] for acc, genome_id in drift.untracked_extractions],
            "empty_assembly_dirs": [[acc, genome_id] for acc, genome_id in drift.empty_assembly_dirs],
            "dangling_links": list(drift.dangling_links),
        }

    @staticmethod
    def _reselect_command(criteria: Dict[str, Any], output: str) -> str:
        """A runnable ``select_datasets`` command that redoes a selection with ``--skip-excluded``.

        Built from the criteria the original ``--no-skip-excluded`` run recorded, targeting the same
        ``--output`` so rerunning it corrects that selection's file in place. Reproduces every
        criterion ``record_selection`` stores that changes which accessions are chosen (metadata
        filter, top-N cap, run size, spot and platform filters, source table), not just the genome
        column and threshold, so the suggested command redoes the same selection rather than a
        looser one. Every value that came from the registry rather than this method's own literal
        flag text is passed through ``shlex.quote``, so a value containing a space or shell
        metacharacter (a metadata value like "New York", say) still produces a command that is safe
        to paste into a shell and run as-is. The threshold is coerced with ``float``, the top-N
        count with ``int`` (positive only) and ``require`` must be ``any`` or ``all``; a recorded
        value that fails that check (a hand-edited registry, say) leaves its flag out rather than
        being pasted into the command.
        """
        raw_threshold = criteria.get("threshold", DEFAULT_CONTAINMENT_THRESHOLD)
        threshold = _as_float(raw_threshold)
        if threshold is None and raw_threshold is not None:
            _warn_malformed("threshold", raw_threshold, "is not a number; the suggested command uses the default")
        threshold_part = f" --threshold {threshold}" if threshold is not None else ""
        genome_ids = criteria.get("genome_ids")
        if genome_ids:
            raw_require = criteria.get("require", "any")
            quoted_ids = " ".join(shlex.quote(str(g)) for g in genome_ids)
            genome_part = f"--genome-ids {quoted_ids}"
            if raw_require in ("any", "all"):
                genome_part += f" --require {raw_require}"
            elif raw_require is not None:
                _warn_malformed("require", raw_require, "is not 'any' or 'all'; the suggested command omits --require")
        else:
            column = criteria.get("column") or "max_containment"
            genome_part = f"--genome-id {shlex.quote(str(column))}"
        command = (
            f"metaquest select_datasets {genome_part}{threshold_part} "
            f"--skip-excluded --output {shlex.quote(str(output))}"
        )

        metadata_file = criteria.get("metadata_file")
        if metadata_file:
            command += f" --metadata-file {shlex.quote(str(metadata_file))}"
        metadata_column = criteria.get("metadata_column")
        metadata_value = criteria.get("metadata_value")
        if metadata_column and metadata_value is not None:
            command += (
                f" --metadata-column {shlex.quote(str(metadata_column))}"
                f" --metadata-value {shlex.quote(str(metadata_value))}"
            )
        raw_top_n = criteria.get("top_n")
        top_n = _as_positive_int(raw_top_n)
        if top_n:
            command += f" --top-n {top_n}"
        elif raw_top_n is not None:
            _warn_malformed("top_n", raw_top_n, "is not a number; the suggested command uses the default")
        command += _run_filter_flags(criteria)
        table = criteria.get("table")
        if table and str(table) != DEFAULT_PARSED_CONTAINMENT_FILE:
            command += f" --parsed-containment {shlex.quote(str(table))}"
        return command

    @staticmethod
    def _download_next_steps(registry: Registry) -> List[Dict[str, Any]]:
        to_download = [
            acc
            for acc, record in registry.datasets.items()
            if record.get("selection", {}).get("selected")
            and not record.get("exclusion", {}).get("excluded")
            and record.get("download", {}).get("state") != "downloaded"
        ]
        if not to_download:
            return []
        # A selection recorded with --no-skip-excluded may still list an excluded
        # accession, so its output file is never suggested for direct download;
        # such accessions instead point at re-running select_datasets with
        # --skip-excluded so the excluded run is dropped before download.
        by_output: Dict[str, List[str]] = {}
        reselect_groups: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
        for acc in to_download:
            selection = registry.datasets[acc].get("selection", {})
            criteria = selection.get("criteria") or {}
            output = selection.get("output") or "accessions.txt"
            if criteria.get("skip_excluded") is False:
                group = reselect_groups.setdefault(output, (criteria, []))
                group[1].append(acc)
                continue
            by_output.setdefault(output, []).append(acc)
        steps = [
            {"command": f"metaquest download_sra --accessions-file {output}", "accessions": accs}
            for output, accs in by_output.items()
        ]
        for output, (criteria, accs) in reselect_groups.items():
            steps.append({"command": StatusCommand._reselect_command(criteria, output), "accessions": accs})
        return steps

    @staticmethod
    def _selection_table(registry: Registry) -> str:
        """The containment table a selection was recorded from, else the default name."""
        for record in registry.datasets.values():
            selection = record.get("selection") or {}
            table = (selection.get("criteria") or {}).get("table") if selection.get("selected") else None
            if table:
                return str(table)
        return DEFAULT_PARSED_CONTAINMENT_FILE

    @staticmethod
    def _genome_fasta(registry: Registry, paths: ProjectPaths, genome_id: str) -> Path:
        """The genome's FASTA: the recorded one, else a file on disk, else the conventional name."""
        recorded = (registry.genomes.get(genome_id) or {}).get("fasta")
        if recorded:
            return resolve_project_path(registry, recorded)
        for pattern in GENOME_FASTA_GLOBS:
            candidate = paths.genomes / pattern.replace("*", genome_id)
            if candidate.exists():
                return candidate
        return paths.genomes / f"{genome_id}.fna"

    @staticmethod
    def _extraction_next_steps(registry: Registry, paths: ProjectPaths) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        excluded = set(query(registry, "excluded"))
        downloaded = [acc for acc in query(registry, "downloaded") if acc not in excluded]
        table = StatusCommand._selection_table(registry)
        for genome_id in sorted(known_genome_ids(registry)):
            genome_fasta = StatusCommand._genome_fasta(registry, paths, genome_id)
            base = (
                f"metaquest extract_target_reads --parsed-containment {table} "
                f"--genome-id {genome_id} --genome-fasta {genome_fasta}"
            )
            # Any record, including a zero-mapped one, means the sample has been tried.
            recorded = {acc for acc in registry.datasets if extraction_record(registry, acc, genome_id) is not None}
            to_extract = [acc for acc in downloaded if acc not in recorded]
            if to_extract:
                steps.append({"command": base, "accessions": to_extract})
            assembled = set(query(registry, "assembled", genome_id))
            to_assemble = [
                acc for acc in query(registry, "extracted", genome_id) if acc not in assembled and acc not in excluded
            ]
            if to_assemble:
                steps.append({"command": f"{base} --assemble", "accessions": to_assemble})
        return steps

    def _next_steps(self, registry: Registry, paths: ProjectPaths) -> List[Dict[str, Any]]:
        return self._download_next_steps(registry) + self._extraction_next_steps(registry, paths)

    def _export_tsv(self, registry: Registry, prefix: str) -> None:
        datasets_df, extractions_df = to_dataframes(registry)
        datasets_path = f"{prefix}_datasets.tsv"
        extractions_path = f"{prefix}_extractions.tsv"
        write_csv(datasets_df, datasets_path, sep="\t")
        write_csv(extractions_df, extractions_path, sep="\t", index=False)
        self.logger.info("Wrote %s and %s", datasets_path, extractions_path)

    # -------------------------------------------------------------------- store

    @staticmethod
    def _store_report(root: Path) -> Dict[str, Any]:
        """Dataset counts by state from the shared data store's catalogue at ``root``."""
        paths = store_paths(root)
        with Catalog(paths) as catalog:
            rows = catalog.conn.execute(
                "SELECT state, COUNT(*) AS n FROM datasets WHERE state IS NOT 'unknown' GROUP BY state"
            ).fetchall()
        return {"root": str(root), "available": True, "datasets": {row["state"]: row["n"] for row in rows}}

    def _resolve_store_root(self, args, registry) -> Tuple[Optional[Path], bool]:
        """``(root, available)``: the store this project points at, and whether it can be read.

        A status report is about the project, so an unmounted volume or a root that has moved
        must not stop it: the block still names the root, marked unavailable, and everything
        else in the report (dangling links above all, which is exactly what a missing store
        produces) is reported as usual.
        """
        try:
            return resolve_store_root(args.data_root, registry.store.get("root")), True
        except DataAccessError as e:
            self.logger.warning("store unavailable: %s; continuing without it", e)
            return resolve_store_root(args.data_root, registry.store.get("root"), require_marker=False), False

    def _store_block(self, root: Path, available: bool) -> Dict[str, Any]:
        """The report's store block: dataset counts when readable, else root and a flag."""
        if not available:
            return {"root": str(root), "available": False, "datasets": {}}
        try:
            return self._store_report(root)
        except DataAccessError as e:
            self.logger.warning("store unavailable: %s; continuing without it", e)
            return {"root": str(root), "available": False, "datasets": {}}

    # ---------------------------------------------------------------- printing

    @staticmethod
    def _print_store(store: Dict[str, Any]) -> None:
        print("\nStore")
        print("=====")
        print(f"  Root : {store['root']}")
        if not store.get("available", True):
            print("  Unavailable: the store could not be read from here")
            return
        for state, count in sorted(store["datasets"].items()):
            print(f"  {state:<10s}: {count}")

    @staticmethod
    def _print_inventory(report: Dict[str, Any], list_missing: bool) -> None:
        od = report["on_disk"]
        print("Local inventory")
        print("===============")
        print(f"  FASTQ accessions on disk : {od['fastq_accessions']}")
        print(f"  Metadata XML on disk     : {od['metadata_xml']}")
        print(f"  Genome FASTA on disk     : {od['genome_fasta']}")

        w = report.get("wanted")
        if w:
            incomplete_links = w.get("fastq_incomplete_store_links") or []
            print(f"\nReconciled against {w['total']} wanted accession(s)")
            fastq_line = f"  FASTQ    : {w['fastq_present']} present, {len(w['fastq_missing'])} missing"
            if incomplete_links:
                fastq_line += f", {len(incomplete_links)} linked to a store dataset that is not complete"
            print(fastq_line)
            print(f"  Metadata : {w['metadata_present']} present, {len(w['metadata_missing'])} missing")
            if list_missing:
                if w["fastq_missing"]:
                    print("  Missing FASTQ    : " + ", ".join(w["fastq_missing"]))
                if incomplete_links:
                    print("  Incomplete store links : " + ", ".join(incomplete_links))
                if w["metadata_missing"]:
                    print("  Missing metadata : " + ", ".join(w["metadata_missing"]))

    @staticmethod
    def _selection_detail(registry: Registry) -> str:
        """The criteria and date of the most recent selection, for the selected stage row."""
        latest: Dict[str, Any] = {}
        for record in registry.datasets.values():
            selection = record.get("selection") or {}
            if selection.get("selected") and str(selection.get("date", "")) >= str(latest.get("date", "")):
                latest = selection
        if not latest:
            return ""
        criteria = latest.get("criteria") or {}
        parts = []
        if criteria.get("column"):
            parts.append(f"column {criteria['column']}")
        if criteria.get("threshold") is not None:
            parts.append(f"threshold {criteria['threshold']}")
        if criteria.get("metadata_column"):
            parts.append(f"{criteria['metadata_column']} = {criteria.get('metadata_value')}")
        parts.extend(_run_filter_detail(criteria))
        parts.append(str(latest.get("date", "")))
        return ", ".join(p for p in parts if p)

    @staticmethod
    def _exclusion_detail(registry: Registry) -> str:
        """How many accessions carry each exclusion reason."""
        reasons: Dict[str, int] = {}
        for record in registry.datasets.values():
            exclusion = record.get("exclusion") or {}
            if exclusion.get("excluded"):
                reason = str(exclusion.get("reason") or "no reason given")
                reasons[reason] = reasons.get(reason, 0) + 1
        return ", ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))

    @staticmethod
    def _download_verdicts(registry: Registry) -> Dict[str, List[str]]:
        """Accessions whose recorded completeness verdict is "truncated" or "unverified"."""
        truncated = []
        unverified = []
        for acc, record in registry.datasets.items():
            verdict = (record.get("download") or {}).get("complete", {}).get("verdict")
            if verdict == "truncated":
                truncated.append(acc)
            elif verdict == "unverified":
                unverified.append(acc)
        return {"truncated": sorted(truncated), "unverified": sorted(unverified)}

    @staticmethod
    def _print_stages(stages: Dict[str, Any], registry: Registry) -> None:
        print("\nStages")
        print("======")
        details = {
            "selected": StatusCommand._selection_detail(registry),
            "excluded": StatusCommand._exclusion_detail(registry),
        }
        for stage in STAGES:
            info = stages[stage]
            detail = details.get(stage)
            print(f"  {stage:<10s} : {info['count']}" + (f"   {detail}" if detail else ""))
        truncated = StatusCommand._download_verdicts(registry)["truncated"]
        if truncated:
            print(f"  truncated downloads: {len(truncated)} (" + ", ".join(truncated) + ")")

    @staticmethod
    def _print_genomes(genomes: Dict[str, Any]) -> None:
        if not genomes:
            return
        print("\nGenomes")
        print("=======")
        for genome_id, info in genomes.items():
            zero_mapped = len(info["zero_mapped"])
            empty_dirs = len(info["empty_assembly_dirs"])
            print(
                f"  {genome_id}: extracted {info['extracted']} ({zero_mapped} with 0 mapped reads), "
                f"assembled {info['assembled']}, {empty_dirs} empty assembly dir(s)"
            )

    @staticmethod
    def _print_stage_filter(registry: Registry, stage: Optional[str], genomes: Optional[List[str]]) -> None:
        if not stage:
            return
        accs = StatusCommand._stage_filter_accessions(registry, stage, genomes)
        detail = f" (genome {', '.join(genomes)})" if genomes else ""
        print(f"\nStage '{stage}'{detail}: " + (", ".join(accs) if accs else "(none)"))

    @staticmethod
    def _print_gaps(registry: Registry) -> None:
        selected = set(query(registry, "selected"))
        excluded = set(query(registry, "excluded"))
        downloaded_set = set(query(registry, "downloaded"))
        not_downloaded = sorted(selected - excluded - downloaded_set)
        print("\nGaps")
        print("====")
        print("  Selected but not downloaded : " + (", ".join(not_downloaded) if not_downloaded else "(none)"))
        for genome_id in sorted(known_genome_ids(registry)):
            gap = sorted(downloaded_set - set(query(registry, "extracted", genome_id)))
            if gap:
                print(f"  Downloaded but not extracted for {genome_id} : " + ", ".join(gap))

    @staticmethod
    def _print_drift(drift: Dict[str, Any]) -> None:
        if not drift:
            return
        print("\nDrift against disk")
        print("===================")
        print(
            "  Recorded downloaded but missing on disk : "
            + (", ".join(drift["recorded_missing"]) if drift["recorded_missing"] else "(none)")
        )
        print(
            "  On disk but not tracked as downloaded   : "
            + (", ".join(drift["untracked_fastq"]) if drift["untracked_fastq"] else "(none)")
        )
        if drift["untracked_extractions"]:
            pairs = ", ".join(f"{acc}/{genome_id}" for acc, genome_id in drift["untracked_extractions"])
            print(f"  Untracked extractions                   : {pairs}")
        if drift["empty_assembly_dirs"]:
            pairs = ", ".join(f"{acc}/{genome_id}" for acc, genome_id in drift["empty_assembly_dirs"])
            print(f"  Empty assembly directories               : {pairs}")
        if drift.get("dangling_links"):
            print("  Store links with a missing target       : " + ", ".join(drift["dangling_links"]))

    @staticmethod
    def _print_next(steps: List[Dict[str, Any]]) -> None:
        if not steps:
            return
        print("\nSuggested next steps")
        print("=====================")
        for step in steps:
            print(f"  {step['command']}")
            print("    accessions: " + ", ".join(step["accessions"]))

    def _print_report(self, args: argparse.Namespace, report: Dict[str, Any], registry: Registry) -> None:
        self._print_inventory(report, args.list_missing)
        if report.get("store"):
            self._print_store(report["store"])
        self._print_stages(report["stages"], registry)
        self._print_genomes(report["genomes"])
        self._print_stage_filter(registry, args.stage, args.genome)
        if args.list_missing:
            self._print_gaps(registry)
        self._print_drift(report["drift"])
        if args.next:
            self._print_next(report.get("next", []))

    def _emit(self, args: argparse.Namespace, report: Dict[str, Any], registry: Registry) -> None:
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            self._print_report(args, report, registry)

    # ------------------------------------------------------------------ execute

    def execute(self, args: argparse.Namespace) -> int:
        try:
            paths = ProjectPaths(
                Path(args.fastq_folder),
                Path(args.metadata_folder),
                Path(args.genomes_folder),
                Path(args.targeted_folder),
                Path(args.matches_folder),
            )
            registry_file = registry_path(args.registry)
            existed = registry_file.exists()
            if existed and args.init:
                self.logger.error(
                    "Registry already exists at %s; run status --reconcile to update it from disk, "
                    "or remove the file to rebuild it",
                    registry_file,
                )
                return 1
            if args.reconcile and not existed:
                self.logger.error(
                    "No registry at %s to reconcile; create one first with: metaquest status --init",
                    registry_file,
                )
                return 1
            if existed:
                registry = load_registry(registry_file)
            else:
                registry = bootstrap_from_disk(paths, args.accessions_file, args.parsed_containment, registry_file)
                if args.init:
                    save_registry(registry)
                    self.logger.info("Registry written to %s", registry_file)

            drift = None
            if args.reconcile:
                drift = reconcile(registry, paths)
                save_registry(registry)

            store_root, store_available = self._resolve_store_root(args, registry)
            store: Optional[StorePaths] = None
            if store_root is not None and store_available:
                store = store_paths(store_root)
                self.logger.info("Using shared data store at %s", store_root)

            report = self._inventory_report(args, registry, store)
            report["registry"] = {
                "path": str(registry_file),
                "version": registry.version,
                "exists": existed or args.init,
                "updated": registry.updated,
            }
            if store_root is not None:
                report["store"] = self._store_block(store_root, store_available)
            counts = stage_counts(registry)
            report["stages"] = {s: {"count": counts["stages"][s], "accessions": query(registry, s)} for s in STAGES}
            report["downloads"] = self._download_verdicts(registry)
            report["genomes"] = self._genome_report(registry, paths, args.genome, counts)
            report["drift"] = self._drift_report(drift) if drift else {}
            if args.next:
                report["next"] = self._next_steps(registry, paths)
            if args.export_tsv:
                self._export_tsv(registry, args.export_tsv)

            if not existed and not args.init:
                self.logger.info(
                    "No registry yet; the stages above were reconstructed from disk. Run: metaquest status --init"
                )

            self._emit(args, report, registry)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error building status report: %s", e)
            return 1
