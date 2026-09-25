"""
Text rendering of the `status` report.

Each `_print_*` function writes one section of the report through the `emit` callable it is
given (the command's `BaseCommand.emit`), so stdout stays the command's single channel.
"""

import argparse
from typing import Any, Callable, Dict, List, Optional

from metaquest.cli.commands.status.suggest import _as_non_negative_int, _as_positive_int
from metaquest.data import registry_blocks as rb
from metaquest.data.registry import Registry, STAGES, known_genome_ids, query
from metaquest.processing.status_report import _download_verdicts, _stage_filter_accessions


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


def _print_store(store: Dict[str, Any], emit: Callable[[str], None]) -> None:
    emit("\nStore")
    emit("=====")
    emit(f"  Root : {store['root']}")
    if not store.get("available", True):
        emit("  Unavailable: the store could not be read from here")
        return
    for state, count in sorted(store["datasets"].items()):
        emit(f"  {state:<10s}: {count}")


def _print_inventory(report: Dict[str, Any], list_missing: bool, emit: Callable[[str], None]) -> None:
    od = report["on_disk"]
    emit("Local inventory")
    emit("===============")
    emit(f"  FASTQ accessions on disk : {od['fastq_accessions']}")
    emit(f"  Metadata XML on disk     : {od['metadata_xml']}")
    emit(f"  Genome FASTA on disk     : {od['genome_fasta']}")

    w = report.get("wanted")
    if w:
        incomplete_links = w.get("fastq_incomplete_store_links") or []
        emit(f"\nReconciled against {w['total']} wanted accession(s)")
        fastq_line = f"  FASTQ    : {w['fastq_present']} present, {len(w['fastq_missing'])} missing"
        if incomplete_links:
            fastq_line += f", {len(incomplete_links)} linked to a store dataset that is not complete"
        emit(fastq_line)
        emit(f"  Metadata : {w['metadata_present']} present, {len(w['metadata_missing'])} missing")
        if list_missing:
            if w["fastq_missing"]:
                emit("  Missing FASTQ    : " + ", ".join(w["fastq_missing"]))
            if incomplete_links:
                emit("  Incomplete store links : " + ", ".join(incomplete_links))
            if w["metadata_missing"]:
                emit("  Missing metadata : " + ", ".join(w["metadata_missing"]))


def _selection_detail(registry: Registry) -> str:
    """The criteria and date of the most recent selection, for the selected stage row."""
    latest: Optional[rb.SelectionBlock] = None
    for acc in registry.datasets:
        selection = rb.selection_block(registry, acc)
        if selection is not None and selection.selected and str(selection.date) >= str(latest.date if latest else ""):
            latest = selection
    if latest is None:
        return ""
    criteria = latest.criteria or {}
    parts = []
    if criteria.get("column"):
        parts.append(f"column {criteria['column']}")
    if criteria.get("threshold") is not None:
        parts.append(f"threshold {criteria['threshold']}")
    if criteria.get("metadata_column"):
        parts.append(f"{criteria['metadata_column']} = {criteria.get('metadata_value')}")
    parts.extend(_run_filter_detail(criteria))
    parts.append(str(latest.date))
    return ", ".join(p for p in parts if p)


def _exclusion_detail(registry: Registry) -> str:
    """How many accessions carry each exclusion reason."""
    reasons: Dict[str, int] = {}
    for acc in registry.datasets:
        exclusion = rb.exclusion_block(registry, acc)
        if exclusion is not None and exclusion.excluded:
            reason = str(exclusion.reason or "no reason given")
            reasons[reason] = reasons.get(reason, 0) + 1
    return ", ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))


def _print_stages(stages: Dict[str, Any], registry: Registry, emit: Callable[[str], None]) -> None:
    emit("\nStages")
    emit("======")
    details = {
        "selected": _selection_detail(registry),
        "excluded": _exclusion_detail(registry),
    }
    for stage in STAGES:
        info = stages[stage]
        detail = details.get(stage)
        emit(f"  {stage:<10s} : {info['count']}" + (f"   {detail}" if detail else ""))
    truncated = _download_verdicts(registry)["truncated"]
    if truncated:
        emit(f"  truncated downloads: {len(truncated)} (" + ", ".join(truncated) + ")")


def _print_genomes(genomes: Dict[str, Any], emit: Callable[[str], None]) -> None:
    if not genomes:
        return
    emit("\nGenomes")
    emit("=======")
    for genome_id, info in genomes.items():
        zero_mapped = len(info["zero_mapped"])
        empty_dirs = len(info["empty_assembly_dirs"])
        emit(
            f"  {genome_id}: extracted {info['extracted']} ({zero_mapped} with 0 mapped reads), "
            f"assembled {info['assembled']}, {empty_dirs} empty assembly dir(s)"
        )


def _print_stage_filter(
    registry: Registry, stage: Optional[str], genomes: Optional[List[str]], emit: Callable[[str], None]
) -> None:
    if not stage:
        return
    accs = _stage_filter_accessions(registry, stage, genomes)
    detail = f" (genome {', '.join(genomes)})" if genomes else ""
    emit(f"\nStage '{stage}'{detail}: " + (", ".join(accs) if accs else "(none)"))


def _print_gaps(registry: Registry, emit: Callable[[str], None]) -> None:
    selected = set(query(registry, "selected"))
    excluded = set(query(registry, "excluded"))
    downloaded_set = set(query(registry, "downloaded"))
    not_downloaded = sorted(selected - excluded - downloaded_set)
    emit("\nGaps")
    emit("====")
    emit("  Selected but not downloaded : " + (", ".join(not_downloaded) if not_downloaded else "(none)"))
    for genome_id in sorted(known_genome_ids(registry)):
        gap = sorted(downloaded_set - set(query(registry, "extracted", genome_id)))
        if gap:
            emit(f"  Downloaded but not extracted for {genome_id} : " + ", ".join(gap))


def _print_drift(drift: Dict[str, Any], emit: Callable[[str], None]) -> None:
    if not drift:
        return
    emit("\nDrift against disk")
    emit("===================")
    emit(
        "  Recorded downloaded but missing on disk : "
        + (", ".join(drift["recorded_missing"]) if drift["recorded_missing"] else "(none)")
    )
    emit(
        "  On disk but not tracked as downloaded   : "
        + (", ".join(drift["untracked_fastq"]) if drift["untracked_fastq"] else "(none)")
    )
    if drift["untracked_extractions"]:
        pairs = ", ".join(f"{acc}/{genome_id}" for acc, genome_id in drift["untracked_extractions"])
        emit(f"  Untracked extractions                   : {pairs}")
    if drift["empty_assembly_dirs"]:
        pairs = ", ".join(f"{acc}/{genome_id}" for acc, genome_id in drift["empty_assembly_dirs"])
        emit(f"  Empty assembly directories               : {pairs}")
    if drift.get("dangling_links"):
        emit("  Store links with a missing target       : " + ", ".join(drift["dangling_links"]))


def _print_next(steps: List[Dict[str, Any]], emit: Callable[[str], None]) -> None:
    if not steps:
        return
    emit("\nSuggested next steps")
    emit("=====================")
    for step in steps:
        emit(f"  {step['command']}")
        emit("    accessions: " + ", ".join(step["accessions"]))


def _print_report(
    args: argparse.Namespace, report: Dict[str, Any], registry: Registry, emit: Callable[[str], None]
) -> None:
    _print_inventory(report, args.list_missing, emit)
    if report.get("store"):
        _print_store(report["store"], emit)
    _print_stages(report["stages"], registry, emit)
    _print_genomes(report["genomes"], emit)
    _print_stage_filter(registry, args.stage, args.genome, emit)
    if args.list_missing:
        _print_gaps(registry, emit)
    _print_drift(report["drift"], emit)
    if args.next:
        _print_next(report.get("next", []), emit)
