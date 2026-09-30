"""Reconcile the project registry with the disk in two steps: a slow scan, then a fast apply.

``status --reconcile`` walks the FASTQ and targeted folders, counts the reads of every
extraction the registry does not know about and recomputes download completeness verdicts
from the reads on disk. On a large project that takes minutes, far longer than any other
writer should wait for the registry lock. ``scan_reconcile`` therefore does all of that
reading on a snapshot of the registry, without the lock, and returns a ``ReconcilePlan``;
``apply_reconcile`` then applies the plan to the registry loaded inside the lock, checking
each condition again against that registry, and reads no reads. ``reconcile`` runs both
steps on one registry and keeps the behaviour it had before the split.

The module sits next to ``metaquest.data.registry``, which is held at its current size by
the module size ceiling; that module re-exports ``reconcile``.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry as _registry
from metaquest.data import registry_blocks as rb
from metaquest.data.read_extraction import coverage_table_path, summarise_contigs
from metaquest.data.registry import (
    ProjectPaths,
    ReconcileReport,
    Registry,
    _genome_ids_on_disk,
    _infer_coverage,
    empty_assembly_dirs,
    query,
    record_assembly,
    record_download,
    record_extraction,
    scan_assemblies,
    scan_downloads,
    scan_extractions,
    set_download_verdict,
)
from metaquest.data.sra import accession_has_fastq, count_fastq_reads, verify_download

logger = logging.getLogger(__name__)

_CONTIGS_NAME = "final.contigs.fa"
# Returned by ReconcilePlan.value for a figure the scan did not compute.
_NOT_SCANNED = object()


@dataclass
class ReconcilePlan:
    """What ``scan_reconcile`` found on disk, and the figures it computed there.

    The folder listings (``on_disk``, ``extractions``, ``assemblies``) and the report-only
    findings (``empty_assembly_dirs``, ``dangling_links``) are taken once, by the scan.
    ``computed`` holds every figure that required reading files: the read count and coverage
    of an untracked extraction, the contig statistics of its assembly, and the completeness
    verdict of a download that has none. Each is keyed by what it depends on (the accession,
    genome id, recorded spot count or store root), so ``apply_reconcile`` uses a figure only
    when the registry it is given still asks the same question, and skips it otherwise.
    """

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    on_disk: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    genome_ids: List[str] = field(default_factory=list)
    extractions: Dict[str, Dict[str, List[Path]]] = field(default_factory=dict)
    assemblies: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    empty_assembly_dirs: List[Tuple[str, str]] = field(default_factory=list)
    dangling_links: List[str] = field(default_factory=list)
    computed: Dict[Tuple[str, ...], Any] = field(default_factory=dict)

    def value(self, key: Tuple[str, ...], compute: Optional[Callable[[], Any]]) -> Any:
        """The figure stored under ``key``; with ``compute`` given and none stored, compute and store it.

        Returns ``_NOT_SCANNED`` when nothing is stored and ``compute`` is None, which is how the
        apply step learns that a condition appeared only after the scan.
        """
        if key not in self.computed:
            if compute is None:
                return _NOT_SCANNED
            self.computed[key] = compute()
        return self.computed[key]


def scan_reconcile(snapshot: Registry, paths: ProjectPaths) -> ReconcilePlan:
    """List the project's folders and compute every figure a reconcile of ``snapshot`` needs.

    Runs without the registry lock and may take minutes (it counts reads). ``snapshot`` is not
    changed: the reconcile is rehearsed on a copy of it, and each figure that rehearsal needs is
    computed and stored in the plan.
    """
    # Imported here rather than at module level: the store package imports the data layer.
    from metaquest.store.link import dangling_links

    genome_ids = sorted(_genome_ids_on_disk(paths, snapshot))
    plan = ReconcilePlan(
        paths=paths,
        on_disk=scan_downloads(paths.fastq),
        genome_ids=genome_ids,
        extractions=scan_extractions(paths.targeted, genome_ids),
        assemblies=scan_assemblies(paths.targeted, genome_ids),
        empty_assembly_dirs=empty_assembly_dirs(paths.targeted, genome_ids),
        dangling_links=dangling_links(paths.fastq),
    )
    _reconcile(copy.deepcopy(snapshot), plan, compute=True)
    return plan


def apply_reconcile(registry: Registry, plan: ReconcilePlan) -> ReconcileReport:
    """Apply ``plan`` to ``registry`` in place and return what differed; reads no FASTQ.

    Meant to run inside a ``registry_transaction`` on the freshly loaded registry. Every
    condition is checked again against ``registry``: a download recorded since the scan is not
    marked missing twice, an extraction recorded since the scan is not inferred over, and a
    verdict is filled in only when the scan computed it for the same recorded spot count or
    store root. Work the scan did not see is left for the next reconcile.
    """
    return _reconcile(registry, plan, compute=False)


def reconcile(registry: Registry, paths: ProjectPaths) -> ReconcileReport:
    """Compare the registry with the disk: mark missing downloads, register untracked work.

    Untracked FASTQ and extractions are recorded the way ``bootstrap_from_disk`` records
    them, with ``"inferred": true``, so a project worked on outside MetaQuest lands in the
    journal instead of being reported as drift on every run. They stay in the report.

    A link into a shared store whose target has gone (an unmounted store, a dataset removed
    from it) is reported as well, and is not repaired here: removing the link would lose the
    record of which datasets this project uses.

    Equivalent to ``apply_reconcile(registry, scan_reconcile(registry, paths))``; a caller that
    saves the result should run the scan outside the registry lock and only the apply inside it.
    """
    return apply_reconcile(registry, scan_reconcile(registry, paths))


def _reconcile(registry: Registry, plan: ReconcilePlan, compute: bool) -> ReconcileReport:
    """The reconcile itself; with ``compute`` False it only uses figures ``plan`` already holds."""
    report = ReconcileReport()
    for acc in registry.datasets:
        download = rb.download_block(registry, acc)
        if download is not None and download.state == "downloaded" and acc not in plan.on_disk:
            # A download another process finished after the scan is not in the scan's listing;
            # one look at its folder (a few stat calls) keeps it from being marked missing.
            if accession_has_fastq(plan.paths.fastq / acc):
                continue
            download.state, download.date = "missing", _registry._now()
            rb.set_download_block(registry, acc, download)
            report.recorded_missing.append(acc)
    tracked = set(query(registry, "downloaded"))
    report.untracked_fastq = sorted(acc for acc in plan.on_disk if acc not in tracked)
    for acc in report.untracked_fastq:
        record_download(registry, acc, "downloaded", plan.paths.fastq, attempt=False)
        rb.mark_inferred(registry, acc, "download", attempts=0)
    for acc, per_genome_files in plan.extractions.items():
        for genome_id, files in per_genome_files.items():
            if rb.extraction_block(registry, acc, genome_id) is None:
                if _infer_extraction(registry, plan, acc, genome_id, files, compute):
                    report.untracked_extractions.append((acc, genome_id))
    report.empty_assembly_dirs = list(plan.empty_assembly_dirs)
    report.dangling_links = list(plan.dangling_links)
    _fill_verdicts(registry, plan, compute)
    return report


def _count_extraction(files: List[Path], genome_id: str) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Reads in every file of one extraction, and the coverage its table records (or None)."""
    reads = sum(count_fastq_reads(f) for f in files)
    coverage = _infer_coverage(coverage_table_path(files[0].parent, genome_id)) if files else None
    return reads, coverage


def _infer_extraction(
    registry: Registry, plan: ReconcilePlan, acc: str, genome_id: str, files: List[Path], compute: bool
) -> bool:
    """Record one untracked extraction (and its assembly), marked inferred; False when not scanned.

    The read count covers both mates, singles and unpaired reads, as ``bootstrap_from_disk``
    counts them; the assembly is recorded only when it holds contigs.
    """
    counted = plan.value(
        ("extraction", acc, genome_id), (lambda: _count_extraction(files, genome_id)) if compute else None
    )
    if counted is _NOT_SCANNED:
        return False
    reads, coverage = counted
    record_extraction(registry, acc, genome_id, files, reads, False, {}, coverage=coverage)
    rb.mark_inferred(registry, acc, "extractions", genome_id=genome_id)
    asm_dir = plan.assemblies.get(acc, {}).get(genome_id)
    if asm_dir is not None:
        contigs = asm_dir / _CONTIGS_NAME
        stats = plan.value(("assembly", acc, genome_id), (lambda: summarise_contigs(contigs)) if compute else None)
        if stats is not _NOT_SCANNED and stats["contigs"] > 0:
            record_assembly(registry, acc, genome_id, asm_dir, stats, "", {})
            rb.mark_inferred(registry, acc, "extractions", genome_id=genome_id)
    return True


def _fill_missing_download_verdicts(registry: Registry, paths: ProjectPaths) -> None:
    """Compute a completeness verdict for a downloaded accession that never got one.

    A project downloaded before completeness verification existed (or with
    ``--no-verify-downloads``) has metadata recorded but no ``download.complete`` verdict.
    When NCBI's recorded spot count is on file, this recomputes it the same way a fresh
    download would have, against the accession's files on disk. A record whose reads came from
    the shared store is filled in from that dataset's sidecar instead: the store already
    verified it when it was downloaded, and counting the reads again through a link would
    repeat work another project has done.
    """
    _fill_verdicts(registry, ReconcilePlan(paths=paths), compute=True)


def _fill_verdicts(registry: Registry, plan: ReconcilePlan, compute: bool) -> None:
    """``_fill_missing_download_verdicts`` on a plan; with ``compute`` False, only scanned verdicts are used."""
    for acc in registry.datasets:
        download = rb.download_block(registry, acc) or rb.DownloadBlock()
        # A verdict recorded as an empty dict counts as none, as it did before the typed blocks.
        if download.state != "downloaded" or (download.complete is not None and download.complete.to_dict()):
            continue
        if download.source == "store":
            root = rb.store_block(registry).root or ""
            store_verdict = plan.value(
                ("store", acc, root), (lambda: _store_verdict(registry, acc)) if compute else None
            )
            if store_verdict is not _NOT_SCANNED and store_verdict is not None:
                set_download_verdict(registry, acc, store_verdict)
            continue
        spots = (rb.metadata_block(registry, acc) or rb.MetadataBlock()).run_total_spots
        if not spots:
            continue
        acc_dir = plan.paths.fastq / acc
        if not acc_dir.is_dir():
            continue
        verify = plan.value(
            ("spots", acc, str(spots)), (lambda: verify_download(acc, acc_dir, spots)) if compute else None
        )
        if verify is _NOT_SCANNED:
            continue
        set_download_verdict(registry, acc, {"method": "spots", "ratio": verify["ratio"], "verdict": verify["verdict"]})


def _store_verdict(registry: Registry, accession: str) -> Optional[Dict[str, Any]]:
    """The completeness verdict the store's sidecar records for ``accession``, or None.

    Reads the store root the registry itself recorded; a project whose store has moved or is
    not mounted simply gets no verdict this time round, exactly as before.
    """
    root = rb.store_block(registry).root
    if not root:
        return None
    # Imported here, not at module level: metaquest.store imports the data registry.
    from metaquest.store.layout import sidecar_path, store_paths
    from metaquest.store.sidecar import sidecar_completeness

    try:
        return sidecar_completeness(sidecar_path(store_paths(Path(root)), accession))
    except (OSError, DataAccessError) as e:
        logger.warning("Could not read the store sidecar for %s: %s", accession, e)
        return None
