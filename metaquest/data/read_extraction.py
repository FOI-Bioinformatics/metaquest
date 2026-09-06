"""Targeted read extraction prior to assembly.

Given a parsed containment table (samples x target genomes), this module maps
each sample's reads against a chosen target genome and keeps only the reads that
align. The resulting reduced FASTQ set supports a small, targeted assembly rather
than a whole-metagenome assembly.

The mapping uses minimap2 (one aligner for both short and long reads) and samtools
to filter and export the mapped reads. External tools run through
``SecureSubprocess.run_secure`` and must be present on the system.
"""

import gzip
import logging
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from metaquest.core.exceptions import DataAccessError, ProcessingError
from metaquest.data.file_io import ensure_directory
from metaquest.data.sra import MATE1_SUFFIXES, fastq_files, fastq_stem, orphan_fastq, primary_fastq
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)

# minimap2 presets by read type; keys are accepted on the CLI via --preset
MINIMAP2_PRESETS = {
    "sr": "sr",  # short paired/single Illumina reads
    "map-ont": "map-ont",  # Oxford Nanopore
    "map-pb": "map-pb",  # PacBio CLR
    "map-hifi": "map-hifi",  # PacBio HiFi
}

# minimap2 prints this when the two mate files differ in length; it then maps single-end.
UNEQUAL_MATES_MARKER = "different number of records"

# samtools -F flags dropped by the alignment filter: unmapped (0x4), secondary (0x100)
# and supplementary (0x800) alignments.
FILTER_FLAGS = "0x904"


@dataclass
class ExtractionResult:
    """Outcome of mapping one sample against the target genome."""

    files: List[Path]
    mapped_records: int
    unequal_mates: bool = False
    skipped: bool = False
    # Mapped records before the secondary/supplementary/MAPQ filter (0 for a skipped result).
    mapped_total: int = 0


def _notify_result(
    on_result: Optional[Callable[[str, "ExtractionResult"], None]], accession: str, result: "ExtractionResult"
) -> None:
    """Call ``on_result`` in isolation so a failing callback never stops the run.

    The callback records the result (a registry write, which can raise, for example on a
    lock timeout). One sample's bookkeeping failing must not cost the remaining samples
    their extraction.
    """
    if on_result is None:
        return
    try:
        on_result(accession, result)
    except Exception as e:
        logger.warning("Recording the extraction result for %s failed: %s", accession, e)


def _count_records(path: Path, *filter_args: str) -> int:
    """Number of records in a SAM/BAM file, via ``samtools view -c``.

    ``filter_args`` are extra ``samtools view`` flags (e.g. ``"-F", "4"``) applied before
    the count; with none, every record in the file is counted.
    """
    result = SecureSubprocess.run_secure("samtools", ["view", "-c", *filter_args, str(path)])
    text = (result.stdout or "").strip()
    return int(text) if text.isdigit() else 0


def resolve_index_path(genome_fasta: Union[str, Path], preset: str, index_dir: Union[str, Path]) -> Path:
    """The minimap2 index path ``build_index`` reads or writes for this genome and preset."""
    return Path(index_dir) / f"{Path(genome_fasta).stem}.{preset}.mmi"


def build_index(genome_fasta: Union[str, Path], preset: str, index_dir: Union[str, Path]) -> Path:
    """Build (or reuse) a minimap2 index for the target genome, shared across every sample.

    The index lives at ``resolve_index_path(genome_fasta, preset, index_dir)`` and is
    rebuilt only when the FASTA's mtime is newer than the index (or the index does not
    exist yet), so a genome replaced between runs is picked up automatically.
    """
    genome_path = Path(genome_fasta)
    index_root = ensure_directory(index_dir)
    index_path = resolve_index_path(genome_path, preset, index_root)
    if index_path.exists() and index_path.stat().st_mtime >= genome_path.stat().st_mtime:
        return index_path
    SecureSubprocess.run_secure("minimap2", ["-x", preset, "-d", str(index_path), str(genome_path)])
    return index_path


def _run_minimap2(
    accession: str,
    preset: str,
    threads: int,
    sam_path: Path,
    reference: Path,
    genome_fasta: Path,
    reads: List[Path],
) -> Any:
    """Align ``reads`` against the prebuilt index, retrying once against the FASTA directly
    if the index fails (e.g. it was built by an incompatible minimap2 version)."""
    read_args = [str(r) for r in reads]
    args = ["-a", "-x", preset, "-t", str(threads), "-o", str(sam_path), str(reference), *read_args]
    try:
        return SecureSubprocess.run_secure("minimap2", args)
    except Exception as exc:
        logger.warning(
            "%s: minimap2 failed against the prebuilt index %s (%s); retrying against the FASTA directly",
            accession,
            reference,
            exc,
        )
        fallback_args = ["-a", "-x", preset, "-t", str(threads), "-o", str(sam_path), str(genome_fasta), *read_args]
        return SecureSubprocess.run_secure("minimap2", fallback_args)


def _fastq_is_empty(path: Path) -> bool:
    """True when the file is missing or has no records (gzip or plain)."""
    if not path.exists():
        return True
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return handle.readline() == ""


def select_samples_for_genome(containment: pd.DataFrame, genome_id: str, threshold: float) -> List[str]:
    """Return the sample accessions whose containment for a genome meets a threshold.

    Args:
        containment: Parsed containment table indexed by sample accession.
        genome_id: Target genome column to select on.
        threshold: Minimum containment value (inclusive).

    Raises:
        ProcessingError: If the genome column is absent from the table.
    """
    if genome_id not in containment.columns:
        raise ProcessingError(
            f"Genome '{genome_id}' not found in the containment table. "
            f"Available columns include: {', '.join(map(str, containment.columns[:5]))}"
        )
    values = pd.to_numeric(containment[genome_id], errors="coerce")
    return [str(acc) for acc in containment.index[values >= threshold]]


def selected_samples(parsed_containment: Union[str, Path], genome_id: str, threshold: float) -> List[str]:
    """Return the sample accessions that meet containment >= threshold for a genome.

    Reads the parsed containment table and delegates to ``select_samples_for_genome``.
    Exposed so a caller (the CLI) can learn which samples were selected without
    duplicating that logic, or re-running the full extraction.

    Raises:
        DataAccessError: If the table is missing.
        ProcessingError: If the genome column is absent from the table.
    """
    table_path = Path(parsed_containment)
    if not table_path.exists():
        raise DataAccessError(f"Parsed containment table not found: {table_path}")
    containment = pd.read_csv(table_path, sep="\t", index_col=0)
    return select_samples_for_genome(containment, genome_id, threshold)


def _record_matches(record: Dict[str, Any], genome_path: Path, preset: str, threshold: float) -> bool:
    """True when a registry extraction record matches the current call and its files are still usable.

    A parameter that is absent or None is a wildcard. Records rebuilt from disk by
    ``status --init`` carry no parameters at all, and rejecting them would remap every
    sample of an already extracted project.
    """
    recorded_fasta = record.get("genome_fasta")
    if recorded_fasta is not None and Path(recorded_fasta).resolve() != genome_path.resolve():
        return False
    recorded_preset = record.get("preset")
    if recorded_preset is not None and recorded_preset != preset:
        return False
    recorded_threshold = record.get("threshold")
    if recorded_threshold is not None and float(recorded_threshold) != float(threshold):
        return False
    if record.get("mapped_reads") == 0:
        return True
    return all(Path(p).exists() for p in record.get("files", []))


def _skipped_result(record: Dict[str, Any]) -> ExtractionResult:
    """The result of a sample whose recorded extraction still stands."""
    return ExtractionResult(
        files=[Path(p) for p in record.get("files", [])],
        mapped_records=int(record.get("mapped_reads") or 0),
        unequal_mates=bool(record.get("unequal_mates", False)),
        skipped=True,
    )


def _sample_reads(fastq_folder: Path, accession: str) -> List[Path]:
    """Return the FASTQ files for one accession that minimap2 should map, R1 before R2.

    For paired data that is the mate pair alone: the bare ``<acc>.fastq`` file that
    ``--split-3`` writes for unpaired spots would make minimap2 read three files as an
    interleaved set and report mismatched mate counts, so it is left out. Single-end data
    gives the bare file. Zero-byte files and ``.gz.tmp.<pid>`` leftovers of an interrupted
    download are excluded by ``fastq_files``.
    """
    acc_dir = fastq_folder / accession
    files = fastq_files(acc_dir)
    if not files:
        return []

    primary = primary_fastq(acc_dir)
    if primary is None:
        return files

    stem = fastq_stem(primary)
    for marker in MATE1_SUFFIXES:
        if stem.endswith(marker):
            mate_stem = stem[: -len(marker)] + marker[:-1] + "2"
            mate_two = next((p for p in files if fastq_stem(p) == mate_stem), None)
            if mate_two is None:
                return files
            orphan = orphan_fastq(acc_dir)
            if orphan is not None:
                logger.debug("%s: mapping the mate files only; %s holds unpaired reads", accession, orphan.name)
            return [primary, mate_two]

    return [primary]


def _align_and_count_mapped(
    accession: str,
    preset: str,
    threads: int,
    reference: Path,
    genome_fasta: Path,
    sam_paths: List[Path],
    mate_groups: List[List[Path]],
) -> Tuple[int, bool]:
    """Align each read group to its own SAM and return (total mapped records, saw the
    stderr marker). Mapped records are counted with ``-F 4`` (drops unmapped), before any
    other filtering."""
    mapped_total = 0
    stderr_marker_seen = False
    for sam_path, group in zip(sam_paths, mate_groups):
        aligned = _run_minimap2(accession, preset, threads, sam_path, reference, genome_fasta, group)
        if UNEQUAL_MATES_MARKER in (aligned.stderr or ""):
            stderr_marker_seen = True
        mapped_total += _count_records(sam_path, "-F", "4")
    return mapped_total, stderr_marker_seen


def _filter_and_merge_bam(
    sam_paths: List[Path], filter_args: List[str], threads: int, out_dir: Path, genome_id: str, bam_path: Path
) -> None:
    """Filter each SAM into a BAM (``-F 0x904 [-q min_mapq]``); merge with ``samtools cat``
    when there is more than one (the deliberate per-mate single-end fallback)."""
    if len(sam_paths) == 1:
        SecureSubprocess.run_secure(
            "samtools", ["view", "-b", *filter_args, "-@", str(threads), "-o", str(bam_path), str(sam_paths[0])]
        )
        return
    part_bams = [out_dir / f"{genome_id}.mate{i}.bam" for i in range(1, len(sam_paths) + 1)]
    for sam_path, part_bam in zip(sam_paths, part_bams):
        SecureSubprocess.run_secure(
            "samtools", ["view", "-b", *filter_args, "-@", str(threads), "-o", str(part_bam), str(sam_path)]
        )
    SecureSubprocess.run_secure("samtools", ["cat", "-o", str(bam_path), *(str(p) for p in part_bams)])
    for part_bam in part_bams:
        part_bam.unlink(missing_ok=True)


def _export_mapped_fastq(reads: List[Path], bam_path: Path, out_dir: Path, genome_id: str, threads: int) -> List[Path]:
    """Export a filtered BAM back to FASTQ. Reads without a mate flag (single-end mapping,
    or the unequal-mates fallback) go to the ``-0`` file."""
    if len(reads) < 2:
        out0 = out_dir / f"{genome_id}.fastq.gz"
        SecureSubprocess.run_secure("samtools", ["fastq", "-@", str(threads), "-0", str(out0), str(bam_path)])
        return [] if _fastq_is_empty(out0) else [out0]

    out1 = out_dir / f"{genome_id}_1.fastq.gz"
    out2 = out_dir / f"{genome_id}_2.fastq.gz"
    singles = out_dir / f"{genome_id}_s.fastq.gz"
    orphans = out_dir / f"{genome_id}_0.fastq.gz"
    SecureSubprocess.run_secure(
        "samtools",
        [
            "fastq",
            "-@",
            str(threads),
            "-1",
            str(out1),
            "-2",
            str(out2),
            "-s",
            str(singles),
            "-0",
            str(orphans),
            str(bam_path),
        ],
    )
    for path in (out1, out2, singles, orphans):
        if _fastq_is_empty(path):
            path.unlink(missing_ok=True)
    if out1.exists() and out2.exists():
        return [out1, out2]
    if orphans.exists():
        return [orphans]
    return [singles] if singles.exists() else []


def _map_and_extract(
    accession: str,
    reads: List[Path],
    reference: Path,
    genome_fasta: Path,
    out_dir: Path,
    genome_id: str,
    preset: str,
    threads: int,
    min_mapq: int = 0,
    sam_dir: Optional[Path] = None,
    keep_sam: bool = False,
    force_single_end: bool = False,
) -> ExtractionResult:
    """Map one sample's reads to the target genome and write the mapped reads.

    ``reference`` is the prebuilt minimap2 index shared across every sample of this
    ``extract_target_reads`` call; ``genome_fasta`` is the FASTA it was built from, used as
    a one-time fallback if aligning against the index fails. ``force_single_end`` (set when
    the caller already knows the two mate files disagree in read count) maps each mate file
    in its own minimap2 run and merges the two alignments afterwards, deliberately as
    single-end, rather than relying on minimap2's own after-the-fact stderr warning.

    Returns the FASTQ files written (two for paired input, one otherwise, none when
    nothing mapped) together with the mapped-record counts. The SAM alignment(s) live under
    ``sam_dir`` (or ``out_dir`` when not given) and are removed once the filtered BAM exists,
    unless ``keep_sam``; the BAM is always removed once the FASTQ export is written.
    """
    ensure_directory(out_dir)
    sam_root = ensure_directory(sam_dir) if sam_dir is not None else out_dir
    bam_path = out_dir / f"{genome_id}.mapped.bam"
    filter_args = ["-F", FILTER_FLAGS]
    if min_mapq:
        filter_args += ["-q", str(min_mapq)]

    unequal = force_single_end and len(reads) >= 2
    if unequal:
        mate_groups: List[List[Path]] = [[read] for read in reads[:2]]
        sam_paths = [sam_root / f"{genome_id}.mate1.sam", sam_root / f"{genome_id}.mate2.sam"]
        logger.warning(
            "%s: the mate files are recorded with different read counts, so each is mapped "
            "independently as single-end reads; re-download with fasterq-dump (which keeps mates "
            "in step) for paired extraction",
            accession,
        )
    else:
        mate_groups = [reads]
        sam_paths = [sam_root / f"{genome_id}.sam"]

    mapped_total, stderr_marker_seen = _align_and_count_mapped(
        accession, preset, threads, reference, genome_fasta, sam_paths, mate_groups
    )
    if stderr_marker_seen and not unequal:
        unequal = True
        logger.warning(
            "%s: the mate files have different read counts, so minimap2 mapped them as single-end reads; "
            "re-download with fasterq-dump (which keeps mates in step) for paired extraction",
            accession,
        )

    if mapped_total == 0:
        logger.warning("No reads from %s mapped to %s; nothing written", accession, genome_id)
        if not keep_sam:
            for sam_path in sam_paths:
                sam_path.unlink(missing_ok=True)
        return ExtractionResult([], 0, unequal, mapped_total=0)

    _filter_and_merge_bam(sam_paths, filter_args, threads, out_dir, genome_id, bam_path)
    if not keep_sam:
        for sam_path in sam_paths:
            sam_path.unlink(missing_ok=True)

    mapped = _count_records(bam_path)
    if mapped == 0:
        logger.warning("No reads from %s mapped to %s; nothing written", accession, genome_id)
        bam_path.unlink(missing_ok=True)
        return ExtractionResult([], 0, unequal, mapped_total=mapped_total)
    logger.info(
        "%s: kept %d of %d mapped records (secondary/supplementary and MAPQ below %d removed: %d)",
        accession,
        mapped,
        mapped_total,
        min_mapq,
        mapped_total - mapped,
    )

    written = _export_mapped_fastq(reads, bam_path, out_dir, genome_id, threads)
    bam_path.unlink(missing_ok=True)

    return ExtractionResult(written, mapped, unequal, mapped_total=mapped_total)


def _skip_if_truncated(accession: str, verdict: Optional[Dict[str, Any]], allow_truncated: bool) -> bool:
    """True (after logging) when this sample's truncated download should be skipped."""
    if not verdict or allow_truncated:
        return False
    logger.warning(
        "skipped %s: download truncated (%d of %d spots); use --allow-truncated",
        accession,
        verdict.get("reads_r1", 0),
        verdict.get("expected_spots", 0),
    )
    return True


def _extract_one_sample(
    accession: str,
    record: Optional[Dict[str, Any]],
    genome_path: Path,
    preset: str,
    threshold: float,
    force: bool,
    dry_run: bool,
    fastq_root: Path,
    genome_id: str,
    output_root: Path,
    threads: int,
    min_mapq: int,
    sam_dir: Optional[Path],
    keep_sam: bool,
    mate_counts: Dict[str, Tuple[int, int]],
    reference_holder: Dict[str, Path],
) -> Optional[ExtractionResult]:
    """Handle one selected sample: an already-done skip, missing reads, dry-run reporting,
    or a real mapping run. Returns None when the sample contributes nothing to ``results``
    (no FASTQ files found for it). The shared minimap2 index is built on first use here and
    cached in ``reference_holder["reference"]`` for every later sample of this call.
    """
    done = bool(record) and not force and _record_matches(record or {}, genome_path, preset, threshold)
    if done and not dry_run:
        result = _skipped_result(record or {})
        logger.info(
            "%s already extracted against %s (%d mapped reads); use --force to redo",
            accession,
            genome_id,
            result.mapped_records,
        )
        return result

    reads = _sample_reads(fastq_root, accession)
    if not reads:
        logger.warning("No FASTQ files found for %s under %s; skipping", accession, fastq_root)
        return None

    if dry_run:
        result = _skipped_result(record or {}) if done else ExtractionResult([], 0)
        if done:
            logger.info(
                "would skip %s (already extracted, %d mapped reads); use --force to redo",
                accession,
                result.mapped_records,
            )
        return result

    if "reference" not in reference_holder:
        index_dir = output_root / ".index"
        SecureSubprocess.add_allowed_root(index_dir)
        reference_holder["reference"] = build_index(genome_path, preset, index_dir)

    counts = mate_counts.get(accession)
    force_single_end = bool(counts is not None and len(reads) >= 2 and counts[0] != counts[1])
    outcome = _map_and_extract(
        accession,
        reads,
        reference_holder["reference"],
        genome_path,
        output_root / accession,
        genome_id,
        preset,
        threads,
        min_mapq=min_mapq,
        sam_dir=sam_dir,
        keep_sam=keep_sam,
        force_single_end=force_single_end,
    )
    if outcome.files:
        logger.info(
            "Extracted %d mapped records for %s -> %s",
            outcome.mapped_records,
            accession,
            ", ".join(str(p) for p in outcome.files),
        )
    return outcome


def extract_target_reads(
    parsed_containment: Union[str, Path],
    genome_id: str,
    genome_fasta: Union[str, Path],
    fastq_folder: Union[str, Path] = "fastq",
    output_folder: Union[str, Path] = "targeted",
    threshold: float = 0.1,
    preset: str = "sr",
    threads: int = 4,
    dry_run: bool = False,
    force: bool = False,
    already_done: Optional[Dict[str, Dict[str, Any]]] = None,
    on_result: Optional[Callable[[str, ExtractionResult], None]] = None,
    min_mapq: int = 0,
    temp_folder: Optional[Union[str, Path]] = None,
    allow_truncated: bool = False,
    mate_counts: Optional[Dict[str, Tuple[int, int]]] = None,
    truncated_downloads: Optional[Dict[str, Dict[str, Any]]] = None,
    keep_sam: bool = False,
) -> Dict[str, ExtractionResult]:
    """Extract reads mapping to a target genome for every qualifying sample.

    Args:
        parsed_containment: Parsed containment table (samples x genomes).
        genome_id: Target genome to extract against (a column in the table).
        genome_fasta: FASTA file for the target genome.
        fastq_folder: Root folder with per-accession FASTQ subdirectories.
        output_folder: Root folder for the extracted reads (per-accession subdirs).
        threshold: Minimum containment for a sample to be included.
        preset: minimap2 preset (sr, map-ont, map-pb, map-hifi).
        threads: Threads for minimap2 and samtools.
        dry_run: If True, report the qualifying samples without running any tool.
        force: If True, redo extraction even for a sample already recorded in ``already_done``.
        already_done: Accession -> the registry's extraction record for this genome. A sample
            already recorded there is skipped (unless ``force``) when the record's genome FASTA,
            preset and threshold match this call and its files are still on disk (or it mapped 0).
        on_result: Called on the main thread with (accession, result) after every sample of a
            real run, skipped samples included, so the caller can checkpoint each result as it
            lands. A callback that raises is logged and does not stop the run. Dry runs never
            call it.
        min_mapq: Minimum mapping quality (samtools ``-q``) a record must meet to be kept, in
            addition to dropping secondary/supplementary alignments. 0 keeps every mapped
            record regardless of quality; a divergent strain of the target genome can map with
            a genuinely low MAPQ, so a nonzero value is best reserved for close relatives.
        temp_folder: Where the intermediate SAM alignment(s) are written; defaults to the
            sample's own output folder when not given.
        allow_truncated: If True, extract even for a sample whose registry download verdict is
            "truncated" (via ``truncated_downloads``); otherwise that sample is skipped.
        mate_counts: Accession -> (mate 1 reads, mate 2 reads). When the two counts differ,
            the sample's mate files are mapped independently as single-end reads rather than
            as a pair, deliberately rather than relying on minimap2's own stderr warning.
        truncated_downloads: Accession -> the registry's download completeness verdict, for
            samples whose verdict is "truncated". Such a sample is skipped unless
            ``allow_truncated``.
        keep_sam: If True, keep the intermediate SAM alignment(s) instead of removing them
            once the filtered BAM exists (for debugging).

    Returns:
        Mapping of accession to an ``ExtractionResult`` (empty files in dry-run).

    Raises:
        DataAccessError: If inputs are missing.
        ProcessingError: If the genome column or preset is invalid.
    """
    table_path = Path(parsed_containment)
    if not table_path.exists():
        raise DataAccessError(f"Parsed containment table not found: {table_path}")
    if preset not in MINIMAP2_PRESETS:
        raise ProcessingError(f"Unknown minimap2 preset '{preset}'. Choose one of: {', '.join(MINIMAP2_PRESETS)}")

    fastq_root = Path(fastq_folder)
    genome_path = Path(genome_fasta)
    if not dry_run and not genome_path.exists():
        raise DataAccessError(f"Target genome FASTA not found: {genome_path}")

    containment = pd.read_csv(table_path, sep="\t", index_col=0)
    samples = select_samples_for_genome(containment, genome_id, threshold)
    logger.info("%d sample(s) meet containment >= %.3f for %s", len(samples), threshold, genome_id)

    results: Dict[str, ExtractionResult] = {}
    output_root = Path(output_folder)
    SecureSubprocess.add_allowed_root(output_root)
    already_done = already_done or {}
    mate_counts = mate_counts or {}
    truncated_downloads = truncated_downloads or {}

    # The index is built lazily, the first time a sample actually needs mapping, so a run
    # where every sample is already done (or dry) never touches minimap2 at all. Held in a
    # dict (rather than a local reassigned across the loop) so the helper below can cache
    # it too.
    reference_holder: Dict[str, Path] = {}
    sam_dir: Optional[Path] = None
    if temp_folder is not None:
        sam_dir = Path(temp_folder)
        SecureSubprocess.add_allowed_root(sam_dir)

    for accession in samples:
        if _skip_if_truncated(accession, truncated_downloads.get(accession), allow_truncated):
            continue
        result = _extract_one_sample(
            accession,
            already_done.get(accession),
            genome_path,
            preset,
            threshold,
            force,
            dry_run,
            fastq_root,
            genome_id,
            output_root,
            threads,
            min_mapq,
            sam_dir,
            keep_sam,
            mate_counts,
            reference_holder,
        )
        if result is None:
            continue
        results[accession] = result
        if not dry_run:
            _notify_result(on_result, accession, result)

    return results


def resolve_assembly_threads(requested: Optional[int], fallback: int) -> int:
    """Pick the megahit thread count, defaulting to single-thread on macOS.

    megahit 1.2.9's multithreaded k-mer sorting step segfaults on recent macOS
    (the prebuilt binary predates the current runtime), so when the user has not
    asked for a specific value we cap the assembly at one thread on Darwin.
    minimap2/samtools are unaffected and keep using ``fallback`` threads.

    Args:
        requested: An explicit thread count from the user, or None for the default.
        fallback: The thread count to use when no override is given (non-macOS).

    Returns:
        The number of threads to run megahit with.
    """
    if requested is not None:
        return requested
    if platform.system() == "Darwin":
        return 1
    return fallback


def assemble_extracted_reads(
    reads: List[Path],
    output_dir: Union[str, Path],
    threads: int = 4,
    min_contig_len: Optional[int] = None,
    force: bool = False,
) -> Tuple[Path, bool]:
    """Assemble a set of extracted FASTQ files with megahit.

    Single-end input (one file) uses megahit ``-r``; paired input (two files) uses
    ``-1``/``-2``. The reads are expected to be a small, target-filtered set.

    If ``output_dir`` already holds contigs, the assembly is considered done and
    megahit is not rerun unless ``force`` is set. If the folder exists but holds no
    contigs (an interrupted run), an error is raised unless ``force`` is set, in
    which case the folder is removed before megahit runs.

    Args:
        reads: One or two FASTQ files to assemble.
        output_dir: megahit output directory.
        threads: CPU threads.
        min_contig_len: Optional minimum contig length.
        force: If True, redo the assembly even if it already ran.

    Returns:
        The megahit output directory and whether megahit actually ran (False when the
        assembly was already there), so the caller can leave an existing record alone.

    Raises:
        ProcessingError: If the number of reads is unsupported, or the output
            directory exists without contigs and ``force`` is not set.
    """
    out_dir = Path(output_dir)
    contigs_path = out_dir / "final.contigs.fa"
    if out_dir.exists():
        if contigs_path.exists() and not force:
            logger.info("%s already assembled; use --force to redo", out_dir)
            return out_dir, False
        if not contigs_path.exists() and not force:
            raise ProcessingError("Assembly folder exists but holds no contigs (interrupted run?); rerun with --force")
        if force:
            shutil.rmtree(out_dir, ignore_errors=True)

    SecureSubprocess.add_allowed_root(out_dir.parent)
    args: List[str] = []
    if len(reads) == 2:
        args += ["-1", str(reads[0]), "-2", str(reads[1])]
    elif len(reads) == 1:
        args += ["-r", str(reads[0])]
    else:
        raise ProcessingError(f"Expected 1 or 2 FASTQ files to assemble, got {len(reads)}")

    args += ["--num-cpu-threads", str(threads), "-o", str(out_dir)]
    if min_contig_len is not None:
        args += ["--min-contig-len", str(min_contig_len)]

    SecureSubprocess.run_secure("megahit", args)
    logger.info("Assembly written to %s", out_dir)
    return out_dir, True


def summarise_contigs(contigs: Union[str, Path]) -> Dict[str, int]:
    """Contig count, total length, N50 and largest contig of a FASTA file.

    megahit headers carry ``len=<bp>``; when present that value is used, so the
    scan reads only header lines. Otherwise sequence lengths are summed.
    """
    path = Path(contigs)
    if not path.exists():
        return {"contigs": 0, "total_bp": 0, "n50": 0, "largest": 0}
    lengths: List[int] = []
    current = 0
    have_current = False
    header_len = False
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if have_current:
                    lengths.append(current)
                have_current = True
                match = re.search(r"\blen=(\d+)", line)
                current = int(match.group(1)) if match else 0
                header_len = match is not None
            elif have_current and not header_len:
                current += len(line.strip())
    if have_current:
        lengths.append(current)
    lengths.sort(reverse=True)
    total = sum(lengths)
    n50 = 0
    running = 0
    for length in lengths:
        running += length
        if running * 2 >= total:
            n50 = length
            break
    return {"contigs": len(lengths), "total_bp": total, "n50": n50, "largest": lengths[0] if lengths else 0}


def megahit_version() -> str:
    """The installed megahit's version string, or an empty string if it cannot be run."""
    try:
        result = SecureSubprocess.run_secure("megahit", ["--version"])
        return (result.stdout or "").strip()
    except Exception:
        return ""
