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


@dataclass
class ExtractionResult:
    """Outcome of mapping one sample against the target genome."""

    files: List[Path]
    mapped_records: int
    unequal_mates: bool = False
    skipped: bool = False


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


def _count_bam_records(bam_path: Path) -> int:
    """Number of records in a BAM file, via ``samtools view -c``."""
    result = SecureSubprocess.run_secure("samtools", ["view", "-c", str(bam_path)])
    text = (result.stdout or "").strip()
    return int(text) if text.isdigit() else 0


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
    """Return the FASTQ files for one accession, sorted (R1 before R2)."""
    acc_dir = fastq_folder / accession
    if not acc_dir.is_dir():
        return []
    return sorted(p for p in acc_dir.glob("*.fastq*") if p.is_file())


def _map_and_extract(
    accession: str,
    reads: List[Path],
    genome_fasta: Path,
    out_dir: Path,
    genome_id: str,
    preset: str,
    threads: int,
) -> ExtractionResult:
    """Map one sample's reads to the target genome and write the mapped reads.

    Returns the FASTQ files written (two for paired input, one otherwise, none when
    nothing mapped) together with the mapped-record count. Intermediate SAM/BAM
    files are removed.
    """
    ensure_directory(out_dir)
    sam_path = out_dir / f"{genome_id}.sam"
    bam_path = out_dir / f"{genome_id}.mapped.bam"

    # 1. Align reads to the reference (SAM output).
    minimap_args = ["-a", "-x", preset, "-t", str(threads), "-o", str(sam_path), str(genome_fasta)]
    minimap_args.extend(str(r) for r in reads)
    aligned = SecureSubprocess.run_secure("minimap2", minimap_args)
    unequal = UNEQUAL_MATES_MARKER in (aligned.stderr or "")
    if unequal:
        logger.warning(
            "%s: the mate files have different read counts, so minimap2 mapped them as single-end reads; "
            "re-download with fasterq-dump (which keeps mates in step) for paired extraction",
            accession,
        )

    # 2. Keep only mapped records (-F 4 drops the unmapped flag) and count them.
    SecureSubprocess.run_secure(
        "samtools", ["view", "-b", "-F", "4", "-@", str(threads), "-o", str(bam_path), str(sam_path)]
    )
    mapped = _count_bam_records(bam_path)
    if mapped == 0:
        logger.warning("No reads from %s mapped to %s; nothing written", accession, genome_id)
        for tmp in (sam_path, bam_path):
            tmp.unlink(missing_ok=True)
        return ExtractionResult([], 0, unequal)

    # 3. Export mapped reads back to FASTQ. Reads without a mate flag (single-end
    #    mapping, or a fallback after unequal mates) go to the -0 file.
    if len(reads) >= 2:
        out1 = out_dir / f"{genome_id}_1.fastq.gz"
        out2 = out_dir / f"{genome_id}_2.fastq.gz"
        singles = out_dir / f"{genome_id}_s.fastq.gz"
        orphans = out_dir / f"{genome_id}_0.fastq.gz"
        SecureSubprocess.run_secure(
            "samtools",
            ["fastq", "-1", str(out1), "-2", str(out2), "-s", str(singles), "-0", str(orphans), str(bam_path)],
        )
        for path in (out1, out2, singles, orphans):
            if _fastq_is_empty(path):
                path.unlink(missing_ok=True)
        if out1.exists() and out2.exists():
            written = [out1, out2]
        elif orphans.exists():
            written = [orphans]
        else:
            written = [singles] if singles.exists() else []
    else:
        out0 = out_dir / f"{genome_id}.fastq.gz"
        SecureSubprocess.run_secure("samtools", ["fastq", "-0", str(out0), str(bam_path)])
        written = [] if _fastq_is_empty(out0) else [out0]

    for tmp in (sam_path, bam_path):
        tmp.unlink(missing_ok=True)

    return ExtractionResult(written, mapped, unequal)


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

    for accession in samples:
        record = already_done.get(accession)
        done = bool(record) and not force and _record_matches(record or {}, genome_path, preset, threshold)
        if done and not dry_run:
            results[accession] = _skipped_result(record or {})
            logger.info(
                "%s already extracted against %s (%d mapped reads); use --force to redo",
                accession,
                genome_id,
                results[accession].mapped_records,
            )
            _notify_result(on_result, accession, results[accession])
            continue
        reads = _sample_reads(fastq_root, accession)
        if not reads:
            logger.warning("No FASTQ files found for %s under %s; skipping", accession, fastq_root)
            continue
        if dry_run:
            results[accession] = _skipped_result(record or {}) if done else ExtractionResult([], 0)
            if done:
                logger.info(
                    "would skip %s (already extracted, %d mapped reads); use --force to redo",
                    accession,
                    results[accession].mapped_records,
                )
            continue
        outcome = _map_and_extract(accession, reads, genome_path, output_root / accession, genome_id, preset, threads)
        results[accession] = outcome
        _notify_result(on_result, accession, outcome)
        if outcome.files:
            logger.info(
                "Extracted %d mapped records for %s -> %s",
                outcome.mapped_records,
                accession,
                ", ".join(str(p) for p in outcome.files),
            )

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
