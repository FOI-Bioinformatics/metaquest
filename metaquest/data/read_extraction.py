"""Targeted read extraction prior to assembly.

Given a parsed containment table (samples x target genomes), this module maps
each sample's reads against a chosen target genome and keeps only the reads that
align. The resulting reduced FASTQ set supports a small, targeted assembly rather
than a whole-metagenome assembly.

The mapping uses minimap2 (one aligner for both short and long reads) and samtools
to filter and export the mapped reads. External tools run through
``SecureSubprocess.run_secure`` and must be present on the system.
"""

import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Union

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
) -> List[Path]:
    """Map one sample's reads to the target genome and write the mapped reads.

    Returns the list of FASTQ files written (two for paired input, one otherwise).
    Intermediate SAM/BAM files are removed on success.
    """
    ensure_directory(out_dir)
    sam_path = out_dir / f"{genome_id}.sam"
    bam_path = out_dir / f"{genome_id}.mapped.bam"

    # 1. Align reads to the reference (SAM output).
    minimap_args = ["-a", "-x", preset, "-t", str(threads), "-o", str(sam_path), str(genome_fasta)]
    minimap_args.extend(str(r) for r in reads)
    SecureSubprocess.run_secure("minimap2", minimap_args)

    # 2. Keep only mapped records (-F 4 drops the unmapped flag).
    SecureSubprocess.run_secure(
        "samtools", ["view", "-b", "-F", "4", "-@", str(threads), "-o", str(bam_path), str(sam_path)]
    )

    # 3. Export mapped reads back to FASTQ.
    if len(reads) >= 2:
        out1 = out_dir / f"{genome_id}_1.fastq.gz"
        out2 = out_dir / f"{genome_id}_2.fastq.gz"
        singles = out_dir / f"{genome_id}_s.fastq.gz"
        SecureSubprocess.run_secure(
            "samtools", ["fastq", "-1", str(out1), "-2", str(out2), "-s", str(singles), str(bam_path)]
        )
        written = [out1, out2]
    else:
        out0 = out_dir / f"{genome_id}.fastq.gz"
        SecureSubprocess.run_secure("samtools", ["fastq", "-0", str(out0), str(bam_path)])
        written = [out0]

    # Remove intermediates; the reduced FASTQ set is the deliverable.
    for tmp in (sam_path, bam_path):
        tmp.unlink(missing_ok=True)

    return written


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
) -> Dict[str, List[Path]]:
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

    Returns:
        Mapping of accession to the list of FASTQ files written (empty in dry-run).

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

    results: Dict[str, List[Path]] = {}
    output_root = Path(output_folder)

    for accession in samples:
        reads = _sample_reads(fastq_root, accession)
        if not reads:
            logger.warning("No FASTQ files found for %s under %s; skipping", accession, fastq_root)
            continue
        if dry_run:
            results[accession] = []
            continue
        written = _map_and_extract(accession, reads, genome_path, output_root / accession, genome_id, preset, threads)
        results[accession] = written
        logger.info("Extracted mapped reads for %s -> %s", accession, ", ".join(str(p) for p in written))

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
) -> Path:
    """Assemble a set of extracted FASTQ files with megahit.

    Single-end input (one file) uses megahit ``-r``; paired input (two files) uses
    ``-1``/``-2``. The reads are expected to be a small, target-filtered set.

    Args:
        reads: One or two FASTQ files to assemble.
        output_dir: megahit output directory (must not already exist).
        threads: CPU threads.
        min_contig_len: Optional minimum contig length.

    Returns:
        Path to the megahit output directory.

    Raises:
        ProcessingError: If the number of reads is unsupported.
    """
    out_dir = Path(output_dir)
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
    return out_dir
