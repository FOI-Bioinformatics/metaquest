"""Synthetic project registries for the performance regression tests.

``build_registry`` writes a registry of realistic size through the real ``record_*`` writers,
so a performance test measures the same shapes a real project holds: every dataset screened
against each genome, a subset selected, a smaller subset downloaded with FASTQ files on disk,
and a few extracted and assembled.
"""

import random
from pathlib import Path
from typing import Dict, List, Sequence, Union

from metaquest.data.registry import (
    Registry,
    record_assembly,
    record_download,
    record_extraction,
    record_screening,
    record_selection,
    save_registry,
)


def accession_names(n_datasets: int) -> List[str]:
    """The accessions ``build_registry`` writes, in order."""
    return [f"SRR{index:08d}" for index in range(1, n_datasets + 1)]


def build_registry(
    path: Union[str, Path],
    n_datasets: int = 20000,
    genomes: Sequence[str] = ("G1", "G2", "G3"),
    seed: int = 0,
    n_selected: int = 2000,
    n_downloaded: int = 200,
    n_extracted: int = 5,
) -> Dict[str, List[str]]:
    """Write a registry of ``n_datasets`` screened accessions to ``path`` and return its accession lists.

    Every accession gets one screening entry per genome. The first ``n_selected`` are selected,
    the first ``n_downloaded`` of those are recorded as downloaded with one small FASTQ file each
    under ``<path's folder>/fastq/<acc>/``, and the first ``n_extracted`` are recorded as extracted
    and assembled for the first genome. Returns ``{"all", "selected", "downloaded", "extracted"}``.
    """
    target = Path(path)
    root = target.parent
    rng = random.Random(seed)
    registry = Registry(path=target)
    accessions = accession_names(n_datasets)
    csv_path = root / "matches" / "containment.csv"
    for accession in accessions:
        for genome_id in genomes:
            containment = rng.random()
            record_screening(
                registry, accession, genome_id, containment, containment**0.5, "branchwater", 0.1, csv_path
            )
    selected = accessions[:n_selected]
    record_selection(registry, selected, {"min_containment": 0.1}, root / "selected.txt")
    fastq_dir = root / "fastq"
    downloaded = selected[:n_downloaded]
    for accession in downloaded:
        folder = fastq_dir / accession
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{accession}.fastq.gz").write_bytes(b"\x1f\x8b" + bytes(64))
        record_download(registry, accession, "downloaded", fastq_dir, "Downloaded 1 files")
    extracted = downloaded[:n_extracted]
    genome_id = genomes[0]
    for accession in extracted:
        out = root / "targeted" / accession
        record_extraction(
            registry,
            accession,
            genome_id,
            [out / f"{genome_id}_R1.fastq.gz"],
            100,
            False,
            {"preset": "sr", "threshold": 0.1, "genome_fasta": root / "genomes" / f"{genome_id}.fna"},
        )
        record_assembly(
            registry,
            accession,
            genome_id,
            out / f"{genome_id}_assembly",
            {"contigs": 3, "total_bp": 3000, "n50": 1000, "largest": 1200},
            "1.2.9",
            {"threads": 1},
        )
    save_registry(registry, target)
    return {"all": accessions, "selected": selected, "downloaded": downloaded, "extracted": extracted}
