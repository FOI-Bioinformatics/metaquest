"""Synthetic containment data for the containment summary and screening tests.

``containment_data`` returns the accession -> genome -> containment mapping that
``_process_genome_containments`` builds from the match CSVs. Values are drawn from a small set of
two-decimal numbers, so ties are common, and about half the cells are 0. The last genome can be
made all zeros, which the summary and the registry both have to carry without a positive entry.
"""

from typing import Dict

import numpy as np


def containment_data(rows: int, genomes: int, seed: int = 7, zero_column: bool = True) -> Dict[str, Dict[str, float]]:
    """``rows`` accessions by ``genomes`` genomes of containment values, with ties and zeros.

    Accessions are ``SRR<n>`` and genomes ``GCF_<n>``. With ``zero_column`` the last genome is 0 for
    every accession. The same arguments always give the same values.
    """
    rng = np.random.default_rng(seed)
    levels = np.round(np.linspace(0.05, 1.0, 20), 2)
    values = rng.choice(levels, size=(rows, genomes))
    values[rng.random((rows, genomes)) < 0.5] = 0.0
    if zero_column:
        values[:, -1] = 0.0
    names = [f"GCF_{j:03d}" for j in range(genomes)]
    return {f"SRR{1000000 + i}": {names[j]: float(values[i, j]) for j in range(genomes)} for i in range(rows)}
