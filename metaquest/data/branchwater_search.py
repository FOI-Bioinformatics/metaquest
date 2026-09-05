"""Query the Branchwater index of SRA metagenomes with a genome sketch.

The public Branchwater service (https://branchwater.sourmash.bio) indexes about
1.1 million SRA metagenomes as FracMinHash sketches (k=21, scaled=1000). Its
search API takes one sourmash signature and a minimum containment and returns
the matching SRA accessions with their containment. This module builds the
sketch, runs the search and writes the result in the CSV layout the rest of
MetaQuest reads (see ``use_branchwater``). The metadata columns are left empty;
``download_metadata`` fills them from NCBI.
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import requests

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "https://api.branchwater.sourmash.bio"
KSIZE = 21
SCALED = 1000
BRANCHWATER_COLUMNS = [
    "acc",
    "containment",
    "cANI",
    "biosample",
    "bioproject",
    "assay_type",
    "collection_date_sam",
    "geo_loc_name_country_calc",
    "organism",
    "lat_lon",
]
SOURMASH_HINT = "sourmash is required to sketch a genome. Install it with: pip install 'metaquest[sourmash]'"


def sketch_fasta(fasta_path: Union[str, Path]) -> Dict[str, Any]:
    """Build the k=21, scaled=1000 sourmash signature Branchwater expects for one FASTA file.

    Returns the signature as the JSON object sourmash writes (one element of a
    ``.sig`` file), ready to be posted to the search API.

    Raises:
        DataAccessError: If sourmash is not installed, the file is missing, or it holds no sequences.
    """
    try:
        from sourmash import MinHash, SourmashSignature
        from sourmash.signature import save_signatures_to_json
    except ImportError as e:
        raise DataAccessError(SOURMASH_HINT) from e
    from Bio import SeqIO

    path = Path(fasta_path)
    if not path.exists():
        raise DataAccessError(f"Genome FASTA not found: {path}")

    minhash = MinHash(n=0, ksize=KSIZE, scaled=SCALED)
    n_records = 0
    with open(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            minhash.add_sequence(str(record.seq).upper(), force=True)
            n_records += 1
    if n_records == 0:
        raise DataAccessError(f"No sequences found in {path}")

    signature = SourmashSignature(minhash, name=path.stem, filename=path.name)
    buffer = io.BytesIO()
    save_signatures_to_json([signature], buffer)
    payload = json.loads(buffer.getvalue().decode("utf-8"))
    logger.info(
        "Sketched %s: %d sequence(s), %d hashes (k=%d, scaled=%d)",
        path.name,
        n_records,
        len(minhash.hashes),
        KSIZE,
        SCALED,
    )
    return payload[0]


def load_signature(sig_path: Union[str, Path]) -> Dict[str, Any]:
    """Read one sourmash JSON signature (list or single object) and check its sketch parameters."""
    path = Path(sig_path)
    if not path.exists():
        raise DataAccessError(f"Signature file not found: {path}")
    try:
        with open(path) as handle:
            data = json.load(handle)
    except json.JSONDecodeError as e:
        raise DataAccessError(f"Signature file is not valid JSON: {path} ({e})") from e

    signature = data[0] if isinstance(data, list) and data else data
    if not isinstance(signature, dict) or not signature.get("signatures"):
        raise DataAccessError(f"Not a sourmash signature file: {path}")
    _check_sketch_parameters(signature, path)
    return signature


def _check_sketch_parameters(signature: Dict[str, Any], path: Path) -> None:
    sketch = signature["signatures"][0]
    ksize = sketch.get("ksize")
    max_hash = sketch.get("max_hash") or 0
    scaled = round(2**64 / max_hash) if max_hash else None
    if ksize != KSIZE or scaled != SCALED:
        raise DataAccessError(
            f"Branchwater needs k={KSIZE}, scaled={SCALED}; {path.name} has k={ksize}, scaled={scaled}"
        )


def search_index(
    signature: Dict[str, Any], threshold: float, server: str = DEFAULT_SERVER, timeout: int = 600
) -> List[Tuple[str, float]]:
    """Post a signature to the Branchwater search API and return (accession, containment) pairs, best first."""
    url = f"{server.rstrip('/')}/search"
    try:
        response = requests.post(url, json={"threshold": threshold, "signature": signature}, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise DataAccessError(f"Branchwater search failed: {e}") from e
    if response.status_code != 200:
        raise DataAccessError(f"Branchwater search returned HTTP {response.status_code}: {response.text[:200]}")
    return _parse_search_csv(response.text)


def _parse_search_csv(text: str) -> List[Tuple[str, float]]:
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if "SRA accession" not in fields or "containment" not in fields:
        raise DataAccessError(f"Unexpected Branchwater response header: {fields}")
    matches: List[Tuple[str, float]] = []
    for row in reader:
        try:
            matches.append((row["SRA accession"].strip(), float(row["containment"])))
        except (TypeError, ValueError, AttributeError):
            logger.warning("Skipping malformed Branchwater row: %s", row)
    matches.sort(key=lambda match: match[1], reverse=True)
    return matches


def write_branchwater_csv(matches: List[Tuple[str, float]], output_path: Union[str, Path], ksize: int = KSIZE) -> Path:
    """Write matches in Branchwater CSV layout; cANI is derived from containment, metadata columns stay empty."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    empty_metadata = [""] * (len(BRANCHWATER_COLUMNS) - 3)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BRANCHWATER_COLUMNS)
        for accession, containment in matches:
            cani = round(containment ** (1 / ksize), 4) if containment > 0 else 0.0
            writer.writerow([accession, f"{containment:.4f}", f"{cani:.4f}"] + empty_metadata)
    logger.info("Wrote %d match(es) to %s", len(matches), path)
    return path
