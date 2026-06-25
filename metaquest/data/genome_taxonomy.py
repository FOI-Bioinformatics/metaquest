"""Genome taxonomy enrichment via the GTDB API.

This module maps genome accessions (GCF_*/GCA_*) to their taxonomic
classification by querying the GTDB API, and provides utilities for
annotating containment data with taxonomy information.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from metaquest.core.exceptions import DataAccessError
from metaquest.core.models import TaxonomyInfo
from metaquest.data.gtdb import GTDB_API_BASE, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# Mapping from GTDB rank prefixes to TaxonomyInfo field names
_GTDB_RANK_MAP = {
    "d": None,  # domain -- not stored
    "p": "phylum",
    "c": "class_name",
    "o": "order",
    "f": "family",
    "g": "genus",
    "s": "species",
}

_CACHE_COLUMNS = [
    "genome_id",
    "species",
    "genus",
    "family",
    "order",
    "class_name",
    "phylum",
    "organism",
    "tax_id",
]


def parse_gtdb_taxonomy_string(gtdb_string: str, genome_id: str) -> TaxonomyInfo:
    """Parse a GTDB taxonomy string into a TaxonomyInfo dataclass.

    Expects the standard GTDB format, e.g.
    ``d__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Bacillaceae;
    g__Bacillus;s__Bacillus subtilis``
    """
    info = TaxonomyInfo(genome_id=genome_id)
    if not gtdb_string:
        return info

    for token in gtdb_string.split(";"):
        token = token.strip()
        if "__" not in token:
            continue
        prefix, value = token.split("__", 1)
        field = _GTDB_RANK_MAP.get(prefix)
        if field and value:
            setattr(info, field, value)

    return info


def _lookup_genome_taxonomy_gtdb(accession: str) -> Optional[TaxonomyInfo]:
    """Query the GTDB API for a single genome's taxonomy."""
    # Try the genome detail endpoint first
    url = f"{GTDB_API_BASE}/genome/{requests.utils.quote(accession)}"
    logger.debug("Querying GTDB genome endpoint: %s", url)

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            gtdb_taxonomy = data.get("gtdb_taxonomy") or data.get("gtdbTaxonomy") or data.get("taxonomy") or ""
            if gtdb_taxonomy:
                info = parse_gtdb_taxonomy_string(gtdb_taxonomy, accession)
                info.organism = data.get("organism_name") or data.get("organismName")
                info.tax_id = str(data.get("ncbi_taxid") or data.get("taxId") or "")
                if info.tax_id == "":
                    info.tax_id = None
                return info

        # Fallback: search endpoint
        search_url = f"{GTDB_API_BASE}/search/gtdb"
        params = {"search": accession, "page": 1, "itemsPerPage": 1}
        response = requests.get(search_url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            rows = data.get("rows") or data if isinstance(data, list) else []
            if isinstance(data, dict) and not rows:
                rows = data.get("results", [])
            if rows and len(rows) > 0:
                record = rows[0]
                gtdb_taxonomy = (
                    record.get("gtdb_taxonomy") or record.get("gtdbTaxonomy") or record.get("taxonomy") or ""
                )
                if gtdb_taxonomy:
                    info = parse_gtdb_taxonomy_string(gtdb_taxonomy, accession)
                    info.organism = record.get("organism_name") or record.get("organismName")
                    info.tax_id = str(record.get("ncbi_taxid") or record.get("taxId") or "")
                    if info.tax_id == "":
                        info.tax_id = None
                    return info

    except requests.exceptions.RequestException as e:
        raise DataAccessError(f"GTDB API error looking up genome '{accession}': {e}")

    return None


def load_taxonomy_cache(cache_file: Path) -> Dict[str, TaxonomyInfo]:
    """Load a genome-to-taxonomy cache from a TSV file."""
    cache: Dict[str, TaxonomyInfo] = {}
    if not cache_file.exists():
        return cache

    with open(cache_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gid = row.get("genome_id", "")
            if not gid:
                continue
            cache[gid] = TaxonomyInfo(
                genome_id=gid,
                species=row.get("species") or None,
                genus=row.get("genus") or None,
                family=row.get("family") or None,
                order=row.get("order") or None,
                class_name=row.get("class_name") or None,
                phylum=row.get("phylum") or None,
                organism=row.get("organism") or None,
                tax_id=row.get("tax_id") or None,
            )
    logger.info("Loaded %d entries from taxonomy cache %s", len(cache), cache_file)
    return cache


def save_taxonomy_cache(taxonomy: Dict[str, TaxonomyInfo], cache_file: Path) -> None:
    """Save a genome-to-taxonomy mapping to a TSV file."""
    with open(cache_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CACHE_COLUMNS, delimiter="\t")
        writer.writeheader()
        for info in taxonomy.values():
            writer.writerow(
                {
                    "genome_id": info.genome_id,
                    "species": info.species or "",
                    "genus": info.genus or "",
                    "family": info.family or "",
                    "order": info.order or "",
                    "class_name": info.class_name or "",
                    "phylum": info.phylum or "",
                    "organism": info.organism or "",
                    "tax_id": info.tax_id or "",
                }
            )
    logger.info("Saved %d entries to taxonomy cache %s", len(taxonomy), cache_file)


def enrich_genomes_with_taxonomy(
    genome_ids: List[str],
    cache_file: Optional[Path] = None,
) -> Dict[str, TaxonomyInfo]:
    """Map genome accessions to taxonomy via the GTDB API.

    Loads cached results when available and saves new lookups back to the
    cache file after completion.
    """
    taxonomy: Dict[str, TaxonomyInfo] = {}

    if cache_file:
        taxonomy = load_taxonomy_cache(cache_file)

    missing = [gid for gid in genome_ids if gid not in taxonomy]
    if missing:
        logger.info(
            "Enriching %d genome(s) with taxonomy (%d cached)",
            len(missing),
            len(taxonomy),
        )
        for accession in missing:
            try:
                info = _lookup_genome_taxonomy_gtdb(accession)
                if info:
                    taxonomy[accession] = info
                else:
                    logger.warning("No taxonomy found for genome '%s'", accession)
                    taxonomy[accession] = TaxonomyInfo(genome_id=accession)
            except DataAccessError:
                logger.warning("Failed to retrieve taxonomy for '%s'", accession)
                taxonomy[accession] = TaxonomyInfo(genome_id=accession)

        if cache_file:
            save_taxonomy_cache(taxonomy, cache_file)

    return taxonomy


def annotate_containment_with_taxonomy(
    containment_df: pd.DataFrame,
    taxonomy: Dict[str, TaxonomyInfo],
) -> pd.DataFrame:
    """Add taxonomy columns to a parsed containment DataFrame.

    Converts the wide-format containment table (samples as rows, genomes as
    columns) into a long-format DataFrame with columns:
    ``sample, genome, containment, species, genus, family``.
    """
    from metaquest.core.utils import get_genome_columns

    genome_cols = get_genome_columns(containment_df)
    records = []
    for _, row in containment_df.iterrows():
        sample = row.name if isinstance(row.name, str) else str(row.name)
        for genome in genome_cols:
            containment_val = row[genome]
            info = taxonomy.get(genome, TaxonomyInfo(genome_id=genome))
            records.append(
                {
                    "sample": sample,
                    "genome": genome,
                    "containment": float(containment_val),
                    "species": info.species,
                    "genus": info.genus,
                    "family": info.family,
                }
            )

    return pd.DataFrame(records)


def filter_by_taxonomy(
    annotated_df: pd.DataFrame,
    family: Optional[str] = None,
    genus: Optional[str] = None,
    species: Optional[str] = None,
    min_containment: float = 0.0,
) -> pd.DataFrame:
    """Filter an annotated containment DataFrame by taxonomy criteria."""
    df = annotated_df.copy()

    if family:
        df = df[df["family"].str.lower() == family.lower()]
    if genus:
        df = df[df["genus"].str.lower() == genus.lower()]
    if species:
        df = df[df["species"].str.lower() == species.lower()]
    if min_containment > 0:
        df = df[df["containment"] >= min_containment]

    return df


def summarize_by_taxonomy(
    annotated_df: pd.DataFrame,
    level: str = "family",
) -> pd.DataFrame:
    """Aggregate containment by taxonomic level.

    For each sample, computes the maximum containment per taxonomic group
    and returns a pivot table with samples as rows and taxonomy groups as
    columns.
    """
    valid_levels = {"family", "genus", "species"}
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}, got '{level}'")

    df = annotated_df.dropna(subset=[level])
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["sample", level])["containment"].max().reset_index()
    pivot = grouped.pivot_table(index="sample", columns=level, values="containment", fill_value=0.0)
    return pivot
