"""GTDB API client for genome taxonomy database queries."""

import logging
from typing import Dict, List, Optional

import requests

from metaquest.core.exceptions import DataAccessError

logger = logging.getLogger(__name__)

GTDB_API_BASE = "https://gtdb-api.ecogenomic.org"
REQUEST_TIMEOUT = 30


def search_species(species_name: str) -> List[Dict]:
    """Search GTDB for a species, return list of genome records with accessions."""
    url = f"{GTDB_API_BASE}/species/search/{requests.utils.quote(species_name)}"
    logger.debug("Searching GTDB species: %s", species_name)

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise DataAccessError(f"GTDB API error searching species '{species_name}': {e}")

    if not data:
        logger.warning("No results found for species '%s'", species_name)
        return []

    # The API may return a list or a single object with a genomes array
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("genomes", [data])

    return []


def search_taxon(taxon_name: str, limit: int = 100) -> List[Dict]:
    """Search GTDB by any taxonomic name (genus, family, etc.)."""
    url = f"{GTDB_API_BASE}/taxon/search/{requests.utils.quote(taxon_name)}"
    params = {"limit": limit}
    logger.debug("Searching GTDB taxon: %s (limit=%d)", taxon_name, limit)

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise DataAccessError(f"GTDB API error searching taxon '{taxon_name}': {e}")

    if not data:
        logger.warning("No results found for taxon '%s'", taxon_name)
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "matches" in data:
            return [{"name": str(name)} for name in data["matches"]]
        return data.get("results", [data])

    return []


def _record_accession(record: Dict) -> Optional[str]:
    """Pull the assembly accession from a GTDB record under any of its key spellings."""
    return record.get("accession") or record.get("gid") or record.get("ncbi_accession")


def _is_representative(record: Dict) -> bool:
    """Whether a GTDB record is flagged as a representative genome (any key spelling)."""
    return bool(record.get("isRep") or record.get("is_representative") or record.get("gtdb_species_rep"))


def _keep_accession(record: Dict, representative_only: bool) -> Optional[str]:
    """Return the record's accession if it should be kept under the rep filter, else None."""
    accession = _record_accession(record)
    if accession and (not representative_only or _is_representative(record)):
        return accession
    return None


def get_accessions_for_species(species_name: str, representative_only: bool = True) -> List[str]:
    """Get assembly accessions (GCF_*/GCA_*) for a species.

    If representative_only is True, return only the representative genome.
    """
    results = search_species(species_name)
    if not results:
        return []

    accessions = [acc for record in results if (acc := _keep_accession(record, representative_only))]

    # If representative_only but none flagged as representative, return the first accession
    if representative_only and not accessions:
        first = _record_accession(results[0])
        if first:
            accessions.append(first)

    return accessions


def get_accessions_for_genus(genus_name: str, representative_only: bool = True) -> List[str]:
    """Get representative accessions for all species in a genus.

    The live taxon endpoint returns taxon names only; each ``s__`` species name is
    resolved through the species endpoint. Older record-shaped responses (with an
    accession per record) are still handled.
    """
    taxon_results = search_taxon(genus_name)
    if not taxon_results:
        return []

    accessions: List[str] = []
    seen_species = set()
    attempted = 0
    failed = 0
    for record in taxon_results:
        name = str(record.get("species") or record.get("name", ""))
        if name in seen_species:
            continue
        seen_species.add(name)

        if name.startswith("s__") and _record_accession(record) is None:
            species_name = name[3:]
            logger.debug("Resolving species %s for genus %s", species_name, genus_name)
            attempted += 1
            try:
                accessions.extend(get_accessions_for_species(species_name, representative_only))
            except DataAccessError as e:
                failed += 1
                logger.warning("Skipping species %s: %s", species_name, e)
            continue

        accession = _keep_accession(record, representative_only)
        if accession:
            accessions.append(accession)

    if attempted and failed == attempted:
        raise DataAccessError(f"All {attempted} species lookups for genus '{genus_name}' failed")

    if representative_only and not accessions:
        for record in taxon_results:
            accession = _record_accession(record)
            if accession and accession not in accessions:
                accessions.append(accession)

    return accessions
