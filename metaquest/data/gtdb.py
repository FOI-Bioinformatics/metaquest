"""GTDB API client for genome taxonomy database queries."""

import logging
from typing import Dict, List

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
        return data.get("results", [data])

    return []


def get_accessions_for_species(
    species_name: str, representative_only: bool = True
) -> List[str]:
    """Get assembly accessions (GCF_*/GCA_*) for a species.

    If representative_only is True, return only the representative genome.
    """
    results = search_species(species_name)
    if not results:
        return []

    accessions = []
    for record in results:
        accession = record.get("accession") or record.get("gid") or record.get("ncbi_accession")
        is_rep = record.get("isRep") or record.get("is_representative", False)

        if accession:
            if representative_only and not is_rep:
                continue
            accessions.append(accession)

    # If representative_only but none flagged as representative, return first
    if representative_only and not accessions and results:
        first = results[0]
        accession = first.get("accession") or first.get("gid") or first.get("ncbi_accession")
        if accession:
            accessions.append(accession)

    return accessions


def get_accessions_for_genus(
    genus_name: str, representative_only: bool = True
) -> List[str]:
    """Get representative accessions for all species in a genus."""
    taxon_results = search_taxon(genus_name)
    if not taxon_results:
        return []

    accessions = []
    seen_species = set()

    for record in taxon_results:
        species = record.get("species") or record.get("name", "")
        if species in seen_species:
            continue
        seen_species.add(species)

        accession = record.get("accession") or record.get("gid") or record.get("ncbi_accession")
        is_rep = record.get("isRep") or record.get("is_representative", False)

        if accession:
            if representative_only and not is_rep:
                continue
            accessions.append(accession)

    # If representative_only yielded nothing, collect all unique accessions
    if representative_only and not accessions:
        for record in taxon_results:
            accession = record.get("accession") or record.get("gid") or record.get("ncbi_accession")
            if accession and accession not in accessions:
                accessions.append(accession)

    return accessions
