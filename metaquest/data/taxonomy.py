"""
Taxonomy integration and validation for MetaQuest.

This module provides functions for integrating with NCBI taxonomy,
validating taxonomic assignments, and creating taxonomic summaries.
"""

import logging
import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import re

from metaquest.core.exceptions import ProcessingError

logger = logging.getLogger(__name__)


class NCBITaxonomyClient:
    """Client for interacting with NCBI Taxonomy database."""

    def __init__(self, email: str, api_key: Optional[str] = None):
        """
        Initialize NCBI client.

        Args:
            email: Email address for NCBI API access
            api_key: Optional API key for increased rate limits
        """
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.last_request_time = 0
        self.request_delay = 1.0 / 3 if api_key else 1.0 / 10  # Rate limiting

    def _make_request(self, url: str, params: Dict[str, str]) -> str:
        """Make rate-limited request to NCBI API."""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)

        # Add required parameters
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.text
        except requests.RequestException as e:
            logger.error(f"NCBI API request failed: {e}")
            raise ProcessingError(f"Failed to query NCBI: {e}")

    def search_taxonomy(self, query: str) -> List[Dict[str, str]]:
        """
        Search for taxonomic entries by name.

        Args:
            query: Scientific name or partial name to search

        Returns:
            List of matching taxonomy entries
        """
        search_url = f"{self.base_url}esearch.fcgi"
        params = {"db": "taxonomy", "term": query, "retmax": "10", "retmode": "xml"}

        response = self._make_request(search_url, params)

        # Parse XML response
        root = ET.fromstring(response)
        tax_ids = [id_elem.text for id_elem in root.findall(".//Id")]

        if not tax_ids:
            return []

        # Get details for found IDs
        return self.get_taxonomy_details(tax_ids[:5])  # Limit to first 5 results

    def get_taxonomy_details(self, tax_ids: List[str]) -> List[Dict[str, str]]:
        """
        Get detailed taxonomy information for given taxonomy IDs.

        Args:
            tax_ids: List of NCBI taxonomy IDs

        Returns:
            List of taxonomy detail dictionaries
        """
        if not tax_ids:
            return []

        fetch_url = f"{self.base_url}efetch.fcgi"
        params = {"db": "taxonomy", "id": ",".join(tax_ids), "retmode": "xml"}

        response = self._make_request(fetch_url, params)

        # Parse XML response
        root = ET.fromstring(response)
        results = []

        for taxon in root.findall("./Taxon"):
            tax_id = taxon.find("TaxId").text if taxon.find("TaxId") is not None else ""
            sci_name = (
                taxon.find("ScientificName").text
                if taxon.find("ScientificName") is not None
                else ""
            )
            rank = taxon.find("Rank").text if taxon.find("Rank") is not None else ""

            # Get lineage
            lineage = []
            lineage_elem = taxon.find(".//LineageEx")
            if lineage_elem is not None:
                for taxon_elem in lineage_elem.findall("Taxon"):
                    name = taxon_elem.find("ScientificName")
                    rank_elem = taxon_elem.find("Rank")
                    if name is not None and rank_elem is not None:
                        lineage.append(f"{rank_elem.text}:{name.text}")

            results.append(
                {
                    "tax_id": tax_id,
                    "scientific_name": sci_name,
                    "rank": rank,
                    "lineage": ";".join(lineage) if lineage else "",
                }
            )

        return results

    def validate_species_name(self, species_name: str) -> Dict[str, Union[str, bool]]:
        """
        Validate a species name against NCBI taxonomy.

        Args:
            species_name: Species name to validate

        Returns:
            Dictionary with validation results
        """
        try:
            # Clean the species name
            cleaned_name = self._clean_species_name(species_name)

            # Search for exact match first
            results = self.search_taxonomy(f'"{cleaned_name}"[Scientific Name]')

            if not results:
                # Try broader search
                results = self.search_taxonomy(cleaned_name)

            if results:
                best_match = results[0]  # Take first/best result
                return {
                    "original_name": species_name,
                    "cleaned_name": cleaned_name,
                    "validated_name": best_match["scientific_name"],
                    "tax_id": best_match["tax_id"],
                    "rank": best_match["rank"],
                    "lineage": best_match["lineage"],
                    "is_valid": True,
                    "confidence": (
                        "high"
                        if cleaned_name.lower() == best_match["scientific_name"].lower()
                        else "medium"
                    ),
                }
            else:
                return {
                    "original_name": species_name,
                    "cleaned_name": cleaned_name,
                    "validated_name": "",
                    "tax_id": "",
                    "rank": "",
                    "lineage": "",
                    "is_valid": False,
                    "confidence": "low",
                }

        except Exception as e:
            logger.warning(f"Failed to validate species name '{species_name}': {e}")
            return {
                "original_name": species_name,
                "cleaned_name": species_name,
                "validated_name": "",
                "tax_id": "",
                "rank": "",
                "lineage": "",
                "is_valid": False,
                "confidence": "error",
            }

    def _clean_species_name(self, name: str) -> str:
        """Clean and standardize species name for taxonomy lookup."""
        # Remove common prefixes/suffixes
        cleaned = re.sub(
            r"^(candidate|unclassified|uncultured)\s+", "", name, flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"\s+(sp\.|strain|isolate|clone)\s+.*$", "", cleaned, flags=re.IGNORECASE
        )

        # Remove brackets and their contents
        cleaned = re.sub(r"\[.*?\]", "", cleaned)
        cleaned = re.sub(r"\(.*?\)", "", cleaned)

        # Standardize whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned.strip()


def validate_taxonomic_assignments(
    species_list: List[str],
    email: str,
    api_key: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    cache_file: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Validate a list of species names against NCBI taxonomy.

    Args:
        species_list: List of species names to validate
        email: Email for NCBI API access
        api_key: Optional API key
        output_file: File to save validation results
        cache_file: File to cache results for reuse

    Returns:
        DataFrame with validation results
    """
    try:
        # Load cache if available
        cached_results = {}
        if cache_file and Path(cache_file).exists():
            try:
                cache_df = pd.read_csv(cache_file)
                cached_results = dict(
                    zip(cache_df["original_name"], cache_df.to_dict("records"))
                )
                logger.info(f"Loaded {len(cached_results)} cached taxonomy validations")
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")

        # Initialize client
        client = NCBITaxonomyClient(email, api_key)

        # Validate species
        results = []
        unique_species = list(set(species_list))

        logger.info(f"Validating {len(unique_species)} unique species names")

        for i, species in enumerate(unique_species, 1):
            if species in cached_results:
                results.append(cached_results[species])
                logger.debug(f"Using cached result for {species}")
            else:
                logger.info(f"Validating {i}/{len(unique_species)}: {species}")
                result = client.validate_species_name(species)
                results.append(result)

                # Rate limiting for API calls
                if i % 10 == 0:
                    logger.info(
                        f"Progress: {i}/{len(unique_species)} species validated"
                    )

        # Create results DataFrame
        if results:
            results_df = pd.DataFrame(results)
        else:
            # Create empty DataFrame with expected columns
            results_df = pd.DataFrame(
                columns=[
                    "original_name",
                    "validated_name",
                    "is_valid",
                    "tax_id",
                    "rank",
                    "lineage",
                    "confidence",
                ]
            )

        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Taxonomy validation results saved to {output_path}")

        # Update cache
        if cache_file:
            cache_path = Path(cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(cache_path, index=False)
            logger.info(f"Taxonomy validation cache updated: {cache_path}")

        # Log summary
        valid_count = results_df["is_valid"].sum() if not results_df.empty else 0
        logger.info(
            f"Taxonomy validation complete: {valid_count}/{len(results_df)} species validated"  # noqa: E501
        )

        return results_df

    except Exception as e:
        raise ProcessingError(f"Failed to validate taxonomic assignments: {e}")


def create_taxonomic_summary(
    abundance_data: pd.DataFrame,
    taxonomy_data: pd.DataFrame,
    level: str = "genus",
    min_abundance: float = 0.001,
) -> pd.DataFrame:
    """
    Create taxonomic summary at specified level.

    Args:
        abundance_data: Sample x Species abundance matrix
        taxonomy_data: Taxonomy information with lineage
        level: Taxonomic level for summary (genus, family, order, etc.)
        min_abundance: Minimum abundance threshold

    Returns:
        Sample x Taxon summary matrix
    """
    try:
        # Parse taxonomic lineages
        taxonomy_dict = {}
        for _, row in taxonomy_data.iterrows():
            if row["is_valid"] and row["lineage"]:
                lineage_parts = row["lineage"].split(";")
                lineage_dict = {}

                for part in lineage_parts:
                    if ":" in part:
                        rank, name = part.split(":", 1)
                        lineage_dict[rank.lower()] = name

                # Map by original_name (which should match abundance data columns)
                taxonomy_dict[row["original_name"]] = lineage_dict

        # Create summary matrix
        summary_data = []

        for sample_name in abundance_data.index:
            sample_summary = {}

            for species, abundance in abundance_data.loc[sample_name].items():
                if abundance < min_abundance:
                    continue

                # Get taxonomic assignment
                if species in taxonomy_dict and level in taxonomy_dict[species]:
                    taxon_name = taxonomy_dict[species][level]

                    if taxon_name in sample_summary:
                        sample_summary[taxon_name] += abundance
                    else:
                        sample_summary[taxon_name] = abundance
                else:
                    # Unclassified
                    unclassified = f"Unclassified_{level}"
                    if unclassified in sample_summary:
                        sample_summary[unclassified] += abundance
                    else:
                        sample_summary[unclassified] = abundance

            summary_data.append(sample_summary)

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data, index=abundance_data.index)
        summary_df = summary_df.fillna(0)

        logger.info(f"Created taxonomic summary at {level} level: {summary_df.shape}")
        return summary_df

    except Exception as e:
        raise ProcessingError(f"Failed to create taxonomic summary: {e}")


def analyze_taxonomic_composition(
    abundance_data: pd.DataFrame,
    taxonomy_data: pd.DataFrame,
    levels: List[str] = ["phylum", "class", "order", "family", "genus"],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze taxonomic composition at multiple levels.

    Args:
        abundance_data: Sample x Species abundance matrix
        taxonomy_data: Taxonomy information
        levels: List of taxonomic levels to analyze
        output_dir: Directory to save results

    Returns:
        Dictionary of summary matrices for each level
    """
    try:
        results = {}

        for level in levels:
            logger.info(f"Creating taxonomic summary at {level} level")
            summary_df = create_taxonomic_summary(abundance_data, taxonomy_data, level)
            results[level] = summary_df

            if output_dir:
                output_path = Path(output_dir) / f"taxonomy_summary_{level}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                summary_df.to_csv(output_path)
                logger.info(f"Saved {level} summary to {output_path}")

        return results

    except Exception as e:
        raise ProcessingError(f"Failed to analyze taxonomic composition: {e}")


def suggest_species_corrections(
    validation_results: pd.DataFrame, confidence_threshold: str = "medium"
) -> pd.DataFrame:
    """
    Suggest corrections for invalid species names.

    Args:
        validation_results: Results from validate_taxonomic_assignments
        confidence_threshold: Minimum confidence for suggestions

    Returns:
        DataFrame with suggested corrections
    """
    try:
        # Filter invalid species
        invalid_species = validation_results[~validation_results["is_valid"]]

        if len(invalid_species) == 0:
            logger.info("No invalid species names found")
            return pd.DataFrame()

        suggestions = []
        confidence_levels = {"low": 0, "medium": 1, "high": 2}
        # min_confidence = confidence_levels.get(confidence_threshold, 1)  # For future use

        for _, row in invalid_species.iterrows():
            # For now, suggest manual review
            # In future versions, could implement fuzzy matching
            suggestions.append(
                {
                    "original_name": row["original_name"],
                    "suggested_correction": "Manual review required",
                    "confidence": "low",
                    "notes": "Species name not found in NCBI taxonomy",
                }
            )

        suggestions_df = pd.DataFrame(suggestions)
        logger.info(
            f"Generated {len(suggestions_df)} species name correction suggestions"
        )

        return suggestions_df

    except Exception as e:
        raise ProcessingError(f"Failed to generate species corrections: {e}")
