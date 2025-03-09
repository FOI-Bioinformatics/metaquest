"""
Branchwater format plugin for MetaQuest.

This plugin handles reading and processing Branchwater CSV files.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from metaquest.core.exceptions import ValidationError
from metaquest.core.models import Containment, SRAMetadata
from metaquest.core.validation import validate_accession, validate_containment_value
from metaquest.plugins.base import Plugin

logger = logging.getLogger(__name__)


class BranchWaterFormatPlugin(Plugin):
    """Plugin for handling Branchwater format files."""

    name = "branchwater"
    description = "Handler for Branchwater CSV format files"
    version = "0.1.0"

    # Required columns
    REQUIRED_COLS = ["acc", "containment"]

    # Metadata column mapping
    METADATA_MAPPING = {
        "biosample": "biosample",
        "bioproject": "bioproject",
        "organism": "organism",
        "assay_type": "assay_type",
        "collection_date_sam": "collection_date",
        "geo_loc_name_country_calc": "location",
        "lat_lon": "latitude_longitude",
    }

    @classmethod
    def validate_header(cls, headers: List[str]) -> bool:
        """
        Validate that the headers contain required columns.

        Args:
            headers: List of header column names

        Returns:
            True if valid, False otherwise
        """
        return all(col in headers for col in cls.REQUIRED_COLS)

    @classmethod
    def parse_file(
        cls, file_path: Union[str, Path], genome_id: str
    ) -> List[Containment]:
        """
        Parse a Branchwater format file and return containment data.

        Args:
            file_path: Path to the Branchwater CSV file
            genome_id: Identifier for the genome

        Returns:
            List of Containment objects

        Raises:
            ValidationError: If the file is invalid or cannot be parsed
        """
        containments = []
        file_path = Path(file_path)

        try:
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)

                # Validate headers
                if not cls.validate_header(reader.fieldnames or []):
                    missing = [
                        col
                        for col in cls.REQUIRED_COLS
                        if col not in (reader.fieldnames or [])
                    ]
                    raise ValidationError(
                        f"Missing required columns in {file_path}: {', '.join(missing)}"
                    )

                for row in reader:
                    # Extract and validate accession
                    accession = row.get("acc", "")
                    if not validate_accession(accession):
                        logger.warning(
                            f"Skipping row with invalid accession '{accession}' in {file_path}"
                        )
                        continue

                    # Extract and validate containment value
                    containment_value = validate_containment_value(
                        row.get("containment", "")
                    )
                    if containment_value is None:
                        logger.warning(
                            f"Skipping row with invalid containment value for {accession} in {file_path}"
                        )
                        continue

                    # Create containment object with additional metadata
                    additional_data = {}
                    for key, value in row.items():
                        if key not in ("acc", "containment") and value:
                            additional_data[key] = value

                    containment = Containment(
                        accession=accession,
                        value=containment_value,
                        genome_id=genome_id,
                        additional_data=additional_data,
                    )

                    containments.append(containment)

            return containments

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                f"Error parsing Branchwater file {file_path}: {str(e)}"
            )

    @classmethod
    def extract_metadata(cls, containment: Containment) -> Optional[SRAMetadata]:
        """
        Extract metadata from a Containment object.

        Args:
            containment: Containment object with additional data

        Returns:
            SRAMetadata object or None if insufficient data
        """
        if not containment.additional_data:
            return None

        metadata = SRAMetadata(accession=containment.accession)

        # Map known fields to metadata attributes
        for source_field, target_field in cls.METADATA_MAPPING.items():
            if source_field in containment.additional_data:
                setattr(
                    metadata, target_field, containment.additional_data[source_field]
                )

        # Add all other fields to the attributes dictionary
        for key, value in containment.additional_data.items():
            if key not in cls.METADATA_MAPPING:
                metadata.attributes[key] = value

        return metadata
