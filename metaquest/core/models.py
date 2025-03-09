"""
Data models for MetaQuest.

This module contains dataclasses that define the core data structures used throughout
the application.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import datetime


@dataclass
class Containment:
    """Represents a containment relationship between a genome and an SRA accession."""
    accession: str
    value: float
    genome_id: str
    additional_data: Dict[str, Union[str, float, int]] = field(default_factory=dict)


@dataclass
class GenomeInfo:
    """Represents information about a genome."""
    genome_id: str
    path: Optional[Path] = None
    description: Optional[str] = None
    source: Optional[str] = None
    date_added: Optional[datetime.datetime] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SRAMetadata:
    """Represents metadata for an SRA sample."""
    accession: str
    biosample: Optional[str] = None
    bioproject: Optional[str] = None
    organism: Optional[str] = None
    collection_date: Optional[str] = None
    location: Optional[str] = None
    latitude_longitude: Optional[str] = None
    assay_type: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    
    @property
    def country(self) -> Optional[str]:
        """Extract country from location if available."""
        if self.location and ":" in self.location:
            return self.location.split(":")[0]
        return self.location


@dataclass
class ContainmentSummary:
    """Summary of containment data for analysis."""
    thresholds: List[float] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    max_containment: Dict[str, float] = field(default_factory=dict)
    genome_to_samples: Dict[str, List[str]] = field(default_factory=dict)
    sample_to_genomes: Dict[str, List[str]] = field(default_factory=dict)