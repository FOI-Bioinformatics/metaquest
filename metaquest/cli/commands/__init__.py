"""
CLI commands package.

This package contains modular CLI command implementations.
"""

from .branchwater import UseBranchwaterCommand, ExtractBranchwaterMetadataCommand
from .containment import ParseContainmentCommand, PlotContainmentCommand
from .metadata import (
    DownloadMetadataCommand,
    ParseMetadataCommand,
    CountMetadataCommand,
    PlotMetadataCountsCommand,
)
from .samples import SingleSampleCommand
from .sra import DownloadSraCommand, AssembleDatasetsCommand
from .test_data import DownloadTestGenomeCommand

__all__ = [
    "UseBranchwaterCommand",
    "ExtractBranchwaterMetadataCommand",
    "ParseContainmentCommand",
    "PlotContainmentCommand",
    "DownloadMetadataCommand",
    "ParseMetadataCommand",
    "CountMetadataCommand",
    "PlotMetadataCountsCommand",
    "SingleSampleCommand",
    "DownloadSraCommand",
    "AssembleDatasetsCommand",
    "DownloadTestGenomeCommand",
]
