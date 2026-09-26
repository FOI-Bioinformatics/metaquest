"""
SRA data handling for MetaQuest.

This package downloads SRA accessions and checks the FASTQ files they produce. It is split by
concern:

- ``fastq``: finding, counting, compressing and verifying FASTQ files already on disk
- ``sampling``: drawing a uniform sample of records from a dataset's FASTQ files
- ``cleanup``: recognising, sizing, preparing and removing temporary download folders
- ``accession``: downloading one accession with prefetch and fasterq-dump
- ``store_handoff``: routing a download through the shared data store
- ``retry``: running many downloads in parallel and retrying the failures
- ``download``: ``download_sra``, the entry point for a whole accession list

The public names other packages import are re-exported here. Private names are imported from
their submodule, and each submodule refers to a sibling's names through the sibling module
(``accession_mod.download_accession``), so a test patches a name on the module that owns it.
"""

from metaquest.data.sra.fastq import (
    COMPLETE_RATIO_THRESHOLD,
    MATE1_SUFFIXES,
    MATE_SUFFIXES,
    FastqDigest,
    accession_has_fastq,
    compress_fastq,
    count_fastq_reads,
    fastq_digest,
    fastq_files,
    fastq_stem,
    iter_fastq_records,
    orphan_fastq,
    parse_verdict_message,
    primary_fastq,
    verify_download,
)
from metaquest.data.sra.sampling import sample_records
from metaquest.data.sra.cleanup import is_transient_folder, transient_bytes
from metaquest.data.sra.accession import STOP, classify_download_error, download_accession, fasterq_dump_version
from metaquest.data.sra.store_handoff import STORE_LINKED_PREFIX, STORE_READY_STATES
from metaquest.data.sra.download import default_max_workers, download_sra

__all__ = [
    "COMPLETE_RATIO_THRESHOLD",
    "FastqDigest",
    "MATE1_SUFFIXES",
    "MATE_SUFFIXES",
    "STOP",
    "STORE_LINKED_PREFIX",
    "STORE_READY_STATES",
    "accession_has_fastq",
    "classify_download_error",
    "compress_fastq",
    "count_fastq_reads",
    "default_max_workers",
    "download_accession",
    "download_sra",
    "fasterq_dump_version",
    "fastq_digest",
    "fastq_files",
    "fastq_stem",
    "iter_fastq_records",
    "is_transient_folder",
    "orphan_fastq",
    "parse_verdict_message",
    "primary_fastq",
    "sample_records",
    "transient_bytes",
    "verify_download",
]
