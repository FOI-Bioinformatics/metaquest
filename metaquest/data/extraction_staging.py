"""Staging for one sample's extracted FASTQ files, so an interrupted sample leaves nothing visible.

``samtools fastq`` writes a sample's mapped reads into a dot-prefixed folder inside the
sample's output folder (``targeted/<ACC>/.<genome>.<host>.<pid>.<token>.tmp/``). Only once the
sample is complete are the files renamed into ``targeted/<ACC>/``, one rename per file. Every
folder listing (``visible_files``, and so ``scan_extractions`` and ``status --reconcile``)
skips dot-prefixed names, so a sample stopped by a signal, a timeout or a kill never leaves a
partial FASTQ that a later scan could record as a finished extraction.

``require_nonempty`` guards a staged output a tool reported as written (the minimap2 index)
against being published empty.
"""

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

from metaquest.core.exceptions import ProcessingError
from metaquest.data.file_io import unique_temp_path

logger = logging.getLogger(__name__)

__all__ = ["extracted_names", "publish_staged", "require_nonempty", "staged_sample_outputs"]


def require_nonempty(path: Path, what: str) -> None:
    """Raise ``ProcessingError`` when ``path``, which a tool reported as written, is empty.

    Called on a staged output before it is published, so a tool that exits 0 without writing
    (or after a full disk truncated its output) never leaves an empty file under the final name.
    """
    if Path(path).stat().st_size == 0:
        raise ProcessingError(f"{what} is empty although the tool reported success: {path}")


# Mate suffixes ``_export_mapped_fastq`` writes for paired input; single-end input has none.
_PAIRED_SUFFIXES = ("_1", "_2", "_s", "_0")


def extracted_names(genome_id: str) -> List[str]:
    """Every FASTQ name one sample's extraction for ``genome_id`` can write."""
    return [f"{genome_id}.fastq.gz"] + [f"{genome_id}{suffix}.fastq.gz" for suffix in _PAIRED_SUFFIXES]


@contextmanager
def staged_sample_outputs(out_dir: Path, genome_id: str) -> Iterator[Path]:
    """A new dot-prefixed folder inside ``out_dir`` for one sample's FASTQ export.

    The folder and whatever is left in it are removed when the block ends, whether it
    finished or was interrupted; files published with ``publish_staged`` have already left it.
    """
    staging = unique_temp_path(Path(out_dir) / genome_id)
    staging.mkdir(parents=True)
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def publish_staged(staging: Path, returned: List[Path], out_dir: Path, genome_id: str) -> List[Path]:
    """Rename every file left in ``staging`` into ``out_dir``; return the final paths of ``returned``.

    Every non-empty file the export kept is published, one rename per file, as the export
    used to leave them in place; ``returned`` (the files the caller reports) are all among
    them. A file an earlier run left under one of ``genome_id``'s other extraction names is
    removed first, as the export used to overwrite or remove it, so a rerun that now writes
    fewer files never leaves an older run's reads beside the new ones.
    """
    out_dir = Path(out_dir)
    staged = sorted(path for path in Path(staging).iterdir() if path.is_file())
    names = {path.name for path in staged}
    for name in extracted_names(genome_id):
        final = out_dir / name
        if name not in names and final.exists():
            logger.debug("Removing %s, left by an earlier extraction", final)
            final.unlink()
    for path in staged:
        os.replace(path, out_dir / path.name)
    return [out_dir / path.name for path in returned]
