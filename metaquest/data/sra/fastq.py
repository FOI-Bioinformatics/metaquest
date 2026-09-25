"""FASTQ files of a downloaded SRA accession: discovery, read counting, compression and verification.

Every function here reads or rewrites files already on disk; none of them starts a download.
"""

import gzip
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from metaquest.core.constants import FASTQ_GLOBS
from metaquest.core.exceptions import SecurityError
from metaquest.data.file_io import visible_files
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


# What SecureSubprocess.run_secure raises for a tool that fails, times out, is refused or
# cannot be started.
_TOOL_ERRORS = (subprocess.CalledProcessError, subprocess.TimeoutExpired, SecurityError, OSError)


# Ratio of downloaded reads to NCBI's recorded run_total_spots at or above which a download
# counts as complete rather than truncated.
COMPLETE_RATIO_THRESHOLD = 0.99

# FASTQ file-name suffixes marking a mate of a pair: fasterq-dump's --split-3 output uses
# _1/_2, while files from other sources often use _R1/_R2.
MATE1_SUFFIXES = ("_1", "_R1")
MATE_SUFFIXES = ("_1", "_2", "_R1", "_R2")


def fastq_files(acc_dir: Union[str, Path]) -> List[Path]:
    """Non-empty, visible FASTQ files directly in ``acc_dir``, sorted by name (see ``visible_files``).

    Matches ``FASTQ_GLOBS`` (plain and gzipped ``.fastq``/``.fq``). A directory that does not
    exist (or a dangling symlink) yields an empty list; a symlinked directory is followed
    since ``Path.is_dir``/``Path.glob`` already resolve it transparently. A zero-byte file
    (e.g. left behind by an interrupted download) is never returned, and neither is a hidden
    name such as the ``._<name>`` AppleDouble files macOS writes next to every file on a
    volume without native extended attributes.
    """
    found = []
    for candidate in visible_files(acc_dir, *FASTQ_GLOBS):
        try:
            if candidate.stat().st_size > 0:
                found.append(candidate)
        except OSError:
            continue
    return found


def fastq_stem(path: Union[str, Path]) -> str:
    """The file name of ``path`` without its FASTQ extension (``.fastq``/``.fq``, plain or gzipped)."""
    name = Path(path).name
    for suffix in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def primary_fastq(acc_dir: Union[str, Path]) -> Optional[Path]:
    """The FASTQ file in ``acc_dir`` holding one record per sequenced spot, or None if there is none.

    A mate-1 file (``_1``/``_R1``) is preferred, since for paired data it holds exactly one
    record per spot; otherwise the bare ``<acc>.fastq`` file written for single-end data.
    Sorting by name is not enough here: ``<acc>.fastq`` sorts before ``<acc>_1.fastq``, and
    for paired data that bare file holds only the unpaired leftovers of ``--split-3``.
    """
    files = fastq_files(acc_dir)
    if not files:
        return None
    mate_one = next((p for p in files if fastq_stem(p).endswith(MATE1_SUFFIXES)), None)
    if mate_one is not None:
        return mate_one
    bare = next((p for p in files if not fastq_stem(p).endswith(MATE_SUFFIXES)), None)
    return bare if bare is not None else files[0]


def orphan_fastq(acc_dir: Union[str, Path]) -> Optional[Path]:
    """The bare ``<acc>.fastq`` file of unpaired spots ``--split-3`` writes beside a mate pair.

    Returns None when the folder holds no mate-1 file, since then the bare file is the
    single-end data itself rather than a set of leftovers (``primary_fastq`` returns it).
    """
    files = fastq_files(acc_dir)
    if not any(fastq_stem(p).endswith(MATE1_SUFFIXES) for p in files):
        return None
    return next((p for p in files if not fastq_stem(p).endswith(MATE_SUFFIXES)), None)


def accession_has_fastq(acc_dir: Union[str, Path]) -> bool:
    """Return True if the per-accession directory holds at least one usable FASTQ file.

    This is the single source of truth for "this accession is already
    downloaded" used across the download and status paths. A sidecar
    ``<acc_dir>/<acc_dir.name>.json`` recording an in-progress or failed state
    (``partial``, ``failed`` or ``downloading``) overrides an otherwise
    present-looking directory, so a half-written or restarted download is not
    mistaken for a finished one.
    """
    acc_path = Path(acc_dir)
    if not fastq_files(acc_path):
        return False

    sidecar = acc_path / f"{acc_path.name}.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            data = None
        # A sidecar that is unreadable, not JSON, or not a JSON object carries no state.
        state = data.get("state") if isinstance(data, dict) else None
        if state in ("partial", "failed", "downloading"):
            return False

    return True


def count_fastq_reads(path: Union[str, Path]) -> int:
    """Count FASTQ records in ``path`` via a chunked binary newline count (4 lines/record).

    Reads in 1 MiB blocks so a large FASTQ file is never loaded into memory, and counts
    raw ``b"\\n"`` bytes rather than decoding text, which is both faster and immune to
    encoding errors in a corrupted file. Works for both gzip-compressed and plain files.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    block_size = 1024 * 1024
    total_newlines = 0
    last_byte = b""
    with opener(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            total_newlines += block.count(b"\n")
            last_byte = block[-1:]
    # A file whose last line has no trailing newline still ends a record; count it too.
    if last_byte and last_byte != b"\n":
        total_newlines += 1
    return total_newlines // 4


def iter_fastq_records(path: Union[str, Path]):
    """Yield ``(sequence, quality)`` string pairs for each record in ``path``, streaming.

    Works for both gzip-compressed and plain files. This is the one raw four-line reader
    shared by every caller that needs read-level content without Biopython's slower
    per-record parser: ``metaquest.store.stats.compute_dataset_stats``'s reservoir sampler,
    ``metaquest.data.sra_metadata.calculate_read_statistics``'s streaming pass, and
    ``metaquest.sra.analytics.SequenceQualityAnalyzer``'s uniform sampler.

    Raises ``ValueError`` when a header line is not followed by a complete
    sequence/plus/quality triplet, since a truncated trailing record cannot be trusted.
    Truncation is decided on the raw lines: a line that is missing entirely (``readline``
    returns the empty string at end of file) ends the record early, whereas a present but
    empty line does not. A record whose sequence and quality lines are both empty is a
    zero-length read, which is legal FASTQ (a read trimmed to nothing, or the empty mate
    ``fasterq-dump --split-files`` writes for a half-empty spot) and is yielded as such.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            seq_line = handle.readline()
            plus_line = handle.readline()
            qual_line = handle.readline()
            if not seq_line or not plus_line or not qual_line:
                raise ValueError(f"Truncated FASTQ record after header: {header.strip()!r}")
            if not plus_line.startswith("+"):
                raise ValueError(f"Malformed FASTQ record after header: {header.strip()!r}")
            yield seq_line.rstrip("\r\n"), qual_line.rstrip("\r\n")


def verify_download(
    accession: str,
    acc_dir: Union[str, Path],
    expected_spots: Optional[int],
) -> Dict[str, Any]:
    """Compare what actually downloaded for ``accession`` against NCBI's recorded spot count.

    ``reads_r1`` counts one record per sequenced spot: the mate-1 file (or the single-end
    file) plus the bare ``<acc>.fastq`` file of unpaired spots that ``--split-3`` writes
    beside a mate pair, since NCBI's ``total_spots`` covers those too. The verdict is
    ``"complete"`` when the ratio of downloaded reads to ``expected_spots`` is at least
    ``COMPLETE_RATIO_THRESHOLD``, ``"truncated"`` below that, and ``"unverified"`` when
    ``expected_spots`` is unknown (e.g. NCBI metadata was never fetched for this accession).
    """
    files = fastq_files(acc_dir)
    primary = primary_fastq(acc_dir)
    orphan = orphan_fastq(acc_dir)
    reads_r1 = count_fastq_reads(primary) if primary is not None else 0
    if orphan is not None:
        reads_r1 += count_fastq_reads(orphan)
    bytes_total = sum(p.stat().st_size for p in files)

    ratio: Optional[float]
    if expected_spots:
        ratio = round(reads_r1 / expected_spots, 4)
        verdict = "complete" if ratio >= COMPLETE_RATIO_THRESHOLD else "truncated"
    else:
        ratio = None
        verdict = "unverified"

    return {
        "reads_r1": reads_r1,
        "expected_spots": expected_spots,
        "ratio": ratio,
        "verdict": verdict,
        "bytes_total": bytes_total,
    }


_VERDICT_MESSAGE_RE = re.compile(r", (complete|truncated) \((\d+) of (\d+) spots\)")


def parse_verdict_message(message: str) -> Optional[Dict[str, Any]]:
    """Recover the verdict dict encoded in a download result message by ``_handle_download_output``.

    The ``on_result`` callback contract only carries a plain message string across the
    download/registry boundary, so the CLI recovers the verdict from it rather than the data
    layer reaching into the registry directly. Returns ``None`` for a message that carries no
    verdict (a failure message, or "already exists").

    Both patterns are anchored on the ", <verdict>" separator ``_handle_download_output``
    writes, so a failure message that merely contains one of these words is not read as a
    verdict.
    """
    match = _VERDICT_MESSAGE_RE.search(message)
    if match:
        verdict, reads_r1, expected_spots = match.group(1), int(match.group(2)), int(match.group(3))
        ratio = round(reads_r1 / expected_spots, 4) if expected_spots else 0.0
        return {"verdict": verdict, "reads_r1": reads_r1, "expected_spots": expected_spots, "ratio": ratio}
    if ", unverified" in message:
        return {"verdict": "unverified"}
    return None


def compress_fastq(path: Path, threads: int) -> Path:
    """Gzip-compress ``path`` in place, returning the path to the compressed file.

    Uses ``pigz`` (parallel gzip) when it is on PATH, since it is substantially faster
    than single-threaded gzip on the multi-core machines this tool typically runs on;
    otherwise falls back to Python's ``gzip`` module, streaming the file through in
    1 MiB blocks so a large FASTQ file is never fully loaded into memory. Either way
    the uncompressed source is removed and only the ``.gz`` file remains.

    On failure, no partial ``.gz`` is left behind and the uncompressed source survives
    untouched: the Python fallback writes to a process-unique temp file and only
    ``os.replace``s it onto the final ``.gz`` name (and only then unlinks the source)
    once the gzip stream has closed cleanly; a failing ``pigz`` invocation has any
    ``.gz`` it managed to write before dying removed. Either way the original
    exception propagates to the caller.
    """
    target = path.with_suffix(path.suffix + ".gz")

    if shutil.which("pigz"):
        try:
            SecureSubprocess.run_secure("pigz", ["-p", str(threads), "-f", str(path)])
        except _TOOL_ERRORS:
            if target.exists():
                try:
                    target.unlink()
                except OSError as cleanup_error:
                    logger.warning(f"Could not remove partial {target} after a failed pigz run: {cleanup_error}")
            raise
        return target

    tmp_target = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    block_size = 1024 * 1024
    try:
        with open(path, "rb") as source, gzip.open(tmp_target, "wb", compresslevel=6) as dest:
            while True:
                block = source.read(block_size)
                if not block:
                    break
                dest.write(block)
        os.replace(tmp_target, target)
    finally:
        if tmp_target.exists():
            try:
                tmp_target.unlink()
            except OSError as cleanup_error:
                logger.warning(f"Could not remove leftover temp file {tmp_target}: {cleanup_error}")
    path.unlink()
    return target
