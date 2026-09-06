"""
SRA data handling for MetaQuest.

This module provides functions for downloading and processing SRA data.
"""

import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from metaquest.core.constants import DEFAULT_MAX_WORKERS, FAILED_ACCESSIONS_FILE, FASTQ_GLOBS, MAX_CONCURRENT_DOWNLOADS
from metaquest.core.exceptions import DataAccessError, SecurityError
from metaquest.data.file_io import ensure_directory
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)

# Ratio of downloaded reads to NCBI's recorded run_total_spots at or above which a download
# counts as complete rather than truncated.
COMPLETE_RATIO_THRESHOLD = 0.99

# FASTQ file-name suffixes marking a mate of a pair: fasterq-dump's --split-3 output uses
# _1/_2, while files from other sources often use _R1/_R2.
MATE1_SUFFIXES = ("_1", "_R1")
MATE_SUFFIXES = ("_1", "_2", "_R1", "_R2")

# Regexes used by classify_download_error to sort a failure message into a coarse class that
# retry logic can act on: retry network/unknown failures, never retry not-found, and abort the
# whole run on disk-full.
# "could not resolve host" rather than a bare "resolve": prefetch reports a missing accession
# as "failed to resolve accession ... no data ( 404 )", which is a not-found, not a network error.
_NETWORK_ERROR_RE = re.compile(r"timeout|timed out|connection|could not resolve host|network|curl|ssl", re.IGNORECASE)
# "storage exhausted" and "disk-limit exeeded" (sic) are fasterq-dump's own out-of-space messages.
_DISK_FULL_ERROR_RE = re.compile(r"no space left|enospc|disk[ -]full|storage exhausted|disk-limit", re.IGNORECASE)
_NOT_FOUND_ERROR_RE = re.compile(
    r"not[ -]found|invalid accession|cannot be found|403|404|does not exist", re.IGNORECASE
)


def classify_download_error(text: str) -> str:
    """Classify a download failure message into a coarse error class.

    Returns one of ``"network"``, ``"disk-full"``, ``"not-found"``, or ``"unknown"``. Pure
    function of the message text; used both to prefix ``download_accession``'s failure
    messages and to decide, in ``_retry_failed_downloads``, which accessions are worth
    retrying.

    ``not-found`` is tested before ``network`` because a prefetch not-found message mentions
    resolving a query, and each class name classifies back to its own class so an already
    prefixed message (``"disk-full: not attempted"``) keeps its class on a second pass.
    """
    if not text:
        return "unknown"
    if _NOT_FOUND_ERROR_RE.search(text):
        return "not-found"
    if _DISK_FULL_ERROR_RE.search(text):
        return "disk-full"
    if _NETWORK_ERROR_RE.search(text):
        return "network"
    return "unknown"


def default_max_workers(num_threads: int) -> int:
    """Size the download worker pool from the machine's CPU count and per-download thread use.

    Each worker runs its own ``fasterq-dump`` using ``num_threads`` threads, so the pool is
    sized to roughly saturate the CPU without wildly oversubscribing it: divide the CPU count
    by the per-download thread count, floor at 1 worker, cap at ``MAX_CONCURRENT_DOWNLOADS``
    (a hard ceiling regardless of CPU count) and at ``DEFAULT_MAX_WORKERS`` (this project's
    conservative default).
    """
    cpu_count = os.cpu_count() or 4
    return min(MAX_CONCURRENT_DOWNLOADS, max(1, cpu_count // max(1, num_threads)), DEFAULT_MAX_WORKERS)


def is_transient_folder(name: str) -> bool:
    """True for a folder name that is a download-in-progress artifact, not a real accession.

    Covers the ``<acc>_temp`` folder ``download_accession`` builds into (kept on disk after a
    failure for inspection, see its except blocks) and fasterq-dump's own on-disk cache
    directory (``.sra-cache``). Neither should be counted as a downloaded accession by
    ``scan_downloads`` or the status command's on-disk inventory.
    """
    return name.endswith("_temp") or name == ".sra-cache"


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree if present, logging on failure instead of raising."""
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"Could not remove directory {path}: {e}")


def _notify_result(
    on_result: Optional[Callable[[str, bool, str], None]], accession: str, success: bool, message: str
) -> None:
    """Call ``on_result`` in isolation so a failing callback never corrupts the download tally.

    A registry write (or any other callback) can raise, e.g. a lock timeout. That must not be
    mistaken for the download itself failing, so this is never allowed to propagate.
    """
    if on_result is None:
        return
    try:
        on_result(accession, success, message)
    except Exception as e:
        logger.warning(f"Recording the result for {accession} failed: {e}")


def fastq_files(acc_dir: Union[str, Path]) -> List[Path]:
    """Non-empty FASTQ files directly in ``acc_dir``, sorted by name.

    Matches ``FASTQ_GLOBS`` (plain and gzipped ``.fastq``/``.fq``). A directory that does not
    exist (or a dangling symlink) yields an empty list; a symlinked directory is followed
    since ``Path.is_dir``/``Path.glob`` already resolve it transparently. A zero-byte file
    (e.g. left behind by an interrupted download) is never returned.
    """
    acc_path = Path(acc_dir)
    if not acc_path.is_dir():
        return []
    found = set()
    for pattern in FASTQ_GLOBS:
        for candidate in acc_path.glob(pattern):
            if candidate.is_file() and candidate.stat().st_size > 0:
                found.add(candidate)
    return sorted(found)


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
            state = json.loads(sidecar.read_text()).get("state")
        except Exception:
            state = None
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


def _read_blacklist_files(blacklist_files):
    """
    Read accessions from blacklist files.

    Args:
        blacklist_files: List of blacklist file paths

    Returns:
        Set of blacklisted accessions
    """
    blacklisted_accessions: set = set()

    if not blacklist_files:
        return blacklisted_accessions

    for blacklist_file in blacklist_files:
        try:
            file_accessions = set()
            with open(blacklist_file, "r") as f:
                for line in f:
                    accession = line.split("#", 1)[0].strip()
                    if accession:
                        file_accessions.add(accession)
                        blacklisted_accessions.add(accession)
            logger.info(f"Read {len(file_accessions)} blacklisted accessions from {blacklist_file}")
        except Exception as e:
            logger.warning(f"Error reading blacklist file {blacklist_file}: {e}")

    return blacklisted_accessions


def _prepare_temp_folder(temp_folder):
    """
    Prepare the temporary folder for fasterq-dump.

    Args:
        temp_folder: Path to temporary folder

    Returns:
        Path object of the prepared temp folder, or None if not successful
    """
    import tempfile

    if not temp_folder:
        # Create a temporary directory
        try:
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary folder: {temp_dir}")
            return Path(temp_dir)
        except Exception as e:
            logger.warning(f"Could not create temporary folder: {e}")
            return None

    # Ensure temp folder exists
    temp_path_obj = Path(temp_folder)
    try:
        temp_path_obj.mkdir(parents=True, exist_ok=True)
        if not os.access(temp_path_obj, os.W_OK):
            logger.warning(f"Temp folder {temp_folder} exists but is not writable, " "using default temp location")
            return None
        else:
            logger.info(f"Using temp folder: {temp_path_obj.absolute()}")
            SecureSubprocess.add_allowed_root(temp_path_obj)
            return temp_path_obj
    except Exception as e:
        logger.warning(f"Could not create or access temp folder {temp_folder}: {e}, " "using default temp location")
        return None


def _remove_stale_entry(output_path: Path) -> None:
    """Remove whatever is at ``output_path`` (a directory tree or a dangling/valid symlink).

    ``rmdir``/``rmtree`` reject a symlink (even one pointing at an empty directory) on most
    platforms, so a symlink is always ``unlink``'d instead.
    """
    try:
        if output_path.is_symlink():
            output_path.unlink()
        else:
            shutil.rmtree(output_path)
    except Exception as e:
        logger.warning(f"Could not remove {output_path}: {e}")


def _check_existing_download(output_path, force):
    """
    Check if the accession is already downloaded.

    Args:
        output_path: Path to output directory
        force: Whether to force redownload

    Returns:
        True if already downloaded, False otherwise
    """
    # A dangling symlink (e.g. left by an interrupted run) reports False from `.exists()`
    # even though the directory entry itself is still there; clear it so a fresh download can
    # create a real directory at this path instead of failing on FileExistsError.
    if output_path.is_symlink() and not output_path.exists():
        try:
            output_path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove dangling symlink {output_path}: {e}")
        return False

    if force and output_path.exists():
        # Force redownload - remove existing directory
        _remove_stale_entry(output_path)
        if not output_path.exists():
            logger.info(f"Removed existing directory for force redownload: {output_path}")
        return False

    if not force and output_path.exists():
        if accession_has_fastq(output_path):
            return True

        # Found empty (or not-yet-complete) directory, will redownload.
        try:
            if output_path.is_symlink():
                output_path.unlink()
            else:
                output_path.rmdir()
        except Exception as e:
            logger.warning(f"Could not remove empty directory {output_path}: {e}")

    return False


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
        except Exception:
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


def _handle_download_output(
    temp_path,
    output_path,
    expected_spots: Optional[int] = None,
    compress: bool = False,
    num_threads: int = 4,
):
    """
    Move downloaded files from temp path to output path, then verify completeness.

    Args:
        temp_path: Path to temporary folder
        output_path: Path to output directory
        expected_spots: NCBI's recorded total_spots for this accession, if known
        compress: If True, gzip each downloaded FASTQ file (via ``compress_fastq``)
            after the completeness verdict has been computed on the plain files
        num_threads: Thread count passed to ``compress_fastq`` (for pigz's ``-p``)

    Returns:
        Tuple of (success, message); the message carries the completeness verdict
        (``"... complete (n of m spots)"``, ``"... truncated (n of m spots)"``, or
        ``"... unverified"`` when ``expected_spots`` is unknown).
    """
    # Check if files were actually created
    found = fastq_files(temp_path)
    if not found:
        logger.error("No FASTQ files created despite successful command execution")
        # Kept for inspection, consistent with download_accession's failure paths.
        message = "No FASTQ files created"
        return False, f"{classify_download_error(message)}: {message}"

    # Move files to the final location
    # First ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    moved = []
    for file in found:
        dest = output_path / file.name
        shutil.move(str(file), str(dest))
        moved.append(dest)

    # Remove the temporary directory
    try:
        shutil.rmtree(temp_path)
    except Exception as e:
        logger.warning(f"Could not remove temp directory {temp_path}: {e}")

    logger.info(f"Successfully downloaded: {len(found)} files")

    # Compute the verdict on the plain files first: counting reads in an uncompressed
    # file is cheaper, and the verdict message format must stay stable either way.
    verdict = verify_download(output_path.name, output_path, expected_spots=expected_spots)

    compression_failures = []
    if compress:
        for file in moved:
            try:
                compress_fastq(file, num_threads)
            except Exception as e:
                logger.warning(f"Could not compress {file}: {e}")
                compression_failures.append(file.name)

    if verdict["verdict"] == "unverified":
        message = f"Downloaded {len(found)} files, unverified"
    else:
        spots_note = f"({verdict['reads_r1']} of {verdict['expected_spots']} spots)"
        message = f"Downloaded {len(found)} files, {verdict['verdict']} {spots_note}"
        if verdict["verdict"] == "truncated":
            logger.warning(
                f"Truncated download for {output_path.name}: {verdict['reads_r1']} of "
                f"{verdict['expected_spots']} spots (ratio {verdict['ratio']})"
            )

    if compression_failures:
        # Appended, never prepended, so parse_verdict_message's regex/substring checks
        # on the leading verdict text keep working unchanged.
        message += "; compression failed for " + ", ".join(compression_failures)

    return True, message


def _fasterq_dump_args(
    source: str,
    temp_path: Path,
    temp_folder_path: Optional[Path],
    num_threads: int,
    using_prefetch: bool,
) -> List[str]:
    """Build the fasterq-dump argument list.

    ``source`` is the prefetched ``.sra`` archive when prefetch ran, otherwise the accession
    itself, which fasterq-dump then resolves and downloads on its own.
    """
    if using_prefetch:
        args = ["--split-3", "--skip-technical", "--threads", str(num_threads), "-O", str(temp_path)]
        tail = [source]
    else:
        args = ["--threads", str(num_threads), "--progress", source, "-O", str(temp_path)]
        tail = ["--split-3", "--skip-technical"]
    if temp_folder_path:
        args.extend(["--temp", str(temp_folder_path.absolute())])
    return args + tail


def _cached_sra_archive(acc_cache_dir: Path, accession: str) -> Path:
    """The archive prefetch wrote for ``accession``, preferring ``.sra`` over ``.sralite``.

    NCBI serves some runs only in the smaller ``.sralite`` format, which fasterq-dump reads
    just as well. When the folder holds neither, the ``.sra`` path is returned so
    fasterq-dump reports the missing file itself.
    """
    expected = acc_cache_dir / f"{accession}.sra"
    if expected.is_file():
        return expected
    candidates = sorted(p for p in acc_cache_dir.glob(f"{accession}.sra*") if p.is_file())
    return candidates[0] if candidates else expected


def _discard_cached_archive(cache_path: Path, accession: str, message: str) -> None:
    """Remove the prefetched ``.sra`` archive once its FASTQ files are on disk.

    A truncated download drops the archive too: keeping it would make the next attempt dump
    the same short archive again rather than fetch the run afresh.
    """
    verdict = parse_verdict_message(message)
    verdict_name = verdict.get("verdict") if verdict else None
    if verdict_name not in ("complete", "unverified", "truncated"):
        return
    _safe_rmtree(cache_path / accession)
    if verdict_name == "truncated":
        logger.info(f"{accession}: archive removed so the next attempt fetches it again")


def download_accession(
    accession: str,
    output_folder: Union[str, Path],
    num_threads: int = 4,
    force: bool = False,
    temp_folder: Optional[Union[str, Path]] = None,
    expected_spots: Optional[int] = None,
    redownload_truncated: bool = False,
    sra_cache: Optional[Union[str, Path]] = None,
    use_prefetch: bool = True,
    keep_sra: bool = False,
    compress: bool = True,
) -> Tuple[bool, str]:
    """
    Download a single SRA accession using prefetch + fasterq-dump --split-3.

    Args:
        accession: SRA accession to download
        output_folder: Folder to save the downloaded files
        num_threads: Number of threads to use for download
        force: If True, redownload even if files exist
        temp_folder: Directory for temporary files
        expected_spots: NCBI's recorded total_spots for this accession, used to verify
            completeness once the download finishes
        redownload_truncated: If True, treat an existing on-disk copy the same as ``force``
            (i.e. wipe it and redownload) rather than skipping it as already present; used
            for an accession whose registry verdict was "truncated"
        sra_cache: Directory prefetch downloads the ``.sra`` archive into; defaults to
            ``<output_folder>/.sra-cache``
        use_prefetch: If True and ``prefetch`` is on PATH, download the ``.sra`` archive
            first and run fasterq-dump against it; otherwise fasterq-dump is run directly
            against the accession, as before this became configurable
        keep_sra: If True, keep the downloaded ``.sra`` archive after a successful,
            verified download rather than deleting it
        compress: If True, gzip each downloaded FASTQ file once the completeness verdict
            has been computed

    Returns:
        Tuple of (success, message)
    """
    output_path = Path(output_folder) / accession
    SecureSubprocess.add_allowed_root(Path(output_folder))
    cache_path = Path(sra_cache) if sra_cache else Path(output_folder) / ".sra-cache"

    # Check if already downloaded
    redownload = force or redownload_truncated
    if _check_existing_download(output_path, redownload):
        logger.info(f"Skipping {accession}, FASTQ files already exist")
        return True, "already exists"

    # A redownload must not reuse a cached archive: prefetch treats an existing <acc>.sra as
    # already fetched, so a truncated archive would be dumped again and stay truncated.
    if redownload and not keep_sra and (cache_path / accession).exists():
        _safe_rmtree(cache_path / accession)
        logger.info(f"Removed the cached archive for {accession} so the redownload fetches it again")

    # Create a fresh temporary folder for download
    temp_path = Path(output_folder) / f"{accession}_temp"
    _safe_rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)

    temp_folder_path = None
    try:
        logger.info(f"Downloading SRA for {accession}")

        # Handle temp folder for fasterq-dump
        temp_folder_path = _prepare_temp_folder(temp_folder)

        using_prefetch = use_prefetch and shutil.which("prefetch") is not None
        if use_prefetch and not using_prefetch:
            logger.info(
                f"prefetch not found on PATH; running fasterq-dump directly against {accession}, "
                "which downloads and dumps in one step"
            )

        if using_prefetch:
            SecureSubprocess.add_allowed_root(cache_path)
            SecureSubprocess.run_secure(
                "prefetch",
                ["-O", str(cache_path), "--max-size", "100G", "--progress", accession],
            )
            source = str(_cached_sra_archive(cache_path / accession, accession))
        else:
            # Direct call against the accession, without going through prefetch's
            # on-disk .sra archive: used when use_prefetch is False, or prefetch is
            # not installed.
            source = accession

        # Run fasterq-dump command securely
        args = _fasterq_dump_args(source, temp_path, temp_folder_path, num_threads, using_prefetch)
        SecureSubprocess.run_secure("fasterq-dump", args)

        # Handle download output
        success, message = _handle_download_output(
            temp_path, output_path, expected_spots=expected_spots, compress=compress, num_threads=num_threads
        )

        if success and using_prefetch and not keep_sra:
            _discard_cached_archive(cache_path, accession, message)

        return success, message

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading {accession}: {e.stderr}")
        # <acc>_temp is kept for inspection; a new attempt starts clean (the _safe_rmtree
        # above always wipes it before the next fasterq-dump run, so this is not a resume).
        message = f"Download failed: {e.stderr}"
        return False, f"{classify_download_error(message)}: {message}"

    except SecurityError as e:
        logger.error(f"Security error downloading {accession}: {e}")
        # <acc>_temp is kept for inspection; a new attempt starts clean.
        message = f"Security error: {e}"
        return False, f"{classify_download_error(message)}: {message}"

    except Exception as e:
        logger.error(f"Error downloading {accession}: {e}")
        # <acc>_temp is kept for inspection; a new attempt starts clean.
        message = f"Download failed: {str(e)}"
        return False, f"{classify_download_error(message)}: {message}"

    finally:
        # Clean up auto-created temp directory (from tempfile.mkdtemp)
        if temp_folder_path and not temp_folder:
            _safe_rmtree(temp_folder_path)


def _check_existing_downloads(
    accessions: List[str],
    fastq_path: Path,
    force: bool,
    blacklisted_accessions: Optional[Set[str]] = None,
    truncated_accessions: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Check which accessions need downloading and which are already downloaded or blacklisted.

    Args:
        accessions: List of accessions
        fastq_path: Path to FASTQ directory
        force: Whether to force redownload
        blacklisted_accessions: Set of blacklisted accessions
        truncated_accessions: Accessions whose registry verdict is "truncated"; treated like
            ``force`` for that one accession, so a partial download on disk is redownloaded
            rather than counted as already present

    Returns:
        Tuple of (already_downloaded, to_download, blacklisted)
    """
    already_downloaded = []
    to_download = []
    blacklisted = []

    if blacklisted_accessions is None:
        blacklisted_accessions = set()
    if truncated_accessions is None:
        truncated_accessions = set()

    for acc in accessions:
        if acc in blacklisted_accessions:
            blacklisted.append(acc)
            continue

        if not force and acc not in truncated_accessions and accession_has_fastq(fastq_path / acc):
            already_downloaded.append(acc)
        else:
            to_download.append(acc)

    return already_downloaded, to_download, blacklisted


def _process_download_results(futures_results, accessions_to_download, download_results, failed_accessions):
    """
    Process download results from completed futures.

    Args:
        futures_results: List of (accession, future_result) tuples
        accessions_to_download: List of accessions that were attempted
        download_results: Dictionary to store results
        failed_accessions: List to store failed accessions

    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful_count = 0
    failed_count = 0

    for accession, result in futures_results:
        try:
            success, message = result
            download_results[accession] = message

            if success:
                successful_count += 1
                # Log progress periodically
                if successful_count % 5 == 0:
                    logger.info(
                        f"Downloaded {successful_count}/{len(accessions_to_download)} " f"({failed_count} failed)"
                    )
            else:
                failed_count += 1
                failed_accessions.append(accession)
                logger.warning(f"Failed to download {accession}: {message}")

        except Exception as e:
            failed_count += 1
            failed_accessions.append(accession)
            logger.error(f"Error processing download result for {accession}: {e}")
            download_results[accession] = f"Error: {str(e)}"

    return successful_count, failed_count


def _retry_failed_downloads(
    failed_accessions,
    max_retries,
    fastq_path,
    num_threads,
    temp_folder,
    download_results,
    on_result: Optional[Callable[[str, bool, str], None]] = None,
    expected_spots: Optional[Dict[str, int]] = None,
    redownload_truncated: bool = False,
    sra_cache: Optional[Union[str, Path]] = None,
    use_prefetch: bool = True,
    keep_sra: bool = False,
    compress: bool = True,
):
    """
    Retry failed downloads.

    Args:
        failed_accessions: List of accessions that failed
        max_retries: Maximum number of retry attempts
        fastq_path: Path to FASTQ directory
        num_threads: Number of threads to use
        temp_folder: Temporary folder path
        download_results: Dictionary to store results
        on_result: Optional callback invoked with (accession, success, message) after each retry
        expected_spots: Per-accession NCBI total_spots, used to verify completeness
        redownload_truncated: Forwarded to ``download_accession`` for each retry
        sra_cache: Forwarded to ``download_accession`` for each retry
        use_prefetch: Forwarded to ``download_accession`` for each retry
        keep_sra: Forwarded to ``download_accession`` for each retry
        compress: Forwarded to ``download_accession`` for each retry

    Returns:
        Tuple of (retried_successful, failed_accessions, abort_reason). ``abort_reason`` is
        ``"disk-full"`` when a retry attempt hit a disk-full error: the accession that hit it
        is recorded as failed and notified like any other failure, every other accession still
        queued in that round is marked failed with the message ``"disk-full: not attempted"``
        (and notified too) without ever calling ``download_accession``, and no further retry
        round runs. ``abort_reason`` is ``None`` when every round ran to completion normally.
    """
    if max_retries <= 0 or not failed_accessions:
        return 0, failed_accessions, None

    logger.info(f"Retrying {len(failed_accessions)} failed downloads")
    retry_count = 0
    retried_successful = 0
    expected_spots = expected_spots or {}
    abort_reason: Optional[str] = None

    for retry in range(max_retries):
        if not failed_accessions:
            break

        # An accession classified as not-found from its last attempt's message will not
        # succeed on retry (the run genuinely does not exist, or the ID is invalid); skip it
        # but keep it in the failed list rather than burning a retry round on it.
        retry_batch = []
        skipped_not_found = []
        for accession in failed_accessions:
            if classify_download_error(download_results.get(accession, "")) == "not-found":
                skipped_not_found.append(accession)
            else:
                retry_batch.append(accession)

        failed_accessions = list(skipped_not_found)

        if not retry_batch:
            break

        logger.info(f"Retry attempt {retry + 1}/{max_retries}")

        for index, accession in enumerate(retry_batch):
            retry_count += 1
            try:
                success, message = download_accession(
                    accession,
                    fastq_path,
                    num_threads,
                    force=False,
                    temp_folder=temp_folder,
                    expected_spots=expected_spots.get(accession),
                    redownload_truncated=redownload_truncated,
                    sra_cache=sra_cache,
                    use_prefetch=use_prefetch,
                    keep_sra=keep_sra,
                    compress=compress,
                )
            except Exception as e:
                failed_accessions.append(accession)
                logger.error(f"Error retrying download for {accession}: {e}")
                download_results[accession] = f"Retry {retry + 1} error: {str(e)}"
                _notify_result(on_result, accession, False, download_results[accession])
                continue

            download_results[accession] = f"Retry {retry + 1}: {message}"

            if success:
                retried_successful += 1
                logger.info(f"Successfully downloaded {accession} on retry {retry + 1}")
                _notify_result(on_result, accession, success, download_results[accession])
                continue

            failed_accessions.append(accession)
            logger.warning(f"Failed to download {accession} on retry {retry + 1}: {message}")
            _notify_result(on_result, accession, success, download_results[accession])

            if classify_download_error(message) == "disk-full":
                abort_reason = "disk-full"
                logger.error(f"Disk full while downloading {accession}; aborting remaining retries")
                for not_attempted in retry_batch[index + 1 :]:
                    download_results[not_attempted] = "disk-full: not attempted"
                    failed_accessions.append(not_attempted)
                    _notify_result(on_result, not_attempted, False, download_results[not_attempted])
                break

        if abort_reason:
            break

        if failed_accessions and retry < max_retries - 1:
            time.sleep(2**retry)

    return retried_successful, failed_accessions, abort_reason


def _handle_download_failure(fastq_path, failed_accessions):
    """
    Handle failed downloads by writing failed accessions to a file.

    Args:
        fastq_path: Path to FASTQ directory
        failed_accessions: List of failed accessions
    """
    if not failed_accessions:
        return

    # Write failed accessions to file for easier retry
    failed_file = Path(fastq_path) / FAILED_ACCESSIONS_FILE
    with open(failed_file, "w") as f:
        for acc in failed_accessions:
            f.write(f"{acc}\n")

    logger.info(f"Failed accessions written to {failed_file}")
    logger.info(
        f"To retry only failed accessions: metaquest download_sra "
        f"--accessions-file {failed_file} "
        f"--fastq-folder {fastq_path}"
    )


def _execute_parallel_downloads(
    accessions,
    fastq_path,
    num_threads,
    max_workers,
    force,
    temp_folder,
    download_results,
    failed_accessions,
    on_result: Optional[Callable[[str, bool, str], None]] = None,
    expected_spots: Optional[Dict[str, int]] = None,
    redownload_truncated: bool = False,
    sra_cache: Optional[Union[str, Path]] = None,
    use_prefetch: bool = True,
    keep_sra: bool = False,
    compress: bool = True,
):
    """Download accessions concurrently and tally results. Returns (successful, failed)."""
    expected_spots = expected_spots or {}
    futures_results: list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_accession,
                acc,
                fastq_path,
                num_threads,
                force,
                temp_folder,
                expected_spots=expected_spots.get(acc),
                redownload_truncated=redownload_truncated,
                sra_cache=sra_cache,
                use_prefetch=use_prefetch,
                keep_sra=keep_sra,
                compress=compress,
            ): acc
            for acc in accessions
        }
        for future in as_completed(futures):
            acc = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"Download failed for {acc}: {e}")
                futures_results.append((acc, None))
                _notify_result(on_result, acc, False, str(e))
                continue

            futures_results.append((acc, result))
            success, message = result
            _notify_result(on_result, acc, success, message)

    return _process_download_results(futures_results, accessions, download_results, failed_accessions)


def _resolve_fastq_path(fastq_folder: Union[str, Path], dry_run: bool) -> Path:
    """Return the FASTQ output path, creating it unless in dry-run mode.

    In dry-run mode the folder is not created, but an existing non-directory
    at that location is still rejected.
    """
    fastq_path = Path(fastq_folder)
    if dry_run:
        if fastq_path.exists() and not fastq_path.is_dir():
            raise DataAccessError(f"{fastq_folder} exists but is not a directory")
        logger.info(f"Dry run mode: Would use {fastq_path} for downloads")
        return fastq_path
    return ensure_directory(fastq_folder)


def _download_with_retries(
    accessions_to_download,
    fastq_path,
    num_threads,
    max_workers,
    force,
    temp_folder,
    max_retries,
    on_result: Optional[Callable[[str, bool, str], None]] = None,
    expected_spots: Optional[Dict[str, int]] = None,
    redownload_truncated: bool = False,
    sra_cache: Optional[Union[str, Path]] = None,
    use_prefetch: bool = True,
    keep_sra: bool = False,
    compress: bool = True,
) -> Tuple[int, int, List[str], Dict[str, Any], Optional[str]]:
    """Run the parallel downloads and optional retry pass.

    Returns (successful_count, failed_count, failed_accessions, download_results, abort_reason).
    ``abort_reason`` is ``"disk-full"`` when a retry hit a disk-full error (see
    ``_retry_failed_downloads``), else ``None``.
    """
    failed_accessions: list = []
    download_results: dict = {}
    abort_reason: Optional[str] = None
    successful_count, failed_count = _execute_parallel_downloads(
        accessions_to_download,
        fastq_path,
        num_threads,
        max_workers,
        force,
        temp_folder,
        download_results,
        failed_accessions,
        on_result,
        expected_spots,
        redownload_truncated,
        sra_cache,
        use_prefetch,
        keep_sra,
        compress,
    )

    if max_retries > 0 and failed_accessions:
        retried_successful, failed_accessions, abort_reason = _retry_failed_downloads(
            failed_accessions,
            max_retries,
            fastq_path,
            num_threads,
            temp_folder,
            download_results,
            on_result,
            expected_spots,
            redownload_truncated,
            sra_cache,
            use_prefetch,
            keep_sra,
            compress,
        )
        successful_count += retried_successful
        failed_count -= retried_successful
        if retried_successful > 0:
            logger.info(f"Successfully downloaded {retried_successful} accessions on retry")

    return successful_count, failed_count, failed_accessions, download_results, abort_reason


def download_sra(
    fastq_folder: Union[str, Path],
    accessions_file: Union[str, Path],
    max_downloads: Optional[int] = None,
    dry_run: bool = False,
    num_threads: int = 4,
    max_workers: int = 4,
    force: bool = False,
    max_retries: int = 1,
    temp_folder: Optional[Union[str, Path]] = None,
    blacklist: Optional[List[Union[str, Path]]] = None,
    blacklist_accessions: Optional[Set[str]] = None,
    on_result: Optional[Callable[[str, bool, str], None]] = None,
    expected_spots: Optional[Dict[str, int]] = None,
    redownload_truncated: bool = False,
    truncated_accessions: Optional[Set[str]] = None,
    sra_cache: Optional[Union[str, Path]] = None,
    use_prefetch: bool = True,
    keep_sra: bool = False,
    compress: bool = True,
) -> Dict[str, Any]:
    """
    Download multiple SRA datasets.

    Args:
        fastq_folder: Folder to save downloaded FASTQ files
        accessions_file: File containing SRA accessions, one per line
        max_downloads: Maximum number of datasets to download
        dry_run: If True, only count accessions without downloading
        num_threads: Number of threads for each fasterq-dump
        max_workers: Number of parallel downloads
        force: If True, redownload even if files exist
        max_retries: Maximum number of retry attempts for failed downloads
        temp_folder: Directory to use for fasterq-dump temporary files
        blacklist: One or more files containing accessions to skip
        blacklist_accessions: An additional set of accessions to skip (e.g. registry exclusions),
            joined with any accessions read from ``blacklist``
        on_result: Optional callback invoked with (accession, success, message) on the main
            thread as each download (and each retry) completes
        expected_spots: Per-accession NCBI ``run_total_spots``, used to verify each download's
            completeness once it finishes; an accession missing from this mapping is reported
            as "unverified" rather than "complete"/"truncated"
        redownload_truncated: Forwarded to every ``download_accession`` call so a partial copy
            left on disk by a prior truncated attempt is wiped and redownloaded rather than
            reused
        truncated_accessions: Accessions whose registry verdict is "truncated"; excluded from
            ``already_downloaded`` so they are redownloaded even though files exist on disk
        sra_cache: Forwarded to every ``download_accession`` call; directory prefetch downloads
            the ``.sra`` archive into (defaults to ``<fastq_folder>/.sra-cache`` per accession)
        use_prefetch: Forwarded to every ``download_accession`` call; download via prefetch then
            fasterq-dump when True and prefetch is on PATH, else fasterq-dump directly
        keep_sra: Forwarded to every ``download_accession`` call; keep the ``.sra`` archive
            after a successful, verified download instead of deleting it
        compress: Forwarded to every ``download_accession`` call; gzip each downloaded FASTQ
            file once its completeness verdict has been computed

    Returns:
        Dictionary with download statistics

    Raises:
        DataAccessError: If the download fails
    """
    try:
        # Handle the output folder based on dry run status
        fastq_path = _resolve_fastq_path(fastq_folder, dry_run)

        # Read accessions from file
        with open(accessions_file, "r") as f:
            all_accessions = [line.strip() for line in f if line.strip()]

        logger.info(f"Found {len(all_accessions)} accessions in file")

        # Read blacklisted accessions (from files, plus any passed in directly, e.g. registry exclusions)
        blacklisted_accessions = _read_blacklist_files(blacklist)
        if blacklist_accessions:
            blacklisted_accessions |= set(blacklist_accessions)
        if blacklisted_accessions:
            logger.info(f"Found total of {len(blacklisted_accessions)} blacklisted accessions")

        # Check which accessions need downloading
        already_downloaded, accessions_to_download, blacklisted = _check_existing_downloads(
            all_accessions, fastq_path, force, blacklisted_accessions, truncated_accessions
        )

        logger.info(f"{len(already_downloaded)} accessions already downloaded")
        logger.info(f"{len(blacklisted)} accessions blacklisted")
        logger.info(f"{len(accessions_to_download)} accessions need downloading")

        # Accessions that --max-downloads would cut off, computed before any truncation so a
        # dry run can report them too.
        skipped_accessions: List[str] = []
        if max_downloads is not None and max_downloads < len(accessions_to_download):
            skipped_accessions = accessions_to_download[max_downloads:]

        if dry_run:
            logger.info(f"Dry run: would download {len(accessions_to_download)} accessions")
            return {
                "total": len(all_accessions),
                "already_downloaded": len(already_downloaded),
                "blacklisted": len(blacklisted),
                "to_download": len(accessions_to_download),
                "successful": 0,
                "failed": 0,
                "already_downloaded_accessions": sorted(str(a) for a in already_downloaded),
                "blacklisted_accessions": sorted(str(a) for a in blacklisted),
                "skipped_accessions": sorted(str(a) for a in skipped_accessions),
                # No download ran, so nothing could abort; the key is present either way so
                # callers can read it without knowing which mode produced the stats.
                "aborted": None,
            }

        # Limit number of downloads if specified
        if max_downloads is not None and max_downloads < len(accessions_to_download):
            logger.info(f"Limiting to {max_downloads} downloads")
            accessions_to_download = accessions_to_download[:max_downloads]

        # Download accessions in parallel, with an optional retry pass
        successful_count, failed_count, failed_accessions, download_results, abort_reason = _download_with_retries(
            accessions_to_download,
            fastq_path,
            num_threads,
            max_workers,
            force,
            temp_folder,
            max_retries,
            on_result,
            expected_spots,
            redownload_truncated,
            sra_cache,
            use_prefetch,
            keep_sra,
            compress,
        )

        # Log final summary
        logger.info("Download summary:")
        logger.info(f"  Total accessions: {len(all_accessions)}")
        logger.info(f"  Already downloaded: {len(already_downloaded)}")
        logger.info(f"  Blacklisted: {len(blacklisted)}")
        logger.info(f"  Newly downloaded: {successful_count}")
        logger.info(f"  Failed downloads: {failed_count}")

        if abort_reason:
            logger.error(f"Download run aborted: {abort_reason}")
        if failed_count > 0:
            logger.warning("Some downloads failed. Use --force to retry or --max-retries to enable " "automatic retry.")
            _handle_download_failure(fastq_path, failed_accessions)

        download_stats = {
            "total": len(all_accessions),
            "already_downloaded": len(already_downloaded),
            "blacklisted": len(blacklisted),
            "successful": successful_count,
            "failed": failed_count,
            "failed_accessions": failed_accessions,
            "results": download_results,
            "already_downloaded_accessions": sorted(str(a) for a in already_downloaded),
            "blacklisted_accessions": sorted(str(a) for a in blacklisted),
            "skipped_accessions": sorted(str(a) for a in skipped_accessions),
            "aborted": abort_reason,
        }

        return download_stats

    except Exception as e:
        raise DataAccessError(f"Downloading SRA data: {e}")
