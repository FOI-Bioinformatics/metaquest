"""Download of one SRA accession with prefetch and fasterq-dump, and classification of its failures."""

import logging
import re
import shutil
import subprocess
import threading
import zlib
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from metaquest.core.exceptions import DataAccessError, SecurityError
from metaquest.data.sra import cleanup as cleanup_mod
from metaquest.data.sra import fastq as fastq_mod
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


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


# Set when the user interrupts a download run (Ctrl-C). download_accession checks it
# before each prefetch or fasterq-dump call, so a worker thread that has not yet
# started a tool returns without starting one, and the retry pass does not run.
STOP = threading.Event()


class _DownloadInterrupted(Exception):
    """Raised inside download_accession when STOP is set before a tool call."""


def _run_download_tool(executable: str, args: List[str]) -> None:
    """Run prefetch or fasterq-dump through ``run_secure`` unless STOP is set.

    A tool that exits non-zero while STOP is set was stopped by the interrupt (or killed
    as it started), so its failure is reported as an interruption rather than an error.
    """
    if STOP.is_set():
        raise _DownloadInterrupted()
    try:
        SecureSubprocess.run_secure(executable, args)
    except subprocess.CalledProcessError as e:
        if STOP.is_set():
            raise _DownloadInterrupted() from e
        raise


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
    # The callback is supplied by the caller and may raise anything (a registry lock timeout,
    # a JSON error); none of it may be mistaken for the download failing.
    except Exception as e:  # noqa: B902 - callback of unknown type, see the docstring
        logger.warning(f"Recording the result for {accession} failed: {e}")


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
        except OSError as e:
            logger.warning(f"Could not remove dangling symlink {output_path}: {e}")
        return False

    if force and output_path.exists():
        # Force redownload - remove existing directory
        cleanup_mod._remove_stale_entry(output_path)
        if not output_path.exists():
            logger.info(f"Removed existing directory for force redownload: {output_path}")
        return False

    if not force and output_path.exists():
        if fastq_mod.accession_has_fastq(output_path):
            return True

        # Found empty (or not-yet-complete) directory, will redownload.
        try:
            if output_path.is_symlink():
                output_path.unlink()
            else:
                output_path.rmdir()
        except OSError as e:
            logger.warning(f"Could not remove empty directory {output_path}: {e}")

    return False


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
    found = fastq_mod.fastq_files(temp_path)
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
    except OSError as e:
        logger.warning(f"Could not remove temp directory {temp_path}: {e}")

    logger.info(f"Successfully downloaded: {len(found)} files")

    # Compute the verdict on the plain files first: counting reads in an uncompressed
    # file is cheaper, and the verdict message format must stay stable either way.
    verdict = fastq_mod.verify_download(output_path.name, output_path, expected_spots=expected_spots)

    compression_failures = []
    compression_skipped: List[str] = []
    if compress:
        for index, file in enumerate(moved):
            if STOP.is_set():
                # The download itself is complete; leave the rest uncompressed rather than
                # start pigz after an interrupt.
                compression_skipped = [f.name for f in moved[index:]]
                logger.info(f"Compression of {output_path.name} skipped: the run was interrupted")
                break
            try:
                fastq_mod.compress_fastq(file, num_threads)
            except fastq_mod._TOOL_ERRORS as e:
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
    if compression_skipped:
        message += "; compression skipped (interrupted) for " + ", ".join(compression_skipped)

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
    verdict = fastq_mod.parse_verdict_message(message)
    verdict_name = verdict.get("verdict") if verdict else None
    if verdict_name not in ("complete", "unverified", "truncated"):
        return
    cleanup_mod._safe_rmtree(cache_path / accession)
    if verdict_name == "truncated":
        logger.info(f"{accession}: archive removed so the next attempt fetches it again")


def _staging_root(output_folder: Union[str, Path], staging_folder: Optional[Union[str, Path]]) -> Path:
    """Where a download's ``<accession>_temp`` build folder goes: ``staging_folder`` if given,
    else beside the output. Registering it as an allowed root lets fasterq-dump write there."""
    root = Path(staging_folder or output_folder)
    SecureSubprocess.add_allowed_root(root)
    return root


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
    staging_folder: Optional[Union[str, Path]] = None,
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
        staging_folder: Folder the ``<accession>_temp`` build directory is created in;
            defaults to ``output_folder``. The shared store points it at the store's own
            ``tmp`` folder so a half-written download never sits among finished datasets

    Returns:
        Tuple of (success, message)
    """
    output_path = Path(output_folder) / accession
    SecureSubprocess.add_allowed_root(Path(output_folder))
    staging_path = _staging_root(output_folder, staging_folder)
    cache_path = Path(sra_cache) if sra_cache else Path(output_folder) / ".sra-cache"

    # Check if already downloaded
    redownload = force or redownload_truncated
    if _check_existing_download(output_path, redownload):
        logger.info(f"Skipping {accession}, FASTQ files already exist")
        return True, "already exists"

    # A redownload must not reuse a cached archive: prefetch treats an existing <acc>.sra as
    # already fetched, so a truncated archive would be dumped again and stay truncated.
    if redownload and not keep_sra and (cache_path / accession).exists():
        cleanup_mod._safe_rmtree(cache_path / accession)
        logger.info(f"Removed the cached archive for {accession} so the redownload fetches it again")

    # Create a fresh temporary folder for download
    temp_path = staging_path / f"{accession}_temp"
    cleanup_mod._safe_rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)

    temp_folder_path = None
    try:
        logger.info(f"Downloading SRA for {accession}")

        # Handle temp folder for fasterq-dump
        temp_folder_path = cleanup_mod._prepare_temp_folder(temp_folder)

        using_prefetch = use_prefetch and shutil.which("prefetch") is not None
        if use_prefetch and not using_prefetch:
            logger.info(
                f"prefetch not found on PATH; running fasterq-dump directly against {accession}, "
                "which downloads and dumps in one step"
            )

        if using_prefetch:
            SecureSubprocess.add_allowed_root(cache_path)
            _run_download_tool(
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
        _run_download_tool("fasterq-dump", args)

        # Handle download output
        success, message = _handle_download_output(
            temp_path, output_path, expected_spots=expected_spots, compress=compress, num_threads=num_threads
        )

        if success and using_prefetch and not keep_sra:
            _discard_cached_archive(cache_path, accession, message)

        return success, message

    except _DownloadInterrupted:
        logger.info(f"Download of {accession} not started or continued: the run was interrupted")
        return False, "interrupted"

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

    except (OSError, EOFError, ValueError, zlib.error, subprocess.TimeoutExpired, DataAccessError) as e:
        logger.error(f"Error downloading {accession}: {e}")
        # <acc>_temp is kept for inspection; a new attempt starts clean.
        message = f"Download failed: {str(e)}"
        return False, f"{classify_download_error(message)}: {message}"

    finally:
        # Clean up auto-created temp directory (from tempfile.mkdtemp)
        if temp_folder_path and not temp_folder:
            cleanup_mod._safe_rmtree(temp_folder_path)


def fasterq_dump_version() -> str:
    """The installed fasterq-dump's version string, or an empty string if it cannot be run.

    Recorded in a store dataset's sidecar so a later reader knows which tool produced the
    files. The tool prints its name and version on separate lines, so the last non-empty
    line is used.
    """
    try:
        result = SecureSubprocess.run_secure("fasterq-dump", ["--version"])
        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        return lines[-1] if lines else ""
    except fastq_mod._TOOL_ERRORS as e:
        logger.debug(f"Could not read the fasterq-dump version: {e}")
        return ""
