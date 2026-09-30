"""Parallel download of many accessions, tallying of their results, and the retry pass for failures."""

import logging
import subprocess
import time
import zlib
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from metaquest.core.constants import FAILED_ACCESSIONS_FILE
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.file_io import write_text_atomic
from metaquest.data.sra import accession as accession_mod
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


# What a download worker (download_accession, or the store downloader) raises for a dataset
# that could not be fetched: store and catalogue failures arrive as DataAccessError, file and
# FASTQ reading failures as OSError, EOFError, ValueError or (a corrupt gzip stream)
# zlib.error. Anything else is a programming error and propagates.
_DOWNLOAD_ERRORS = (MetaQuestError, OSError, EOFError, ValueError, zlib.error, subprocess.SubprocessError)


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
            else:
                failed_count += 1
                failed_accessions.append(accession)
                logger.warning(f"Failed to download {accession}: {message}")

        # A worker that raised leaves None here (see _execute_parallel_downloads), which does not unpack.
        except (TypeError, ValueError) as e:
            failed_count += 1
            failed_accessions.append(accession)
            logger.error(f"Error processing download result for {accession}: {e}")
            download_results[accession] = f"Error: {str(e)}"

    logger.info("Downloaded %d of %d (%d failed)", successful_count, len(accessions_to_download), failed_count)

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
    downloader: Optional[Callable[..., Tuple[bool, str]]] = None,
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
        downloader: Callable used in place of ``download_accession``; the shared store
            passes one that links the project to the store's copy instead of downloading
            into the project folder

    Returns:
        Tuple of (retried_successful, failed_accessions, abort_reason). ``abort_reason`` is
        ``"disk-full"`` when a retry attempt hit a disk-full error: the accession that hit it
        is recorded as failed and notified like any other failure, every other accession still
        queued in that round is marked failed with the message ``"disk-full: not attempted"``
        (and notified too) without ever calling ``download_accession``, and no further retry
        round runs. ``abort_reason`` is ``None`` when every round ran to completion normally.
    """
    if max_retries <= 0 or not failed_accessions or accession_mod.STOP.is_set():
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
            if accession_mod.classify_download_error(download_results.get(accession, "")) == "not-found":
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
                success, message = (downloader or accession_mod.download_accession)(
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
            except _DOWNLOAD_ERRORS as e:
                failed_accessions.append(accession)
                logger.error(f"Error retrying download for {accession}: {e}")
                download_results[accession] = f"Retry {retry + 1} error: {str(e)}"
                accession_mod._notify_result(on_result, accession, False, download_results[accession])
                continue

            download_results[accession] = f"Retry {retry + 1}: {message}"

            if success:
                retried_successful += 1
                logger.info(f"Successfully downloaded {accession} on retry {retry + 1}")
                accession_mod._notify_result(on_result, accession, success, download_results[accession])
                continue

            failed_accessions.append(accession)
            logger.warning(f"Failed to download {accession} on retry {retry + 1}: {message}")
            accession_mod._notify_result(on_result, accession, success, download_results[accession])

            if accession_mod.classify_download_error(message) == "disk-full":
                abort_reason = "disk-full"
                logger.error(f"Disk full while downloading {accession}; aborting remaining retries")
                for not_attempted in retry_batch[index + 1 :]:
                    download_results[not_attempted] = "disk-full: not attempted"
                    failed_accessions.append(not_attempted)
                    accession_mod._notify_result(on_result, not_attempted, False, download_results[not_attempted])
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
    write_text_atomic(failed_file, "".join(f"{acc}\n" for acc in failed_accessions))

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
    downloader: Optional[Callable[..., Tuple[bool, str]]] = None,
):
    """Download accessions concurrently and tally results. Returns (successful, failed).

    ``downloader`` replaces ``download_accession`` when the project reads through a shared
    store; it takes the same arguments so the tally, retries and callbacks are unchanged.

    On ``KeyboardInterrupt`` STOP is set, downloads not yet started are cancelled, every
    running prefetch or fasterq-dump child is terminated, and the interrupt is re-raised.
    """
    accession_mod.STOP.clear()
    SecureSubprocess.clear_stopping()
    expected_spots = expected_spots or {}
    futures_results: list = []
    worker = downloader or accession_mod.download_accession
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            futures = {
                executor.submit(
                    worker,
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
                except (*_DOWNLOAD_ERRORS, CancelledError) as e:
                    logger.error(f"Download failed for {acc}: {e}")
                    futures_results.append((acc, None))
                    accession_mod._notify_result(on_result, acc, False, str(e))
                    continue

                futures_results.append((acc, result))
                success, message = result
                accession_mod._notify_result(on_result, acc, success, message)
        except KeyboardInterrupt:
            accession_mod.STOP.set()
            logger.warning("Interrupted; cancelling pending downloads and stopping running tools")
            executor.shutdown(wait=False, cancel_futures=True)
            SecureSubprocess.terminate_children()
            raise

    return _process_download_results(futures_results, accessions, download_results, failed_accessions)


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
    downloader: Optional[Callable[..., Tuple[bool, str]]] = None,
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
        downloader,
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
            downloader,
        )
        successful_count += retried_successful
        failed_count -= retried_successful
        if retried_successful > 0:
            logger.info(f"Successfully downloaded {retried_successful} accessions on retry")

    return successful_count, failed_count, failed_accessions, download_results, abort_reason
