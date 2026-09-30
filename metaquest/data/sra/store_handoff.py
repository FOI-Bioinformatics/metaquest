"""Downloads routed through the shared data store: the store keeps the one copy, the project gets a link.

The ``metaquest.store`` imports stay inside the functions because ``metaquest.store`` itself
imports this package.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.data.sra import accession as accession_mod
from metaquest.data.sra import cleanup as cleanup_mod
from metaquest.data.sra import fastq as fastq_mod

logger = logging.getLogger(__name__)

# The publish step moved to ``cleanup.publish_folder`` so plain projects share it; the old
# name stays for callers and tests that use it.
_publish_store_dataset = cleanup_mod.publish_folder


# Sidecar states that mean the store's copy is usable as it stands: verified against NCBI's
# spot count, or downloaded without a spot count to check it against.
STORE_READY_STATES = ("complete", "unverified")

# Prefix ``_link_result`` puts on the message for a dataset the store already held (linked
# rather than downloaded); the single place this string is produced, so every caller that
# needs to tell a link apart from a fresh download (the CLI's result recorder, this module's
# own run summary) matches against this constant rather than a copy of the literal.
STORE_LINKED_PREFIX = "linked from store"


def _metadata_xml(folders, accession: str) -> Optional[Path]:
    """The first ``<accession>_metadata.xml`` found in ``folders``, or None."""
    if folders is None:
        candidates: List[Union[str, Path]] = []
    elif isinstance(folders, (str, Path)):
        candidates = [folders]
    else:
        candidates = list(folders)
    for folder in candidates:
        xml = Path(folder) / f"{accession}_metadata.xml"
        if xml.is_file():
            return xml
    return None


def _store_state(store, accession: str) -> str:
    """What the store holds for ``accession``: ``ready``, ``incomplete`` or ``absent``.

    ``incomplete`` covers a sidecar recording a partial, failed or in-progress download, and
    also files sitting there with no sidecar at all, which is what an interrupted run leaves
    behind and cannot be trusted without re-fetching.
    """
    from metaquest.store.layout import sidecar_path, sra_dir
    from metaquest.store.sidecar import read_sidecar

    sidecar_file = sidecar_path(store, accession)
    if sidecar_file.is_file():
        sidecar = read_sidecar(sidecar_file)
        state = sidecar.state if sidecar else None
        return "ready" if state in STORE_READY_STATES else "incomplete"
    return "incomplete" if fastq_mod.fastq_files(sra_dir(store, accession)) else "absent"


def _link_result(accession: str, project_fastq: Path, store, link_mode: str, note: str) -> Tuple[bool, str]:
    """Link the store's copy into the project and report how many files it holds."""
    from metaquest.store.link import link_dataset

    link = link_dataset(project_fastq, accession, store, mode=link_mode)
    return True, f"{STORE_LINKED_PREFIX}{note}, {len(fastq_mod.fastq_files(link))} files"


def _store_precheck(
    accession: str,
    project_fastq: Path,
    store,
    link_mode: str,
    accept_partial: bool,
    resume_partial: bool,
) -> Optional[Tuple[bool, str]]:
    """Decide what the store alone can settle for ``accession``, without any network call.

    Returns a result when the dataset is already usable (linked), or when it is incomplete
    and the caller asked not to resume it; returns None when it must be downloaded.
    """
    state = _store_state(store, accession)
    if state == "ready":
        return _link_result(accession, project_fastq, store, link_mode, "")
    if state == "incomplete" and not resume_partial:
        if accept_partial:
            return _link_result(accession, project_fastq, store, link_mode, " (partial)")
        return False, f"partial in store; rerun with --resume-partial to finish {accession}"
    return None


def _store_fetch(
    accession: str,
    project_fastq: Path,
    store,
    link_mode: str,
    store_metadata,
    **download_kwargs,
) -> Tuple[bool, str]:
    """Download ``accession`` into the store, describe it, and link the project to it.

    Everything happens in the store's ``tmp`` folder: fasterq-dump builds
    ``tmp/<ACC>_temp``, the files are verified and compressed into ``tmp/<ACC>``, and the
    sidecar describing them is written there too. Only then is the finished folder published
    into ``sra/<ACC>`` with one rename, so that folder never holds an unverified or
    sidecar-less dataset for another project to find, and an existing copy is replaced only
    once its replacement is complete.
    """
    from metaquest.store.catalog import catalog_write
    from metaquest.store.layout import sra_dir
    from metaquest.store.link import link_dataset
    from metaquest.store.sidecar import build_sidecar, ncbi_from_metadata_xml, write_sidecar

    target = sra_dir(store, accession)
    replacing = target.exists()
    staged = store.tmp / accession
    # A forced refetch stays forced even when nothing is in the store yet: it must not reuse
    # a cached archive from an earlier attempt.
    download_kwargs["force"] = bool(download_kwargs.get("force")) or replacing
    download_kwargs.setdefault("sra_cache", None)
    if download_kwargs["sra_cache"] is None:
        download_kwargs["sra_cache"] = store.tmp / ".sra-cache"

    # A caller that gave no temp_folder gets one under the store's own tmp, not the system
    # temp directory: fasterq-dump's scratch space can run to several gigabytes per accession,
    # and download_accession only cleans up a temp_folder it created itself (its finally block
    # skips a folder the caller supplied), so this scratch folder is ours to remove afterwards.
    own_scratch: Optional[Path] = None
    if download_kwargs.get("temp_folder") is None:
        own_scratch = store.tmp / f"{accession}_fqtmp"
        download_kwargs["temp_folder"] = own_scratch

    # Whatever an earlier interrupted attempt left staged is not a resume point: the download
    # would otherwise be skipped as "already exists" and that partial copy published.
    cleanup_mod._safe_rmtree(staged)

    try:
        success, message = accession_mod.download_accession(
            accession, store.tmp, staging_folder=store.tmp, **download_kwargs
        )
    finally:
        if own_scratch is not None:
            cleanup_mod._safe_rmtree(own_scratch)

    if not success:
        cleanup_mod._safe_rmtree(staged)
        return False, message

    files = fastq_mod.fastq_files(staged)
    compression = "gzip" if any(path.name.endswith(".gz") for path in files) else "none"
    xml = _metadata_xml(store_metadata, accession) or _metadata_xml(store.metadata, accession)
    ncbi = ncbi_from_metadata_xml(xml) if xml is not None else {}

    sidecar = build_sidecar(accession, staged, ncbi, accession_mod.fasterq_dump_version(), compression)
    write_sidecar(staged / f"{accession}.json", sidecar)
    cleanup_mod.publish_folder(staged, target, store.tmp)

    with catalog_write(store) as catalog:
        catalog.upsert_dataset(sidecar)

    link_dataset(project_fastq, accession, store, mode=link_mode)
    return True, f"{message}; stored"


def _store_download(
    accession: str,
    project_fastq: Union[str, Path],
    store,
    link_mode: str = "auto",
    accept_partial: bool = False,
    resume_partial: bool = True,
    store_metadata=None,
    force: bool = False,
    lock_wait: float = 0.0,
    **download_kwargs,
) -> Tuple[bool, str]:
    """Get ``accession`` for this project through the shared store.

    The store holds one copy of every dataset; the project only ever gets a link to it. A
    copy that is already there is linked without touching the network. Otherwise the store's
    per-accession lock is taken, the sidecar is checked again (another project may have
    finished the download while this one waited), and the download runs into the store.

    The lock is held for the whole download (``metaquest.store.locks.dataset_lock``, which
    heartbeats while held), so a second project waits rather than writing the same folder.
    ``lock_wait`` of zero waits for as long as the other project keeps working; a positive
    value gives up after that many seconds, naming the accession and the holder.
    """
    from metaquest.store.locks import LockWaitStopped, dataset_lock

    project_path = Path(project_fastq)
    for directory in (store.sra, store.tmp, store.locks):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        if not force:
            settled = _store_precheck(accession, project_path, store, link_mode, accept_partial, resume_partial)
            if settled is not None:
                return settled

        with dataset_lock(store, accession, wait_seconds=lock_wait, should_stop=accession_mod.STOP.is_set):
            if not force:
                settled = _store_precheck(accession, project_path, store, link_mode, accept_partial, resume_partial)
                if settled is not None:
                    return settled
            return _store_fetch(
                accession, project_path, store, link_mode, store_metadata, force=force, **download_kwargs
            )
    except LockWaitStopped:
        logger.info(f"Stopped waiting for the store lock on {accession}: the run was interrupted")
        return False, "interrupted"
    except DataAccessError as e:
        logger.error(f"Store download failed for {accession}: {e}")
        return False, f"store error: {e}"


def _store_downloader(
    store, link_mode: str, accept_partial: bool, resume_partial: bool, store_metadata, lock_wait: float = 0.0
):
    """A ``download_accession``-shaped callable that routes every download through the store.

    The download loops call it with the project's FASTQ folder as ``output_folder``, which is
    where the link is created; the files themselves land in the store. Returns None when no
    store is configured, which leaves the loops downloading into the project folder as before.
    """
    if store is None:
        return None

    def _download(accession, output_folder, num_threads=4, force=False, temp_folder=None, **kwargs):
        return _store_download(
            accession,
            output_folder,
            store,
            link_mode=link_mode,
            accept_partial=accept_partial,
            resume_partial=resume_partial,
            store_metadata=store_metadata,
            force=force,
            lock_wait=lock_wait,
            num_threads=num_threads,
            temp_folder=temp_folder,
            **kwargs,
        )

    return _download
