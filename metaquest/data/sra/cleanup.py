"""Temporary folders of SRA downloads: recognising them, sizing them, preparing and removing them."""

import logging
import os
import shutil
from pathlib import Path
from typing import Union

from metaquest.core.exceptions import SecurityError
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


def is_transient_folder(name: str) -> bool:
    """True for a folder name that is a download-in-progress artifact, not a real accession.

    Covers the ``<acc>_temp`` folder ``download_accession`` builds into (kept on disk after a
    failure for inspection, see its except blocks), the ``<acc>_fqtmp`` scratch folder
    ``_store_fetch`` points fasterq-dump at, and fasterq-dump's own on-disk cache directory
    (``.sra-cache``). None of these should be counted as a downloaded accession by
    ``scan_downloads`` or the status command's on-disk inventory.
    """
    return name.endswith("_temp") or name.endswith("_fqtmp") or name == ".sra-cache"


def transient_bytes(folder: Union[str, Path]) -> int:
    """Total bytes held by every transient artifact directly under ``folder``.

    Sums the size of every file under each entry of ``folder`` whose name
    ``is_transient_folder`` accepts (an ``<acc>_temp`` build directory kept after a failed
    download, or a ``.sra-cache`` archive cache), recursing into their contents. Used to warn
    when a download run has left large temporary artifacts on disk. Returns 0 when
    ``folder`` does not exist or holds no such entry; a file that disappears mid-scan (a
    concurrent cleanup) is simply skipped rather than raising.
    """
    path = Path(folder)
    if not path.is_dir():
        return 0
    total = 0
    for entry in path.iterdir():
        if not entry.is_dir() or not is_transient_folder(entry.name):
            continue
        for sub in entry.rglob("*"):
            if not sub.is_file():
                continue
            try:
                total += sub.stat().st_size
            except OSError:
                continue
    return total


def _ignore_missing(func, target, exc: BaseException) -> None:
    """``onexc`` callback for ``shutil.rmtree``: swallow a missing-file race, re-raise anything else.

    On a volume that stores each file's AppleDouble sidecar (``._<name>``) next to it, macOS can
    delete ``._X`` together with ``X``, so ``rmtree`` reaching ``._X`` afterwards finds it already
    gone. That race is not a real failure to remove the directory, so a ``FileNotFoundError`` is
    ignored; ``shutil.rmtree``'s ``onexc`` (Python 3.12+) hands this callback the exception object
    directly, unlike the older ``onerror`` callback's ``sys.exc_info()`` tuple.
    """
    if isinstance(exc, FileNotFoundError):
        return
    raise exc


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree if present, logging on failure instead of raising."""
    try:
        if path.exists():
            shutil.rmtree(path, onexc=_ignore_missing)
    except OSError as e:
        logger.warning(f"Could not remove directory {path}: {e}")


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
        except OSError as e:
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
    except (OSError, SecurityError) as e:
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
    except OSError as e:
        logger.warning(f"Could not remove {output_path}: {e}")
