"""
The link a project keeps into the shared store.

One dataset is downloaded once, into ``<store>/sra/<ACC>/``. Every project
that uses it gets ``<project>/fastq/<ACC>`` as a symlink to that folder, so
the rest of MetaQuest reads the project path exactly as if the reads were
local. This module creates, removes and recognises those links, and reports
links whose target has gone away.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.store.layout import StorePaths, sra_dir

logger = logging.getLogger(__name__)

# How a project entry points at the store: a symlink whose target is written relative to the
# link ("relative") or as an absolute path ("absolute"), a copy of the folder ("copy"), or
# "auto", which picks relative when the two trees share a parent and absolute otherwise.
LINK_MODES = ("auto", "relative", "absolute", "copy")


def _shares_a_parent(store_root: Path, project_fastq: Path) -> bool:
    """True when both paths sit under a common directory other than the filesystem root.

    A relative symlink only survives moving the project if the store moves with it, which is
    the case when both live under one shared parent (a lab directory, a scratch mount). Two
    trees whose only common ancestor is ``/`` are unrelated, so an absolute link is safer.
    ``os.path.commonpath`` raises for paths on different drives, which is the same answer.
    """
    try:
        common = Path(os.path.commonpath([str(store_root), str(project_fastq)]))
    except ValueError:
        return False
    return str(common) != common.anchor


def _symlink_target(store_dataset: Path, link_parent: Path, mode: str) -> str:
    """The string to write into a symlink in ``link_parent`` pointing at ``store_dataset``.

    Both paths are already resolved by the caller, so a relative target is computed between
    real locations and a relative ``--fastq-folder`` still gets a link that works from
    anywhere.
    """
    if mode == "relative":
        return os.path.relpath(store_dataset, link_parent)
    return str(store_dataset)


def _clear_existing(link: Path) -> None:
    """Make room for a new link at ``link``, refusing to destroy real project data."""
    if link.is_symlink():
        link.unlink()
        return
    if link.exists():
        raise DataAccessError(
            f"{link} already exists and is not a store link; remove it first if the reads " "should come from the store"
        )


def link_dataset(
    project_fastq: Union[str, Path],
    accession: str,
    paths: StorePaths,
    mode: str = "auto",
) -> Path:
    """Point ``<project_fastq>/<accession>`` at the store's copy of ``accession``.

    Returns the project-side path. ``mode`` is one of ``auto`` (relative when the store and
    the project share a parent directory, absolute otherwise), ``relative``, ``absolute`` or
    ``copy``, which copies the dataset folder instead of linking it (for a project that must
    keep working when the store is unmounted). An existing symlink at the target is replaced;
    a real directory is never replaced, since it may hold reads this project downloaded
    itself.
    """
    if mode not in LINK_MODES:
        raise DataAccessError(f"Unknown link mode '{mode}'; expected one of {', '.join(LINK_MODES)}")

    store_dataset = sra_dir(paths, accession)
    if not store_dataset.is_dir():
        raise DataAccessError(f"Store has no dataset folder for {accession}: {store_dataset}")

    project_path = Path(project_fastq)
    project_path.mkdir(parents=True, exist_ok=True)
    link = project_path / accession
    # Resolved forms: the caller may pass a relative folder, and a relative symlink target
    # can only be worked out between two absolute paths.
    resolved_project = project_path.resolve()
    resolved_dataset = store_dataset.resolve()

    _clear_existing(link)

    if mode == "copy":
        shutil.copytree(store_dataset, link)
        logger.info("Copied %s from the store into %s", accession, link)
        return link

    if mode == "auto":
        mode = "relative" if _shares_a_parent(paths.root.resolve(), resolved_project) else "absolute"

    target = _symlink_target(resolved_dataset, resolved_project, mode)
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as e:
        raise DataAccessError(f"Cannot link {accession} into {project_path}: {e}") from e
    logger.info("Linked %s to the store copy at %s", link, target)
    return link


def unlink_dataset(project_fastq: Union[str, Path], accession: str) -> bool:
    """Remove ``<project_fastq>/<accession>`` when it is a symlink; return whether it was removed.

    Only a symlink is ever removed: a real directory holds this project's own reads, and the
    store's copy is never touched either way.
    """
    link = Path(project_fastq) / accession
    if not link.is_symlink():
        return False
    try:
        link.unlink()
    except OSError as e:
        raise DataAccessError(f"Cannot remove the store link {link}: {e}") from e
    return True


def is_store_link(path: Union[str, Path], paths: StorePaths) -> bool:
    """True when ``path`` is a symlink pointing into this store's ``sra`` folder.

    A dangling link still counts: it says where the dataset was meant to come from, which is
    what a caller deciding whether to relink needs to know.
    """
    link = Path(path)
    if not link.is_symlink():
        return False
    target = Path(os.path.realpath(link))
    store_sra = Path(os.path.realpath(paths.sra))
    return target == store_sra or store_sra in target.parents


def dangling_links(project_fastq: Union[str, Path]) -> List[str]:
    """Names in ``project_fastq`` that are symlinks with nothing at the other end, sorted.

    Reported rather than repaired: a missing target usually means the store is unmounted, in
    which case removing the links would lose the record of what the project uses.
    """
    folder = Path(project_fastq)
    if not folder.is_dir():
        return []
    return sorted(entry.name for entry in folder.iterdir() if entry.is_symlink() and not entry.exists())
