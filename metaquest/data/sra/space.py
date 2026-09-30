"""Free-space guard for downloads: refuse to start an accession that would not fit on disk.

A download writes to three places: the FASTQ output folder (a project's ``fastq/`` or the
store's ``tmp/``, where the uncompressed files are built before gzip), ``fasterq-dump``'s
temporary folder, and, with ``prefetch``, the ``.sra`` cache. Before each accession starts, the
guard estimates what it needs on each filesystem and checks it against the free space left
once the downloads already running are counted, so parallel workers cannot each see the same
free space and together fill the disk.

The estimate for an accession with a known run size (NCBI's ``.sra`` size, from the project
registry) is ``FASTQ_EXPANSION`` times that size for the output folder and again for the
temporary folder, plus the ``.sra`` size for the cache when ``prefetch`` is used. Needs of
locations that share a filesystem are added. An accession without a known size needs
``floor_bytes`` (``--min-free-gb``) on each filesystem instead. A filesystem whose free space
cannot be read is assumed to have room, as ``store_adopt`` does: refusing on an unreadable
``disk_usage`` would be worse than trying.
"""

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - import cycle: metaquest.store imports this package
    from metaquest.store.layout import StorePaths

logger = logging.getLogger(__name__)

# Uncompressed FASTQ bytes per byte of .sra archive. Measured on seven runs of the crispatus
# test store (Illumina paired-end, 2x150 bp): 6.98, 7.28, 7.33, 7.34, 7.63, 7.69 and 7.73;
# rounded up to the next whole number.
FASTQ_EXPANSION = 8

GB = 1024**3

# The locations a download writes to, as keys of the ``locations`` mapping.
OUTPUT, TEMP, CACHE = "output", "temp", "cache"

DISK_FULL_PREFIX = "disk-full: insufficient free space"


def _existing(path: Path) -> Path:
    """``path`` itself, or its nearest parent that exists (a folder a download will create)."""
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _device(path: Path) -> Optional[int]:
    """The filesystem (``st_dev``) ``path`` lies on, or None when it cannot be read."""
    try:
        return os.stat(_existing(path)).st_dev
    except OSError:
        return None


def mount_point(path: Path) -> Path:
    """The top folder of the filesystem ``path`` lies on (used to name it in messages)."""
    current = _existing(path)
    try:
        device = os.stat(current).st_dev
        while current != current.parent and os.stat(current.parent).st_dev == device:
            current = current.parent
    except OSError:
        pass
    return current


def _size(value: Any) -> Optional[int]:
    """A run size as a positive whole number of bytes, or None (registry values may be strings)."""
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def _gb(value: int) -> str:
    """Bytes as gigabytes with one decimal."""
    return f"{value / GB:.1f}"


def download_locations(
    fastq_path: Union[str, Path],
    temp_folder: Optional[Union[str, Path]] = None,
    sra_cache: Optional[Union[str, Path]] = None,
    store: Optional["StorePaths"] = None,
) -> Dict[str, Path]:
    """Where a download writes: the output, temporary and ``.sra`` cache folders.

    Mirrors the defaults of ``download_accession`` and the store downloader: with a store the
    files are built in the store's ``tmp`` folder, which also holds the cache and temporary
    folder unless they are given; without one they are built under ``fastq_path``, the cache
    defaults to ``<fastq>/.sra-cache`` and the temporary folder to the system's.
    """
    if store is not None:
        output = Path(store.tmp)
        return {
            OUTPUT: output,
            TEMP: Path(temp_folder) if temp_folder else output,
            CACHE: Path(sra_cache) if sra_cache else output / ".sra-cache",
        }
    fastq = Path(fastq_path)
    return {
        OUTPUT: fastq,
        TEMP: Path(temp_folder) if temp_folder else Path(tempfile.gettempdir()),
        CACHE: Path(sra_cache) if sra_cache else fastq / ".sra-cache",
    }


class SpaceGuard:
    """Reserve disk space for each accession before its download starts; see the module docstring.

    ``reserve`` and ``release`` are called from the worker threads; the reservations of
    downloads in progress are kept per filesystem under one lock. ``floor_bytes`` of zero
    turns the guard off.
    """

    def __init__(
        self,
        locations: Mapping[str, Path],
        floor_bytes: int,
        run_sizes: Mapping[str, Any],
        use_prefetch: bool,
        exempt: Iterable[str] = (),
    ) -> None:
        """Group ``locations`` by filesystem; ``exempt`` accessions (already in the store) need nothing."""
        self.floor_bytes = max(0, int(floor_bytes))
        self.use_prefetch = use_prefetch
        self._sizes = {acc: size for acc, value in (run_sizes or {}).items() if (size := _size(value)) is not None}
        self._exempt = frozenset(exempt)
        self._lock = threading.Lock()
        self._reserved: Dict[int, int] = {}
        self._held: Dict[str, Dict[int, int]] = {}
        # One representative folder and the roles it serves, per filesystem.
        self._paths: Dict[int, Path] = {}
        self._roles: Dict[int, List[str]] = {}
        for role, path in locations.items():
            device = _device(Path(path))
            if device is None:
                logger.debug("Free-space guard: cannot read the filesystem of %s; not checked", path)
                continue
            self._paths.setdefault(device, _existing(Path(path)))
            self._roles.setdefault(device, []).append(role)

    @property
    def enabled(self) -> bool:
        """False when the guard was turned off with a floor of zero."""
        return self.floor_bytes > 0

    def needs(self, accession: str) -> Dict[int, int]:
        """Bytes ``accession`` is expected to need, per filesystem."""
        if accession in self._exempt:
            return {}
        size = self._sizes.get(accession)
        if size is None:
            return {device: self.floor_bytes for device in self._paths}
        per_role = {
            OUTPUT: FASTQ_EXPANSION * size,
            TEMP: FASTQ_EXPANSION * size,
            CACHE: size if self.use_prefetch else 0,
        }
        needs = {device: sum(per_role.get(role, 0) for role in roles) for device, roles in self._roles.items()}
        return {device: need for device, need in needs.items() if need > 0}

    def _free(self, device: int) -> Optional[int]:
        """Free bytes on ``device``, or None when they cannot be read."""
        path = self._paths[device]
        try:
            return shutil.disk_usage(path).free
        except OSError as e:
            logger.warning("Could not check free space on %s: %s", path, e)
            return None

    def reserve(self, accession: str) -> Optional[str]:
        """Reserve the space ``accession`` needs; None when it fits, else the refusal message.

        The message starts with ``disk-full:``, so the download loops treat it like a disk
        that filled up during a download.
        """
        if not self.enabled:
            return None
        needs = self.needs(accession)
        with self._lock:
            for device, need in needs.items():
                free = self._free(device)
                if free is None:
                    continue
                available = free - self._reserved.get(device, 0)
                if available < need:
                    return (
                        f"{DISK_FULL_PREFIX} on {mount_point(self._paths[device])}: "
                        f"{_gb(max(available, 0))} GB free, about {_gb(need)} GB needed"
                    )
            for device, need in needs.items():
                self._reserved[device] = self._reserved.get(device, 0) + need
            self._held[accession] = needs
        return None

    def release(self, accession: str) -> None:
        """Return the space reserved for ``accession`` (its download ended, however it ended)."""
        with self._lock:
            for device, need in self._held.pop(accession, {}).items():
                self._reserved[device] = max(0, self._reserved.get(device, 0) - need)

    def preflight(self, accessions: Iterable[str]) -> List[str]:
        """Warnings for filesystems that cannot hold every download of ``accessions`` at once.

        Only accessions with a known run size are added up; the run still starts, and each
        download is checked again by ``reserve`` before it begins.
        """
        if not self.enabled:
            return []
        totals: Dict[int, int] = {}
        counted = 0
        for accession in accessions:
            if accession in self._exempt or accession not in self._sizes:
                continue
            counted += 1
            for device, need in self.needs(accession).items():
                totals[device] = totals.get(device, 0) + need
        warnings = []
        for device, total in totals.items():
            free = self._free(device)
            if free is not None and free < total:
                warnings.append(
                    f"The {counted} downloads with a known size together need up to about {_gb(total)} GB on "
                    f"{mount_point(self._paths[device])}, which has {_gb(free)} GB free; "
                    "downloads that do not fit when they are due to start will be refused"
                )
        return warnings
