"""CPUs and memory available to this process, as a batch scheduler or container limits them.

``os.cpu_count()`` reports every CPU of the node and ``/proc/meminfo`` all of its memory, while a
SLURM job or a container may use only part of either. These helpers read the limits the kernel
applies (the CPU affinity mask and the cgroup memory limit) and fall back to the SLURM job
variables, so a tool is not started with more threads or memory than the job was given.

The ``/proc`` and ``/sys`` roots are parameters so tests can point them at a fake tree.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# A cgroup v1 limit at or above this is the kernel's way of saying "no limit"
# (9223372036854771712 on 64-bit systems).
_CGROUP_V1_UNLIMITED = 2**60

# Share of a detected memory limit handed to a tool under --assembly-memory auto; the rest is
# left for the Python process itself and the page cache.
AUTO_MEMORY_FRACTION = 0.9

_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?|\.\d+)([KMGT]?)$", re.IGNORECASE)


def _positive_int(text: Optional[str]) -> Optional[int]:
    """``text`` as a whole number above zero, or None when it is missing or not one."""
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def available_cpus() -> int:
    """CPUs this process may run on.

    The affinity mask (``os.sched_getaffinity``, Linux only) reflects a SLURM cpuset or a
    ``taskset``; without it ``SLURM_CPUS_PER_TASK`` is used, then ``os.cpu_count()``, and 1 when
    nothing is known.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            cpus = len(getaffinity(0))
        except OSError:
            cpus = 0
        if cpus > 0:
            return cpus
    slurm = _positive_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    if slurm is not None:
        return slurm
    return os.cpu_count() or 1


def _read_text(path: Path) -> Optional[str]:
    """The stripped content of ``path``, or None when it cannot be read."""
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _cgroup_v2_path(proc_root: Path) -> Optional[str]:
    """This process's cgroup v2 path from ``/proc/self/cgroup`` (the ``0::<path>`` line), or None."""
    text = _read_text(proc_root / "self" / "cgroup")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:].strip() or "/"
    return None


def _cgroup_v2_limit(proc_root: Path, sys_root: Path) -> Optional[int]:
    """The smallest ``memory.max`` from this process's cgroup up to the root; None if all are ``max``.

    A parent group's limit also binds its children, so every level is read, not only the leaf.
    """
    group = _cgroup_v2_path(proc_root)
    if group is None:
        return None
    base = sys_root / "fs" / "cgroup"
    current = base.joinpath(*[part for part in group.split("/") if part])
    limits = []
    while True:
        value = _positive_int(_read_text(current / "memory.max"))
        if value is not None:
            limits.append(value)
        if current == base or base not in current.parents:
            break
        current = current.parent
    return min(limits) if limits else None


def _cgroup_v1_limit(sys_root: Path) -> Optional[int]:
    """``memory.limit_in_bytes`` of the cgroup v1 memory controller, or None when it means no limit."""
    value = _positive_int(_read_text(sys_root / "fs" / "cgroup" / "memory" / "memory.limit_in_bytes"))
    if value is None or value >= _CGROUP_V1_UNLIMITED:
        return None
    return value


def memory_limit_bytes(proc_root: Union[str, Path] = "/proc", sys_root: Union[str, Path] = "/sys") -> Optional[int]:
    """Memory this process may use, in bytes, or None when no limit can be found.

    Looked up in order: the cgroup v2 ``memory.max`` (smallest along the path to the root), the
    cgroup v1 ``memory.limit_in_bytes``, and ``SLURM_MEM_PER_NODE`` (megabytes). On macOS, and on
    a Linux host without a limit, the result is None.
    """
    proc_root, sys_root = Path(proc_root), Path(sys_root)
    limit = _cgroup_v2_limit(proc_root, sys_root)
    if limit is None:
        limit = _cgroup_v1_limit(sys_root)
    if limit is None:
        megabytes = _positive_int(os.environ.get("SLURM_MEM_PER_NODE"))
        limit = megabytes * 1024**2 if megabytes is not None else None
    return limit


def parse_memory(value: str, limit: Optional[int]) -> Optional[Union[int, float]]:
    """Resolve an ``--assembly-memory`` value to what megahit's ``--memory`` receives.

    - ``auto``: ``0.9 * limit`` in bytes when a limit was detected, else None (flag left out).
    - A fraction between 0 and 1 (``0.5``, ``.25``, ``1``): returned as a float; megahit applies
      it to the node's total memory, not to a cgroup limit.
    - A size (``32G``, ``32000M``, ``512K``, ``1T``, binary units) or a whole number of bytes:
      returned as an int.

    Raises:
        ValueError: For anything else, including 0 and a number above 1 with a decimal part
            but no unit.
    """
    text = str(value).strip()
    if text.lower() == "auto":
        return int(AUTO_MEMORY_FRACTION * limit) if limit else None
    match = _SIZE_RE.match(text)
    if match is None:
        raise ValueError(
            f"not a memory value: {value!r} (expected auto, a fraction such as 0.5, or a size such as 32G)"
        )
    number, unit = float(match.group(1)), match.group(2).upper()
    if not unit and 0 < number <= 1:
        return number
    if not unit and "." in match.group(1):
        raise ValueError(f"not a memory value: {value!r} (a fraction is at most 1; give a size with a unit)")
    size = int(number * _SIZE_UNITS[unit])
    if size <= 0:
        raise ValueError(f"not a memory value: {value!r} (must be above zero)")
    return size
