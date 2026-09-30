"""The external tools MetaQuest runs: one table, one version probe, one pre-flight check.

``TOOLS`` names each tool with the conda package that provides it, the oldest version known to
work (``min_version``, None when any version will do), and the commands that use it. A tool
marked ``optional`` is one those commands can run without: ``prefetch`` (``download_sra`` then
runs ``fasterq-dump`` directly), ``pigz`` (gzip is used instead), ``seqkit`` (statistics are
counted in Python) and ``megahit`` (needed only for ``extract_target_reads --assemble``).

``require_tools`` is the check a command runs before any work starts: it looks every named
tool up on ``PATH`` and, unless told not to, reads its version, and raises one
``ConfigurationError`` (exit code 3) listing every missing or too-old tool with a conda install
hint. A version that cannot be read is logged and not held against the tool, since the floor
cannot be judged; the tool's own error then surfaces if it really is unusable.
"""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from metaquest.core.constants import VERSION_PROBE_TIMEOUT
from metaquest.core.exceptions import ConfigurationError, SecurityError
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)

# Channels that between them provide every tool in the table (ncbi-datasets-cli and pigz are on
# conda-forge, the rest on bioconda).
CONDA_CHANNELS = "-c conda-forge -c bioconda"

_VERSION_PATTERN = re.compile(r"\d+(?:\.\d+)+")


@dataclass(frozen=True)
class ToolSpec:
    """One external tool: its conda package, oldest supported version and the commands that use it.

    ``optional`` marks a tool the commands in ``used_by`` can run without (``note`` says what
    happens then); ``version_args`` are the arguments that make it print its version.
    """

    name: str
    conda_package: str
    min_version: Optional[str]
    used_by: Tuple[str, ...]
    optional: bool = False
    version_args: Tuple[str, ...] = ("--version",)
    note: str = ""

    def install_hint(self) -> str:
        """The conda command that installs a suitable version of this tool."""
        package = f"'{self.conda_package}>={self.min_version}'" if self.min_version else self.conda_package
        provides = f" (provides {self.name})" if self.conda_package != self.name else ""
        return f"conda install {CONDA_CHANNELS} {package}{provides}"


_SPECS = (
    ToolSpec("fasterq-dump", "sra-tools", "3.0", ("download_sra",)),
    ToolSpec(
        "prefetch",
        "sra-tools",
        "3.0",
        ("download_sra",),
        optional=True,
        note="without it download_sra runs fasterq-dump directly against NCBI",
    ),
    ToolSpec(
        "pigz",
        "pigz",
        None,
        ("download_sra", "store_adopt"),
        optional=True,
        note="without it FASTQ files are compressed with gzip",
    ),
    # 2.17 is the first minimap2 release with --sam-hit-only (minimap2 NEWS.md).
    ToolSpec("minimap2", "minimap2", "2.17", ("extract_target_reads",)),
    # 1.10 is the first samtools release with the coverage subcommand.
    ToolSpec("samtools", "samtools", "1.10", ("extract_target_reads",)),
    ToolSpec(
        "megahit",
        "megahit",
        "1.2.9",
        ("extract_target_reads",),
        optional=True,
        note="needed only for extract_target_reads --assemble",
    ),
    ToolSpec("datasets", "ncbi-datasets-cli", None, ("genome_download", "genome_prepare")),
    ToolSpec(
        "seqkit",
        "seqkit",
        None,
        ("download_sra", "sra_profile", "sra_report"),
        optional=True,
        version_args=("version",),
        note="without it read and base counts are computed in Python, which is slower",
    ),
)

TOOLS: Dict[str, ToolSpec] = {spec.name: spec for spec in _SPECS}


def parse_version(text: str) -> Optional[Tuple[int, ...]]:
    """The first dotted number in ``text`` (``2.28-r1209`` gives ``(2, 28)``), or None when there is none."""
    match = _VERSION_PATTERN.search(text or "")
    if match is None:
        return None
    return tuple(int(part) for part in match.group(0).split("."))


def version_at_least(version: Tuple[int, ...], floor: str) -> bool:
    """Whether ``version`` is ``floor`` or later, comparing part by part with missing parts as 0."""
    wanted = tuple(int(part) for part in floor.split("."))
    width = max(len(version), len(wanted))
    return tuple(version) + (0,) * (width - len(version)) >= wanted + (0,) * (width - len(wanted))


@dataclass(frozen=True)
class ToolStatus:
    """What a probe found for one tool: its path, the line naming its version, and that version.

    ``path`` is None when the tool is not on ``PATH``; ``version`` is None when it could not be
    read, and ``error`` then says why when the tool could not be run.
    """

    name: str
    path: Optional[str]
    version_text: str = ""
    version: Optional[Tuple[int, ...]] = None
    error: str = ""

    @property
    def spec(self) -> ToolSpec:
        """The tool's entry in ``TOOLS``."""
        return TOOLS[self.name]

    @property
    def found(self) -> bool:
        """Whether the tool is on ``PATH``."""
        return self.path is not None

    @property
    def version_string(self) -> str:
        """The parsed version as dotted text, or an empty string when it is unknown."""
        return ".".join(str(part) for part in self.version) if self.version else ""

    @property
    def meets_floor(self) -> bool:
        """False only when the version is known and older than the table's ``min_version``."""
        floor = self.spec.min_version
        return self.version is None or floor is None or version_at_least(self.version, floor)

    def problem(self) -> Optional[str]:
        """Why this tool cannot be used (missing, or older than its floor), with an install hint; else None."""
        if not self.found:
            return f"{self.name} not found on PATH; install with: {self.spec.install_hint()}"
        if not self.meets_floor:
            return (
                f"{self.name} {self.version_string} at {self.path} is older than the {self.spec.min_version} "
                f"MetaQuest needs; install a newer one with: {self.spec.install_hint()}"
            )
        return None


def _output_text(value: object) -> str:
    """Captured output as text; anything else (a test double's attribute) as an empty string."""
    return value if isinstance(value, str) else ""


def probe_tool(name: str, timeout: float = VERSION_PROBE_TIMEOUT) -> ToolStatus:
    """Find ``name`` on ``PATH`` and read its version from what it prints (stdout, then stderr).

    The tool runs through ``run_secure`` with the table's ``version_args`` and a fixed
    ``timeout``, whatever the run's own tool timeout. A tool that is missing is not run. Raises
    ``KeyError`` for a name not in ``TOOLS``.
    """
    spec = TOOLS[name]
    path = shutil.which(name)
    if path is None:
        return ToolStatus(name, None)
    try:
        result = SecureSubprocess.run_secure(name, list(spec.version_args), timeout=timeout, check=False)
    except (SecurityError, subprocess.SubprocessError, OSError) as e:
        logger.debug("Could not run %s %s: %s", name, " ".join(spec.version_args), e)
        return ToolStatus(name, path, error=str(e))
    output = _output_text(result.stdout) + "\n" + _output_text(result.stderr)
    for line in output.splitlines():
        version = parse_version(line)
        if version is not None:
            return ToolStatus(name, path, line.strip(), version)
    returncode = result.returncode if isinstance(result.returncode, int) else 0
    error = f"exited with code {returncode}" if returncode else "printed no version number"
    return ToolStatus(name, path, error=error)


def require_tools(names: Iterable[str], check_versions: bool = True) -> None:
    """Check that every tool in ``names`` is on ``PATH`` and, with ``check_versions``, new enough.

    Raises one ``ConfigurationError`` listing every problem, each with its conda install hint,
    so a user fixes them all at once. A version that cannot be read is logged, not refused.
    """
    problems = []
    for name in names:
        spec = TOOLS[name]
        if not check_versions or spec.min_version is None:
            status = ToolStatus(name, shutil.which(name))
        else:
            status = probe_tool(name)
            if status.found and status.version is None:
                logger.warning(
                    "Could not read the %s version (%s); continuing without checking it against %s",
                    name,
                    status.error or "no version number printed",
                    spec.min_version,
                )
        problem = status.problem()
        if problem is not None:
            problems.append(problem)
    if problems:
        raise ConfigurationError(
            "External tools are missing or too old:\n  - " + "\n  - ".join(problems) + "\n"
            "Run 'metaquest doctor' to check every tool at once."
        )
