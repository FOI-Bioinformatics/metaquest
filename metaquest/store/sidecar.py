"""
Per-dataset sidecar: ``<root>/sra/<ACC>/<ACC>.json``.

The sidecar makes the store self-describing: what files exist for one
downloaded accession, their size, md5 and read count, the run layout
(``PAIRED``/``SINGLE``), NCBI's recorded facts about the run, whether the
download is complete against those facts, and whatever per-run statistics a
later step computes. Nothing here talks to NCBI or the network; it only
reads what is already on disk plus a caller-supplied ``ncbi`` dict.
"""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from metaquest.core.exceptions import DataAccessError
from metaquest.data.metadata import parse_metadata_xml
from metaquest.data.sra import (
    MATE1_SUFFIXES,
    MATE_SUFFIXES,
    count_fastq_reads,
    fastq_files,
    fastq_stem,
    verify_download,
)

logger = logging.getLogger(__name__)

# Bump when the sidecar's fields change shape; a sidecar written by an older schema is still
# read back by Sidecar.from_dict, which fills in defaults for whatever is missing.
SIDECAR_SCHEMA = 1

# Suffixes marking the second mate of a pair, i.e. MATE_SUFFIXES minus MATE1_SUFFIXES.
_MATE2_SUFFIXES = tuple(suffix for suffix in MATE_SUFFIXES if suffix not in MATE1_SUFFIXES)


@dataclass
class Sidecar:
    """One dataset's sidecar record.

    Every field has a default so ``from_dict`` tolerates a sidecar written by an older or
    partial schema, or one missing keys because a later processing step has not run yet.
    """

    accession: str = ""
    state: str = "unknown"
    layout: str = "SINGLE"
    downloaded: Optional[str] = None
    tool: str = "fasterq-dump"
    tool_version: str = ""
    compression: str = "none"
    files: List[Dict[str, Any]] = field(default_factory=list)
    reads_per_mate: Optional[int] = None
    bases_total: Optional[int] = None
    ncbi: Dict[str, Any] = field(default_factory=dict)
    completeness: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    stats_computed: Optional[str] = None
    schema: int = SIDECAR_SCHEMA
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form suitable for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sidecar":
        """Build a ``Sidecar`` from a plain dict, ignoring unknown keys and defaulting missing ones."""
        known_names = {f.name for f in dataclass_fields(cls)}
        filtered = {key: value for key, value in (data or {}).items() if key in known_names}
        return cls(**filtered)


def _md5_file(path: Union[str, Path]) -> str:
    """MD5 hex digest of the file at ``path``, read in 1 MiB chunks so a large file is never
    loaded into memory at once."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_layout(files: List[Path]) -> str:
    """``PAIRED`` when a mate-1 (``_1``/``_R1``) and a mate-2 (``_2``/``_R2``) file both exist,
    else ``SINGLE``."""
    stems = [fastq_stem(p) for p in files]
    has_mate1 = any(stem.endswith(MATE1_SUFFIXES) for stem in stems)
    has_mate2 = any(stem.endswith(_MATE2_SUFFIXES) for stem in stems)
    return "PAIRED" if has_mate1 and has_mate2 else "SINGLE"


def build_sidecar(
    accession: str,
    acc_dir: Union[str, Path],
    ncbi: Dict[str, Any],
    tool_version: str,
    compression: str,
) -> Sidecar:
    """Build a sidecar describing the files already downloaded for ``accession`` in ``acc_dir``.

    Stats every FASTQ file present (size, md5, read count via ``count_fastq_reads``); the
    layout comes from the file names, and completeness reuses
    ``metaquest.data.sra.verify_download`` against ``ncbi.get("spots")``. A truncated gzip
    file makes ``count_fastq_reads`` raise ``EOFError`` (or plain garbage raise ``OSError``);
    either is caught per file and turns the whole result into ``state="failed"`` with the
    error recorded, since a corrupt file cannot be verified against NCBI's spot count.
    """
    acc_path = Path(acc_dir)
    files = fastq_files(acc_path)
    layout = _detect_layout(files)

    reads_by_path: Dict[Path, Optional[int]] = {}
    error: Optional[str] = None
    for file_path in files:
        try:
            reads_by_path[file_path] = count_fastq_reads(file_path)
        except (EOFError, OSError) as exc:
            reads_by_path[file_path] = None
            error = f"{file_path.name}: {exc}"

    file_records = [
        {
            "name": file_path.name,
            "bytes": file_path.stat().st_size,
            "md5": _md5_file(file_path),
            "reads": reads_by_path[file_path],
        }
        for file_path in files
    ]

    reads_per_mate: Optional[int]
    if error is not None:
        completeness = {"method": "unverified", "ratio": None, "verdict": "unverified"}
        reads_per_mate = None
        state = "failed"
    else:
        verify = verify_download(accession, acc_path, ncbi.get("spots"))
        reads_per_mate = verify["reads_r1"]
        method = "spots" if ncbi.get("spots") else "unverified"
        completeness = {"method": method, "ratio": verify["ratio"], "verdict": verify["verdict"]}
        state = "partial" if verify["verdict"] == "truncated" else "complete"

    return Sidecar(
        accession=accession,
        state=state,
        layout=layout,
        downloaded=datetime.now(timezone.utc).isoformat(),
        tool="fasterq-dump",
        tool_version=tool_version,
        compression=compression,
        files=file_records,
        reads_per_mate=reads_per_mate,
        bases_total=None,
        ncbi=dict(ncbi),
        completeness=completeness,
        stats={},
        stats_computed=None,
        schema=SIDECAR_SCHEMA,
        error=error,
    )


def _to_int(value: Any) -> Optional[int]:
    """Best-effort int conversion; ``None`` for anything not numeric (including ``None`` itself)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ncbi_from_metadata_xml(xml_path: Union[str, Path]) -> Dict[str, Any]:
    """Map one NCBI metadata XML file's fields into the flat ``ncbi`` dict a sidecar records.

    ``spots``/``bases``/``size`` come from ``Run_Total_Spots``/``Run_Total_Bases``/``Run_Size``
    (int when present and numeric, else None); ``layout`` from ``Experiment_Library_Layout``;
    ``files`` holds one ``{"name", "md5"}`` entry built from ``Run_Filename``/``Run_MD5`` when
    either is present, else an empty list. Returns ``{}`` when the XML file is missing or
    unparsable (``parse_metadata_xml`` already logs a warning in that case).
    """
    parsed = parse_metadata_xml(xml_path)
    if not parsed:
        return {}

    files: List[Dict[str, Any]] = []
    name = parsed.get("Run_Filename")
    md5 = parsed.get("Run_MD5")
    if name is not None or md5 is not None:
        files.append({"name": name, "md5": md5})

    return {
        "spots": _to_int(parsed.get("Run_Total_Spots")),
        "bases": _to_int(parsed.get("Run_Total_Bases")),
        "size": _to_int(parsed.get("Run_Size")),
        "layout": parsed.get("Experiment_Library_Layout"),
        "files": files,
    }


def write_sidecar(path: Union[str, Path], sidecar: Sidecar) -> Path:
    """Write ``sidecar`` to ``path`` atomically (temp file plus ``os.replace``), sorted keys, indent 2."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(sidecar.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, target)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise DataAccessError(f"Cannot write sidecar {target}: {e}") from e
    return target


def read_sidecar(path: Union[str, Path]) -> Optional[Sidecar]:
    """Read the sidecar at ``path``, returning None with a logged warning when missing or invalid."""
    sidecar_path = Path(path)
    if not sidecar_path.is_file():
        logger.warning(f"Sidecar file not found: {sidecar_path}")
        return None
    try:
        data = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read sidecar {sidecar_path}: {e}")
        return None
    return Sidecar.from_dict(data)
