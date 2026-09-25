"""Typed forms of the blocks a project registry records, per dataset and per project.

The registry file stays plain JSON (layout and ``SCHEMA_VERSION`` unchanged); these dataclasses
are the in-memory view writers build and readers consume instead of chains of ``dict.get``.

Every block converts with ``from_dict`` (tolerant of missing keys) and ``to_dict``, and the
conversion is faithful in both directions:

- A key the block does not declare is kept in ``extra`` and written back unchanged, so a key
  added by a newer version, or by hand, is never lost by a rewrite.
- A block loaded with ``from_dict`` writes back exactly the keys it was loaded with, plus any
  field assigned or changed in place since (``discard`` removes one again). A block that some
  writer only ever wrote in part, e.g. a download record holding nothing but a verdict, keeps
  that shape.
- A block constructed directly writes every field, except the optional ones (``ranked``,
  ``inferred``, a download's ``source``, ...) while they hold their default, which matches
  what the writers wrote before these classes existed.

Values are passed through without coercion, so an integer stays an integer and a JSON
``null`` stays ``None``.
"""

from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Self, Set, Tuple, Type, TypeVar

if TYPE_CHECKING:
    # Only for annotations: metaquest.data.registry imports this module.
    from metaquest.data.registry import Registry

# Field metadata keys: "omit" marks a field left out of a fresh block's dict while it holds its
# default; "load"/"dump" convert a nested value (skipped for None).
_OMIT = "omit"
_LOAD = "load"
_DUMP = "dump"
# Attributes of the base class itself, never written as keys.
_INTERNAL = ("extra", "_keys")


def _optional(default: Any = None) -> Any:
    """A field whose key a fresh block leaves out while it holds ``default``."""
    return field(default=default, metadata={_OMIT: True})


def _copy(value: Any) -> Any:
    """A shallow copy of a list or dict, so a block never aliases the registry's own containers."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _default_of(f: "Field[Any]") -> Any:
    """The value a field takes when its key is absent."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return None


_FIELDS: Dict[type, Tuple["Field[Any]", ...]] = {}


def _block_fields(cls: type) -> Tuple["Field[Any]", ...]:
    """The declared (key-carrying) fields of a block class, cached per class."""
    found = _FIELDS.get(cls)
    if found is None:
        found = _FIELDS[cls] = tuple(f for f in fields(cls) if f.name not in _INTERNAL)
    return found


@dataclass
class RegistryBlock:
    """Shared conversion logic; see the module docstring for the round-trip rules."""

    extra: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    # Declared keys the source dict held, plus fields assigned since; None for a fresh block.
    _keys: Optional[Set[str]] = field(default=None, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        keys = self.__dict__.get("_keys")
        if keys is not None and name not in _INTERNAL:
            keys.add(name)
        object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> Self:
        """Build the block from its JSON form; missing keys take the field defaults."""
        data = data or {}
        declared = {f.name: f for f in _block_fields(cls)}
        values: Dict[str, Any] = {}
        for name, f in declared.items():
            if name in data:
                value = data[name]
                load: Optional[Callable[[Any], Any]] = f.metadata.get(_LOAD)
                values[name] = load(value) if load is not None and value is not None else _copy(value)
        block = cls(**values)
        block.extra = {k: _copy(v) for k, v in data.items() if k not in declared}
        object.__setattr__(block, "_keys", set(values))
        return block

    def to_dict(self) -> Dict[str, Any]:
        """The JSON form: the keys this block holds (see the module docstring), then ``extra``."""
        out: Dict[str, Any] = {}
        for f in _block_fields(type(self)):
            value = getattr(self, f.name)
            if self._keys is not None and f.name not in self._keys and value == _default_of(f):
                continue
            if self._keys is None and f.metadata.get(_OMIT) and value == _default_of(f):
                continue
            dump: Optional[Callable[[Any], Any]] = f.metadata.get(_DUMP)
            out[f.name] = dump(value) if dump is not None and value is not None else _copy(value)
        out.update(self.extra)
        return out

    def discard(self, name: str) -> None:
        """Reset field ``name`` to its default and leave its key out of ``to_dict``."""
        f = {f.name: f for f in _block_fields(type(self))}[name]
        object.__setattr__(self, name, _default_of(f))
        if self._keys is not None:
            self._keys.discard(name)


def _nested(cls: type[RegistryBlock], default: Any = None, omit: bool = False) -> Any:
    """A field holding one nested block (or None)."""
    metadata = {_LOAD: cls.from_dict, _DUMP: lambda block: block.to_dict(), _OMIT: omit}
    return field(default=default, metadata=metadata)


def _nested_list(cls: type[RegistryBlock]) -> Any:
    """A field holding a list of nested blocks."""
    metadata = {
        _LOAD: lambda items: [cls.from_dict(item) for item in items],
        _DUMP: lambda blocks: [block.to_dict() for block in blocks],
    }
    return field(default_factory=list, metadata=metadata)


def _nested_map(cls: type[RegistryBlock], omit: bool = False) -> Any:
    """A field holding a mapping of name to nested block."""
    metadata = {
        _LOAD: lambda items: {k: cls.from_dict(v) for k, v in items.items()},
        _DUMP: lambda blocks: {k: block.to_dict() for k, block in blocks.items()},
        _OMIT: omit,
    }
    return field(default_factory=dict, metadata=metadata)


# ------------------------------------------------------------------ screening


@dataclass
class ScreeningEntry(RegistryBlock):
    """One genome's screening result for one accession."""

    containment: float = 0.0
    cani: Optional[float] = None
    csv: Optional[str] = None
    source: str = ""
    query_threshold: float = 0.0


@dataclass
class ScreeningBlock(RegistryBlock):
    """An accession's screening results, keyed by genome id."""

    date: str = ""
    genomes: Dict[str, ScreeningEntry] = _nested_map(ScreeningEntry)
    inferred: bool = _optional(False)


# ---------------------------------------------------------- selection, exclusion


@dataclass
class SelectionBlock(RegistryBlock):
    """Whether an accession is selected, and by which criteria and output file."""

    selected: bool = False
    date: str = ""
    criteria: Dict[str, Any] = field(default_factory=dict)
    output: str = ""
    ranked: Optional[List[Dict[str, Any]]] = _optional()
    inferred: bool = _optional(False)


@dataclass
class ExclusionBlock(RegistryBlock):
    """An accession's exclusion (or a reversed one, with ``excluded`` False)."""

    excluded: bool = False
    reason: str = ""
    source: str = ""
    date: str = ""


# -------------------------------------------------------------------- download


@dataclass
class FileEntry(RegistryBlock):
    """One downloaded read file: project-relative path, size and modification time."""

    path: str = ""
    bytes: int = 0
    mtime: str = ""


@dataclass
class Verdict(RegistryBlock):
    """A download's completeness verdict; producers differ in which keys they fill in."""

    method: Optional[str] = _optional()
    ratio: Optional[float] = _optional()
    verdict: Optional[str] = _optional()
    expected_spots: Optional[int] = _optional()
    reads_r1: Optional[int] = _optional()

    def carrying_counts_from(self, previous: Optional[Verdict]) -> Verdict:
        """This verdict, about to replace ``previous``, with the read counts it may inherit filled in.

        A count this verdict leaves as ``None`` inherits the previous count only when both carry
        the same verdict (a complete-to-complete relink describes the same reads). Across a
        verdict change the previous count describes other files, e.g. a truncated project copy
        relinked to a complete store copy, so it stays ``None``; ``store_verify --rescan`` can
        fill in a real count. Only ``reads_r1`` is carried: ``ratio`` and ``expected_spots``
        always come from the new verdict, so the result never mixes two downloads' verdicts.
        """
        if previous is not None and previous.verdict == self.verdict:
            if self.reads_r1 is None and previous.reads_r1 is not None:
                self.reads_r1 = previous.reads_r1
        return self


@dataclass
class DownloadBlock(RegistryBlock):
    """An accession's download outcome, completeness verdict and cached mate read counts."""

    attempts: int = 0
    state: str = ""
    date: str = ""
    files: List[FileEntry] = _nested_list(FileEntry)
    bytes_total: int = 0
    message: str = ""
    complete: Optional[Verdict] = _nested(Verdict, omit=True)
    source: Optional[str] = _optional()
    store_name: Optional[str] = _optional()
    mate_reads: Optional[List[int]] = _optional()
    # ``[[file name, size bytes, mtime], ...]`` of the mate files ``mate_reads`` was counted from.
    mate_reads_signature: Optional[List[List[Any]]] = _optional()
    inferred: bool = _optional(False)


# -------------------------------------------------------------------- metadata


@dataclass
class MetadataBlock(RegistryBlock):
    """The NCBI run metadata recorded for an accession."""

    xml: str = ""
    date: str = ""
    run_size: Any = None
    run_md5: Any = None
    assay_type: Any = None
    organism: Any = None
    collection_date: Any = None
    library_layout: Any = None
    platform: Any = None
    library_strategy: Any = None
    run_total_spots: Optional[int] = None
    run_total_bases: Optional[int] = None
    inferred: bool = _optional(False)


# ---------------------------------------------------------- extraction, assembly


@dataclass
class AssemblyBlock(RegistryBlock):
    """One assembly's provenance and stats; stats beyond the four required ones live in ``extra``."""

    date: str = ""
    dir: str = ""
    tool: str = ""
    version: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    contigs: int = 0
    total_bp: int = 0
    n50: int = 0
    largest: int = 0


@dataclass
class ExtractionBlock(RegistryBlock):
    """One accession's targeted read extraction against one genome, with its assembly if any."""

    date: str = ""
    genome_fasta: Optional[str] = None
    preset: Any = None
    threshold: Any = None
    filter_flags: Any = None
    min_mapq: Any = None
    index: Any = None
    mapped_reads: Optional[int] = None
    mapped_total: Optional[int] = None
    unequal_mates: bool = False
    files: List[str] = field(default_factory=list)
    breadth: Optional[float] = None
    mean_depth: Optional[float] = None
    coverage_tsv: Optional[str] = None
    assembly: Optional[AssemblyBlock] = _nested(AssemblyBlock)
    inferred: bool = _optional(False)


# ------------------------------------------------------------ analyses, exports


@dataclass
class AnalysisEntry(RegistryBlock):
    """The latest run of one named analysis for an accession."""

    date: str = ""
    output: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportEntry(RegistryBlock):
    """The latest run of one project-level export, e.g. the results table."""

    date: str = ""
    output: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------- project, store


@dataclass
class ProjectBlock(RegistryBlock):
    """This project's identity (bound by ``store_init`` or minted on first store use) and its exports."""

    id: Optional[str] = _optional()
    name: Optional[str] = _optional()
    path: Optional[str] = _optional()
    created: Optional[str] = _optional()
    exports: Dict[str, ExportEntry] = _nested_map(ExportEntry, omit=True)


@dataclass
class StoreBlock(RegistryBlock):
    """The shared data store this project is bound to, and the accessions it links from it."""

    root: Optional[str] = None
    mode: str = "symlink"
    linked: List[str] = field(default_factory=list)


# ------------------------------------------------------------ typed block access
#
# Readers take a typed copy of a block with ``<name>_block``; writers change the copy and store
# it with ``set_<name>_block``. The copy never aliases the registry's dicts, so a block is only
# written back when a setter is called. The module docstring says how unknown
# keys and partly written blocks survive the round trip.

_B = TypeVar("_B", bound=RegistryBlock)


def _dataset_block(registry: Registry, accession: str, key: str, cls: Type[_B]) -> Optional[_B]:
    raw = registry.datasets.get(accession, {}).get(key)
    return cls.from_dict(raw) if isinstance(raw, dict) else None


def screening_block(registry: Registry, accession: str) -> Optional[ScreeningBlock]:
    """``accession``'s screening block, or None when it was never screened."""
    return _dataset_block(registry, accession, "screening", ScreeningBlock)


def selection_block(registry: Registry, accession: str) -> Optional[SelectionBlock]:
    """``accession``'s selection block, or None when no selection ever named it."""
    return _dataset_block(registry, accession, "selection", SelectionBlock)


def exclusion_block(registry: Registry, accession: str) -> Optional[ExclusionBlock]:
    """``accession``'s exclusion block, or None when it was never excluded."""
    return _dataset_block(registry, accession, "exclusion", ExclusionBlock)


def download_block(registry: Registry, accession: str) -> Optional[DownloadBlock]:
    """``accession``'s download block, or None when nothing about its download was recorded."""
    return _dataset_block(registry, accession, "download", DownloadBlock)


def download_block_for_write(registry: Registry, accession: str) -> DownloadBlock:
    """``accession``'s download block, or the ``{"attempts": 0}`` block a first write starts from."""
    return download_block(registry, accession) or DownloadBlock.from_dict({"attempts": 0})


def set_mate_reads(registry: Registry, accession: str, mate_reads: List[int], signature: List[List[Any]]) -> None:
    """Cache ``accession``'s mate read counts, with the ``[name, size, mtime]`` signature of the files counted."""
    download = download_block_for_write(registry, accession)
    download.mate_reads = mate_reads
    download.mate_reads_signature = signature
    set_download_block(registry, accession, download)


def metadata_block(registry: Registry, accession: str) -> Optional[MetadataBlock]:
    """``accession``'s NCBI metadata block, or None when none was recorded."""
    return _dataset_block(registry, accession, "metadata", MetadataBlock)


def extraction_block(registry: Registry, accession: str, genome_id: str) -> Optional[ExtractionBlock]:
    """``accession``'s extraction against ``genome_id``, or None when there is none."""
    raw = registry.datasets.get(accession, {}).get("extractions", {}).get(genome_id)
    return ExtractionBlock.from_dict(raw) if isinstance(raw, dict) else None


def extraction_blocks(registry: Registry, accession: str) -> Dict[str, ExtractionBlock]:
    """Every recorded extraction of ``accession``, keyed by genome id (entries set to null are skipped)."""
    extractions = registry.datasets.get(accession, {}).get("extractions") or {}
    return {g: ExtractionBlock.from_dict(raw) for g, raw in extractions.items() if isinstance(raw, dict)}


def project_block(registry: Registry) -> ProjectBlock:
    """This project's identity and exports (empty fields when ``store_init`` never ran)."""
    return ProjectBlock.from_dict(registry.project)


def store_block(registry: Registry) -> StoreBlock:
    """The shared store this project is bound to (``root`` None when it is bound to none)."""
    return StoreBlock.from_dict(registry.store)


def set_screening_block(registry: Registry, accession: str, block: ScreeningBlock) -> None:
    """Store ``block`` as ``accession``'s screening block."""
    registry.datasets.setdefault(accession, {})["screening"] = block.to_dict()


def set_selection_block(registry: Registry, accession: str, block: SelectionBlock) -> None:
    """Store ``block`` as ``accession``'s selection block."""
    registry.datasets.setdefault(accession, {})["selection"] = block.to_dict()


def set_exclusion_block(registry: Registry, accession: str, block: ExclusionBlock) -> None:
    """Store ``block`` as ``accession``'s exclusion block."""
    registry.datasets.setdefault(accession, {})["exclusion"] = block.to_dict()


def set_download_block(registry: Registry, accession: str, block: DownloadBlock) -> None:
    """Store ``block`` as ``accession``'s download block."""
    registry.datasets.setdefault(accession, {})["download"] = block.to_dict()


def set_metadata_block(registry: Registry, accession: str, block: MetadataBlock) -> None:
    """Store ``block`` as ``accession``'s metadata block."""
    registry.datasets.setdefault(accession, {})["metadata"] = block.to_dict()


def set_extraction_block(registry: Registry, accession: str, genome_id: str, block: ExtractionBlock) -> None:
    """Store ``block`` as ``accession``'s extraction against ``genome_id``."""
    registry.datasets.setdefault(accession, {}).setdefault("extractions", {})[genome_id] = block.to_dict()


def set_project_block(registry: Registry, block: ProjectBlock) -> None:
    """Store ``block`` as this project's identity and exports."""
    registry.project = block.to_dict()


def set_store_block(registry: Registry, block: StoreBlock) -> None:
    """Store ``block`` as this project's store binding."""
    registry.store = block.to_dict()


# The dataset blocks bootstrap and reconcile can mark as reconstructed from disk.
_INFERABLE: Dict[str, Type[RegistryBlock]] = {
    "screening": ScreeningBlock,
    "selection": SelectionBlock,
    "download": DownloadBlock,
    "metadata": MetadataBlock,
    "extractions": ExtractionBlock,
}


def mark_inferred(registry: Registry, accession: str, key: str, genome_id: Optional[str] = None, **values: Any) -> None:
    """Flag a block just recorded for ``accession`` as reconstructed from disk, and set ``values`` on it.

    ``key`` names the dataset block (``"screening"``, ``"selection"``, ``"download"``,
    ``"metadata"``), or ``"extractions"`` together with ``genome_id`` for one extraction. A block
    that was never recorded is left alone.
    """
    container = registry.datasets.get(accession, {})
    name = key
    if genome_id is not None:
        container, name = container.get(key) or {}, genome_id
    raw = container.get(name)
    if not isinstance(raw, dict):
        return
    block = _INFERABLE[key].from_dict(raw)
    for field_name, value in {"inferred": True, **values}.items():
        setattr(block, field_name, value)
    container[name] = block.to_dict()
