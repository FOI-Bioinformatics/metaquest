"""Round-trip tests for the typed registry blocks in ``metaquest.data.registry_blocks``.

The fixture ``tests/fixtures/registry_v0.4.0.json`` was written by the ``record_*`` writers as
they stood before the blocks existed. Every block must come back out of ``from_dict``/``to_dict``
exactly as it went in, including keys no writer knows about.
"""

import json
import shutil
from pathlib import Path

import pytest

from metaquest.data import registry as R
from metaquest.data import registry_blocks as B
from metaquest.data.registry_blocks import (
    AnalysisEntry,
    AssemblyBlock,
    DownloadBlock,
    ExclusionBlock,
    ExportEntry,
    ExtractionBlock,
    FileEntry,
    MetadataBlock,
    ProjectBlock,
    ScreeningBlock,
    ScreeningEntry,
    SelectionBlock,
    StoreBlock,
    Verdict,
)

FIXTURE = Path(__file__).parent / "fixtures" / "registry_v0.4.0.json"

DATASET_BLOCKS = {
    "screening": ScreeningBlock,
    "selection": SelectionBlock,
    "exclusion": ExclusionBlock,
    "download": DownloadBlock,
    "metadata": MetadataBlock,
}


@pytest.fixture
def fixture_data():
    return json.loads(FIXTURE.read_text())


def test_fixture_covers_every_block(fixture_data):
    seen = set()
    for record in fixture_data["datasets"].values():
        seen.update(record)
    assert set(DATASET_BLOCKS) | {"extractions", "analyses"} <= seen
    assert len(fixture_data["datasets"]) == 5


def test_every_dataset_block_round_trips(fixture_data):
    for accession, record in fixture_data["datasets"].items():
        for key, cls in DATASET_BLOCKS.items():
            if key in record:
                assert cls.from_dict(record[key]).to_dict() == record[key], (accession, key)
        for genome_id, extraction in record.get("extractions", {}).items():
            assert ExtractionBlock.from_dict(extraction).to_dict() == extraction, (accession, genome_id)
        for name, analysis in record.get("analyses", {}).items():
            assert AnalysisEntry.from_dict(analysis).to_dict() == analysis, (accession, name)


def test_project_and_store_keep_unknown_keys(fixture_data):
    project = ProjectBlock.from_dict(fixture_data["project"])
    store = StoreBlock.from_dict(fixture_data["store"])
    assert project.extra == {"notes": "kept by round trip"}
    assert store.extra == {"comment": "kept by round trip"}
    assert project.to_dict() == fixture_data["project"]
    assert store.to_dict() == fixture_data["store"]


def test_nested_blocks_are_typed(fixture_data):
    first = fixture_data["datasets"]["SRR0000001"]
    download = DownloadBlock.from_dict(first["download"])
    assert isinstance(download.files[0], FileEntry)
    assert isinstance(download.complete, Verdict)
    assert download.complete.verdict == "complete"
    assert download.mate_reads == [2, 2]
    extraction = ExtractionBlock.from_dict(first["extractions"]["GCF_000001"])
    assert isinstance(extraction.assembly, AssemblyBlock)
    assert extraction.assembly.extra["genome_fraction_estimate"] == 0.8
    screening = ScreeningBlock.from_dict(first["screening"])
    assert isinstance(screening.genomes["GCF_000001"], ScreeningEntry)
    project = ProjectBlock.from_dict(fixture_data["project"])
    assert isinstance(project.exports["results_table"], ExportEntry)


def test_partial_blocks_keep_their_shape(fixture_data):
    """A block a writer only ever wrote in part must not gain keys on the way back out."""
    verdict_only = fixture_data["datasets"]["SRR0000003"]["download"]
    assert DownloadBlock.from_dict(verdict_only).to_dict() == {"attempts": 0, "complete": {"verdict": "unverified"}}
    assembly_only = fixture_data["datasets"]["SRR0000004"]["extractions"]["GCF_000002"]
    assert set(ExtractionBlock.from_dict(assembly_only).to_dict()) == {"assembly", "files", "inferred", "mapped_reads"}


def test_from_dict_tolerates_missing_keys():
    block = DownloadBlock.from_dict({})
    assert block.state == ""
    assert block.files == []
    assert block.complete is None
    assert block.to_dict() == {}
    assert ExtractionBlock.from_dict(None).to_dict() == {}


def test_assignment_after_load_writes_the_key():
    block = DownloadBlock.from_dict({"attempts": 0})
    block.message = ""
    assert block.to_dict() == {"attempts": 0, "message": ""}
    block.discard("message")
    assert block.to_dict() == {"attempts": 0}


def test_fresh_blocks_omit_optional_keys_left_at_their_default():
    selection = SelectionBlock(selected=True, date="d", criteria={}, output="o")
    assert selection.to_dict() == {"selected": True, "date": "d", "criteria": {}, "output": "o"}
    store = StoreBlock(root="/r", linked=["A"])
    assert store.to_dict() == {"root": "/r", "mode": "symlink", "linked": ["A"]}


def test_registry_accessors_and_setters_round_trip(tmp_path, fixture_data):
    """Reading every block through an accessor and writing it back through its setter leaves the file as it was."""
    target = tmp_path / R.REGISTRY_FILENAME
    shutil.copy(FIXTURE, target)
    registry = R.load_registry(target)
    for accession, record in list(registry.datasets.items()):
        for key, getter, setter in (
            ("screening", B.screening_block, B.set_screening_block),
            ("selection", B.selection_block, B.set_selection_block),
            ("exclusion", B.exclusion_block, B.set_exclusion_block),
            ("download", B.download_block, B.set_download_block),
            ("metadata", B.metadata_block, B.set_metadata_block),
        ):
            if key in record:
                setter(registry, accession, getter(registry, accession))
        for genome_id in record.get("extractions", {}):
            B.set_extraction_block(registry, accession, genome_id, B.extraction_block(registry, accession, genome_id))
    B.set_project_block(registry, B.project_block(registry))
    B.set_store_block(registry, B.store_block(registry))
    R.save_registry(registry)

    written = json.loads(target.read_text())
    original = dict(fixture_data)
    assert written.pop("updated") and original.pop("updated")
    assert written == original
