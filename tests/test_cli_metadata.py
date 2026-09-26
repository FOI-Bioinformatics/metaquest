"""Tests for the metadata CLI commands' handling of parsed metadata (single-pass parsing, task 6)."""

import argparse
import json
from unittest.mock import patch

import pandas as pd

import metaquest.data.metadata as metadata_module
from metaquest.cli.commands.metadata import DownloadMetadataCommand, ParseMetadataCommand
from tests.perf_fixtures import write_metadata_folder


def _parse_args(tmp_path, folder):
    return argparse.Namespace(
        metadata_folder=str(folder),
        metadata_table_file=str(tmp_path / "metadata_table.txt"),
        registry=str(tmp_path / "metaquest_registry.json"),
    )


def test_parse_metadata_command_records_every_run_without_iterating_rows_as_series(tmp_path):
    """Every run is recorded from the table without building one pandas Series per row."""
    folder = tmp_path / "metadata"
    write_metadata_folder(folder, count=6, per_file=14, pool=30)

    with patch.object(pd.DataFrame, "iterrows", side_effect=AssertionError("iterrows used")):
        assert ParseMetadataCommand().execute(_parse_args(tmp_path, folder)) == 0

    datasets = json.loads((tmp_path / "metaquest_registry.json").read_text())["datasets"]
    assert sorted(datasets) == [f"SRR{1000000 + i}" for i in range(6)]
    first = datasets["SRR1000000"]["metadata"]
    assert first["run_total_spots"] == 1000000
    assert first["run_md5"] == f"{1:032x}"
    assert first["organism"] == "soil metagenome"
    assert first["collection_date"] == "value 0 of collection_date"
    assert first["library_layout"] == "SINGLE"
    assert first["platform"] == "ILLUMINA"
    assert first["xml"] == "metadata/SRR1000000_metadata.xml"


def test_parse_metadata_command_records_nothing_for_an_empty_table(tmp_path):
    """An empty folder yields an empty table and no dataset entries."""
    folder = tmp_path / "metadata"
    folder.mkdir()
    assert ParseMetadataCommand().execute(_parse_args(tmp_path, folder)) == 0
    assert json.loads((tmp_path / "metaquest_registry.json").read_text()).get("datasets", {}) == {}


def test_download_metadata_command_parses_each_downloaded_file_once(tmp_path):
    """Recording the fields of freshly downloaded files costs one lxml parse per file."""
    folder = tmp_path / "metadata"
    paths = write_metadata_folder(folder, count=3, per_file=14, pool=30)
    downloaded = {path.name.split("_")[0]: path for path in paths}
    args = argparse.Namespace(
        email="a@b.c",
        matches_folder=str(tmp_path / "matches"),
        metadata_folder=str(folder),
        threshold=0.0,
        dry_run=False,
        accessions_file=None,
        api_key=None,
        batch_size=200,
        registry=str(tmp_path / "metaquest_registry.json"),
        data_root=None,
    )

    real_parse = metadata_module.etree.parse
    with (
        patch("metaquest.cli.commands.metadata.download_metadata", return_value=downloaded),
        patch.object(metadata_module.etree, "parse", side_effect=real_parse) as spy,
    ):
        assert DownloadMetadataCommand().execute(args) == 0

    assert sorted(str(c.args[0]) for c in spy.call_args_list) == sorted(str(p) for p in paths)
    datasets = json.loads((tmp_path / "metaquest_registry.json").read_text())["datasets"]
    assert datasets["SRR1000002"]["metadata"]["run_total_spots"] == 1000026
