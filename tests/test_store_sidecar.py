"""
Tests for metaquest.store.sidecar: the per-dataset JSON sidecar
``<root>/sra/<ACC>/<ACC>.json`` that makes the store self-describing.
"""

import hashlib
import json
import logging

from metaquest.store.sidecar import (
    SIDECAR_SCHEMA,
    Sidecar,
    build_sidecar,
    ncbi_from_metadata_xml,
    read_sidecar,
    write_sidecar,
)

NCBI_XML = """<?xml version="1.0"?>
<EXPERIMENT_PACKAGE_SET>
    <EXPERIMENT_PACKAGE>
        <EXPERIMENT>
            <IDENTIFIERS>
                <PRIMARY_ID>EXP1</PRIMARY_ID>
            </IDENTIFIERS>
            <LIBRARY_DESCRIPTOR>
                <LIBRARY_STRATEGY>WGS</LIBRARY_STRATEGY>
                <LIBRARY_LAYOUT>
                    <PAIRED/>
                </LIBRARY_LAYOUT>
            </LIBRARY_DESCRIPTOR>
            <PLATFORM>
                <ILLUMINA>
                    <INSTRUMENT_MODEL>Illumina HiSeq 2500</INSTRUMENT_MODEL>
                </ILLUMINA>
            </PLATFORM>
        </EXPERIMENT>
        <RUN_SET>
            <RUN accession="SRR1" total_spots="10" total_bases="1400" size="4744553813">
                <IDENTIFIERS>
                    <PRIMARY_ID>SRR1</PRIMARY_ID>
                </IDENTIFIERS>
                <SRAFiles>
                    <SRAFile filename="SRR1" md5="abc" semantic_name="run"/>
                </SRAFiles>
            </RUN>
        </RUN_SET>
    </EXPERIMENT_PACKAGE>
</EXPERIMENT_PACKAGE_SET>"""


def _write_fastq(path, n_records):
    """Write ``n_records`` plain FASTQ records (4 lines each) to ``path``."""
    lines = []
    for i in range(n_records):
        lines.append(f"@read{i}\nACGT\n+\nIIII\n")
    path.write_text("".join(lines))


def _make_sidecar(**overrides):
    fields = dict(
        accession="SRR1",
        state="complete",
        layout="PAIRED",
        downloaded="2026-09-06T00:00:00+00:00",
        tool="fasterq-dump",
        tool_version="3.0.0",
        compression="gzip",
        files=[{"name": "SRR1_1.fastq.gz", "bytes": 100, "md5": "deadbeef", "reads": 10}],
        reads_per_mate=10,
        bases_total=None,
        ncbi={"spots": 10},
        completeness={"method": "spots", "ratio": 1.0, "verdict": "complete"},
        stats={},
        stats_computed=None,
    )
    fields.update(overrides)
    return Sidecar(**fields)


def test_sidecar_round_trip(tmp_path):
    sidecar = _make_sidecar()
    path = tmp_path / "SRR1.json"

    written_path = write_sidecar(path, sidecar)
    assert written_path == path
    assert path.is_file()

    loaded = read_sidecar(path)
    assert loaded == sidecar


def test_write_sidecar_is_atomic_and_sorted(tmp_path):
    sidecar = _make_sidecar()
    path = tmp_path / "SRR1.json"
    write_sidecar(path, sidecar)

    raw = path.read_text()
    data = json.loads(raw)
    assert data["accession"] == "SRR1"
    assert data["schema"] == SIDECAR_SCHEMA
    # sorted keys means "accession" (first alphabetically among our keys) precedes "tool"
    assert raw.index('"accession"') < raw.index('"tool"')
    # no leftover temp file
    leftovers = [p for p in tmp_path.iterdir() if p.name != "SRR1.json"]
    assert leftovers == []


def test_read_sidecar_missing_file_returns_none(tmp_path, caplog):
    path = tmp_path / "missing.json"
    with caplog.at_level(logging.WARNING):
        result = read_sidecar(path)
    assert result is None


def test_read_sidecar_invalid_json_returns_none(tmp_path, caplog):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with caplog.at_level(logging.WARNING):
        result = read_sidecar(path)
    assert result is None
    assert "bad.json" in caplog.text


def test_sidecar_from_dict_tolerant_of_missing_keys():
    sidecar = Sidecar.from_dict({})
    assert sidecar.accession == ""
    assert sidecar.schema == SIDECAR_SCHEMA
    assert sidecar.files == []


def test_sidecar_from_dict_ignores_unknown_keys():
    sidecar = Sidecar.from_dict({"accession": "SRR1", "made_up_field": "x"})
    assert sidecar.accession == "SRR1"
    assert not hasattr(sidecar, "made_up_field")


def test_build_sidecar_paired_layout_and_md5(tmp_path):
    acc_dir = tmp_path / "SRR1"
    acc_dir.mkdir()
    mate1 = acc_dir / "SRR1_1.fastq"
    mate2 = acc_dir / "SRR1_2.fastq"
    _write_fastq(mate1, 10)
    _write_fastq(mate2, 10)

    sidecar = build_sidecar("SRR1", acc_dir, {"spots": 10}, "3.0.0", "none")

    assert sidecar.layout == "PAIRED"
    assert sidecar.accession == "SRR1"
    assert sidecar.tool_version == "3.0.0"
    assert sidecar.compression == "none"
    assert len(sidecar.files) == 2

    by_name = {f["name"]: f for f in sidecar.files}
    assert by_name["SRR1_1.fastq"]["md5"] == hashlib.md5(mate1.read_bytes()).hexdigest()
    assert by_name["SRR1_2.fastq"]["md5"] == hashlib.md5(mate2.read_bytes()).hexdigest()
    assert by_name["SRR1_1.fastq"]["bytes"] == mate1.stat().st_size
    assert by_name["SRR1_1.fastq"]["reads"] == 10


def test_build_sidecar_verdict_complete(tmp_path):
    acc_dir = tmp_path / "SRR1"
    acc_dir.mkdir()
    _write_fastq(acc_dir / "SRR1.fastq", 10)

    sidecar = build_sidecar("SRR1", acc_dir, {"spots": 10}, "3.0.0", "none")

    assert sidecar.layout == "SINGLE"
    assert sidecar.state == "complete"
    assert sidecar.completeness["method"] == "spots"
    assert sidecar.completeness["verdict"] == "complete"
    assert sidecar.reads_per_mate == 10


def test_build_sidecar_verdict_partial(tmp_path):
    acc_dir = tmp_path / "SRR1"
    acc_dir.mkdir()
    _write_fastq(acc_dir / "SRR1.fastq", 10)

    sidecar = build_sidecar("SRR1", acc_dir, {"spots": 1000}, "3.0.0", "none")

    assert sidecar.state == "partial"
    assert sidecar.completeness["method"] == "spots"
    assert sidecar.completeness["verdict"] == "truncated"


def test_build_sidecar_verdict_unverified(tmp_path):
    acc_dir = tmp_path / "SRR1"
    acc_dir.mkdir()
    _write_fastq(acc_dir / "SRR1.fastq", 10)

    sidecar = build_sidecar("SRR1", acc_dir, {}, "3.0.0", "none")

    assert sidecar.state == "complete"
    assert sidecar.completeness["method"] == "unverified"
    assert sidecar.completeness["verdict"] == "unverified"


def test_build_sidecar_truncated_gzip_is_failed_state(tmp_path):
    acc_dir = tmp_path / "SRR1"
    acc_dir.mkdir()
    (acc_dir / "SRR1.fastq.gz").write_bytes(b"not a real gzip stream")

    sidecar = build_sidecar("SRR1", acc_dir, {"spots": 10}, "3.0.0", "gzip")

    assert sidecar.state == "failed"
    assert sidecar.error is not None
    assert sidecar.completeness["verdict"] == "unverified"
    assert sidecar.reads_per_mate is None


def test_ncbi_from_metadata_xml(tmp_path):
    xml_path = tmp_path / "SRR1.xml"
    xml_path.write_text(NCBI_XML)

    ncbi = ncbi_from_metadata_xml(xml_path)

    assert ncbi["spots"] == 10
    assert ncbi["bases"] == 1400
    assert ncbi["size"] == 4744553813
    assert ncbi["layout"] == "PAIRED"
    assert ncbi["files"] == [{"name": "SRR1", "md5": "abc"}]


def test_ncbi_from_metadata_xml_missing_file_returns_empty(tmp_path, caplog):
    xml_path = tmp_path / "missing.xml"

    with caplog.at_level(logging.WARNING):
        assert ncbi_from_metadata_xml(xml_path) == {}

    assert "Could not read NCBI metadata" in caplog.text
    assert "missing.xml" in caplog.text


def test_ncbi_from_metadata_xml_corrupt_xml_returns_empty(tmp_path, caplog):
    xml_path = tmp_path / "corrupt.xml"
    xml_path.write_text("<EXPERIMENT_PACKAGE_SET><unclosed>")

    with caplog.at_level(logging.WARNING):
        assert ncbi_from_metadata_xml(xml_path) == {}

    assert "Could not read NCBI metadata" in caplog.text
    assert "corrupt.xml" in caplog.text
