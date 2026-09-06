"""Tests for the `status` (local inventory) CLI command."""

import argparse
import gzip
import json
import tempfile
from pathlib import Path

from metaquest.cli.commands.status import StatusCommand
from metaquest.data.registry import (
    SCHEMA_VERSION,
    load_registry,
    record_exclusion,
    record_extraction,
    record_genome,
    record_selection,
    save_registry,
)


def _args(registry, **kwargs):
    base = dict(
        registry=str(registry),
        data_root=None,
        fastq_folder="fastq",
        metadata_folder="metadata",
        genomes_folder="genomes",
        accessions_file=None,
        parsed_containment=None,
        list_missing=False,
        json=False,
        targeted_folder="targeted",
        matches_folder="matches",
        stage=None,
        genome=None,
        init=False,
        reconcile=False,
        export_tsv=None,
        next=False,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def _make_tree(tmp):
    """Create a small fixture tree: one accession fully present, one absent."""
    root = Path(tmp)
    (root / "fastq" / "SRR1").mkdir(parents=True)
    (root / "fastq" / "SRR1" / "SRR1.fastq.gz").write_text("@r\nACGT\n+\nIIII\n")
    (root / "metadata").mkdir()
    (root / "metadata" / "SRR1_metadata.xml").write_text("<xml/>")
    (root / "genomes").mkdir()
    (root / "genomes" / "GCF_000006945.2.fna").write_text(">s\nACGT\n")
    return root


class TestStatusCommand:
    def test_command_properties(self):
        cmd = StatusCommand()
        assert cmd.name == "status"
        assert "local" in cmd.help.lower()

    def test_on_disk_inventory(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                )
            )
        assert result == 0
        out = capsys.readouterr().out
        assert "FASTQ accessions on disk : 1" in out
        assert "Metadata XML on disk     : 1" in out
        assert "Genome FASTA on disk     : 1" in out

    def test_reconcile_present_and_missing(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            accs = root / "accs.txt"
            accs.write_text("SRR1\nSRR2\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    accessions_file=str(accs),
                    list_missing=True,
                )
            )
        assert result == 0
        out = capsys.readouterr().out
        assert "FASTQ    : 1 present, 1 missing" in out
        assert "Metadata : 1 present, 1 missing" in out
        assert "SRR2" in out

    def test_json_output(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            accs = root / "accs.txt"
            accs.write_text("SRR1\nSRR2\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    accessions_file=str(accs),
                    json=True,
                )
            )
        assert result == 0
        report = json.loads(capsys.readouterr().out)
        assert report["on_disk"]["fastq_accessions"] == 1
        assert report["wanted"]["total"] == 2
        assert report["wanted"]["fastq_missing"] == ["SRR2"]

    def test_on_disk_inventory_ignores_transient_temp_folder(self, capsys):
        """A <acc>_temp folder holding a partial FASTQ must not inflate the on-disk count."""
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            (root / "fastq" / "SRR2_temp").mkdir(parents=True)
            (root / "fastq" / "SRR2_temp" / "SRR2_temp_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    json=True,
                )
            )
        assert result == 0
        report = json.loads(capsys.readouterr().out)
        assert report["on_disk"]["fastq_accessions"] == 1

    def test_parsed_containment_supplies_wanted_list(self, capsys):
        cmd = StatusCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_tree(tmp)
            table = root / "parsed_containment.txt"
            table.write_text("accession\tGCF_x\nSRR1\t0.9\nSRR2\t0.4\n")
            result = cmd.execute(
                _args(
                    root / "metaquest_registry.json",
                    fastq_folder=str(root / "fastq"),
                    metadata_folder=str(root / "metadata"),
                    genomes_folder=str(root / "genomes"),
                    parsed_containment=str(table),
                    json=True,
                )
            )
        assert result == 0
        report = json.loads(capsys.readouterr().out)
        assert report["wanted"]["total"] == 2

    def test_missing_accessions_file_errors(self, tmp_path):
        cmd = StatusCommand()
        result = cmd.execute(_args(tmp_path / "metaquest_registry.json", accessions_file="/nonexistent/accs.txt"))
        assert result == 1


def _status_args(root, **overrides):
    base = dict(
        fastq_folder=str(root / "fastq"),
        metadata_folder=str(root / "metadata"),
        genomes_folder=str(root / "genomes"),
        targeted_folder=str(root / "targeted"),
        matches_folder=str(root / "matches"),
        registry=str(root / "metaquest_registry.json"),
        data_root=None,
        accessions_file=None,
        parsed_containment=None,
        stage=None,
        genome=None,
        init=False,
        reconcile=False,
        export_tsv=None,
        next=False,
        list_missing=False,
        json=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _project_tree(root):
    for acc in ("SRR1", "SRR2"):
        d = root / "fastq" / acc
        d.mkdir(parents=True)
        (d / f"{acc}_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
    (root / "matches").mkdir()
    (root / "matches" / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\nSRR2,0.4,0.9\nSRR3,0.2,0.8\n")
    (root / "accessions.txt").write_text("SRR1\nSRR2\nSRR3\n")


class TestStatusWithRegistry:
    def test_without_registry_bootstraps_in_memory_and_hints(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(_status_args(tmp_path, accessions_file=str(tmp_path / "accessions.txt")))
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["registry"]["exists"] is False
        assert out["stages"]["screened"]["count"] == 3 and out["stages"]["downloaded"]["count"] == 2
        assert out["wanted"]["total"] == 3 and out["on_disk"]["fastq_accessions"] == 2
        assert not (tmp_path / "metaquest_registry.json").exists()

    def test_init_persists_bootstrap(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(
            _status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt"))
        )
        assert rc == 0 and (tmp_path / "metaquest_registry.json").exists()
        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR1"]["download"]["inferred"] is True

    def test_second_init_refuses_to_overwrite(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        capsys.readouterr()
        registry_path = tmp_path / "metaquest_registry.json"
        before = registry_path.read_text()
        rc = StatusCommand().execute(_status_args(tmp_path, init=True))
        assert rc == 1
        assert registry_path.read_text() == before

    def test_stage_and_genome_filters_list_accessions(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        rc = StatusCommand().execute(_status_args(tmp_path, stage="screened", genome=["GCF_1"], json=False))
        out = capsys.readouterr().out
        assert rc == 0 and "SRR1" in out and "SRR3" in out

    def test_next_suggests_download_and_extraction(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        capsys.readouterr()
        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        commands = [n["command"] for n in out["next"]]
        assert any(c.startswith("metaquest download_sra") for c in commands)  # SRR3 selected, not downloaded
        assert any(
            "extract_target_reads" in c and "GCF_1" in c for c in commands
        )  # SRR1/SRR2 downloaded, not extracted

    def test_next_extraction_command_is_runnable(self, tmp_path, capsys):
        """The extract suggestion carries the table it was selected from and a FASTA that exists."""
        _project_tree(tmp_path)
        (tmp_path / "genomes").mkdir()
        (tmp_path / "genomes" / "GCF_1.fasta").write_text(">s\nACGT\n")
        table = tmp_path / "tables" / "containment.txt"
        table.parent.mkdir()
        table.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.4\n")
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_selection(seeded, ["SRR1", "SRR2"], {"column": "GCF_1", "table": str(table)}, tmp_path / "acc.txt")
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        extract = next(s for s in out["next"] if "extract_target_reads" in s["command"])
        assert f"--parsed-containment {table}" in extract["command"]
        assert f"--genome-fasta {tmp_path / 'genomes' / 'GCF_1.fasta'}" in extract["command"]

    def test_next_uses_the_recorded_genome_fasta(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_genome(seeded, "GCF_1", tmp_path / "refs" / "wMel.fna", tmp_path / "manifest.csv")
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        extract = next(s for s in out["next"] if "extract_target_reads" in s["command"])
        assert f"--genome-fasta {tmp_path / 'refs' / 'wMel.fna'}" in extract["command"]

    def test_a_moved_project_still_reports_the_recorded_genome_fasta(self, tmp_path, capsys):
        """A registry recorded under the old project location still resolves its genome FASTA
        after the project directory is renamed, since the path is stored relative to the
        registry file rather than to the working directory at record time."""
        old_root = tmp_path / "a"
        _project_tree(old_root)
        StatusCommand().execute(_status_args(old_root, init=True))
        seeded = load_registry(old_root / "metaquest_registry.json")
        record_genome(seeded, "GCF_1", old_root / "refs" / "wMel.fna", old_root / "manifest.csv")
        save_registry(seeded)
        capsys.readouterr()

        new_root = tmp_path / "b"
        old_root.rename(new_root)

        StatusCommand().execute(_status_args(new_root, next=True))
        out = json.loads(capsys.readouterr().out)
        extract = next(s for s in out["next"] if "extract_target_reads" in s["command"])
        assert f"--genome-fasta {new_root / 'refs' / 'wMel.fna'}" in extract["command"]

    def test_next_drops_excluded_and_already_extracted_accessions(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, accessions_file=str(tmp_path / "accessions.txt")))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_exclusion(seeded, "SRR3", "16S amplicon")  # selected, not downloaded
        record_exclusion(seeded, "SRR2", "16S amplicon")  # downloaded
        record_extraction(seeded, "SRR1", "GCF_1", [], 0, False, {})  # zero mapped, but recorded
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, next=True))
        out = json.loads(capsys.readouterr().out)
        download = [s for s in out["next"] if "download_sra" in s["command"]]
        extract = [
            s for s in out["next"] if "extract_target_reads" in s["command"] and "--assemble" not in s["command"]
        ]
        assert download == []  # SRR3 was the only one left to download
        assert extract == []  # SRR1 has a record, SRR2 is excluded

    def test_reconcile_marks_missing_and_untracked(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        (tmp_path / "fastq" / "SRR2" / "SRR2_1.fastq").unlink()
        (tmp_path / "fastq" / "SRR8").mkdir()
        (tmp_path / "fastq" / "SRR8" / "SRR8_1.fastq").write_text("@r\nA\n+\nI\n")
        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        out = json.loads(capsys.readouterr().out)
        assert out["drift"]["recorded_missing"] == ["SRR2"] and out["drift"]["untracked_fastq"] == ["SRR8"]
        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR2"]["download"]["state"] == "missing"

    def test_reconcile_registers_untracked_work(self, tmp_path, capsys):
        """Untracked FASTQ and extractions are recorded as inferred, and still reported."""
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        (tmp_path / "fastq" / "SRR8").mkdir()
        (tmp_path / "fastq" / "SRR8" / "SRR8_1.fastq").write_text("@r\nACGT\n+\nIIII\n")
        extracted = tmp_path / "targeted" / "SRR1"
        extracted.mkdir(parents=True)
        for name in ("GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"):
            with gzip.open(extracted / name, "wt") as handle:
                handle.write("@r1\nACGT\n+\nIIII\n")

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        out = json.loads(capsys.readouterr().out)
        assert out["drift"]["untracked_fastq"] == ["SRR8"]
        assert out["drift"]["untracked_extractions"] == [["SRR1", "GCF_1"]]

        datasets = json.loads((tmp_path / "metaquest_registry.json").read_text())["datasets"]
        assert datasets["SRR8"]["download"]["state"] == "downloaded"
        assert datasets["SRR8"]["download"]["inferred"] is True
        assert datasets["SRR8"]["download"]["attempts"] == 0
        extraction = datasets["SRR1"]["extractions"]["GCF_1"]
        assert extraction["mapped_reads"] > 0 and extraction["inferred"] is True

    def test_reconcile_without_a_registry_refuses(self, tmp_path, caplog):
        _project_tree(tmp_path)
        with caplog.at_level("ERROR"):
            rc = StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        assert rc == 1
        assert not (tmp_path / "metaquest_registry.json").exists()
        assert "status --init" in caplog.text

    def test_report_carries_the_schema_version(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        out = json.loads(capsys.readouterr().out)
        assert out["registry"]["version"] == SCHEMA_VERSION

    def test_text_report_shows_selection_criteria_and_exclusion_reasons(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        record_selection(
            seeded,
            ["SRR1"],
            {"column": "GCF_1", "threshold": 0.5, "metadata_column": "organism", "metadata_value": "soil"},
            tmp_path / "accessions.txt",
        )
        record_exclusion(seeded, "SRR2", "16S amplicon")
        record_exclusion(seeded, "SRR3", "16S amplicon")
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "column GCF_1" in out and "threshold 0.5" in out and "organism = soil" in out
        assert "16S amplicon: 2" in out

    def test_export_tsv(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True, export_tsv=str(tmp_path / "registry")))
        assert (tmp_path / "registry_datasets.tsv").exists() and (tmp_path / "registry_extractions.tsv").exists()

    def test_text_report_shows_stage_matrix(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()
        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "Local inventory" in out and "Stages" in out and "screened" in out and "GCF_1" in out

    def test_downloads_report_lists_truncated_and_unverified(self, tmp_path, capsys):
        """report['downloads'] surfaces registry verdicts; stages.downloaded is unaffected."""
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        seeded.datasets["SRR1"]["download"]["complete"] = {"verdict": "truncated"}
        seeded.datasets["SRR2"]["download"]["complete"] = {"verdict": "unverified"}
        save_registry(seeded)
        capsys.readouterr()

        rc = StatusCommand().execute(_status_args(tmp_path))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["downloads"]["truncated"] == ["SRR1"]
        assert out["downloads"]["unverified"] == ["SRR2"]
        assert out["stages"]["downloaded"]["count"] == 2

    def test_text_report_shows_truncated_downloads_line(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        seeded.datasets["SRR1"]["download"]["complete"] = {"verdict": "truncated"}
        save_registry(seeded)
        capsys.readouterr()

        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "truncated downloads: 1 (SRR1)" in out


class TestStatusStorePlumbing:
    def test_no_store_configured_omits_store_section(self, tmp_path, capsys):
        _project_tree(tmp_path)
        rc = StatusCommand().execute(_status_args(tmp_path))
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert "store" not in out

    def test_no_store_configured_text_report_has_no_store_block(self, tmp_path, capsys):
        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, json=False))
        out = capsys.readouterr().out
        assert "Store" not in out

    def test_data_root_flag_adds_store_section(self, tmp_path, capsys):
        from metaquest.store.catalog import catalog_write
        from metaquest.store.layout import init_store

        _project_tree(tmp_path)
        store_root = tmp_path / "store"
        paths = init_store(store_root)
        with catalog_write(paths) as cat:
            cat.upsert_project("proj1", "P1", "/p1", "reg1")

        rc = StatusCommand().execute(_status_args(tmp_path, data_root=str(store_root)))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["store"]["root"] == str(store_root.resolve())
        assert out["store"]["datasets"] == {}

    def test_data_root_flag_adds_text_store_block_after_local_inventory(self, tmp_path, capsys):
        from metaquest.store.layout import init_store

        _project_tree(tmp_path)
        store_root = tmp_path / "store"
        init_store(store_root)

        StatusCommand().execute(_status_args(tmp_path, data_root=str(store_root), json=False))
        out = capsys.readouterr().out

        assert "Local inventory" in out
        assert "Store" in out
        assert out.index("Local inventory") < out.index("Store")

    def test_registry_recorded_store_root_is_used_when_no_flag(self, tmp_path, capsys):
        from metaquest.store.layout import init_store

        _project_tree(tmp_path)
        store_root = tmp_path / "store"
        init_store(store_root)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        seeded = load_registry(tmp_path / "metaquest_registry.json")
        seeded.store = {"root": str(store_root), "mode": "symlink", "linked": []}
        save_registry(seeded)
        capsys.readouterr()

        rc = StatusCommand().execute(_status_args(tmp_path))
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["store"]["root"] == str(store_root.resolve())


class TestStatusReconcileVerdict:
    def test_reconcile_fills_missing_download_verdict(self, tmp_path, capsys):
        """A download recorded before completeness verification existed (no `download.complete`)
        gets a verdict computed from what is on disk, once the registry records NCBI's spot count."""
        from metaquest.data.registry import record_metadata

        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()

        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_metadata(registry, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_total_spots": 100})
        save_registry(registry)

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        capsys.readouterr()

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        complete = data["datasets"]["SRR1"]["download"]["complete"]
        assert complete["verdict"] == "truncated"

    def test_reconcile_leaves_existing_verdict_alone(self, tmp_path, capsys):
        """A verdict already on file (e.g. from a fresh download run) is never recomputed."""
        from metaquest.data.registry import record_download, record_metadata

        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()

        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_metadata(registry, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_total_spots": 100})
        record_download(
            registry,
            "SRR1",
            "downloaded",
            tmp_path / "fastq",
            attempt=False,
            complete={"method": "spots", "ratio": 1.0, "verdict": "complete"},
        )
        save_registry(registry)

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        capsys.readouterr()

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert data["datasets"]["SRR1"]["download"]["complete"]["verdict"] == "complete"

    def test_reconcile_skips_store_sourced_downloads(self, tmp_path, capsys):
        """A download whose reads came from the shared store already has its own verdict
        pipeline; reconcile must not overwrite it."""
        from metaquest.data.registry import record_download, record_metadata

        _project_tree(tmp_path)
        StatusCommand().execute(_status_args(tmp_path, init=True))
        capsys.readouterr()

        registry = load_registry(tmp_path / "metaquest_registry.json")
        record_metadata(registry, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_total_spots": 100})
        record_download(
            registry, "SRR1", "downloaded", tmp_path / "fastq", attempt=False, source="store", store_name="SRR1"
        )
        save_registry(registry)

        StatusCommand().execute(_status_args(tmp_path, reconcile=True))
        capsys.readouterr()

        data = json.loads((tmp_path / "metaquest_registry.json").read_text())
        assert "complete" not in data["datasets"]["SRR1"]["download"]
