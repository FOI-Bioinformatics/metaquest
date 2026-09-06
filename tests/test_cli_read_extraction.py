"""Tests for the extract_target_reads CLI command."""

import argparse
import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from metaquest.cli.commands.read_extraction import ExtractTargetReadsCommand
from metaquest.cli.commands.status import StatusCommand
from metaquest.core.exceptions import ProcessingError
from metaquest.data.registry import load_registry, record_extraction, save_registry
from helpers_extraction import _fake_tools


def _args(tmp, **kwargs):
    base = dict(
        parsed_containment="parsed_containment.txt",
        genome_id="GCF_1",
        genome_fasta="GCF_1.fna",
        fastq_folder="fastq",
        output_folder="targeted",
        threshold=0.1,
        preset="sr",
        threads=4,
        min_mapq=0,
        temp_folder=None,
        allow_truncated=False,
        debug_keep_sam=False,
        assemble=False,
        assembly_threads=None,
        min_contig_len=None,
        dry_run=False,
        force=False,
        registry=str(Path(tmp) / "registry.json"),
        data_root=None,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def _tree(tmp):
    root = Path(tmp)
    table = root / "parsed_containment.txt"
    table.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.05\n")
    d = root / "fastq" / "SRR1"
    d.mkdir(parents=True)
    (d / "SRR1_1.fastq.gz").write_text("x")
    (d / "SRR1_2.fastq.gz").write_text("x")
    genome = root / "GCF_1.fna"
    genome.write_text(">s\nACGT\n")
    return root, table, genome


def _bootstrapped_tree(tmp):
    """A project whose extraction already happened on disk, with no registry yet."""
    root = Path(tmp)
    table = root / "parsed_containment.txt"
    table.write_text("\tGCF_1\nSRR1\t0.9\n")
    reads = root / "fastq" / "SRR1"
    reads.mkdir(parents=True)
    (reads / "SRR1_1.fastq.gz").write_text("x")
    (reads / "SRR1_2.fastq.gz").write_text("x")
    genome = root / "genomes" / "GCF_1.fna"
    genome.parent.mkdir(parents=True)
    genome.write_text(">s\nACGT\n")
    extracted = root / "targeted" / "SRR1"
    extracted.mkdir(parents=True)
    for name in ("GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz"):
        with gzip.open(extracted / name, "wt") as handle:
            handle.write("@r1\nACGT\n+\nIIII\n")
    return root, table, genome


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
        init=True,
        reconcile=False,
        export_tsv=None,
        next=False,
        list_missing=False,
        json=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _gzip_tree(tmp, mate1_reads=3, mate2_reads=3):
    """Like _tree, but with real gzip FASTQ content so count_fastq_reads can read it."""
    root = Path(tmp)
    table = root / "parsed_containment.txt"
    table.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.05\n")
    d = root / "fastq" / "SRR1"
    d.mkdir(parents=True)
    for name, n in (("SRR1_1.fastq.gz", mate1_reads), ("SRR1_2.fastq.gz", mate2_reads)):
        with gzip.open(d / name, "wt") as handle:
            handle.write("@r\nACGT\n+\nIIII\n" * n)
    genome = root / "GCF_1.fna"
    genome.write_text(">s\nACGT\n")
    return root, table, genome


def _two_sample_tree(tmp):
    """Like _tree, but with two samples above the threshold."""
    root = Path(tmp)
    table = root / "parsed_containment.txt"
    table.write_text("\tGCF_1\nSRR1\t0.9\nSRR2\t0.8\n")
    for acc in ("SRR1", "SRR2"):
        d = root / "fastq" / acc
        d.mkdir(parents=True)
        (d / f"{acc}_1.fastq.gz").write_text("x")
        (d / f"{acc}_2.fastq.gz").write_text("x")
    genome = root / "GCF_1.fna"
    genome.write_text(">s\nACGT\n")
    return root, table, genome


class TestExtractTargetReadsCommand:
    def test_command_properties(self):
        cmd = ExtractTargetReadsCommand()
        assert cmd.name == "extract_target_reads"
        assert "target" in cmd.help.lower()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_extracts(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                )
            )
        assert rc == 0
        assert mock_run.called

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_returns_1_when_no_sample_yields_reads(self, mock_run, caplog):
        mock_run.side_effect = _fake_tools({"mapped": 0})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            with caplog.at_level("ERROR"):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                    )
                )
        assert rc == 1
        assert "No reads mapped to GCF_1" in caplog.text

    def test_execute_returns_1_when_no_sample_meets_threshold(self, caplog):
        """threshold above every sample's containment -> nothing selected."""
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            with caplog.at_level("ERROR"):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=2.0,
                    )
                )
        assert rc == 1
        assert "No sample meets containment >= 2.0 for GCF_1" in caplog.text

    def test_execute_returns_1_when_no_fastq_for_selected_samples(self, caplog):
        """A sample is selected but --fastq-folder does not contain its reads."""
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            with caplog.at_level("ERROR"):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "no-such-fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                    )
                )
        assert rc == 1
        assert "No FASTQ files found for the 1 selected sample(s) under" in caplog.text

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_dry_run(self, mock_run):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    dry_run=True,
                )
            )
        assert rc == 0
        mock_run.assert_not_called()

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_with_assembly(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    assemble=True,
                )
            )
        assert rc == 0
        tools = [c.args[0] for c in mock_run.call_args_list]
        assert "megahit" in tools

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_assembly_single_thread_on_macos(self, mock_run, monkeypatch):
        mock_run.side_effect = _fake_tools({})
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    threads=4,
                    assemble=True,
                )
            )
        assert rc == 0
        megahit_call = next(c for c in mock_run.call_args_list if c.args[0] == "megahit" and "-o" in c.args[1])
        args = megahit_call.args[1]
        assert args[args.index("--num-cpu-threads") + 1] == "1"

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_assembly_threads_override_on_macos(self, mock_run, monkeypatch):
        mock_run.side_effect = _fake_tools({})
        monkeypatch.setattr("metaquest.data.read_extraction.platform.system", lambda: "Darwin")
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    threads=4,
                    assemble=True,
                    assembly_threads=6,
                )
            )
        assert rc == 0
        megahit_call = next(c for c in mock_run.call_args_list if c.args[0] == "megahit" and "-o" in c.args[1])
        args = megahit_call.args[1]
        assert args[args.index("--num-cpu-threads") + 1] == "6"

    def test_execute_missing_genome_column_returns_1(self):
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_id="GCF_absent",
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                )
            )
        assert rc == 1

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_records_extraction_and_assembly(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    assemble=True,
                    registry=str(registry_file),
                )
            )
            assert rc == 0
            data = json.loads(registry_file.read_text())
        extraction = data["datasets"]["SRR1"]["extractions"]["GCF_1"]
        assert extraction["mapped_reads"] > 0
        assert extraction["assembly"]["contigs"] == 2

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_records_usage_for_extraction_and_assembly(self, mock_run):
        """Extraction and assembly are recorded as store catalogue usage for this genome."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.catalog import Catalog
        from metaquest.store.layout import init_store, store_paths

        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            store_root = root / "store"
            init_store(store_root)
            registry_file = root / "registry.json"

            registry = _load(registry_file)
            registry.project = {"id": "proj1", "name": "demo", "path": str(root), "created": "now"}
            _save(registry)

            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    assemble=True,
                    registry=str(registry_file),
                    data_root=str(store_root),
                )
            )
            assert rc == 0

            with Catalog(store_paths(store_root)) as catalog:
                catalog.migrate()
                rows = {
                    (r["accession"], r["genome_id"], r["stage"])
                    for r in catalog.conn.execute("SELECT accession, genome_id, stage FROM usage").fetchall()
                }
        assert ("SRR1", "GCF_1", "extracted") in rows
        assert ("SRR1", "GCF_1", "assembled") in rows

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_catalog_failure_leaves_extraction_outcome_unchanged(self, mock_run):
        """A broken catalogue write never changes the extraction's exit code or registry record."""
        from metaquest.data.registry import load_registry as _load, save_registry as _save
        from metaquest.store.layout import init_store

        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            store_root = root / "store"
            init_store(store_root)
            registry_file = root / "registry.json"

            registry = _load(registry_file)
            registry.project = {"id": "proj1", "name": "demo", "path": str(root), "created": "now"}
            _save(registry)

            with patch("metaquest.store.usage.catalog_write", side_effect=RuntimeError("locked")):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                        assemble=True,
                        registry=str(registry_file),
                        data_root=str(store_root),
                    )
                )
            assert rc == 0
            data = json.loads(registry_file.read_text())
        extraction = data["datasets"]["SRR1"]["extractions"]["GCF_1"]
        assert extraction["mapped_reads"] > 0
        assert extraction["assembly"]["contigs"] == 2

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_zero_mapped_sample_is_never_assembled(self, mock_run):
        """Leftover files of a zero-mapped sample must not reach megahit."""
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            leftovers = root / "targeted" / "SRR1"
            leftovers.mkdir(parents=True)
            names = ["GCF_1_1.fastq.gz", "GCF_1_2.fastq.gz", "GCF_1_s.fastq.gz"]
            for name in names:
                (leftovers / name).write_text("")
            seeded = load_registry(registry_file)
            record_extraction(
                seeded,
                "SRR1",
                "GCF_1",
                [leftovers / name for name in names],
                0,
                False,
                {"genome_fasta": str(genome), "preset": "sr", "threshold": 0.5},
            )
            save_registry(seeded)

            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    assemble=True,
                    registry=str(registry_file),
                )
            )
        assert rc == 1
        assert "megahit" not in [c.args[0] for c in mock_run.call_args_list]

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_dry_run_reports_would_be_skips(self, mock_run, caplog):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            args = _args(
                tmp,
                parsed_containment=str(table),
                genome_fasta=str(genome),
                fastq_folder=str(root / "fastq"),
                output_folder=str(root / "targeted"),
                threshold=0.5,
                registry=str(registry_file),
            )
            assert cmd.execute(args) == 0
            with caplog.at_level("INFO"):
                rc = cmd.execute(_args(tmp, **{**vars(args), "dry_run": True}))
        assert rc == 0
        assert "already extracted, would be skipped: SRR1" in caplog.text

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_rerun_keeps_the_recorded_assembly(self, mock_run):
        """A rerun that does not run megahit again leaves the recorded assembly untouched."""
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            args = _args(
                tmp,
                parsed_containment=str(table),
                genome_fasta=str(genome),
                fastq_folder=str(root / "fastq"),
                output_folder=str(root / "targeted"),
                threshold=0.5,
                assemble=True,
                registry=str(registry_file),
            )
            assert cmd.execute(args) == 0
            # Stand in for the megahit version that produced the assembly on the first run.
            seeded = load_registry(registry_file)
            seeded.datasets["SRR1"]["extractions"]["GCF_1"]["assembly"]["version"] = "v1.2.9"
            save_registry(seeded)

            assert cmd.execute(args) == 0
            second = json.loads(registry_file.read_text())["datasets"]["SRR1"]["extractions"]["GCF_1"]["assembly"]
        assert second["version"] == "v1.2.9"

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_bootstrapped_registry_skips_the_extraction(self, mock_run):
        """status --init infers records without parameters; a rerun must still skip the mapping."""
        mock_run.side_effect = _fake_tools({})
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _bootstrapped_tree(tmp)
            registry_file = root / "metaquest_registry.json"
            assert StatusCommand().execute(_status_args(root)) == 0
            assert registry_file.exists()
            calls_before = len(mock_run.call_args_list)

            rc = ExtractTargetReadsCommand().execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    registry=str(registry_file),
                )
            )
        assert rc == 0
        assert "minimap2" not in [c.args[0] for c in mock_run.call_args_list[calls_before:]]

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_records_are_checkpointed_when_a_later_assembly_fails(self, mock_run):
        """Each extraction and assembly is written as it completes, so a later failure keeps them."""
        mock_run.side_effect = _fake_tools({})

        def fail_on_second(reads, out_dir, **kwargs):
            if "SRR2" in str(out_dir):
                raise ProcessingError("megahit crashed")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "final.contigs.fa").write_text(">c1 len=100\nACGT\n")
            return Path(out_dir), True

        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _two_sample_tree(tmp)
            registry_file = root / "registry.json"
            with patch("metaquest.cli.commands.read_extraction.assemble_extracted_reads", side_effect=fail_on_second):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                        assemble=True,
                        registry=str(registry_file),
                    )
                )
            assert rc == 1
            datasets = json.loads(registry_file.read_text())["datasets"]
        assert datasets["SRR1"]["extractions"]["GCF_1"]["mapped_reads"] > 0
        assert datasets["SRR2"]["extractions"]["GCF_1"]["mapped_reads"] > 0
        assert datasets["SRR1"]["extractions"]["GCF_1"]["assembly"]["contigs"] == 1
        assert datasets["SRR2"]["extractions"]["GCF_1"].get("assembly") is None

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_second_run_skips_and_returns_0(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            args = _args(
                tmp,
                parsed_containment=str(table),
                genome_fasta=str(genome),
                fastq_folder=str(root / "fastq"),
                output_folder=str(root / "targeted"),
                threshold=0.5,
                registry=str(registry_file),
            )
            rc = cmd.execute(args)
            assert rc == 0
            calls_before = len(mock_run.call_args_list)

            rc = cmd.execute(args)
            assert rc == 0
            tools_after_second_run = [c.args[0] for c in mock_run.call_args_list[calls_before:]]
            assert "minimap2" not in tools_after_second_run
            assert "samtools" not in tools_after_second_run

            forced_args = _args(
                tmp,
                parsed_containment=str(table),
                genome_fasta=str(genome),
                fastq_folder=str(root / "fastq"),
                output_folder=str(root / "targeted"),
                threshold=0.5,
                registry=str(registry_file),
                force=True,
            )
            calls_before_forced = len(mock_run.call_args_list)
            rc = cmd.execute(forced_args)
            assert rc == 0
            tools_after_forced_run = [c.args[0] for c in mock_run.call_args_list[calls_before_forced:]]
            assert "minimap2" in tools_after_forced_run
            assert "samtools" in tools_after_forced_run

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_second_run_skips_after_the_project_directory_is_renamed(self, mock_run, tmp_path):
        """The registry stores paths relative to itself, so a project directory renamed
        between two runs still lets the second run recognise the earlier extraction."""
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        old_root = tmp_path / "a"
        old_root.mkdir()
        root, table, genome = _tree(old_root)
        registry_file = root / "registry.json"
        rc = cmd.execute(
            _args(
                str(old_root),
                parsed_containment=str(table),
                genome_fasta=str(genome),
                fastq_folder=str(root / "fastq"),
                output_folder=str(root / "targeted"),
                threshold=0.5,
                registry=str(registry_file),
            )
        )
        assert rc == 0
        calls_before = len(mock_run.call_args_list)

        new_root = tmp_path / "b"
        old_root.rename(new_root)
        registry_file = new_root / "registry.json"
        rc = cmd.execute(
            _args(
                str(new_root),
                parsed_containment=str(new_root / "parsed_containment.txt"),
                genome_fasta=str(new_root / "GCF_1.fna"),
                fastq_folder=str(new_root / "fastq"),
                output_folder=str(new_root / "targeted"),
                threshold=0.5,
                registry=str(registry_file),
            )
        )
        assert rc == 0
        tools_after_rename = [c.args[0] for c in mock_run.call_args_list[calls_before:]]
        assert "minimap2" not in tools_after_rename
        assert "samtools" not in tools_after_rename

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_records_filter_flags_min_mapq_and_index(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    min_mapq=20,
                    registry=str(registry_file),
                )
            )
            assert rc == 0
            data = json.loads(registry_file.read_text())
        extraction = data["datasets"]["SRR1"]["extractions"]["GCF_1"]
        assert extraction["filter_flags"] == "0x904"
        assert extraction["min_mapq"] == 20
        assert extraction["index"].endswith("GCF_1.sr.mmi")
        assert ".index" in extraction["index"]

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_caches_mate_read_counts_in_the_registry(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _gzip_tree(tmp, mate1_reads=3, mate2_reads=3)
            registry_file = root / "registry.json"
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    registry=str(registry_file),
                )
            )
            assert rc == 0
            data = json.loads(registry_file.read_text())
        assert data["datasets"]["SRR1"]["download"]["mate_reads"] == [3, 3]

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_skips_a_truncated_download_unless_allowed(self, mock_run, caplog):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            registry_file = root / "registry.json"
            registry = load_registry(registry_file)
            registry.datasets["SRR1"] = {
                "download": {"complete": {"verdict": "truncated", "reads_r1": 5, "expected_spots": 20}}
            }
            save_registry(registry)

            with caplog.at_level("WARNING"):
                rc = cmd.execute(
                    _args(
                        tmp,
                        parsed_containment=str(table),
                        genome_fasta=str(genome),
                        fastq_folder=str(root / "fastq"),
                        output_folder=str(root / "targeted"),
                        threshold=0.5,
                        registry=str(registry_file),
                    )
                )
        assert rc == 1
        assert "skipped SRR1: download truncated (5 of 20 spots); use --allow-truncated" in caplog.text
        assert not mock_run.called

        mock_run.reset_mock()
        with tempfile.TemporaryDirectory() as tmp2:
            root, table, genome = _tree(tmp2)
            registry_file = root / "registry.json"
            registry = load_registry(registry_file)
            registry.datasets["SRR1"] = {
                "download": {"complete": {"verdict": "truncated", "reads_r1": 5, "expected_spots": 20}}
            }
            save_registry(registry)
            rc = cmd.execute(
                _args(
                    tmp2,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    registry=str(registry_file),
                    allow_truncated=True,
                )
            )
        assert rc == 0
        assert mock_run.called

    @patch("metaquest.data.read_extraction.SecureSubprocess.run_secure")
    def test_execute_debug_keep_sam_leaves_the_sam_on_disk(self, mock_run):
        mock_run.side_effect = _fake_tools({})
        cmd = ExtractTargetReadsCommand()
        with tempfile.TemporaryDirectory() as tmp:
            root, table, genome = _tree(tmp)
            rc = cmd.execute(
                _args(
                    tmp,
                    parsed_containment=str(table),
                    genome_fasta=str(genome),
                    fastq_folder=str(root / "fastq"),
                    output_folder=str(root / "targeted"),
                    threshold=0.5,
                    debug_keep_sam=True,
                )
            )
            assert rc == 0
            assert list((root / "targeted" / "SRR1").glob("*.sam"))
