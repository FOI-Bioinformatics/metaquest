"""Tests for the per-project dataset registry (metaquest.data.registry)."""

import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from metaquest.core.exceptions import DataAccessError
from metaquest.data import registry as reg
from metaquest.data.read_extraction import summarise_contigs


def _fastq(path: Path, reads: int = 2, gz: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(reads))
    if gz:
        with gzip.open(path, "wt") as handle:
            handle.write(body)
    else:
        path.write_text(body)
    return path


def _project(tmp_path: Path) -> reg.ProjectPaths:
    return reg.ProjectPaths(
        fastq=tmp_path / "fastq",
        metadata=tmp_path / "metadata",
        genomes=tmp_path / "genomes",
        targeted=tmp_path / "targeted",
        matches=tmp_path / "matches",
    )


class TestLoadSave:
    def test_absent_registry_is_empty(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        assert r.datasets == {} and r.version == reg.SCHEMA_VERSION and r.path == tmp_path / "metaquest_registry.json"

    def test_round_trip_and_atomic_write(self, tmp_path):
        target = tmp_path / "metaquest_registry.json"
        r = reg.load_registry(target)
        reg.record_exclusion(r, "SRR1", "amplicon", source="user")
        reg.save_registry(r)
        assert target.exists()
        assert not list(tmp_path.glob("*.tmp.*")) and not list(tmp_path.glob("*.lock"))
        again = reg.load_registry(target)
        assert again.datasets["SRR1"]["exclusion"]["reason"] == "amplicon"
        assert json.loads(target.read_text())["version"] == reg.SCHEMA_VERSION

    def test_invalid_json_raises(self, tmp_path):
        target = tmp_path / "metaquest_registry.json"
        target.write_text("{not json")
        with pytest.raises(DataAccessError, match="not valid JSON"):
            reg.load_registry(target)

    def test_load_registry_wraps_os_error(self, tmp_path):
        """A read failure (permissions, I/O error) surfaces as DataAccessError, not a raw traceback."""
        target = tmp_path / "metaquest_registry.json"
        target.write_text("{}")
        with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            with pytest.raises(DataAccessError, match="Cannot read registry"):
                reg.load_registry(target)

    def test_save_registry_wraps_os_error(self, tmp_path):
        """A write failure (disk full, permissions) surfaces as DataAccessError, not a raw traceback."""
        target = tmp_path / "metaquest_registry.json"
        r = reg.load_registry(target)
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(DataAccessError, match="Cannot write registry"):
                reg.save_registry(r)
        # The lock and any temp file must not be left behind after the failure.
        assert not list(tmp_path.glob("*.lock"))
        assert not list(tmp_path.glob("*.tmp.*"))

    def test_registry_path_walks_up(self, tmp_path, monkeypatch):
        (tmp_path / "metaquest_registry.json").write_text("{}")
        sub = tmp_path / "targeted" / "SRR1"
        sub.mkdir(parents=True)
        assert reg.registry_path(start=sub).resolve() == (tmp_path / "metaquest_registry.json").resolve()
        assert reg.registry_path(explicit=str(tmp_path / "x.json")) == tmp_path / "x.json"

    def test_stale_lock_is_removed_and_fresh_lock_times_out(self, tmp_path, monkeypatch):
        target = tmp_path / "metaquest_registry.json"
        lock = tmp_path / "metaquest_registry.json.lock"
        monkeypatch.setattr(reg, "LOCK_STALE_SECONDS", 0.3)
        monkeypatch.setattr(reg, "LOCK_WAIT_SECONDS", 0.3)
        lock.write_text("1")
        os.utime(lock, (0, 0))  # ancient -> stale -> removed
        reg.save_registry(reg.load_registry(target))
        assert target.exists() and not lock.exists()
        lock.write_text("1")  # fresh lock held by "another process"
        with pytest.raises(DataAccessError, match="locked"):
            reg.save_registry(reg.load_registry(target))
        lock.unlink()

    def test_wait_deadline_is_its_own_constant(self, tmp_path, monkeypatch):
        """A long staleness window still gives up after the (separate) wait deadline."""
        target = tmp_path / "metaquest_registry.json"
        lock = tmp_path / "metaquest_registry.json.lock"
        monkeypatch.setattr(reg, "LOCK_STALE_SECONDS", 600.0)
        monkeypatch.setattr(reg, "LOCK_WAIT_SECONDS", 0.2)
        lock.write_text("1")
        with pytest.raises(DataAccessError, match="locked"):
            reg.save_registry(reg.load_registry(target))
        assert lock.exists()  # not reclaimed as stale
        lock.unlink()


class TestRegistryTransaction:
    def test_transaction_reloads_before_writing(self, tmp_path):
        """A change made between transactions survives, because each one loads from disk."""
        target = tmp_path / "metaquest_registry.json"
        reg.save_registry(reg.load_registry(target))
        other = reg.load_registry(target)
        reg.record_exclusion(other, "SRR9", "amplicon")
        reg.save_registry(other)

        with reg.registry_transaction(target) as r:
            reg.record_download(r, "SRR1", "failed", tmp_path / "fastq", message="timeout")

        data = json.loads(target.read_text())
        assert data["datasets"]["SRR9"]["exclusion"]["excluded"] is True
        assert data["datasets"]["SRR1"]["download"]["state"] == "failed"
        assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp.*"))

    def test_transaction_writes_nothing_on_error(self, tmp_path):
        target = tmp_path / "metaquest_registry.json"
        with pytest.raises(ValueError, match="boom"):
            with reg.registry_transaction(target) as r:
                reg.record_exclusion(r, "SRR1", "amplicon")
                raise ValueError("boom")
        assert not target.exists()
        assert not list(tmp_path.glob("*.lock"))


class TestNanToNone:
    def test_nan_to_none(self):
        assert reg.nan_to_none(float("nan")) is None
        assert reg.nan_to_none("SRR1") == "SRR1"


class TestRecords:
    def test_screening_selection_exclusion(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_screening(r, "SRR1", "GCF_1", 0.94611, 0.9974, "branchwater", 0.1, tmp_path / "bw.csv")
        reg.record_screening(r, "SRR1", "GCF_2", 0.2, None, "branchwater", 0.1, None)
        reg.record_selection(r, ["SRR1"], {"column": "GCF_1", "threshold": 0.5}, tmp_path / "accessions.txt")
        reg.record_selection(r, ["SRR2"], {"column": "GCF_1", "threshold": 0.9}, tmp_path / "accessions.txt")
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_1"]["containment"] == 0.9461
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_2"]["cani"] is None
        assert r.datasets["SRR1"]["selection"]["selected"] is False
        assert r.datasets["SRR2"]["selection"]["selected"] is True
        assert r.datasets["SRR2"]["selection"]["criteria"]["threshold"] == 0.9
        reg.record_exclusion(r, "SRR2", "16S amplicon")
        assert reg.query(r, "excluded") == ["SRR2"]
        reg.clear_exclusion(r, "SRR2")
        assert reg.query(r, "excluded") == []

    def test_download_records_files_and_attempts(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        _fastq(tmp_path / "fastq" / "SRR1" / "SRR1_1.fastq")
        _fastq(tmp_path / "fastq" / "SRR1" / "SRR1_2.fastq")
        reg.record_download(r, "SRR1", "failed", tmp_path / "fastq", message="timeout")
        reg.record_download(r, "SRR1", "downloaded", tmp_path / "fastq")
        dl = r.datasets["SRR1"]["download"]
        assert dl["state"] == "downloaded" and dl["attempts"] == 2
        assert [Path(f["path"]).name for f in dl["files"]] == ["SRR1_1.fastq", "SRR1_2.fastq"]
        assert dl["bytes_total"] == sum(f["bytes"] for f in dl["files"]) > 0
        reg.record_download(r, "SRR3", "skipped", tmp_path / "fastq", message="--max-downloads")
        assert r.datasets["SRR3"]["download"]["attempts"] == 0

        # A "downloaded" state recorded without an actual attempt (e.g. already present on
        # disk) must not bump attempts.
        _fastq(tmp_path / "fastq" / "SRR4" / "SRR4_1.fastq")
        reg.record_download(r, "SRR4", "downloaded", tmp_path / "fastq", attempt=False)
        assert r.datasets["SRR4"]["download"]["state"] == "downloaded"
        assert r.datasets["SRR4"]["download"]["attempts"] == 0

    def test_record_download_stores_and_preserves_the_completeness_verdict(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        verdict = {"verdict": "truncated", "reads_r1": 300000, "expected_spots": 48000000, "ratio": 0.0063}
        reg.record_download(r, "SRR1", "downloaded", tmp_path / "fastq", complete=verdict)
        assert r.datasets["SRR1"]["download"]["complete"] == verdict

        # A later call with no verdict (e.g. a skip recorded with no real attempt) must not
        # erase the verdict already on file.
        reg.record_download(r, "SRR1", "downloaded", tmp_path / "fastq", attempt=False)
        assert r.datasets["SRR1"]["download"]["complete"] == verdict

    def test_screening_source_and_threshold_are_per_genome(self, tmp_path):
        """A later screening from another source must not overwrite the first genome's provenance."""
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_screening(r, "SRR1", "GCF_1", 0.9, 0.99, "branchwater", 0.1, tmp_path / "bw.csv")
        reg.record_screening(r, "SRR1", "GCF_2", 0.3, None, "matches", 0.0, tmp_path / "m.csv")
        screening = r.datasets["SRR1"]["screening"]
        assert screening["genomes"]["GCF_1"]["source"] == "branchwater"
        assert screening["genomes"]["GCF_1"]["query_threshold"] == 0.1
        assert screening["genomes"]["GCF_2"]["source"] == "matches"
        assert "source" not in screening and "query_threshold" not in screening
        assert screening["date"]

    def test_real_records_clear_the_inferred_flag(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_screening(r, "SRR1", "GCF_1", 0.5, None, "matches", 0.0, None)
        r.datasets["SRR1"]["screening"]["inferred"] = True
        reg.record_screening(r, "SRR1", "GCF_1", 0.9, 0.99, "branchwater", 0.1, None)
        assert "inferred" not in r.datasets["SRR1"]["screening"]

        reg.record_extraction(r, "SRR1", "GCF_1", [], 3, False, {})
        r.datasets["SRR1"]["extractions"]["GCF_1"]["inferred"] = True
        reg.record_assembly(r, "SRR1", "GCF_1", Path("d"), {"contigs": 1}, "v1.2.9", {})
        assert "inferred" not in r.datasets["SRR1"]["extractions"]["GCF_1"]

    def test_screening_cap_keeps_the_best(self, tmp_path, caplog):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        table = tmp_path / "parsed_containment.txt"
        table.write_text("\tGCF_A\nSRR1\t0.9\nSRR2\t0.2\nSRR3\t0.5\n")
        with caplog.at_level("WARNING"):
            reg.record_screening_from_table(r, table, tmp_path / "matches", max_screened=2)
        assert sorted(reg.query(r, "screened")) == ["SRR1", "SRR3"]
        assert "SRR2" not in r.datasets  # nothing else was recorded for it
        assert "2" in caplog.text and "screen" in caplog.text.lower()

    def test_screening_cap_keeps_accessions_with_other_records(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_exclusion(r, "SRR2", "16S amplicon")
        table = tmp_path / "parsed_containment.txt"
        table.write_text("\tGCF_A\nSRR1\t0.9\nSRR2\t0.2\n")
        reg.record_screening_from_table(r, table, tmp_path / "matches", max_screened=1)
        assert reg.query(r, "screened") == ["SRR1"]
        assert r.datasets["SRR2"]["exclusion"]["excluded"] is True and "screening" not in r.datasets["SRR2"]

    def test_screening_from_table_skips_non_numeric_cells(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        table = tmp_path / "parsed_containment.txt"
        table.write_text("\tGCF_A\tGCF_B\nSRR1\t0.9\tnot-a-number\n")
        count = reg.record_screening_from_table(r, table, tmp_path / "matches")
        assert count == 1
        assert list(r.datasets["SRR1"]["screening"]["genomes"]) == ["GCF_A"]

    def test_registry_path_logs_a_registry_above_the_working_directory(self, tmp_path, caplog):
        (tmp_path / "metaquest_registry.json").write_text("{}")
        sub = tmp_path / "targeted"
        sub.mkdir()
        with caplog.at_level("INFO"):
            found = reg.registry_path(start=sub)
        assert found.resolve() == (tmp_path / "metaquest_registry.json").resolve()
        assert str(found) in caplog.text

    def test_record_screening_from_table(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        table = tmp_path / "parsed_containment.txt"
        table.write_text(
            "\tGCF_A\tGCF_B\tmax_containment\tmax_containment_annotation\n"
            "SRR1\t0.9\t0.0\t0.9\tGCF_A\n"
            "SRR2\t0.0\t0.3\t0.3\tGCF_B\n"
        )
        matches_folder = tmp_path / "matches"
        count = reg.record_screening_from_table(r, table, matches_folder)
        assert count == 2
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_A"]["containment"] == 0.9
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_A"]["csv"] == str(matches_folder / "GCF_A.csv")
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_A"]["source"] == "matches"
        assert "GCF_B" not in r.datasets["SRR1"]["screening"]["genomes"]
        assert r.datasets["SRR2"]["screening"]["genomes"]["GCF_B"]["containment"] == 0.3
        assert "GCF_A" not in r.datasets["SRR2"]["screening"]["genomes"]

    def test_record_screening_from_table_missing_returns_zero(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        count = reg.record_screening_from_table(r, tmp_path / "missing.txt", tmp_path / "matches")
        assert count == 0
        assert r.datasets == {}

    def test_metadata_analysis_extraction_assembly(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_metadata(
            r, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {"run_size": "1234", "organism": "x"}
        )
        reg.record_analysis(r, "SRR1", "sra_stats", tmp_path / "sra_statistics.csv", {"total_reads": 10})
        reg.record_extraction(
            r,
            "SRR1",
            "GCF_1",
            [tmp_path / "t" / "GCF_1_1.fastq.gz"],
            239464,
            False,
            {"genome_fasta": "genomes/GCF_1.fna", "preset": "sr", "threshold": 0.1},
        )
        reg.record_assembly(
            r,
            "SRR1",
            "GCF_1",
            tmp_path / "t" / "GCF_1_assembly",
            {"contigs": 188, "total_bp": 1209849, "n50": 15400, "largest": 57491},
            "v1.2.9",
            {"threads": 1},
        )
        rec = reg.extraction_record(r, "SRR1", "GCF_1")
        assert (
            rec["mapped_reads"] == 239464 and rec["assembly"]["n50"] == 15400 and rec["assembly"]["version"] == "v1.2.9"
        )
        assert r.datasets["SRR1"]["metadata"]["run_size"] == "1234"
        assert r.datasets["SRR1"]["analyses"]["sra_stats"]["summary"]["total_reads"] == 10
        # a new extraction record keeps the assembly block
        reg.record_extraction(r, "SRR1", "GCF_1", [], 0, True, {"genome_fasta": "g", "preset": "sr", "threshold": 0.1})
        assert reg.extraction_record(r, "SRR1", "GCF_1")["assembly"]["contigs"] == 188

    def test_record_metadata_keeps_spots_bases_layout_platform_as_int(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_metadata(
            r,
            "SRR1",
            tmp_path / "metadata" / "SRR1_metadata.xml",
            {
                "run_size": "4744553813",
                "run_md5": "abc",
                "run_total_spots": "47964651",
                "run_total_bases": "14389395300",
                "library_layout": "PAIRED",
                "platform": "ILLUMINA",
                "library_strategy": "WGS",
            },
        )
        metadata = r.datasets["SRR1"]["metadata"]
        assert metadata["run_total_spots"] == 47964651
        assert isinstance(metadata["run_total_spots"], int)
        assert metadata["run_total_bases"] == 14389395300
        assert isinstance(metadata["run_total_bases"], int)
        assert metadata["library_layout"] == "PAIRED"
        assert metadata["platform"] == "ILLUMINA"
        assert metadata["library_strategy"] == "WGS"

    def test_record_metadata_spots_none_when_not_numeric(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_metadata(r, "SRR1", tmp_path / "metadata" / "SRR1_metadata.xml", {})
        metadata = r.datasets["SRR1"]["metadata"]
        assert metadata["run_total_spots"] is None
        assert metadata["run_total_bases"] is None
        assert metadata["library_layout"] is None
        assert metadata["platform"] is None
        assert metadata["library_strategy"] is None

    def test_query_and_stage_counts(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        for acc, cont in (("SRR1", 0.9), ("SRR2", 0.5), ("SRR3", 0.05)):
            reg.record_screening(r, acc, "GCF_1", cont, None, "branchwater", 0.01, None)
        reg.record_selection(r, ["SRR1", "SRR2"], {"threshold": 0.1}, Path("accessions.txt"))
        reg.record_download(r, "SRR1", "downloaded", tmp_path / "fastq")
        reg.record_extraction(r, "SRR1", "GCF_1", [Path("a")], 100, False, {})
        reg.record_extraction(r, "SRR2", "GCF_1", [], 0, False, {})
        reg.record_assembly(
            r, "SRR1", "GCF_1", Path("d"), {"contigs": 3, "total_bp": 9, "n50": 3, "largest": 4}, "v", {}
        )
        assert reg.query(r, "screened") == ["SRR1", "SRR2", "SRR3"]
        assert reg.query(r, "selected") == ["SRR1", "SRR2"]
        assert reg.query(r, "downloaded") == ["SRR1"]
        assert reg.query(r, "extracted", "GCF_1") == ["SRR1"]
        assert reg.query(r, "assembled", "GCF_1") == ["SRR1"]
        counts = reg.stage_counts(r)
        assert counts["stages"]["selected"] == 2 and counts["genomes"]["GCF_1"]["zero_mapped"] == ["SRR2"]
        with pytest.raises(DataAccessError, match="Unknown stage"):
            reg.query(r, "bogus")


class TestScanners:
    def test_summarise_contigs_uses_len_headers(self, tmp_path):
        fa = tmp_path / "final.contigs.fa"
        fa.write_text(
            ">k141_1 flag=1 multi=2.0 len=10\nACGTACGTAC\n>k141_2 flag=1 multi=2.0 len=4\nACGT\n>k141_3 len=6\nACGTAC\n"
        )
        assert summarise_contigs(fa) == {"contigs": 3, "total_bp": 20, "n50": 10, "largest": 10}
        fa.write_text(">a\nACGTACGT\n>b\nAC\n")
        assert summarise_contigs(fa) == {"contigs": 2, "total_bp": 10, "n50": 8, "largest": 8}
        assert summarise_contigs(tmp_path / "missing.fa") == {"contigs": 0, "total_bp": 0, "n50": 0, "largest": 0}

    def test_count_fastq_reads_plain_and_gz(self, tmp_path):
        assert reg.count_fastq_reads(_fastq(tmp_path / "a.fastq", reads=3)) == 3
        assert reg.count_fastq_reads(_fastq(tmp_path / "b.fastq.gz", reads=5, gz=True)) == 5

    def test_count_fastq_reads_across_chunk_boundaries(self, tmp_path):
        """The chunked binary reader must still count correctly for a file over 1 MiB."""
        record = "@r{0}\n" + "A" * 32 + "\n+\n" + "I" * 32 + "\n"
        path = tmp_path / "big.fastq"
        with open(path, "w") as handle:
            for i in range(20000):
                handle.write(record.format(i))
        assert path.stat().st_size > 1024 * 1024
        assert reg.count_fastq_reads(path) == 20000

    def test_split_extract_filename(self):
        genomes = ["GCF_000008025.1", "wMel_ref_1"]
        assert reg.split_extract_filename("GCF_000008025.1_1.fastq.gz", genomes) == ("GCF_000008025.1", "_1")
        assert reg.split_extract_filename("GCF_000008025.1.fastq.gz", genomes) == ("GCF_000008025.1", "")
        assert reg.split_extract_filename("wMel_ref_1_s.fastq.gz", genomes) == ("wMel_ref_1", "_s")
        assert reg.split_extract_filename("GCF_9_1.fastq.gz", genomes) == ("GCF_9", "_1")  # suffix fallback
        assert reg.split_extract_filename("notes.txt", genomes) is None

    def test_empty_assembly_dirs(self, tmp_path):
        paths = _project(tmp_path)
        empty_asm = paths.targeted / "SRR1" / "GCF_1_assembly"
        empty_asm.mkdir(parents=True)  # interrupted, no contigs file
        full_asm = paths.targeted / "SRR2" / "GCF_1_assembly"
        full_asm.mkdir(parents=True)
        (full_asm / "final.contigs.fa").write_text(">k len=5\nACGTA\n")
        (paths.targeted / "SRR3").mkdir(parents=True)  # no assembly folder at all

        pairs = reg.empty_assembly_dirs(paths.targeted, ["GCF_1"])
        assert pairs == [("SRR1", "GCF_1")]

    def test_bootstrap_handles_non_numeric_cani(self, tmp_path):
        paths = _project(tmp_path)
        paths.matches.mkdir(parents=True)
        (paths.matches / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.5,NA\nSRR2,0.6,\n")
        r = reg.bootstrap_from_disk(paths)
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_1"]["containment"] == 0.5
        assert r.datasets["SRR1"]["screening"]["genomes"]["GCF_1"]["cani"] is None
        assert r.datasets["SRR2"]["screening"]["genomes"]["GCF_1"]["containment"] == 0.6
        assert r.datasets["SRR2"]["screening"]["genomes"]["GCF_1"]["cani"] is None

    def test_bootstrap_and_reconcile(self, tmp_path):
        paths = _project(tmp_path)
        _fastq(paths.fastq / "SRR1" / "SRR1_1.fastq")
        _fastq(paths.fastq / "SRR2" / "SRR2_1.fastq")
        (paths.metadata).mkdir()
        (paths.metadata / "SRR1_metadata.xml").write_text("<x/>")
        paths.genomes.mkdir()
        (paths.genomes / "GCF_1.fna").write_text(">c\nACGT\n")
        _fastq(paths.targeted / "SRR1" / "GCF_1_1.fastq.gz", gz=True)
        _fastq(paths.targeted / "SRR1" / "GCF_1_2.fastq.gz", gz=True)
        asm = paths.targeted / "SRR1" / "GCF_1_assembly"
        asm.mkdir(parents=True)
        (asm / "final.contigs.fa").write_text(">k len=5\nACGTA\n")
        (paths.targeted / "SRR2" / "GCF_1_assembly").mkdir(parents=True)  # interrupted, empty
        paths.matches.mkdir()
        (paths.matches / "GCF_1.csv").write_text("acc,containment,cANI\nSRR1,0.9,0.99\nSRR2,0.3,0.9\nSRR9,0.2,0.8\n")
        acc_file = tmp_path / "accessions.txt"
        acc_file.write_text("SRR1\nSRR2\n")

        r = reg.bootstrap_from_disk(paths, accessions_file=acc_file)
        assert r.datasets["SRR9"]["screening"]["genomes"]["GCF_1"]["containment"] == 0.2
        assert (
            r.datasets["SRR1"]["selection"]["selected"] is True and r.datasets["SRR1"]["selection"]["inferred"] is True
        )
        assert (
            r.datasets["SRR1"]["download"]["state"] == "downloaded"
            and r.datasets["SRR1"]["download"]["inferred"] is True
        )
        assert r.datasets["SRR1"]["metadata"]["inferred"] is True and "metadata" not in r.datasets["SRR2"]
        ext = reg.extraction_record(r, "SRR1", "GCF_1")
        # both mates are counted, so the inferred count matches what a real extraction records
        assert ext["mapped_reads"] == 4 and ext["assembly"]["contigs"] == 1 and ext["inferred"] is True
        assert "GCF_1" in r.genomes

        (paths.fastq / "SRR2" / "SRR2_1.fastq").unlink()
        _fastq(paths.fastq / "SRR7" / "SRR7_1.fastq")
        report = reg.reconcile(r, paths)
        assert report.recorded_missing == ["SRR2"]
        assert report.untracked_fastq == ["SRR7"]
        assert report.empty_assembly_dirs == [("SRR2", "GCF_1")]
        assert r.datasets["SRR2"]["download"]["state"] == "missing"

    def test_to_dataframes(self, tmp_path):
        r = reg.load_registry(tmp_path / "metaquest_registry.json")
        reg.record_selection(r, ["SRR1"], {"threshold": 0.1}, Path("a.txt"))
        reg.record_extraction(r, "SRR1", "GCF_1", [], 7, False, {"preset": "sr"})
        datasets, extractions = reg.to_dataframes(r)
        assert list(datasets.index) == ["SRR1"] and bool(datasets.loc["SRR1", "selected"]) is True
        assert extractions.loc[0, "genome_id"] == "GCF_1" and int(extractions.loc[0, "mapped_reads"]) == 7
