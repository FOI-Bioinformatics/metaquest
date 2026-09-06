"""Opt-in smoke tests that talk to NCBI SRA (run with: make test-network).

SRR2517620 is a 425-spot MiSeq mosquito metagenome; fasterq-dump fetches it in
a few seconds, which is enough to prove the download command line and the
output layout end to end.
"""

import gzip
import shutil
import subprocess

import pytest

from metaquest.data.sra import download_accession

pytestmark = pytest.mark.network

TINY_RUN = "SRR2517620"
TINY_RUN_SPOTS = 425

needs_fasterq_dump = pytest.mark.skipif(shutil.which("fasterq-dump") is None, reason="fasterq-dump not on PATH")
needs_cli = pytest.mark.skipif(shutil.which("metaquest") is None, reason="metaquest CLI not on PATH")


def _read_count(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return sum(1 for _ in handle) // 4


@needs_fasterq_dump
def test_download_accession_writes_paired_fastq(tmp_path, monkeypatch):
    """Default settings (prefetch when available, compression on): files land as .fastq.gz."""
    monkeypatch.chdir(tmp_path)
    ok, message = download_accession(TINY_RUN, tmp_path / "fastq", num_threads=2)
    assert ok, message

    files = sorted(p.name for p in (tmp_path / "fastq" / TINY_RUN).glob("*.fastq*"))
    assert files == [f"{TINY_RUN}_1.fastq.gz", f"{TINY_RUN}_2.fastq.gz"]
    assert _read_count(tmp_path / "fastq" / TINY_RUN / f"{TINY_RUN}_1.fastq.gz") == TINY_RUN_SPOTS


@needs_fasterq_dump
def test_download_accession_writes_paired_fastq_uncompressed(tmp_path, monkeypatch):
    """compress=False leaves the plain FASTQ files fasterq-dump wrote, uncompressed."""
    monkeypatch.chdir(tmp_path)
    ok, message = download_accession(TINY_RUN, tmp_path / "fastq", num_threads=2, compress=False)
    assert ok, message

    files = sorted(p.name for p in (tmp_path / "fastq" / TINY_RUN).glob("*.fastq*"))
    assert files == [f"{TINY_RUN}_1.fastq", f"{TINY_RUN}_2.fastq"]
    assert _read_count(tmp_path / "fastq" / TINY_RUN / f"{TINY_RUN}_1.fastq") == TINY_RUN_SPOTS


@needs_fasterq_dump
@needs_cli
def test_download_sra_cli_round_trip(tmp_path):
    (tmp_path / "accessions.txt").write_text(f"{TINY_RUN}\n")
    result = subprocess.run(
        ["metaquest", "download_sra", "--accessions-file", "accessions.txt", "--fastq-folder", "fastq"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    # Compression defaults to on, so the CLI writes the gzipped file, not the plain one.
    assert (tmp_path / "fastq" / TINY_RUN / f"{TINY_RUN}_1.fastq.gz").exists()
    assert "Successfully downloaded: 1 datasets" in result.stderr + result.stdout
