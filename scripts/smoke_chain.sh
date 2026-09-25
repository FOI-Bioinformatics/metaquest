#!/usr/bin/env bash
# Nightly real-data smoke chain: download a tiny real SRA run, validate it, profile it,
# extract reads against a reference genome, and write the results table. Exercises the
# same command line a person runs, against real NCBI/SRA services, using the tools the
# "metaquest" conda environment installs (fasterq-dump, prefetch, minimap2, samtools).
#
# SRR2517620 is a 425-spot MiSeq mosquito metagenome: small enough that fasterq-dump
# fetches it in a few seconds, so this chain finishes in well under a minute of tool time.
# A mosquito metagenome maps almost nothing onto the bacterial reference genome used
# below, so this script checks that each step exits 0 and writes its expected output,
# not how many reads mapped.
#
# Usage: bash scripts/smoke_chain.sh [PROJECT_DIR]
#   PROJECT_DIR defaults to a fresh temporary directory, which is left behind for
#   inspection; the caller (or CI, via the failure-only artifact upload) is responsible
#   for cleaning it up. Run from a "metaquest" environment that has fasterq-dump,
#   prefetch, minimap2, and samtools on PATH (e.g. "conda run -n metaquest bash
#   scripts/smoke_chain.sh").
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${1:-$(mktemp -d -t metaquest-smoke-XXXXXX)}"
mkdir -p "$PROJECT_DIR"
echo "Smoke chain project directory: $PROJECT_DIR"

# Invoked as "python -m metaquest.cli.main" rather than the "metaquest" console script
# so this always runs against whatever interpreter is active (the "metaquest" conda env
# in CI and locally), rather than whichever "metaquest" happens to be first on PATH.
#
# PYTHONPATH is pinned to this checkout's repository root so the chain always exercises
# the code sitting beside this script. This matters most in a local multi-worktree setup:
# an editable "pip install -e" only ever points at whichever checkout it was run from, so
# running from a different worktree (or a scratch project directory elsewhere) can
# otherwise silently import a different checkout's metaquest package. In CI there is only
# ever one checkout, so this is a no-op there.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
run_metaquest() {
    python -m metaquest.cli.main "$@"
}

GENOME_ID="GCF_000008985.1"
ACCESSION="SRR2517620"

cd "$PROJECT_DIR"

echo "== Step 1/6: download_test_genome =="
run_metaquest download_test_genome --output-folder test_data

echo "== Step 2/6: download_sra ($ACCESSION) =="
echo "$ACCESSION" >acc.txt
run_metaquest download_sra --accessions-file acc.txt --num-threads 2

echo "== Step 3/6: sra_validate =="
run_metaquest sra_validate --accessions-file acc.txt --check-pairs

echo "== Step 4/6: sra_profile =="
run_metaquest sra_profile --accessions-file acc.txt --sample-size 1000

echo "== Step 5/6: extract_target_reads =="
# A synthetic two-line parsed containment table: one genome column (GCF_000008985.1)
# plus the max_containment / max_containment_annotation columns results_table also
# reads. The real value from a Branchwater/sourmash screen would be much lower for a
# mosquito metagenome; 0.5 here only needs to clear --threshold 0.1 so the extraction
# step has a sample to run on.
cat >parsed_containment.txt <<PARSED
	${GENOME_ID}	max_containment	max_containment_annotation
${ACCESSION}	0.5	0.5	${GENOME_ID}
PARSED
run_metaquest extract_target_reads \
    --parsed-containment parsed_containment.txt \
    --genome-id "$GENOME_ID" \
    --genome-fasta "test_data/${GENOME_ID}.fasta" \
    --threshold 0.1 \
    --threads 2

echo "== Step 6/6: results_table =="
run_metaquest results_table --output results.tsv

echo "== Verifying results.tsv =="
lines=$(wc -l <results.tsv | tr -d ' ')
if [ "$lines" -lt 2 ]; then
    echo "ERROR: expected a header plus at least one data row in results.tsv, got $lines line(s)"
    exit 1
fi
echo "results.tsv: $lines line(s) (header + $((lines - 1)) data row(s))"

echo "Smoke chain completed successfully."
