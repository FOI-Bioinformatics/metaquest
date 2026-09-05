#!/bin/bash
# local_test.sh: end-to-end CLI walkthrough on the bundled Branchwater sample.
# Runs in build/local_test so tracked files under test_data/ are never rewritten.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$ROOT/build/local_test"
rm -rf "$WORK"
mkdir -p "$WORK/branchwater" "$WORK/genomes"
cp "$ROOT/test_data/branchwater/salmonella_subset.csv" "$WORK/branchwater/"
if [ -f "$ROOT/test_data/GCF_000008985.1.fasta" ]; then
    cp "$ROOT/test_data/GCF_000008985.1.fasta" "$WORK/genomes/GCF_000008985.1.fna"
fi
cd "$WORK"

check() { if [ -e "$1" ]; then echo "ok   $1"; else echo "FAIL $1 missing"; exit 1; fi; }

echo "use_branchwater"
metaquest use_branchwater --branchwater-folder branchwater --matches-folder matches
check matches/salmonella_subset.csv

echo "parse_containment"
metaquest parse_containment --matches-folder matches --parsed-containment-file parsed_containment.txt \
    --summary-containment-file summary_containment.txt --step-size 0.1
check parsed_containment.txt
check summary_containment.txt

echo "extract_branchwater_metadata"
metaquest extract_branchwater_metadata --branchwater-folder branchwater --metadata-folder metadata
check metadata/branchwater_metadata.txt

echo "count_metadata (metadata table resolved from metadata/branchwater_metadata.txt)"
metaquest count_metadata --metadata-column Sample_Scientific_Name --threshold 0.9 --output-file metadata_counts.txt
check metadata_counts.txt

echo "plot_metadata_counts"
metaquest plot_metadata_counts --file-path metadata_counts.txt --plot-type bar --save-format png
check metadata_counts_bar.png

echo "plot_containment"
metaquest plot_containment --file-path parsed_containment.txt --column max_containment --plot-type rank --save-format png
check parsed_containment.txt_rank_max_containment.png

echo "select_datasets"
metaquest select_datasets --threshold 0.95 --output accessions.txt
check accessions.txt
test "$(wc -l < accessions.txt)" -gt 0

# use_branchwater/parse_containment/select_datasets already wrote metaquest_registry.json as they
# ran; remove it here so status --init can demonstrate building the registry from an existing
# project (matches folder, parsed containment table, accessions file) in one step.
echo "status --init"
rm -f metaquest_registry.json
metaquest status --init --parsed-containment parsed_containment.txt --accessions-file accessions.txt
check metaquest_registry.json

echo "blacklist"
metaquest blacklist --add SRR31320538 --reason "walkthrough example"

echo "status --stage excluded"
metaquest status --stage excluded

echo "status"
metaquest status --parsed-containment parsed_containment.txt --list-missing

if [ -f genomes/GCF_000008985.1.fna ]; then
    echo "extract_target_reads --dry-run"
    metaquest extract_target_reads --parsed-containment parsed_containment.txt --genome-id salmonella_subset \
        --genome-fasta genomes/GCF_000008985.1.fna --threshold 0.95 --dry-run
fi

echo "status --json"
metaquest status --json > status.json
python3 -c "
import json
with open('status.json') as f:
    data = json.load(f)
selected = data['stages']['selected']['count']
excluded = data['stages']['excluded']['count']
assert selected > 0, f'expected stages.selected.count > 0, got {selected}'
assert excluded == 1, f'expected stages.excluded.count == 1, got {excluded}'
print(f'status.json ok: selected={selected} excluded={excluded}')
" || { echo "FAIL status.json stage counts"; exit 1; }

echo "All steps passed (outputs in $WORK)"
