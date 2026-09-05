# MetaQuest audit findings (spec for the repair plan)

Source: functionality and usability audit of commit 0cbe7a4 on 2026-09-04, with a real end-to-end
Wolbachia run (report: https://claude.ai/code/artifact/19de3114-36e8-4a2e-819d-8d62444388f4).

## Verified defects, ranked

1. CRITICAL. All three SRA download commands (`download_sra`, `sra_download`, `sra-download-intelligent`)
   fail before fasterq-dump starts with `Invalid SRA accession format: 4`.
   `SecureSubprocess._build_validated_command` (metaquest/utils/security.py:174) validates args index 1
   as the accession; every caller passes `--threads N` first. The boolean flag `--progress` also swallows
   the accession as its "value", so the real accession is never validated. Present since 2025-08-26.
   Tests mock `run_secure`; `tests/test_security_comprehensive.py:473` asserts the wrong contract.
2. HIGH. `validate_path` (security.py:104-113) allows only cwd, home and the literal `/tmp`.
   `tempfile.mkdtemp()` on macOS returns `/var/folders/...` (used by default for fasterq-dump `--temp`),
   `/tmp` resolves to `/private/tmp`, and `/Volumes/...` is rejected. Prefix matching also accepts
   `/tmpfoo`. Verified: `extract_target_reads --output-folder <scratch>` is refused.
3. HIGH. `extract_target_reads` logs "Extracted mapped reads" and writes three empty files when the two
   mate files have different read counts: minimap2 warns "query files have different number of records"
   and maps single-end; `samtools fastq -1 -2 -s` sends unflagged reads to stdout, which run_secure
   discards. No mapped-read count is reported. 295,443 reads were extractable for SRR11011979.
4. HIGH. `genome_search --genus X` (and `genome_download`/`genome_prepare --genus`) never return
   anything. GTDB `/taxon/search/{name}` answers `{"matches": ["g__Wolbachia", "s__Wolbachia pipientis", ...]}`;
   `search_taxon` (metaquest/data/gtdb.py:41-63) wraps that dict as one record without an accession.
   `/species/search/{name}` returns `{"name", "genomes": [{"accession", "gtdb_species_rep", ...}]}`;
   `_is_representative` only checks `isRep`/`is_representative`.
5. HIGH. `sra-dashboard` and `sra-compare` produce zero-filled reports (GC 0.0%, 0 reads, grade "poor",
   all p = 1.0) and exit 0. `SRADatasetAnalyzer._locate_fastq_file` (metaquest/sra/analytics.py:644)
   only checks flat `fastq/<acc>.fastq.gz`, never `<folder>/<acc>/<acc>_1.fastq`. Intelligent commands
   default to `sra_downloads/` while every other command defaults to `fastq/`.
6. HIGH. `plot_metadata_counts` cannot read `count_metadata` output ("no numeric data to plot"):
   `_load_counts_df` (metaquest/visualization/plots.py:220) reads with `header=None` while
   `count_metadata` writes a header row and one column per genome. The bar branch (plots.py:306-315)
   never passes `output_file`, so `--save-format` writes nothing.
7. HIGH. `sra_download` renames fasterq-dump output to `<acc>_R1.fastq.gz` although it is uncompressed
   (metaquest/data/sra_enhanced.py:249-257); `sra_stats`/`sra-profile-quality` pick the opener by suffix.
8. MEDIUM. Default hand-offs break: `extract_branchwater_metadata` writes `metadata/branchwater_metadata.txt`
   but `count_metadata`, `single_sample`, `check_metadata_attributes` default to `metadata_table.txt`;
   `interactive_plot`, `diversity_analysis`, `taxonomic_summary` read CSV while `parse_containment`
   writes TSV (heatmap fails with "empty distance matrix").
9. MEDIUM. No command produces the `accessions.txt` every download command requires.
10. LOW. Cartopy emits a WARNING on every invocation (plugins/visualizers/map.py:28), before logging is
    configured, so `--log-level` cannot silence it.
11. LOW. Documentation drift: README output names (`summary.txt`, `containment.txt`,
    `counts_Sample_Scientific_Name.txt`), `docs/branchwater_workflow.md` documents non-existent flags
    (`parse_containment --threshold`, `--batch-size`, `--streaming --chunk-size`,
    `use_branchwater --max-workers`); `make pipeline` rewrites tracked `test_data/` outputs; no install
    guidance for sra-tools, minimap2, samtools, megahit, NCBI datasets.

## Out of scope for the first plan (separate plans)

- `branchwater_search` command (sourmash sketch, POST to https://api.branchwater.sourmash.bio/search).
- Consolidating the three download commands and removing inert `sra-download-intelligent` flags.
- CLI polish: defaults in help, grouped command list, hidden aliases, unified threshold semantics.

## Global constraints

- Python >= 3.12; black line length 120; flake8 and mypy clean (`make check`); `make test` green.
- CLI arguments use dashes. External tools are never run in unit tests; tests mock `run_secure`
  or `subprocess.run`, never the code under test.
- Language in code and docs is plain and modest; no Unicode symbols in new log or help text.
