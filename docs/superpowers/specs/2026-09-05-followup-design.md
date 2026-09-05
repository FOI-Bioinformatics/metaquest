# MetaQuest follow-up design: Branchwater search, download consolidation, taxonomy tables, CLI polish

Approved in conversation on 2026-09-05. Base: local `main` at ee0238e (the merged audit repairs).
Four sub-projects, built in order as stacked branches, each PR-sized.

Decisions taken with the maintainer:
- The two extra download commands are removed now, not deprecated.
- `branchwater_search` leaves metadata columns empty; `download_metadata` remains the metadata source.
- Threshold comparisons become greater-or-equal everywhere.

## A. `branchwater_search` command

Purpose: start the pipeline from a genome FASTA instead of a manually downloaded Branchwater CSV.

Module `metaquest/data/branchwater_search.py`:
- `sketch_fasta(fasta_path) -> dict`: reads sequences with Biopython `SeqIO` (already a dependency),
  builds one sourmash `MinHash(n=0, ksize=21, scaled=1000)` with `add_sequence(seq, force=True)`,
  wraps it in `SourmashSignature(mh, name=<fasta stem>, filename=<fasta name>)` and returns the single
  signature object (a `dict`) obtained from `sourmash.signature.save_signatures_to_json([sig], buffer)`
  (the JSON list's first element). sourmash is imported inside the function; `ImportError` becomes
  `DataAccessError("sourmash is required to sketch a genome. Install it with: pip install 'metaquest[sourmash]'")`.
- `load_signature(sig_path) -> dict`: reads a sourmash JSON signature file; accepts a list (first
  element) or a single object; raises `DataAccessError` if the ksize is not 21 or scaled is not 1000.
- `search_index(signature, threshold, server=DEFAULT_SERVER, timeout=600) -> List[Tuple[str, float]]`:
  `requests.post(f"{server}/search", json={"threshold": threshold, "signature": signature}, timeout=timeout)`;
  raises `DataAccessError` on HTTP or connection errors (message includes status and body head);
  parses the response as CSV with header `SRA accession,containment`; returns `(accession, containment)`
  pairs sorted by containment descending. `DEFAULT_SERVER = "https://api.branchwater.sourmash.bio"`.
- `write_branchwater_csv(matches, output_path, ksize=21) -> Path`: writes the standard header
  `acc,containment,cANI,biosample,bioproject,assay_type,collection_date_sam,geo_loc_name_country_calc,organism,lat_lon`
  with `containment` to 4 decimals, `cANI = round(containment ** (1 / ksize), 4)` (0 when containment is 0)
  and the remaining columns empty. Creates the parent folder.

Command `branchwater_search` (`metaquest/cli/commands/branchwater_search.py`, class `BranchwaterSearchCommand`):
- `--genome-fasta PATH` or `--signature PATH` (mutually exclusive, one required).
- `--threshold` float, default 0.1 (Branchwater's own default).
- `--branchwater-folder`, default `branchwater`.
- `--output`, default `<branchwater-folder>/<input stem>.csv`; the stem is the genome id later shown as
  the containment column, matching `use_branchwater`.
- `--server`, default `DEFAULT_SERVER`.
- Logs the number of matches and the top containment; when zero matches, logs a warning that the public
  index has returned nothing for known positives before and suggests testing with a control genome; the
  empty CSV is still written and the exit code is 0. `MetaQuestError` -> logged, exit 1.
- Registered in `register_all_commands()`; added to `expected_commands` in `tests/test_cli_main.py`.

Tests: `tests/test_branchwater_search.py` (module) mocking `requests.post`; sketch test runs a tiny FASTA
through the real sourmash API, skipped with `pytest.importorskip("sourmash")`; `tests/test_cli_branchwater_search.py`.
`pytest.importorskip` keeps CI green without the extra.

Docs: README section 1 ("Getting Containment Files from Branchwater") rewritten to show the command
first and the web download as the alternative; `docs/branchwater_workflow.md` gets a short
"Searching from a FASTA" block; `environment.yml` already lists sourmash.

## B. Download consolidation

`download_sra` is the only download command.

Removed:
- Command `sra_download` (`SRADownloadEnhancedCommand` in `metaquest/cli/commands/sra_enhanced.py`).
- Command `sra-download-intelligent` and alias (`SRAIntelligentDownloadCommand` in
  `metaquest/cli/commands/sra_intelligent.py`), the `--download-session` flag of the dashboard command
  and its dead branch.
- Module `metaquest/sra/download_manager.py` and its exports in `metaquest/sra/__init__.py`;
  `create_download_summary` and `_create_download_plots` in `metaquest/sra/reporting.py` (they need
  `DownloadSession`) and the dead `export_metadata_enriched`.
- Module `metaquest/data/sra_enhanced.py` entirely (`EnhancedSRADownloader`, `create_download_report`,
  `verify_sra_tools`, which loses its only caller). Its one surviving function `estimate_download_time`
  moves to `metaquest/data/sra_metadata.py` next to `create_download_preview`. `sra_info` calls
  `create_download_preview(accessions, SRAMetadataClient(email, api_key))` directly.
- Tests: `tests/test_sra_intelligent_download.py`; the downloader classes in `tests/test_sra_enhanced.py`
  (keep the metadata, technology and statistics tests; move the `estimate_download_time` test to
  `tests/test_sra_metadata_extended.py`); all of `tests/test_sra_enhanced_extended.py`;
  `TestSRADownloadEnhancedCommand` in `tests/test_cli_commands_sra_enhanced.py`;
  `TestSRAIntelligentDownloadCommand` and download-session cases in `tests/test_cli_sra_intelligent.py`.
- `examples/sra_groups_example.json` stays (used by `sra-compare`).

Added to `download_sra`:
- A start-up tool check: `shutil.which("fasterq-dump")` is None -> error
  "fasterq-dump not found on PATH. Install sra-tools, for example: conda install -c bioconda sra-tools"
  and exit 1, unless `--dry-run`.
- `--report-file PATH`: after a real run, writes a CSV with columns `accession,status,message` where
  status is `downloaded`, `failed`, `already_present` or `blacklisted`, built from the returned
  `results`, `failed_accessions`, and the sets of already-downloaded and blacklisted accessions
  (`download_sra` in `metaquest/data/sra.py` adds `already_downloaded_accessions` and
  `blacklisted_accessions` lists to its returned dictionary so the CLI can label rows).

Docs: README "Which SRA download command should I use?", "Intelligent SRA Downloading" and
"Enhanced SRA Features" sections replaced by one "Downloading reads" section documenting `download_sra`
(including `--report-file`) and pointing to `sra_info` for the pre-download preview; the external-tools
table row lists only `download_sra`. `docs/SRA_ENHANCED_FEATURES.md` rewritten to cover `sra_info`,
`sra_stats`, `sra_validate` only (no emoji). `CLAUDE.md` and `AGENTS.md`: remove the intelligent-download
bullets and the download_manager architecture line; the "four main CLI commands" list becomes three.
`CONTRIBUTING.md` network-test paragraph unchanged.

## C. Taxonomy table unification

`taxonomic_summary --taxonomy-file` accepts both shapes.

- `metaquest/data/defaults.py` gains `read_records(path) -> pd.DataFrame` (separator by suffix, no index
  column) using the existing `_separator_for`.
- `metaquest/data/taxonomy.py`: `_build_taxonomy_lineage_map(taxonomy_data)` detects the shape:
  - `validate_taxonomy` CSV (columns `original_name`, `is_valid`, `lineage`): existing parsing.
  - `enrich_taxonomy` map (column `genome_id` plus any of `species`, `genus`, `family`, `order`,
    `class_name`, `phylum`): each row becomes `{rank: value}` for non-empty values, with `class_name`
    stored under rank `class`, keyed by `genome_id`.
  - Neither: `ProcessingError("Taxonomy file must come from validate_taxonomy (columns original_name, is_valid, lineage) or enrich_taxonomy (column genome_id with rank columns)")`.
- `TaxonomicSummaryCommand` reads the taxonomy file with `read_records` (TSV or CSV) and the help text
  names both producers.
- Tests in `tests/test_taxonomy.py` for the map shape (rank lookup, `class_name` mapping, unknown genome
  bucketed as `Unclassified_<level>`) and the error case; a CLI test with a `parsed_containment.txt`
  and an `enrich_taxonomy` TSV that produces `taxonomy_summary_genus.csv`.
- README "Taxonomic Summary Analysis" documents both routes with the `enrich_taxonomy` example first.

## D. CLI polish

- `metaquest/cli/base.py`: `BaseCommand.group` property (default `"Other"`); `CommandRegistry.setup_parsers`
  passes `formatter_class=argparse.ArgumentDefaultsHelpFormatter` to every `add_parser`, registers each
  alias as its own parser with `help=argparse.SUPPRESS` (same `configure_parser` and `func`), and sets
  the subparsers `metavar` to `COMMAND`.
- `metaquest/cli/main.py`: `create_parser` builds an epilog "Commands by pipeline step" from the registry:
  one block per group in this order: `Containment` (branchwater_search, use_branchwater,
  parse_containment, plot_containment, explore_containment, enrich_taxonomy, find_by_taxonomy),
  `Metadata` (extract_branchwater_metadata, download_metadata, parse_metadata,
  check_metadata_attributes, count_metadata, single_sample, plot_metadata_counts), `Genomes`
  (genome_search, genome_download, genome_prepare, download_test_genome), `Reads` (select_datasets,
  download_sra, status, sra_info, sra_stats, sra_validate, sra_profile_quality, sra_dashboard,
  sra_compare, extract_target_reads), `Analysis` (diversity_analysis, interactive_plot,
  validate_taxonomy, taxonomic_summary). Each command declares its group. `formatter_class` of the
  main parser becomes `RawDescriptionHelpFormatter` so the epilog keeps its lines.
- Canonical names: `sra_profile_quality`, `sra_dashboard`, `sra_compare`; the kebab spellings become
  hidden aliases. README, CLAUDE.md, AGENTS.md updated; `tests/test_cli_main.py` alias test inverted.
- Thresholds: greater-or-equal at `metaquest/processing/counts.py` (two sites),
  `metaquest/processing/containment.py` (four sites), `metaquest/data/branchwater.py:328`,
  `metaquest/data/metadata.py:66`, `metaquest/visualization/plots.py:399`; help strings say
  "(inclusive)"; tests adjusted where they encoded strict behaviour.
- Removed: `assemble_datasets` command and `metaquest/data/sra.py::assemble_datasets` with their
  tests; `parse_containment --file-format` (and the unused parameter it fed); dead functions with no
  callers: `filter_samples_by_containment`, `find_co_occurring_genomes` (processing/containment.py),
  `count_metadata_by_category`, `summarize_metadata_column` (processing/counts.py),
  `process_files_in_directory`, `read_json`, `write_json` (data/file_io.py), `validate_csv_file`
  (core/validation.py), `plot_heatmap` (visualization/plots.py), `create_beta_diversity_plot`
  (visualization/interactive.py), `calculate_dispersion` (processing/diversity.py),
  `create_presence_heatmap` (plugins/visualizers/heatmap.py), `create_grouped_bar_chart`
  (plugins/visualizers/bar.py), `create_choropleth` (plugins/visualizers/map.py),
  `suggest_species_corrections` (data/taxonomy.py), `download_from_file`, `create_genome_manifest`,
  `check_datasets_available` (data/genome_download.py), `discover_plugins`, `register_discovered_plugins`
  (plugins/base.py), `validate_file_path` (utils/security.py); each with its tests. Kept on purpose:
  `metaquest/visualization/reporting.py` (a whole report generator with its own test files; removing a
  subsystem is a separate decision) and the sourmash plugin classes (all four are registered entry points).
- Dead allowlist entries: remove `--gzip` from the fasterq-dump `safe_params`, `--split-3` from
  `FASTERQ_DUMP_BOOLEAN_FLAGS` and `-e` from `FASTERQ_DUMP_INTEGER_FLAGS`; add `-0` and `-s` to
  `PATH_VALUE_FLAGS` (their values are output paths under registered roots) with a validator test.
- `interactive_plot --plot-type tsne` calls `create_interactive_tsne(data_df, metadata_df, color_by=...,
  title=..., output_file=..., show_plot=...)`; the "not yet implemented" branch goes away; a CLI test
  patches the function and asserts the call.

## Out of scope

Byte-level resume for downloads; fetching metadata inside `branchwater_search`; renaming output files
of plot commands; the pre-existing test-isolation artifact in `tests/test_sra_analytics.py`.

## Global constraints

- Python >= 3.12 target; black line length 120; flake8, mypy and the radon ceiling via `make check`;
  `make test` green; `make pipeline` green after each sub-project.
- CLI flags use dashes. Plain, modest language; no Unicode symbols in new code, help or docs.
- Unit tests never call the network or external tools; `requests` and `SecureSubprocess.run_secure`
  or `subprocess.run` are patched. sourmash-dependent tests use `pytest.importorskip("sourmash")`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
