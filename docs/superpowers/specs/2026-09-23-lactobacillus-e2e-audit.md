# Lactobacillus end-to-end audit of metaquest 0.3.1 (2026-09-23)

## Context

The 2026-09-06 audit programme could not run the screening stage against a live Branchwater index and never ran the
whole pipeline on full-size public runs through a store created from scratch. This audit did that on 2026-09-23 with
metaquest 0.3.1 at commit 39c8fa3 (working tree unchanged during the run), two Lactobacillus genomes, seven public
runs of 56-560 MB, a fresh shared store on an external ExFAT disk, and three projects sharing it. Every command, exit
code, wall time and output excerpt is in `/Volumes/sekvens2/metaquest-e2e/audit_log.md`; raw logs are under `logs/`,
snapshots under `snapshots/`, the documentation review in `docs_review.md`. Defects were recorded, not fixed.

Environment: macOS 26 (Darwin 25.2.0), miniforge Python 3.11 (`metaquest` console script in the base env; the conda
envs sra-tools 3.4.1, mt (minimap2 2.30, samtools 1.22.1), mq-megahit 1.2.9, seqkit, and pigz from envs/claude on the
PATH), store and projects on `/Volumes/sekvens2` (ExFAT via FSKit), root disk 46 GB free.

## Dataset

| Role | Run | Genome | Containment | Spots | NCBI size |
|---|---|---|---|---|---|
| crispatus positive | SRR23946461 | GCF_000091765.1 | 0.913 | 4 866 463 | 460 MB |
| crispatus positive | SRR23946404 | GCF_000091765.1 | 0.910 | 5 165 262 | 485 MB |
| crispatus positive | SRR23946450 | GCF_000091765.1 | 0.918 | 5 521 346 | 551 MB |
| crispatus positive | SRR23946417 | GCF_000091765.1 | 0.894 | 5 611 799 | 556 MB |
| crispatus positive | SRR23946447 | GCF_000091765.1 | 0.925 | 5 653 745 | 559 MB |
| low-containment control | ERR4233039 | GCF_000091765.1 | 0.299 | 1 547 139 | 56 MB |
| GG positive | ERR5164054 | GCF_000026505.1 | 0.9997 | 6 247 314 | 177 MB |
| blacklist case | SRR2182552 | GCF_000026505.1 | 0.9997 | 346 400 | 63 MB |
| adoption demo | SRR12066581 | GCF_000091765.1 | 0.295 | 62 875 | 12 MB |
| interruption tests | SRR8354140, SRR2182556 | GCF_000026505.1 | 0.9997 | 5.1 M, 3.6 M | 616, 637 MB |

Genomes: Lactobacillus crispatus ST1 GCF_000091765.1 (2 043 161 bp) and Lacticaseibacillus rhamnosus GG
GCF_000026505.1 (3 010 111 bp).

## Stage results

| Stage | Result | Wall time |
|---|---|---|
| 0 baseline | make check passes; make test passes on a plain PATH (1 888 passed, re-run after the audit: unchanged) and fails 3 tests with sra-tools on PATH (S0-1) | 20 min |
| 1 genomes | GTDB search and download work; old genus name gives a raw 400; manifest and registry polluted by `._` files | 1 min |
| 2 screen | 16 603 and 12 587 rows at 0.1 (predicted exactly), 216 and 992 above 0.9; cache hit 3 s; scale probe 734 679 rows in 11 s / 1.2 GB; parse 36 s | 5 min |
| 3 metadata | 300 XML in 15 s, 1 898 in 72 s without rate limiting, all key fields populated, spot counts equal runinfo; sra_info sizes 0; library strategy empty | 5 min |
| 4 select/exclude | multi-genome counts exact; blacklist works; last selection redefines the target list | 1 min |
| 5 store/download | 7 runs in 29 min (38 min on the re-download); every sidecar `failed` (S5-2); adopt crash; stale lock removed after 600 s; reindex loses usage; gc removed everything (S5-14/15) | 150 min |
| 6 analyse | sra_stats 5 min then 1 min cached; validate ok; profiles ok; dashboard needs `._` cleanup; sra_compare crashes | 15 min |
| 7 extract/assemble | six runs extracted in 24 min (about 15 percent of records at containment 0.9, 0.4 percent for the control); megahit cannot run on ExFAT (no FIFOs, S7-5); on an APFS copy GG assembled to 317 contigs / 2.90 Mb / N50 15.7 kb with 99 percent of reads mapping back; crispatus SRR23946461: 4754 contigs / 3942054 bp / N50 1433, mapping back 0.827; SRR23946447: 4281 contigs / 3764661 bp / N50 1602 | 180 min |
| 8 cross-cutting | pre-flight messages clear; dry runs flood warnings; docs review with 17 mismatches; status 2-4 s on a 16 MB registry | 10 min |

## Findings

Severity: Critical = data loss, wrong result or a stage that cannot complete; Important = wrong or missing behaviour with a
workaround; Minor = cosmetic or wording; Usability = a detour a first-time user must take.

### The AppleDouble family (one root cause, many symptoms)

macOS sets `com.apple.provenance` on every file a process writes; on a volume without native extended attributes
(ExFAT, SMB, some NAS) that becomes a 4 kB `._<name>` file next to the real one. metaquest lists folders with plain
globs and `iterdir()` and treats those files as data. Evidence for each symptom is in the audit log under the id.

| Id | Sev | Command | Observed | Fix |
|---|---|---|---|---|
| S1-3 | Important | genome_prepare | `._GCF_*.fna` become manifest rows and registry genomes | skip dot-prefixed names in `_create_manifest` (genome.py:312) |
| S2-3, S4-3 | Minor | status | counts `._*.fna` and `._*_metadata.xml` in the inventory | same filter in status.py:142-147 |
| S2-6 | Important | parse_containment, use_branchwater, extract_branchwater_metadata | ERROR per `._*.csv`, "Processed 2 files with 2 errors", exit 0 | filter and non-zero exit on unreadable match files (data/branchwater.py) |
| S5-2 | Critical | download_sra (store) | sidecar lists `._<ACC>_1.fastq.gz`, gzip check fails, dataset `failed`/`unverified` although complete; store_verify says corrupt | filter in the store's file listing (store/sidecar.py, layout.py); ignore names starting with `._` everywhere |
| S5-7 | Critical | store_adopt --move | FileNotFoundError on `._*.fastq` after the project files were removed; only copy left `failed`; registry not updated | adopt.py:233 rglob filter; never remove project files before the store copy verified |
| S5-11 | Critical | download_sra after a lock wait | waiter sees `failed` store copy and downloads again (three downloads of one run), previous copy shuffled to `<ACC>_old` | consequence of S5-2; also re-check the store after acquiring the lock |
| S6-6 | Important | sra_dashboard | reads `._*.json` profiles, utf-8 error, exit 1 | filter in sra/reporting.py |
| misc | Minor | download_sra, store_gc | pigz warnings on `._*.fastq`; rmtree failures "No such file '._X'" (macOS deletes the pair) | filter; tolerate ENOENT in rmtree |

Test to add: a fixture that drops `._name` files into every folder the code lists, on a plain filesystem.

### Store and download

| Id | Sev | Command | Expected | Observed | Fix |
|---|---|---|---|---|---|
| S5-14 | Critical | store_reindex | catalogue rebuilt with projects and usage | projects 4 -> 0, usage empty | persist project and usage records in sidecars or a journal; refuse to reindex without them; warn |
| S5-15 | Critical | store_gc after reindex | linked datasets kept | all 10 datasets "unused", `--yes` removed 6.2 GB still linked by three projects | before removal, check every registered project's fastq links and registry; treat "no usage rows" as unknown, not unused |
| S5-1 | Important | download_sra without --temp-folder | scratch under `<store>/tmp` | `tempfile.mkdtemp()` on the root disk (sra.py:398-403), 3 GB per pair of runs | default the scratch to `<store>/tmp` when a store is active; document --temp-folder |
| S5-5 | Important | store_verify --fix-state | recover from a wrong file list | verifies against the recorded list; "._X: missing" forever | add a rescan-from-disk mode |
| S5-9, S5-13 | Important | store_verify --spots | verify against NCBI spots | only the sidecar `ncbi` block filled at download time is used; adopted or metadata-less datasets stay unverified and `failed` | read `<store>/metadata/<ACC>_metadata.xml` and the project metadata folder; let --fix-state promote a bytes-ok dataset |
| S5-8 | Critical | store_adopt on an empty folder | refuse | "Adopted 1", dataset with no files marked complete | require at least one FASTQ; never complete with an empty file list |
| S5-10 | Important | Ctrl-C during download_sra | stop | SIGINT ignored, workers continue, lock kept | cancel the executor and child processes on KeyboardInterrupt |
| S5-12 | Important | store_adopt --copy (dedup) | project folder left as is | replaced by a symlink | make the dedup path honour --copy or fix the help |
| S5-4 | Minor | status with a failed store dataset | "linked, store state failed" | "FASTQ 0 present, 7 missing" | wording |
| S5-3, S5-6 | Minor | download_sra summary, --fix-state | "Downloaded 7/7"; error cleared | "Downloaded 5/7 (0 failed)"; stale error string kept | counting; clear `error` on success |

### Screening, metadata and selection

| Id | Sev | Command | Observed | Fix |
|---|---|---|---|---|
| S3-2 | Critical | sra_info | every size 0, "Total estimated size: 0.00 GB", rows keyed by SRX, release date empty | read RUN attributes total_bases/size/published (sra_metadata.py:200-210); report the run accession |
| S3-3 | Important | parse_metadata | Experiment_Library_Strategy and _Source empty for all rows although the XML has them | fix the attribute path; add a test on a real XML |
| S4-1 | Important | select_datasets / status | the last selection is the project's target list; exploratory runs redefine "wanted" | keep a named selection or make status take the accessions file by default |
| S4-2 | Important | status --next | suggests the `--no-skip-excluded` output, i.e. downloading a blacklisted run | suggest a skip-excluded list or apply the blacklist |
| S2-5 | Important | plot_containment | writes nothing and says nothing without --save-format | default to png, or print where the figure went |
| S3-1 | Usability | extract_branchwater_metadata | API CSVs have no metadata; output is Run_ID + cANI with no warning | warn and point at download_metadata |
| S2-2, S2-4, S2-7 | Minor | branchwater_search, plots | tie order of "best containment" unstable; no "Next:" hints on parse/plot/explore; plot name keeps ".txt" | sort ties; hints; stem |
| S1-1, S1-4 | Usability | genome_search, genome_download | raw GTDB 400 for an old genus name; genome_download records nothing in the registry | message; record_genome in genome_download |
| S0-1 | Important | make test | three TestDownloadAccession tests fail when real sra-tools are on PATH | patch `shutil.which` in those tests |

### Analysis, extraction and cross-cutting

| Id | Sev | Command | Observed | Fix |
|---|---|---|---|---|
| S6-3 | Important | sra_compare | "Object of type bool is not JSON serializable" with and without --statistical-tests | cast numpy bools; test with two real groups |
| S8-1 | Important | extract_target_reads --dry-run | one WARNING per non-downloaded sample (16 597 lines) | intersect with downloaded accessions first; summarise |
| S6-2 | Minor | sra_profile_quality --accession, sra_dashboard | --accessions-file required even with --accession / --quality-profiles | make it optional |
| S6-1, S6-4, S6-7, S6-8 | Minor | sra_stats, sra_compare, profiles, dashboard | mates vs spots unlabelled; "Mean total reads: 10,000" is the sample size; JSON keys differ from console names; dashboard re-profiles instead of reading --quality-profiles | labels; use the profiles |
| S6-5 | Minor | sra_validate | "failed at NCBI download" for a dataset whose files are fine (store state) | validate the files, report the store state separately |
| S7-3, S7-5 | Important | extract_target_reads --assemble | megahit fails at once on ExFAT/SMB (mkfifo unsupported) and the user sees only "CalledProcessError: Command [...]" without megahit's stderr | include the captured stderr in the error; pass `--tmp-dir` on a local temp folder or pre-check mkfifo and explain |
| S7-6 | Important | extract_target_reads rerun | a parameter change (min_mapq) silently re-extracts every sample; same parameters skip correctly | log the changed parameter or require --force |
| S7-1, S7-2, S7-7, S7-8 | Minor | extract_target_reads, download_sra --redownload-truncated, status --export-tsv | "MAPQ below 0 removed"; dangling links unreported; relink path drops reads_r1 and says "Newly downloaded"; export prints no paths and the extractions TSV carries a pandas index column | wording; carry the count; print the paths; drop the index |
| S8-2 | Minor | extract_target_reads pre-flight | each missing-tool error printed twice | dedupe |
| S2-1 | Usability | branchwater_search hint | "pip install 'metaquest[sourmash]'" does not say which interpreter | print sys.executable |
| docs | Usability | README, pipeline_overview | 17 mismatches, undocumented flags, misleading wording, order gaps (docs_review.md) | doc pass |

## Prioritised fix list

1. Filter `._*` and dotfiles in every folder listing (store sidecar/layout/adopt/gc, matches, metadata, genomes, profiles);
   add the fixture test. Closes S1-3, S2-3, S2-6, S4-3, S5-2, S5-7, S5-11, S6-6 and the pigz/rmtree noise.
2. store_gc must never remove a dataset a registered project links (walk project registries and fastq links); store_reindex
   must preserve or rebuild project and usage records and warn when it cannot. (S5-14, S5-15)
3. store_adopt: verify the store copy before removing project files, refuse empty folders, honour --copy. (S5-7, S5-8, S5-12)
4. download_sra: scratch under `<store>/tmp` when a store is active; handle KeyboardInterrupt; re-check the store after a
   lock wait; fix the summary count. (S5-1, S5-10, S5-11, S5-3)
5. store_verify: rescan mode, read the store metadata for spots, let --fix-state promote a bytes-ok dataset. (S5-5, S5-9, S5-13)
6. sra_info sizes and run ids; library strategy in parse_metadata. (S3-2, S3-3)
7. Selection and status: stable target list, blacklist-aware --next. (S4-1, S4-2)
8. plot_containment default output; sra_compare serialization; extraction dry-run summary; megahit stderr and a local
   tmp dir for assembly; parameter-change message. (S2-5, S6-3, S8-1, S7-3, S7-5, S7-6)
9. Tests independent of PATH. (S0-1)
10. Documentation pass from docs_review.md.

## Appendix A: documentation review

## (1) Documented commands, flags, defaults or file names that differ from the run
1. PO:38-39 says parse_containment writes summary_containment.txt; the default is top_containments.txt (containment.py:42). R:116/119 pass the name explicitly.
2. PO:35 parse_containment example has no step size; default --step-size 0.1 (containment.py:47), R:116 uses 0.05; neither doc gives the default.
3. PO:28 branchwater_search "(writes branchwater/<name>.csv)"; neither doc says the API ignores --threshold and the client filters.
4. R:74 "metadata columns are empty until step 5 (download_metadata) fills them": download_metadata writes XML to metadata/ and never fills the CSV columns.
5. R:100 "extract basic metadata directly from Branchwater CSV files"; PO:162 "Fast: the metadata embedded in the Branchwater CSVs": API CSVs have no metadata columns; output is Run_ID and cANI only.
6. R:106-109 "organism becomes Sample_Scientific_Name ... --metadata-column Sample_Scientific_Name": on the API path the column does not exist in the Branchwater table (R:160, PO:170 examples fail or come out empty).
7. R:138 --metadata-table-file parsed_metadata.txt vs default metadata_table.txt (metadata.py:194); R:155 says the later commands use metadata_table.txt when it exists. Same name drift at R:148, R:169, R:179.
8. PO:171 plot_metadata_counts --file-path counts_Sample_Scientific_Name.txt names a file nothing writes; count_metadata default is metadata_counts.txt (metadata.py:295, R:163).
9. PO:170 count_metadata without threshold; default 0.5 (metadata.py:290) undocumented.
10. R:197-199, R:39: genome_download leaves genomes/ncbi_dataset.zip (data/genome_download.py:86) and writes nothing to the registry; only genome_prepare writes genome_manifest.csv and the registry; genome_prepare is never described.
11. R:369 "To see sizes and sequencing technology before downloading, use sra_info": sizes were 0 and rows keyed by SRX.
12. PO:175 "The registry keeps ... library strategy and organism per accession": Experiment_Library_Strategy was empty.
13. R:229, PO:65-66 "Each select_datasets run replaces the previous selection" (true); no size filter, not stated, while R:368-369 points at sizes.
14. R:415-418 documents sra_profile_quality --accession ... alone; refused because --accessions-file is required (sra_intelligent.py:79, :93-95).
15. R:455-459 sra_compare --statistical-tests: crashes (JSON serialization). Groups JSON format (R:462-468) is right. PO:17 lists sra_compare without caveat.
16. R:274-275 "--lock-wait SECONDS (default: wait indefinitely)": code default 0.0 = wait as long as the other project keeps working (sra.py:196-198, store.py:509); 0 = unbounded is not stated.
17. R:352-353 "writes fastq/<accession>/<accession>_1.fastq": downloads are gzip by default (R:373), files are .fastq.gz.

## (2) Flags: documented but missing (none) and existing but undocumented
- plot_containment --save-format: only in the R:477 example; required for any output (default None, containment.py:110-113); output name <file>_rank_max_containment.png. PO:41 omits it.
- plot_metadata_counts --save-format: default None (metadata.py:343-346); PO:171 example writes nothing.
- download_sra --temp-folder (sra.py:99-101) and extract_target_reads --temp-folder (read_extraction.py:103): nowhere; without it data/sra.py:396-401 uses tempfile.mkdtemp() on the system disk.
- --data-root: R:294 lists seven commands; store_* only implied through R:252.
- sra_dashboard --accessions-file: required=True (sra_intelligent.py:402), never stated (PO:103, R:434-443; PO:110-111 implies --quality-profiles alone suffices).
- select_datasets --genome-ids/--require (select.py:46-51) and --parsed-containment (select.py:39): unmentioned.
- download_metadata --accessions-file/--batch-size: only PO:174-177, not R step 5.
- store_status --verbose, store_usage --project/--organism/--unused: unmentioned; store_verify --fix-state/--md5 at R:264-265 only.
- genome_search --species/--all/--format (genome.py:59-69), explore_containment --parsed-containment/--min-containment (writes containment_explorer.html, taxonomy_cache.tsv), find_by_taxonomy --genus/--format, enrich_taxonomy --cache: none documented; genome_search, explore_containment, find_by_taxonomy appear in neither doc.
- extract_target_reads --no-coverage (PO:156 only), --threads (R:322 only, default 4 unstated), --data-root.

## (3) Wording a first-time user would misread
- PO:41 "plot_containment draws the distribution", R:474 "Plot the distribution": implies a plot is shown; nothing is written, shown or printed without --save-format. Same for plot_metadata_counts (R:486, PO:171).
- R:254 "--move # move reads into the store, link back"; --copy unexplained; help (store.py:482-486) says "Leave each accession's project folder as is", but --copy on a duplicate replaced the project folder with a symlink.
- R:288 "store_verify --spots and status --reconcile compute a verdict for a dataset that lacks one", R:264-265: neither says a spot count exists only if metadata XML was present at download or adoption time (store_adopt --metadata-folder, store.py:501-503).
- R:290-292 (1 GB temp warning) and R:269-270 (leftover temp under tmp/) suggest all temp stays off the root disk; fasterq-dump scratch goes to the system temp dir without --temp-folder.
- R:74 (item 4). R:385-388 dry run does not apply the blacklist: accurate, but a dry run lists blacklisted accessions as planned.
- PO:133 "--dry-run lists what would be extracted and what would be skipped": no warning about one line per non-downloaded sample (16 597).
- R:99-100 "Extract Basic Metadata (Optional)", PO:162 "Fast": overpromise (item 5). R:368-369 sizes (item 11).

## (4) Pipeline order
- PO:3-4, 10-21: "screen, select, download, analyse, extract, assemble"; PO:21/159 "Metadata commands run alongside stages 1 and 2". Neither says download_metadata must finish before download_sra for the spot-count verdict (R:377-380, PO:80-82); store_verify --spots cannot fix it later.
- R numbering (1 branchwater ... 8 count_metadata, 10 status, 11 extract) never places download_metadata relative to download_sra; select_datasets and download_sra only appear under "Advanced SRA Operations" (R:336-362); R:126 uses the matches-folder route rather than --accessions-file after selection.
- parse_metadata before count_metadata: R order right, but R:155-157 makes parse_metadata look optional; on the API path the fallback has no usable columns. PO:205-207 walkthrough runs count_metadata right after extract_branchwater_metadata, which only works on bundled sample data.
- genome_prepare placed nowhere although branchwater_search needs a FASTA (R:71, PO:29); status listed as step 10 before downloads.

## Appendix B: rulings taken during the run

- Ruling: store at /Volumes/sekvens2/metaquest-e2e/store (user asked for all data and processing under metaquest-e2e) - replaces /Volumes/sekvens2/metaquest-store in the plan - cost if wrong: none, path only.
- Ruling: per-stage TDD and commits do not apply to an audit run; a stage is complete when its checks are answered in audit_log.md - cost if wrong: none.
- Ruling: AppleDouble ._ files are left in place at later stages and each recurrence is recorded (S1-3 family) - cost if wrong: extra workarounds
- Ruling: envs/claude/bin moved to the end of the audit PATH (its own metaquest shadowed the base install) - cost if wrong: none
- Ruling: Stage 3 download_metadata runs without --data-root (no store yet); the store copy is exercised in Stage 5 as the plan already schedules - cost if wrong: one extra rerun
- Ruling: store data re-downloaded after the gc data loss (7 runs, ~30 min), this time with --temp-folder <store>/tmp to test that flag as well - cost: 30 min
- Ruling: assembly stage on an APFS sparse bundle copy (ExFAT has no FIFOs, S7-5) - cost: assembled stage recorded in the copy's registry

## Appendix C: evidence

All logs, the audit ledger (`audit_log.md`, `progress.md`), snapshots and the store live under `/Volumes/sekvens2/metaquest-e2e/` (store 4.5 GB, extracted reads, dashboards; the APFS sparse bundle `mqapfs.sparsebundle` holds the three assemblies). Finding ids in this spec refer to the entries in `audit_log.md`.
