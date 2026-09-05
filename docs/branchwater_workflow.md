# Advanced Branchwater Workflows

This guide covers advanced workflows and decision points for using MetaQuest with Branchwater containment data. For basic usage, see the main [README](../README.md).

## Workflow Decision Points

### Metadata Strategy Selection

Choose your metadata approach based on analysis requirements:

#### Option A: Embedded Metadata (Fast)
Use metadata already present in Branchwater CSV files. Suitable for rapid analysis with basic sample information:

```bash
metaquest extract_branchwater_metadata --branchwater-folder branchwater_files --metadata-folder metadata
```

**Advantages:**
- No NCBI API calls required
- Immediate availability
- Includes: organism, bioproject, biosample, geographic data

**Limitations:** 
- Limited depth compared to full NCBI records
- May lack study-specific metadata

#### Option B: Full NCBI Metadata (Comprehensive)
Download complete metadata records from NCBI for detailed analysis:

```bash
metaquest download_metadata --matches-folder matches --metadata-folder metadata --threshold 0.95 --email your@email.com
metaquest parse_metadata --metadata-folder metadata --metadata-table-file parsed_metadata.txt
```

**Advantages:**
- Complete sample descriptions
- Study protocols and methods
- Enhanced publication metadata

**Considerations:**
- Requires NCBI API access
- Longer processing time
- Rate-limited downloads

## Searching from a FASTA

`branchwater_search` replaces the manual download. It needs the sourmash extra and network access:

```bash
metaquest branchwater_search --genome-fasta genomes/GCF_000008025.1.fna --threshold 0.1 --branchwater-folder branchwater
metaquest use_branchwater --branchwater-folder branchwater --matches-folder matches
```

A run that returns zero matches is reported as a warning, not an error; verify the index with a genome
known to be abundant in metagenomes before trusting an empty result.

## Advanced Filtering and Thresholds

### Containment steps

`parse_containment` does not filter; it records every sample and summarizes how many samples reach
each containment step. Choose the step size for the summary and apply thresholds downstream:

```bash
metaquest parse_containment --matches-folder matches --step-size 0.05
metaquest select_datasets --threshold 0.9 --output accessions.txt
metaquest count_metadata --metadata-column Sample_Scientific_Name --threshold 0.9
```

## What the registry records

Most commands in this workflow update the project registry (`metaquest_registry.json`) as they run;
`status` reads it back. One line per command:

- `branchwater_search` / `parse_containment`: screening, the containment per genome
- `select_datasets`: selection, the threshold and metadata filter used
- `blacklist`: exclusions, with a reason
- `download_sra`: download outcomes, with file sizes and dates
- `download_metadata` / `parse_metadata`: metadata
- `sra_stats` / `sra_validate` / `sra_profile_quality`: analyses
- `extract_target_reads`: extraction and assembly, per target genome

See the README's "Project state" section and `metaquest status --help` for the full set of flags.

## Troubleshooting Common Issues

### Format Validation Errors
If you encounter format errors, verify your CSV structure:

```bash
# Check file headers
head -1 your_branchwater_file.csv

# Expected format: acc,containment,organism,[additional_columns]
```

---

*For basic Branchwater usage, see the main [README](../README.md). For technical details, refer to [ARCHITECTURE.md](ARCHITECTURE.md).*