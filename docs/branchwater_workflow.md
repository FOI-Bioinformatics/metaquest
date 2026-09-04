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

## Advanced Filtering and Thresholds

### Containment steps

`parse_containment` does not filter; it records every sample and summarizes how many samples exceed
each containment step. Choose the step size for the summary and apply thresholds downstream:

```bash
metaquest parse_containment --matches-folder matches --step-size 0.05
metaquest select_datasets --threshold 0.9 --output accessions.txt
metaquest count_metadata --metadata-column Sample_Scientific_Name --threshold 0.9
```

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