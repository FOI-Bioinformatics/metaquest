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

### Containment Threshold Optimization
Experiment with different containment thresholds based on your research questions:

```bash
# Conservative analysis (high specificity)
metaquest parse_containment --matches-folder matches --threshold 0.95 --step-size 0.01

# Exploratory analysis (higher sensitivity)  
metaquest parse_containment --matches-folder matches --threshold 0.85 --step-size 0.05
```

### Large Dataset Optimization

For processing many genome files efficiently:

```bash
# Use dry-run to estimate processing time
metaquest download_metadata --matches-folder matches --dry-run --email your@email.com

# Process in batches to manage memory usage
metaquest parse_containment --matches-folder matches --batch-size 1000

# Parallel processing for large datasets
metaquest use_branchwater --branchwater-folder large_dataset --max-workers 8
```

## Troubleshooting Common Issues

### Format Validation Errors
If you encounter format errors, verify your CSV structure:

```bash
# Check file headers
head -1 your_branchwater_file.csv

# Expected format: acc,containment,organism,[additional_columns]
```

### Memory Management
For large datasets, consider using streaming processing:

```bash
# Enable streaming mode for large files
metaquest parse_containment --streaming --chunk-size 10000
```

---

*For basic Branchwater usage, see the main [README](../README.md). For technical details, refer to [ARCHITECTURE.md](ARCHITECTURE.md).*