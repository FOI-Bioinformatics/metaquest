#!/bin/bash
# local_test.sh

# Create test directories
mkdir -p test_data/branchwater
mkdir -p test_data/processed
mkdir -p test_data/metadata

# Create sample Branchwater data
echo "acc,containment,cANI,biosample,bioproject,assay_type,collection_date_sam,geo_loc_name_country_calc,organism,lat_lon" > test_data/branchwater/example.csv
echo "SRR12345678,0.95,0.98,SAMN12345678,PRJNA123456,AMPLICON,2019-05-01,USA,Francisella adeliensis,35.7N 100.2W" >> test_data/branchwater/example.csv
echo "SRR87654321,0.92,0.96,SAMN87654321,PRJNA123456,AMPLICON,2019-06-15,USA,Francisella adeliensis,36.1N 99.8W" >> test_data/branchwater/example.csv

# Test core functionality
echo "Testing download_test_genome..."
metaquest download_test_genome --output-folder test_data

echo "Testing use_branchwater..."
metaquest use_branchwater --branchwater-folder test_data/branchwater --matches-folder test_data/processed

echo "Testing extract_branchwater_metadata..."
metaquest extract_branchwater_metadata --branchwater-folder test_data/branchwater --metadata-folder test_data/metadata

echo "Testing parse_containment..."
metaquest parse_containment --matches-folder test_data/processed --parsed-containment-file test_data/parsed_containment.txt --summary-containment-file test_data/summary_containment.txt

# Verify outputs
echo "Verifying outputs..."
if [ -f test_data/GCF_000008985.1.fasta ]; then
    echo "✓ Test genome downloaded successfully"
else
    echo "✗ Test genome download failed"
    exit 1
fi

if [ -f test_data/processed/example.csv ]; then
    echo "✓ Branchwater file processed successfully"
else
    echo "✗ Branchwater file processing failed"
    exit 1
fi

if [ -f test_data/metadata/branchwater_metadata.txt ]; then
    echo "✓ Metadata extracted successfully"
else
    echo "✗ Metadata extraction failed"
    exit 1
fi

if [ -f test_data/parsed_containment.txt ]; then
    echo "✓ Containment parsed successfully"
else
    echo "✗ Containment parsing failed"
    exit 1
fi

if [ -f test_data/summary_containment.txt ]; then
    echo "✓ Summary created successfully"
else
    echo "✗ Summary creation failed"
    exit 1
fi

echo "All tests passed!"
