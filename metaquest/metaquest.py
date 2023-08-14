
import argparse
from metaquest.data_processing import summarize, count_single_sample, download_test_genome, run_mastiff
from metaquest.visualization import plot_heatmap
from metaquest.metadata import download_metadata, parse_metadata

# argparse setup
def main():
    parser = argparse.ArgumentParser(description='MetaQuest: A toolkit for handling genomic metadata.')
    subparsers = parser.add_subparsers(title='commands', description='valid commands', help='additional help')

    # Download test genome command
    parser_download_test_genome = subparsers.add_parser('download_test_genome', help='Download test genome')
    parser_download_test_genome.set_defaults(func=download_test_genome)

    # Run mastiff command
    parser_run_mastiff = subparsers.add_parser('run_mastiff', help='Run MASTIFF')
    parser_run_mastiff.set_defaults(func=run_mastiff)
    
    # Summarize command
    parser_summarize = subparsers.add_parser('summarize', help='Summarize matches from MASH')
    parser_summarize.add_argument('--matches-folder', required=True, help='Folder containing _matches.csv files from MASH')
    parser_summarize.add_argument('--summary-file', required=True, help='Output summary file')
    parser_summarize.add_argument('--containment-file', required=True, help='Output containment file')
    parser_summarize.set_defaults(func=summarize)

    # Download metadata command
    parser_download_metadata = subparsers.add_parser('download-metadata', help='Download metadata from NCBI')
    parser_download_metadata.add_argument('--email', required=True, help='Email for NCBI Entrez')
    parser_download_metadata.add_argument('--matches-folder', required=True, help='Folder containing _matches.csv files')
    parser_download_metadata.add_argument('--metadata-folder', required=True, help='Folder to save metadata')
    parser_download_metadata.add_argument('--threshold', type=float, default=0.0, help='Containment threshold')
    parser_download_metadata.set_defaults(func=download_metadata)

    # Parse metadata command
    parser_parse_metadata = subparsers.add_parser('parse-metadata', help='Parse metadata from XML')
    parser_parse_metadata.add_argument('--metadata-folder', required=True, help='Folder containing downloaded metadata XML files')
    parser_parse_metadata.add_argument('--metadata-table-file', required=True, help='Output metadata table file')
    parser_parse_metadata.set_defaults(func=parse_metadata)

    # Count single sample command
    parser_count_single_sample = subparsers.add_parser('count_single_sample', help='Count single sample')
    parser_count_single_sample.add_argument('--summary-file', required=True, help='Input summary file')
    parser_count_single_sample.add_argument('--metadata-file', required=True, help='Input metadata file')
    parser_count_single_sample.add_argument('--metadata-column', required=True, help='Column in metadata file to count')
    parser_count_single_sample.add_argument('--threshold', type=float, default=0.0, help='Containment threshold')
    parser_count_single_sample.add_argument('--output-file', required=True, help='Output file for genome counts')
    parser_count_single_sample.set_defaults(func=count_single_sample)

    # Collect genome counts command
    parser_collect_genome_counts = subparsers.add_parser('collect_genome_counts', help='Collect genome counts')
    parser_collect_genome_counts.add_argument('--summary-file', required=True, help='Input summary file')
    parser_collect_genome_counts.add_argument('--metadata-file', required=True, help='Input metadata file')
    parser_collect_genome_counts.add_argument('--metadata-column', required=True, help='Column in metadata file to count')
    parser_collect_genome_counts.add_argument('--threshold', type=float, default=0.0, help='Containment threshold')
    parser_collect_genome_counts.add_argument('--output-file', required=True, help='Output file for genome counts')
    parser_collect_genome_counts.set_defaults(func=collect_genome_counts)
    
    # Plot heatmap command
    parser_plot_heatmap = subparsers.add_parser('plot_heatmap', help='Plot heatmap from summary')
    parser_plot_heatmap.add_argument('--summary-file', required=True, help='Input summary file')
    parser_plot_heatmap.add_argument('--heatmap-file', required=True, help='Output heatmap file')
    parser_plot_heatmap.add_argument('--threshold', type=float, default=0.0, help='Containment threshold')
    parser_plot_heatmap.set_defaults(func=plot_heatmap)

    # Plot UpSet command
    parser_plot_upset = subparsers.add_parser('plot_upset', help='Generate an UpSet plot from a summary file')
    parser_plot_upset.add_argument('--summary-file', required=True, help='Input summary file')
    parser_plot_upset.add_argument('--output-file', required=True, help='Output file for the UpSet plot')
    parser_plot_upset.add_argument('--threshold', type=float, default=0.1, help='Containment threshold for the plot')
    parser_plot_upset.set_defaults(func=plot_upset)

    # Plot PCA command
    parser_plot_pca = subparsers.add_parser('plot_pca', help='Generate a PCA plot from a summary file')
    parser_plot_pca.add_argument('--summary-file', required=True, help='Input summary file')
    parser_plot_pca.add_argument('--pca-file', required=True, help='Output file for the PCA plot')
    parser_plot_pca.add_argument('--threshold', type=float, default=0.1, help='Containment threshold for the plot')
    parser_plot_pca.set_defaults(func=plot_pca)
    
    # Download SRA argparse setup
    parser_download_sra = subparsers.add_parser('download_sra', help='Download Sequence Read Archive (SRA) data given its accession number.')
    parser_download_sra.add_argument('--sra-number', required=True, help='Accession number of the SRA data to be downloaded.')
    parser_download_sra.add_argument('--output-folder', required=True, help='Directory to save the downloaded SRA data.')
    parser_download_sra.set_defaults(func=download_sra)

    # Assemble datasets argparse setup
    parser_assemble_datasets = subparsers.add_parser('assemble_datasets', help='Assemble datasets based on input data files.')
    parser_assemble_datasets.add_argument('--data-files', required=True, nargs='+', help='List of paths to data files.')
    parser_assemble_datasets.add_argument('--output-file', required=True, help='Path to save the assembled dataset.')
    parser_assemble_datasets.set_defaults(func=assemble_datasets)


    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

