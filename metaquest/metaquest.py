
import argparse
from metaquest.data_processing import run_mastiff, summarize, count_single_sample, download_test_genome, collect_genome_counts, download_sra, assemble_datasets
from metaquest.visualization import plot_heatmap, plot_pca, plot_upset
from metaquest.metadata import download_metadata, parse_metadata

# argparse setup
def main():
    parser = argparse.ArgumentParser(description='MetaQuest: A toolkit for handling genomic metadata.')
    subparsers = parser.add_subparsers(title='commands', description='valid commands', help='additional help')

    # Download test genome command
    parser_download_test_genome = subparsers.add_parser('download_test_genome', help='Download test genome')
    parser_download_test_genome.add_argument('--output-folder', default='genomes',
                                             help='Folder to save the downloaded fasta files.')
    parser_download_test_genome.set_defaults(func=download_test_genome)

    # Run mastiff command
    parser_mastiff = subparsers.add_parser('mastiff', help='Runs mastiff on the genome data in the specified folder.')
    parser_mastiff.add_argument('--genomes-folder', default='genomes',
                                help='Folder containing the genomes to be analyzed.')
    parser_mastiff.add_argument('--matches-folder', default='matches',
                                help='Folder where the output matches will be stored.')
    parser_mastiff.set_defaults(func=run_mastiff)
    
    # Summarize command
    parser_summarize = subparsers.add_parser('summarize',
                                             help='Summarizes the data from the .csv files in the matches directory.')
    parser_summarize.add_argument('--matches-folder', default='matches',
                                  help='Folder containing the matches to be summarized.')
    parser_summarize.add_argument('--summary-file', default='SRA-summary.txt',
                                  help='File where the summary will be stored.')
    parser_summarize.add_argument('--containment-file', default='top_containments.txt',
                                  help='File where the top containments will be stored.')
    parser_summarize.set_defaults(func=summarize)

    # Download metadata command
    parser_download_metadata = subparsers.add_parser('download-metadata',
                                                     help='Download metadata for each SRA accession in the .csv files in the matches directory')
    parser_download_metadata.add_argument('--email', required=True, help='Your email address for NCBI API access.')
    parser_download_metadata.add_argument('--matches_folder', default='matches',
                                          help='Folder containing match .csv files')
    parser_download_metadata.add_argument('--metadata_folder', default='metadata',
                                          help='Folder to save downloaded metadata')
    parser_download_metadata.add_argument('--threshold', type=float, default=0.0,
                                          help="Threshold for containment values (default: 0.0)")
    parser_download_metadata.add_argument('--dry-run', action='store_true',
                                          help='If enabled, no downloads are performed. Only calculates the total number of accessions and the number of accessions to download.')
    parser_download_metadata.set_defaults(func=download_metadata)

    # Parse metadata command
    parser_parse_metadata = subparsers.add_parser('parse-metadata',
                                                  help='Parse metadata for each *_metadata.xml file in the specified directory')
    parser_parse_metadata.add_argument('--metadata-folder', default='metadata',
                                       help='Folder containing *_metadata.xml files to parse')
    parser_parse_metadata.add_argument('--metadata-table-file', default='metadata_table.txt',
                                       help='File where the parsed metadata will be stored')
    parser_parse_metadata.set_defaults(func=parse_metadata)

    # Count single sample command
    parser_single_sample = subparsers.add_parser('single_sample',
                                                 help='Counts the occurrences of unique values in a column for a single sample.')
    parser_single_sample.add_argument('--summary-file', default='SRA-summary.txt', help='Path to the summary file.')
    parser_single_sample.add_argument('--metadata-file', default='metadata_table.txt',
                                      help='Path to the metadata file.')
    parser_single_sample.add_argument('--summary-column', required=True,
                                      help='Name of the column in the summary file to compare with the threshold.')
    parser_single_sample.add_argument('--metadata-column', required=True,
                                      help='Name of the column in the metadata file to count the unique values of.')
    parser_single_sample.add_argument('--threshold', type=float, default=0.1,
                                      help='Threshold for the column in the summary file.')
    parser_single_sample.add_argument('--top-n', type=int, default=100, help='Number of top items to keep.')
    parser_single_sample.set_defaults(func=count_single_sample)

    # Collect genome counts command
    parser_genome_count = subparsers.add_parser('genome_count',
                                                help='Collects genome counts and outputs a table with sample files.')
    parser_genome_count.add_argument('--summary-file', default='SRA-summary.txt', help='Path to the summary file.')
    parser_genome_count.add_argument('--metadata-file', default='metadata_table.txt', help='Path to the metadata file.')
    parser_genome_count.add_argument('--metadata-column', required=True, help='Name of the column in th metadata fil.')
    parser_genome_count.add_argument('--threshold', type=float, default=0.5,
                                     help='Threshold for the column in the summary file.')
    parser_genome_count.add_argument('--output-file', default='collected_table.txt', help='Path to the output file.')
    parser_genome_count.add_argument('--stat-file', default="collected_stats.txt", help='Path to the statistics file.')
    parser_genome_count.set_defaults(func=collect_genome_counts)

    
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

