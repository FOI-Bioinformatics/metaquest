import logging
import argparse
from typing import Dict

from metaquest.data_processing import (
    use_branchwater_files, parse_containment, count_single_sample,
    download_test_genome, count_metadata, extract_branchwater_metadata,
    download_sra, assemble_datasets
)
from metaquest.visualization import plot_containment_column, plot_metadata_counts, plot_heatmap, plot_pca, plot_upset
from metaquest.metadata import download_metadata, parse_metadata, check_metadata_attributes
from argparse import Namespace


# Wrapper functions
def download_test_genome_wrapper(args):
    # Add additional logic here if needed.
    print("Starting the download of the test genome.")

    # Unpack the arguments and call the original function
    download_test_genome(args.output_folder)

    # Add any post-call logic here if needed.
    print("Successfully downloaded the test genome.")


def use_branchwater_files_wrapper(args):
    """
    Wrapper function for use_branchwater_files to be used with argparse.
    
    Parameters:
    - args (Namespace): Namespace object from argparse containing the command-line arguments.
    """
    use_branchwater_files(args.branchwater_folder, args.matches_folder)


def parse_containment_wrapper(args):
    """
    Wrapper function for parse_containment to be used with argparse.
    
    Parameters:
    - args (Namespace): Namespace object from argparse containing the command-line arguments.
    """
    parse_containment(
        args.matches_folder, 
        args.parsed_containment_file, 
        args.summary_containment_file, 
        args.step_size,
        args.file_format if hasattr(args, 'file_format') and args.file_format else None
    )


def extract_branchwater_metadata_wrapper(args):
    """
    Wrapper function for extract_branchwater_metadata to be used with argparse.
    
    Parameters:
    - args (Namespace): Namespace object from argparse containing the command-line arguments.
    """
    extract_branchwater_metadata(args.branchwater_folder, args.metadata_folder)


def download_metadata_wrapper(args):
    download_metadata(email=args.email, matches_folder=args.matches_folder, metadata_folder=args.metadata_folder,
                      threshold=args.threshold, dry_run=args.dry_run)


def parse_metadata_wrapper(args):
    """Wrapper function for parse_metadata.

    This function extracts individual arguments from the argparse Namespace object and passes them to parse_metadata.
    It also includes any additional logic before or after calling parse_metadata, if needed.
    """
    # Extract individual arguments from the argparse Namespace object
    metadata_folder = args.metadata_folder
    metadata_table_file = args.metadata_table_file

    # Add any pre-call logic here, if needed

    # Call the original function
    parse_metadata(metadata_folder, metadata_table_file)

    # Add any post-call logic here, if needed
    logging.info("Metadata parsing completed and table saved.")

def count_single_sample_wrapper(args):
    """
    Wrapper function for count_single_sample to be used with argparse.

    Parameters:
    - args (Namespace): Namespace object from argparse containing the command-line arguments.

    Calls count_single_sample function with unpacked arguments from the Namespace object.
    """

    # Unpack arguments from the Namespace object
    summary_file = args.summary_file
    metadata_file = args.metadata_file
    summary_column = args.summary_column
    metadata_column = args.metadata_column
    threshold = args.threshold
    top_n = args.top_n

    # Call the original function
    count_dict = count_single_sample(summary_file, metadata_file, summary_column, metadata_column, threshold, top_n)

    # You can log or print `count_dict` here, if needed.
    logging.info(f"Count dictionary generated: {count_dict}")


def count_metadata_wrapper(args: Namespace) -> None:
    """
    Wrapper function for collect_genome_counts to be called from the command-line interface.

    Parameters:
    - args (Namespace): An argparse Namespace object containing all the required arguments.
    """
    count_metadata(
        summary_file=args.summary_file,
        metadata_file=args.metadata_file,
        metadata_column=args.metadata_column,
        threshold=args.threshold,
        output_file=args.output_file,
        stat_file=args.stat_file
    )

def plot_metadata_counts_wrapper(args):
    plot_metadata_counts(file_path=args.file_path,
                         title=args.title,
                         plot_type=args.plot_type,
                         colors=args.colors,
                         show_title=args.show_title,
                         save_format=args.save_format)

# Wrapper function for count_metadata_attributes
def check_metadata_attributes_wrapper(args) -> Dict[str, int]:
    sorted_summary = check_metadata_attributes(file_path=args.file_path, output_file=args.output_file)
    print(f"Summary saved to {args.output_file}")
    return sorted_summary

def plot_containment_column_wrapper(args):
    """
    Wrapper function for plot_containment_column to use with argparse.
    """
    plot_containment_column(file_path=args.file_path, column=args.column, title=args.title,
                            colors=args.colors, show_title=args.show_title, save_format=args.save_format,
                            threshold=args.threshold, plot_type=args.plot_type)

def download_sra_wrapper(args):
    return download_sra(
        fastq_folder=args.fastq_folder,
        accessions_file=args.accessions_file,
        max_downloads=args.max_downloads,
        dry_run=args.dry_run,
        num_threads=args.num_threads,
        max_workers=args.max_workers
    )

# argparse setup
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='MetaQuest: A toolkit for handling genomic metadata.')
    subparsers = parser.add_subparsers(title='commands', description='valid commands', help='additional help', dest='command')
    
    # Make subcommand required
    parser.set_defaults(func=lambda _: parser.print_help())

    # Download test genome command
    parser_download_test_genome = subparsers.add_parser('download_test_genome', help='Download test genome')
    parser_download_test_genome.add_argument('--output-folder', default='genomes',
                                             help='Folder to save the downloaded fasta files.')
    parser_download_test_genome.set_defaults(func=download_test_genome_wrapper)

    # Use Branchwater files command
    parser_branchwater = subparsers.add_parser('use_branchwater', 
                                             help='Process pre-downloaded Branchwater files')
    parser_branchwater.add_argument('--branchwater-folder', required=True,
                                  help='Folder containing Branchwater output files')
    parser_branchwater.add_argument('--matches-folder', default='matches',
                                  help='Folder to save processed matches')
    parser_branchwater.set_defaults(func=use_branchwater_files_wrapper)

    # Extract metadata from Branchwater files command
    parser_extract_metadata = subparsers.add_parser('extract_branchwater_metadata',
                                                  help='Extract metadata from Branchwater files')
    parser_extract_metadata.add_argument('--branchwater-folder', required=True,
                                       help='Folder containing Branchwater output files')
    parser_extract_metadata.add_argument('--metadata-folder', default='metadata',
                                       help='Folder to save extracted metadata')
    parser_extract_metadata.set_defaults(func=extract_branchwater_metadata_wrapper)

    # Summarize command
    parser_parse_containment = subparsers.add_parser('parse_containment',
                                             help='Parse the containment data from the match files.')
    parser_parse_containment.add_argument('--matches_folder', default='matches',
                                  help='Folder containing the containment matches to be parsed.')
    parser_parse_containment.add_argument('--parsed_containment_file', default='parsed_containment.txt',
                                  help='File where the parsed containment will be stored.')
    parser_parse_containment.add_argument('--summary_containment_file', default='top_containments.txt',
                                  help='File where the top containments will be stored.')
    parser_parse_containment.add_argument('--step_size', default='0.1', type=float,
                                          help='Size of steps for the containment values.')
    parser_parse_containment.add_argument('--file_format', default=None, choices=['branchwater', 'mastiff'],
                                         help='Format of the input files (branchwater or mastiff). If not specified, format will be auto-detected.')
    parser_parse_containment.set_defaults(func=parse_containment_wrapper)

    # Plot containment column command
    parser_plot_containment = subparsers.add_parser('plot_containment',
                                                    help='Plot containment column from summary file')
    parser_plot_containment.add_argument('--file_path', required=True, help='Path to the summary file.')
    parser_plot_containment.add_argument('--column', default='max_containment', help='Column to plot. Defaults to max_containment.')
    parser_plot_containment.add_argument('--title', help='Title of the plot.')
    parser_plot_containment.add_argument('--colors', help='Colors to use in the plot.')
    parser_plot_containment.add_argument('--show_title', action='store_true', help='Whether to display the title.')
    parser_plot_containment.add_argument('--save_format', help='Format in which to save the plot (e.g., png).')
    parser_plot_containment.add_argument('--threshold', type=float, help='Minimum value to be included in the plot.')
    parser_plot_containment.add_argument('--plot_type', default='rank', help="Type of plot to generate. Options are 'rank', 'histogram', 'box', 'violin'.")
    parser_plot_containment.set_defaults(func=plot_containment_column_wrapper)

    # Download metadata command
    parser_download_metadata = subparsers.add_parser('download_metadata',
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
    parser_download_metadata.set_defaults(func=download_metadata_wrapper)

    # Parse metadata command
    parser_parse_metadata = subparsers.add_parser('parse_metadata',
                                                  help='Parse metadata for each *_metadata.xml file in the specified directory')
    parser_parse_metadata.add_argument('--metadata_folder', default='metadata',
                                       help='Folder containing *_metadata.xml files to parse')
    parser_parse_metadata.add_argument('--metadata_table_file', default='metadata_table.txt',
                                       help='File where the parsed metadata will be stored')
    parser_parse_metadata.set_defaults(func=parse_metadata_wrapper)

    # Subparser for check_metadata_attributes
    parser_check_metadata_attributes = subparsers.add_parser('check_metadata_attributes',
                                                  help='Count metadata attribute presence from a parsed metadata file')
    parser_check_metadata_attributes.add_argument('--file-path', required=True,
                                       help='Path to the input parsed_metadata.txt file')
    parser_check_metadata_attributes.add_argument('--output-file', default='metadata_counts.txt',
                                       help='Path to the output summary file')
    parser_check_metadata_attributes.set_defaults(func=check_metadata_attributes_wrapper)


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
    parser_single_sample.set_defaults(func=count_single_sample_wrapper)

    # Collect genome counts command
    parser_count_metadata = subparsers.add_parser('count_metadata',
                                                help='Collects genome counts and outputs a table with sample files.')
    parser_count_metadata.add_argument('--summary-file', default='SRA-summary.txt', help='Path to the summary file.')
    parser_count_metadata.add_argument('--metadata-file', default='metadata_table.txt', help='Path to the metadata file.')
    parser_count_metadata.add_argument('--metadata-column', required=True,
                                     help='Name of the column in the metadata file.')
    parser_count_metadata.add_argument('--threshold', type=float, default=0.5,
                                     help='Threshold for the column in the summary file.')
    parser_count_metadata.add_argument('--output-file', default='collected_table.txt', help='Path to the output file.')
    parser_count_metadata.add_argument('--stat-file', default=None, help='Path to the statistics file.')
    parser_count_metadata.set_defaults(func=count_metadata_wrapper)

    # Plot metadata counts command
    parser_plot_metadata = subparsers.add_parser('plot_metadata_counts',
                                                 help='Plot metadata counts as bar, pie, or radar chart.')
    parser_plot_metadata.add_argument('--file_path', required=True, help='Path to the metadata counts file.')
    parser_plot_metadata.add_argument('--title', default=None,
                                      help='Title for the plot. Default is the header of the first column.')
    parser_plot_metadata.add_argument('--plot_type', default='bar', choices=['bar', 'pie', 'radar'],
                                      help='Type of plot to generate. Choices are bar, pie, and radar. Default is bar.')
    parser_plot_metadata.add_argument('--colors', default=None,
                                      help='List of colors or colormap name. Default is viridis.')
    parser_plot_metadata.add_argument('--show_title', type=bool, default=True,
                                      help='Whether to display the title on the plot. Default is True.')
    parser_plot_metadata.add_argument('--save_format', default=None, choices=['png', 'jpg', 'pdf'],
                                      help='Format to save the figure. Choices are png, jpg, and pdf. Default is None (do not save).')
    parser_plot_metadata.set_defaults(func=plot_metadata_counts_wrapper)

    # Download SRA argparse setup
    parser_download_sra = subparsers.add_parser('download_sra',
                                                help='Download SRA datasets based on the given accessions file.')
    parser_download_sra.add_argument('--fastq_folder', default='fastq',
                                     help='Folder to save downloaded FASTQ files.')
    parser_download_sra.add_argument('--accessions_file', required=True,
                                     help='File containing SRA accessions, one per line.')
    parser_download_sra.add_argument('--max_downloads', type=int, default=None,
                                     help='Maximum number of datasets to download.')
    parser_download_sra.add_argument('--num_threads', type=int, default=4,
                                     help='Number of threads for each fasterq-dump.')
    parser_download_sra.add_argument('--max_workers', type=int, default=4,
                                     help='Number of threads for parallel downloads.')
    parser_download_sra.add_argument('--dry-run', action='store_true',
                                     help='If enabled, no downloads are performed. Only calculates the total number of accessions.')
    parser_download_sra.set_defaults(func=download_sra_wrapper)

    # Assemble datasets argparse setup
    parser_assemble_datasets = subparsers.add_parser('assemble_datasets', help='Assemble datasets based on input data files.')
    parser_assemble_datasets.add_argument('--data-files', required=True, nargs='+', help='List of paths to data files.')
    parser_assemble_datasets.add_argument('--output-file', required=True, help='Path to save the assembled dataset.')
    parser_assemble_datasets.set_defaults(func=assemble_datasets)


    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()