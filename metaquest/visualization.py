import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from upsetplot import UpSet
import logging
import numpy as np
import os
from typing import Union, Optional, List



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


####################################################################################################
#
#  Section: Metadata file visualization
#


def plot_bar(df_sorted: pd.DataFrame, title: str, colors, show_title: bool):
    plt.barh(df_sorted[df_sorted.columns[0]], df_sorted[df_sorted.columns[1]], color=colors)
    plt.xlabel('Number of Samples')
    plt.ylabel(df_sorted.columns[0])
    if show_title:
        plt.title(title)
    plt.gca().invert_yaxis()
    plt.tick_params(axis='y', which='major', labelsize=8)

def plot_pie(df_sorted: pd.DataFrame, title: str, colors, show_title: bool):
    total_count = df_sorted[df_sorted.columns[1]].sum()
    labels = [
        label if count / total_count > 0.02 else ''
        for label, count in zip(df_sorted[df_sorted.columns[0]], df_sorted[df_sorted.columns[1]])
    ]
    plt.pie(df_sorted[df_sorted.columns[1]], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    if show_title:
        plt.title(title)

def plot_radar(df_sorted: pd.DataFrame, title: str, show_title: bool):
    df_limited = df_sorted.head(10)
    categories = list(df_limited[df_limited.columns[0]])
    values = list(df_limited[df_limited.columns[1]])
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]

    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    ax.fill(angles, values, color='teal', alpha=0.5)

    if show_title:
        plt.title(title, size=20, color='blue', y=1.1)

def plot_metadata_counts(file_path: str, title: str = None, plot_type: str = 'bar', colors=None,
                         show_title: bool = True, save_format: str = None) -> None:
    logging.info(f"Generating {plot_type} plot.")
    df = pd.read_csv(file_path, sep='\t')
    df_sorted = df.sort_values(by=df.columns[1], ascending=False)

    if title is None and show_title:
        title = df.columns[0]

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    plt.figure(figsize=(15, 6))

    plot_func = {
        'bar': plot_bar,
        'pie': plot_pie,
        'radar': plot_radar,
    }.get(plot_type, None)

    if plot_func:
        plot_func(df_sorted, title, colors, show_title)
    else:
        logging.error("Invalid plot_type. Choose between 'bar', 'pie', or 'radar'.")
        return

    if save_format:
        file_name, _ = os.path.splitext(os.path.basename(file_path))
        output_file_name = f"{file_name}_{plot_type}.{save_format}"
        plt.savefig(output_file_name, format=save_format)
        logging.info(f"Plot saved as {output_file_name}.")

####################################################################################################
#
#  Section: Summary file visualization


def plot_containment_column(file_path: str, column: str = 'max_containment',
                            title: str = None, colors=None, show_title: bool = True,
                            save_format: str = None, threshold: float = None,
                            plot_type: str = 'rank') -> None:
    """
    Plots the values of a given column using various types of plots.

    Parameters:
        file_path (str): The path to the summary file.
        column (str): The column to plot. Defaults to 'max_containment'.
        title (str): The title of the plot.
        colors: The colors to use in the plot.
        show_title (bool): Whether to display the title.
        save_format (str): The format in which to save the plot (e.g., 'png').
        threshold (float): Minimum value to be included in the plot.
        plot_type (str): Type of plot to generate. Options are 'rank', 'histogram', 'box', 'violin'.
    """
    logging.info(f"Generating {plot_type} plot for column {column}.")

    # Read the DataFrame
    df = pd.read_csv(file_path, sep='\t')

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        logging.error(f"Column {column} does not exist in the DataFrame.")
        return

    # Filter the data based on the threshold
    if threshold is not None:
        df = df[df[column] >= threshold]
        if df.empty:
            logging.warning(f"No data above the threshold of {threshold}.")
            return

    # Sort the DataFrame by the selected column
    df_sorted = df.sort_values(by=column, ascending=False)

    # Create a rank column for rank plot
    df_sorted['rank'] = np.arange(1, len(df_sorted) + 1)

    # Prepare plot settings
    if title is None and show_title:
        title = f"{plot_type.capitalize()} Plot of {column}"

    if colors is None:
        colors = 'blue'

    # Create the plot
    plt.figure(figsize=(10, 6))

    if plot_type == 'rank':
        plt.scatter(df_sorted['rank'], df_sorted[column], color=colors)
        plt.xlabel('Rank')
        plt.ylabel('Containment Score (%)')
    elif plot_type == 'histogram':
        plt.hist(df_sorted[column], bins=20, color=colors)
        plt.xlabel(column)
        plt.ylabel('Frequency')
    elif plot_type == 'box':
        plt.boxplot(df_sorted[column])
        plt.ylabel(column)
    elif plot_type == 'violin':
        plt.violinplot(df_sorted[column])
        plt.ylabel(column)
    else:
        logging.error("Invalid plot_type. Choose between 'rank', 'histogram', 'box', or 'violin'.")
        return

    if show_title:
        plt.title(title)

    # Save the plot if save_format is specified
    if save_format:
        file_name, _ = os.path.splitext(os.path.basename(file_path))
        output_file_name = f"{file_name}_{plot_type}_plot_{column}.{save_format}"
        plt.savefig(output_file_name, format=save_format)
        logging.info(f"Plot saved as {output_file_name}.")


def load_and_filter_data(summary_file: str, threshold: Union[int, float]) -> pd.DataFrame:
    """
    Load and filter the summary data based on a given threshold.

    Parameters:
        summary_file (str): The file path to the tab-separated summary data file.
                            The first column is assumed to be the index.

        threshold (int, float): A numerical threshold for filtering the data.
                                 Values below this threshold will be set to zero.

    Returns:
        pd.DataFrame: A filtered DataFrame with values below the given threshold set to zero,
                      and with columns and rows entirely consisting of zero values removed.
    """
    df = pd.read_csv(summary_file, sep='\t', index_col=0)

    # Keep only columns that contain 'GCF' or 'GCA'
    df = df[[col for col in df.columns if 'GCF' in col or 'GCA' in col]]

    # Fill NaN values with zero
    df.fillna(0, inplace=True)

    # Set values below the threshold to zero
    df[df < threshold] = 0

    # Remove columns and rows that are all zeros
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.loc[(df != 0).any(axis=1), :]

    return df

def plot_heatmap(args):
    """Plot heatmap for the summary file."""
    logging.info("Generating heatmap...")
    df = load_and_filter_data(args.summary_file, args.threshold)
    plt.figure(figsize=(10, 10))
    sns.clustermap(df, cmap='viridis')
    plt.savefig(args.heatmap_file)
    logging.info(f"Heatmap saved to {args.heatmap_file}")

def plot_pca(args):
    """Plot PCA for the summary file."""
    logging.info("Generating PCA plot...")
    df = load_and_filter_data(args.summary_file, args.threshold)
    df_T = df.T
    X = StandardScaler().fit_transform(df_T)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)
    ax.scatter(principalDf['PC1'], principalDf['PC2'])
    plt.savefig(args.pca_file)
    logging.info(f"PCA plot saved to {args.pca_file}")

def plot_upset(args):
    """Generate an UpSet plot based on the summary data."""
    logging.info("Generating UpSet plot...")
    df = load_and_filter_data(args.summary_file, args.threshold)
    upset = UpSet(df, subset_size='auto')
    upset.plot()
    plt.savefig(args.output_file)
    logging.info(f"UpSet plot saved to {args.output_file}")
