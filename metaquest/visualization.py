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

def plot_metadata_counts(file_path: str, title: str = None, plot_type: str = 'bar', colors = None,
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
        output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '.' + save_format
        plt.savefig(output_file_name, format=save_format)
        logging.info(f"Plot saved as {output_file_name}.")

####################################################################################################
#
#  Section: Summary file visualization

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
