import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from upsetplot import UpSet
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_filter_data(summary_file, threshold):
    """Load and filter the summary data."""
    df = pd.read_csv(summary_file, sep='\t', index_col=0)
    df = df[[col for col in df.columns if 'GCF' in col or 'GCA' in col]]
    df.fillna(0, inplace=True)
    df[df < threshold] = 0
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
