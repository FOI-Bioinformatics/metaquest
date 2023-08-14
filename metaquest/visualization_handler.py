
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(args):
    """Plot heatmap for the summary file.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the required attributes.
    """
    df = pd.read_csv(args.summary_file, sep='	', index_col=0)
    df = df[[col for col in df.columns if 'GCF' in col or 'GCA' in col]]
    df.fillna(0, inplace=True)
    df[df < args.threshold] = 0
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.loc[(df != 0).any(axis=1), :]
    plt.figure(figsize=(10, 10))
    sns.clustermap(df, cmap='viridis')
    plt.savefig(args.heatmap_file)



def plot_pca(args):
    """Plot PCA for the summary file.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the required attributes.
    """
    df = pd.read_csv(args.summary_file, sep='	', index_col=0)
    df = df[[col for col in df.columns if 'GCF' in col or 'GCA' in col]]
    df.fillna(0, inplace=True)
    df[df < args.threshold] = 0
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



def plot_upset(args):
    """Generate an UpSet plot based on the summary data.
    
    Parameters:
    - args (Namespace): An argparse Namespace object containing the following attributes:
        * summary_file (str): Path to the summary data file.
        * output_file (str): Path to save the generated UpSet plot.
        * threshold (float): Minimum value to consider for the plot.

    The function reads the summary data, processes it, and then generates an UpSet plot.
    The resulting plot is saved to the specified output path.
    """
    df = pd.read_csv(args.summary_file, sep='	', index_col=0)
    # only keep columns containing GCF or GCA
    df = df[[col for col in df.columns if 'GCF' in col or 'GCA' in col]]
    # replace missing values with 0
    df.fillna(0, inplace=True)
    # set values less than the threshold to 0
    df[df < args.threshold] = 0
    # remove rows and columns that only contain 0
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.loc[(df != 0).any(axis=1), :]
    
    # UpSet plot
    upset = UpSet(df, subset_size='auto')
    upset.plot()
    plt.savefig(args.output_file)
