"""
Interactive visualization functions for MetaQuest.

This module provides interactive plotting capabilities using Plotly,
essential for modern data exploration and publication-quality figures.
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.offline as pyo
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Optional, Union
from pathlib import Path

from metaquest.core.exceptions import VisualizationError

logger = logging.getLogger(__name__)


def create_interactive_pca(
    data: Union[pd.DataFrame, np.ndarray],
    metadata: Optional[pd.DataFrame] = None,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    n_components: int = 3,
    title: str = "Interactive PCA Plot",
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> go.Figure:
    """
    Create an interactive 3D PCA plot with metadata overlay.

    Args:
        data: Sample x Features matrix
        metadata: Sample metadata for coloring/sizing points
        color_by: Metadata column for point colors
        size_by: Metadata column for point sizes
        n_components: Number of PCA components (2 or 3)
        title: Plot title
        output_file: HTML file to save plot
        show_plot: Whether to display plot

    Returns:
        Plotly Figure object

    Raises:
        VisualizationError: If PCA fails
    """
    try:
        # Prepare data
        if isinstance(data, pd.DataFrame):
            X = data.values
            sample_names = data.index
        else:
            X = data
            sample_names = [f"Sample_{i}" for i in range(X.shape[0])]

        # Perform PCA
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
        X_pca = pca.fit_transform(X)

        # Create DataFrame for plotting
        plot_data = pd.DataFrame(
            {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Sample": sample_names}
        )

        if n_components >= 3 and X_pca.shape[1] >= 3:
            plot_data["PC3"] = X_pca[:, 2]

        # Add metadata if provided
        if metadata is not None and color_by is not None:
            if color_by in metadata.columns:
                plot_data[color_by] = metadata.loc[sample_names, color_by]
            else:
                logger.warning(f"Column '{color_by}' not found in metadata")
                color_by = None

        if metadata is not None and size_by is not None:
            if size_by in metadata.columns:
                size_values = metadata.loc[sample_names, size_by]
                # Check for NaN values and handle them
                if size_values.isna().any():
                    logger.warning(
                        f"Column '{size_by}' contains NaN values, removing size mapping"
                    )
                    size_by = None
                else:
                    plot_data[size_by] = size_values
                    # Additional safety check
                    if plot_data[size_by].isna().any():
                        logger.warning(
                            f"NaN values detected in plot data for '{size_by}', removing size mapping"
                        )
                        column_to_drop = size_by
                        size_by = None
                        plot_data = plot_data.drop(columns=[column_to_drop])
            else:
                logger.warning(f"Column '{size_by}' not found in metadata")
                size_by = None

        # Create plot
        if n_components >= 3 and X_pca.shape[1] >= 3:
            fig = px.scatter_3d(
                plot_data,
                x="PC1",
                y="PC2",
                z="PC3",
                color=color_by,
                size=size_by,
                hover_name="Sample",
                title=title,
                labels={
                    "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                    "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                    "PC3": f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)",
                },
            )
        else:
            fig = px.scatter(
                plot_data,
                x="PC1",
                y="PC2",
                color=color_by,
                size=size_by,
                hover_name="Sample",
                title=title,
                labels={
                    "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                    "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                },
            )

        # Customize layout
        fig.update_layout(width=800, height=600, showlegend=True, hovermode="closest")

        # Add explained variance to title
        total_variance = sum(pca.explained_variance_ratio_[:n_components])
        fig.update_layout(
            title=f"{title}<br><sub>Total variance explained: {total_variance:.1%}</sub>"
        )

        # Save plot if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info(f"Interactive PCA plot saved to {output_path}")

        # Show plot if requested
        if show_plot:
            fig.show()

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create interactive PCA plot: {e}")


def create_interactive_tsne(
    data: Union[pd.DataFrame, np.ndarray],
    metadata: Optional[pd.DataFrame] = None,
    color_by: Optional[str] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    title: str = "Interactive t-SNE Plot",
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> go.Figure:
    """
    Create an interactive t-SNE plot.

    Args:
        data: Sample x Features matrix
        metadata: Sample metadata for coloring points
        color_by: Metadata column for point colors
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        title: Plot title
        output_file: HTML file to save plot
        show_plot: Whether to display plot

    Returns:
        Plotly Figure object
    """
    try:
        # Prepare data
        if isinstance(data, pd.DataFrame):
            X = data.values
            sample_names = data.index
        else:
            X = data
            sample_names = [f"Sample_{i}" for i in range(X.shape[0])]

        # Perform t-SNE
        tsne = TSNE(
            n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42
        )
        X_tsne = tsne.fit_transform(X)

        # Create DataFrame for plotting
        plot_data = pd.DataFrame(
            {"tSNE1": X_tsne[:, 0], "tSNE2": X_tsne[:, 1], "Sample": sample_names}
        )

        # Add metadata if provided
        if metadata is not None and color_by is not None:
            if color_by in metadata.columns:
                plot_data[color_by] = metadata.loc[sample_names, color_by]
            else:
                logger.warning(f"Column '{color_by}' not found in metadata")
                color_by = None

        # Create plot
        fig = px.scatter(
            plot_data,
            x="tSNE1",
            y="tSNE2",
            color=color_by,
            hover_name="Sample",
            title=title,
        )

        # Customize layout
        fig.update_layout(width=800, height=600, showlegend=True, hovermode="closest")

        # Save and show plot
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info(f"Interactive t-SNE plot saved to {output_path}")

        if show_plot:
            fig.show()

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create interactive t-SNE plot: {e}")


def create_interactive_heatmap(
    data: Union[pd.DataFrame, np.ndarray],
    sample_metadata: Optional[pd.DataFrame] = None,
    feature_metadata: Optional[pd.DataFrame] = None,
    cluster_samples: bool = True,
    cluster_features: bool = True,
    distance_metric: str = "euclidean",
    linkage_method: str = "ward",
    title: str = "Interactive Heatmap",
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> go.Figure:
    """
    Create an interactive clustered heatmap with dendrograms.

    Args:
        data: Data matrix (samples x features or features x samples)
        sample_metadata: Metadata for sample annotations
        feature_metadata: Metadata for feature annotations
        cluster_samples: Whether to cluster samples
        cluster_features: Whether to cluster features
        distance_metric: Distance metric for clustering
        linkage_method: Linkage method for clustering
        title: Plot title
        output_file: HTML file to save plot
        show_plot: Whether to display plot

    Returns:
        Plotly Figure object
    """
    try:
        # Prepare data
        if isinstance(data, pd.DataFrame):
            data_matrix = data.values
            sample_names = data.index
            feature_names = data.columns
        else:
            data_matrix = data
            sample_names = [f"Sample_{i}" for i in range(data_matrix.shape[0])]
            feature_names = [f"Feature_{i}" for i in range(data_matrix.shape[1])]

        # Clustering
        sample_order = list(range(len(sample_names)))
        feature_order = list(range(len(feature_names)))

        if cluster_samples:
            sample_linkage = linkage(
                data_matrix, method=linkage_method, metric=distance_metric
            )
            sample_dendro = dendrogram(sample_linkage, no_plot=True)
            sample_order = sample_dendro["leaves"]

        if cluster_features:
            feature_linkage = linkage(
                data_matrix.T, method=linkage_method, metric=distance_metric
            )
            feature_dendro = dendrogram(feature_linkage, no_plot=True)
            feature_order = feature_dendro["leaves"]

        # Reorder data
        clustered_data = data_matrix[np.ix_(sample_order, feature_order)]
        clustered_sample_names = [sample_names[i] for i in sample_order]
        clustered_feature_names = [feature_names[i] for i in feature_order]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=clustered_data,
                x=clustered_feature_names,
                y=clustered_sample_names,
                colorscale="RdBu_r",
                showscale=True,
                hoverongaps=False,
                hovertemplate="Sample: %{y}<br>Feature: %{x}<br>Value: %{z}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            width=max(800, len(clustered_feature_names) * 20),
            height=max(600, len(clustered_sample_names) * 15),
            xaxis={"side": "top"},
            yaxis={"autorange": "reversed"},
        )

        # Save and show plot
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info(f"Interactive heatmap saved to {output_path}")

        if show_plot:
            fig.show()

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create interactive heatmap: {e}")


def create_diversity_comparison_plot(
    alpha_diversity: pd.DataFrame,
    metadata: pd.DataFrame,
    group_by: str,
    diversity_metric: str = "shannon",
    plot_type: str = "box",
    title: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> go.Figure:
    """
    Create interactive diversity comparison plots.

    Args:
        alpha_diversity: DataFrame with diversity metrics
        metadata: Sample metadata
        group_by: Column to group samples by
        diversity_metric: Which diversity metric to plot
        plot_type: Type of plot ("box", "violin", "strip")
        title: Plot title
        output_file: HTML file to save plot
        show_plot: Whether to display plot

    Returns:
        Plotly Figure object
    """
    try:
        # Prepare data
        plot_data = alpha_diversity.copy()
        plot_data[group_by] = metadata.loc[alpha_diversity.index, group_by]

        # Set default title
        if title is None:
            title = f"{diversity_metric.title()} Diversity by {group_by}"

        # Create plot based on type
        if plot_type == "box":
            fig = px.box(
                plot_data, x=group_by, y=diversity_metric, title=title, points="all"
            )
        elif plot_type == "violin":
            fig = px.violin(
                plot_data,
                x=group_by,
                y=diversity_metric,
                title=title,
                box=True,
                points="all",
            )
        elif plot_type == "strip":
            fig = px.strip(plot_data, x=group_by, y=diversity_metric, title=title)
        else:
            raise VisualizationError(f"Unknown plot type: {plot_type}")

        # Update layout
        fig.update_layout(width=800, height=600, showlegend=True)

        # Save and show plot
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info(f"Diversity comparison plot saved to {output_path}")

        if show_plot:
            fig.show()

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create diversity comparison plot: {e}")


def create_beta_diversity_plot(
    abundance_matrix: Union[pd.DataFrame, np.ndarray],
    metadata: pd.DataFrame,
    color_by: str,
    distance_metric: str = "bray_curtis",
    method: str = "PCA",
    title: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> go.Figure:
    """
    Create beta diversity ordination plot.

    Args:
        abundance_matrix: Sample x Species abundance matrix
        metadata: Sample metadata
        color_by: Metadata column for coloring points
        distance_metric: Distance metric for beta diversity
        method: Ordination method ("PCA", "PCoA", "NMDS")
        title: Plot title
        output_file: HTML file to save plot
        show_plot: Whether to display plot

    Returns:
        Plotly Figure object
    """
    try:
        # Calculate beta diversity (for future PCoA/NMDS implementations)
        # distance_df = calculate_beta_diversity(
        #     abundance_matrix, metric=distance_metric, return_dataframe=True
        # )

        if method == "PCA":
            # Use PCA on abundance data directly
            return create_interactive_pca(
                abundance_matrix,
                metadata=metadata,
                color_by=color_by,
                title=title or f"PCA of {distance_metric} distances",
                output_file=output_file,
                show_plot=show_plot,
            )
        else:
            # For PCoA and NMDS, would need additional implementation
            logger.warning(f"Method {method} not yet implemented, using PCA")
            return create_interactive_pca(
                abundance_matrix,
                metadata=metadata,
                color_by=color_by,
                title=title or f"PCA of {distance_metric} distances",
                output_file=output_file,
                show_plot=show_plot,
            )

    except Exception as e:
        raise VisualizationError(f"Failed to create beta diversity plot: {e}")
