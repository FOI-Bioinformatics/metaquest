"""
Plotting functions for MetaQuest.

This module provides functions for creating various types of plots.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

from metaquest.core.exceptions import VisualizationError
from metaquest.plugins.base import visualizer_registry
from metaquest.plugins.visualizers.bar import BarChartPlugin
from metaquest.plugins.visualizers.heatmap import HeatmapPlugin
from metaquest.plugins.visualizers.map import MapVisualizerPlugin

logger = logging.getLogger(__name__)

# Register visualizer plugins
visualizer_registry.register(BarChartPlugin)
visualizer_registry.register(HeatmapPlugin)
visualizer_registry.register(MapVisualizerPlugin)


def plot_containment(
    file_path: Union[str, Path],
    column: str = "max_containment",
    title: Optional[str] = None,
    colors: Optional[Union[str, List[str]]] = None,
    show_title: bool = True,
    save_format: Optional[str] = None,
    threshold: Optional[float] = None,
    plot_type: str = "rank",
) -> Optional[plt.Figure]:
    """
    Plot containment data with various plot types.

    Args:
        file_path: Path to the containment file
        column: Column to plot
        title: Title for the plot
        colors: Colors to use in the plot
        show_title: Whether to display the title
        save_format: Format to save the plot (png, jpg, pdf, svg)
        threshold: Minimum value to be included in the plot
        plot_type: Type of plot to generate ('rank', 'histogram', 'box', 'violin')

    Returns:
        Matplotlib Figure if successful, None otherwise

    Raises:
        VisualizationError: If the visualization fails
    """
    try:
        # Load data
        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Check if column exists
        if column not in df.columns:
            raise VisualizationError(
                f"Column '{column}' not found in file. "
                f"Available columns: {', '.join(df.columns)}"
            )

        # Apply threshold if specified
        if threshold is not None:
            df = df[df[column] >= threshold]
            if df.empty:
                logger.warning(f"No data above threshold {threshold}")
                return None

        # Set title if not specified
        if title is None and show_title:
            title = f"{plot_type.capitalize()} Plot of {column}"
        elif not show_title:
            title = None

        # Set default colors if not specified
        if colors is None:
            colors = "blue"

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create plot based on plot_type
        if plot_type == "rank":
            # Sort data
            df_sorted = df.sort_values(by=column, ascending=False)

            # Add rank column
            df_sorted["rank"] = np.arange(1, len(df_sorted) + 1)

            # Plot rank vs value
            ax.scatter(df_sorted["rank"], df_sorted[column], color=colors)
            ax.set_xlabel("Rank")
            ax.set_ylabel(f"{column} Value")

        elif plot_type == "histogram":
            # Create histogram
            ax.hist(df[column], bins=20, color=colors, alpha=0.7)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

        elif plot_type == "box":
            # Create box plot
            ax.boxplot(df[column])
            ax.set_ylabel(column)
            ax.set_xticklabels([column])

        elif plot_type == "violin":
            # Create violin plot
            ax.violinplot(df[column])
            ax.set_ylabel(column)
            ax.set_xticklabels([column])

        else:
            raise VisualizationError(
                f"Unknown plot type: {plot_type}. "
                "Supported types: rank, histogram, box, violin"
            )

        # Add title if specified
        if title:
            ax.set_title(title)

        # Adjust layout
        plt.tight_layout()

        # Save plot if format specified
        if save_format:
            output_file = f"{Path(file_path).stem}_{plot_type}_{column}.{save_format}"
            plt.savefig(output_file, format=save_format, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {output_file}")

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Error plotting containment: {e}")


def plot_metadata_counts(
    file_path: Union[str, Path],
    title: Optional[str] = None,
    plot_type: str = "bar",
    colors: Optional[Union[str, List[str]]] = None,
    show_title: bool = True,
    save_format: Optional[str] = None,
    limit: int = 20,
) -> Optional[plt.Figure]:
    """
    Plot metadata counts with various plot types.

    Args:
        file_path: Path to the metadata counts file
        title: Title for the plot
        plot_type: Type of plot to generate ('bar', 'pie', 'radar')
        colors: Colors to use in the plot
        show_title: Whether to display the title
        save_format: Format to save the plot (png, jpg, pdf, svg)
        limit: Maximum number of items to include in the plot

    Returns:
        Matplotlib Figure if successful, None otherwise

    Raises:
        VisualizationError: If the visualization fails
    """
    try:
        # Load data
        df = pd.read_csv(file_path, sep="\t", header=None)

        # If file has two columns, assume it's [category, count]
        if df.shape[1] >= 2:
            # Set column names
            df.columns = ["category", "count"] + [
                f"col{i+3}" for i in range(df.shape[1] - 2)
            ]

            # Sort by count and limit number of items
            df = df.sort_values(by="count", ascending=False).head(limit)

        else:
            # Single column format, can't create plot
            raise VisualizationError(
                "File must have at least two columns (category and count)"
            )

        # Set default title if not specified
        if title is None and show_title:
            title = f"Top {len(df)} Categories"
        elif not show_title:
            title = None

        # Create plot based on plot_type
        if plot_type == "bar":
            # Use bar chart plugin
            plugin = visualizer_registry.get("bar")
            fig = plugin.create_plot(
                data=df,
                x_column="category",
                y_column="count",
                title=title,
                colors=colors,
                horizontal=True,
                output_format=save_format if save_format else None,
            )

        elif plot_type == "pie":
            # Create pie chart (limit to top items for readability)
            fig, ax = plt.subplots(figsize=(10, 10))

            # Calculate percentage for "Other" category if data was limited
            total_count = df["count"].sum()

            # Create pie chart
            df["category"] = df["category"].astype(str)
            ax.pie(
                df["count"],
                labels=df["category"],
                autopct="%1.1f%%",
                startangle=90,
                colors=colors if colors else plt.cm.tab20.colors,
            )
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

            if title:
                ax.set_title(title)

        elif plot_type == "radar":
            # Create radar chart (requires at least 3 categories)
            if len(df) < 3:
                logger.warning(
                    "Radar chart requires at least 3 categories, falling back to bar chart"
                )
                return plot_metadata_counts(
                    file_path, title, "bar", colors, show_title, save_format
                )

            # Set up radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

            # Get categories and values
            categories = df["category"].tolist()
            values = df["count"].tolist()

            # Number of categories
            N = len(categories)

            # Create angle values
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

            # Make the plot circular by repeating the first value
            values += [values[0]]
            angles += [angles[0]]

            # Plot data
            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                color=colors if isinstance(colors, str) else "blue",
            )
            ax.fill(
                angles,
                values,
                alpha=0.25,
                color=colors if isinstance(colors, str) else "blue",
            )

            # Set category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            if title:
                ax.set_title(title)

        else:
            raise VisualizationError(
                f"Unknown plot type: {plot_type}. " "Supported types: bar, pie, radar"
            )

        # Save plot if format specified
        if save_format and plot_type != "bar":  # bar plugin handles saving
            output_file = f"{Path(file_path).stem}_{plot_type}.{save_format}"
            plt.savefig(output_file, format=save_format, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {output_file}")

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Error plotting metadata counts: {e}")


def plot_heatmap(
    data: Union[str, Path, pd.DataFrame],
    title: Optional[str] = None,
    threshold: float = 0.0,
    cluster: bool = True,
    output_file: Optional[Union[str, Path]] = None,
    output_format: str = "png",
) -> Optional[plt.Figure]:
    """
    Create a heatmap visualization.

    Args:
        data: DataFrame or path to data file
        title: Title for the plot
        threshold: Minimum value threshold
        cluster: Whether to cluster rows and columns
        output_file: Path to save the plot
        output_format: Format to save the plot

    Returns:
        Matplotlib Figure if successful, None otherwise

    Raises:
        VisualizationError: If the visualization fails
    """
    try:
        # Load data if string or Path
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data, sep="\t", index_col=0)
        else:
            df = data.copy()

        # Apply threshold
        df = df.applymap(lambda x: x if x > threshold else 0)

        # Remove metadata columns if present
        metadata_cols = ["max_containment", "max_containment_annotation"]
        for col in metadata_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Use heatmap plugin
        plugin = visualizer_registry.get("heatmap")
        fig = plugin.create_plot(
            data=df,
            title=title,
            cluster=cluster,
            output_file=output_file,
            output_format=output_format,
        )

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Error creating heatmap: {e}")


def plot_correlation_matrix(
    data: Union[str, Path, pd.DataFrame],
    title: Optional[str] = None,
    method: str = "pearson",
    output_file: Optional[Union[str, Path]] = None,
    output_format: str = "png",
) -> Optional[plt.Figure]:
    """
    Create a correlation matrix visualization.

    Args:
        data: DataFrame or path to data file
        title: Title for the plot
        method: Correlation method ('pearson', 'spearman', 'kendall')
        output_file: Path to save the plot
        output_format: Format to save the plot

    Returns:
        Matplotlib Figure if successful, None otherwise

    Raises:
        VisualizationError: If the visualization fails
    """
    try:
        # Load data if string or Path
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data, sep="\t", index_col=0)
        else:
            df = data.copy()

        # Use heatmap plugin to create correlation matrix
        plugin = visualizer_registry.get("heatmap")
        fig = plugin.create_correlation_heatmap(
            data=df,
            title=title,
            method=method,
            output_file=output_file,
            output_format=output_format,
        )

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Error creating correlation matrix: {e}")