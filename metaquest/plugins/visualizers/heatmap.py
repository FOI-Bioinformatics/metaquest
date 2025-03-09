"""
Heatmap visualization plugin for MetaQuest.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from metaquest.core.exceptions import VisualizationError
from metaquest.plugins.base import Plugin

logger = logging.getLogger(__name__)


class HeatmapPlugin(Plugin):
    """Plugin for creating heatmap visualizations."""
    
    name = "heatmap"
    description = "Heatmap visualization"
    version = "0.1.0"
    
    @classmethod
    def create_plot(
        cls,
        data: pd.DataFrame,
        title: Optional[str] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (12, 10),
        cluster: bool = True,
        annot: bool = False,
        linewidths: float = 0,
        output_file: Optional[Union[str, Path]] = None,
        output_format: str = "png",
        **kwargs
    ) -> plt.Figure:
        """
        Create a heatmap visualization.
        
        Args:
            data: DataFrame containing data to plot
            title: Title for the plot
            cmap: Colormap to use
            figsize: Figure size (width, height) in inches
            cluster: If True, cluster rows and columns
            annot: If True, annotate cells with values
            linewidths: Width of lines separating cells
            output_file: Path to save the plot
            output_format: Format to save the plot (png, jpg, pdf, svg)
            **kwargs: Additional arguments to pass to plotting function
            
        Returns:
            Matplotlib Figure object
            
        Raises:
            VisualizationError: If the plot cannot be created
        """
        try:
            # Prepare data
            df = data.copy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create plot
            if cluster:
                clustergrid = sns.clustermap(
                    df, 
                    cmap=cmap,
                    linewidths=linewidths,
                    annot=annot,
                    **kwargs
                )
                
                # Add title if specified
                if title:
                    clustergrid.fig.suptitle(title, y=1.02)
                
                # Get figure from clustergrid
                fig = clustergrid.fig
                
            else:
                # Create regular heatmap
                sns.heatmap(
                    df,
                    cmap=cmap,
                    linewidths=linewidths,
                    annot=annot,
                    ax=ax,
                    **kwargs
                )
                
                # Add title if specified
                if title:
                    ax.set_title(title)
            
                # Adjust layout
                plt.tight_layout()
            
            # Save plot if output_file specified
            if output_file:
                fig.savefig(output_file, format=output_format, dpi=300, bbox_inches='tight')
                logger.info(f"Saved heatmap to {output_file}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating heatmap: {e}")
    
    @classmethod
    def create_presence_heatmap(
        cls,
        data: pd.DataFrame,
        threshold: float = 0.0,
        title: Optional[str] = None,
        cmap: str = "Blues",
        figsize: Tuple[int, int] = (14, 12),
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        output_format: str = "png",
        **kwargs
    ) -> plt.Figure:
        """
        Create a presence/absence heatmap visualization.
        
        Args:
            data: DataFrame containing containment data
            threshold: Threshold for binary presence
            title: Title for the plot
            cmap: Colormap to use
            figsize: Figure size (width, height) in inches
            cluster_rows: If True, cluster rows
            cluster_cols: If True, cluster columns
            output_file: Path to save the plot
            output_format: Format to save the plot (png, jpg, pdf, svg)
            **kwargs: Additional arguments to pass to plotting function
            
        Returns:
            Matplotlib Figure object
            
        Raises:
            VisualizationError: If the plot cannot be created
        """
        try:
            # Prepare data
            df = data.copy()
            
            # Filter out metadata columns if present
            metadata_cols = ['max_containment', 'max_containment_annotation']
            for col in metadata_cols:
                if col in df.columns:
                    df = df.drop(col, axis=1)
            
            # Apply threshold
            binary_df = (df > threshold).astype(int)
            
            # Calculate clustering
            if cluster_rows or cluster_cols:
                g = sns.clustermap(
                    binary_df,
                    cmap=cmap,
                    figsize=figsize,
                    row_cluster=cluster_rows,
                    col_cluster=cluster_cols,
                    **kwargs
                )
                
                # Add title if specified
                if title:
                    g.fig.suptitle(title, y=1.02)
                
                fig = g.fig
                
            else:
                # Create figure
                fig, ax = plt.subplots(figsize=figsize)
                
                # Create plot without clustering
                sns.heatmap(
                    binary_df,
                    cmap=cmap,
                    ax=ax,
                    **kwargs
                )
                
                # Add title if specified
                if title:
                    ax.set_title(title)
                
                # Adjust layout
                plt.tight_layout()
            
            # Save plot if output_file specified
            if output_file:
                fig.savefig(output_file, format=output_format, dpi=300, bbox_inches='tight')
                logger.info(f"Saved presence heatmap to {output_file}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating presence heatmap: {e}")
    
    @classmethod
    def create_correlation_heatmap(
        cls,
        data: pd.DataFrame,
        method: str = 'pearson',
        title: Optional[str] = None,
        cmap: str = "coolwarm",
        figsize: Tuple[int, int] = (12, 10),
        mask_upper: bool = True,
        annot: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        output_format: str = "png",
        **kwargs
    ) -> plt.Figure:
        """
        Create a correlation heatmap visualization.
        
        Args:
            data: DataFrame containing data to calculate correlations from
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            title: Title for the plot
            cmap: Colormap to use
            figsize: Figure size (width, height) in inches
            mask_upper: If True, mask the upper triangle
            annot: If True, annotate cells with values
            output_file: Path to save the plot
            output_format: Format to save the plot (png, jpg, pdf, svg)
            **kwargs: Additional arguments to pass to plotting function
            
        Returns:
            Matplotlib Figure object
            
        Raises:
            VisualizationError: If the plot cannot be created
        """
        try:
            # Calculate correlation matrix
            corr_matrix = data.corr(method=method)
            
            # Create mask for upper triangle if requested
            if mask_upper:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            else:
                mask = None
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create plot
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                annot=annot,
                fmt=".2f",
                square=True,
                linewidths=.5,
                ax=ax,
                **kwargs
            )
            
            # Add title if specified
            if title:
                ax.set_title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if output_file specified
            if output_file:
                fig.savefig(output_file, format=output_format, dpi=300, bbox_inches='tight')
                logger.info(f"Saved correlation heatmap to {output_file}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating correlation heatmap: {e}")