"""
Map visualization plugin for MetaQuest.
"""

import logging
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Union

from metaquest.core.exceptions import VisualizationError
from metaquest.plugins.base import Plugin

logger = logging.getLogger(__name__)

# Conditional import for cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    logger.warning("Cartopy not available. Map visualization will be limited.")


class MapVisualizerPlugin(Plugin):
    """Plugin for creating geographic map visualizations."""

    name = "map"
    description = "Geographic map visualization"
    version = "0.1.0"

    @classmethod
    def create_plot(
        cls,
        data: pd.DataFrame,
        lat_lon_column: Optional[str] = None,
        country_column: Optional[str] = None,
        value_column: Optional[str] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (12, 8),
        projection: str = "PlateCarree",
        marker_size: Union[int, List[int]] = 50,
        output_file: Optional[Union[str, Path]] = None,
        output_format: str = "png",
        **kwargs,
    ) -> plt.Figure:
        """
        Create a map visualization.

        Args:
            data: DataFrame containing geographic data
            lat_lon_column: Column containing latitude/longitude as "lat, lon"
            country_column: Column containing country names
            value_column: Column containing values for coloring/sizing
            title: Title for the plot
            cmap: Colormap to use
            figsize: Figure size (width, height) in inches
            projection: Map projection to use
            marker_size: Size of markers or base size for scaled markers
            output_file: Path to save the plot
            output_format: Format to save the plot (png, jpg, pdf, svg)
            **kwargs: Additional arguments to pass to plotting function

        Returns:
            Matplotlib Figure object

        Raises:
            VisualizationError: If the plot cannot be created
        """
        if not CARTOPY_AVAILABLE:
            raise VisualizationError(
                "Cartopy library is required for map visualization. "
                "Please install with 'pip install cartopy'"
            )

        try:
            # Create figure with projection
            proj_class = getattr(ccrs, projection)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=proj_class())

            # Add map features
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")

            # Extract coordinates if lat_lon_column is provided
            if lat_lon_column and lat_lon_column in data.columns:
                # Create latitude and longitude columns
                lats = []
                lons = []

                for coord_str in data[lat_lon_column]:
                    if pd.isna(coord_str) or not coord_str:
                        lats.append(np.nan)
                        lons.append(np.nan)
                        continue

                    # Try to parse coordinates
                    try:
                        # Handle various formats
                        if "," in coord_str:
                            # Format: "lat, lon"
                            lat_str, lon_str = coord_str.split(",")
                            lat = float(lat_str.strip().rstrip("NS"))
                            lon = float(lon_str.strip().rstrip("EW"))

                            # Handle N/S and E/W designations
                            if lat_str.strip().endswith("S"):
                                lat = -lat
                            if lon_str.strip().endswith("W"):
                                lon = -lon

                        elif " " in coord_str:
                            # Format: "lat lon"
                            lat_str, lon_str = coord_str.split()
                            lat = float(re.sub(r"[NS]", "", lat_str))
                            lon = float(re.sub(r"[EW]", "", lon_str))

                            # Handle N/S and E/W designations
                            if "S" in lat_str:
                                lat = -lat
                            if "W" in lon_str:
                                lon = -lon
                        else:
                            # Unknown format
                            lats.append(np.nan)
                            lons.append(np.nan)
                            continue

                        lats.append(lat)
                        lons.append(lon)

                    except (ValueError, IndexError):
                        lats.append(np.nan)
                        lons.append(np.nan)

                # Add to dataframe
                plot_df = data.copy()
                plot_df["latitude"] = lats
                plot_df["longitude"] = lons

                # Filter out rows with missing coordinates
                plot_df = plot_df.dropna(subset=["latitude", "longitude"])

                if not plot_df.empty:
                    # Plot points
                    if value_column and value_column in plot_df.columns:
                        # Use values for coloring
                        scatter = ax.scatter(
                            plot_df["longitude"],
                            plot_df["latitude"],
                            transform=ccrs.PlateCarree(),
                            c=plot_df[value_column],
                            s=marker_size,
                            cmap=cmap,
                            **kwargs,
                        )

                        # Add colorbar
                        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
                        cbar.set_label(value_column)

                    else:
                        # Simple scatter plot
                        ax.scatter(
                            plot_df["longitude"],
                            plot_df["latitude"],
                            transform=ccrs.PlateCarree(),
                            s=marker_size,
                            **kwargs,
                        )

            # Set global extent
            ax.set_global()

            # Add title if specified
            if title:
                ax.set_title(title)

            # Save plot if output_file specified
            if output_file:
                fig.savefig(
                    output_file, format=output_format, dpi=300, bbox_inches="tight"
                )
                logger.info(f"Saved map to {output_file}")

            return fig

        except Exception as e:
            raise VisualizationError(f"Error creating map: {e}")

    @classmethod
    def create_choropleth(
        cls,
        data: pd.DataFrame,
        country_column: str,
        value_column: str,
        title: Optional[str] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (12, 8),
        projection: str = "Robinson",
        output_file: Optional[Union[str, Path]] = None,
        output_format: str = "png",
        **kwargs,
    ) -> plt.Figure:
        """
        Create a choropleth map visualization.

        Args:
            data: DataFrame containing country data
            country_column: Column containing country names
            value_column: Column containing values for coloring
            title: Title for the plot
            cmap: Colormap to use
            figsize: Figure size (width, height) in inches
            projection: Map projection to use
            output_file: Path to save the plot
            output_format: Format to save the plot (png, jpg, pdf, svg)
            **kwargs: Additional arguments to pass to plotting function

        Returns:
            Matplotlib Figure object

        Raises:
            VisualizationError: If the plot cannot be created
        """
        if not CARTOPY_AVAILABLE:
            raise VisualizationError(
                "Cartopy library is required for choropleth visualization. "
                "Please install with 'pip install cartopy'"
            )

        try:
            # Create figure with projection
            proj_class = getattr(ccrs, projection)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=proj_class())

            # Get natural earth feature
            countries = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_0_countries",
                scale="50m",
                facecolor="none",
            )

            # Add features
            ax.add_feature(countries, edgecolor="black")
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)

            # Aggregate data by country
            country_data = data.groupby(country_column)[value_column].mean().to_dict()

            # Add country polygons
            for country in countries.geometries():
                country_name = country.attributes.get("NAME", "")
                if country_name in country_data:
                    value = country_data[country_name]
                    ax.add_geometries(
                        [country],
                        ccrs.PlateCarree(),
                        facecolor=plt.cm.get_cmap(cmap)(value),
                        edgecolor="black",
                        **kwargs,
                    )
                else:
                    ax.add_geometries(
                        [country],
                        ccrs.PlateCarree(),
                        facecolor="lightgray",
                        edgecolor="black",
                    )

            # Set global extent
            ax.set_global()

            # Add title if specified
            if title:
                ax.set_title(title)

            # Save plot if output_file specified
            if output_file:
                fig.savefig(
                    output_file, format=output_format, dpi=300, bbox_inches="tight"
                )
                logger.info(f"Saved choropleth map to {output_file}")

            return fig

        except Exception as e:
            raise VisualizationError(f"Error creating choropleth map: {e}")
