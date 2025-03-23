"""
Map visualization plugin for MetaQuest.
"""

import logging
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast
from cartopy.mpl.geoaxes import GeoAxes

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


def _validate_cartopy_availability():
    """
    Validate that cartopy is available.

    Raises:
        VisualizationError: If cartopy is not available
    """
    if not CARTOPY_AVAILABLE:
        raise VisualizationError(
            "Cartopy library is required for map visualization. "
            "Please install with 'pip install cartopy'"
        )


def _create_map_figure(figsize, projection):
    """
    Create a map figure with projection.

    Args:
        figsize: Figure size
        projection: Map projection to use

    Returns:
        Figure and axes objects
    """
    proj_class = getattr(ccrs, projection)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=proj_class())
    ax = cast(GeoAxes, ax)
    return fig, ax


def _add_map_features(ax):
    """
    Add standard map features to axes.

    Args:
        ax: Matplotlib axes with map projection
    """
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")


def _parse_coordinate_string(coord_str):
    """
    Parse a coordinate string into latitude and longitude.

    Args:
        coord_str: String containing latitude and longitude

    Returns:
        Tuple of (latitude, longitude) or (None, None) if parsing fails
    """
    if pd.isna(coord_str) or not coord_str:
        return None, None

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
            return None, None

        return lat, lon

    except (ValueError, IndexError):
        return None, None


def _extract_coordinates(data, lat_lon_column):
    """
    Extract latitude and longitude from a column.

    Args:
        data: DataFrame containing data
        lat_lon_column: Column containing lat/lon data

    Returns:
        DataFrame with additional latitude and longitude columns
    """
    if not lat_lon_column or lat_lon_column not in data.columns:
        return None

    # Create latitude and longitude columns
    lats = []
    lons = []

    for coord_str in data[lat_lon_column]:
        lat, lon = _parse_coordinate_string(coord_str)
        lats.append(lat)
        lons.append(lon)

    # Add to dataframe
    plot_df = data.copy()
    plot_df["latitude"] = lats
    plot_df["longitude"] = lons

    # Filter out rows with missing coordinates
    plot_df = plot_df.dropna(subset=["latitude", "longitude"])

    return plot_df


def _plot_points(ax, plot_df, value_column, marker_size, cmap, **kwargs):
    """
    Plot points on a map.

    Args:
        ax: Matplotlib axes with map projection
        plot_df: DataFrame with latitude and longitude columns
        value_column: Column to use for point colors
        marker_size: Size of markers
        cmap: Colormap
        **kwargs: Additional arguments for scatter plot

    Returns:
        Scatter plot object if successful, None otherwise
    """
    if plot_df is None or plot_df.empty:
        return None

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

        return scatter
    else:
        # Simple scatter plot
        scatter = ax.scatter(
            plot_df["longitude"],
            plot_df["latitude"],
            transform=ccrs.PlateCarree(),
            s=marker_size,
            **kwargs,
        )

        return scatter


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
        _validate_cartopy_availability()

        try:
            # Create figure with projection
            fig, ax = _create_map_figure(figsize, projection)

            # Add map features
            _add_map_features(ax)

            # Extract coordinates if lat_lon_column is provided
            plot_df = _extract_coordinates(data, lat_lon_column)

            # Plot points
            _plot_points(ax, plot_df, value_column, marker_size, cmap, **kwargs)

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
            ax = cast(GeoAxes, ax)

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
