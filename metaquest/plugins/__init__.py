"""
Plugin system for MetaQuest.

This package provides the plugin system for extending MetaQuest functionality.
"""

# Import commonly used components for easier access
from metaquest.plugins.base import Plugin, PluginRegistry
from metaquest.plugins.base import format_registry, visualizer_registry
from metaquest.plugins.base import register_discovered_plugins