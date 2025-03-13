"""
Base plugin system for MetaQuest.

This module provides the foundational classes and registries for MetaQuest's plugin system.
Plugins allow extending functionality without modifying the core codebase.
"""

import abc
import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Generic, List, Set, Type, TypeVar

from metaquest.core.exceptions import PluginError

logger = logging.getLogger(__name__)

# Type variable for plugin classes
T = TypeVar("T", bound="Plugin")


class Plugin(abc.ABC):
    """Base class for all MetaQuest plugins."""

    # Plugin metadata
    name: str = "base"
    description: str = "Base plugin class"
    version: str = "0.1.0"

    @classmethod
    def get_name(cls) -> str:
        """Get the name of the plugin."""
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Get the description of the plugin."""
        return cls.description

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the plugin."""
        return cls.version


class PluginRegistry(Generic[T]):
    """
    Registry for managing plugin classes.

    This class maintains a registry of plugin classes of a specific type.
    It provides methods for registering, retrieving, and discovering plugins.
    """

    def __init__(self):
        self._plugins: Dict[str, Type[T]] = {}

    def register(self, plugin_class: Type[T]) -> None:
        """
        Register a plugin class.

        Args:
            plugin_class: The plugin class to register

        Raises:
            PluginError: If a plugin with the same name already exists
        """
        name = plugin_class.get_name()
        if name in self._plugins:
            raise PluginError(f"Plugin '{name}' is already registered")

        self._plugins[name] = plugin_class
        logger.debug(
            f"Registered plugin: {name} ({plugin_class.__module__}.{plugin_class.__name__})"
        )

    def unregister(self, name: str) -> None:
        """
        Unregister a plugin by name.

        Args:
            name: The name of the plugin to unregister

        Raises:
            PluginError: If the plugin is not registered
        """
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' is not registered")

        del self._plugins[name]
        logger.debug(f"Unregistered plugin: {name}")

    def get(self, name: str) -> Type[T]:
        """
        Get a plugin class by name.

        Args:
            name: The name of the plugin to retrieve

        Returns:
            The plugin class

        Raises:
            PluginError: If the plugin is not registered
        """
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' is not registered")

        return self._plugins[name]

    def list(self) -> List[str]:
        """
        List all registered plugin names.

        Returns:
            A list of registered plugin names
        """
        return list(self._plugins.keys())

    def get_all(self) -> Dict[str, Type[T]]:
        """
        Get all registered plugins.

        Returns:
            A dictionary mapping plugin names to plugin classes
        """
        return self._plugins.copy()


def discover_plugins(package: str, base_class: Type[T]) -> Set[Type[T]]:
    """
    Discover all plugins in a package that inherit from a base class.

    Args:
        package: The package to search for plugins
        base_class: The base class that plugins must inherit from

    Returns:
        A set of discovered plugin classes
    """
    discovered_plugins = set()

    try:
        package_obj = importlib.import_module(package)
        package_path = package_obj.__path__
        prefix = package_obj.__name__ + "."

        for _, name, is_pkg in pkgutil.iter_modules(package_path, prefix):
            try:
                module = importlib.import_module(name)

                for item_name, item in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(item, base_class)
                        and item is not base_class
                        and item.__module__ == module.__name__
                    ):
                        discovered_plugins.add(item)
                        logger.debug(
                            f"Discovered plugin: {item.get_name()} in {module.__name__}"
                        )

                # If it's a package, recursively discover plugins
                if is_pkg:
                    sub_plugins = discover_plugins(name, base_class)
                    discovered_plugins.update(sub_plugins)

            except Exception as e:
                logger.warning(f"Error loading module {name}: {e}")

    except ImportError as e:
        logger.warning(f"Could not import package {package}: {e}")

    return discovered_plugins


def register_discovered_plugins(
    registry: PluginRegistry[T], package: str, base_class: Type[T]
) -> None:
    """
    Discover and register all plugins in a package.

    Args:
        registry: The plugin registry to register plugins with
        package: The package to search for plugins
        base_class: The base class that plugins must inherit from
    """
    plugins = discover_plugins(package, base_class)
    for plugin in plugins:
        try:
            registry.register(plugin)
        except PluginError as e:
            logger.warning(f"Failed to register plugin {plugin.get_name()}: {e}")


# Define a type variable for the Plugin type
FormatPluginT = TypeVar("FormatPluginT", bound=Plugin)
VisualizerPluginT = TypeVar("VisualizerPluginT", bound=Plugin)

# Then change your registry definitions to:
format_registry: PluginRegistry[Plugin] = PluginRegistry()
visualizer_registry: PluginRegistry[Plugin] = PluginRegistry()
