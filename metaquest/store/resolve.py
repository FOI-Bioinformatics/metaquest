"""
Store root discovery and the small user config file that can name one.

The shared data store is one folder holding a single copy of every
downloaded metagenome, used across organism projects. Its root is found, in
order, from: an explicit flag, the METAQUEST_DATA environment variable, a
project registry's recorded root, or the user config file. This module
resolves that precedence and reads/writes the user config.
"""

import json
import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from metaquest.core.constants import CONFIG_DIRNAME, CONFIG_FILENAME, STORE_ENV
from metaquest.core.exceptions import DataAccessError
from metaquest.store.layout import read_marker
from metaquest.utils.security import SecureSubprocess

logger = logging.getLogger(__name__)


def config_path() -> Path:
    """Path to the user config file: $XDG_CONFIG_HOME/metaquest/config.toml, or
    ~/.config/metaquest/config.toml when XDG_CONFIG_HOME is not set."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / CONFIG_DIRNAME / CONFIG_FILENAME


def read_config() -> Dict[str, Any]:
    """Read the user config file, returning {} if it does not exist."""
    path = config_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _strip_store_table(text: str) -> str:
    """Remove an existing top-level [store] table block from config text."""
    lines = text.splitlines(keepends=True)
    kept = []
    in_store_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_store_table = stripped == "[store]"
            if in_store_table:
                continue
        if in_store_table:
            continue
        kept.append(line)
    return "".join(kept)


def write_config_data_root(root: Path) -> Path:
    """Write the store root into the user config's [store] table.

    Any other top-level tables already present in the file are preserved
    textually; an existing [store] table is replaced.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_text = path.read_text() if path.exists() else ""
    remaining_text = _strip_store_table(existing_text)

    if remaining_text and not remaining_text.endswith("\n"):
        remaining_text += "\n"

    root_posix = Path(root).as_posix()
    # json.dumps produces a double-quoted string with '"' and '\' escaped the
    # same way TOML basic strings require, so it doubles as a safe TOML
    # string literal for a plain path (no control characters to worry about).
    new_block = f"[store]\ndata_root = {json.dumps(root_posix)}\n"

    path.write_text(remaining_text + new_block)
    return path


def resolve_store_root(
    explicit: Optional[str],
    registry_root: Optional[str],
    require_marker: bool = True,
) -> Optional[Path]:
    """Resolve the store root, in order: explicit, METAQUEST_DATA, registry_root, config.

    Returns None when none of the rules name a root. When a rule names a root
    whose store marker is missing and require_marker is True, raises
    DataAccessError naming the rule that produced the root. On success, the
    resolved root is registered with SecureSubprocess.add_allowed_root.
    """
    env_value = os.environ.get(STORE_ENV)
    config_value = read_config().get("store", {}).get("data_root")

    candidates = [
        ("--data-root", explicit),
        (STORE_ENV, env_value),
        ("registry store.root", registry_root),
        ("config data_root", config_value),
    ]

    for rule, candidate in candidates:
        if not candidate:
            continue

        root = Path(candidate).resolve()

        if require_marker and read_marker(root) is None:
            raise DataAccessError(
                f"Store root '{root}' from {rule} has no store marker " "(missing or invalid metaquest_store.json)"
            )

        logger.info("Resolved store root from %s: %s", rule, root)
        SecureSubprocess.add_allowed_root(root)
        return root

    return None
