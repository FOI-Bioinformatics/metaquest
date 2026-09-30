"""
Runtime settings: one place that resolves every value a run can be tuned with.

Each setting is looked up, in order, from a command-line flag, a ``METAQUEST_<NAME>``
environment variable, the ``[runtime]`` table of the user config file
(``$XDG_CONFIG_HOME/metaquest/config.toml``, or ``~/.config/metaquest/config.toml``), and
finally a built-in default. The resolved value keeps a note of where it came from, so a run
can log, and ``metaquest doctor`` can show, why a value is what it is.

``main()`` calls :func:`activate` once with the parsed arguments; library code reads
:func:`active`, which builds the settings from the environment and config file alone when
``activate`` was never called. The shared data store root keeps its own precedence
(``metaquest.store.resolve``); only the config file reader is shared with it.

A bad value (an environment variable or config entry that does not parse) raises
``ConfigurationError`` naming the variable or the config key and file, never a silent default.
"""

import argparse
import logging
import math
import os
import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from metaquest.core.constants import (
    CATALOG_LOCK_WAIT_SECONDS,
    CONFIG_DIRNAME,
    CONFIG_FILENAME,
    DATASET_LOCK_STALE_SECONDS,
    DEFAULT_LOG_LEVEL,
    LOCK_HEARTBEAT_SECONDS,
    LOG_LEVELS,
    SHORT_LOCK_HEARTBEAT_SECONDS,
)
from metaquest.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

ENV_PREFIX = "METAQUEST_"
RUNTIME_TABLE = "runtime"
# The registry's own lock limits; metaquest.data.registry keeps the same numbers as module
# attributes (a test checks they agree), since core must not import the data layer.
REGISTRY_LOCK_WAIT_SECONDS = 30.0
REGISTRY_LOCK_STALE_SECONDS = 120.0


# --- the user config file ----------------------------------------------------


def config_path() -> Path:
    """Path to the user config file: $XDG_CONFIG_HOME/metaquest/config.toml, or
    ~/.config/metaquest/config.toml when XDG_CONFIG_HOME is not set."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / CONFIG_DIRNAME / CONFIG_FILENAME


def read_config() -> Dict[str, Any]:
    """Read the user config file, returning {} if it does not exist.

    A file that cannot be read or is not valid TOML raises ``ConfigurationError`` naming it.
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as e:
        raise ConfigurationError(f"Config file {path} is not valid TOML: {e}") from e
    except OSError as e:
        raise ConfigurationError(f"Config file {path} could not be read: {e}") from e


def _runtime_table(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The config file's ``[runtime]`` table, or {} when it has none."""
    table = config.get(RUNTIME_TABLE, {})
    if not isinstance(table, dict):
        raise ConfigurationError(f"Config file {config_path()}: [{RUNTIME_TABLE}] must be a table")
    return table


def runtime_table() -> Dict[str, Any]:
    """The user config file's ``[runtime]`` table as it is now, or {}; ``ConfigurationError`` if unreadable."""
    return _runtime_table(read_config())


# --- parsers: text in, typed value out, ValueError on a bad value -------------


def _number(text: str) -> float:
    """A finite float; ValueError otherwise."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def _non_negative_seconds(text: str) -> float:
    """Seconds, 0 or more."""
    value = _number(text)
    if value < 0:
        raise ValueError("expected a number of seconds, 0 or more")
    return value


def _positive_seconds(text: str) -> float:
    """Seconds, more than 0."""
    value = _number(text)
    if value <= 0:
        raise ValueError("expected a number of seconds greater than 0")
    return value


def _non_negative_number(text: str) -> float:
    """A number, 0 or more."""
    value = _number(text)
    if value < 0:
        raise ValueError("expected a number, 0 or more")
    return value


def _integer(text: str, minimum: int) -> int:
    """A whole number of at least ``minimum``."""
    value = int(text)
    if value < minimum:
        raise ValueError(f"expected a whole number, {minimum} or more")
    return value


def _non_negative_int(text: str) -> int:
    """A whole number, 0 or more."""
    return _integer(text, 0)


def _positive_int(text: str) -> int:
    """A whole number, 1 or more."""
    return _integer(text, 1)


def _boolean(text: str) -> bool:
    """true/false, yes/no, on/off or 1/0, in any case."""
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError("expected true or false")


def _log_level(text: str) -> str:
    """One of the logging level names, in upper case."""
    level = text.strip().upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"expected one of {', '.join(LOG_LEVELS)}")
    return level


def _text(text: str) -> str:
    """A non-empty string, stripped of surrounding space."""
    value = text.strip()
    if not value:
        raise ValueError("expected a non-empty value")
    return value


def _email(text: str) -> str:
    """An email address (checked only for an ``@`` between two non-empty parts)."""
    value = _text(text)
    local, _, domain = value.partition("@")
    if not local or not domain:
        raise ValueError("expected an email address")
    return value


def _path(text: str) -> str:
    """A filesystem path, with a leading ``~`` expanded."""
    return str(Path(_text(text)).expanduser())


_MEMORY_PATTERN = re.compile(r"^(auto|0?\.\d+|1(\.0*)?|[1-9]\d*|\d+(\.\d+)?[KMGT])$", re.IGNORECASE)


def _memory(text: str) -> str:
    """``auto``, a fraction of the node's memory (0 to 1), or a size such as ``32G`` or whole bytes."""
    value = _text(text)
    if not _MEMORY_PATTERN.match(value):
        raise ValueError("expected 'auto', a fraction such as 0.5, or a size such as 32G, 32000M or bytes")
    return value


# --- the settings table ------------------------------------------------------


@dataclass(frozen=True)
class SettingSpec:
    """One runtime setting: its parser, default, environment variable and matching flag.

    ``config_key`` is the name inside ``[runtime]`` (the setting's own name unless given);
    ``aliases`` are further environment variables read after ``env``; ``secret`` hides the
    value when the settings are logged.
    """

    name: str
    parse: Callable[[str], Any]
    default: Any
    env: str
    doc: str
    cli_dest: Optional[str] = None
    config_key: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    secret: bool = False

    @property
    def key(self) -> str:
        """The setting's key inside the config file's ``[runtime]`` table."""
        return self.config_key or self.name

    @property
    def flag(self) -> Optional[str]:
        """The command-line flag that sets this value, as a user types it."""
        return f"--{self.cli_dest.replace('_', '-')}" if self.cli_dest else None


def _spec(name: str, parse: Callable[[str], Any], default: Any, doc: str, **extra: Any) -> SettingSpec:
    """A ``SettingSpec`` whose environment variable is ``METAQUEST_<NAME>`` unless ``env`` is given."""
    env = extra.pop("env", ENV_PREFIX + name.upper())
    return SettingSpec(name=name, parse=parse, default=default, env=env, doc=doc, **extra)


_SPECS = (
    _spec(
        "subprocess_timeout",
        _non_negative_seconds,
        0.0,
        "Seconds before an external tool is stopped; 0 means no limit",
        env=ENV_PREFIX + "TIMEOUT",
        config_key="timeout",
        cli_dest="timeout",
    ),
    _spec("max_workers_cap", _positive_int, 4, "Upper bound on parallel download workers"),
    _spec(
        "lock_wait",
        _non_negative_seconds,
        0.0,
        "Seconds to wait for another run's work on the same accession; 0 waits while it is alive",
        cli_dest="lock_wait",
    ),
    _spec("registry_lock_wait", _positive_seconds, REGISTRY_LOCK_WAIT_SECONDS, "Seconds to wait for the registry lock"),
    _spec(
        "registry_lock_stale",
        _positive_seconds,
        REGISTRY_LOCK_STALE_SECONDS,
        "Seconds without a heartbeat after which a registry lock is taken over",
    ),
    _spec(
        "dataset_lock_stale",
        _positive_seconds,
        DATASET_LOCK_STALE_SECONDS,
        "Seconds without a heartbeat after which a dataset, index or extraction lock is taken over",
    ),
    _spec(
        "lock_heartbeat", _positive_seconds, LOCK_HEARTBEAT_SECONDS, "Seconds between refreshes of a held dataset lock"
    ),
    _spec("catalog_lock_wait", _positive_seconds, CATALOG_LOCK_WAIT_SECONDS, "Seconds to wait for the store catalogue"),
    _spec("ncbi_email", _email, None, "Email address sent with NCBI requests", cli_dest="email"),
    _spec(
        "ncbi_api_key",
        _text,
        None,
        "NCBI API key for a higher request rate",
        cli_dest="api_key",
        aliases=("NCBI_API_KEY",),
        secret=True,
    ),
    _spec("temp_folder", _path, None, "Folder for temporary files of external tools", cli_dest="temp_folder"),
    _spec("log_file", _path, None, "File that receives a copy of every log line", cli_dest="log_file"),
    _spec("log_level", _log_level, DEFAULT_LOG_LEVEL, "Console logging level", cli_dest="log_level"),
    _spec("progress_every", _non_negative_int, 50, "Items between progress summaries", cli_dest="progress_every"),
    _spec("log_host", _boolean, False, "Put the host name on every log line"),
    _spec("min_free_gb", _non_negative_number, 10.0, "Free space to keep, in GB; 0 disables", cli_dest="min_free_gb"),
    _spec(
        "assembly_memory",
        _memory,
        "auto",
        "Memory given to megahit: auto, a fraction, or a size",
        cli_dest="assembly_memory",
    ),
)

SETTINGS: Dict[str, SettingSpec] = {spec.name: spec for spec in _SPECS}


# --- resolution --------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    """One setting's value and where it came from: a flag, a variable, a config key or ``default``."""

    name: str
    value: Any
    source: str


def _parse(spec: SettingSpec, raw: Any, source: str) -> Any:
    """``raw`` parsed by the spec, or ``ConfigurationError`` naming ``source``."""
    text = ("true" if raw else "false") if isinstance(raw, bool) else str(raw)
    if isinstance(raw, (list, dict)):
        text = ""  # a TOML array or table is never a valid setting value
    try:
        if not text:
            raise ValueError("expected a single value")
        return spec.parse(text)
    except ValueError as e:
        raise ConfigurationError(f"Invalid value {raw!r} for {spec.name} from {source}: {e}") from e


def _resolve(spec: SettingSpec, cli_value: Any, runtime: Mapping[str, Any]) -> Resolved:
    """Resolve one setting against a flag value and an already-read ``[runtime]`` table."""
    if cli_value is not None and spec.flag is not None:
        return Resolved(spec.name, cli_value, spec.flag)
    for variable in (spec.env, *spec.aliases):
        raw = os.environ.get(variable)
        if raw:
            return Resolved(spec.name, _parse(spec, raw, variable), variable)
    if spec.key in runtime:
        source = f"config [{RUNTIME_TABLE}] {spec.key}"
        return Resolved(spec.name, _parse(spec, runtime[spec.key], f"{source} ({config_path()})"), source)
    return Resolved(spec.name, spec.default, "default")


def resolve_setting(name: str, cli_value: Any = None) -> Resolved:
    """Resolve the setting ``name``: ``cli_value`` when given, else environment, config, default.

    Raises ``KeyError`` for a name not in ``SETTINGS`` and ``ConfigurationError`` for a value
    that does not parse.
    """
    spec = SETTINGS[name]
    return _resolve(spec, cli_value, _runtime_table(read_config()))


def _cli_values(args: Optional[argparse.Namespace]) -> Dict[str, Any]:
    """The attributes a namespace actually holds (not those a mock would invent)."""
    if args is None:
        return {}
    try:
        return dict(vars(args))
    except TypeError:
        return {}


def config_warnings(runtime: Mapping[str, Any]) -> Tuple[str, ...]:
    """One message per key in the config file's ``[runtime]`` table that names no setting."""
    known = {spec.key for spec in _SPECS}
    return tuple(
        f"Config file {config_path()}: unknown key '{key}' in [{RUNTIME_TABLE}] is ignored"
        for key in sorted(set(runtime) - known)
    )


def _resolve_table(args: Optional[argparse.Namespace], runtime: Mapping[str, Any]) -> Dict[str, Resolved]:
    given = _cli_values(args)
    return {spec.name: _resolve(spec, given.get(spec.cli_dest or ""), runtime) for spec in _SPECS}


def resolve_all(args: Optional[argparse.Namespace] = None) -> Dict[str, Resolved]:
    """Resolve every setting, taking flag values from ``args``.

    Unknown config keys are not reported here: ``activate`` keeps them in
    ``RuntimeSettings.warnings`` for the caller to log once logging is configured.
    """
    return _resolve_table(args, _runtime_table(read_config()))


def _build(args: Optional[argparse.Namespace]) -> "RuntimeSettings":
    runtime = _runtime_table(read_config())
    return RuntimeSettings.from_resolved(_resolve_table(args, runtime), warnings=config_warnings(runtime))


@dataclass(frozen=True)
class RuntimeSettings:
    """Every runtime setting's value for this run; ``sources`` says where each one came from."""

    subprocess_timeout: float
    max_workers_cap: int
    lock_wait: float
    registry_lock_wait: float
    registry_lock_stale: float
    dataset_lock_stale: float
    lock_heartbeat: float
    catalog_lock_wait: float
    ncbi_email: Optional[str]
    ncbi_api_key: Optional[str]
    temp_folder: Optional[str]
    log_file: Optional[str]
    log_level: str
    progress_every: int
    log_host: bool
    min_free_gb: float
    assembly_memory: str
    sources: Mapping[str, str] = field(default_factory=dict)
    # Messages about the config file (unknown keys) for the caller to log; activate() runs
    # before logging is set up, so logging them there would lose them.
    warnings: Tuple[str, ...] = ()

    @classmethod
    def from_resolved(cls, resolved: Mapping[str, Resolved], warnings: Tuple[str, ...] = ()) -> "RuntimeSettings":
        """Build the settings from ``resolve_all``'s result, checking the lock limits agree."""
        values = {name: item.value for name, item in resolved.items()}
        runtime = cls(**values, sources={name: item.source for name, item in resolved.items()}, warnings=warnings)
        runtime._check_lock_limits()
        return runtime

    def _check_lock_limits(self) -> None:
        """A heartbeat must come inside the stale limit, or a live holder's lock would be taken over."""
        if self.lock_heartbeat >= self.dataset_lock_stale:
            raise ConfigurationError(
                f"lock_heartbeat ({self.lock_heartbeat} s, from {self.source('lock_heartbeat')}) must be less "
                f"than dataset_lock_stale ({self.dataset_lock_stale} s, from {self.source('dataset_lock_stale')})"
            )
        if self.registry_lock_stale <= SHORT_LOCK_HEARTBEAT_SECONDS:
            raise ConfigurationError(
                f"registry_lock_stale ({self.registry_lock_stale} s, from {self.source('registry_lock_stale')}) "
                f"must be more than the registry heartbeat of {SHORT_LOCK_HEARTBEAT_SECONDS} s"
            )

    def source(self, name: str) -> str:
        """Where the setting ``name`` came from."""
        return self.sources.get(name, "default")

    def describe(self) -> List[str]:
        """One ``name = value (source)`` line per setting, with secret values hidden."""
        lines = []
        for spec in _SPECS:
            value = getattr(self, spec.name)
            shown = "(set)" if spec.secret and value is not None else repr(value) if isinstance(value, str) else value
            lines.append(f"{spec.name} = {shown} ({self.source(spec.name)})")
        return lines


_lock = threading.Lock()
_active: Optional[RuntimeSettings] = None


def activate(args: Optional[argparse.Namespace]) -> RuntimeSettings:
    """Resolve every setting with the flags in ``args`` and make the result the active one.

    Nothing is logged: the caller logs ``runtime.warnings`` once logging is configured.
    """
    global _active
    runtime = _build(args)
    with _lock:
        _active = runtime
    return runtime


def activate_defaults() -> RuntimeSettings:
    """Make the built-in defaults the active settings, ignoring flags, environment and config file.

    Used by ``main()`` for ``metaquest doctor`` alone when ``activate`` fails, so the command
    still runs and reports the value that does not parse instead of stopping before it starts.
    """
    global _active
    resolved = {spec.name: Resolved(spec.name, spec.default, "default") for spec in _SPECS}
    runtime = RuntimeSettings.from_resolved(resolved)
    with _lock:
        _active = runtime
    return runtime


def active() -> RuntimeSettings:
    """The active settings; built from environment, config and defaults when ``activate`` was not called."""
    global _active
    with _lock:
        if _active is None:
            # Library use without activate(): the host has configured logging by now.
            _active = _build(None)
            for message in _active.warnings:
                logger.warning(message)
        return _active


def reset_for_tests() -> None:
    """Forget the active settings, so the next ``active()`` resolves them again."""
    global _active
    with _lock:
        _active = None


# --- helpers for consumers -----------------------------------------------------


def setting_for(args: Optional[argparse.Namespace], name: str) -> Any:
    """The setting ``name`` for a command: its flag in ``args`` when given, else the active value.

    Commands are also run directly with a hand-built namespace (tests, library callers) that
    ``activate`` never saw, so the flag is looked up here rather than only through ``active()``.
    """
    spec = SETTINGS[name]
    given = _cli_values(args).get(spec.cli_dest or "")
    if given is not None and spec.flag is not None:
        return given
    return getattr(active(), name)


def setting_or(name: str, fallback: Any) -> Any:
    """The active value of ``name`` when the user set it, else ``fallback``.

    Lock policies pass their module attribute as ``fallback``, read at call time, so a test
    that shrinks the attribute still takes effect while a flag, variable or config entry wins.
    """
    runtime = active()
    return getattr(runtime, name) if runtime.source(name) != "default" else fallback


def settings_or(**fallbacks: Any) -> Tuple[Any, ...]:
    """``setting_or`` for several names at once, in the order given."""
    return tuple(setting_or(name, fallback) for name, fallback in fallbacks.items())


def require_email(args: Optional[argparse.Namespace]) -> str:
    """The NCBI email address for a command, or ``ConfigurationError`` saying how to give one."""
    email = setting_for(args, "ncbi_email")
    if not email:
        spec = SETTINGS["ncbi_email"]
        raise ConfigurationError(
            f"NCBI requires an email address: give --email, set {spec.env}, "
            f"or add {spec.key} to [{RUNTIME_TABLE}] in {config_path()}"
        )
    return str(email)
