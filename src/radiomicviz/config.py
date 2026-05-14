"""
Configuration and preset management for radiomics extraction.

Ships built-in presets (YAML files) for common MRI radiomics workflows.
Users can use presets directly, inspect them, or provide custom YAML configs.

Usage:
    >>> from radiomicviz import list_presets, show_preset
    >>> list_presets()
    ['mri-default', 'mri-texture', 'mri-firstorder', 'mri-habitat',
     'mri-all-transforms', 'minimal']
    >>> show_preset("mri-texture")  # prints the YAML
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml

logger = logging.getLogger("radiomicviz.config")

# ---------------------------------------------------------------------------
# Preset discovery
# ---------------------------------------------------------------------------
PRESETS_DIR = Path(__file__).parent / "presets"


def _preset_path(name: str) -> Path:
    """Resolve a preset name to its YAML file path."""
    # Accept with or without .yaml extension
    stem = name.replace(".yaml", "").replace(".yml", "")
    candidates = [
        PRESETS_DIR / f"{stem}.yaml",
        PRESETS_DIR / f"{stem}.yml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Preset '{name}' not found. Available presets: {list_presets()}"
    )


def list_presets() -> list[str]:
    """
    List all available built-in preset names.

    Returns
    -------
    list of str
        Sorted list of preset names (without .yaml extension).

    Examples
    --------
    >>> from radiomicviz import list_presets
    >>> list_presets()
    ['minimal', 'mri-all-transforms', 'mri-default', ...]
    """
    if not PRESETS_DIR.exists():
        return []
    return sorted(
        p.stem for p in PRESETS_DIR.glob("*.yaml")
    )


def show_preset(name: str) -> str:
    """
    Print and return the raw YAML content of a built-in preset.

    Parameters
    ----------
    name : str
        Preset name (e.g. ``"mri-default"``).

    Returns
    -------
    str
        The YAML file content.

    Examples
    --------
    >>> from radiomicviz import show_preset
    >>> yaml_str = show_preset("mri-texture")
    """
    path = _preset_path(name)
    content = path.read_text()
    print(f"# Preset: {name}")
    print(f"# Source: {path}")
    print(content)
    return content


def load_preset(name: str) -> dict[str, Any]:
    """
    Load a preset as a parsed dictionary.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    dict
        Parsed YAML config.
    """
    path = _preset_path(name)
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded preset '%s' from %s", name, path)
    return config


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def resolve_config(
    preset: Optional[str] = None,
    config: Optional[Union[str, Path, dict]] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    """
    Resolve extraction configuration from preset, custom file, or dict.

    Priority: ``config`` > ``preset``. If both are None, falls back to
    ``"mri-default"``.  ``overrides`` are merged on top of the resolved
    config (useful for changing ``label``, ``binWidth``, etc.).

    Parameters
    ----------
    preset : str, optional
        Name of a built-in preset.
    config : str, Path, or dict, optional
        Path to a custom YAML file, or an already-parsed dict.
    overrides : dict, optional
        Key-value pairs merged into the ``setting`` section of the config.

    Returns
    -------
    tuple of (dict, str)
        (resolved_config, source_description) — the config dict and a
        human-readable string describing where it came from.

    Raises
    ------
    FileNotFoundError
        If a preset name or config path doesn't exist.
    ValueError
        If the resolved config is missing required sections.
    """
    source = ""

    if config is not None:
        if isinstance(config, dict):
            resolved = config.copy()
            source = "user-provided dict"
        else:
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path) as f:
                resolved = yaml.safe_load(f)
            source = f"custom file: {config_path}"
    elif preset is not None:
        resolved = load_preset(preset)
        source = f"preset: {preset}"
    else:
        resolved = load_preset("mri-default")
        source = "preset: mri-default (default)"

    # Apply overrides to the setting section
    if overrides:
        if "setting" not in resolved:
            resolved["setting"] = {}
        resolved["setting"].update(overrides)
        source += f" + overrides: {list(overrides.keys())}"

    # Validate config structure
    _validate_config(resolved)

    logger.info("Resolved config from %s", source)
    return resolved, source


def _validate_config(config: dict[str, Any]) -> None:
    """Basic structural validation of a PyRadiomics config dict."""
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict, got {type(config).__name__}")

    # Must have at least imageType or featureClass
    if "imageType" not in config and "featureClass" not in config:
        raise ValueError(
            "Config must contain at least 'imageType' or 'featureClass'. "
            "See PyRadiomics docs or run `radiomicviz show-preset mri-default`."
        )

    # Warn about common mistakes
    if "setting" in config:
        settings = config["setting"]
        if settings and "binWidth" in settings and "binCount" in settings:
            logger.warning(
                "Config specifies both 'binWidth' and 'binCount'. "
                "PyRadiomics will use binWidth. Remove one to avoid confusion."
            )


def config_to_yaml(config: dict[str, Any]) -> str:
    """Serialize a config dict back to YAML string."""
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def save_config(config: dict[str, Any], path: Union[str, Path]) -> Path:
    """
    Save a config dict to a YAML file.

    Parameters
    ----------
    config : dict
        Config to save.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config to %s", path)
    return path
