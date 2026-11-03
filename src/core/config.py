"""Small, strict YAML/JSON configuration loader; CLI options take precedence."""
import json
from pathlib import Path
import yaml

from .paths import ROOT


def merge(base, update):
    result = dict(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"Cyclic configuration include: {path}")
    text = path.read_text(encoding="utf-8")
    config = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    defaults = config.pop("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must map section names to YAML/JSON files")
    composed = {}
    for section, relative in defaults.items():
        included = load_config(path.parent / relative, (*_stack, path))
        composed[section] = included
    return merge(composed, config)


def project_path(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value
