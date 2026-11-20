"""Load and validate paired demo configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .specs import DemoSpec


def load_demo_spec(path: str | Path) -> DemoSpec:
    """Load one YAML demo document into a validated :class:`DemoSpec`."""

    config_path = Path(path)
    try:
        payload: Any = yaml.safe_load(config_path.read_text())
    except OSError as exc:
        raise ValueError(f"Unable to read demo configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in demo configuration {config_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("demo configuration must contain one YAML mapping")
    try:
        return DemoSpec.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Invalid demo configuration {config_path}: {exc}") from exc
