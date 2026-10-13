"""Versioned, provider-neutral integration profile loading.

Profiles describe the complete agent-facing integration surface.  They are
data, not executable installers: platform materializers decide how to apply
the declared files, MCP endpoints, dependencies, and lifecycle contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ProfileError(ValueError):
    """Raised when an integration profile is invalid or unsafe."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class AssetSpec:
    source: str
    target: str
    mode: int | None = None

    @classmethod
    def from_dict(cls, value: Any, index: int) -> AssetSpec:
        item = _mapping(value, f"assets[{index}]")
        mode = item.get("mode")
        if mode is not None and not isinstance(mode, int):
            raise ProfileError(f"assets[{index}].mode must be an integer")
        return cls(
            source=_string(item.get("source"), f"assets[{index}].source"),
            target=_string(item.get("target"), f"assets[{index}].target"),
            mode=mode,
        )


@dataclass(frozen=True)
class PlatformProfile:
    name: str
    min_version: str | None
    assets: tuple[AssetSpec, ...]
    mcp: dict[str, Any]
    required_commands: tuple[str, ...]

    @classmethod
    def from_dict(cls, name: str, value: Any) -> PlatformProfile:
        item = _mapping(value, f"platforms.{name}")
        raw_assets = item.get("assets", [])
        if not isinstance(raw_assets, list):
            raise ProfileError(f"platforms.{name}.assets must be a list")
        mcp = item.get("mcp", {})
        if not isinstance(mcp, dict):
            raise ProfileError(f"platforms.{name}.mcp must be a mapping")
        raw_commands = item.get("required_commands", [])
        if not isinstance(raw_commands, list) or not all(
            isinstance(command, str) and command.strip() for command in raw_commands
        ):
            raise ProfileError(
                f"platforms.{name}.required_commands must be a list of strings"
            )
        return cls(
            name=name,
            min_version=item.get("min_version"),
            assets=tuple(
                AssetSpec.from_dict(asset, index)
                for index, asset in enumerate(raw_assets)
            ),
            mcp=mcp,
            required_commands=tuple(raw_commands),
        )


@dataclass(frozen=True)
class IntegrationProfile:
    path: Path
    api_version: str
    kind: str
    name: str
    version: str
    memory_system: dict[str, Any]
    topology: dict[str, Any]
    capabilities: tuple[str, ...]
    required_context: tuple[str, ...]
    dependencies: dict[str, Any]
    platforms: dict[str, PlatformProfile]
    lifecycle: dict[str, Any]

    @property
    def profile_id(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def from_dict(cls, path: Path, value: Any) -> IntegrationProfile:
        data = _mapping(value, "profile")
        metadata = _mapping(data.get("metadata"), "metadata")
        memory_system = _mapping(data.get("memory_system"), "memory_system")
        topology = _mapping(data.get("topology"), "topology")
        dependencies = _mapping(data.get("dependencies", {}), "dependencies")
        raw_platforms = _mapping(data.get("platforms"), "platforms")
        lifecycle = _mapping(data.get("lifecycle", {}), "lifecycle")

        raw_capabilities = data.get("capabilities", [])
        if not isinstance(raw_capabilities, list) or not all(
            isinstance(capability, str) and capability.strip()
            for capability in raw_capabilities
        ):
            raise ProfileError("capabilities must be a list of strings")

        raw_context = data.get("required_context", [])
        if not isinstance(raw_context, list) or not all(
            isinstance(value, str) and value.strip() for value in raw_context
        ):
            raise ProfileError("required_context must be a list of strings")

        if data.get("kind") != "AgentMemoryIntegration":
            raise ProfileError("kind must be AgentMemoryIntegration")

        return cls(
            path=path.resolve(),
            api_version=_string(data.get("api_version"), "api_version"),
            kind="AgentMemoryIntegration",
            name=_string(metadata.get("name"), "metadata.name"),
            version=_string(metadata.get("version"), "metadata.version"),
            memory_system=memory_system,
            topology=topology,
            capabilities=tuple(raw_capabilities),
            required_context=tuple(raw_context),
            dependencies=dependencies,
            platforms={
                platform_name: PlatformProfile.from_dict(platform_name, platform)
                for platform_name, platform in raw_platforms.items()
            },
            lifecycle=lifecycle,
        )


def load_profile(path: str | Path) -> IntegrationProfile:
    """Load and validate a profile YAML file."""

    profile_path = Path(path).resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    with profile_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return IntegrationProfile.from_dict(profile_path, data)
