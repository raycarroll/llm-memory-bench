"""Platform materialization and static conformance checks."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .manifest import IntegrationProfile, PlatformProfile, ProfileError


@dataclass(frozen=True)
class MaterializationContext:
    """Non-secret values used to render a run-specific integration."""

    project_id: str | None = None
    values: Mapping[str, str] | None = None

    def as_dict(self) -> dict[str, str]:
        values = dict(self.values or {})
        if self.project_id is not None:
            values.setdefault("project_id", self.project_id)
        return values


@dataclass(frozen=True)
class MaterializationResult:
    profile_id: str
    platform: str
    target: Path
    files: tuple[Path, ...]
    effective_manifest: Path
    effective_hash: str


def _safe_relative(path_value: str, field: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ProfileError(f"{field} must be a relative path without '..'")
    if not path.parts or path == Path("."):
        raise ProfileError(f"{field} must name a file")
    return path


def _safe_target(root: Path, path_value: str, field: str) -> Path:
    relative = _safe_relative(path_value, field)
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ProfileError(f"{field} escapes the materialization target") from exc
    return target


def _render(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        rendered = value
        for key, replacement in context.items():
            rendered = rendered.replace("{{ " + key + " }}", replacement)
            rendered = rendered.replace("{{" + key + "}}", replacement)
        return rendered
    if isinstance(value, list):
        return [_render(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, context) for key, item in value.items()}
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_asset(
    profile: IntegrationProfile,
    target_root: Path,
    source_value: str,
    target_value: str,
    mode: int | None,
    context: Mapping[str, str],
) -> Path:
    source_relative = _safe_relative(source_value, "asset.source")
    source = (profile.path.parent / source_relative).resolve()
    try:
        source.relative_to(profile.path.parent.resolve())
    except ValueError as exc:
        raise ProfileError("asset.source escapes the profile directory") from exc
    if not source.is_file():
        raise ProfileError(f"asset source does not exist: {source_relative}")

    destination = _safe_target(target_root, target_value, "asset.target")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_text(encoding="utf-8")
    destination.write_text(_render(raw, context), encoding="utf-8")
    if mode is not None:
        destination.chmod(mode)
    else:
        destination.chmod(source.stat().st_mode & 0o777)
    return destination


def _materialize_mcp(
    platform: PlatformProfile,
    target_root: Path,
    context: Mapping[str, str],
) -> Path | None:
    if not platform.mcp:
        return None
    path_value = platform.mcp.get("path", ".mcp.json")
    relative = _safe_relative(path_value, "mcp.path")
    destination = _safe_target(target_root, str(relative), "mcp.path")
    servers = platform.mcp.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise ProfileError("mcp.servers must be a non-empty mapping")
    payload = {"mcpServers": _render(servers, context)}
    _write_json(destination, payload)
    destination.chmod(0o644)
    return destination


def materialize_profile(
    profile: IntegrationProfile,
    platform_name: str,
    target: str | Path,
    context: MaterializationContext | None = None,
) -> MaterializationResult:
    """Apply a profile to an isolated agent workspace.

    Secrets are deliberately not rendered.  MCP configs retain environment
    placeholders such as ``${MEMORYHUB_API_KEY}``, and hooks resolve secrets
    at runtime inside the container.
    """

    try:
        platform = profile.platforms[platform_name]
    except KeyError as exc:
        raise ProfileError(
            f"profile {profile.profile_id} has no platform {platform_name!r}"
        ) from exc

    target_root = Path(target).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    values = (context or MaterializationContext()).as_dict()
    missing_context = [
        key for key in profile.required_context if not values.get(key)
    ]
    if missing_context:
        raise ProfileError(
            f"profile {profile.profile_id} requires context values: "
            + ", ".join(missing_context)
        )

    files: list[Path] = []
    for asset in platform.assets:
        files.append(
            _materialize_asset(
                profile,
                target_root,
                asset.source,
                asset.target,
                asset.mode,
                values,
            )
        )
    mcp_path = _materialize_mcp(platform, target_root, values)
    if mcp_path is not None:
        files.append(mcp_path)

    effective_manifest = target_root / ".llm-memory-bench" / "integration-manifest.json"
    effective_manifest.parent.mkdir(parents=True, exist_ok=True)
    file_records = [
        {
            "path": str(path.relative_to(target_root)),
            "sha256": _sha256(path),
        }
        for path in sorted(files)
    ]
    manifest = {
        "api_version": profile.api_version,
        "profile": profile.profile_id,
        "platform": platform.name,
        "memory_system": profile.memory_system,
        "topology": profile.topology,
        "capabilities": list(profile.capabilities),
        "required_context": list(profile.required_context),
        "dependencies": profile.dependencies,
        "lifecycle": profile.lifecycle,
        "files": file_records,
    }
    _write_json(effective_manifest, manifest)
    effective_hash = _sha256(effective_manifest)

    return MaterializationResult(
        profile_id=profile.profile_id,
        platform=platform.name,
        target=target_root,
        files=tuple(files),
        effective_manifest=effective_manifest,
        effective_hash=effective_hash,
    )


class ProfileVerifier:
    """Static verifier for a materialized platform profile."""

    def __init__(self, profile: IntegrationProfile, platform_name: str):
        self.profile = profile
        try:
            self.platform = profile.platforms[platform_name]
        except KeyError as exc:
            raise ProfileError(
                f"profile {profile.profile_id} has no platform {platform_name!r}"
            ) from exc

    def verify_workspace(self, target: str | Path) -> list[str]:
        root = Path(target).resolve()
        errors: list[str] = []
        for asset in self.platform.assets:
            destination = _safe_target(root, asset.target, "asset.target")
            if not destination.is_file():
                errors.append(f"missing asset: {asset.target}")
            elif asset.mode is not None and destination.stat().st_mode & 0o777 != asset.mode:
                errors.append(f"wrong mode for asset: {asset.target}")

        mcp_path = self.platform.mcp.get("path", ".mcp.json") if self.platform.mcp else None
        if mcp_path:
            destination = _safe_target(root, str(mcp_path), "mcp.path")
            if not destination.is_file():
                errors.append(f"missing MCP config: {mcp_path}")
            else:
                try:
                    payload = json.loads(destination.read_text(encoding="utf-8"))
                    servers = payload["mcpServers"]
                    expected = self.platform.mcp.get("servers", {})
                    for name in expected:
                        if name not in servers:
                            errors.append(f"missing MCP server: {name}")
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid MCP config {mcp_path}: {exc}")

        manifest_path = root / ".llm-memory-bench" / "integration-manifest.json"
        if not manifest_path.is_file():
            errors.append("missing effective integration manifest")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("profile") != self.profile.profile_id:
                    errors.append("effective manifest profile does not match profile")
                if manifest.get("platform") != self.platform.name:
                    errors.append("effective manifest platform does not match platform")
                for record in manifest.get("files", []):
                    relative = record["path"]
                    expected_hash = record["sha256"]
                    path = _safe_target(root, relative, "manifest.files.path")
                    if not path.is_file():
                        errors.append(f"manifest file is missing: {relative}")
                    elif _sha256(path) != expected_hash:
                        errors.append(f"manifest hash mismatch: {relative}")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid effective integration manifest: {exc}")

        return errors

    def verify_commands(self) -> list[str]:
        return [
            f"missing runtime command: {command}"
            for command in self.platform.required_commands
            if shutil.which(command) is None
        ]
