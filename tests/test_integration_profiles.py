from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_memory_bench.integrations import (
    MaterializationContext,
    ProfileError,
    ProfileVerifier,
    load_profile,
    materialize_profile,
)
from llm_memory_bench.integrations.manifest import AssetSpec, PlatformProfile
from llm_memory_bench.bench import RunConfig, run_benchmark


PROFILE = Path(__file__).parents[1] / "integrations/profiles/memoryhub/remote-http-v1/profile.yaml"


def test_memoryhub_profile_materializes_full_claude_surface(tmp_path: Path) -> None:
    profile = load_profile(PROFILE)
    result = materialize_profile(
        profile,
        "claude-code",
        tmp_path,
        MaterializationContext(project_id="benchmark-abc"),
    )

    assert ProfileVerifier(profile, "claude-code").verify_workspace(tmp_path) == []
    assert {
        path.relative_to(tmp_path).as_posix() for path in result.files
    } == {
        ".memoryhub.yaml",
        ".mcp.json",
        ".claude/rules/memoryhub-loading.md",
        ".claude/hooks/load-memories.sh",
        ".claude/settings.json",
        ".llm-memory-bench/memoryhub-mcp-proxy.py",
    }
    assert 'project_id: "benchmark-abc"' in (tmp_path / ".memoryhub.yaml").read_text()
    assert "mh-dev-" not in (tmp_path / ".mcp.json").read_text()
    assert "MEMORYHUB_API_KEY" in (
        tmp_path / ".claude/hooks/load-memories.sh"
    ).read_text()
    hook = (tmp_path / ".claude/hooks/load-memories.sh").read_text()
    assert '--scope project' in hook
    proxy = (tmp_path / ".llm-memory-bench/memoryhub-mcp-proxy.py").read_text()
    assert "_isolation_arguments" in proxy
    assert 'args["scope"] = "project"' in proxy
    assert json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["memoryhub"]["type"] == "stdio"

    manifest = json.loads(result.effective_manifest.read_text())
    assert manifest["profile"] == "memoryhub-remote-http@1.0.0"
    assert all("api_key" not in json.dumps(record).lower() for record in manifest["files"])


def test_profile_requires_run_context(tmp_path: Path) -> None:
    profile = load_profile(PROFILE)

    with pytest.raises(ProfileError, match="project_id"):
        materialize_profile(profile, "claude-code", tmp_path)


def test_verifier_detects_effective_config_drift(tmp_path: Path) -> None:
    profile = load_profile(PROFILE)
    materialize_profile(
        profile,
        "claude-code",
        tmp_path,
        MaterializationContext(project_id="benchmark-abc"),
    )
    (tmp_path / ".claude/settings.json").write_text("{}\n")

    errors = ProfileVerifier(profile, "claude-code").verify_workspace(tmp_path)
    assert "manifest hash mismatch: .claude/settings.json" in errors


def test_materializer_rejects_unsafe_asset_target(tmp_path: Path) -> None:
    profile = load_profile(PROFILE)

    with pytest.raises(ProfileError):
        unsafe_platform = PlatformProfile(
            name="claude-code",
            min_version=None,
            assets=(AssetSpec("assets/.memoryhub.yaml", "../escape"),),
            mcp={},
            required_commands=(),
        )
        unsafe_profile = profile.__class__(
            path=profile.path,
            api_version=profile.api_version,
            kind=profile.kind,
            name=profile.name,
            version=profile.version,
            memory_system=profile.memory_system,
            topology=profile.topology,
            capabilities=profile.capabilities,
            required_context=profile.required_context,
            dependencies=profile.dependencies,
            platforms={"claude-code": unsafe_platform},
            lifecycle=profile.lifecycle,
        )
        materialize_profile(
            unsafe_profile,
            "claude-code",
            tmp_path,
            MaterializationContext(project_id="benchmark-abc"),
        )


def test_container_runner_rejects_unsafe_remote_reuse() -> None:
    config = RunConfig.from_yaml(Path("configs/claude-memoryhub-remote.yaml"))
    config = replace(
        config,
        dataset_path="datasets/test_sequential.yaml",
        max_conversations=2,
    )

    with pytest.raises(RuntimeError, match="lifecycle.reset is unsupported"):
        run_benchmark(config)
