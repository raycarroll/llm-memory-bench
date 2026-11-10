"""Orchestrates benchmark runs: build image, start container, drive conversations, collect results."""

from __future__ import annotations

import json
import base64
from functools import lru_cache
import os
import shlex
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console

from .dataset import Conversation, Dataset, load_dataset
from .integrations import (
    MaterializationContext,
    ProfileError,
    ProfileVerifier,
    IntegrationProfile,
    load_profile,
    materialize_profile,
)
from .seeding import (
    SeedError,
    SeedPlan,
    SeedResult,
    cleanup_benchmark_data,
    load_seed_plan,
    seed_benchmark_data,
)

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONTAINER_LOG_FIFO = "/run/llm-memory-bench/events"


@dataclass
class RunConfig:
    host_name: str
    host_version: str
    system_name: str
    system_version: str | None
    system_env: dict[str, str]
    model: str
    dataset_path: str
    max_conversations: int | None
    agent_sdk_version: str = "0.2.165"
    integration_profile: str | None = None
    integration_platform: str | None = None
    seed_plan: SeedPlan | None = None
    keep_container: bool = False
    require_isolation: bool = True  # Enforce conversation isolation (reset between conversations)
    cleanup_before_run: bool = True  # Run integration cleanup before benchmark
    experiment_group: str | None = None  # Group ID for comparing/aggregating runs
    experiment_metadata: dict[str, str] = field(default_factory=dict)  # Additional experiment context
    experiment_runs: int = 1  # Number of times to run this experiment

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        host = data["host"]
        system = data.get("system", {})
        integration = data.get("integration", {})
        benchmark = data.get("benchmark", {})
        if not isinstance(benchmark, dict):
            raise ValueError("benchmark must be a mapping")
        seed_config = benchmark.get("seed")
        seed_plan = None
        if seed_config is not None:
            if not isinstance(seed_config, dict):
                raise ValueError("benchmark.seed must be a mapping")
            fixture = seed_config.get("fixture")
            if not isinstance(fixture, str) or not fixture.strip():
                raise ValueError("benchmark.seed.fixture must be a path")
            fixture_path = Path(fixture)
            if not fixture_path.is_absolute():
                config_relative = Path(path).resolve().parent / fixture_path
                fixture_path = (
                    config_relative
                    if config_relative.exists()
                    else PROJECT_ROOT / fixture_path
                )
            try:
                seed_plan = load_seed_plan(fixture_path)
            except (FileNotFoundError, SeedError) as exc:
                raise ValueError(f"Invalid benchmark seed: {exc}") from exc
        integration_profile = integration.get("profile")
        container = data.get("container", {})
        if not isinstance(container, dict):
            raise ValueError("container must be a mapping")
        keep_container = container.get("keep_alive", False)
        if not isinstance(keep_container, bool):
            raise ValueError("container.keep_alive must be a boolean")
        cleanup_before_run = container.get("cleanup_before_run", True)
        if not isinstance(cleanup_before_run, bool):
            raise ValueError("container.cleanup_before_run must be a boolean")
        if integration_profile:
            profile_path = Path(integration_profile)
            if not profile_path.is_absolute():
                candidate = PROJECT_ROOT / profile_path
                if candidate.exists():
                    integration_profile = str(candidate)
        profile_name = (
            Path(integration_profile).parent.name
            if integration_profile
            else "profile"
        )
        experiment = data.get("experiment", {})
        if not isinstance(experiment, dict):
            raise ValueError("experiment must be a mapping")

        experiment_runs = experiment.get("runs", 1)
        if not isinstance(experiment_runs, int) or experiment_runs < 1:
            raise ValueError("experiment.runs must be a positive integer")

        return cls(
            host_name=host["name"],
            host_version=host["version"],
            system_name=system.get("name", profile_name),
            system_version=system.get("version"),
            system_env={**system.get("env", {}), **integration.get("env", {})},
            model=data["model"],
            dataset_path=data["dataset"],
            max_conversations=data.get("max_conversations"),
            agent_sdk_version=host.get("sdk_version", "0.2.165"),
            integration_profile=integration_profile,
            integration_platform=integration.get("platform"),
            seed_plan=seed_plan,
            keep_container=keep_container,
            require_isolation=container.get("require_isolation", True),
            cleanup_before_run=cleanup_before_run,
            experiment_group=experiment.get("group_id"),
            experiment_metadata=experiment.get("metadata", {}),
            experiment_runs=experiment_runs,
        )

    @property
    def uses_integration_profile(self) -> bool:
        return self.integration_profile is not None

    @property
    def image_tag(self) -> str:
        return (
            f"llm-bench/{self.host_name}:{self.host_version}"
            f"-sdk{self.agent_sdk_version}"
        )

    def container_name(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"llm-bench-{self.host_name}-{self.system_name}-{ts}"


@dataclass
class ConversationResult:
    conversation_id: str
    stored: list[dict] = field(default_factory=list)  # Extracted memories (final output)
    ground_truth: list[dict] = field(default_factory=list)
    driver_log: str = ""
    conversation_turns: list[dict] = field(default_factory=list)  # Original user/assistant turns
    hook_storage_events: list[dict] = field(default_factory=list)  # What hooks captured
    retention_calls: list[dict] = field(default_factory=list)  # Raw data sent to retain API


@dataclass
class BenchResult:
    config: RunConfig
    conversation_results: list[ConversationResult] = field(default_factory=list)
    environment: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _container_runtime() -> str:
    runtime = os.environ.get("LLM_MEMORY_BENCH_CONTAINER_RUNTIME", "").strip().lower()
    if runtime:
        if runtime not in {"docker", "podman"}:
            raise RuntimeError(
                "LLM_MEMORY_BENCH_CONTAINER_RUNTIME must be 'docker' or 'podman'"
            )
        return runtime

    # Prefer Podman on this project, then fall back to Docker for hosts where
    # the Podman machine is unavailable. An explicit environment override can
    # still force either runtime.
    for candidate in ("podman", "docker"):
        try:
            probe = subprocess.run(
                [candidate, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return candidate

    raise RuntimeError(
        "No reachable container runtime. Start Docker Desktop or run "
        "`podman machine start`; optionally set "
        "LLM_MEMORY_BENCH_CONTAINER_RUNTIME to docker or podman."
    )


def _docker(*args, check=True, **kwargs) -> subprocess.CompletedProcess:
    """Run a Docker-compatible container CLI selected for this benchmark."""

    return subprocess.run([_container_runtime(), *args], **kwargs, check=check)


def _resolve_config_env(key: str, value: str) -> str:
    """Resolve the small env-reference form allowed in run YAML."""

    if not isinstance(value, str):
        raise RuntimeError(f"Environment value for {key} must be a string")
    if not (value.startswith("${") and value.endswith("}")):
        return value
    expression = value[2:-1]
    if ":-" in expression:
        env_name, default = expression.split(":-", 1)
        return os.environ.get(env_name, default)
    return os.environ.get(expression, "")


def _vertex_enabled(config: RunConfig) -> bool:
    """Return whether the container is configured to use Vertex AI."""

    configured = config.system_env.get("CLAUDE_CODE_USE_VERTEX")
    value = (
        _resolve_config_env("CLAUDE_CODE_USE_VERTEX", configured)
        if configured is not None
        else os.environ.get("CLAUDE_CODE_USE_VERTEX", "")
    )
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _vertex_credentials_path(config: RunConfig) -> Path:
    """Resolve the host-side ADC file that must be mounted into Docker."""

    configured = config.system_env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if configured is not None:
        raw_path = _resolve_config_env("GOOGLE_APPLICATION_CREDENTIALS", configured)
    else:
        raw_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    path = (
        Path(raw_path).expanduser()
        if raw_path
        else Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    return path.resolve()


def build_host_image(config: RunConfig) -> None:
    host_dir = PROJECT_ROOT / "hosts" / config.host_name
    if not host_dir.exists():
        raise FileNotFoundError(f"No host definition at {host_dir}")

    result = _docker(
        "image", "inspect", config.image_tag,
        capture_output=True, check=False,
    )
    if result.returncode == 0:
        console.print(f"Image {config.image_tag} exists.", style="dim")
        return

    console.print(f"Building {config.image_tag}...")
    _docker(
        "build",
        "--build-arg", f"CLAUDE_CODE_VERSION={config.host_version}",
        "--build-arg", f"CLAUDE_AGENT_SDK_VERSION={config.agent_sdk_version}",
        "-t", config.image_tag,
        str(host_dir),
    )


def load_integration(config: RunConfig) -> IntegrationProfile | None:
    if not config.integration_profile:
        return None
    try:
        return load_profile(config.integration_profile)
    except (FileNotFoundError, ProfileError) as exc:
        raise RuntimeError(f"Invalid integration profile: {exc}") from exc


def materialize_integration(
    config: RunConfig,
    profile: IntegrationProfile,
    target: str | Path,
    project_id: str,
) -> str:
    platform = config.integration_platform or config.host_name

    # Expand env vars needed for materialization (e.g., MCP HTTP URLs)
    context_values = {"project_id": project_id}
    for key, value in config.system_env.items():
        if value:
            context_values[key] = _resolve_config_env(key, value)

    try:
        result = materialize_profile(
            profile,
            platform,
            target,
            MaterializationContext(
                project_id=project_id,
                values=context_values,
            ),
        )
        errors = ProfileVerifier(profile, platform).verify_workspace(target)
    except ProfileError as exc:
        raise RuntimeError(f"Integration materialization failed: {exc}") from exc
    if errors:
        raise RuntimeError(
            "Materialized integration failed verification:\n- "
            + "\n- ".join(errors)
        )
    return result.effective_hash


def validate_integration_environment(
    config: RunConfig,
    profile: IntegrationProfile,
) -> None:
    runtime_env = profile.topology.get("runtime_env", {})
    if not isinstance(runtime_env, dict):
        return

    def value_for(key: str) -> str:
        configured = config.system_env.get(key)
        if configured is not None:
            return _resolve_config_env(key, configured)
        return os.environ.get(key, "")

    missing = [
        key for key in runtime_env.get("required", [])
        if isinstance(key, str) and not value_for(key)
    ]
    if missing:
        raise RuntimeError(
            f"Integration {profile.profile_id} requires runtime environment: "
            + ", ".join(missing)
        )

    one_of = runtime_env.get("one_of", [])
    if isinstance(one_of, list) and one_of and not any(
        isinstance(key, str) and value_for(key) for key in one_of
    ):
        raise RuntimeError(
            f"Integration {profile.profile_id} requires one of: "
            + ", ".join(str(key) for key in one_of)
        )


def resolved_system_environment(config: RunConfig) -> dict[str, str]:
    """Resolve configured provider environment without exposing secret values."""

    resolved: dict[str, str] = {}
    for key, value in config.system_env.items():
        resolved[key] = _resolve_config_env(key, value)
    return resolved


def start_container(
    config: RunConfig,
    profile: IntegrationProfile | None = None,
    project_id: str | None = None,
    legacy_install: bool = True,
) -> str:
    system_dir = PROJECT_ROOT / "systems" / config.system_name
    if profile is None and legacy_install and not system_dir.exists():
        raise FileNotFoundError(f"No system definition at {system_dir}")

    host_dir = PROJECT_ROOT / "hosts" / config.host_name

    env_args = []
    for key in [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLOUD_ML_REGION",
        "CLAUDE_CODE_USE_VERTEX",
    ]:
        val = os.environ.get(key)
        if val:
            env_args.extend(["-e", f"{key}={val}"])

    for key, val in config.system_env.items():
        # Host paths must be rewritten to the mounted path below. Passing the
        # host path through would make it meaningless inside the container.
        if key in {"GOOGLE_APPLICATION_CREDENTIALS", "MEMORYHUB_SEED_URL"}:
            continue
        env_args.extend(["-e", f"{key}={_resolve_config_env(key, val)}"])

    if project_id:
        env_args.extend(["-e", f"MEMORYHUB_PROJECT_ID={project_id}"])

    # Hooks, plugins, and bridge scripts use this provider-neutral event sink
    # to report lifecycle/context delivery without putting memory contents or
    # credentials in the benchmark log.
    env_args.extend([
        "-e",
        "LLM_MEMORY_BENCH_EVENT_LOG=/run/llm-memory-bench/integration-events.jsonl",
        "-e",
        "LLM_MEMORY_BENCH_CAPTURE_HOOK_OUTPUT=true",
    ])

    if config.system_version:
        env_args.extend([
            "-e", f"{config.system_name.upper().replace('-', '_')}_VERSION={config.system_version}",
        ])

    name = config.container_name()
    mounts = ["-v", f"{host_dir}:/host-driver:ro"]
    if _vertex_enabled(config):
        adc_path = _vertex_credentials_path(config)
        if not adc_path.is_file():
            raise RuntimeError(
                "Claude Code Vertex runs require Google Application Default "
                f"Credentials, but no file was found at {adc_path}. Run "
                "`gcloud auth application-default login` or set "
                "GOOGLE_APPLICATION_CREDENTIALS to a valid host-side file."
            )
        container_adc_path = "/run/llm-memory-bench/google-application-default-credentials.json"
        mounts.extend(["-v", f"{adc_path}:{container_adc_path}:ro"])
        env_args.extend(["-e", f"GOOGLE_APPLICATION_CREDENTIALS={container_adc_path}"])
    if profile is None and legacy_install:
        mounts.extend([
            "-v", f"{system_dir}:/system:ro",
            "-e", "SYSTEM_INSTALL_SCRIPT=/system/install.sh",
        ])
    result = _docker(
        "run", "-d",
        "--name", name,
        *mounts,
        *env_args,
        config.image_tag,
        "bash", "-lc",
        (
            "echo 'llm-memory-bench container log stream ready' && "
            "mkdir -p /run/llm-memory-bench && "
            f"rm -f {_CONTAINER_LOG_FIFO} && "
            f"mkfifo {_CONTAINER_LOG_FIFO} && "
            # Keep the FIFO write end open in PID 1. Otherwise cat exits with
            # status 0 after the first short-lived exec writer closes, which
            # leaves subsequent podman exec calls targeting a stopped container.
            f"exec 3<> {_CONTAINER_LOG_FIFO} && exec cat <&3"
        ),
        capture_output=True, text=True,
    )
    container_id = result.stdout.strip()
    state = _docker(
        "inspect",
        "--format",
        "{{.State.Running}}",
        container_id,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or state.returncode != 0 or state.stdout.strip() != "true":
        logs = _docker(
            "logs",
            container_id,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = logs.stderr.strip() or logs.stdout.strip() or result.stderr.strip()
        raise RuntimeError(
            f"Container {name} did not remain running: {detail[:2000]}"
        )
    console.print(
        f"Container {name} ({container_id[:12]}) started; "
        f"runtime events are available in the container log pane.",
        style="dim",
    )
    return container_id


def _emit_container_output(container_id: str, text: str) -> None:
    """Forward captured exec output to the container's main log stream."""

    if not text:
        return
    encoded = base64.b64encode(text.encode("utf-8", errors="replace")).decode("ascii")
    result = _docker(
        "exec",
        container_id,
        "bash",
        "-lc",
        f"printf '%s' '{encoded}' | base64 -d > {_CONTAINER_LOG_FIFO}",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        console.print(
            f"  Container log forwarding failed: {result.stderr[:300]}",
            style="yellow",
        )


def _exec(container_id: str, cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [_container_runtime(), "exec", container_id, *cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    output = f"$ {' '.join(shlex.quote(part) for part in cmd)}\n"
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    _emit_container_output(container_id, output)
    return result


def install_system(container_id: str) -> None:
    console.print("Installing memory system...", style="dim")
    result = _exec(container_id, ["bash", "/system/install.sh"], timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"System install failed:\n{result.stderr}")
    console.print("System installed.", style="dim")


def install_integration(
    container_id: str,
    workspace: str | Path,
    profile: IntegrationProfile,
) -> None:
    """Copy a materialized workspace and install its pinned runtime deps."""

    console.print("Materializing agent integration...", style="dim")
    _docker(
        "cp",
        f"{Path(workspace).resolve()}/.",
        f"{container_id}:/workspace/",
        capture_output=True,
    )

    packages = profile.dependencies.get("python", [])
    if packages and (
        not isinstance(packages, list)
        or not all(isinstance(package, str) and package.strip() for package in packages)
    ):
        raise RuntimeError("integration dependencies.python must be a list of package strings")

    if packages:
        runtime = "/opt/llm-memory-bench/integration-runtime"
        result = _exec(container_id, ["python3", "-m", "venv", runtime], timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Integration runtime creation failed:\n{result.stderr}")
        result = _exec(
            container_id,
            [
                f"{runtime}/bin/pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                *packages,
            ],
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Integration dependency install failed:\n{result.stderr}")
        console.print("Integration runtime installed.", style="dim")

    # Run lifecycle.install hook if present (e.g., for Hindsight npm package)
    install_spec = profile.lifecycle.get("install", {})
    command = install_spec.get("command") if isinstance(install_spec, dict) else None
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        console.print("  Running integration installer...", style="dim")
        result = _exec(container_id, command, timeout=300)  # 5 min timeout for npm installs
        if result.returncode != 0:
            raise RuntimeError(f"Integration install failed:\n{result.stderr}")
        output = result.stdout.strip()
        if output:
            console.print(f"  {output}", style="dim")
        console.print("  Integration installed successfully.", style="green")


def drive_conversation(
    container_id: str,
    conversation: Conversation,
    model: str | None = None,
    profile_mode: bool = False,
    session_id: str | None = None,
) -> str:
    user_turns = [
        {"role": t.role, "content": t.content}
        for t in conversation.turns
        if t.role == "user"
    ]

    turns_json = json.dumps(user_turns)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(turns_json)
        tmp_path = f.name

    try:
        _docker(
            "cp", tmp_path, f"{container_id}:/tmp/turns.json",
            check=True, capture_output=True,
        )
    finally:
        os.unlink(tmp_path)

    cmd = [
        "python3", "/host-driver/driver.py",
        "--turns-file", "/tmp/turns.json",
    ]
    if model:
        cmd.extend(["--model", model])
    if profile_mode:
        cmd.append("--profile-mode")
    if session_id:
        cmd.extend(["--resume", session_id])

    result = _exec(container_id, cmd, timeout=1800)  # 30 minutes for long conversations

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"Container driver failed with exit code {result.returncode}: "
            f"{detail[:2000]}"
        )
    if not result.stdout.strip():
        detail = result.stderr.strip() or "no stderr output"
        raise RuntimeError(
            "Container driver returned empty stdout instead of JSON: "
            f"{detail[:2000]}"
        )

    return result.stdout


def query_stored(container_id: str) -> list[dict]:
    result = _exec(container_id, ["bash", "/system/query.sh"], timeout=30)
    if result.returncode != 0:
        console.print(f"  Query error: {result.stderr[:300]}", style="red")
        return []
    try:
        data = json.loads(result.stdout)
        return data.get("stored", [])
    except json.JSONDecodeError:
        console.print("  Query returned invalid JSON", style="red")
        return []


def query_integration(
    container_id: str,
    profile: IntegrationProfile,
    project_id: str,
) -> tuple[list[dict], list[dict]]:
    observe = profile.lifecycle.get("observe", {})
    command = observe.get("command") if isinstance(observe, dict) else None
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise RuntimeError(
            f"Integration {profile.profile_id} has no safe lifecycle.observe command"
        )
    expanded = [part.replace("${MEMORYHUB_PROJECT_ID}", project_id) for part in command]
    result = _exec(container_id, expanded, timeout=60)
    if result.returncode != 0:
        console.print(f"  Integration observation failed: {result.stderr[:300]}", style="red")
        return ([], [])
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        console.print("  Integration observation returned invalid JSON", style="red")
        return ([], [])

    # Handle different memory system response formats:
    # - MemoryHub CLI: {"status": "ok", "data": {"results": [...]}} or {"results": [...]}
    # - Hindsight API: {"items": [...], "total": N, "_benchmark_retention_calls": [...]}
    # Return tuple: (memories, retention_calls)
    if isinstance(data, dict):
        retention_calls = data.get("_benchmark_retention_calls", [])

        if isinstance(data.get("items"), list):
            return (data["items"], retention_calls)
        if isinstance(data.get("results"), list):
            return (data["results"], retention_calls)
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("results"), list):
            return (data["data"]["results"], retention_calls)

    if isinstance(data, list):
        return (data, [])
    return ([], [])


def cleanup_integration(
    container_id: str,
    profile: IntegrationProfile,
    project_id: str,
) -> None:
    """Run integration-specific cleanup (e.g., delete all memories)."""
    cleanup_spec = profile.lifecycle.get("cleanup", {})
    command = cleanup_spec.get("command") if isinstance(cleanup_spec, dict) else None
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        console.print("  No cleanup command configured, skipping", style="dim")
        return

    expanded = [part.replace("${MEMORYHUB_PROJECT_ID}", project_id) for part in command]
    console.print("  Cleaning integration state...", style="dim")
    result = _exec(container_id, expanded, timeout=120)  # Increased timeout for Hindsight API calls
    if result.returncode != 0:
        console.print(f"  Cleanup warning: {result.stderr[:200]}", style="yellow")
    else:
        output = result.stdout.strip()
        if output:
            console.print(f"  {output}", style="dim")


def cleanup_system(container_id: str) -> None:
    result = _exec(container_id, ["bash", "/system/cleanup.sh"], timeout=30)
    if result.returncode != 0:
        console.print(f"  Cleanup warning: {result.stderr[:200]}", style="yellow")


def collect_environment(container_id: str, config: RunConfig) -> dict:
    env = {
        "host": {"name": config.host_name, "version": config.host_version},
        "system": {"name": config.system_name, "version": config.system_version},
        "model": config.model,
        "system_env": config.system_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    result = _docker(
        "inspect", "--format", "{{.Image}}", container_id,
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        env["image_digest"] = result.stdout.strip()

    # Capture configuration context for debugging
    config_context = _collect_config_context(container_id)
    if config_context:
        env["config_context"] = config_context

    return env


def _collect_config_context(container_id: str) -> dict:
    """Capture MCP tools, rules, and other config details from the container."""
    context = {}

    # MCP configuration
    result = _exec(container_id, ["cat", "/workspace/.mcp.json"], timeout=10)
    if result.returncode == 0:
        try:
            mcp_config = json.loads(result.stdout)
            context["mcp_servers"] = list(mcp_config.get("mcpServers", {}).keys())
        except json.JSONDecodeError:
            pass

    # Available MCP tools (requires claude CLI)
    result = _exec(container_id, ["claude", "mcp", "list", "--json"], timeout=30)
    if result.returncode == 0:
        try:
            mcp_list = json.loads(result.stdout)
            # Extract tool names from each server
            tools = []
            for server in mcp_list.get("servers", []):
                for tool in server.get("tools", []):
                    tools.append({
                        "server": server.get("name"),
                        "tool": tool.get("name"),
                    })
            context["available_tools"] = tools
        except json.JSONDecodeError:
            pass

    # Claude Code rules
    result = _exec(container_id, ["find", "/workspace/.claude/rules/", "-type", "f", "-name", "*.md"], timeout=10)
    if result.returncode == 0:
        rules = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        context["rules"] = [r.replace("/workspace/.claude/rules/", "") for r in rules]

    # Hooks
    result = _exec(container_id, ["find", "/workspace/.claude/hooks/", "-type", "f"], timeout=10)
    if result.returncode == 0:
        hooks = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        context["hooks"] = [h.replace("/workspace/.claude/hooks/", "") for h in hooks]

    # Claude Code settings
    result = _exec(container_id, ["cat", "/workspace/.claude/settings.json"], timeout=10)
    if result.returncode == 0:
        try:
            settings = json.loads(result.stdout)
            # Only include non-sensitive config keys
            safe_keys = ["permissionMode", "model", "allowedTools", "mcpServers"]
            context["settings"] = {k: v for k, v in settings.items() if k in safe_keys}
        except json.JSONDecodeError:
            pass

    return context


def stop_container(container_id: str, keep: bool = False) -> None:
    if keep:
        console.print(
            f"Container {container_id[:12]} retained for inspection.",
            style="yellow",
        )
        return
    _docker("rm", "-f", container_id, capture_output=True, check=False)
    console.print("Container removed.", style="dim")


def extract_ground_truth(conversation: Conversation) -> list[dict]:
    """Extract ground truth from conversation.

    Handles both per-turn and cumulative ground truth:
    - Cumulative: conversation.cumulative_ground_truth (BFCL datasets)
    - Per-turn: turn.ground_truth (AlpsBench datasets)
    """
    facts = []

    # Check for cumulative ground truth first (BFCL pattern)
    if conversation.cumulative_ground_truth:
        for ef in conversation.cumulative_ground_truth.should_store:
            facts.append({"fact": ef.fact, "type": ef.type.value})
        return facts

    # Fall back to per-turn ground truth (AlpsBench pattern)
    for turn in conversation.turns:
        for ef in turn.ground_truth.should_store:
            facts.append({"fact": ef.fact, "type": ef.type.value})
    return facts


def _smoke_test_integration(
    container_id: str,
    profile: IntegrationProfile | None,
    project_id: str | None,
) -> dict:
    """Verify MCP tools can be called before running the full benchmark.

    Uses the same driver.py path as the actual benchmark to ensure we're
    testing the real code path, not a different configuration.
    """
    if profile is None or project_id is None:
        return {"status": "skipped", "reason": "no integration profile"}

    console.print("Running MCP tool smoke test...", style="dim")
    start_time = datetime.now(timezone.utc)

    # Use the profile's smoke_test command if specified, otherwise skip
    smoke_test_spec = profile.lifecycle.get("smoke_test", {})
    command = smoke_test_spec.get("command") if isinstance(smoke_test_spec, dict) else None

    if not (isinstance(command, list) and all(isinstance(part, str) for part in command)):
        return {"status": "skipped", "reason": "no smoke_test command in profile"}

    # Smoke test runs a full Claude conversation with MCP/hooks - needs time for:
    # - MCP server startup, - Hook execution (SessionStart/Stop)
    # - Agent SDK query loop with tool calls
    result = _exec(container_id, command, timeout=180)

    # Parse smoke_test.py output
    try:
        smoke_output = json.loads(result.stdout)
        tool_calls = smoke_output.get("tool_calls", [])
        test_error = smoke_output.get("error")

        if result.returncode != 0:
            # smoke_test.py returns 1 if no memory tools were called
            raise RuntimeError(
                f"MCP smoke test FAILED: no memory tools called. "
                f"Tool calls: {[tc.get('name') for tc in tool_calls]}"
            )

        if test_error:
            raise RuntimeError(f"MCP smoke test error: {test_error}")

        # Filter for memory storage tools (same logic as smoke_test.py)
        memory_keywords = ["memory", "hindsight", "retain", "remember", "store", "save"]
        memory_tools = [
            tc for tc in tool_calls
            if any(keyword in tc.get("name", "").lower() for keyword in memory_keywords)
        ]

        if not memory_tools:
            raise RuntimeError(
                f"MCP smoke test FAILED: no memory tools called. "
                f"Tool calls: {[tc.get('name') for tc in tool_calls]}"
            )

        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        console.print(
            f"  ✓ MCP smoke test passed ({len(memory_tools)} memory tool call(s))",
            style="green"
        )

        return {
            "status": "passed",
            "duration_ms": duration_ms,
            "tool_calls": tool_calls,
            "test_content": smoke_output.get("test_content", ""),
            "timestamp": start_time.isoformat(),
        }

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"MCP smoke test FAILED: invalid output. "
            f"Error: {e}, Output: {result.stdout[:200]}"
        )


def run_benchmark(config: RunConfig) -> BenchResult:
    dataset = load_dataset(Path(config.dataset_path))
    console.print(f"Dataset: {json.dumps(dataset.summary(), indent=2)}")

    conversations = dataset.conversations
    if config.max_conversations:
        conversations = conversations[:config.max_conversations]

    profile = load_integration(config)
    seed_plan = config.seed_plan
    project_id = None
    seed_result: SeedResult | None = None
    seed_environment: dict[str, str] = {}
    workspace_tmp: tempfile.TemporaryDirectory[str] | None = None
    effective_profile_hash = None
    if profile is not None or seed_plan is not None:
        project_id = (
            f"llm-bench-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            f"-{uuid.uuid4().hex[:10]}"
        )

    if profile is not None:
        reset = profile.lifecycle.get("reset", {})
        reset_supported = isinstance(reset, dict) and reset.get("support") == "supported"

        if len(conversations) > 1 and not reset_supported and config.require_isolation:
            raise RuntimeError(
                f"Integration {profile.profile_id} cannot safely run "
                "multiple conversations: lifecycle.reset is unsupported. "
                "Set require_isolation: false in config to allow cross-conversation bleed."
            )
        validate_integration_environment(config, profile)
        workspace_tmp = tempfile.TemporaryDirectory(prefix="llm-bench-integration-")
        effective_profile_hash = materialize_integration(
            config, profile, workspace_tmp.name, project_id
        )

    if seed_plan is not None:
        assert project_id is not None
        if profile is not None:
            profile_provider = str(profile.memory_system.get("name", "")).lower()
            if profile_provider and profile_provider != seed_plan.provider:
                raise RuntimeError(
                    f"Benchmark seed provider {seed_plan.provider!r} does not match "
                    f"integration provider {profile_provider!r}"
                )
        try:
            seed_environment = resolved_system_environment(config)
            seed_result = seed_benchmark_data(
                seed_plan,
                project_id,
                seed_environment,
            )
        except SeedError as exc:
            raise RuntimeError(f"Benchmark seed failed: {exc}") from exc

    container_id: str | None = None

    try:
        # Keep image-build failures inside the seed cleanup boundary.  A
        # failed build after a successful seed must not leave benchmark data
        # behind.
        build_host_image(config)
        container_id = start_container(config, profile=profile, project_id=project_id)
        if profile is None:
            install_system(container_id)
        else:
            assert workspace_tmp is not None
            install_integration(container_id, workspace_tmp.name, profile)

        # Clean integration state before benchmark (if enabled)
        if config.cleanup_before_run and profile is not None and project_id is not None:
            cleanup_integration(container_id, profile, project_id)

        # Run smoke test before benchmark conversations
        try:
            smoke_test_result = _smoke_test_integration(container_id, profile, project_id)
        except Exception as e:
            smoke_test_result = {
                "status": "failed",
                "error": str(e)[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            console.print(f"  ⚠ Integration smoke test FAILED (continuing anyway): {str(e)[:100]}", style="yellow")
            # Continue - smoke test uses direct Claude calls which don't expand env vars
            # Benchmark uses driver.py which DOES expand vars properly

        environment = collect_environment(container_id, config)
        if profile is not None:
            environment["integration"] = {
                "profile": profile.profile_id,
                "platform": config.integration_platform or config.host_name,
                "effective_hash": effective_profile_hash,
                "project_id": project_id,
                "smoke_test": smoke_test_result,
                "memory_system": profile.memory_system.get("name") if profile.memory_system else None,
            }
            # Override system name with actual memory system name from profile
            if profile.memory_system and "name" in profile.memory_system:
                environment["system"]["name"] = profile.memory_system["name"]
        if seed_result is not None:
            environment["benchmark_seed"] = seed_result.as_dict()
        if config.experiment_group:
            environment["experiment"] = {
                "group_id": config.experiment_group,
                "metadata": config.experiment_metadata,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        result = BenchResult(config=config, environment=environment)

        for i, conversation in enumerate(conversations):
            console.print(
                f"  [{i+1}/{len(conversations)}] {conversation.id} "
                f"({len(conversation.turns)} turns)"
            )

            output = drive_conversation(
                container_id,
                conversation,
                model=config.model,
                profile_mode=profile is not None,
            )
            if profile is None:
                stored = query_stored(container_id)
                retention_calls = []
            else:
                assert project_id is not None
                stored, retention_calls = query_integration(container_id, profile, project_id)
            ground_truth = extract_ground_truth(conversation)

            # Extract hook storage events for visibility (similar to tool calls)
            hook_storage_events = []
            try:
                driver_data = json.loads(output)
                for response in driver_data.get("responses", []):
                    for event in response.get("integration_events", []):
                        if event.get("name") == "integration.hook_lifecycle":
                            details = event.get("details", {})
                            # Capture Stop hook responses which contain what was stored
                            if details.get("hook_event") == "Stop" and details.get("phase") == "hook_response":
                                hook_storage_events.append({
                                    "hook": details.get("hook_event"),
                                    "output_preview": details.get("hook_output", "")[:500] if "hook_output" in details else None,
                                    "exit_code": details.get("exit_code"),
                                })
            except (json.JSONDecodeError, KeyError):
                pass

            conv_result = ConversationResult(
                conversation_id=conversation.id,
                stored=stored,
                ground_truth=ground_truth,
                driver_log=output,
                conversation_turns=[
                    {"role": t.role, "content": t.content}
                    for t in conversation.turns
                ],
                hook_storage_events=hook_storage_events,
                retention_calls=retention_calls,
            )
            result.conversation_results.append(conv_result)

            # Show breakdown of direct vs derived for Hindsight
            direct = sum(1 for item in stored if item.get("_benchmark_metadata", {}).get("is_direct", False))
            derived = sum(1 for item in stored if item.get("_benchmark_metadata", {}).get("is_derived", False))

            if direct > 0 or derived > 0:
                console.print(
                    f"    stored {len(stored)} items ({direct} direct, {derived} derived), "
                    f"expected {len(ground_truth)} facts",
                    style="dim",
                )
            else:
                console.print(
                    f"    stored {len(stored)} items, "
                    f"expected {len(ground_truth)} facts",
                    style="dim",
                )

            if i < len(conversations) - 1:
                if profile is None:
                    cleanup_system(container_id)

        return result

    finally:
        if container_id is not None:
            stop_container(container_id, keep=config.keep_container)
        if seed_result is not None and project_id is not None:
            try:
                cleanup_benchmark_data(
                    seed_result.provider,
                    project_id,
                    seed_environment,
                    seed_result,
                )
            except SeedError as exc:
                raise RuntimeError(f"Benchmark cleanup failed: {exc}") from exc
        if workspace_tmp is not None:
            workspace_tmp.cleanup()
