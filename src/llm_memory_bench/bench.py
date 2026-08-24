"""Orchestrates benchmark runs: build image, start container, drive conversations, collect results."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console

from .dataset import Conversation, Dataset, load_dataset

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        host = data["host"]
        system = data["system"]
        return cls(
            host_name=host["name"],
            host_version=host["version"],
            system_name=system["name"],
            system_version=system.get("version"),
            system_env=system.get("env", {}),
            model=data["model"],
            dataset_path=data["dataset"],
            max_conversations=data.get("max_conversations"),
        )

    @property
    def image_tag(self) -> str:
        return f"llm-bench/{self.host_name}:{self.host_version}"

    def container_name(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"llm-bench-{self.host_name}-{self.system_name}-{ts}"


@dataclass
class ConversationResult:
    conversation_id: str
    stored: list[dict] = field(default_factory=list)
    ground_truth: list[dict] = field(default_factory=list)
    driver_log: str = ""


@dataclass
class BenchResult:
    config: RunConfig
    conversation_results: list[ConversationResult] = field(default_factory=list)
    environment: dict = field(default_factory=dict)


def _docker(*args, check=True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], **kwargs, check=check)


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
        "-t", config.image_tag,
        str(host_dir),
    )


def start_container(config: RunConfig) -> str:
    system_dir = PROJECT_ROOT / "systems" / config.system_name
    if not system_dir.exists():
        raise FileNotFoundError(f"No system definition at {system_dir}")

    host_dir = PROJECT_ROOT / "hosts" / config.host_name

    env_args = []
    for key in ["ANTHROPIC_API_KEY", "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION"]:
        val = os.environ.get(key)
        if val:
            env_args.extend(["-e", f"{key}={val}"])

    for key, val in config.system_env.items():
        env_args.extend(["-e", f"{key}={val}"])

    if config.system_version:
        env_args.extend([
            "-e", f"{config.system_name.upper().replace('-', '_')}_VERSION={config.system_version}",
        ])

    name = config.container_name()
    result = _docker(
        "run", "-d",
        "--name", name,
        "-v", f"{system_dir}:/system:ro",
        "-v", f"{host_dir}:/host-driver:ro",
        "-e", f"SYSTEM_INSTALL_SCRIPT=/system/install.sh",
        *env_args,
        config.image_tag,
        "sleep", "infinity",
        capture_output=True, text=True,
    )
    container_id = result.stdout.strip()
    console.print(f"Container {container_id[:12]} started.", style="dim")
    return container_id


def _exec(container_id: str, cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", container_id, *cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def install_system(container_id: str) -> None:
    console.print("Installing memory system...", style="dim")
    result = _exec(container_id, ["bash", "/system/install.sh"], timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"System install failed:\n{result.stderr}")
    console.print("System installed.", style="dim")


def drive_conversation(container_id: str, conversation: Conversation, model: str | None = None) -> str:
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

    result = _exec(container_id, cmd, timeout=600)

    if result.returncode != 0:
        console.print(f"  Driver error: {result.stderr[:300]}", style="red")

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

    return env


def stop_container(container_id: str) -> None:
    _docker("rm", "-f", container_id, capture_output=True, check=False)
    console.print("Container removed.", style="dim")


def extract_ground_truth(conversation: Conversation) -> list[dict]:
    facts = []
    for turn in conversation.turns:
        for ef in turn.ground_truth.should_store:
            facts.append({"fact": ef.fact, "type": ef.type.value})
    return facts


def run_benchmark(config: RunConfig) -> BenchResult:
    dataset = load_dataset(Path(config.dataset_path))
    console.print(f"Dataset: {json.dumps(dataset.summary(), indent=2)}")

    conversations = dataset.conversations
    if config.max_conversations:
        conversations = conversations[:config.max_conversations]

    build_host_image(config)
    container_id = start_container(config)

    try:
        install_system(container_id)
        environment = collect_environment(container_id, config)
        result = BenchResult(config=config, environment=environment)

        for i, conversation in enumerate(conversations):
            console.print(
                f"  [{i+1}/{len(conversations)}] {conversation.id} "
                f"({len(conversation.turns)} turns)"
            )

            output = drive_conversation(container_id, conversation, model=config.model)
            stored = query_stored(container_id)
            ground_truth = extract_ground_truth(conversation)

            conv_result = ConversationResult(
                conversation_id=conversation.id,
                stored=stored,
                ground_truth=ground_truth,
                driver_log=output,
            )
            result.conversation_results.append(conv_result)

            console.print(
                f"    stored {len(stored)} items, "
                f"expected {len(ground_truth)} facts",
                style="dim",
            )

            if i < len(conversations) - 1:
                cleanup_system(container_id)

        return result

    finally:
        stop_container(container_id)
