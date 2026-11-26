"""Run the τ² Value adapter inside a self-contained container image."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TAU2_REF = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"
# Include the harness revision in the default tag. The runner only builds an
# image when its tag is absent; keeping only the τ² revision would allow an
# image containing stale harness code to survive source changes.
HARNESS_IMAGE_REV = "tau2-live-memoryhub-v3-simple-ben-prompt"
DEFAULT_IMAGE = f"llm-memory-bench-tau2:{TAU2_REF[:12]}-{HARNESS_IMAGE_REV}"
CONTAINER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHERAI_API_KEY",
    "MEMORYHUB_URL",
    "MEMORYHUB_API_KEY",
    "MEMORYHUB_AUTH_TOKEN",
    "MEMORYHUB_AUTH_URL",
    "MEMORYHUB_CLIENT_ID",
    "MEMORYHUB_CLIENT_SECRET",
)


def _runtime() -> str:
    configured = os.environ.get("LLM_MEMORY_BENCH_CONTAINER_RUNTIME", "").strip()
    if configured:
        if configured not in {"docker", "podman"}:
            raise RuntimeError(
                "LLM_MEMORY_BENCH_CONTAINER_RUNTIME must be 'docker' or 'podman'"
            )
        return configured

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
        "`podman machine start`."
    )


def _ensure_image(runtime: str, image: str) -> None:
    inspected = subprocess.run(
        [runtime, "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspected.returncode == 0:
        return

    subprocess.run(
        [
            runtime,
            "build",
            "--build-arg",
            f"TAU2_REF={TAU2_REF}",
            "-t",
            image,
            "-f",
            str(PROJECT_ROOT / "hosts" / "tau2" / "Dockerfile"),
            str(PROJECT_ROOT),
        ],
        check=True,
    )


def run_tau2_container(
    *,
    domain: str,
    task_ids: list[str],
    memory_plan: Path,
    agent_model: str,
    user_model: str,
    memory_system: str,
    seed: int,
    max_steps: int,
    memory_setup_mode: str = "capture_first",
    memoryhub_live: bool = False,
    memoryhub_cleanup: bool = True,
    output: Path,
    image: str = DEFAULT_IMAGE,
    build: bool = True,
) -> int:
    """Build if needed and run the τ² image, returning its exit code."""

    memory_plan = memory_plan.resolve()
    output = output.resolve()
    if not memory_plan.is_file():
        raise FileNotFoundError(f"Memory plan does not exist: {memory_plan}")
    output.parent.mkdir(parents=True, exist_ok=True)

    runtime = _runtime()
    if build:
        _ensure_image(runtime, image)

    command = [
        runtime,
        "run",
        "--rm",
        "--volume",
        f"{memory_plan}:/run/memory-plan.yaml:ro",
        "--volume",
        f"{output.parent}:/workspace/results",
    ]
    for key in CONTAINER_ENV_KEYS:
        if os.environ.get(key):
            command.extend(["--env", key])

    command.extend(
        [
            image,
            "--domain",
            domain,
            "--task-ids",
            ",".join(task_ids),
            "--memory-plan",
            "/run/memory-plan.yaml",
            "--agent-model",
            agent_model,
            "--user-model",
            user_model,
            "--memory-system",
            memory_system,
            "--memory-setup-mode",
            memory_setup_mode,
            "--seed",
            str(seed),
            "--max-steps",
            str(max_steps),
            "--memoryhub-live" if memoryhub_live else "--no-memoryhub-live",
            "--memoryhub-cleanup" if memoryhub_cleanup else "--no-memoryhub-cleanup",
            "--output",
            f"/workspace/results/{output.name}",
        ]
    )
    completed = subprocess.run(command, check=False)
    return completed.returncode
