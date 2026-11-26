"""Adapter for running Continual Learning Bench through the value harness.

CL-Bench is an optional Python 3.13 dependency with its own task/evaluation
runtime.  The adapter deliberately keeps it out of the main environment and
invokes a small worker in the configured CL-Bench interpreter instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal


MemorySetupMode = Literal["none", "seeded", "capture_first"]
CLBENCH_REPOSITORY = "https://github.com/pgasawa/continual-learning-bench.git"


class ClBenchUnavailableError(RuntimeError):
    """Raised when the optional CL-Bench runtime is not configured."""


def default_clbench_python(root: Path | None) -> str:
    """Return the conventional CL-Bench interpreter when it exists."""

    if root is not None:
        candidate = root / ".venv" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def validate_clbench_runtime(root: Path, python: str) -> None:
    """Validate the external checkout before starting a paid run."""

    if not root.is_dir():
        raise ClBenchUnavailableError(
            f"CL-Bench checkout not found: {root}. Run "
            "`llm-memory-bench clbench-setup` once, then rerun the benchmark."
        )
    if not (root / "src" / "interface.py").is_file():
        raise ClBenchUnavailableError(
            f"Not a CL-Bench checkout: {root} (missing src/interface.py)."
        )
    python_path = Path(python)
    if not python_path.is_file():
        raise ClBenchUnavailableError(
            f"CL-Bench Python interpreter not found: {python}."
        )


def setup_clbench(
    root: Path,
    *,
    base_python: str | None = None,
    task: str = "database_exploration",
    install_dependencies: bool = True,
) -> dict[str, str]:
    """Install the optional CL-Bench checkout and one-time task artifacts.

    This is intentionally an explicit setup command rather than a package
    install hook: the upstream checkout and database artifacts are large, and
    network access should not happen as a side effect of importing this package.
    """

    root = root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", CLBENCH_REPOSITORY, str(root)],
            check=True,
        )
    validate_clbench_runtime(root, sys.executable)

    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        interpreter = base_python or sys.executable
        subprocess.run([interpreter, "-m", "venv", str(root / ".venv")], check=True)

    if install_dependencies:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(root)],
            check=True,
        )
    subprocess.run(
        [str(venv_python), "-m", "src.cli", "setup", task],
        cwd=str(root),
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"root": str(root), "python": str(venv_python), "revision": revision}


def load_seed_memories(path: Path | None) -> list[dict[str, Any]]:
    """Load a small seed-memory file without coupling CL-Bench to our schema."""

    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"CL-Bench seed memories file not found: {path}")
    import yaml

    payload = yaml.safe_load(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("memories", [])
    if not isinstance(payload, list):
        raise ValueError("CL-Bench seed memories must be a list or {memories: [...]}")
    memories: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, str):
            memories.append({"fact": item, "category": "general"})
        elif isinstance(item, dict) and item.get("fact"):
            memories.append(
                {
                    "fact": str(item["fact"]),
                    "category": str(item.get("category", "general")),
                }
            )
    return memories


def run_clbench_value_benchmark(
    *,
    clbench_root: Path,
    clbench_python: str,
    task: str,
    schedule: str | None,
    variant: str | None,
    max_instances: int | None,
    seed: int,
    memory_setup_mode: MemorySetupMode,
    memory_system: str,
    memory_prompt_version: str | None,
    seed_memories_path: Path | None,
    model: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run one CL-Bench task through the shared capture/replay protocol."""

    validate_clbench_runtime(clbench_root, clbench_python)
    request = {
        "clbench_root": str(clbench_root),
        "task": task,
        "schedule": schedule,
        "variant": variant,
        "max_instances": max_instances,
        "seed": seed,
        "memory_setup_mode": memory_setup_mode,
        "memory_system": memory_system,
        "memory_prompt_version": memory_prompt_version,
        "seed_memories": load_seed_memories(seed_memories_path),
        "model": model,
    }
    env = os.environ.copy()
    project_src = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_src
        if not existing_pythonpath
        else project_src + os.pathsep + existing_pythonpath
    )
    command = [clbench_python, "-m", "llm_memory_bench.value.clbench_worker"]
    print(
        f"[clbench] starting task={task} mode={memory_setup_mode} "
        f"python={clbench_python}",
        file=sys.stderr,
        flush=True,
    )
    completed = subprocess.run(
        command,
        input=json.dumps(request),
        text=True,
        stdout=subprocess.PIPE,
        # Keep worker diagnostics and progress visible while the run is active.
        stderr=None,
        env=env,
        cwd=str(clbench_root),
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"CL-Bench worker failed with status {completed.returncode}: {detail}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "CL-Bench worker returned invalid JSON. "
            f"Output: {completed.stdout[-1000:]}"
        ) from exc
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
    return result

