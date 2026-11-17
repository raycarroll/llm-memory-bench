from pathlib import Path
from types import SimpleNamespace

from llm_memory_bench.value import tau2_container


def test_tau2_container_command_mounts_plan_and_output(
    tmp_path: Path, monkeypatch
) -> None:
    plan = tmp_path / "memory-plan.yaml"
    plan.write_text("domain: airline\ntasks: []\n")
    output = tmp_path / "results" / "run.json"
    commands: list[list[str]] = []

    monkeypatch.setattr(tau2_container, "_runtime", lambda: "docker")
    monkeypatch.setattr(tau2_container, "_ensure_image", lambda runtime, image: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tau2_container.subprocess, "run", fake_run)

    assert (
        tau2_container.run_tau2_container(
            domain="airline",
            task_ids=["3"],
            memory_plan=plan,
            agent_model="openai/gpt-4.1",
            user_model="openai/gpt-4.1",
            memory_system="simple",
            seed=300,
            max_steps=20,
            output=output,
        )
        == 0
    )

    command = commands[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert f"{plan.resolve()}:/run/memory-plan.yaml:ro" in command
    assert f"{output.parent.resolve()}:/workspace/results" in command
    assert "--env" in command
    assert "OPENAI_API_KEY" in command
    assert "/run/memory-plan.yaml" in command
    assert "/workspace/results/run.json" in command
