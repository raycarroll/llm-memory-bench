from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llm_memory_bench.bench import RunConfig
from llm_memory_bench.seeding import (
    MemoryHubSeedAdapter,
    SeedError,
    SeedMemory,
    SeedPlan,
    SeedRecord,
    SeedResult,
    load_seed_plan,
)


def test_load_seed_plan_is_versioned_and_hashed() -> None:
    path = Path("benchmarks/seeds/memoryhub-recall-v1.yaml")
    plan = load_seed_plan(path)

    assert plan.plan_id == "memoryhub-recall@1.0.0"
    assert plan.provider == "memoryhub"
    assert plan.mode == "declarative"
    assert len(plan.memories) == 2
    assert len(plan.fixture_hash) == 64


def test_run_config_loads_benchmark_seed_relative_to_config(tmp_path: Path) -> None:
    fixture = tmp_path / "seed.yaml"
    fixture.write_text(
        """\
api_version: llm-memory-bench.dev/v1
kind: MemorySeed
metadata:
  name: test-seed
  version: 1.0.0
provider: memoryhub
mode: declarative
memories:
  - content: seeded fact
"""
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"""\
host:
  name: claude-code
  version: '1.0.33'
benchmark:
  seed:
    fixture: {fixture.name}
model: test-model
dataset: datasets/test_sequential.yaml
"""
    )

    config = RunConfig.from_yaml(config_path)

    assert config.seed_plan is not None
    assert config.seed_plan.plan_id == "test-seed@1.0.0"


def test_seed_plan_rejects_invalid_weight(tmp_path: Path) -> None:
    path = tmp_path / "seed.yaml"
    path.write_text(
        """\
api_version: llm-memory-bench.dev/v1
kind: MemorySeed
metadata:
  name: bad-seed
  version: 1.0.0
provider: memoryhub
memories:
  - content: fact
    weight: 1.5
"""
    )

    with pytest.raises(SeedError, match="weight"):
        load_seed_plan(path)


def test_memoryhub_cleanup_removes_project_and_recorded_seed_memories() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []
            self.remaining = {"project-memory"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def list(self, **kwargs):
            assert kwargs["scope"] == "project"
            assert kwargs["project_id"] == "run-project"
            return {
                "results": [{"id": memory_id} for memory_id in self.remaining],
                "cursor": None,
            }

        async def delete(self, memory_id: str, *, project_id: str):
            self.deleted.append((memory_id, project_id))
            self.remaining.discard(memory_id)

    client = FakeClient()
    seed_result = SeedResult(
        plan_id="fixture@1.0.0",
        provider="memoryhub",
        mode="declarative",
        fixture_hash="hash",
        project_id="run-project",
        records=(SeedRecord(index=0, provider_id="user-memory", content_hash="hash"),),
    )

    asyncio.run(
        MemoryHubSeedAdapter()._cleanup_project(client, "run-project", seed_result)
    )

    assert client.deleted == [
        ("project-memory", "run-project"),
        ("user-memory", "run-project"),
    ]


def test_memoryhub_seed_cleans_partial_writes() -> None:
    class FakeMemory:
        def __init__(self, memory_id: str, content: str | None = None) -> None:
            self.id = memory_id
            self.content = content

    class FakeResult:
        def __init__(self, memory_id: str) -> None:
            self.memory = FakeMemory(memory_id)

    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []
            self.read_count = 0
            self.write_count = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def write(self, *_args, **_kwargs):
            self.write_count += 1
            return FakeResult(f"written-memory-{self.write_count}")

        async def read(self, *_args, **_kwargs):
            self.read_count += 1
            if self.read_count == 1:
                return FakeMemory("written-memory-1", "one")
            raise RuntimeError("read-back failed")

        async def delete(self, memory_id: str, *, project_id: str):
            self.deleted.append((memory_id, project_id))

    plan = SeedPlan(
        path=Path("<test>"),
        api_version="test/v1",
        seed_id="test",
        version="1.0.0",
        provider="memoryhub",
        mode="declarative",
        memories=(SeedMemory(content="one"), SeedMemory(content="two")),
        fixture_hash="hash",
    )
    client = FakeClient()

    with pytest.raises(RuntimeError, match="read-back failed"):
        asyncio.run(MemoryHubSeedAdapter()._write_and_verify(client, plan, "run-project"))

    assert client.deleted == [
        ("written-memory-2", "run-project"),
        ("written-memory-1", "run-project"),
    ]
