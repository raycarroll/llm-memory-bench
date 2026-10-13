"""Benchmark-owned memory seeding.

Seed plans describe the initial memory state for a benchmark.  They are kept
separate from agent integration profiles: a profile describes how the agent
uses a memory system, while a seed plan describes the state the benchmark
expects that system to have before the first agent turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

import yaml


class SeedError(RuntimeError):
    """Raised when a benchmark seed cannot be applied or verified."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SeedError(f"{field_name} must be a mapping")
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SeedError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SeedMemory:
    content: str
    scope: str = "user"
    weight: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)
    domains: tuple[str, ...] = ()
    content_type: str | None = None
    owner_id: str | None = None
    force: bool = False

    @classmethod
    def from_dict(cls, value: Any, index: int) -> SeedMemory:
        item = _mapping(value, f"memories[{index}]")
        content = _non_empty_string(item.get("content"), f"memories[{index}].content")
        scope = _non_empty_string(item.get("scope", "user"), f"memories[{index}].scope")
        weight = item.get("weight", 0.7)
        if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
            raise SeedError(f"memories[{index}].weight must be between 0 and 1")

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SeedError(f"memories[{index}].metadata must be a mapping")

        domains = item.get("domains", [])
        if not isinstance(domains, list) or not all(
            isinstance(domain, str) and domain.strip() for domain in domains
        ):
            raise SeedError(f"memories[{index}].domains must be a list of strings")

        content_type = item.get("content_type")
        if content_type is not None:
            content_type = _non_empty_string(
                content_type, f"memories[{index}].content_type"
            )
        owner_id = item.get("owner_id")
        if owner_id is not None:
            owner_id = _non_empty_string(owner_id, f"memories[{index}].owner_id")

        force = item.get("force", False)
        if not isinstance(force, bool):
            raise SeedError(f"memories[{index}].force must be a boolean")

        return cls(
            content=content,
            scope=scope,
            weight=float(weight),
            metadata=dict(metadata),
            domains=tuple(domains),
            content_type=content_type,
            owner_id=owner_id,
            force=force,
        )


@dataclass(frozen=True)
class SeedPlan:
    path: Path
    api_version: str
    seed_id: str
    version: str
    provider: str
    mode: str
    memories: tuple[SeedMemory, ...]
    fixture_hash: str

    @property
    def plan_id(self) -> str:
        return f"{self.seed_id}@{self.version}"


def load_seed_plan(path: str | Path) -> SeedPlan:
    """Load and validate a versioned benchmark seed fixture."""

    seed_path = Path(path).resolve()
    if not seed_path.is_file():
        raise FileNotFoundError(seed_path)
    with seed_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    document = _mapping(data, "seed")
    if document.get("kind") != "MemorySeed":
        raise SeedError("kind must be MemorySeed")
    api_version = _non_empty_string(document.get("api_version"), "api_version")
    metadata = _mapping(document.get("metadata"), "metadata")
    seed_id = _non_empty_string(metadata.get("name"), "metadata.name")
    version = _non_empty_string(metadata.get("version"), "metadata.version")
    provider = _non_empty_string(document.get("provider"), "provider").lower()
    mode = _non_empty_string(document.get("mode", "declarative"), "mode").lower()
    if mode not in {"declarative", "replay"}:
        raise SeedError("mode must be declarative or replay")

    raw_memories = document.get("memories", [])
    if not isinstance(raw_memories, list):
        raise SeedError("memories must be a list")

    return SeedPlan(
        path=seed_path,
        api_version=api_version,
        seed_id=seed_id,
        version=version,
        provider=provider,
        mode=mode,
        memories=tuple(
            SeedMemory.from_dict(memory, index)
            for index, memory in enumerate(raw_memories)
        ),
        fixture_hash=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    )


@dataclass(frozen=True)
class SeedRecord:
    index: int
    provider_id: str
    content_hash: str


@dataclass(frozen=True)
class SeedResult:
    plan_id: str
    provider: str
    mode: str
    fixture_hash: str
    project_id: str
    records: tuple[SeedRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan_id,
            "provider": self.provider,
            "mode": self.mode,
            "fixture_hash": self.fixture_hash,
            "project_id": self.project_id,
            "count": len(self.records),
            "records": [
                {
                    "index": record.index,
                    "provider_id": record.provider_id,
                    "content_hash": record.content_hash,
                }
                for record in self.records
            ],
        }


class SeedAdapter(Protocol):
    provider: str

    def seed(
        self,
        plan: SeedPlan,
        project_id: str,
        environment: Mapping[str, str],
    ) -> SeedResult:
        ...

    def cleanup(
        self,
        project_id: str,
        environment: Mapping[str, str],
        seed_result: SeedResult | None = None,
    ) -> None:
        """Remove benchmark-owned data for a run namespace."""
        ...


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Expose configured provider credentials without persisting them."""

    names = {
        "MEMORYHUB_URL",
        "MEMORYHUB_API_KEY",
        "MEMORYHUB_AUTH_TOKEN",
        "MEMORYHUB_AUTH_URL",
        "MEMORYHUB_CLIENT_ID",
        "MEMORYHUB_CLIENT_SECRET",
    }
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        for name in names:
            if values.get(name):
                os.environ[name] = values[name]
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class MemoryHubSeedAdapter:
    provider = "memoryhub"

    def seed(
        self,
        plan: SeedPlan,
        project_id: str,
        environment: Mapping[str, str],
    ) -> SeedResult:
        if plan.mode != "declarative":
            raise SeedError(
                "MemoryHub replay seeding is not implemented; use a declarative "
                "fixture or add a replay adapter"
            )

        try:
            from memoryhub import MemoryHubClient
        except ImportError as exc:
            raise SeedError(
                "MemoryHub seeding requires the optional 'memoryhub' Python package"
            ) from exc

        with _temporary_environment(environment):
            try:
                client = MemoryHubClient.from_env(auto_discover_config=False)
                records = asyncio.run(self._write_and_verify(client, plan, project_id))
            except SeedError:
                raise
            except Exception as exc:  # provider exceptions are implementation-specific
                raise SeedError(f"MemoryHub seed failed: {exc}") from exc

        return SeedResult(
            plan_id=plan.plan_id,
            provider=plan.provider,
            mode=plan.mode,
            fixture_hash=plan.fixture_hash,
            project_id=project_id,
            records=tuple(records),
        )

    def cleanup(
        self,
        project_id: str,
        environment: Mapping[str, str],
        seed_result: SeedResult | None = None,
    ) -> None:
        """Soft-delete all benchmark memories created for the run."""

        try:
            from memoryhub import MemoryHubClient
        except ImportError as exc:
            raise SeedError(
                "MemoryHub cleanup requires the optional 'memoryhub' Python package"
            ) from exc

        with _temporary_environment(environment):
            try:
                client = MemoryHubClient.from_env(auto_discover_config=False)
                asyncio.run(self._cleanup_project(client, project_id, seed_result))
            except SeedError:
                raise
            except Exception as exc:  # provider exceptions are implementation-specific
                raise SeedError(f"MemoryHub cleanup failed: {exc}") from exc

    async def _cleanup_project(
        self,
        client: Any,
        project_id: str,
        seed_result: SeedResult | None = None,
    ) -> None:
        """Delete and verify all current memories in the benchmark project."""

        async def list_memory_ids() -> list[str]:
            memory_ids: list[str] = []
            cursor: str | None = None
            while True:
                page = await client.list(
                    scope="project",
                    project_id=project_id,
                    max_results=200,
                    cursor=cursor,
                    include_branches=True,
                    current_only=True,
                )
                for item in page.get("results", []):
                    memory_id = (
                        item.get("id")
                        if isinstance(item, dict)
                        else getattr(item, "id", None)
                    )
                    if memory_id:
                        memory_ids.append(str(memory_id))
                cursor = page.get("cursor")
                if not cursor:
                    break
            return memory_ids

        async with client:
            # Enumerate before deleting so cursor pagination is stable.
            memory_ids = await list_memory_ids()

            # A generic declarative fixture may intentionally contain a
            # user-scoped seed. Delete those recorded IDs explicitly; the
            # project enumeration above cannot see them.
            if seed_result is not None:
                memory_ids.extend(
                    record.provider_id
                    for record in seed_result.records
                    if record.provider_id not in memory_ids
                )

            for memory_id in dict.fromkeys(memory_ids):
                await client.delete(memory_id, project_id=project_id)

            remaining = await list_memory_ids()
            if remaining:
                raise SeedError(
                    "MemoryHub cleanup left current memories in project "
                    f"{project_id}: {', '.join(remaining[:20])}"
                )

    async def _write_and_verify(
        self,
        client: Any,
        plan: SeedPlan,
        project_id: str,
    ) -> list[SeedRecord]:
        records: list[SeedRecord] = []
        written_ids: list[str] = []
        async with client:
            try:
                for index, memory in enumerate(plan.memories):
                    result = await client.write(
                        memory.content,
                        scope=memory.scope,
                        weight=memory.weight,
                        metadata=memory.metadata or None,
                        project_id=project_id,
                        domains=list(memory.domains) or None,
                        content_type=memory.content_type,
                        owner_id=memory.owner_id,
                        force=memory.force,
                    )
                    provider_memory = getattr(result, "memory", None)
                    provider_id = getattr(provider_memory, "id", None)
                    if not provider_id:
                        curation = getattr(result, "curation", None)
                        gated = getattr(curation, "gated", None)
                        existing_id = getattr(curation, "existing_memory_id", None)
                        reason = getattr(curation, "reason", None)
                        details = []
                        if gated is not None:
                            details.append(f"gated={gated}")
                        if existing_id:
                            details.append(f"existing_memory_id={existing_id}")
                        if reason:
                            details.append(f"reason={reason}")
                        detail_text = f" ({', '.join(details)})" if details else ""
                        raise SeedError(
                            f"MemoryHub did not create seed memory {index}; "
                            "the write may have been gated as a duplicate"
                            f"{detail_text}"
                        )
                    provider_id = str(provider_id)
                    # Track the ID before read-back verification so a failed
                    # verification cannot leave a partial seed behind.
                    written_ids.append(provider_id)

                    read_back = await client.read(provider_id, project_id=project_id)
                    read_content = getattr(read_back, "content", None)
                    if read_content != memory.content:
                        raise SeedError(
                            f"MemoryHub seed verification failed for memory {index}"
                        )
                    records.append(
                        SeedRecord(
                            index=index,
                            provider_id=provider_id,
                            content_hash=hashlib.sha256(
                                memory.content.encode("utf-8")
                            ).hexdigest(),
                        )
                    )
            except Exception as exc:
                cleanup_errors: list[str] = []
                for provider_id in reversed(dict.fromkeys(written_ids)):
                    try:
                        await client.delete(provider_id, project_id=project_id)
                    except Exception as cleanup_exc:  # preserve the seed error
                        cleanup_errors.append(str(cleanup_exc))
                if cleanup_errors:
                    raise SeedError(
                        f"{exc}; partial-seed cleanup failed: "
                        + "; ".join(cleanup_errors)
                    ) from exc
                raise
        return records


_ADAPTERS: dict[str, SeedAdapter] = {
    "memoryhub": MemoryHubSeedAdapter(),
}


def seed_benchmark_data(
    plan: SeedPlan,
    project_id: str,
    environment: Mapping[str, str],
) -> SeedResult:
    """Apply and verify a benchmark seed using its provider adapter."""

    adapter = _ADAPTERS.get(plan.provider)
    if adapter is None:
        available = ", ".join(sorted(_ADAPTERS)) or "none"
        raise SeedError(
            f"No seed adapter for provider {plan.provider!r}; available: {available}"
        )
    return adapter.seed(plan, project_id, environment)


def cleanup_benchmark_data(
    provider: str,
    project_id: str,
    environment: Mapping[str, str],
    seed_result: SeedResult | None = None,
) -> None:
    """Clean provider data allocated for a benchmark run."""

    adapter = _ADAPTERS.get(provider)
    if adapter is None:
        available = ", ".join(sorted(_ADAPTERS)) or "none"
        raise SeedError(
            f"No cleanup adapter for provider {provider!r}; available: {available}"
        )
    adapter.cleanup(project_id, environment, seed_result)
