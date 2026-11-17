"""Live MemoryHub adapter used by the τ² benchmark bridge.

The normal benchmark memory systems are deterministic harness adapters.  This
module is deliberately separate: when enabled it sends the same canonical
``memory(action=...)`` calls to a configured MemoryHub instance.
"""

from __future__ import annotations

import asyncio
import threading
from queue import Queue
from typing import Any, Callable, Mapping

from ..providers.base import ToolCall
from ..systems.memoryhub import MemoryHubMemorySystem


class MemoryHubLiveError(RuntimeError):
    """Raised when live MemoryHub access cannot be configured or completed."""


def _run_async(coroutine: Any) -> Any:
    """Run a coroutine from sync τ² tool dispatch, including inside an event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def worker() -> None:
        try:
            result.put((True, asyncio.run(coroutine)))
        except BaseException as exc:  # propagate provider failures to the caller
            result.put((False, exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    succeeded, value = result.get()
    if not succeeded:
        raise value
    return value


def _jsonable(value: Any) -> Any:
    """Convert MemoryHub SDK models into the JSON shape stored in results."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _jsonable(method())
            except TypeError:
                continue
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _provider_id(value: Any) -> str | None:
    payload = _jsonable(value)
    if not isinstance(payload, Mapping):
        return None
    memory = payload.get("memory")
    if isinstance(memory, Mapping) and memory.get("id"):
        return str(memory["id"])
    if payload.get("id"):
        return str(payload["id"])
    curation = payload.get("curation")
    if isinstance(curation, Mapping) and curation.get("existing_memory_id"):
        return str(curation["existing_memory_id"])
    return None


class MemoryHubLiveAdapter(MemoryHubMemorySystem):
    """Execute the MemoryHub prompt/schema against a real provider instance."""

    def __init__(
        self,
        *,
        project_id: str,
        prompt_version: str,
        client_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(prompt_version=prompt_version, **kwargs)
        self.project_id = project_id
        self._client_factory = client_factory
        self.seeded_memories: list[dict[str, Any]] = []
        self._written_ids: list[str] = []

    @staticmethod
    def ensure_available() -> None:
        try:
            from memoryhub import MemoryHubClient  # noqa: F401
        except ImportError as exc:
            raise MemoryHubLiveError(
                "Live MemoryHub mode requires the optional 'memoryhub' Python "
                "package in the environment running τ²."
            ) from exc

    def _new_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        self.ensure_available()
        from memoryhub import MemoryHubClient

        try:
            return MemoryHubClient.from_env(auto_discover_config=False)
        except Exception as exc:
            raise MemoryHubLiveError(
                "Could not configure MemoryHubClient from MEMORYHUB_* environment "
                f"variables: {exc}"
            ) from exc

    @staticmethod
    async def _call_method(client: Any, action: str, arguments: dict[str, Any]) -> Any:
        options = arguments.get("options")
        options = options if isinstance(options, dict) else {}
        scope = arguments.get("scope", "user")
        content = arguments.get("content")
        memory_id = arguments.get("memory_id")
        project_id = arguments.get("project_id")

        if action == "write":
            return await client.write(
                content,
                scope=scope,
                weight=options.get("weight", 0.7),
                metadata=options.get("metadata"),
                project_id=project_id,
                domains=options.get("domains"),
                content_type=options.get("content_type"),
                owner_id=options.get("owner_id"),
                force=options.get("force", False),
            )
        if action == "list":
            return await client.list(
                scope=scope,
                project_id=project_id,
                max_results=options.get("max_results", 100),
                cursor=options.get("cursor"),
                include_branches=options.get("include_branches", True),
                current_only=options.get("current_only", True),
            )
        if action == "read":
            return await client.read(memory_id, project_id=project_id)
        if action == "delete":
            return await client.delete(memory_id, project_id=project_id)

        method = getattr(client, action, None)
        if method is None:
            raise MemoryHubLiveError(
                f"The configured MemoryHub client does not support action {action!r}."
            )
        kwargs = {
            "scope": scope,
            "project_id": project_id,
            "max_results": options.get("max_results", 100),
        }
        if action == "search":
            return await method(arguments.get("query", ""), **kwargs)
        if action == "update":
            return await method(memory_id, content=content, **kwargs)
        return await method(**kwargs)

    async def _execute_async(self, tool_call: ToolCall) -> dict[str, Any]:
        action = tool_call.arguments.get("action", "")
        arguments = dict(tool_call.arguments)
        arguments["project_id"] = self.project_id
        client = self._new_client()
        try:
            async with client:
                result = await self._call_method(client, action, arguments)
        except MemoryHubLiveError:
            raise
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        payload = _jsonable(result)
        provider_id = _provider_id(payload)
        if action in {"write", "update"} and provider_id:
            self._written_ids.append(provider_id)
        if isinstance(payload, dict):
            return payload
        return {"status": "ok", "result": payload}

    def execute_tool_call(self, tool_call: ToolCall, mode: str) -> dict[str, Any]:
        error = self.validate_tool_call_for_mode(tool_call, mode)
        if error:
            return {"status": "error", "error": error}
        return _run_async(self._execute_async(tool_call))

    def seed_memories(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Write the baseline snapshot into this replay project's scope."""

        seeded: list[dict[str, Any]] = []
        for item in memories:
            fact = str(item.get("fact", "")).strip()
            if not fact:
                continue
            call = ToolCall(
                name="memory",
                arguments={"action": "write", "content": fact, "scope": "user"},
            )
            response = self.execute_tool_call(call, "replay")
            if response.get("status") == "error":
                raise MemoryHubLiveError(
                    f"Could not seed replay memory: {response.get('error', 'unknown error')}"
                )
            record = dict(item)
            provider_id = _provider_id(response)
            if provider_id:
                record["provider_id"] = provider_id
            seeded.append(record)
        self.seeded_memories = seeded
        return seeded

    async def _cleanup_async(self) -> None:
        client = self._new_client()
        async with client:
            cursor: str | None = None
            ids: list[str] = []
            while True:
                page = await client.list(
                    scope="project",
                    project_id=self.project_id,
                    max_results=200,
                    cursor=cursor,
                    include_branches=True,
                    current_only=True,
                )
                payload = _jsonable(page)
                for item in (payload.get("results", []) if isinstance(payload, dict) else []):
                    if isinstance(item, dict) and item.get("id"):
                        ids.append(str(item["id"]))
                cursor = payload.get("cursor") if isinstance(payload, dict) else None
                if not cursor:
                    break
            # User-scoped writes can still be associated with this benchmark
            # project. They are not necessarily returned by project listing,
            # so retain their provider IDs explicitly.
            ids.extend(self._written_ids)
            for memory_id in dict.fromkeys(ids):
                await client.delete(memory_id, project_id=self.project_id)

    def cleanup(self) -> None:
        """Delete current memories in this adapter's benchmark project."""

        _run_async(self._cleanup_async())

