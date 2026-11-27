# OpenShift Demo Transplant Plan

## Objective

Integrate the proven OpenShift NetworkPolicy demo behavior from
`/Users/bcapper/Documents/src/memory-demo-harness` into the newer
`llm-memory-bench` application without copying the old application wholesale
or reintroducing the newer failed MCP verifier and nested completion path.

The old harness is the behavioral reference. The newer benchmark remains the
host application for providers, MemoryHub configuration and seeding, result
persistence, and UI conventions.

## Settled architecture

- The OpenShift demo is a separate experiment kind, not a value-benchmark
  scenario.
- Memory and baseline execute concurrently through a paired driver.
- Each side has its own Kubernetes MCP process, endpoint, kubeconfig,
  ServiceAccount, namespace, and namespace-scoped RBAC.
- There are no shared asyncio locks and no shared MCP client between sides.
- Cluster verification uses direct, bounded `oc` subprocesses rather than
  Kubernetes MCP calls.
- Verification is independent of the model's claimed answer.
- Agent completion and independent verification are separate statuses.
- The memory side must search MemoryHub before Kubernetes mutation.
- Application validation and Kubernetes RBAC both restrict writes to the two
  configured NetworkPolicies in the assigned namespace.
- Existing provider selection, MemoryHub setup, result persistence, and UI
  infrastructure remain in place.
- Scenario-specific names and rules belong in configuration, not generic
  runtime branches.

## Phase 0: Establish the integration baseline

Purpose: freeze the starting point and prevent accidental mixing with the
abandoned implementation.

Tasks:

- Record the current branch and commit.
- Preserve the intended untracked boundary: local `Dockerfile`, `deploy/`,
  and `docs/agent-handoff.md` are not committed unless explicitly requested.
- Remove or ignore stale compiled demo artifacts as cleanup only.
- Confirm the current provider, async execution, result serialization, UI, and
  test boundaries.
- Confirm the active branch has no dependency on the newer MCP verifier or
  nested completion path.

Gate:

- The file boundary is documented.
- Existing benchmark tests pass.
- Old and new demo runtimes cannot be imported accidentally.

Commit boundary: baseline cleanup and non-functional scaffolding only.

## Phase 1: Define the demo domain contract

Add a dedicated demo package containing validated configuration and result
models.

The contract must cover:

- demo ID, version, and fixture identity/hash;
- provider, model, memory system, and memory mode selected by the user;
- prompt and paired side definitions;
- namespace and endpoint environment variable per side;
- Kubernetes and MemoryHub tool allowlists;
- read-only and mutation tool classifications;
- writable resource allowlist;
- turn, tool-call, connection, and total-time limits;
- fixture reset configuration;
- direct verification configuration;
- per-side execution and evidence results.

Execution status must be distinct from verification status. Verification must
represent passed, failed-with-evidence, and unavailable/error states.

Likely files:

- `src/llm_memory_bench/demo/specs.py`
- `src/llm_memory_bench/demo/results.py`
- `src/llm_memory_bench/demo/config.py`
- corresponding unit tests.

Gate:

- Invalid side configuration fails before connection.
- Required fields and allowlists are validated.
- Results are JSON serializable.
- Provider/model values are not hardcoded into runtime execution.

Commit boundary: demo contracts and validation tests.

## Phase 2: Add the OpenShift scenario configuration

Adapt the old harness configuration to the benchmark's configuration
conventions.

The NetworkPolicy demo configuration defines:

- `network-policy@1.0.0`;
- `demo-memory` and `demo-baseline`;
- `K8S_MEMORY_MCP_URL` and `K8S_BASELINE_MCP_URL`;
- `memoryhub_search` only on the memory side;
- exact read-only Kubernetes tools;
- `resources_create_or_update` as the only mutation tool;
- `networkpolicy/dependency-ingress` and
  `networkpolicy/frontend-egress` as writable resources;
- the deliberately broken fixture patches;
- expected healthy policy specifications;
- selectors, ports, protocols, DNS rules, services, and log markers.

The generic runtime must not contain hardcoded demo-specific policy names.

Gate:

- The YAML loads into a complete validated demo spec.
- Broken and healthy policy states are unambiguous.
- Configuration changes alter scenario behavior without generic code changes.

Commit boundary: scenario YAML and configuration-loader tests.

## Phase 3: Recreate the deployment and cluster boundary

Align the deployment manifests with the old working implementation.

Components:

- dependency and frontend workloads;
- dependency Service;
- deliberately broken NetworkPolicies;
- both application namespaces;
- side-specific ServiceAccounts, Roles, and RoleBindings;
- side-specific MCP server deployments/configuration;
- image/build configuration.

RBAC permits reads for pods, logs, events, Services, Endpoints,
EndpointSlices, Deployments, and NetworkPolicies. It permits patching only the
two named NetworkPolicies in the side's namespace. It does not permit deletes,
arbitrary resource creation, cross-namespace writes, or cluster-wide access.

Gate:

- `oc auth can-i` confirms each side's intended permissions.
- Each side cannot patch the other namespace.
- The fixture reaches and remains in the deliberately broken state.

Commit boundary: deployment/RBAC/MCP manifests and manifest validation.

## Phase 4: Port the MCP session boundary

Implement the old harness's simple Streamable HTTP session shape inside the
demo integration.

Requirements:

- one session object per endpoint;
- `AsyncExitStack` lifecycle;
- explicit connection and read timeouts;
- no class-level or module-level asyncio lock;
- deterministic cleanup after partial startup, timeout, and exception;
- tool discovery before execution;
- endpoint resolution from the selected side configuration.

Gate:

- Both sessions initialize concurrently.
- Each discovers only its configured tools.
- No event-loop-bound objects survive between runs.

Commit boundary: MCP session adapter and lifecycle tests.

## Phase 5: Port the scenario tool runtime

Implement the old `ScenarioToolRuntime` behavior as a reusable demo runtime.

Responsibilities:

- open the side's Kubernetes MCP session;
- open and register MemoryHub for the memory side;
- expose only discovered, configured tools;
- hide mutation tools until the required MemoryHub search succeeds;
- force a search after read-only inspection and before finalization;
- enforce assigned namespace arguments;
- validate mutation manifests and exact resource allowlists;
- allow only `networking.k8s.io/v1` NetworkPolicy writes;
- apply per-tool timeouts;
- normalize MCP result text and error status.

Gate:

- Baseline has no MemoryHub tools.
- Memory-side mutation before search is blocked.
- Cross-namespace, namespace-less, malformed, and unrelated writes are
  blocked.
- A successful search unlocks only the configured mutation path.

Commit boundary: runtime policy/gating implementation and unit tests.

## Phase 6: Port the direct model/tool loop

Implement the old `ToolScenarioAgent` behavior using the newer provider
abstractions.

Responsibilities:

- construct system, developer, and user messages;
- use provider/model values from the submitted run spec;
- expose tools for the current turn;
- disable parallel tool calls where supported;
- execute returned calls sequentially;
- enforce deadline, turn, and tool-call limits;
- force required MemoryHub search;
- collect complete tool traces, token usage, timing, and errors;
- never treat text-only completion as verified completion.

Provider adapters must normalize OpenAI/Anthropic tool responses without
changing existing benchmark provider behavior.

Gate:

- Supported tool-calling providers normalize correctly.
- Calls execute sequentially.
- One side's provider failure preserves the other side's result.
- Timeouts are bounded and explicit.

Commit boundary: agent loop/provider adapter and execution tests.

## Phase 7: Port fixture reset and independent verification

Implement the old direct-`oc` environment module.

Reset behavior:

- Patch both namespaces to the deliberately broken state.
- Poll until both sides match the expected broken state.
- Return resource evidence when reset cannot be confirmed.

Verification checks:

- active pods and readiness conditions;
- Service endpoints and dependency pod target;
- actual labels and policy selectors;
- ingress allowance for the actual frontend on TCP 8080;
- egress DNS allowance exactly once;
- egress dependency TCP 8080 allowance exactly once;
- narrow/canonical final policy shapes;
- current frontend logs showing DNS resolution and HTTP success.

Verification states must distinguish passed, failed with evidence, and
unavailable due to timeout, cancellation, or cluster access.

The verifier must not call MCP tools or recursively invoke the paired runner.

Gate:

- Broken fixture produces the expected negative result.
- Corrected policies produce a complete positive result.
- Kubernetes rule ordering does not affect comparison.
- Unavailable verification is not reported as ordinary task failure.

Commit boundary: fixture reset, verifier, and deterministic verifier tests.

## Phase 8: Add paired demo orchestration

Connect the demo runtime to the newer application layer.

Run flow:

1. Load and validate the demo spec.
2. Resolve provider/model/memory values from the submitted run spec.
3. Run cluster preflight.
4. Reset both fixtures.
5. Construct two independent runtimes and agents.
6. Execute both through `asyncio.gather`.
7. Preserve one side's result if the other fails.
8. Perform final independent verification.
9. Build a structured paired result.
10. Persist normal, raw, and log outputs through `RunService`.

Likely integration points:

- `application/specs.py` for the demo run specification;
- `application/experiments.py` for `DemoExperiment`;
- `application/run_service.py` for dispatch and persistence;
- `demo/runner.py` for orchestration;
- `demo/dual_driver.py` or a shared paired-execution utility;
- `demo/results.py` for serialization.

The existing value, extraction, and τ² execution paths must remain unchanged.

Gate:

- Both sides execute concurrently.
- Final evidence is independent per side.
- No nested runner or completion-verification invocation exists.
- Paired cancellation and exceptions are bounded and explicit.

Commit boundary: application integration and orchestration tests.

## Phase 9: Integrate the existing UI

Extend the current Streamlit UI rather than copying the old Streamlit app.

Expose and persist:

- demo selection;
- provider;
- model;
- memory system;
- memory mode;
- run limits;
- any existing preflight/reset controls that fit the UI model.

Render per-side:

- turn and tool-call progress;
- tool arguments and results;
- MemoryHub searches;
- blocked calls;
- agent status;
- verification status;
- unavailable verification;
- raw evidence;
- final policy, readiness, endpoint, and log summaries.

The UI must never display PASS for agent completion alone, cancelled
verification, unavailable verification, or partial evidence.

Gate:

- Sidebar choices appear in the persisted run spec.
- Reruns clear stale output.
- Partial side failure remains visible.
- The UI remains responsive during the bounded async run.

Commit boundary: UI integration and rendering tests.

## Phase 10: Add tests before cluster execution

Add tests for:

- demo specification and configuration validation;
- endpoint resolution and MCP result normalization;
- namespace and mutation enforcement;
- MemoryHub gating and tool visibility;
- sequential tool handling and provider normalization;
- fixture broken-state and healthy-state verification;
- missing endpoints, selectors, rules, logs, and command timeouts;
- paired concurrency and one-sided failure preservation;
- independent final verification;
- result serialization and UI status distinctions.

Run the complete existing test suite together with the new demo tests.

Commit boundary: complete test coverage for the integrated implementation.

## Phase 11: Local image and manifest validation

Before using the cluster:

- run formatting, static checks, and all tests;
- build locally with Podman on macOS;
- build for the cluster's target architecture;
- push to `quay.io/rh_et_wd/llm-bench`;
- validate manifests statically;
- confirm environment-variable and secret wiring.

No cluster run begins until the local image and manifests pass validation.

Commit boundary: only required deployment/image changes; never local secrets.

## Phase 12: Controlled cluster validation

Run in this order:

1. Cluster preflight.
2. Deploy/update the image.
3. Deploy side MCP servers.
4. Verify ServiceAccounts and RBAC.
5. Deploy/reset the fixture.
6. Independently confirm the broken state.
7. Run one paired demo.
8. Inspect both complete event histories.
9. Inspect direct verifier evidence.
10. Confirm both final states independently.
11. Repeat only after the first result is understood.

Acceptance criteria:

- Distinct endpoints and identities are visible for both sides.
- The memory side searches MemoryHub before mutation.
- Each side can repair only its own namespace.
- No shared-lock stall or event-loop binding error occurs.
- No nested verifier cancellation occurs.
- Both sides finish or fail independently.
- Direct verification agrees with actual cluster state.
- Successful repair produces `passed=true` with complete evidence.

## Phase 13: Cleanup and logical commits

After all gates pass:

- remove obsolete active-branch demo verifier/runtime files;
- remove stale imports and dead completion paths;
- remove compiled artifacts from the working tree;
- preserve unrelated benchmark functionality;
- exclude local docs/instructions unless explicitly requested.

Expected logical commit sequence:

1. Demo contracts and configuration.
2. Deployment, RBAC, and MCP manifests.
3. MCP session and scenario runtime.
4. Agent loop and paired orchestration.
5. Direct fixture reset and verification.
6. Application/UI integration.
7. Tests and final cleanup.

Each commit must pass the checks relevant to its phase before the next phase
begins.
