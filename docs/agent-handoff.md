# LLM Memory Bench: OpenShift Demo Handoff

## Objective

This repository is being extended with an OpenShift operational Demo mode. It
runs the same incident prompt against two isolated namespaces:

- `demo-memory`: Kubernetes tools plus MemoryHub access.
- `demo-baseline`: Kubernetes tools without MemoryHub access.

The comparison tests whether operational memory helps diagnose and repair a
deliberately broken NetworkPolicy scenario.

## Deployment

Deployment-specific routes, API endpoints, image tags, and OpenShift context
names must be supplied by the operator. Do not commit them to this handoff
document. In a deployed pod, the verifier uses the application's ServiceAccount
credentials; local execution can provide `KUBECONFIG` and `OPENSHIFT_CONTEXT`.

## Scenario

Prompt:

> Investigate the network policy incident, use relevant operational memory,
> make the required change, and verify the resulting connectivity policy.

Namespaces are `demo-memory` and `demo-baseline`. The fixture intentionally
contains broken policies. A successful repair must establish:

- dependency ingress from the frontend on TCP `8080`;
- frontend egress to the dependency on TCP `8080`;
- frontend DNS egress through OpenShift DNS on UDP/TCP `5353`;
- Ready frontend and dependency pods in both namespaces.

The original harness is the behavioral reference for strict verification. Its
checks cover pod readiness/restarts, service endpoints, exact policy rules, DNS
egress, dependency connectivity, frontend logs, and final functional state. Do
not replace this with generic “evidence collected” output.

## Important implementation files

- `src/llm_memory_bench/ui/app.py`: Demo sidebar, live panels, Markdown,
  prompt/event rendering, reset button, and post-run verification.
- `src/llm_memory_bench/application/experiments.py`: demo experiment integration.
- `src/llm_memory_bench/demo/driver.py`: preparation, paired execution, and result assembly.
- `src/llm_memory_bench/demo/agent.py`: provider-neutral agent loop and tool events.
- `src/llm_memory_bench/demo/verification.py`: direct, independent fixture reset and verification.
- `src/llm_memory_bench/demo/runtime.py`: per-side MCP discovery and safety
  rules, including the required memory-search gate.
- `src/llm_memory_bench/demo/mcp.py`: Streamable-HTTP MCP session adapter.

The focused tests previously passed (`7 passed`), but re-run them against the
current worktree before committing.

## Historical failure (superseded)

An earlier implementation reported:

`Demo side failed: CancelledError: Cancelled via cancel scope ...`

The historical pod log showed the concrete traceback in post-run verification:

`DemoService.verify_sync -> verify -> StreamableHTTPSession.call_tool(resources_list) -> MCP ClientSession.send_request -> anyio memory stream -> asyncio.exceptions.CancelledError`

The traceback is from `ui/app.py` around line 956. It proves that the
verification request loses its Kubernetes MCP response stream. It does not by
itself prove that both agent loops failed.

The previous UI caught `Exception` around verification, but
`asyncio.CancelledError` inherits from `BaseException`, so it escaped and
killed the Streamlit script thread. This is retained as historical context;
the current implementation uses direct, bounded `oc` verification and an
explicit OpenShift context.

The UI must distinguish:

1. verified pass;
2. verified failure with concrete pod/policy evidence;
3. verification unavailable/cancelled/error.

Never report a pass when verification did not complete.

## Investigation commands

Run these from a machine with cluster access:

```bash
oc logs deployment/llm-memory-bench -n llm-memory-bench --tail=300
oc get deployment,pods -n llm-memory-bench -o wide
oc rollout status deployment/llm-memory-bench -n llm-memory-bench --timeout=180s
oc get pods,networkpolicy,service,endpoints -n demo-memory
oc get pods,networkpolicy,service,endpoints -n demo-baseline
```

Also inspect the Kubernetes MCP deployment/logs using its actual deployed name.
Compare its Streamable-HTTP lifecycle with the old harness. Preserve the
existing transport-neutral runtime boundary; do not invent a second Kubernetes
client architecture unless MCP cannot provide a required operation.

Add focused tests for a verification MCP call raising
`asyncio.CancelledError`; the result must be explicit and non-fatal.

## Separate MemoryHub issue

Earlier runs reported:

`Not authorized for this project.`

This is separate from the OpenAI provider key. It indicates that the deployed
MemoryHub client is not authorized for project scope `demo-network-policy`.
The `bcapper` client was reactivated and project configuration was updated, but
real MemoryHub search must still be verified from the deployed bench. Never
expose or commit keys.

The memory side uses actual MemoryHub tool delivery for the required search
gate. Prompt recall may also be enabled with `memory_mode="both"`, but it is
not a substitute for the required tool call.

## Build, rollout, and Git rules

OpenShift builds are slow; the user prefers local image builds in future. Do
not mix deployment changes into implementation-only commits unless requested.

After a confirmed image build:

```bash
oc rollout restart deployment/llm-memory-bench -n llm-memory-bench
oc rollout status deployment/llm-memory-bench -n llm-memory-bench --timeout=180s
```

Do not claim a live change until rollout and pod health are verified. Inspect
`git status` first and preserve unrelated work. The user prefers focused
commits and does not want `Validation:` sections in commit messages. Never
commit credentials, tokens, or deployment secrets.

## Definition of done

- A cancelled verification request cannot crash the Streamlit script thread.
- The UI distinguishes unavailable, failed, and passed verification.
- Both live panels retain event history and per-entity metrics.
- The underlying MCP cancellation cause is identified or instrumented.
- Focused tests pass before rebuilding and rolling out the image.

## Successor update: 2026-11-20

The project is an interactive Streamlit benchmark/demo. Its central experiment
is to run two otherwise equivalent operational agents against the same broken
OpenShift NetworkPolicy incident: the `demo-memory` side has MemoryHub access
and the `demo-baseline` side does not. The agent should diagnose the incident,
make the permitted NetworkPolicy changes through Kubernetes MCP, and prove the
functional result. Agent completion alone is not success; independent cluster
verification is authoritative.

### Current implementation boundary

The demo implementation is under `src/llm_memory_bench/demo/` and is integrated
through `application/experiments.py` and `application/run_service.py`. Preserve
unrelated worktree changes when switching branches or applying deployment fixes.

### Changes made recently

- The live demo panels were moved outside the submit branch so they are
  visible before a run and retain their fixed-height, scrollable containers.
- Each user/agent/tool/verification event is rendered in its own bordered card.
- Agent messages use native Streamlit Markdown rather than raw HTML/Markdown.
- Tool results that contain JSON are rendered vertically with `st.json`; full
  output is behind a collapsed expander. Long text/log output is shortened,
  duplicate lines are removed, and the full response remains available.
- Token metadata no longer uses the distracting number emoji.
- Verification cards now show a readable passed/not-passed status and named
  checks (frontend readiness, dependency readiness, endpoint, ingress, egress,
  DNS, and HTTP check), with raw evidence in an expander.
- `render_live_event` catches Streamlit's internal `StopException` so a browser
  rerun does not recursively crash the shared paired run. Both sides are still
  intended to stop together when the shared run is cancelled.
- The direct verifier uses structured Pod and Endpoints responses from `oc`,
  while the agent continues to use the configured Kubernetes MCP tools.

The latest UI edit is in `src/llm_memory_bench/ui/app.py`. It compiles with:

```bash
python -m compileall -q src/llm_memory_bench/ui/app.py
```

### Cluster/RBAC state

The current intended reset state is: dependency pods ready, frontend pods
intentionally broken/not ready, and only the canonical
`dependency-ingress`/`frontend-egress` policies present. The MCP service
account is:

```text
system:serviceaccount:kubernetes-mcp-server:kubernetes-mcp-server
```

RBAC was expanded to allow `patch`, `update`, and `create` for NetworkPolicies
in both demo namespaces. Deployment write permissions were deliberately not
added; the old harness only allowed agents to modify the two canonical
NetworkPolicies. Check permissions rather than assuming a model's 403 is a
provider problem:

```bash
oc auth can-i patch networkpolicies -n demo-memory \
  --as=system:serviceaccount:kubernetes-mcp-server:kubernetes-mcp-server
oc auth can-i update networkpolicies -n demo-memory \
  --as=system:serviceaccount:kubernetes-mcp-server:kubernetes-mcp-server
oc auth can-i create networkpolicies -n demo-baseline \
  --as=system:serviceaccount:kubernetes-mcp-server:kubernetes-mcp-server
```

The agents should have read access to Pods, Services, Endpoints, policies, and
logs, and write access only to the intended policies. Do not add broad guards
or Deployment permissions merely to make the model succeed: the experiment is
specifically testing whether memory helps the agent repair the incident using
the constrained operational interface.

### Build and rollout workflow

The OpenShift project is `llm-memory-bench`; the deployed image stream is also
`llm-memory-bench`. First verify API/DNS availability, then inspect the build:

```bash
oc whoami
oc get buildconfig,build,deployment -n llm-memory-bench -o wide
oc get istag/llm-memory-bench:latest -n llm-memory-bench
```

The last known verifier image was build 38, but its rollout was not confirmed.
The most recent requested UI build could not start because the API hostname
failed DNS resolution:

```text
lookup <cluster-api-host>: no such host

## Handoff update: 2026-11-20

The active branch is `openshift-demos`, based on `8f3e272`. The old working
harness remains the
behavioral source of truth. The newer MCP-based verifier and nested
completion-verification path are not part of this branch and must not be
reintroduced.

### Completed commits

- `fd6b5c1` — paired demo contracts
- `d52e5ef` — NetworkPolicy scenario configuration and loader
- `36f9ee7` — side-specific MCP/RBAC deployment manifests
- `1ddd4ce` — isolated Streamable HTTP MCP sessions
- `a137212` — scenario tool runtime guards
- `0e787cc` — paired provider/tool agent loop
- `613ce4d` — independent direct-`oc` verification
- `8036099` — paired orchestration/application integration
- `3813fb5` — OpenShift demo UI
- `8bf2397` — Dockerfile build-context correction

Deployment-specific notes and credentials must remain outside this document:

- `docs/agent-handoff.md`
- `docs/openshift-demo-transplant-plan.md`

### Current implementation locations

- `src/llm_memory_bench/demo/specs.py` — typed paired demo contracts
- `src/llm_memory_bench/demo/config.py` — YAML loader
- `src/llm_memory_bench/demo/mcp.py` — one MCP session per side endpoint
- `src/llm_memory_bench/demo/runtime.py` — tool discovery, namespace and write guards
- `src/llm_memory_bench/demo/agent.py` — bounded sequential provider/tool loop
- `src/llm_memory_bench/demo/verification.py` — direct `oc` reset and verification
- `src/llm_memory_bench/demo/driver.py` — paired concurrent orchestration
- `scenarios/openshift/network-policy.yaml` — scenario source of truth
- `deploy/kubernetes-mcp-sides.yaml` — side MCP deployments and RBAC
- `deploy/openshift-demo.yaml` — benchmark deployment and endpoint environment

The reference behavior preserved here is: separate side MCP servers and
identities, no shared asyncio lock, sequential tool calls, MemoryHub search
before memory-side mutation, exact NetworkPolicy write allowlists, direct
`oc` verification, and independent per-side outcomes.

### Local validation and image

Podman was started on macOS and the image was built for `linux/amd64`:

`quay.io/rh_et_wd/llm-bench:local`

The local Streamlit health endpoint returned `ok`, and the image was pushed to
Quay successfully. The Dockerfile was committed in `8bf2397` after removing
the obsolete `COPY demos ./demos` line; this branch has no `demos/` directory.

A local environment contains the test dependencies. Current validation is:

```text
18 focused demo/provider tests passed; 95 repository tests passed when the
network-dependent integration-profile module was excluded.
```

The remaining integration-profile test requires local socket/network access;
run it in an environment where that capability is available.

### Current blocker

The user reports the cluster is up, but this agent environment still cannot
resolve the configured API hostname:

`<cluster-api-host>`

Both `oc` and `curl` fail before authentication with:

```text
lookup <cluster-api-host>: no such host
```

Do not infer that the cluster is down. Treat this as a local DNS/network-path
problem. Do not apply manifests until these succeed:

```bash
oc whoami
oc whoami --show-server
```

### Next agent actions

1. Retry local DNS/API access and run `oc whoami`.
2. Inspect existing namespaces, demo pods, MCP pods, Services, Routes, and
   NetworkPolicies before applying anything.
3. Verify each side's RBAC with `oc auth can-i`; only the two named policies
   in that side's namespace may be patched.
4. Confirm the fixture is deliberately broken before the first paired run.
5. Deploy/update the already-pushed image and MCP manifests only after the
   preflight is successful.
6. Run one paired demo and inspect both complete histories.
7. Use the direct verifier as authoritative evidence; agent completion alone
   is not success.

Keep verification states distinct: passed, failed with evidence, and
unavailable due to cancellation, timeout, or cluster access.
```

Once the cluster is reachable, use the repository's configured BuildConfig
(inspect it before starting a build), wait for completion, then:

```bash
oc start-build llm-memory-bench -n llm-memory-bench --from-dir=. --follow
oc rollout restart deployment/llm-memory-bench -n llm-memory-bench
oc rollout status deployment/llm-memory-bench -n llm-memory-bench --timeout=180s
oc get pods -n llm-memory-bench -o wide
oc logs deployment/llm-memory-bench -n llm-memory-bench --tail=300
```

If the demo cluster has just been restarted, wait for CNI and the registry
before diagnosing image pulls. Previously `ovnkube-node` was `8/8 Running` and
the image registry was Available/Not Degraded, while stale pods remained in
`ImagePullBackOff`. Check events and recreate only affected stale pods after
the infrastructure is healthy:

```bash
oc get co network image-registry
oc get pods -A | rg 'ImagePull|CrashLoop|Pending'
oc describe pod <pod> -n <namespace>
oc get events -A --sort-by=.lastTimestamp | tail -80
```

Do not claim a build or rollout succeeded from a completed local command alone;
confirm the Build object, deployment revision, ready pod, and application log.

### Verification expectations

The verifier must report actual Pod names and phases. A result containing
`frontend_pod=No pod` while `oc get pods` shows a frontend is a verifier/data
shape bug, not evidence that the workload is absent. Use the old implementation
as the behavioural reference:

```text
the original behavioral reference harness
```

The strict pass requires exact repaired policy rules plus outcome evidence:
ready frontend and dependency pods, a Service endpoint for dependency, DNS
resolution, and a successful frontend dependency HTTP check. A policy-only
pass is insufficient. A verifier transport cancellation or MCP error must be
shown as “verification unavailable”, never converted to a failed empty object
or a false pass.

Useful live checks:

```bash
oc get pods,networkpolicy,service,endpoints -n demo-memory -o wide
oc get pods,networkpolicy,service,endpoints -n demo-baseline -o wide
oc logs deployment/llm-memory-bench -n llm-memory-bench --tail=300
```

### Provider note

The code supports `openai`, `anthropic`, `litellm`, and `vertex`. The direct
OpenAI provider currently reads `OPENAI_API_KEY` (or the configured
`api_key_env`) but does not expose a custom `base_url`. Routing through EnMaaS
is possible later if its OpenAI-compatible endpoint is used, most likely via
LiteLLM or by adding explicit base URL configuration. Never put an API key in
this document, source, manifests, logs, or commit messages. Keys previously
pasted into chat should be rotated.

### Practical debugging order

1. Check `git status` and preserve unrelated changes.
2. Check `oc whoami` and API DNS before changing application code.
3. Check cluster operators/CNI/registry before interpreting ImagePullBackOff.
4. Inspect application and MCP logs for the exact failing side and turn.
5. Compare verifier output with direct `oc` output; especially Pod and
   Endpoints data shapes.
6. Run focused tests and `compileall`.
7. Build, restart, wait for rollout, and verify the live pod before reporting
   completion.
