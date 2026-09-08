# MemoryHub Prompts & Instructions - Complete Map

Comprehensive mapping of ALL system instructions, prompts, and guidance related to memory storage in MemoryHub.

## Overview

MemoryHub has **THREE layers** of instructions:

1. **Agent Instructions** (Client-Side) - Tell Claude *when* and *how* to store memories
2. **Tool Descriptions** (MCP Schema) - Embedded in tool definitions, describe parameters
3. **Extraction Prompts** (Server-Side) - Tell MemoryHub's services how to process stored data

---

## 1. Agent Instructions (Client-Side)

### 1.1 Template Source
**Location**: `memoryhub-cli/src/memoryhub_cli/project_config.py`

**Purpose**: Templates that generate agent instructions for different use cases

**Key Components**:

#### Universal Header
```python
_UNIVERSAL_HEADER = """
# MemoryHub Agent Instructions: {pattern_title}

This project uses MemoryHub for persistent, centralized agent memory across
conversations. You MUST use it.
"""
```

#### Memory Hygiene Block
```python
_HYGIENE_BLOCK = """
## Memory hygiene

- Keep memories concise and self-contained. Another agent should
  understand them without re-loading the conversation that produced them.
- DO write preferences, decisions, architectural choices, tool
  configuration, and workflow patterns. Skip ephemeral things like "user
  asked me to read a file."
- Use `update_memory` (not `write_memory`) to revise an existing entry —
  this preserves version history. Use `write_memory` only for new facts.
- Set weights deliberately: `1.0` for critical policies, `0.8-0.9` for
  strong preferences, `0.5-0.7` for nice-to-know context.
- Add rationale branches via `parent_id` + `branch_type="rationale"` when
  the "why" behind a preference is load-bearing.
- Use the right scope: `user` for personal preferences, `project` for
  project-specific context, `organizational` for team/org patterns,
  `enterprise` for mandated policies.
"""
```

**This is the PRIMARY storage policy guidance for agents.**

#### Loading Pattern Blocks (4 variants)

**Lazy Pattern** (most common):
```python
"lazy": """
## At session start

Authenticate with `register_session(api_key=<your_key>)`, then wait until
the user states a task or question with enough detail to form a meaningful
search query. Derive a 1-2 sentence summary and call
`search_memory(query=<summary>)`. If the opening turn is low-signal
("hi", "can you help me?"), do not search yet -- wait for a follow-up
that reveals the topic.

## During the session

- Trust your working set. Re-search only when the user explicitly
  references a concept you don't have loaded.
"""
```

**Eager Pattern**:
- Load full working set at session start
- Call `search_memory(query="", mode="index", max_results=50)` immediately

**Lazy + Rebias on Pivot**:
- Like lazy but re-search when topic changes (subsystem change, unknown concept, explicit switch)
- ADD results to working set, don't replace

**Just-in-Time**:
- No working set
- Search only when needed for specific questions
- Let results drop from context after use

#### Contradiction Handling
```python
_CONTRADICTION_ENABLED = """
## Contradiction handling

When you notice the user's behavior contradicting a memory you have
loaded, call `report_contradiction` with the memory_id and a one-sentence
description of the observed behavior. The server tracks contradiction
counts and surfaces stale memories for review.
"""
```

### 1.2 Generated Output Files
**Location**: `.claude/rules/memoryhub-loading.md` (in each project)

**Created by**: `memoryhub config init`

**Format Options**: 
- `claude-code` → `.claude/rules/memoryhub-loading.md`
- `system-prompt` → Stdout (paste into system prompt)
- `agents-md` → `AGENTS.md`
- `ogx` → OGX config
- `raw` → Framework-agnostic

**Example Generated File**: See `.claude/rules/memoryhub-loading.md` in memory-hub repo

---

## 2. Tool Descriptions (MCP Schema)

These are embedded in the tool definitions and visible to the agent when it calls tools.

### 2.1 write_memory Tool
**Location**: `memory-hub-mcp/src/tools/write_memory.py`

**Tool Description**:
```python
@mcp.tool()
async def write_memory(
    content: Annotated[
        str,
        Field(description="The memory text. Should be clear and self-contained."),
    ],
    scope: Annotated[
        str,
        Field(
            default="user",
            description=(
                "One of: user, project, campaign, role, organizational, enterprise. "
                "Defaults to 'user' -- most agent-created memories are user-scoped. "
            ),
        ),
    ] = "user",
    weight: Annotated[
        float,
        Field(
            description=(
                "Injection priority from 0.0 to 1.0. High-weight (0.8-1.0) "
                "memories get full content injected. Default 0.7."
            ),
        ),
    ] = 0.7,
    parent_id: Annotated[
        str | None,
        Field(
            description=(
                "UUID of the parent memory node when creating a branch. "
                "Omit for root-level memories."
            ),
        ),
    ] = None,
    branch_type: Annotated[
        str | None,
        Field(
            description=(
                "Required when parent_id is set. Common types: rationale, "
                "provenance, description, evidence, approval."
            ),
        ),
    ] = None,
    domains: Annotated[
        list[str] | None,
        Field(
            description=(
                "Knowledge domain tags (e.g., ['authentication', 'React']). "
                "Used for domain-scoped retrieval and contradiction detection."
            ),
        ),
    ] = None,
    content_type: Annotated[
        str | None,
        Field(
            description=(
                "One of: experiential, declarative, behavioral. "
                "Behavioral memories define how to act; experiential/declarative "
                "describe what happened or what is known."
            ),
        ),
    ] = None,
    # ... more parameters
)
```

**Key Guidance in Tool**:
- "Should be clear and self-contained"
- "most agent-created memories are user-scoped"
- Weight ranges: 0.8-1.0 high priority, 0.7 default, 0.5-0.7 nice-to-know
- Branch types: rationale, provenance, description, evidence, approval
- Content types: experiential, declarative, behavioral

### 2.2 memory() Unified Tool
**Location**: `memory-hub-mcp/src/tools/memory.py`

**Docstring** (visible to agent):
```python
async def memory(...) -> dict[str, Any]:
    """All-in-one memory operations. Call register_session(api_key=...) first.

    Read actions:
      search(query, [scope, project_id, options: max_results, focus, domains, ...])
        Semantic search. Returns cache-optimized stable ordering by default.
      read(memory_id, [project_id, options: include_versions, hydrate])
        Retrieve memory by UUID with optional version history.
      # ... 10 more read actions

    Write actions:
      write(content, [scope, project_id, options: weight, parent_id, branch_type, ...])
        Create memory node or branch. scope defaults to "user" if omitted.
      update(memory_id, [content, options: weight, metadata, domains])
        New version with a **new UUID**; old version preserved with is_current=false.
      delete(memory_id, [project_id])
        Soft-delete with cascade.
      relate(options: {source_id, target_id, relationship_type})
        Create directed graph edge between memories.
      report(memory_id, options: {observed_behavior})
        Flag contradiction against a stored memory.
      # ... 7 more write actions
    """
```

**28 total actions** documented in the docstring.

### 2.3 update_memory Tool
**Location**: `memory-hub-mcp/src/tools/update_memory.py`

**Tool Description**:
```python
async def update_memory(
    memory_id: Annotated[
        str,
        Field(description="UUID of the memory to update."),
    ],
    content: Annotated[
        str | None,
        Field(
            description=(
                "New content text. Omit if only updating metadata. "
                "Creates a new version with a new UUID; the old version is "
                "preserved with is_current=false."
            ),
        ),
    ] = None,
    # ...
)
```

**Key Guidance**:
- "Creates a new version with a new UUID"
- "old version is preserved with is_current=false"
- Emphasizes version preservation pattern

---

## 3. Extraction Prompts (Server-Side)

These are used by MemoryHub's backend services to process stored data.

### 3.1 Conversation Extraction
**Location**: `prompts/conversation_extraction.yaml`

**Purpose**: Extract discrete facts from conversation threads

**Used By**: 
- `thread(action="extract")` MCP tool
- Background "dreaming" pipeline
- `memoryhub_core.services.dreaming.extract_from_thread()`

**System Prompt**:
```yaml
system_prompt: |
  You are a memory extraction system. Given a sequence of conversation
  messages, extract discrete facts, preferences, decisions, and knowledge
  that are worth remembering for future conversations.

  Return a JSON object with an "extractions" array. Each extraction has:
    - "content": a clear, self-contained statement of the fact or decision.
      Another agent should understand it without seeing the original conversation.
    - "weight": a float between 0.0 and 1.0 indicating importance:
      - 1.0: critical policy or hard constraint
      - 0.8-0.9: strong preference or important decision
      - 0.5-0.7: useful context or nice-to-know
      - below 0.5: trivial or ephemeral (skip these entirely)
    - "domains": an array of 0-3 short domain tags

  Rules:
  - Extract only information that would be valuable in a future conversation.
  - Each extraction must be self-contained.
  - Do not extract greetings, acknowledgments, or procedural chatter.
  - Do not extract purely ephemeral information.
  - Merge related facts into a single extraction.
  - Prefer concrete facts over vague summaries.
```

**Note**: This runs **after** the conversation, not during.

### 3.2 Fact Extraction
**Location**: `prompts/fact_extraction.yaml`

**Purpose**: Break down stored memories into independently searchable facts

**Used By**:
- Write-time eager extraction pipeline
- Background dreaming path
- `memoryhub_core.services.memory.create_fact_children()`

**System Prompt**:
```yaml
system_prompt: |
  You are a memory extraction agent. Given a document, extract every
  discrete fact as a separate, self-contained statement. Each fact should
  be understandable without the rest of the document.

  Return a JSON object with a "facts" array. Each fact has:
    - "content": a clear, self-contained statement of the fact.
    - "weight": a float between 0.0 and 1.0 indicating importance:
      - 1.0: critical policy or hard constraint
      - 0.8-0.9: strong preference or important decision
      - 0.5-0.7: useful context or nice-to-know
      - below 0.5: trivial or ephemeral (skip these entirely)
    - "domains": an array of 0-3 short domain tags

  Include: preferences, opinions, experiences, biographical details,
  relationships, habits, goals, likes/dislikes, stated intentions.

  Do NOT include: inferences not explicitly stated, generalizations,
  meta-commentary about the conversation format, purely ephemeral information.

  Be thorough -- capture every fact, even minor ones.
```

**Note**: This creates child memory nodes from a parent memory.

### 3.3 Entity Extraction
**Location**: `prompts/entity_extraction.yaml`

**Purpose**: Extract POLE+O entities (Person, Organization, Location, Event, Object) and relationships

**Used By**:
- Stage 3 LLM fallback in entity extraction pipeline
- `memoryhub_core.services.extraction.LLMEntityExtractor`

**System Prompt**:
```yaml
system_prompt: |
  You are an entity extraction system. Extract entities and relationships
  from the given text. Return a JSON object with two arrays:

  "entities": each with:
    - "name": the canonical name of the entity
    - "type": one of "person", "organization", "location", "event", "object"
    - "confidence": a float between 0.0 and 1.0

  "relationships": each with:
    - "source_name": name of the source entity
    - "target_name": name of the target entity
    - "relationship_type": uses, works_at, part_of, located_in, created_by, etc.

  Rules:
  - Only extract entities that are clearly named in the text.
  - Use "object" for software, tools, technologies, frameworks, databases.
  - Do not invent entities or relationships not supported by the text.
```

**Note**: This populates the knowledge graph.

---

## 4. Weight Guidance Summary

Consistent across all layers:

| Weight | Meaning | Use For | Examples |
|--------|---------|---------|----------|
| **1.0** | Critical policy / hard constraint | Must-follow rules, breaking changes | "Never commit secrets to git" |
| **0.8-0.9** | Strong preference / important decision | Architectural choices, key decisions | "Use PostgreSQL for persistence" |
| **0.7** | Default / useful context | General facts, nice-to-know | "User prefers dark mode" |
| **0.5-0.7** | Nice-to-know context | Background info, minor preferences | "User mentioned liking coffee" |
| **< 0.5** | Trivial / ephemeral | Skip these entirely | "User said hi at 3pm" |

---

## 5. Scope Guidance Summary

Consistent across all layers:

| Scope | Purpose | Owner | Examples |
|-------|---------|-------|----------|
| **user** | Personal preferences | User ID | "Prefers concise responses" |
| **project** | Project-specific context | Project ID | "Authentication uses OAuth2" |
| **campaign** | Cross-project knowledge | Campaign UUID | "Company coding standards" |
| **role** | Role-based patterns | Role ID | "DevOps team deployment process" |
| **organizational** | Team/org patterns | Org ID | "PR review requires 2 approvals" |
| **enterprise** | Mandated policies | Enterprise ID | "GDPR compliance requirements" |

---

## 6. Content Type Guidance

From tool descriptions:

| Type | Purpose | Examples |
|------|---------|----------|
| **experiential** | What happened | "User encountered bug X on 2024-01-15" |
| **declarative** | What is known | "System uses Redis for caching" |
| **behavioral** | How to act | "Always run tests before committing" |

**Note**: Behavioral memories are used by `reconstruct()` action for agent behavior synthesis.

---

## 7. Branch Type Guidance

From tool descriptions:

| Branch Type | Purpose | Parent-Child Relationship |
|-------------|---------|---------------------------|
| **rationale** | Explain "why" behind decision | Decision → Rationale |
| **provenance** | Source/derivation trail | Fact → Source |
| **description** | Extended details | Summary → Full Description |
| **evidence** | Supporting evidence | Claim → Evidence |
| **approval** | Authorization record | Request → Approval |
| **chunk** | Semantic chunk (system-managed) | Document → Chunk |

---

## 8. DO vs DON'T Lists

### DO Write (from _HYGIENE_BLOCK):
- Preferences
- Decisions
- Architectural choices
- Tool configuration
- Workflow patterns

### DON'T Write (from _HYGIENE_BLOCK):
- Ephemeral things ("user asked me to read a file")
- Code patterns (derivable from code)
- Git history (use git log)
- Debugging solutions (in commit messages)
- Temporary state

### DO Include (from fact_extraction.yaml):
- Preferences, opinions
- Experiences, biographical details
- Relationships, habits, goals
- Likes/dislikes
- Stated intentions
- Changes in preference over time

### DON'T Include (from fact_extraction.yaml):
- Inferences not explicitly stated
- Generalizations
- Meta-commentary about conversation format
- Purely ephemeral information

---

## 9. Update vs Write Pattern

**From agent instructions and tool descriptions**:

- **Use `update_memory`** to revise existing entry
  - Preserves version history
  - Creates new UUID
  - Old version kept with `is_current=false`
  
- **Use `write_memory`** only for new facts
  - Creates root-level memory
  - First version

**Pattern**: Search → Read → Update (not Search → Write)

---

## 10. Contradiction Detection

**From agent instructions**:

When user behavior contradicts a loaded memory:
1. Call `report_contradiction(memory_id, observed_behavior)`
2. Server tracks contradiction counts
3. Surfaces stale memories for review

**Resolution actions**:
- `accept_new` - Trust new behavior, mark old memory stale
- `keep_old` - Reject contradiction, keep existing memory
- `mark_both_invalid` - Both are wrong
- `manual_merge` - Human intervention needed

---

## 11. Complete File Locations

### Agent Instructions (Client-Side)
```
memoryhub-cli/
└── src/
    └── memoryhub_cli/
        └── project_config.py          # Templates source
            ├── _UNIVERSAL_HEADER
            ├── _HYGIENE_BLOCK
            ├── _UNIVERSAL_PATTERN_BLOCKS
            │   ├── "eager"
            │   ├── "lazy"
            │   ├── "lazy_with_rebias"
            │   └── "jit"
            ├── _CONTRADICTION_ENABLED
            └── _CAMPAIGN_BLOCK

<project-root>/
└── .claude/
    └── rules/
        └── memoryhub-loading.md       # Generated output
```

### Tool Descriptions (MCP Schema)
```
memory-hub-mcp/
└── src/
    └── tools/
        ├── memory.py                  # Unified tool (28 actions)
        ├── write_memory.py            # Standalone write
        ├── update_memory.py           # Standalone update
        ├── search_memory.py           # Standalone search
        └── thread.py                  # Thread operations (9 actions)
```

### Extraction Prompts (Server-Side)
```
prompts/
├── conversation_extraction.yaml       # Thread → Memories
├── fact_extraction.yaml               # Memory → Facts
└── entity_extraction.yaml             # Text → Entities + Relationships
```

---

## 12. Key Insights

### Separation of Concerns

**Agent Layer** (Client):
- When to write (trigger conditions)
- What to write (content selection)
- How to organize (scope, weight, branches)

**Tool Layer** (MCP):
- How to format (parameters, schemas)
- What options exist (actions, parameters)
- What constraints apply (required fields)

**Extraction Layer** (Server):
- How to decompose (facts, entities)
- How to process (background dreaming)
- How to enrich (graph relationships)

### Consistency Patterns

**Weight ranges** are consistent across:
- Agent instructions (_HYGIENE_BLOCK)
- Tool descriptions (write_memory, memory)
- Extraction prompts (conversation_extraction, fact_extraction)

**Scope taxonomy** is consistent across:
- Agent instructions (_HYGIENE_BLOCK)
- Tool descriptions (all write tools)
- Schema validation (MemoryNode model)

**Self-contained requirement** appears in:
- Agent instructions ("concise and self-contained")
- Tool descriptions ("Should be clear and self-contained")
- Extraction prompts ("self-contained statement")

### Design Philosophy

**Lazy by default**: Most instructions use lazy pattern (wait for task, then search)

**Version preservation**: Update creates new UUID, preserves old version

**User scope default**: "most agent-created memories are user-scoped"

**Concrete over vague**: Repeated in extraction prompts

**Proactive storage**: No explicit "when to store" trigger list, relies on hygiene DO/DON'T

---

## 13. Missing / Implicit Guidance

What's **NOT** explicitly stated but implied:

1. **When to store during conversation**: No explicit trigger list (e.g., "store after user corrects you"). Relies on DO/DON'T lists.

2. **How many memories per turn**: No guidance on volume. Agent must infer from hygiene rules.

3. **Deduplication strategy**: Mentioned in extraction prompts but not in agent instructions.

4. **Search before write**: Implied by update vs write pattern, not explicit.

5. **Memory lifecycle**: When to delete, archive, or mark stale not in agent instructions.

---

## 14. Comparison to Other Systems

### vs BFCL Memory KV
- **BFCL**: Vague ("actively manage important information")
- **MemoryHub**: Explicit DO/DON'T lists, weight guidance, examples

### vs Claude Code auto-memory
- **Claude Code**: Type-based (user, feedback, project, reference)
- **MemoryHub**: Scope-based (user, project, campaign, role, org, enterprise)
- **Claude Code**: File-based storage
- **MemoryHub**: Database with graph relationships

### vs GBrain
- **GBrain**: "Brain-first protocol", route by question shape
- **MemoryHub**: Search patterns (eager, lazy, jit)
- **GBrain**: Single namespace
- **MemoryHub**: Multi-scope hierarchy

---

## 15. For llm_memory_bench Integration

### What to Extract for Benchmarking

**Agent Instructions** (from project_config.py):
- _HYGIENE_BLOCK (DO/DON'T lists)
- Pattern blocks (lazy most relevant)
- Weight guidance (1.0 → 0.5 ranges)

**Tool Descriptions** (from memory.py):
- write() action docstring
- Parameter descriptions (content, scope, weight)

**Exclusions** (not relevant for benchmarking):
- Extraction prompts (server-side, post-storage)
- Entity extraction (graph building, not storage decision)
- Loading patterns (retrieval, not storage)

### Prompt Version for Benchmarking

Create: `prompts/memoryhub/v1_lazy_2026-08.yaml`

Include:
- _HYGIENE_BLOCK as `storage_policy`
- Lazy pattern as `loading_pattern`
- Weight/scope/branch guidance as `metadata_guidance`
- write_memory tool description as `tool_schema`

Exclude:
- Extraction prompts (separate pipeline)
- Thread operations (not core to storage)
- Admin operations (not agent-facing)
