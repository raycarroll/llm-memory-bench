# Prompt Versioning Architecture

All memory system prompts are stored as versioned artifacts, not hardcoded in Python files.

## Design Principles

1. **Design-time extraction**: Pull prompts from source projects manually, not at runtime
2. **Version tracking**: Each prompt version has metadata (source, commit, pull date)
3. **Reproducibility**: Lock prompt version per experiment for exact reproducibility
4. **Provenance**: Clear audit trail from source repo → commit → local file
5. **No runtime dependencies**: Don't fetch from source projects during benchmarks

## Directory Structure

```
prompts/
├── bfcl_memory_kv/
│   ├── metadata.yaml          # Source repo, versions available
│   ├── v4_2025-07.yaml        # Versioned prompt content
│   └── v5_2026-01.yaml        # Future version (when released)
├── memoryhub/
│   ├── metadata.yaml
│   └── v1_lazy_2026-08.yaml   # Version + pattern + date
├── gbrain/
│   ├── metadata.yaml
│   └── v0.46.19_2024-12.yaml  # Version from package.json
├── simple/
│   ├── metadata.yaml
│   └── v1_2024-11.yaml        # Internal test system
└── openclaw/
    ├── metadata.yaml
    └── v1_2026-09.yaml        # AGENTS.md Memory section + write/edit tools
```

## File Formats

### metadata.yaml

Tracks source and available versions:

```yaml
name: bfcl_memory_kv
description: BFCL Memory system with Key-Value backend

source:
  repo: https://github.com/ShishirPatil/gorilla
  path: berkeley-function-call-leaderboard/bfcl/eval_checker/eval_runner_helper.py
  documentation: https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html

versions:
  - version: v4_2025-07
    commit: abc123def456  # git commit hash when pulled
    pulled: 2026-08-31
    notes: |
      BFCL V4 memory category (July 2025 release).
      5 scenarios: student, customer, finance, healthcare, notetaker.
    default: true

  - version: v5_2026-01
    commit: def456abc123
    pulled: 2026-01-15
    notes: |
      BFCL V5 with updated storage policy prompts.

default_version: v4_2025-07
```

### {version}.yaml

Actual prompt content for that version:

```yaml
version: v4_2025-07
source_commit: abc123def456
pulled_date: 2026-08-31

# System prompt template
system_prompt: |
  You are an assistant with memory capabilities...

# Or structured by component
memory_instruction: |
  You have access to an advanced memory system...

scenarios:
  student: "You are an academic-support assistant..."
  customer: "You are a customer support assistant..."

# Tool schemas (optional - may be in separate tools.yaml)
tools:
  - name: core_memory_add
    description: "Add to core memory..."
    input_schema: {...}
```

## Python System Classes

All systems load from prompts/ directory:

```python
from pathlib import Path
import yaml

class BFCLMemoryKVSystem(MemorySystem):
    """BFCL Memory with Key-Value backend."""

    name = "bfcl_memory_kv"
    PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "bfcl_memory_kv"

    def __init__(self, scenario: str | None = None, prompt_version: str | None = None):
        self.scenario = scenario
        self.prompt_version = prompt_version or self._get_default_version()
        self._prompts = self._load_prompts(self.prompt_version)

    @classmethod
    def _get_default_version(cls) -> str:
        """Load default version from metadata.yaml."""
        metadata_path = cls.PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        return metadata["default_version"]

    @classmethod
    def _load_prompts(cls, version: str) -> dict:
        """Load prompts from versioned YAML file."""
        prompt_path = cls.PROMPTS_DIR / f"{version}.yaml"
        with open(prompt_path) as f:
            return yaml.safe_load(f)

    def system_prompt(self) -> str:
        # Build from loaded prompts, not hardcoded strings
        return self._prompts["system_prompt"].format(scenario=self.scenario)

    def version_info(self) -> dict:
        """Report which version is active."""
        return {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": f"{metadata['source']['repo']} {metadata['source']['path']}",
            "source_commit": self._prompts["source_commit"],
            "pulled_date": str(self._prompts["pulled_date"]),
        }
```

## Workflow

### Design-time: Adding a New System

1. Identify source repository and file paths
2. Create `prompts/{system_name}/` directory
3. Create `metadata.yaml` with source information
4. Extract prompts from source to `{version}.yaml`
5. Implement Python system class to load from YAML
6. Test that prompts load correctly

### Design-time: Updating an Existing System

1. Pull updated prompts from source repository
2. Note git commit hash
3. Create new `{version}.yaml` file (don't overwrite old)
4. Add version entry to `metadata.yaml`
5. Optionally update `default_version`
6. Old experiments still reference old version (reproducibility)

### Run-time: Using a System

```bash
# Uses default version from metadata.yaml
llm-memory-bench run --system bfcl_memory_kv --scenario student

# Future: explicit version override (not yet implemented)
llm-memory-bench run --system bfcl_memory_kv --scenario student --prompt-version v4_2025-07
```

Run results include version info:

```json
{
  "memory_system": {
    "system": "bfcl_memory_kv",
    "prompt_version": "v4_2025-07",
    "prompt_source": "https://github.com/ShishirPatil/gorilla ...",
    "source_commit": "abc123def456",
    "pulled_date": "2026-08-31",
    "scenario": "student"
  }
}
```

## Benefits

**Reproducibility**
- Lock exact prompt version per experiment
- Re-run with same prompts months later
- Compare prompt evolution (v4 vs v5) side-by-side

**Provenance**
- Clear trail: source repo → commit → pull date → local file
- Results cite exact source commit
- Can verify against upstream if needed

**Offline operation**
- No runtime dependency on source repos
- Works in air-gapped environments
- Faster (no network fetches)

**Version comparison**
- Test prompt changes: old vs new
- Measure impact of prompt engineering
- Regression testing when upstream updates

**Audit trail**
- When was this prompt pulled?
- What commit was it from?
- What changed between versions?

## Migration Plan

### Phase 1: BFCL (Complete)
- ✅ Created prompts/bfcl_memory_kv/ structure
- ✅ Extracted v4_2025-07.yaml
- ✅ Updated Python class to load from YAML
- ✅ Tested loading and version_info

### Phase 2: MemoryHub
- Extract current prompt to prompts/memoryhub/
- Note: Uses "lazy" loading pattern (there are other patterns)
- Version: v1_lazy_2026-08
- Update memoryhub.py to load from YAML

### Phase 3: GBrain
- Extract current prompt to prompts/gbrain/
- Version from package.json: v0.46.19
- Version: v0.46.19_2024-12
- Update gbrain.py to load from YAML

### Phase 4: Documentation
- Update README with prompt versioning
- Document extraction workflow
- Add examples of comparing versions

## Open Questions

1. **Tool schemas**: Should tools be in same YAML or separate tools.yaml?
2. **Prompt override**: Should CLI support --prompt-version flag?
3. **Validation**: Should we validate prompt structure on load?
4. **Caching**: Should we cache loaded prompts or reload each time?
5. **Multi-file prompts**: How to handle systems with many prompt components?

## Related

- [BFCL System](../src/llm_memory_bench/systems/bfcl_memory_kv.py) - First system using versioned prompts
- [MemoryHub System](../src/llm_memory_bench/systems/memoryhub.py) - To be migrated
- [GBrain System](../src/llm_memory_bench/systems/gbrain.py) - To be migrated
