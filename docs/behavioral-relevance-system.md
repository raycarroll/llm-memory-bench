# Behavioral Relevance System

A memory storage system designed to maximize both precision and recall through a clear behavioral relevance test.

## Core Principle

**Store facts that change how you should behave in future interactions**

## The Test

Before storing any fact, ask:

> "If I forgot this, would my next response be less helpful or misaligned?"

- **YES** → Store it with appropriate weight
- **NO** → Skip it

## Design Goals

Based on analysis of three existing systems:
- **BFCL**: 80% recall, 19% precision (too vague)
- **MemoryHub**: 55% recall, 22% precision (too restrictive)
- **GBrain**: 85% recall, 7% precision (no filtering)

**Target Performance**:
- Recall: **75-80%** (capture most important facts)
- Precision: **30-35%** (significantly better than current best 22%)
- F1: **50-55%** (vs current best 31%)

## Five STORE Categories

### 1. Preferences (Weight: 0.7-1.0)
How the user wants things done

**Examples**:
- "Prefers concise responses"
- "Uses PostgreSQL, not MySQL"
- "Wants code without explanations"
- "Dislikes verbose error messages"

### 2. Constraints (Weight: 0.9-1.0)
Rules that limit options

**Examples**:
- "Must comply with GDPR"
- "Never commit secrets to git"
- "Production uses Python 3.9, can't upgrade"
- "PRs require 2 approvals"

### 3. Domain Context (Weight: 0.6-0.8)
Background that improves relevance

**Examples**:
- "auth service = OAuth2 implementation"
- "Senior backend engineer, new to frontend"
- "Focusing on performance optimization this quarter"
- "Chose microservices over monolith for scalability"

### 4. Corrections (Weight: 0.8-0.9)
Learning from mistakes

**Examples**:
- "Actually prefers tabs over spaces"
- "Don't use deprecated API v1, use v2"
- "When I say 'deploy', I mean staging, not prod"

### 5. Patterns (Weight: 0.7-0.8)
Repeated behaviors that predict future needs

**Examples**:
- "Always runs tests locally before pushing"
- "Deploys every Friday afternoon"
- "Frequently asks about database schema"

## Six SKIP Categories

### 1. Observations
Noticing without action

**Examples**:
- ✗ "User seems tired"
- ✗ "User is working on feature X right now"
- ✗ "User mentioned coffee" (unless it's a preference)

### 2. Procedural Details
What you did, not what to remember

**Examples**:
- ✗ "I read file authentication.py"
- ✗ "I ran npm install"
- ✗ "I searched the codebase for API usage"

### 3. Transient State
Temporary conditions

**Examples**:
- ✗ "Build is currently running"
- ✗ "Waiting for deployment"
- ✗ "Database migration in progress"

### 4. Generic Knowledge
Facts not specific to this user/project

**Examples**:
- ✗ "Python is a programming language"
- ✗ "Git is version control"
- ✗ "REST APIs use HTTP"

### 5. Acknowledgments
Social/conversational lubricant

**Examples**:
- ✗ "User said hello"
- ✗ "User thanked me"
- ✗ "User acknowledged the fix"

### 6. Duplicates
Already stored or derivable

**Examples**:
- ✗ Don't store if already in codebase/docs
- ✗ Don't store rephrasing of existing memory
- ✗ Don't store obvious implications

## Weight Assignment

Weight based on impact to future behavior:

| Weight | Meaning | Use For |
|--------|---------|---------|
| **1.0** | Critical constraint | "NEVER do X", breaking policies |
| **0.9** | Strong preference | "ALWAYS do Y", hard requirements |
| **0.8** | Important context | Corrections, key background |
| **0.7** | Moderate preference | Workflow patterns, preferences |
| **0.6** | Nice-to-know | Domain terminology, context |
| **<0.6** | Don't store | Fails behavioral relevance test |

## When Uncertain

**Default: Don't store**

- If behavioral impact is unclear, skip it
- Better to miss borderline facts than pollute with noise
- User can always restate important information

## Tool Schema

```json
{
  "name": "store_fact",
  "description": "Store a fact that changes how you should behave in future conversations",
  "input_schema": {
    "type": "object",
    "properties": {
      "fact": {
        "type": "string",
        "description": "The fact to store. Must be clear and self-contained."
      },
      "category": {
        "type": "string",
        "enum": ["preference", "constraint", "domain_context", "correction", "pattern"],
        "description": "Type of fact"
      },
      "weight": {
        "type": "number",
        "description": "Impact on future behavior (0.6-1.0)",
        "minimum": 0.6,
        "maximum": 1.0
      }
    },
    "required": ["fact", "category", "weight"]
  }
}
```

## Why This Should Work

### High Recall (>75%)
1. **Five broad categories** cover most valuable facts
2. **Includes indirect statements** ("I usually" → preference)
3. **Stores corrections** (catches when initially missed)
4. **Patterns category** catches repeated implicit preferences

### High Precision (>30%)
1. **Behavioral relevance test** provides objective decision boundary
2. **Six skip categories** with concrete examples
3. **Default to skip** when uncertain
4. **Examples teach the pattern** better than abstract rules

### Improvements Over Current Systems

**vs BFCL** (19% precision):
- Replaces "important" with "changes behavior" (objective test)
- Adds skip categories with examples
- Expected: +10-15% precision, -5% recall

**vs MemoryHub** (55% recall):
- Removes "skip ephemeral" (too broad)
- Adds "patterns" category (catches implicit preferences)
- Expected: +20-25% recall, -2% precision

**vs GBrain** (7% precision):
- Adds behavioral relevance filter (not everything mentioned matters)
- Keeps high recall through 5 broad STORE categories
- Expected: +25% precision, -10% recall

## Research Foundation

Based on cognitive science memory taxonomy:
- **Declarative memory**: Episodic + semantic facts
- **Procedural memory**: How-to knowledge (behavioral)
- **Prospective memory**: Future intentions

And LLM memory research findings:
- Personalization facts have highest utility
- Temporal facts decay quickly
- Behavioral patterns > individual actions

## Testing Strategy

### Phase 1: Initial Benchmark
Run on same dataset as BFCL/MemoryHub/GBrain:
- 10 conversations, 50 expected facts
- Cumulative ground truth
- Hybrid matcher evaluation

### Phase 2: Analysis
If precision low: Add more skip examples
If recall low: Broaden STORE categories
If both low: Revise behavioral relevance test

### Phase 3: Iteration
- Variant A: "Changes behavior" test (current)
- Variant B: "Improves future responses" test
- Variant C: Weight-first approach

## Usage

```bash
# Run benchmark
llm-memory-bench run \
  --dataset datasets/test_dataset_cumulative.yaml \
  --system behavioral_relevance \
  --provider vertex \
  --model claude-sonnet-4-5@20250929

# Evaluate
llm-memory-bench evaluate \
  results/vertex_*_behavioral_relevance_*.json \
  --matcher hybrid \
  --judge-provider anthropic \
  --judge-model claude-sonnet-4-20250514
```

## Files

**Prompts**:
- `prompts/behavioral_relevance/metadata.yaml` - Version metadata
- `prompts/behavioral_relevance/v1_2026-08.yaml` - Prompt content

**System**:
- `src/llm_memory_bench/systems/behavioral_relevance.py` - Implementation

**Documentation**:
- `docs/behavioral-relevance-system.md` - This file
- `docs/three-system-comparison.md` - Comparison context

## Expected Results

**Conservative estimate**:
- Recall: 75-80%
- Precision: 30-35%
- F1: 50-55%
- Tool calls: ~180 (between MemoryHub's 138 and BFCL's 211)

**Optimistic estimate**:
- Recall: 80-85%
- Precision: 35-40%
- F1: 55-60%
- Tool calls: ~160

## Key Innovation

**Objective decision criterion**: "Changes behavior" is more testable than "important"

**Cognitive alignment**: Mirrors human procedural memory (actionable vs observational)

**Examples over rules**: 11 concrete examples demonstrate the pattern

**Clear default**: When uncertain, skip (precision over recall in edge cases)
