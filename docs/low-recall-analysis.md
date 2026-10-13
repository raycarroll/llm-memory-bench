# Low Recall Analysis: GPT-4o Simple System (7.1%)

## The Problem

GPT-4o with `simple` system achieved only **7.1% recall** (1 of 14 facts stored across 5 conversations).

## Root Cause: Prompt vs Dataset Philosophy Mismatch

### The Conflict

**AlpsBench Philosophy:**
- Store facts mentioned in **one-off task requests**
- Example: "Write an affidavit with my birthdate" → Store the birthdate

**Simple Prompt Philosophy:**
- DO NOT store facts from **one-off task requests**
- Store only persistent preferences/background for future sessions

### Concrete Example

**Turn 18 - User Request:**
```
"Can you write an affidavit for me which contains the following 
information regarding me: Abidullah Zarghoon resident of Sultani Tower, 
Behind Habibya Lycee, Darul Aman Road, Police District 7, Kabul, born 
on 19 January 1993 in Wardak, Afghanistan; cousin of the groom..."
```

**Ground Truth:** Store 3 facts
- Born on 19 January 1993 in Wardak, Afghanistan
- Resident of Sultani Tower...
- Cousin of Samiullah Mahaal (groom)

**Simple Prompt Guidance:**
```yaml
DO NOT save:
- One-off task requests ('fix this bug', 'translate this sentence')
- Ephemeral context
```

**GPT-4o Decision:** This IS a one-off task (write affidavit), so storing is ambiguous.

**What GPT-4o Did:** Stored a combined fact (got credit for 1/3)

## The Dataset Design

Looking at the AlpsBench conversations:

```python
# Conversation sess_0517169e56bb has 35 turns
# Only turns with facts to store: 18, 31
# Turn 18: "Write an affidavit with this info about me"
# Turn 31: "Add my nationality to the affidavit"

# Both are task requests, not explicit "remember this" statements
```

**AlpsBench measures:** Can the model extract facts mentioned in task context?

**Simple prompt optimizes for:** Only store explicit persistent preferences

## Why This Matters

### Noise Turns (171 in 5 conversations)
The prompt's conservative stance prevents false positives:
- 98.2% noise resistance (only 3 false positives)
- Examples of correctly skipped noise:
  - "Create a description for procurement of furniture"
  - "Write an email requesting a quotation"

### Signal Turns (14 facts expected)
But the same conservative stance causes false negatives:
- User provides personal info FOR A TASK
- Prompt says "don't store one-off tasks"
- Model skips storage

## Is This a Bug or Design Choice?

**Not a bug in implementation** - the system works as designed.

**Design tension:**
1. **High precision mode** (current): Conservative, avoids noise, misses embedded facts
2. **High recall mode**: Aggressive, captures all facts, increases false positives

## What the Numbers Show

| System | Recall | Precision | Noise Resistance | Philosophy |
|--------|--------|-----------|------------------|------------|
| Simple | 7.1% | 25% | 98.2% | "Only persistent facts" |
| MemoryHub | 0.0% | N/A | 100% | GPT-4o didn't use it at all |

## Solutions

### 1. Revise Simple Prompt (Increase Recall)

```yaml
SAVE when user shares:
- Personal preferences, background, constraints
- Facts about themselves mentioned in ANY context
  (even if part of a task request)

Examples that SHOULD be stored:
✅ "Write an affidavit with my birthdate 1993-01-19" → Store birthdate
✅ "I'm allergic to nuts" (mentioned while asking for recipes)
✅ "I live in Kabul" (mentioned while asking for directions)

DO NOT save:
- Pure task output ("create a description of X")
- Facts about OTHER people unless explicitly requested
```

### 2. Test with Claude Models

Claude typically has better proactive tool-calling judgment. Run same dataset:
```bash
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system simple \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --max-conversations 5
```

### 3. Use Behavioral Relevance System

The `behavioral_relevance` system uses a different test:
"If I forgot this, would my next response be less helpful?"

This might better align with AlpsBench's philosophy.

## Conclusion

**The 7.1% recall is not a bug** - it's the result of:
1. Simple prompt designed for high precision
2. AlpsBench testing extraction from task context
3. GPT-4o following the conservative guidance

**To improve recall**, we need to either:
- Adjust the prompt to be more aggressive
- Use a different model (Claude)
- Use a different system (behavioral_relevance)
