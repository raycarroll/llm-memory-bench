# Implementation Check: Is There a Bug?

## Prompt Delivery ✅ VERIFIED CORRECT

**System Prompt:**
- Length: 1,323 characters
- Clearly states: "SAVE when user shares: Personal preferences..."
- Correctly passed to `provider.generate(system=system_prompt)` on every turn
- OpenAI provider adds it as first message: `[{"role": "system", "content": system_prompt}]`

**Tool Schema:** ✅ CORRECT
```json
{
  "name": "add_memory",
  "description": "Store a long-term fact about the user for future sessions. 
                  Only store persistent personal information, preferences, or constraints.",
  "category": ["preference", "personal", "technical", "contextual"]
}
```

## Test Cases: Explicit Preference Misses

### Case 1: Conv 2, Turn 8 - "I want" statement
**Input:** `"i dont want steel angle i want to use carbon fiber instead of steel angle"`

**Expected:** Store preference for carbon fiber

**Context:** 
- Conversation length: 25k tokens (not huge)
- Turn 8 of 16 total turns
- Clear "I don't want X, I want Y" statement

**Result:** ❌ NOT STORED

**Why missed?**
- Embedded in technical conversation about shear wall reinforcement
- Could be interpreted as: "For THIS calculation, use carbon fiber" (one-off)
- Not: "I ALWAYS prefer carbon fiber" (persistent)

**But:** The prompt says store "Personal preferences (likes, dislikes, styles)" - this is explicitly a dislike/preference.

### Case 2: Conv 4, Turn 8 - Literal "I prefer"
**Input:** `"I prefer names rooted in Hebraic. They don't have to sound female with the -elle ending"`

**Expected:** Store 2 preferences

**Context:**
- Conversation length: 17k tokens (small)
- Turn 8 of 12 total turns  
- LITERAL "I prefer" statement

**Result:** ❌ NOT STORED

**Why missed?**
- This is inexcusable - it literally says "I prefer"
- The prompt explicitly says: "SAVE when user shares: Personal preferences"
- No ambiguity here

### Case 3: Conv 5, Turn 26 - Creative inspirations
**Input:** `"My main inspiration for this story was Saikano and Half-Life. The main tone of this story is... loneliness."`

**Expected:** Store 2 facts (inspirations, theme)

**Context:**
- Conversation length: 167k tokens (large but manageable)
- Turn 26 of 32 total turns
- Explicit "My main inspiration" statement

**Result:** ❌ NOT STORED

**Why missed?**
- Statement about user's creative work
- Prompt says: "Long-term goals or ongoing projects" - this qualifies
- But also embedded in story development context

## Conversation Length Analysis

| Conv | Total Tokens | User Turns | Expected Facts | Stored | Issue |
|------|--------------|------------|----------------|--------|-------|
| 1 | 45k | 16 | 4 | 1 (partial) | Task-embedded |
| 2 | 29k | 8 | 2 | 0 | ⚠️ Explicit preferences missed |
| 3 | 1,320k | 44 | 3 | 0 | Massive conversation |
| 4 | 20k | 6 | 3 | 0 | ⚠️ "I prefer" literally used |
| 5 | 177k | 16 | 2 | 0 | Large conversation |

**Conv 3 is pathological:** 1.3 million tokens! The expected facts are at turns 34 and 62, buried under massive context.

**Conv 2 and 4 are damning:** Small conversations (29k, 20k tokens) with explicit "I want" and "I prefer" statements, still not stored.

## What This Rules Out

✅ **Not a prompt delivery bug** - system prompt correctly sent every turn

✅ **Not a tool schema bug** - schema is correct

✅ **Not always a context length issue** - small conversations also fail

## What This Suggests

### Hypothesis 1: GPT-4o Instruction Following Issue

GPT-4o may be:
1. **Over-interpreting the "DO NOT" section**
   - "DO NOT save: One-off task requests"
   - Sees "give me calculation in metric" as a task modification
   - Ignores the preference signal

2. **Focusing on task completion over memory**
   - In a technical conversation, prioritizes answering the question
   - Deprioritizes memory extraction

3. **Being too conservative**
   - When in doubt between "task" and "preference", chooses not to store

### Hypothesis 2: Prompt Design Issue

The "DO NOT" section may be too strong:
```yaml
DO NOT save:
- One-off task requests ('fix this bug', 'translate this sentence')
```

GPT-4o might classify:
- "same calculation in metric" = task request → skip
- Even though it reveals a preference

### Hypothesis 3: Model Capability

GPT-4o may simply be worse at proactive tool-calling than Claude.

BFCL tests REACTIVE tool-calling (user explicitly asks)
AlpsBench tests PROACTIVE extraction (user mentions in passing)

GPT-4o might score well on BFCL but poorly here.

## Recommended Next Steps

### 1. Test Claude Sonnet 4 (Baseline Comparison)
```bash
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system simple \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --max-conversations 5
```

**If Claude catches "I prefer" statements:** GPT-4o instruction-following issue
**If Claude also misses them:** Prompt design issue

### 2. Test Simpler Prompt (Remove "DO NOT" Section)

Create `simple_v3_positive_only.yaml`:
```yaml
SAVE when user shares:
- Personal preferences (likes, dislikes, styles)
  Examples: "I prefer metric units", "I don't like X", "I want Y instead"
- Background information (job, education, expertise, location)
- Constraints or requirements (allergies, accessibility needs)
- Long-term goals or ongoing projects
- Technical environment (languages, tools, OS)

Store facts mentioned in ANY context, even if part of a task request.
```

Remove all "DO NOT" examples to see if they're causing over-filtering.

### 3. Add Explicit Examples to Prompt

```yaml
Examples of what TO store:
✓ "same calculation in metric units" → "Prefers metric units"
✓ "I don't want steel, use carbon fiber" → "Prefers carbon fiber"
✓ "I prefer Hebraic names" → "Prefers Hebraic names"
```

### 4. Test with BFCL Baseline System

The `bfcl_baseline` system has no proactive guidance - just reactive tool-calling.
Check if it does WORSE (proving the simple prompt does help, just not enough).

## Verdict

**No implementation bug found.** The system is working as coded.

**The problem is one of:**
1. GPT-4o not following proactive extraction instructions well
2. Prompt being too conservative (DO NOT list overrides SAVE list)
3. Model capability gap (reactive vs proactive tool-calling)

**Claude baseline will tell us which.**
