# AlpsBench Baseline Comparison Investigation

## The Gap

**AlpsBench Paper Results (GPT-4.1-mini):** 33% F1  
**Our Results (GPT-4o with simple system):** 11.1% F1

**This is a 3x performance gap on the same dataset with a newer/better model.**

## Critical Questions

### 1. Metric Equivalence
**Question:** Is their F1 calculated the same way as ours?

**Our metrics (per-turn):**
```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
```

**To check:**
- [ ] Review AlpsBench evaluation code at https://github.com/ThisIsCosine/AlpsBench
- [ ] Check if they use per-turn vs cumulative evaluation
- [ ] Check their matching criteria (exact vs semantic)
- [ ] Check if they filter low-confidence annotations (we filter confidence < 0.5)

### 2. Prompt Differences
**Question:** What prompt did they use?

**Our simple prompt:**
```yaml
SAVE when user shares:
- Personal preferences (likes, dislikes, styles)
- Background information (job, education, expertise, location)
...

DO NOT save:
- One-off task requests
- Greetings or small talk
...
```

**To check:**
- [ ] Find their prompts in the repo
- [ ] Compare prompt length, examples, tone
- [ ] Check if they use explicit memory instructions
- [ ] Check if they provide few-shot examples

### 3. Tool Schema Differences
**Question:** What tool did they expose to the model?

**Our tool:**
```json
{
  "name": "add_memory",
  "description": "Store a long-term fact...",
  "parameters": {
    "fact": "string",
    "category": "enum[preference, personal, technical, contextual]"
  }
}
```

**To check:**
- [ ] Find their tool schema
- [ ] Check if they use a simpler or more explicit tool
- [ ] Check if they provide usage examples in the description

### 4. Evaluation Matching
**Question:** How lenient is their matching?

**Our matching:** Semantic via embedding similarity (threshold ~0.7) or LLM judge

**Possible differences:**
- Exact string match (too strict)
- Very lenient substring match
- Different similarity threshold
- Post-processing (lemmatization, normalization)

**To check:**
- [ ] Review their evaluation/matching code
- [ ] Check if they use GPT to judge matches
- [ ] Check threshold values

### 5. Data Processing
**Question:** Did they use all annotations or filter some?

**Our processing:**
- Filters `confidence < 0.5` (removes 187 ERROR-tagged items)
- Uses `utterance_index` directly (84% correct for task1)
- English-only by default (--all-languages flag available)

**To check:**
- [ ] Check if they filter low-confidence
- [ ] Check language filtering
- [ ] Check if they fix utterance_index misalignments

### 6. Model Configuration
**Question:** Any special model settings?

**Our setup:**
- System prompt passed as system message
- Tools in Anthropic/OpenAI format
- max_completion_tokens: 1024
- No temperature (uses model default)

**To check:**
- [ ] Check their temperature setting
- [ ] Check max_tokens
- [ ] Check if they use chat history differently
- [ ] Check if they use any special prompting techniques

## Hypothesis Ranking

### Most Likely Causes (in order)

1. **Prompt Design (HIGH LIKELIHOOD)**
   - Their prompt may be more explicit about when to store
   - May include few-shot examples
   - May be less conservative (no "DO NOT" list)

2. **Evaluation Matching (MEDIUM LIKELIHOOD)**
   - Their matcher may be more lenient
   - May use different similarity threshold
   - May have post-processing we don't

3. **Tool Schema (MEDIUM LIKELIHOOD)**
   - Simpler tool = easier to use
   - Better description = clearer usage
   - Different parameters = different behavior

4. **Data Filtering (LOW LIKELIHOOD)**
   - Unless they include low-confidence annotations
   - Or unless they fix more data quality issues

5. **Model Configuration (LOW LIKELIHOOD)**
   - Temperature/max_tokens unlikely to cause 3x gap
   - Unless they use a very different approach

## Action Plan

### Immediate (Can Do Now)
1. ✅ Clone AlpsBench repo: `git clone https://github.com/ThisIsCosine/AlpsBench`
2. Find and read:
   - Paper (if available)
   - Evaluation code
   - Baseline prompts
   - Tool schemas
3. Document exact differences

### Then Test
1. Replicate their exact setup (prompt + tool + evaluation)
2. Run on 5 conversations to verify we get similar scores
3. If yes, A/B test each difference to find what matters:
   - Their prompt vs ours
   - Their tool vs ours
   - Their matching vs ours

### Finally
1. Update our systems with learnings
2. Document best practices
3. Re-run full evaluation

## Expected Findings

**If the gap is in prompting:**
- Our conservative "DO NOT" list is hurting recall
- Need more explicit examples
- Need to reframe task requests

**If the gap is in matching:**
- Our embedding threshold is too strict
- Need to adjust or use LLM matching

**If the gap is in tool schema:**
- Simpler tool = better performance
- Description clarity matters

**If the gap is elsewhere:**
- Document unexpected finding
- Adjust our implementation

## Notes

The fact that GPT-4.1-mini (older, smaller) got 33% F1 while our GPT-4o gets 11.1% F1 suggests:
- **Not a model capability issue** (GPT-4o should be better)
- **Likely a setup/prompt issue** (we're not using the model optimally)
- **Possibly an evaluation issue** (we're being too strict)

This is actually good news - it means there's likely a straightforward fix.
