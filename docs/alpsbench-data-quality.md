# AlpsBench Data Quality Notes

This document records known issues in the [AlpsBench dataset](https://huggingface.co/datasets/Cosineyx/Alpsbench) and the corrections applied during conversion.

## Annotation Source

AlpsBench annotations are **LLM-generated** (DeepSeek-v3.2) and then **human-reviewed** by senior practitioners. Reviewer-rejected annotations are marked with `[ERROR:]` tags in the value field and `confidence: 0.1`.

## Issue 1: Low-Confidence Reviewer Rejections

**Problem:** 187 task1 annotations and additional task2 annotations carry `confidence: 0.1` and `[ERROR:]` prefixes, indicating the human reviewer rejected them. Including these as ground truth inflates false negatives.

**Fix:** The adapter filters out all annotations with `confidence < 0.5` during conversion. This removed ~14% of task1 facts (1822 -> 1572).

**Verification:** Clean split confirmed — all 187 ERROR-tagged items have confidence 0.1, no non-error items do.

## Issue 2: Task2 `utterance_index` Misalignment

**Problem:** In task2 (memory updating), new memory annotations (`m_new_*`) have an `evidence` field with both an `utterance_index` and an `evidence.text` substring. The `utterance_index` is **unreliable**:

- `m_new_*` entries overwhelmingly have `utterance_index: 0` regardless of where the evidence text actually appears in the dialogue.
- Across dev+validation splits: **0 out of 469 dev** and **2 out of 469 validation** entries had `utterance_index` matching the turn containing the evidence text.

Updated memories (`m*_updated`) have `evidence: null` with no source turn indicated.

**Fix:** The adapter now uses text matching (`evidence.text` substring search) against dialogue turns instead of `utterance_index` to determine which turn a fact belongs to. For updated memories with no evidence, the fact is attributed to the last turn of the conversation (reflecting cumulative understanding).

**Verification:** Text matching resolves 100% of `m_new_*` entries to the correct turn (469/469 in both splits).

## Issue 3: Task1 `utterance_index` Minor Misalignment

**Problem:** Task1 `utterance_index` is mostly correct (84% match rate) but has ~154 mismatches in dev, primarily:

- Off-by-one errors (evidence text in adjacent turn)
- Evidence text that's a paraphrase rather than exact substring

**Current status:** Not corrected. Task1 still uses `utterance_index` from the annotation. The impact is minor — facts may be attributed to an adjacent turn, which doesn't significantly affect extraction evaluation since the model sees the full conversation.
