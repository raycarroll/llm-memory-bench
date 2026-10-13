from __future__ import annotations

from dataclasses import dataclass, field

import jsonschema

from .config import RunConfig, get_provider
from .dataset import Conversation, Dataset, ExpectedFact, GroundTruthType
from .matchers.base import FactMatcher, MatchVerdict
from .matchers.llm import LLMMatcher
from .providers.base import ToolCall
from .runner import ConversationResult, RunResult
from .systems import get_system
from .systems.base import MemorySystem


def _to_matcher_input(expected: ExpectedFact) -> str | dict:
    """Convert ExpectedFact to matcher input format.

    For AlpsBench facts (with label field), returns full dict for tuple matching.
    For legacy facts, returns just the fact string.
    """
    # Check if this is an AlpsBench fact (has label field)
    if expected.label is not None:
        return {
            "fact": expected.fact,
            "label": expected.label,
            "type": expected.type.value,
            "preference_attitude": expected.preference_attitude,
            "time_scope": expected.time_scope,
            "emotion": expected.emotion,
        }
    # Legacy format: just the fact string
    return expected.fact


@dataclass
class MatchCandidate:
    """A potential match with its similarity score."""
    tool_call: ToolCall
    similarity: float
    verdict: MatchVerdict


@dataclass
class FactMatch:
    expected_fact: str
    expected_type: str
    matched_tool_call: ToolCall | None
    verdict: MatchVerdict
    candidates: list[MatchCandidate] = field(default_factory=list)  # Top ranked matches
    pair_score: float | None = None  # AlpsBench pair score for matched pairs

    def to_dict(self) -> dict:
        d = {
            "expected_fact": self.expected_fact,
            "expected_type": self.expected_type,
            "verdict": self.verdict.value,
        }
        if self.matched_tool_call:
            d["matched_tool_call"] = {
                "name": self.matched_tool_call.name,
                "arguments": self.matched_tool_call.arguments,
            }
        if self.pair_score is not None:
            d["pair_score"] = round(self.pair_score, 4)
        # Include top candidates with scores for debugging
        if self.candidates:
            d["top_candidates"] = [
                {
                    "tool_call": {
                        "name": c.tool_call.name,
                        "arguments": c.tool_call.arguments,
                    },
                    "similarity": round(c.similarity, 4),
                    "verdict": c.verdict.value,
                }
                for c in self.candidates
            ]
        return d


@dataclass
class TurnEvaluation:
    turn_index: int
    is_noise_turn: bool
    expected_facts: list[str]
    tool_calls_made: list[ToolCall]
    fact_matches: list[FactMatch] = field(default_factory=list)
    false_positive_calls: int = 0
    schema_valid_calls: int = 0
    schema_invalid_calls: int = 0

    def to_dict(self) -> dict:
        d: dict = {
            "turn_index": self.turn_index,
            "is_noise_turn": self.is_noise_turn,
        }
        if self.expected_facts:
            d["expected_facts"] = self.expected_facts
        if self.tool_calls_made:
            d["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls_made
            ]
        if self.fact_matches:
            d["fact_matches"] = [fm.to_dict() for fm in self.fact_matches]
        if self.false_positive_calls:
            d["false_positive_calls"] = self.false_positive_calls
        d["schema_valid"] = self.schema_valid_calls
        d["schema_invalid"] = self.schema_invalid_calls
        return d


@dataclass
class ConversationEvaluation:
    conversation_id: str
    turn_evaluations: list[TurnEvaluation] = field(default_factory=list)

    @property
    def true_positives(self) -> float:
        """Count true positives with partial matches counted as 0.5."""
        tp = 0.0
        for te in self.turn_evaluations:
            for fm in te.fact_matches:
                if fm.verdict == MatchVerdict.MATCH:
                    tp += 1.0
                elif fm.verdict == MatchVerdict.PARTIAL:
                    tp += 0.5
        return tp

    @property
    def false_negatives(self) -> float:
        """Count false negatives with partial matches counted as 0.5."""
        fn = 0.0
        for te in self.turn_evaluations:
            for fm in te.fact_matches:
                if fm.verdict == MatchVerdict.NO_MATCH:
                    fn += 1.0
                elif fm.verdict == MatchVerdict.PARTIAL:
                    fn += 0.5
        return fn

    @property
    def false_positives(self) -> int:
        return sum(te.false_positive_calls for te in self.turn_evaluations)

    @property
    def noise_turns(self) -> int:
        return sum(1 for te in self.turn_evaluations if te.is_noise_turn)

    @property
    def noise_violations(self) -> int:
        return sum(
            1
            for te in self.turn_evaluations
            if te.is_noise_turn and len(te.tool_calls_made) > 0
        )

    def to_dict(self) -> dict:
        active_turns = [
            te.to_dict() for te in self.turn_evaluations
            if te.tool_calls_made or te.expected_facts
        ]
        return {
            "conversation_id": self.conversation_id,
            "summary": {
                "true_positives": self.true_positives,
                "false_negatives": self.false_negatives,
                "false_positives": self.false_positives,
                "noise_turns": self.noise_turns,
                "noise_violations": self.noise_violations,
                "total_tool_calls": sum(
                    len(te.tool_calls_made) for te in self.turn_evaluations
                ),
            },
            "turns": active_turns,
        }


@dataclass
class EvaluationResult:
    conversation_evaluations: list[ConversationEvaluation] = field(
        default_factory=list
    )

    def metrics(self, dataset_type: GroundTruthType = GroundTruthType.PER_TURN) -> dict:
        """Compute metrics with names that reflect the evaluation type."""
        tp = sum(ce.true_positives for ce in self.conversation_evaluations)
        fn = sum(ce.false_negatives for ce in self.conversation_evaluations)
        fp = sum(ce.false_positives for ce in self.conversation_evaluations)
        noise_turns = sum(ce.noise_turns for ce in self.conversation_evaluations)
        noise_violations = sum(
            ce.noise_violations for ce in self.conversation_evaluations
        )

        total_schema_valid = sum(
            te.schema_valid_calls
            for ce in self.conversation_evaluations
            for te in ce.turn_evaluations
        )
        total_schema_invalid = sum(
            te.schema_invalid_calls
            for ce in self.conversation_evaluations
            for te in ce.turn_evaluations
        )
        total_calls = total_schema_valid + total_schema_invalid

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        noise_resistance = (
            1 - noise_violations / noise_turns if noise_turns > 0 else 1.0
        )
        schema_validity = (
            total_schema_valid / total_calls if total_calls > 0 else 1.0
        )

        # AlpsBench avg_pair_score: mean of matched pair scores only
        pair_scores = []
        for ce in self.conversation_evaluations:
            for te in ce.turn_evaluations:
                for fm in te.fact_matches:
                    # Only include matched pairs (MATCH or PARTIAL)
                    if fm.verdict != MatchVerdict.NO_MATCH and fm.pair_score is not None:
                        pair_scores.append(fm.pair_score)

        avg_pair_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

        # AlpsBench blended score: 60% quality + 40% coverage
        # algo_score_100 = 100 * (0.60 * avg_pair_score + 0.40 * F1)
        alpsbench_score = 0.60 * avg_pair_score + 0.40 * f1 if pair_scores else None

        # Prefix metrics based on dataset type
        if dataset_type == GroundTruthType.PER_TURN:
            metrics_dict = {
                "per_turn_extraction_precision": round(precision, 4),
                "per_turn_extraction_recall": round(recall, 4),
                "per_turn_extraction_f1": round(f1, 4),
                "per_turn_noise_resistance_rate": round(noise_resistance, 4),
                "schema_validity_rate": round(schema_validity, 4),
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "noise_turns": noise_turns,
                "noise_violations": noise_violations,
                "total_tool_calls": total_calls,
            }
            # Add AlpsBench metrics if available
            if alpsbench_score is not None:
                metrics_dict["alpsbench_avg_pair_score"] = round(avg_pair_score, 4)
                metrics_dict["alpsbench_blended_score"] = round(alpsbench_score, 4)
                metrics_dict["alpsbench_matched_pairs"] = len(pair_scores)
            return metrics_dict
        else:  # CUMULATIVE
            metrics_dict = {
                "cumulative_extraction_precision": round(precision, 4),
                "cumulative_extraction_recall": round(recall, 4),
                "cumulative_extraction_f1": round(f1, 4),
                "schema_validity_rate": round(schema_validity, 4),
                # Note: noise_resistance not applicable for cumulative
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "total_tool_calls": total_calls,
            }
            # Add AlpsBench metrics if available
            if alpsbench_score is not None:
                metrics_dict["alpsbench_avg_pair_score"] = round(avg_pair_score, 4)
                metrics_dict["alpsbench_blended_score"] = round(alpsbench_score, 4)
                metrics_dict["alpsbench_matched_pairs"] = len(pair_scores)
            return metrics_dict

    def metrics_by_fact_type(self) -> dict[str, dict]:
        by_type: dict[str, dict[str, float]] = {}
        for ce in self.conversation_evaluations:
            for te in ce.turn_evaluations:
                for fm in te.fact_matches:
                    t = fm.expected_type
                    if t not in by_type:
                        by_type[t] = {"tp": 0.0, "fn": 0.0}
                    if fm.verdict == MatchVerdict.MATCH:
                        by_type[t]["tp"] += 1.0
                    elif fm.verdict == MatchVerdict.PARTIAL:
                        by_type[t]["tp"] += 0.5
                        by_type[t]["fn"] += 0.5
                    elif fm.verdict == MatchVerdict.NO_MATCH:
                        by_type[t]["fn"] += 1.0

        result = {}
        for t, counts in by_type.items():
            tp, fn = counts["tp"], counts["fn"]
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            result[t] = {"recall": round(recall, 4), "tp": tp, "fn": fn}
        return result


def validate_tool_call_schema(tool_call: ToolCall, memory_system: MemorySystem) -> bool:
    tool_defs = {t["name"]: t for t in memory_system.tool_definitions()}
    tool_def = tool_defs.get(tool_call.name)
    if not tool_def:
        return False
    try:
        jsonschema.validate(
            instance=tool_call.arguments,
            schema=tool_def["input_schema"],
        )
        return True
    except jsonschema.ValidationError:
        return False


async def evaluate_conversation_per_turn(
    conversation: Conversation,
    conv_result: ConversationResult,
    memory_system: MemorySystem,
    matcher: FactMatcher,
) -> ConversationEvaluation:
    """Evaluate per-turn ground truth - timing matters."""
    evaluation = ConversationEvaluation(conversation_id=conversation.id)

    turn_result_map = {tr.turn_index: tr for tr in conv_result.turn_results}

    for i, turn in enumerate(conversation.turns):
        tr = turn_result_map.get(i)
        all_tool_calls = tr.tool_calls if tr else []

        # Filter to only storage tool calls (ignore search/retrieval)
        tool_calls = [
            tc for tc in all_tool_calls
            if memory_system.extract_stored_fact(tc)
        ]

        expected_facts = turn.ground_truth.should_store
        is_noise = turn.is_noise

        te = TurnEvaluation(
            turn_index=i,
            is_noise_turn=is_noise,
            expected_facts=[f.fact for f in expected_facts],
            tool_calls_made=tool_calls,
        )

        for tc in tool_calls:
            valid = validate_tool_call_schema(tc, memory_system)
            if valid:
                te.schema_valid_calls += 1
            else:
                te.schema_invalid_calls += 1

        if is_noise:
            te.false_positive_calls = len(tool_calls)
        else:
            matched_calls = set()

            # For embedding matcher, use similarity scores for tiebreaking
            from .matchers.embedding import EmbeddingMatcher
            from .matchers.alpsbench import AlpsBenchMatcher
            use_similarity_scores = isinstance(matcher, EmbeddingMatcher)
            use_alpsbench_scores = isinstance(matcher, AlpsBenchMatcher)

            for ef in expected_facts:
                best_verdict = MatchVerdict.NO_MATCH
                best_tc = None
                best_j = None
                best_similarity = -1.0
                best_pair_score: float | None = None

                # Collect all candidates with scores
                all_candidates: list[tuple[int, ToolCall, float, MatchVerdict]] = []

                # Convert expected fact to matcher input format
                expected_input = _to_matcher_input(ef)

                for j, tc in enumerate(tool_calls):
                    if j in matched_calls:
                        continue
                    stored = memory_system.extract_stored_fact(tc)
                    if not stored:
                        continue
                    verdict = await matcher.match(expected_input, stored)

                    # Get similarity/pair score based on matcher type
                    similarity = -1.0
                    if use_similarity_scores:
                        # Embedding matcher needs strings, not dicts
                        expected_str = ef.fact if isinstance(expected_input, dict) else expected_input
                        stored_str = stored.get("value") if isinstance(stored, dict) else stored
                        embeddings = matcher.model.encode([expected_str, stored_str], normalize_embeddings=True)
                        similarity = float(embeddings[0] @ embeddings[1])
                    elif use_alpsbench_scores:
                        # Get AlpsBench pair score for this match
                        pair_sc = matcher.get_last_pair_score()
                        if pair_sc is not None:
                            similarity = pair_sc

                    # Store this candidate
                    all_candidates.append((j, tc, similarity, verdict))

                    # Update best match if better verdict or same verdict with higher similarity
                    if verdict.score > best_verdict.score or (
                        verdict.score == best_verdict.score and similarity > best_similarity
                    ):
                        best_verdict = verdict
                        best_tc = tc
                        best_j = j
                        best_similarity = similarity
                        if use_alpsbench_scores:
                            best_pair_score = similarity  # Store the pair score
                        if verdict == MatchVerdict.MATCH:
                            break  # MATCH is best possible, stop searching

                # Mark the best match as used
                if best_verdict == MatchVerdict.MATCH and best_j is not None:
                    matched_calls.add(best_j)

                # Sort candidates by similarity and take top 5
                all_candidates.sort(key=lambda x: x[2], reverse=True)
                top_candidates = [
                    MatchCandidate(tool_call=tc, similarity=sim, verdict=v)
                    for (j, tc, sim, v) in all_candidates[:5]
                ]

                te.fact_matches.append(
                    FactMatch(
                        expected_fact=ef.fact,
                        expected_type=ef.type.value,
                        matched_tool_call=best_tc,
                        verdict=best_verdict,
                        candidates=top_candidates,
                        pair_score=best_pair_score,
                    )
                )

            unmatched_calls = len(tool_calls) - len(matched_calls)
            te.false_positive_calls = max(0, unmatched_calls - len(
                [fm for fm in te.fact_matches if fm.verdict == MatchVerdict.NO_MATCH]
            ))

        evaluation.turn_evaluations.append(te)

    return evaluation


async def evaluate_conversation_cumulative(
    conversation: Conversation,
    conv_result: ConversationResult,
    memory_system: MemorySystem,
    matcher: FactMatcher,
) -> ConversationEvaluation:
    """Evaluate cumulative ground truth - timing doesn't matter, only completeness."""

    if not conversation.cumulative_ground_truth:
        raise ValueError(
            f"Cumulative evaluation requires cumulative_ground_truth for conversation {conversation.id}"
        )

    evaluation = ConversationEvaluation(conversation_id=conversation.id)

    # Expected facts (from cumulative ground truth)
    expected_facts = conversation.cumulative_ground_truth.should_store

    # Collect ALL facts stored across entire conversation
    all_stored_facts: list[tuple[ToolCall, str]] = []  # (tool_call, extracted_fact)
    total_tool_calls = 0
    schema_valid = 0
    schema_invalid = 0

    for turn_result in conv_result.turn_results:
        for tool_call in turn_result.tool_calls:
            # Extract fact - only count storage tool calls (not search/retrieval)
            fact = memory_system.extract_stored_fact(tool_call)
            if not fact:
                continue

            total_tool_calls += 1

            # Validate schema
            valid = validate_tool_call_schema(tool_call, memory_system)
            if valid:
                schema_valid += 1
            else:
                schema_invalid += 1

            all_stored_facts.append((tool_call, fact))

    # Match all stored vs all expected (order doesn't matter)
    matched_indices: set[int] = set()
    fact_matches: list[FactMatch] = []

    # For embedding matcher, we need to compute all similarities first to find true best match
    # (not just first PARTIAL when multiple PARTIALs exist)
    from .matchers.embedding import EmbeddingMatcher
    from .matchers.alpsbench import AlpsBenchMatcher
    use_similarity_scores = isinstance(matcher, EmbeddingMatcher)
    use_alpsbench_scores = isinstance(matcher, AlpsBenchMatcher)

    for expected in expected_facts:
        best_verdict = MatchVerdict.NO_MATCH
        best_tc = None
        best_idx = None
        best_similarity = -1.0
        best_pair_score: float | None = None

        # Collect all candidates with scores (for ranking)
        all_candidates: list[tuple[int, ToolCall, float, MatchVerdict]] = []

        # Convert expected fact to matcher input format
        expected_input = _to_matcher_input(expected)

        # Find best match among all stored facts
        for idx, (tool_call, stored_fact) in enumerate(all_stored_facts):
            if idx in matched_indices:
                continue  # Already matched to another expected fact

            verdict = await matcher.match(expected_input, stored_fact)

            # Get similarity/pair score based on matcher type
            similarity = -1.0
            if use_similarity_scores:
                # Embedding matcher needs strings, not dicts
                expected_str = expected.fact if isinstance(expected_input, dict) else expected_input
                stored_str = stored_fact.get("value") if isinstance(stored_fact, dict) else stored_fact
                embeddings = matcher.model.encode([expected_str, stored_str], normalize_embeddings=True)
                similarity = float(embeddings[0] @ embeddings[1])
            elif use_alpsbench_scores:
                # Get AlpsBench pair score for this match
                pair_sc = matcher.get_last_pair_score()
                if pair_sc is not None:
                    similarity = pair_sc

            # Store this candidate
            all_candidates.append((idx, tool_call, similarity, verdict))

            # Update best match if:
            # 1. This verdict is better (MATCH > PARTIAL > NO_MATCH), OR
            # 2. Same verdict but higher similarity score (for embedding matcher)
            if verdict.score > best_verdict.score or (
                verdict.score == best_verdict.score and similarity > best_similarity
            ):
                best_verdict = verdict
                best_tc = tool_call
                best_idx = idx
                best_similarity = similarity
                if use_alpsbench_scores:
                    best_pair_score = similarity  # Store the pair score

        # Mark as matched if we found any match (MATCH or PARTIAL)
        if best_verdict != MatchVerdict.NO_MATCH and best_idx is not None:
            matched_indices.add(best_idx)

        # Sort candidates by similarity and take top 5
        all_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        top_candidates = [
            MatchCandidate(tool_call=tc, similarity=sim, verdict=v)
            for (idx, tc, sim, v) in all_candidates[:5]
        ]

        fact_matches.append(
            FactMatch(
                expected_fact=expected.fact,
                expected_type=expected.type.value,
                matched_tool_call=best_tc,
                verdict=best_verdict,
                pair_score=best_pair_score,
                candidates=top_candidates,  # Top 5 ranked by similarity
            )
        )

    # Create a single turn evaluation representing the whole conversation
    # (cumulative doesn't have per-turn granularity)
    fp_calls = len(all_stored_facts) - len(matched_indices)  # Unmatched stored facts

    te = TurnEvaluation(
        turn_index=0,  # Not meaningful for cumulative
        is_noise_turn=False,
        expected_facts=[f.fact for f in expected_facts],
        tool_calls_made=[tc for tc, _ in all_stored_facts],
        fact_matches=fact_matches,
        false_positive_calls=fp_calls,
        schema_valid_calls=schema_valid,
        schema_invalid_calls=schema_invalid,
    )

    evaluation.turn_evaluations.append(te)

    return evaluation


async def evaluate_dataset_cumulative(
    dataset: Dataset,
    run_result: RunResult,
    memory_system: MemorySystem,
    matcher: FactMatcher,
) -> ConversationEvaluation:
    """Evaluate dataset-level cumulative ground truth across ALL conversations."""

    if not dataset.cumulative_ground_truth:
        raise ValueError("Dataset-level cumulative evaluation requires dataset.cumulative_ground_truth")

    # Collect ALL tool calls from ALL conversations
    all_stored_facts: list[tuple[ToolCall, str]] = []
    total_tool_calls = 0
    schema_valid = 0
    schema_invalid = 0

    for conv_result in run_result.conversation_results:
        for turn_result in conv_result.turn_results:
            for tool_call in turn_result.tool_calls:
                # Extract fact - only count storage tool calls (not search/retrieval)
                fact = memory_system.extract_stored_fact(tool_call)
                if not fact:
                    continue

                total_tool_calls += 1

                # Validate schema
                valid = validate_tool_call_schema(tool_call, memory_system)
                if valid:
                    schema_valid += 1
                else:
                    schema_invalid += 1

                all_stored_facts.append((tool_call, fact))

    # Expected facts from dataset-level ground truth
    expected_facts = dataset.cumulative_ground_truth.should_store

    # Match all stored vs all expected (order doesn't matter)
    matched_indices: set[int] = set()
    fact_matches: list[FactMatch] = []

    # For embedding matcher, use similarity scores for tiebreaking
    from .matchers.embedding import EmbeddingMatcher
    use_similarity_scores = isinstance(matcher, EmbeddingMatcher)

    for expected in expected_facts:
        best_verdict = MatchVerdict.NO_MATCH
        best_tc = None
        best_idx = None
        best_similarity = -1.0

        # Collect all candidates with scores (for ranking)
        all_candidates: list[tuple[int, ToolCall, float, MatchVerdict]] = []

        # Convert expected fact to matcher input format
        expected_input = _to_matcher_input(expected)

        # Find best match among all stored facts
        for idx, (tool_call, stored_fact) in enumerate(all_stored_facts):
            if idx in matched_indices:
                continue  # Already matched to another expected fact

            verdict = await matcher.match(expected_input, stored_fact)

            # For embedding matcher, get actual similarity score
            similarity = -1.0
            if use_similarity_scores:
                # Embedding matcher needs strings, not dicts
                expected_str = expected.fact if isinstance(expected_input, dict) else expected_input
                embeddings = matcher.model.encode([expected_str, stored_fact], normalize_embeddings=True)
                similarity = float(embeddings[0] @ embeddings[1])

            # Store this candidate
            all_candidates.append((idx, tool_call, similarity, verdict))

            # Update best match if better verdict or same verdict with higher similarity
            if verdict.score > best_verdict.score or (
                verdict.score == best_verdict.score and similarity > best_similarity
            ):
                best_verdict = verdict
                best_tc = tool_call
                best_idx = idx
                best_similarity = similarity

        # Mark as matched if we found any match (MATCH or PARTIAL)
        if best_verdict != MatchVerdict.NO_MATCH and best_idx is not None:
            matched_indices.add(best_idx)

        # Sort candidates by similarity and take top 5
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = [
            MatchCandidate(tool_call=tc, similarity=sim, verdict=v)
            for (idx, tc, sim, v) in all_candidates[:5]
        ]

        fact_matches.append(
            FactMatch(
                expected_fact=expected.fact,
                expected_type=expected.type.value,
                matched_tool_call=best_tc,
                verdict=best_verdict,
                candidates=top_candidates,
            )
        )

    # Create a single turn evaluation representing the entire dataset
    fp_calls = len(all_stored_facts) - len(matched_indices)

    te = TurnEvaluation(
        turn_index=0,  # Not meaningful for dataset-level
        is_noise_turn=False,
        expected_facts=[f.fact for f in expected_facts],
        tool_calls_made=[tc for tc, _ in all_stored_facts],
        fact_matches=fact_matches,
        false_positive_calls=fp_calls,
        schema_valid_calls=schema_valid,
        schema_invalid_calls=schema_invalid,
    )

    # Use a synthetic conversation ID to indicate dataset-level evaluation
    evaluation = ConversationEvaluation(conversation_id="dataset-cumulative")
    evaluation.turn_evaluations.append(te)

    return evaluation


async def evaluate_run(
    dataset: Dataset,
    run_result: RunResult,
    matcher: FactMatcher | None = None,
    judge_config: RunConfig | None = None,
) -> EvaluationResult:
    if matcher is None:
        if judge_config is None:
            judge_config = run_result.config
        matcher = LLMMatcher(get_provider(judge_config))

    system_kwargs = {}
    if run_result.config.scenario:
        system_kwargs["scenario"] = run_result.config.scenario
    memory_system = get_system(run_result.config.system, **system_kwargs)

    conv_map = {c.id: c for c in dataset.conversations}
    eval_result = EvaluationResult()

    try:
        # Check if dataset has dataset-level cumulative ground truth
        if dataset.ground_truth_type == GroundTruthType.CUMULATIVE and dataset.cumulative_ground_truth:
            # Dataset-level cumulative: evaluate ALL conversations together
            conv_eval = await evaluate_dataset_cumulative(
                dataset, run_result, memory_system, matcher
            )
            eval_result.conversation_evaluations.append(conv_eval)
        else:
            # Per-turn or per-conversation cumulative: evaluate each conversation separately
            for conv_result in run_result.conversation_results:
                conversation = conv_map.get(conv_result.conversation_id)
                if not conversation:
                    continue

                # Route to correct evaluation function based on dataset type
                if dataset.ground_truth_type == GroundTruthType.PER_TURN:
                    conv_eval = await evaluate_conversation_per_turn(
                        conversation, conv_result, memory_system, matcher
                    )
                else:  # CUMULATIVE (per-conversation)
                    conv_eval = await evaluate_conversation_cumulative(
                        conversation, conv_result, memory_system, matcher
                    )

                eval_result.conversation_evaluations.append(conv_eval)
    finally:
        await matcher.close()

    return eval_result
