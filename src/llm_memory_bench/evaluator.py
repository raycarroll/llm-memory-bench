from __future__ import annotations

from dataclasses import dataclass, field

import jsonschema

from .config import RunConfig, get_provider
from .dataset import Conversation, Dataset
from .matchers.base import FactMatcher, MatchVerdict
from .matchers.llm import LLMMatcher
from .providers.base import ToolCall
from .runner import ConversationResult, RunResult
from .systems import get_system
from .systems.base import MemorySystem


@dataclass
class FactMatch:
    expected_fact: str
    expected_type: str
    matched_tool_call: ToolCall | None
    verdict: MatchVerdict

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
    def true_positives(self) -> int:
        return sum(
            1
            for te in self.turn_evaluations
            for fm in te.fact_matches
            if fm.verdict == MatchVerdict.MATCH
        )

    @property
    def false_negatives(self) -> int:
        return sum(
            1
            for te in self.turn_evaluations
            for fm in te.fact_matches
            if fm.verdict == MatchVerdict.NO_MATCH
        )

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

    def metrics(self) -> dict:
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

        return {
            "extraction_precision": round(precision, 4),
            "extraction_recall": round(recall, 4),
            "extraction_f1": round(f1, 4),
            "noise_resistance_rate": round(noise_resistance, 4),
            "schema_validity_rate": round(schema_validity, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "noise_turns": noise_turns,
            "noise_violations": noise_violations,
            "total_tool_calls": total_calls,
        }

    def metrics_by_fact_type(self) -> dict[str, dict]:
        by_type: dict[str, dict[str, int]] = {}
        for ce in self.conversation_evaluations:
            for te in ce.turn_evaluations:
                for fm in te.fact_matches:
                    t = fm.expected_type
                    if t not in by_type:
                        by_type[t] = {"tp": 0, "fn": 0}
                    if fm.verdict == MatchVerdict.MATCH:
                        by_type[t]["tp"] += 1
                    elif fm.verdict == MatchVerdict.NO_MATCH:
                        by_type[t]["fn"] += 1

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


async def evaluate_conversation(
    conversation: Conversation,
    conv_result: ConversationResult,
    memory_system: MemorySystem,
    matcher: FactMatcher,
) -> ConversationEvaluation:
    evaluation = ConversationEvaluation(conversation_id=conversation.id)

    turn_result_map = {tr.turn_index: tr for tr in conv_result.turn_results}

    for i, turn in enumerate(conversation.turns):
        tr = turn_result_map.get(i)
        tool_calls = tr.tool_calls if tr else []

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

            for ef in expected_facts:
                best_verdict = MatchVerdict.NO_MATCH
                best_tc = None

                for j, tc in enumerate(tool_calls):
                    if j in matched_calls:
                        continue
                    stored = memory_system.extract_stored_fact(tc)
                    if not stored:
                        continue
                    verdict = await matcher.match(ef.fact, stored)
                    if verdict.value > best_verdict.value or (
                        verdict == MatchVerdict.MATCH
                    ):
                        best_verdict = verdict
                        best_tc = tc
                        if verdict == MatchVerdict.MATCH:
                            matched_calls.add(j)
                            break

                te.fact_matches.append(
                    FactMatch(
                        expected_fact=ef.fact,
                        expected_type=ef.type.value,
                        matched_tool_call=best_tc,
                        verdict=best_verdict,
                    )
                )

            unmatched_calls = len(tool_calls) - len(matched_calls)
            te.false_positive_calls = max(0, unmatched_calls - len(
                [fm for fm in te.fact_matches if fm.verdict == MatchVerdict.NO_MATCH]
            ))

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

    memory_system = get_system(run_result.config.system)

    conv_map = {c.id: c for c in dataset.conversations}
    eval_result = EvaluationResult()

    try:
        for conv_result in run_result.conversation_results:
            conversation = conv_map.get(conv_result.conversation_id)
            if not conversation:
                continue

            conv_eval = await evaluate_conversation(
                conversation, conv_result, memory_system, matcher
            )
            eval_result.conversation_evaluations.append(conv_eval)
    finally:
        await matcher.close()

    return eval_result
