from llm_memory_bench.value.runner import _should_stop_after_agent


def test_goal_mode_does_not_stop_on_terminal_sounding_response_before_budget():
    assert not _should_stop_after_agent(
        "The recommended approach addresses the request with a concrete sequence of steps and should solve the issue.",
        turn_number=0,
        max_turns=5,
        goal_mode=True,
    )


def test_goal_mode_stops_at_turn_budget():
    assert _should_stop_after_agent(
        "The recommended approach addresses the request with a concrete sequence of steps and should solve the issue.",
        turn_number=4,
        max_turns=5,
        goal_mode=True,
    )


def test_conversational_mode_keeps_terminal_heuristic():
    assert _should_stop_after_agent(
        "The recommended approach addresses the request with a concrete sequence of steps and should solve the issue.",
        turn_number=0,
        max_turns=5,
        goal_mode=False,
    )
