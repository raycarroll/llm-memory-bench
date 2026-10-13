from llm_memory_bench.value.evaluator import OUTCOME_JUDGE_PROMPT


def test_outcome_prompt_includes_conversation_evidence():
    prompt = OUTCOME_JUDGE_PROMPT.format(
        task_prompt="Suggest dinner",
        response="Here is a vegetarian recipe without garlic.",
        memory_context="None",
        conversation_context="user_sim: I am vegetarian and dislike garlic.",
        outcomes="- personalised_recipe: respect the user's preferences",
    )

    assert "facts revealed by the user" in prompt
    assert "dislike garlic" in prompt
    assert "concrete observations" in prompt
