from llm_memory_bench.value.runner import USER_SIMULATOR_PROMPT


def test_user_simulator_prompt_has_private_persona_and_fact_context():
    prompt = USER_SIMULATOR_PROMPT.format(
        persona="A vegetarian user",
        user_facts="- User is vegetarian",
        task_prompt="Suggest dinner",
        outcomes="- vegetarian_recipe: Suggest a vegetarian recipe",
    )

    assert "private persona context" in prompt
    assert "User is vegetarian" in prompt
    assert "Suggest dinner" in prompt
    assert "vegetarian_recipe" in prompt
