from llm_memory_bench.demo.results import DemoRunResult


def test_demo_run_result_requires_both_sides_to_pass():
    result = DemoRunResult(
        demo_id="demo@1.0.0", demo_version="1.0.0", provider="openai",
        model="gpt-4o", memory_system="memoryhub", memory_mode="both", sides={},
    )
    assert result.independent_passed() is False
