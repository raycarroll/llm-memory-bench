from llm_memory_bench.systems import get_system, list_prompt_versions


def test_prompt_versions_are_scoped_to_memory_system() -> None:
    assert list_prompt_versions("simple") == [
        "v3_2026-11_explicit_capture",
        "v1_2024-11",
        "v2_2026-09_grounded",
    ]
    assert list_prompt_versions("memoryhub") == [
        "v5_clbench_capture_2026-11",
        "v1_lazy_2026-08",
        "v2_proactive_2026-11",
        "v3_ben_capture_only_2026-11",
        "v4_ben_clbench_2026-11",
    ]


def test_clbench_prompt_requires_capture_writes() -> None:
    system = get_system("memoryhub", prompt_version="v5_clbench_capture_2026-11")

    capture_prompt = system.system_prompt_for_mode("capture_only")
    replay_prompt = system.system_prompt_for_mode("default")

    assert "call the MemoryHub tool" in capture_prompt
    assert 'action="write"' in capture_prompt
    assert "return the required CL-Bench JSON action" in capture_prompt
    assert "The database may change schema during the run" in replay_prompt


def test_prompt_versions_follow_inherited_system_for_sweep_variant() -> None:
    assert list_prompt_versions("simple_sweep") == list_prompt_versions("simple")
