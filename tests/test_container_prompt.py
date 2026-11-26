from pathlib import Path

from llm_memory_bench.real_system_container_extraction_runner import RunConfig, _load_container_system_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_memoryhub_container_config_loads_prompt_snapshot() -> None:
    config = RunConfig.from_yaml(PROJECT_ROOT / "configs/claude-memoryhub-remote.yaml")

    assert config.system_prompt_mode == "append"
    assert config.system_prompt_file is not None
    assert Path(config.system_prompt_file).name == "v1_2026-08.yaml"
    assert "Types of memory" in _load_container_system_prompt(config.system_prompt_file)


def test_container_prompt_yaml_requires_system_prompt(tmp_path: Path) -> None:
    invalid = tmp_path / "prompt.yaml"
    invalid.write_text("version: test\n", encoding="utf-8")

    try:
        _load_container_system_prompt(str(invalid))
    except RuntimeError as exc:
        assert "system_prompt" in str(exc)
    else:
        raise AssertionError("invalid prompt YAML should be rejected")
