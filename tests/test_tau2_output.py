import json
from datetime import datetime as real_datetime, timezone
from pathlib import Path

import llm_memory_bench.cli as cli
from llm_memory_bench.cli import _write_json_without_overwrite


class _FixedDateTime:
    @classmethod
    def now(cls, tz):
        return real_datetime(2026, 11, 11, 12, 13, 14, tzinfo=timezone.utc)


def test_tau2_output_does_not_overwrite_existing_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "datetime", _FixedDateTime)
    output = tmp_path / "tau2-value-airline.json"
    output.write_text("original")

    written = _write_json_without_overwrite(output, {"run": 1})

    assert written == tmp_path / "tau2-value-airline_20261111-121314.json"
    assert output.read_text() == "original"
    assert json.loads(written.read_text()) == {"run": 1}


def test_tau2_output_increments_until_unique(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "datetime", _FixedDateTime)
    output = tmp_path / "tau2-value-airline.json"
    output.write_text("original")
    (tmp_path / "tau2-value-airline_20261111-121314.json").write_text("previous")

    written = _write_json_without_overwrite(output, {"run": 2})

    assert written == tmp_path / "tau2-value-airline_20261111-121314-2.json"
