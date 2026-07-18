import json
from datetime import UTC, datetime

from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dataset.matrix import matrix_path

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def test_bare_config_collects_all_absent_without_raising(tmp_path):
    ctx = collect_context(write_config(tmp_path), now=NOW)
    assert ctx.manifest is None
    assert ctx.truth_minute is None
    assert ctx.truth_hourly is None
    assert ctx.hourly_matrix is None
    assert ctx.score_frames == {}
    assert ctx.history is None
    assert ctx.latest_forecast is None
    assert ctx.releases == ()
    assert ctx.alignment is None
    assert ctx.drift is None
    assert ctx.observability_states == ()
    assert ctx.observability_history.is_empty()
    assert ctx.runs.is_empty()


def test_corrupted_manifest_loads_as_none(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    (config.dataset.dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert collect_context(config, now=NOW).manifest is None


def test_populated_context_loads_matrix_and_manifest(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    matrix = synthetic_hourly_matrix(days=5)
    matrix.write_parquet(matrix_path(config.dataset.dir, "hourly", "live"))
    (config.dataset.dir / "manifest.json").write_text(
        json.dumps({"fingerprint": "abc", "sources": ["alpha", "beta"]}),
        encoding="utf-8",
    )
    ctx = collect_context(config, now=NOW)
    assert ctx.hourly_matrix is not None
    assert ctx.hourly_matrix.height == matrix.height
    assert ctx.manifest is not None
    assert ctx.manifest["fingerprint"] == "abc"
