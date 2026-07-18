from datetime import UTC, datetime

from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.scores import scores_path, write_scores
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.dashboard import write_dashboard
from grounded_weather_forecast.dataset.matrix import matrix_path

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def test_cold_config_writes_full_not_yet_dashboard(tmp_path):
    config = write_config(tmp_path)
    path = write_dashboard(config, now=NOW)
    assert path == config.reports_dir / "dashboard.html"
    text = path.read_text(encoding="utf-8")
    for zone_id in "ABCDEFG":
        assert f'id="zone-{zone_id}"' in text
    assert "alert-strip" in text
    assert "not evaluable yet" in text
    assert "dashboard-data" in text
    # idempotent overwrite
    assert write_dashboard(config, now=NOW) == path


def test_real_scores_populate_the_leaderboard_panels(tmp_path):
    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
    )
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 4.0}, noise_sd=0.3)
    matrix.write_parquet(matrix_path(config.dataset.dir, "hourly", "live"))
    request = BacktestRequest(
        variables=(hourly_variable("temp_c"),),
        methods=("equal_weight", "best_provider", "climatology"),
    )
    scores = run_backtest(matrix, request, config)
    assert not scores.is_empty()
    destination = scores_path(config.dataset.dir / "scores", "hourly", "live")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_scores(scores, destination)

    text = write_dashboard(config, now=NOW).read_text(encoding="utf-8")
    assert "Leaderboard — hourly_live" in text
    assert "equal_weight" in text
    assert "chart-d3-scores_hourly_live" in text  # baseline floor canvas
    assert "Provider error correlation" in text
