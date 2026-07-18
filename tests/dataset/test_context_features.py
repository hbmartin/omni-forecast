from datetime import timedelta

import polars as pl
import pytest
from conftest import canonical_minute_frame, make_forecast_db, write_config

from grounded_weather_forecast.blenders.gbm import build_features
from grounded_weather_forecast.contracts import (
    CONTEXT_FEATURE_COLUMNS,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import (
    build_hourly_matrix,
    to_supervised_slice,
)
from grounded_weather_forecast.dataset.providers import read_hourly_long
from grounded_weather_forecast.dataset.truth import truth_hourly
from grounded_weather_forecast.timeutil import utc

ISSUE = utc(2026, 3, 22, 12, 0, 30)
VALID = utc(2026, 3, 22, 18, 0)


def _snapshots(*times):
    return pl.DataFrame(
        {"issue_time": list(times)},
        schema={"issue_time": pl.Datetime("us", "UTC")},
    )


@pytest.fixture
def matrix(tmp_path):
    make_forecast_db(
        tmp_path / "fx.sqlite",
        [
            {
                "completed_at": "2026-03-22T12:00:30+00:00",
                "results": [
                    {
                        "provider": "nws",
                        "fetched_at": "2026-03-22T11:30:00+00:00",
                        "hourly": [(VALID, {"temperature": 10.0})],
                    }
                ],
            }
        ],
    )
    config = write_config(tmp_path, min_hour_coverage=0.1)
    # a clean +0.1 degC/min ramp into the issue instant: trend = 6 degC/h
    ts = [ISSUE - timedelta(minutes=m) for m in range(30, -1, -1)]
    temps = [20.0 + 0.1 * i for i in range(len(ts))]
    minute = canonical_minute_frame(ts, temp_c=temps)
    hourly_truth = truth_hourly(minute, config)
    return build_hourly_matrix(
        read_hourly_long(config.forecasts),
        _snapshots(ISSUE),
        hourly_truth,
        minute,
        config,
    )


class TestContextFeatures:
    def test_matrix_carries_all_context_columns(self, matrix):
        for column in CONTEXT_FEATURE_COLUMNS:
            assert column in matrix.columns, column

    def test_cyclical_wrap(self, matrix):
        row = matrix.row(0, named=True)
        assert row["hour_sin"] ** 2 + row["hour_cos"] ** 2 == pytest.approx(1.0)
        assert row["doy_sin"] ** 2 + row["doy_cos"] ** 2 == pytest.approx(1.0)

    def test_solar_elevation_daytime_at_18z(self, matrix):
        # 18:00 UTC = 11:00 local in March: the sun is well up
        assert matrix.row(0, named=True)["solar_elevation_deg"] > 30.0

    def test_observation_trend_recovers_ramp(self, matrix):
        trend = matrix.row(0, named=True)["obs__temp_c__trend15m"]
        assert trend == pytest.approx(6.0, abs=0.1)

    def test_features_reach_slices_and_gbm(self, matrix):
        slice_ = to_supervised_slice(matrix, hourly_variable("temp_c"))
        for column in CONTEXT_FEATURE_COLUMNS:
            assert column in slice_.x.features.columns
        assert "obs__temp_c__trend15m" in slice_.x.features.columns
        _, names = build_features(slice_.x)
        assert "solar_elevation_deg" in names
        assert "hour_sin" in names
        assert "obs__temp_c__trend15m" in names
