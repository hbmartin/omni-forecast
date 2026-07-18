import numpy as np
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.idr import pava_isotonic
from grounded_weather_forecast.blenders.protocol import finalize_quantiles
from grounded_weather_forecast.contracts import TargetKind, hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.probabilistic import pit_from_quantiles
from grounded_weather_forecast.reports.leaderboard import leaderboard

TEMP = hourly_variable("temp_c")
WIND = hourly_variable("wind_speed_ms")


def gaussian_matrix(days=40, sd=1.5, seed=31):
    """Homoscedastic Gaussian errors: EMOS should recover sigma ~ sd."""
    return synthetic_hourly_matrix(days=days, noise_sd=sd, seed=seed)


class TestPava:
    def test_pools_violators(self):
        values = np.array([1.0, 3.0, 2.0, 4.0])
        fitted = pava_isotonic(values)
        assert (np.diff(fitted) >= 0).all()
        assert fitted[1] == pytest.approx(fitted[2]) == pytest.approx(2.5)

    def test_already_isotonic_unchanged(self):
        values = np.array([1.0, 2.0, 3.0])
        assert pava_isotonic(values).tolist() == [1.0, 2.0, 3.0]


class TestFinalizeQuantiles:
    def test_sorts_and_clamps(self):
        crossed = np.array([[3.0, 1.0, -2.0]])
        fixed = finalize_quantiles(crossed, TargetKind.CONTINUOUS, WIND)
        assert fixed.tolist() == [[0.0, 1.0, 3.0]]


class TestEmos:
    def test_recovers_dispersion_and_calibrates(self):
        train = to_supervised_slice(gaussian_matrix(), TEMP)
        emos = get_factory("emos")().fit(train)
        result = emos.predict(train.x)
        assert result.quantiles is not None
        # non-crossing by construction
        assert (np.diff(result.quantiles, axis=1) >= -1e-9).all()
        # the 10-90 interval of a calibrated Gaussian with two averaged
        # sources (noise 1.5 each -> blend sd ~ 1.06) is ~2.56 * sd wide
        width = float(np.nanmean(result.quantiles[:, -2] - result.quantiles[:, 1]))
        assert 1.5 < width < 4.5
        pit = pit_from_quantiles(train.y, result.quantiles, result.quantile_levels)
        coverage = float(np.mean((pit > 0.1) & (pit < 0.9)))
        assert coverage == pytest.approx(0.8, abs=0.08)

    def test_thin_data_degrades_to_point(self):
        thin = synthetic_hourly_matrix(days=1, max_lead=20, seed=31)  # 40 rows < 60
        train = to_supervised_slice(thin, TEMP)
        emos = get_factory("emos")().fit(train)
        assert emos.predict(train.x).quantiles is None


class TestIdr:
    def test_pit_is_calibrated_in_sample(self):
        train = to_supervised_slice(gaussian_matrix(), TEMP)
        idr = get_factory("idr")().fit(train)
        result = idr.predict(train.x)
        assert result.quantiles is not None
        assert (np.diff(result.quantiles, axis=1) >= -1e-9).all()
        pit = pit_from_quantiles(train.y, result.quantiles, result.quantile_levels)
        coverage = float(np.mean((pit > 0.1) & (pit < 0.9)))
        assert coverage == pytest.approx(0.8, abs=0.1)

    def test_monotone_in_the_covariate(self):
        train = to_supervised_slice(gaussian_matrix(), TEMP)
        idr = get_factory("idr")().fit(train)
        result = idr.predict(train.x)
        base = idr._base.predict(train.x).point
        order = np.argsort(base)
        medians = result.quantiles[order, 9]
        assert (np.diff(medians) >= -1e-9).all()


class TestLeaderboardProbabilisticColumns:
    def test_columns_appear_for_quantile_emitters(self, tmp_path):
        config = write_config(
            tmp_path,
            extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
        )
        matrix = gaussian_matrix(days=25)
        request = BacktestRequest(variables=(TEMP,), methods=("equal_weight", "emos"))
        scores = run_backtest(matrix, request, config)
        board = leaderboard(scores)
        emos_rows = board.filter(board["method_id"] == "emos")
        point_rows = board.filter(board["method_id"] == "equal_weight")
        assert emos_rows["crps"].null_count() == 0
        assert emos_rows["coverage80"].null_count() == 0
        assert emos_rows["sharpness"].null_count() == 0
        assert point_rows["crps"].null_count() == point_rows.height
        coverage = emos_rows["coverage80"].to_numpy()
        assert float(np.median(coverage)) == pytest.approx(0.8, abs=0.15)
