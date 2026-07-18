import numpy as np
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.trimmed import trimmed_row_mean
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

TEMP = hourly_variable("temp_c")

FOUR_STEADY_ONE_WILD = ("a", "b", "c", "d", "wild")
STEADY_NOISE = dict.fromkeys(("a", "b", "c", "d"), 0.4) | {"wild": 15.0}


class TestTrimmedRowMean:
    def test_plain_mean_below_three_sources(self):
        values = np.array([[1.0, 3.0], [2.0, np.nan]])
        out = trimmed_row_mean(values, ~np.isnan(values))
        assert out.tolist() == [2.0, 2.0]

    def test_trims_one_extreme_per_side_at_five_sources(self):
        values = np.array([[0.0, 1.0, 2.0, 3.0, 100.0]])
        out = trimmed_row_mean(values, ~np.isnan(values))
        assert out[0] == 2.0

    def test_no_available_sources_is_nan(self):
        values = np.full((1, 4), np.nan)
        out = trimmed_row_mean(values, ~np.isnan(values))
        assert np.isnan(out[0])

    def test_trim_counts_follow_availability_not_width(self):
        values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan]])
        out = trimmed_row_mean(values, ~np.isnan(values))
        assert out[0] == 3.0  # five available: trims 1.0 and 5.0


class TestTrimmedBlenders:
    def test_shrugs_off_a_wild_source(self):
        matrix = synthetic_hourly_matrix(
            days=30, sources=FOUR_STEADY_ONE_WILD, noise_sd=STEADY_NOISE
        )
        train = to_supervised_slice(matrix, TEMP)
        trimmed = get_factory("trimmed_mean")().fit(train).predict(train.x).point
        equal = get_factory("equal_weight")().fit(train).predict(train.x).point
        assert mae(trimmed, train.y) < mae(equal, train.y)

    def test_grounded_variant_also_removes_shared_bias(self):
        matrix = synthetic_hourly_matrix(
            days=30,
            sources=FOUR_STEADY_ONE_WILD,
            biases=dict.fromkeys(FOUR_STEADY_ONE_WILD, 3.0),
            noise_sd=STEADY_NOISE,
        )
        train = to_supervised_slice(matrix, TEMP)
        grounded = (
            get_factory("grounded_trimmed_mean")().fit(train).predict(train.x).point
        )
        raw = get_factory("trimmed_mean")().fit(train).predict(train.x).point
        assert abs(float(np.nanmean(grounded - train.y))) < 0.3
        assert float(np.nanmean(raw - train.y)) > 2.0
