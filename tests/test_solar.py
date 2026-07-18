from datetime import timedelta

import numpy as np
import pytest

from grounded_weather_forecast.solar import solar_elevation_deg, toa_irradiance_wm2
from grounded_weather_forecast.timeutil import utc

LATITUDE = 34.2768
LONGITUDE = -117.1692


def _epochs(start, minutes):
    return np.array(
        [(start + timedelta(minutes=m)).timestamp() for m in range(minutes)]
    )


class TestSolarElevation:
    def test_equinox_noon_elevation(self):
        """At the March equinox the noon sun stands at 90 - latitude."""
        day = _epochs(utc(2026, 3, 20), 1440)
        elevation = solar_elevation_deg(day, LATITUDE, LONGITUDE)
        assert float(elevation.max()) == pytest.approx(90.0 - LATITUDE, abs=0.6)

    def test_solstice_noon_elevation(self):
        day = _epochs(utc(2026, 6, 21), 1440)
        elevation = solar_elevation_deg(day, LATITUDE, LONGITUDE)
        assert float(elevation.max()) == pytest.approx(90.0 - LATITUDE + 23.44, abs=0.6)

    def test_solar_noon_near_expected_utc_time(self):
        """Solar noon at 117.17°W falls near 19:49 UTC (12:00 - lon/15)."""
        day = _epochs(utc(2026, 3, 20), 1440)
        elevation = solar_elevation_deg(day, LATITUDE, LONGITUDE)
        noon_minute = int(np.argmax(elevation))
        expected = (12.0 - LONGITUDE / 15.0) * 60.0
        assert abs(noon_minute - expected) < 20  # equation of time stays small

    def test_night_is_below_horizon(self):
        midnight = np.array([utc(2026, 3, 20, 9, 49).timestamp()])  # solar midnight
        elevation = solar_elevation_deg(midnight, LATITUDE, LONGITUDE)
        assert float(elevation[0]) < -30.0


class TestToaIrradiance:
    def test_zero_at_night_and_bounded_by_day(self):
        day = _epochs(utc(2026, 7, 1), 1440)
        toa = toa_irradiance_wm2(day, LATITUDE, LONGITUDE)
        assert float(toa.min()) == 0.0
        assert 1000.0 < float(toa.max()) < 1420.0

    def test_january_beats_july_distance_factor(self):
        """Earth is closest to the sun in January: same elevation, more flux."""
        january = toa_irradiance_wm2(
            np.array([utc(2026, 1, 4, 20, 0).timestamp()]), 0.0, LONGITUDE
        )
        july = toa_irradiance_wm2(
            np.array([utc(2026, 7, 4, 20, 0).timestamp()]), 0.0, LONGITUDE
        )
        assert float(january[0]) > float(july[0])
