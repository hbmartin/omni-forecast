"""Solar geometry: dependency-free NOAA solar position, vectorized.

Solar elevation is the physically right phase variable for diurnal bias
structure at a fixed site: a bias curve in elevation is approximately
season-invariant (it absorbs most of the diurnal-by-season interaction that
an hour-of-day curve cannot), and every quantity here is a deterministic
function of the valid instant and the fixed site — leakage-safe by
construction, never a function of truth.

Equations follow the NOAA Global Monitoring Division solar calculator
(Fourier-series equation of time and declination), accurate to well under a
degree — far below the sensor and model errors this project deals in. Day of
year comes from the calendar (never ``unix // 86400 mod 365``, which drifts a
day every leap cycle and is two weeks off by the 2020s).
"""

import numpy as np

from grounded_weather_forecast.contracts import FloatArray

_SOLAR_CONSTANT_WM2 = 1361.0
_MINUTES_PER_DAY = 1440.0
_DAYS_PER_YEAR = 365.0


def _day_of_year_and_utc_minutes(
    unix_seconds: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Calendar day-of-year (0-based) and UTC minute-of-day per instant."""
    stamps = np.asarray(unix_seconds, dtype=np.float64).astype("datetime64[s]")
    days = stamps.astype("datetime64[D]")
    years = days.astype("datetime64[Y]")
    day_of_year = (days - years).astype(np.float64)
    utc_minutes = (stamps - days).astype("timedelta64[s]").astype(np.float64) / 60.0
    return day_of_year, utc_minutes


def _fractional_year_rad(
    day_of_year: FloatArray, utc_minutes: FloatArray
) -> FloatArray:
    return (
        2.0
        * np.pi
        / _DAYS_PER_YEAR
        * (day_of_year + (utc_minutes / 60.0 - 12.0) / 24.0)
    )


def _equation_of_time_minutes(gamma: FloatArray) -> FloatArray:
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )


def _declination_rad(gamma: FloatArray) -> FloatArray:
    return (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )


def solar_elevation_deg(
    unix_seconds: FloatArray, latitude: float, longitude: float
) -> FloatArray:
    """Solar elevation angle in degrees (negative below the horizon)."""
    day_of_year, utc_minutes = _day_of_year_and_utc_minutes(unix_seconds)
    gamma = _fractional_year_rad(day_of_year, utc_minutes)
    eqtime = _equation_of_time_minutes(gamma)
    declination = _declination_rad(gamma)
    true_solar_minutes = (utc_minutes + eqtime + 4.0 * longitude) % _MINUTES_PER_DAY
    hour_angle_rad = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = np.deg2rad(latitude)
    cos_zenith = np.sin(latitude_rad) * np.sin(declination) + np.cos(
        latitude_rad
    ) * np.cos(declination) * np.cos(hour_angle_rad)
    return np.rad2deg(np.arcsin(np.clip(cos_zenith, -1.0, 1.0)))


def toa_irradiance_wm2(
    unix_seconds: FloatArray, latitude: float, longitude: float
) -> FloatArray:
    """Top-of-atmosphere horizontal irradiance: the clear-sky ceiling.

    Zero at and below the horizon; includes the ±3.3% annual sun-distance
    correction. A cheap proxy for radiative forcing (radiation-shield error
    models regress on exactly this shape).
    """
    elevation = solar_elevation_deg(unix_seconds, latitude, longitude)
    day_of_year, _ = _day_of_year_and_utc_minutes(unix_seconds)
    distance_factor = 1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / _DAYS_PER_YEAR)
    return (
        _SOLAR_CONSTANT_WM2
        * distance_factor
        * np.clip(np.sin(np.deg2rad(elevation)), 0.0, None)
    )
