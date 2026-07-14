"""Archive of everything this system has itself emitted.

Every served forecast is appended here so it can later be scored against the
truth that arrives afterwards. Backtest skill is an estimate of live skill; the
history is what turns that estimate into a measurement, and it is the only way
to catch a live path that has quietly diverged from the backtested one.
"""

from pathlib import Path

import polars as pl

from omni_forecast.serve.schema import Forecast

HISTORY_SCHEMA: pl.Schema = pl.Schema(
    {
        "issued_at": pl.Datetime("us", "UTC"),
        "product": pl.String(),
        "variable": pl.String(),
        "valid_time": pl.Datetime("us", "UTC"),
        "lead_hours": pl.Float64(),
        "method_id": pl.String(),
        "y_pred": pl.Float64(),
        "dataset_fingerprint": pl.String(),
    }
)

_HOURS_PER_DAY = 24.0


def forecast_to_rows(forecast: Forecast) -> pl.DataFrame:
    """Flatten an emitted document into scoreable rows."""
    rows: list[dict[str, object]] = []
    for point in forecast.hourly:
        for variable, value in point.values.items():
            if value is None:
                continue
            rows.append(
                {
                    "issued_at": forecast.issued_at,
                    "product": "hourly",
                    "variable": variable,
                    "valid_time": point.valid_time,
                    "lead_hours": point.lead_hours,
                    "method_id": point.methods.get(variable, "unknown"),
                    "y_pred": value,
                    "dataset_fingerprint": forecast.dataset_fingerprint,
                }
            )
    for daily in forecast.daily:
        for variable, value in daily.values.items():
            if value is None:
                continue
            rows.append(
                {
                    "issued_at": forecast.issued_at,
                    "product": "daily",
                    "variable": variable,
                    # a daily point is keyed by its local date; store the date's
                    # start so the column stays a single timestamp type
                    "valid_time": f"{daily.date_local}T00:00:00+00:00",
                    "lead_hours": daily.lead_days * _HOURS_PER_DAY,
                    "method_id": daily.methods.get(variable, "unknown"),
                    "y_pred": value,
                    "dataset_fingerprint": forecast.dataset_fingerprint,
                }
            )
    if not rows:
        return pl.DataFrame(schema=HISTORY_SCHEMA)
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("issued_at").str.to_datetime(time_unit="us", time_zone="UTC"),
            pl.col("valid_time").str.to_datetime(time_unit="us", time_zone="UTC"),
        )
        .cast(HISTORY_SCHEMA)
    )


def append_history(forecast: Forecast, path: Path) -> int:
    """Append an emitted forecast; returns the number of rows added."""
    fresh = forecast_to_rows(forecast)
    if fresh.is_empty():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = pl.concat([pl.read_parquet(path), fresh]) if path.exists() else fresh
    combined.write_parquet(path)
    return fresh.height


def load_history(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=HISTORY_SCHEMA)
    return pl.read_parquet(path)
