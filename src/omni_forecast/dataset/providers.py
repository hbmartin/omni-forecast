"""Read the omni-weather-forecast-apis archive into canonical long frames.

One row per (source, fetched_at, valid_time). Leads are ALWAYS recomputed from
the ISO ``fetched_at`` text: the archive's ``horizon_hours``, ``fetched_at_unix``
and ``run_cycle`` columns are NULL in damaged databases and are never trusted.
Missing tables, empty sources, and NULL fetch timestamps are tolerated.
"""

import sqlite3
from collections.abc import Mapping

import polars as pl

from omni_forecast.config import ForecastsConfig
from omni_forecast.contracts import SourceKind
from omni_forecast.dataset.station import sqlite_uri
from omni_forecast.timeutil import parse_iso_utc_expr

HOURLY_COLUMN_MAP: Mapping[str, str] = {
    "temperature": "temp_c",
    "humidity": "humidity_pct",
    "dew_point": "dew_point_c",
    "wind_speed": "wind_speed_ms",
    "wind_gust": "wind_gust_ms",
    "pressure_sea": "pressure_sea_hpa",
    "precipitation": "precip_mm",
    "precipitation_probability": "pop",
}

DAILY_COLUMN_MAP: Mapping[str, str] = {
    "temperature_max": "temp_max_c",
    "temperature_min": "temp_min_c",
    "precipitation_probability_max": "pop",
    "precipitation_sum": "precip_sum_mm",
}

_SECONDS_PER_HOUR = 3600.0


def source_slug(provider: str, model: str) -> str:
    """One stable slug per forecast stream."""
    if model in (provider, "best_match", ""):
        return provider
    return f"{provider}_{model}"


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _fetch_points(
    connection: sqlite3.Connection, points_table: str, point_columns: list[str]
) -> list[tuple[object, ...]]:
    for required in (points_table, "source_forecasts", "provider_results"):
        if not _table_exists(connection, required):
            return []
    columns = ", ".join(f"p.{column}" for column in point_columns)
    query = (
        f"SELECT pr.run_id, sf.provider, sf.model, pr.fetched_at, {columns} "
        f"FROM {points_table} AS p "
        "JOIN source_forecasts AS sf ON sf.id = p.source_forecast_id "
        "JOIN provider_results AS pr ON pr.id = sf.provider_result_id "
        "WHERE pr.status = 'success'"
    )
    return connection.execute(query).fetchall()


def _base_schema(point_columns: Mapping[str, pl.DataType]) -> dict[str, pl.DataType]:
    return {
        "run_id": pl.Int64(),
        "provider": pl.String(),
        "model": pl.String(),
        "fetched_at_raw": pl.String(),
        **dict(point_columns),
    }


def _with_provenance(frame: pl.DataFrame, forecasts: ForecastsConfig) -> pl.DataFrame:
    slugged = frame.with_columns(
        pl.struct("provider", "model")
        .map_elements(
            lambda row: source_slug(row["provider"], row["model"]),
            return_dtype=pl.String,
        )
        .alias("source"),
        parse_iso_utc_expr(pl.col("fetched_at_raw")).alias("fetched_at"),
        pl.lit(SourceKind.LIVE.value).alias("source_kind"),
    ).drop_nulls("fetched_at")
    if forecasts.sources:
        slugged = slugged.filter(pl.col("source").is_in(list(forecasts.sources)))
    return slugged


def _open(forecasts: ForecastsConfig) -> sqlite3.Connection:
    if not forecasts.db_path.exists():
        msg = f"cannot open forecast archive {forecasts.db_path}: file not found"
        raise OSError(msg)
    return sqlite3.connect(sqlite_uri(forecasts.db_path, immutable=True), uri=True)


def read_hourly_long(forecasts: ForecastsConfig) -> pl.DataFrame:
    """Hourly forecast points as a canonical long frame."""
    point_columns = ["timestamp_unix", *HOURLY_COLUMN_MAP.keys()]
    connection = _open(forecasts)
    try:
        rows = _fetch_points(connection, "hourly_points", point_columns)
    finally:
        connection.close()
    schema = _base_schema(
        {"timestamp_unix": pl.Int64()}
        | {column: pl.Float64() for column in HOURLY_COLUMN_MAP}
    )
    raw = pl.DataFrame(rows, schema=schema, orient="row")
    return (
        _with_provenance(raw.drop_nulls("timestamp_unix"), forecasts)
        .with_columns(
            pl.from_epoch("timestamp_unix", time_unit="s")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
            .alias("valid_time"),
        )
        .with_columns(
            (
                (pl.col("valid_time") - pl.col("fetched_at")).dt.total_seconds()
                / _SECONDS_PER_HOUR
            ).alias("lead_hours")
        )
        .rename(dict(HOURLY_COLUMN_MAP))
        .sort("source", "fetched_at", "valid_time")
        .unique(subset=["source", "fetched_at", "valid_time"], keep="last")
        .sort("source", "fetched_at", "valid_time")
        .select(
            "run_id",
            "source",
            "source_kind",
            "fetched_at",
            "valid_time",
            "lead_hours",
            *HOURLY_COLUMN_MAP.values(),
        )
    )


def read_daily_long(forecasts: ForecastsConfig) -> pl.DataFrame:
    """Daily forecast points as a canonical long frame keyed by forecast date."""
    point_columns = ["forecast_date", *DAILY_COLUMN_MAP.keys()]
    connection = _open(forecasts)
    try:
        rows = _fetch_points(connection, "daily_points", point_columns)
    finally:
        connection.close()
    schema = _base_schema(
        {"forecast_date": pl.String()}
        | {column: pl.Float64() for column in DAILY_COLUMN_MAP}
    )
    raw = pl.DataFrame(rows, schema=schema, orient="row")
    return (
        _with_provenance(raw.drop_nulls("forecast_date"), forecasts)
        .with_columns(pl.col("forecast_date").str.to_date("%Y-%m-%d", strict=False))
        .drop_nulls("forecast_date")
        .rename(dict(DAILY_COLUMN_MAP))
        .sort("source", "fetched_at", "forecast_date")
        .unique(subset=["source", "fetched_at", "forecast_date"], keep="last")
        .sort("source", "fetched_at", "forecast_date")
        .select(
            "run_id",
            "source",
            "source_kind",
            "fetched_at",
            "forecast_date",
            *DAILY_COLUMN_MAP.values(),
        )
    )


def read_minutely_long(forecasts: ForecastsConfig) -> pl.DataFrame:
    """Minutely precipitation points (the only minutely content providers emit)."""
    point_columns = [
        "timestamp_unix",
        "precipitation_intensity",
        "precipitation_probability",
    ]
    connection = _open(forecasts)
    try:
        rows = _fetch_points(connection, "minutely_points", point_columns)
    finally:
        connection.close()
    schema = _base_schema(
        {
            "timestamp_unix": pl.Int64(),
            "precipitation_intensity": pl.Float64(),
            "precipitation_probability": pl.Float64(),
        }
    )
    raw = pl.DataFrame(rows, schema=schema, orient="row")
    return (
        _with_provenance(raw.drop_nulls("timestamp_unix"), forecasts)
        .with_columns(
            pl.from_epoch("timestamp_unix", time_unit="s")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
            .alias("valid_time")
        )
        .rename(
            {
                "precipitation_intensity": "precip_intensity_mmh",
                "precipitation_probability": "pop",
            }
        )
        .sort("source", "fetched_at", "valid_time")
        .unique(subset=["source", "fetched_at", "valid_time"], keep="last")
        .sort("source", "fetched_at", "valid_time")
        .select(
            "run_id",
            "source",
            "source_kind",
            "fetched_at",
            "valid_time",
            "precip_intensity_mmh",
            "pop",
        )
    )


def read_run_completions(forecasts: ForecastsConfig) -> pl.DataFrame:
    """``forecast_runs.completed_at`` instants (snapshot anchors)."""
    connection = _open(forecasts)
    try:
        if not _table_exists(connection, "forecast_runs"):
            return pl.DataFrame(schema={"completed_at": pl.Datetime("us", "UTC")})
        rows = connection.execute(
            "SELECT completed_at FROM forecast_runs WHERE completed_at IS NOT NULL"
        ).fetchall()
    finally:
        connection.close()
    raw = pl.DataFrame(rows, schema={"completed_at_raw": pl.String()}, orient="row")
    return (
        raw.with_columns(
            parse_iso_utc_expr(pl.col("completed_at_raw")).alias("completed_at")
        )
        .drop_nulls("completed_at")
        .select("completed_at")
        .sort("completed_at")
    )
