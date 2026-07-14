"""Emit the three products for right now.

The as-of snapshot machinery is the same code the dataset build uses, with a
single snapshot at ``now``: each source contributes its latest forecast fetched
at or before now and not older than the staleness cap. Blenders are fitted on
the live matrix's scoreable history and applied to the future rows of that
snapshot, so serving and backtesting run through one code path.

The minutely product is the anchored nowcast: the hourly path interpolated to
minutes, plus the current observation's residual against the blend, decayed
with lead. Providers' native minutely precipitation is blended separately,
because it is the only genuinely minute-resolution signal they publish.
"""

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from omni_forecast.blenders.registry import get_factory
from omni_forecast.config import Config
from omni_forecast.contracts import (
    DAILY_VARIABLES,
    HOURLY_VARIABLES,
    BlendResult,
    ForecastMatrix,
    TruthSemantics,
    VariableSpec,
)
from omni_forecast.dataset.matrix import (
    build_daily_matrix,
    build_hourly_matrix,
    build_truth,
    matrix_path,
    matrix_sources,
    to_forecast_matrix,
    to_supervised_slice,
)
from omni_forecast.dataset.providers import (
    read_daily_long,
    read_hourly_long,
    read_minutely_long,
)
from omni_forecast.leads import daily_bucket, hourly_bucket
from omni_forecast.serve.schema import (
    SCHEMA_VERSION,
    DailyPoint,
    Forecast,
    HourlyPoint,
    MinutelyPoint,
)
from omni_forecast.serve.selection import Selection, SelectionMap, method_for

MINUTELY_HORIZON_MINUTES = 60
HOURLY_HORIZON_HOURS = 48
DAILY_HORIZON_DAYS = 10
_OBS_STALENESS = timedelta(minutes=30)
_DEFAULT_TAU_HOURS = 3.0
_MINUTELY_VARIABLES = (
    "temp_c",
    "humidity_pct",
    "dew_point_c",
    "wind_speed_ms",
)


class NoForecastDataError(RuntimeError):
    """No provider forecast is fresh enough to serve from."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything needed to serve one issue time."""

    issue_time: datetime
    hourly: pl.DataFrame
    daily: pl.DataFrame
    minutely: pl.DataFrame
    observation: dict[str, float]
    observation_at: datetime | None


def _latest_observation(
    minute: pl.DataFrame, issue_time: datetime
) -> tuple[dict[str, float], datetime | None]:
    if minute.is_empty():
        return {}, None
    recent = minute.filter(
        (pl.col("ts") <= issue_time) & (pl.col("ts") > issue_time - _OBS_STALENESS)
    ).sort("ts")
    if recent.is_empty():
        return {}, None
    row = recent.row(recent.height - 1, named=True)
    values = {
        key: float(value)
        for key, value in row.items()
        if key != "ts" and isinstance(value, float) and not math.isnan(value)
    }
    return values, row["ts"]


def build_snapshot(config: Config, issue_time: datetime) -> Snapshot:
    """The as-of view of every source at ``issue_time``, plus current truth."""
    minute, hourly_truth, daily_truth = build_truth(config)
    snapshots = pl.DataFrame(
        {"issue_time": [issue_time]},
        schema={"issue_time": pl.Datetime("us", "UTC")},
    )
    hourly_long = read_hourly_long(config.forecasts)
    hourly = build_hourly_matrix(
        hourly_long, snapshots, hourly_truth, minute, config
    ).filter(pl.col("lead_hours") <= HOURLY_HORIZON_HOURS)
    daily = build_daily_matrix(
        read_daily_long(config.forecasts), snapshots, hourly, daily_truth, config
    ).filter(pl.col("lead_days") < DAILY_HORIZON_DAYS)
    minutely_long = read_minutely_long(config.forecasts)
    minutely = minutely_long.filter(
        (pl.col("fetched_at") <= issue_time)
        & (pl.col("valid_time") > issue_time)
        & (
            pl.col("valid_time")
            <= issue_time + timedelta(minutes=MINUTELY_HORIZON_MINUTES)
        )
    )
    observation, observation_at = _latest_observation(minute, issue_time)
    if hourly.is_empty() and daily.is_empty():
        msg = (
            f"no provider forecast within {config.forecasts.max_forecast_age_hours}h "
            f"of {issue_time.isoformat()}"
        )
        raise NoForecastDataError(msg)
    return Snapshot(
        issue_time=issue_time,
        hourly=hourly,
        daily=daily,
        minutely=minutely,
        observation=observation,
        observation_at=observation_at,
    )


def _fit_methods(
    train: pl.DataFrame,
    predict_frame: pl.DataFrame,
    variable: VariableSpec,
    method_ids: set[str],
    *,
    daily: bool,
    semantics: TruthSemantics,
) -> tuple[dict[str, BlendResult], ForecastMatrix] | None:
    """Fit each needed method on history and predict the snapshot's rows."""
    truth_sources = matrix_sources(train)
    if not truth_sources:
        return None
    slice_ = to_supervised_slice(train, variable, daily=daily, semantics=semantics)
    if slice_.x.n_rows == 0:
        return None
    x = to_forecast_matrix(predict_frame, variable, daily=daily, sources=truth_sources)
    results: dict[str, BlendResult] = {}
    for method_id in sorted(method_ids):
        blender = get_factory(method_id)().fit(slice_)
        results[method_id] = blender.predict(x)
    return results, x


def _blend_variable(
    train: pl.DataFrame,
    predict_frame: pl.DataFrame,
    variable: VariableSpec,
    selections: SelectionMap,
    config: Config,
    *,
    daily: bool,
    semantics: TruthSemantics,
    force_method: str | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    """Per row: the prediction of the method selected for that row's bucket."""
    product = "daily" if daily else "hourly"
    buckets: list[str | None] = [
        daily_bucket(lead) if daily else hourly_bucket(lead)
        for lead in (
            predict_frame["lead_days"] if daily else predict_frame["lead_hours"]
        ).to_list()
    ]
    chosen: list[Selection] = [
        Selection(force_method, reason="forced by --method")
        if force_method
        else method_for(selections, product, variable.name, bucket, config)
        for bucket in buckets
    ]
    fitted = _fit_methods(
        train,
        predict_frame,
        variable,
        {c.method_id for c in chosen},
        daily=daily,
        semantics=semantics,
    )
    if fitted is None:
        return None
    results, _ = fitted
    point = np.full(predict_frame.height, np.nan)
    for row, selection in enumerate(chosen):
        point[row] = results[selection.method_id].point[row]
    return point, [c.method_id for c in chosen]


def _finite(value: float) -> float | None:
    return None if math.isnan(value) else round(float(value), 3)


def hourly_product(
    snapshot: Snapshot,
    train: pl.DataFrame,
    selections: SelectionMap,
    config: Config,
    semantics: TruthSemantics,
    force_method: str | None = None,
) -> tuple[list[HourlyPoint], dict[str, np.ndarray]]:
    """Blended hourly path, plus the raw per-variable arrays for anchoring."""
    frame = snapshot.hourly.sort("valid_time")
    blended: dict[str, np.ndarray] = {}
    methods: dict[str, list[str]] = {}
    for variable in HOURLY_VARIABLES:
        result = _blend_variable(
            train,
            frame,
            variable,
            selections,
            config,
            daily=False,
            semantics=semantics,
            force_method=force_method,
        )
        if result is None:
            continue
        blended[variable.name], methods[variable.name] = result
    points = [
        HourlyPoint(
            valid_time=row["valid_time"].isoformat(),
            lead_hours=round(float(row["lead_hours"]), 2),
            lead_bucket=row["lead_bucket"],
            values={name: _finite(values[index]) for name, values in blended.items()},
            methods={name: chosen[index] for name, chosen in methods.items()},
        )
        for index, row in enumerate(frame.iter_rows(named=True))
    ]
    return points, blended


def daily_product(
    snapshot: Snapshot,
    train: pl.DataFrame,
    selections: SelectionMap,
    config: Config,
    force_method: str | None = None,
) -> list[DailyPoint]:
    """Daily hi/lo and PoP as their own supervised targets (decision 8).

    The hourly-blend aggregates ride along as ``ewagg__*`` features inside the
    matrix, so a method that can use them (the GBM) does, and one that cannot
    still sees every provider's native daily value.
    """
    frame = snapshot.daily.sort("forecast_date")
    if frame.is_empty():
        return []
    blended: dict[str, np.ndarray] = {}
    methods: dict[str, list[str]] = {}
    for variable in DAILY_VARIABLES:
        result = _blend_variable(
            train,
            frame,
            variable,
            selections,
            config,
            daily=True,
            semantics=TruthSemantics.INSTANTANEOUS,
            force_method=force_method,
        )
        if result is None:
            continue
        blended[variable.name], methods[variable.name] = result
    return [
        DailyPoint(
            date_local=row["forecast_date"].isoformat(),
            lead_days=int(row["lead_days"]),
            values={name: _finite(values[index]) for name, values in blended.items()},
            methods={name: chosen[index] for name, chosen in methods.items()},
        )
        for index, row in enumerate(frame.iter_rows(named=True))
    ]


def _anchor_weight(minutes_ahead: int, tau_hours: float) -> float:
    weight = math.exp(-(minutes_ahead / 60.0) / tau_hours)
    return 0.0 if weight < 0.05 else weight


def _minutely_precip(snapshot: Snapshot) -> dict[datetime, tuple[float, float]]:
    """Availability-weighted mean of providers' native minutely precipitation."""
    if snapshot.minutely.is_empty():
        return {}
    latest = (
        snapshot.minutely.sort("fetched_at")
        .group_by("source", "valid_time")
        .last()
        .group_by("valid_time")
        .agg(
            pl.col("precip_intensity_mmh").mean().alias("intensity"),
            pl.col("pop").mean().alias("pop"),
        )
    )
    return {
        row["valid_time"]: (row["intensity"], row["pop"])
        for row in latest.iter_rows(named=True)
    }


def minutely_product(
    snapshot: Snapshot, hourly_blend: dict[str, np.ndarray]
) -> list[MinutelyPoint]:
    """The anchored nowcast: hourly path interpolated to minutes + decayed residual."""
    frame = snapshot.hourly.sort("valid_time")
    if frame.is_empty():
        return []
    leads = frame["lead_hours"].to_numpy().astype(float)
    precip = _minutely_precip(snapshot)
    points: list[MinutelyPoint] = []
    for minute in range(1, MINUTELY_HORIZON_MINUTES + 1):
        valid = snapshot.issue_time + timedelta(minutes=minute)
        lead = minute / 60.0
        values: dict[str, float | None] = {}
        for name in _MINUTELY_VARIABLES:
            path = hourly_blend.get(name)
            if path is None or not np.isfinite(path).any():
                continue
            usable = np.isfinite(path)
            interpolated = float(np.interp(lead, leads[usable], path[usable]))
            observed = snapshot.observation.get(name)
            if observed is not None:
                now_blend = float(np.interp(0.0, leads[usable], path[usable]))
                residual = observed - now_blend
                interpolated += _anchor_weight(minute, _DEFAULT_TAU_HOURS) * residual
            values[name] = round(interpolated, 3)
        intensity, pop = precip.get(valid, (None, None))
        points.append(
            MinutelyPoint(
                valid_time=valid.isoformat(),
                minutes_ahead=minute,
                precip_intensity_mmh=intensity,
                pop=pop,
                **values,  # type: ignore[arg-type]
            )
        )
    return points


def _dataset_fingerprint(config: Config) -> str:
    manifest = config.dataset.dir / "manifest.json"
    if not manifest.exists():
        return "unknown"
    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    return str(loaded.get("fingerprint", "unknown"))


def _training_matrix(config: Config, product: str, *, required: bool) -> pl.DataFrame:
    path = matrix_path(config.dataset.dir, product, "live")
    if path.exists():
        return pl.read_parquet(path)
    if required:
        msg = f"missing {path}; run build-dataset before predicting"
        raise NoForecastDataError(msg)
    return pl.DataFrame()


def predict(
    config: Config,
    selections: SelectionMap,
    now: datetime | None = None,
    semantics: TruthSemantics = TruthSemantics.INSTANTANEOUS,
    force_method: str | None = None,
) -> Forecast:
    """Assemble the whole forecast document for one issue time."""
    issue_time = (now or datetime.now(tz=UTC)).replace(second=0, microsecond=0)
    snapshot = build_snapshot(config, issue_time)
    hourly_train = _training_matrix(config, "hourly", required=True)
    daily_train = _training_matrix(config, "daily", required=False)
    hourly, hourly_blend = hourly_product(
        snapshot, hourly_train, selections, config, semantics, force_method
    )
    daily = (
        daily_product(snapshot, daily_train, selections, config, force_method)
        if not daily_train.is_empty()
        else []
    )
    return Forecast(
        schema_version=SCHEMA_VERSION,
        issued_at=issue_time.isoformat(),
        latitude=config.station.latitude,
        longitude=config.station.longitude,
        dataset_fingerprint=_dataset_fingerprint(config),
        sources=list(matrix_sources(snapshot.hourly)),
        observation_at=snapshot.observation_at.isoformat()
        if snapshot.observation_at
        else None,
        minutely=minutely_product(snapshot, hourly_blend),
        hourly=hourly,
        daily=daily,
    )
