"""Assemble and write the self-contained dashboard page."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from grounded_weather_forecast import __version__
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.dashboard.context import (
    DashboardContext,
    collect_context,
)
from grounded_weather_forecast.dashboard.derive import Derived, derive
from grounded_weather_forecast.dashboard.html import render_page
from grounded_weather_forecast.dashboard.zones import ALL_ZONES
from grounded_weather_forecast.evaluation import dataset_fingerprint
from grounded_weather_forecast.reports.alerts import AlertInputs, evaluate_alerts

_MINUTELY_CAP = 120


def _empty() -> pl.DataFrame:
    return pl.DataFrame()


def _alert_inputs(ctx: DashboardContext, derived: Derived) -> AlertInputs:
    forecast = ctx.latest_forecast
    board = pl.DataFrame()
    if derived.live_stem is not None:
        board = derived.boards[derived.live_stem]
    elif derived.boards:
        board = next(iter(derived.boards.values()))
    return AlertInputs(
        config=ctx.config,
        now=ctx.now,
        manifest=ctx.manifest,
        runs=ctx.runs,
        minute_truth=ctx.truth_minute if ctx.truth_minute is not None else _empty(),
        hourly_truth=ctx.truth_hourly if ctx.truth_hourly is not None else _empty(),
        daily_truth=ctx.truth_daily if ctx.truth_daily is not None else _empty(),
        qc=ctx.qc if ctx.qc is not None else _empty(),
        hourly_matrix=(
            ctx.hourly_matrix if ctx.hourly_matrix is not None else _empty()
        ),
        board=board,
        live_vs_backtest=derived.verification,
        drift=ctx.drift,
        latest_status=(
            (forecast.status, forecast.status_reason) if forecast is not None else None
        ),
        releases=ctx.releases,
        observability_history=ctx.observability_history,
        archive_location=(
            (forecast.latitude, forecast.longitude) if forecast is not None else None
        ),
    )


def _latest_inputs(
    matrix: pl.DataFrame | None,
) -> dict[str, dict[str, dict[str, object]]]:
    if matrix is None or matrix.is_empty() or "issue_time" not in matrix.columns:
        return {}
    newest = matrix.filter(pl.col("issue_time") == matrix["issue_time"].max())
    if "lead_hours" in newest.columns:
        newest = newest.sort("lead_hours")
    row = newest.row(0, named=True)
    inputs: dict[str, dict[str, dict[str, object]]] = {}
    for column in matrix.columns:
        if not column.startswith("fx__"):
            continue
        _prefix, source, variable = column.split("__", 2)
        value = row.get(column)
        if not isinstance(value, (int, float)):
            continue
        age = row.get(f"age__{source}")
        inputs.setdefault(variable, {})[source] = {
            "value": round(float(value), 3),
            "age_hours": round(float(age), 2) if isinstance(age, (int, float)) else None,
        }
    return inputs


def _forecast_payload(ctx: DashboardContext) -> dict[str, object] | None:
    if ctx.latest_forecast is None:
        return None
    try:
        document = json.loads(ctx.latest_forecast.to_json())
    except (TypeError, ValueError):
        return None
    minutely = document.get("minutely")
    if isinstance(minutely, list) and len(minutely) > _MINUTELY_CAP:
        document["minutely"] = minutely[:_MINUTELY_CAP]
    return document


def write_dashboard(config: Config, *, now: datetime | None = None) -> Path:
    """Collect artifacts, evaluate alerts, render, and write dashboard.html."""
    ctx = collect_context(config, now=now)
    derived = derive(ctx)
    zones = tuple(build(ctx, derived) for build in ALL_ZONES)
    alerts = evaluate_alerts(_alert_inputs(ctx, derived))
    charts = {
        panel.panel_id: panel.chart.config
        for zone in zones
        for panel in zone.panels
        if panel.chart is not None
    }
    payload: dict[str, object] = {
        "charts": charts,
        "forecast": _forecast_payload(ctx),
        "latest_inputs": _latest_inputs(ctx.hourly_matrix),
    }
    manifest_print = (
        str(ctx.manifest.get("fingerprint", "unknown"))
        if ctx.manifest is not None
        else dataset_fingerprint(config)
    )
    page = render_page(
        title="grounded-weather-forecast · operator console",
        generated_at=ctx.now.replace(microsecond=0).isoformat(),
        fingerprints={"dataset": manifest_print},
        version=__version__,
        alerts=alerts,
        zones=zones,
        payload=payload,
    )
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    destination = config.reports_dir / "dashboard.html"
    destination.write_text(page, encoding="utf-8")
    return destination


__all__ = ["write_dashboard"]
