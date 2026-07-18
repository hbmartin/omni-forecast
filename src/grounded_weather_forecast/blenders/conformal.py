"""Online conformal intervals: guaranteed coverage under drift.

A compact conformal-PID-style tracker (Angelopoulos, Candès & Tibshirani
2023; SAOCP-adjacent): per (lead bucket x day/night) cell, an online quantile
tracker follows the absolute-residual quantile of the base blend (the P term)
and a slow integrator nudges the radius whenever realized coverage drifts
from target (the I term, which is what recovers coverage through regime
shifts and archive gaps). Distribution-free, ~a dozen floats of state per
cell, and the state serializes for the serve path.

This supersedes plain ACI from the original improvement plan: ACI's single
learning rate either oscillates or lags; tracking the score quantile at a
residual-scaled step plus an explicit coverage integrator removes the knob.

Intervals are symmetric around the base point — the honest first cut; the
EMOS/IDR heads carry asymmetry where the data supports it, and the
leaderboard arbitrates.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.blenders.ewma_grounding import EwmaGroundedBlend
from grounded_weather_forecast.blenders.protocol import (
    finalize_point,
    finalize_quantiles,
)
from grounded_weather_forecast.blenders.registry import BlenderFactory, register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import LeadBucket, buckets_for_product

# target central coverages and the quantile levels they imply
COVERAGES: tuple[float, ...] = (0.5, 0.8, 0.9)
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)
_STEP = 0.05  # quantile-tracker step, in units of the running score scale
_INTEGRATOR_GAIN = 0.02
_INTEGRATOR_SATURATION = 25.0
_SCALE_DECAY = 0.02
_MIN_UPDATES = 20


@dataclass
class _CellState:
    """Radii for each coverage target plus the shared score scale."""

    radii: FloatArray
    integrals: FloatArray
    scale: float = 1.0
    updates: int = 0

    def update(self, score: float) -> None:
        self.updates += 1
        self.scale = (1.0 - _SCALE_DECAY) * self.scale + _SCALE_DECAY * score
        step = _STEP * max(self.scale, 1e-6)
        for index, coverage in enumerate(COVERAGES):
            covered = score <= self.effective_radius(index)
            # P: pinball-gradient quantile tracking of the score distribution
            self.radii[index] += step * (coverage if not covered else coverage - 1.0)
            self.radii[index] = max(self.radii[index], 0.0)
            # I: realized-coverage error accumulates and nudges the radius
            self.integrals[index] = float(
                np.clip(
                    self.integrals[index] + (coverage - float(covered)),
                    -_INTEGRATOR_SATURATION,
                    _INTEGRATOR_SATURATION,
                )
            )

    def effective_radius(self, index: int) -> float:
        integrator = _INTEGRATOR_GAIN * self.integrals[index] * max(self.scale, 1e-6)
        return max(float(self.radii[index]) + integrator, 0.0)

    def ready(self) -> bool:
        return self.updates >= _MIN_UPDATES


def _fresh_cell() -> _CellState:
    return _CellState(
        radii=np.zeros(len(COVERAGES)), integrals=np.zeros(len(COVERAGES))
    )


def _bucket_labels(
    lead_hours: FloatArray, buckets: tuple[LeadBucket, ...]
) -> list[str]:
    labels = []
    for lead in lead_hours:
        for bucket in buckets:
            if bucket.contains(float(lead)):
                labels.append(bucket.label)
                break
        else:
            labels.append("__outside__")
    return labels


def _day_flags(x: ForecastMatrix) -> np.ndarray:
    if "solar_elevation_deg" in x.features.columns:
        elevation = (
            x.features["solar_elevation_deg"].cast(float).fill_null(0.0).to_numpy()
        )
        return elevation > 0.0
    return np.zeros(x.n_rows, dtype=bool)


def _replay_order(x: ForecastMatrix) -> np.ndarray:
    if "issue_time" not in x.features.columns:
        return np.arange(x.n_rows)
    return np.argsort(x.features["issue_time"].to_numpy(), kind="stable")


@dataclass
class Conformal:
    """Wrapper: any base blend plus tracked symmetric interval radii."""

    base_factory: BlenderFactory
    method_id: str = "conformal_gew"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _cells: dict[tuple[str, bool], _CellState] = field(default_factory=dict)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        scores = np.abs(train.y - base_point)
        labels = _bucket_labels(
            train.x.lead_hours, buckets_for_product(train.x.product)
        )
        day = _day_flags(train.x)
        for row in _replay_order(train.x):
            if not np.isfinite(scores[row]):
                continue
            key = (labels[row], bool(day[row]))
            cell = self._cells.setdefault(key, _fresh_cell())
            cell.update(float(scores[row]))
        return self

    def _cell_for(self, label: str, is_day: bool) -> _CellState | None:
        # fall back across day/night before giving up on the bucket
        for key in ((label, is_day), (label, not is_day)):
            cell = self._cells.get(key)
            if cell is not None and cell.ready():
                return cell
        return None

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        labels = _bucket_labels(x.lead_hours, buckets_for_product(x.product))
        day = _day_flags(x)
        radii = np.full((x.n_rows, len(COVERAGES)), np.nan)
        for row in range(x.n_rows):
            cell = self._cell_for(labels[row], bool(day[row]))
            if cell is None:
                continue
            radii[row] = [
                cell.effective_radius(index) for index in range(len(COVERAGES))
            ]
        if not np.isfinite(radii).any():
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        # symmetric grid: (0.05, 0.1, 0.25) mirror (0.95, 0.9, 0.75)
        half = radii[:, ::-1]  # widest first: 0.9, 0.8, 0.5
        quantiles = np.column_stack(
            [
                base_point - half[:, 0],
                base_point - half[:, 1],
                base_point - half[:, 2],
                base_point + half[:, 2],
                base_point + half[:, 1],
                base_point + half[:, 0],
            ]
        )
        return BlendResult(
            point=finalize_point(base_point, self._kind, self._variable),
            quantiles=finalize_quantiles(quantiles, self._kind, self._variable),
            quantile_levels=QUANTILE_LEVELS,
        )

    def to_state(self) -> dict[str, object]:
        return {
            "cells": {
                f"{label}|{'day' if is_day else 'night'}": {
                    "radii": cell.radii.tolist(),
                    "integrals": cell.integrals.tolist(),
                    "scale": cell.scale,
                    "updates": cell.updates,
                }
                for (label, is_day), cell in sorted(self._cells.items())
            },
            "coverages": list(COVERAGES),
        }


def _conformal_gew() -> Blender:
    return Conformal(GroundedEqualWeight, "conformal_gew")


def _conformal_ewma() -> Blender:
    return Conformal(EwmaGroundedBlend, "conformal_ewma")


register("conformal_gew", _conformal_gew)
register("conformal_ewma", _conformal_ewma)
