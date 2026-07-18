"""Trimmed-mean blenders: robust centers with zero fitted parameters.

With many correlated sources, estimated combination weights mostly relearn
noise (the forecast-combination puzzle), but a single misbehaving provider —
an outage, a silent backend swap, a unit bug — drags a plain mean. A symmetric
trimmed mean drops the extremes per row and averages the rest: protection
against the one bad source without estimating anything.

- ``trimmed_mean``: trim the raw sources.
- ``grounded_trimmed_mean``: bias-only affine-ground each source first (as
  ``grounded_equal_weight`` does), then trim the corrected values.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.grounding import AffineGrounding
from grounded_weather_forecast.blenders.protocol import finalize_point
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    BlendResult,
    BoolArray,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)

TRIM_FRACTION = 0.2


def trimmed_row_mean(
    values: FloatArray, availability: BoolArray, trim_fraction: float = TRIM_FRACTION
) -> FloatArray:
    """Per-row symmetric trimmed mean over the available sources.

    Drops ``floor(trim_fraction * k)`` values from each end of the row's sorted
    available values, so rows with fewer than three sources degenerate to the
    plain mean. Rows with no available source are NaN.
    """
    filled = np.where(availability, values, np.nan)
    order = np.argsort(filled, axis=1)  # NaN (unavailable) sorts to the end
    sorted_values = np.take_along_axis(filled, order, axis=1)
    counts = availability.sum(axis=1)
    trim = np.floor(trim_fraction * counts).astype(np.int64)
    columns = np.arange(values.shape[1])[np.newaxis, :]
    keep = (columns >= trim[:, np.newaxis]) & (columns < (counts - trim)[:, np.newaxis])
    kept_totals = np.where(keep, sorted_values, 0.0).sum(axis=1)
    kept_counts = keep.sum(axis=1)
    return np.where(kept_counts > 0, kept_totals / np.maximum(kept_counts, 1), np.nan)


@dataclass
class TrimmedMean:
    """Symmetric trimmed mean, optionally over grounded sources."""

    method_id: str = "trimmed_mean"
    ground: bool = False
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _grounding: AffineGrounding | None = field(default=None)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        if self.ground:
            self._grounding = AffineGrounding().fit(train)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        values = self._grounding.transform(x) if self._grounding else x.values
        point = trimmed_row_mean(values, x.availability)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))


def _grounded_trimmed_mean() -> TrimmedMean:
    return TrimmedMean(method_id="grounded_trimmed_mean", ground=True)


register("trimmed_mean", TrimmedMean)
register("grounded_trimmed_mean", _grounded_trimmed_mean)
