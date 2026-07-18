"""Adaptive grounding: hour-binned decaying-average bias correction.

The static per-lead-bucket intercept averages the diurnal and seasonal bias
structure away — measured here as grounding *losing* to raw equal weight at
24-96 h — and any fixed training window is seasonally unrepresentative until
the archive spans a year. The operational cure (NCEP since 2006; the NBM per
grid point/projection/element) is a decaying-average bias per cell:

    bias <- (1 - w) * bias + w * (forecast - truth)

keyed by (source, lead bucket, hour-of-day bin). No training window to
choose, seasonal drift tracked automatically, archive gaps survived (state
goes stale and re-converges in ~1/w updates), and the diurnal structure kept
because clock-hour bins never pool midnight with noon.

Cells warm up with count-based shrinkage ``n / (n + n0)`` toward zero
correction, so a thin cell degrades smoothly to the raw forecast instead of
falling off the ``min_rows`` cliff the static fitter has.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import finalize_point, masked_average
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import LeadBucket, buckets_for_product

LEARNING_RATE = 0.05
_WARMUP_N0 = 10.0
_HOURS_PER_BIN = 3
_N_HOUR_BINS = 24 // _HOURS_PER_BIN


def _bucket_indices(
    lead_hours: FloatArray, buckets: tuple[LeadBucket, ...]
) -> np.ndarray:
    """Bucket index per row; -1 where the lead falls outside every bucket."""
    indices = np.full(lead_hours.shape[0], -1, dtype=np.int64)
    for position, bucket in enumerate(buckets):
        mask = (lead_hours >= bucket.lo) & (lead_hours < bucket.hi)
        indices[mask] = position
    return indices


def _hour_bins(x: ForecastMatrix) -> np.ndarray:
    """Hour-of-day bin per row; a single bin when the feature is absent.

    The daily matrix carries no ``valid_hour_local``; there the state
    degrades to one adaptive bias per (source, lead bucket), which is still
    the decaying average — just without diurnal resolution.
    """
    if "valid_hour_local" not in x.features.columns:
        return np.zeros(x.n_rows, dtype=np.int64)
    hours = x.features["valid_hour_local"].cast(int).fill_null(0).to_numpy()
    return np.clip(hours // _HOURS_PER_BIN, 0, _N_HOUR_BINS - 1).astype(np.int64)


def _replay_order(x: ForecastMatrix) -> np.ndarray:
    """Training rows in issue-time order, so the decay follows real time."""
    if "issue_time" not in x.features.columns:
        return np.arange(x.n_rows)
    issue = x.features["issue_time"].to_numpy()
    return np.argsort(issue, kind="stable")


@dataclass
class EwmaBiasGrounding:
    """Decaying-average bias per (source, lead bucket, hour bin)."""

    learning_rate: float = LEARNING_RATE
    _buckets: tuple[LeadBucket, ...] = ()
    _bias: dict[str, np.ndarray] = field(default_factory=dict)
    _count: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, train: SupervisedSlice) -> Self:
        self._buckets = buckets_for_product(train.x.product)
        shape = (len(self._buckets), _N_HOUR_BINS)
        for source in train.x.sources:
            self._bias[source] = np.zeros(shape)
            self._count[source] = np.zeros(shape)
        bucket_index = _bucket_indices(train.x.lead_hours, self._buckets)
        hour_bin = _hour_bins(train.x)
        values, y, availability = train.x.values, train.y, train.x.availability
        w = self.learning_rate
        for row in _replay_order(train.x):
            bucket = bucket_index[row]
            if bucket < 0:
                continue
            cell = (bucket, hour_bin[row])
            for position, source in enumerate(train.x.sources):
                if not availability[row, position]:
                    continue
                error = values[row, position] - y[row]
                self._bias[source][cell] = (1.0 - w) * self._bias[source][
                    cell
                ] + w * error
                self._count[source][cell] += 1.0
        return self

    def transform(self, x: ForecastMatrix) -> FloatArray:
        """Corrected forecasts; unknown sources and unseen cells pass through."""
        corrected = x.values.copy()
        bucket_index = _bucket_indices(x.lead_hours, self._buckets)
        hour_bin = _hour_bins(x)
        usable = bucket_index >= 0
        for position, source in enumerate(x.sources):
            bias = self._bias.get(source)
            count = self._count.get(source)
            if bias is None or count is None:
                continue
            cells = (bucket_index[usable], hour_bin[usable])
            shrinkage = count[cells] / (count[cells] + _WARMUP_N0)
            corrected[usable, position] -= shrinkage * bias[cells]
        return corrected

    def to_state(self) -> dict[str, object]:
        """JSON-serializable state for the artifact store."""
        return {
            "learning_rate": self.learning_rate,
            "bucket_labels": [bucket.label for bucket in self._buckets],
            "sources": {
                source: {
                    "bias": self._bias[source].tolist(),
                    "count": self._count[source].tolist(),
                }
                for source in sorted(self._bias)
            },
        }


@dataclass
class EwmaGroundedBlend:
    """EWMA-grounded sources combined by equal weight or inverse training MAE.

    The inverse-MAE variant weights by each source's overall corrected MAE —
    deliberately global, not per lead bucket, so it stays stable on the thin
    slices where the per-bucket ``inverse_mae`` incumbent gets noisy.
    """

    method_id: str = "ewma_grounded_equal_weight"
    weighting: str = "equal"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _grounding: EwmaBiasGrounding = field(default_factory=EwmaBiasGrounding)
    _weights: FloatArray | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._grounding = EwmaBiasGrounding().fit(train)
        if self.weighting == "inverse_mae":
            corrected = self._grounding.transform(train.x)
            errors = np.abs(corrected - train.y[:, np.newaxis])
            with np.errstate(invalid="ignore"):
                mae_per_source = np.nanmean(
                    np.where(train.x.availability, errors, np.nan), axis=0
                )
            fallback = (
                float(np.nanmax(mae_per_source))
                if np.isfinite(mae_per_source).any()
                else 1.0
            )
            mae_per_source = np.where(
                np.isnan(mae_per_source), fallback, mae_per_source
            )
            self._weights = 1.0 / np.maximum(mae_per_source, 1e-6)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        point = masked_average(corrected, x.availability, self._weights)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))


def _ewma_inverse_mae() -> EwmaGroundedBlend:
    return EwmaGroundedBlend(method_id="ewma_inverse_mae", weighting="inverse_mae")


register("ewma_grounded_equal_weight", EwmaGroundedBlend)
register("ewma_inverse_mae", _ewma_inverse_mae)
