"""Online expert aggregation: EWA and BOA with sleeping experts and fixed share.

The experts are the grounded sources. Weights update sequentially in
training-row order, separately per lead bucket, because skill orderings differ
by horizon. Two mechanisms make this family the drift specialist of the lineup:

- **Sleeping experts.** A source outside its horizon is simply absent from the
  round: it is neither updated nor penalized, because the reduction assigns a
  sleeping expert the awake mixture's loss, i.e. an update factor of exactly
  one. Ragged provider horizons need no special casing.
- **Fixed share.** After each multiplicative update, a small fraction of the
  awake mass is redistributed uniformly. This floors every expert's weight and
  bounds the weight ratio. Without it the learning rate decays, an expert that
  dominated early accumulates an insurmountable lead, and the aggregator cannot
  follow a regime change — a provider silently swapping its backend model would
  keep its weight forever. With it, the weights track the best *sequence* of
  experts, and recovery takes days rather than never.

``ewa`` uses a known-horizon exponential update. ``boa`` (Bernstein Online
Aggregation) uses a second-order update whose per-expert rate shrinks with
accumulated regret variance, so it adapts faster when losses are volatile.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders.grounding import AffineGrounding
from grounded_weather_forecast.blenders.protocol import finalize_point
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    BlendResult,
    BoolArray,
    FloatArray,
    ForecastMatrix,
    Product,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import bucket_for_product

_MAX_ETA = 0.5
# The drift-vs-average-case knob. The loser's steady-state weight is roughly
# share / (2 * per-round learning rate), so a share this small costs nothing
# against a stationary best expert while still capping the weight ratio, which
# is what lets a recovering source climb back after a regime change.
_SHARE = 0.005
_MIN_AWAKE = 2
_GLOBAL_BUCKET = "__global__"


@dataclass
class _BucketState:
    weights: FloatArray
    regret_variance: FloatArray
    horizon: int
    steps: int = 0


def _uniform(n_experts: int) -> _BucketState:
    return _BucketState(
        weights=np.full(n_experts, 1.0 / max(n_experts, 1)),
        regret_variance=np.full(n_experts, 1e-12),
        horizon=1,
    )


def _awake_weights(weights: FloatArray, awake: BoolArray) -> FloatArray | None:
    """Weights renormalized over the awake set; ``None`` if nobody is awake."""
    if not awake.any():
        return None
    subset = weights[awake]
    total = float(subset.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(subset.shape[0], 1.0 / subset.shape[0])
    return subset / total


@dataclass
class OnlineExperts:
    """Sequential expert aggregation over grounded sources."""

    method_id: str
    scheme: Literal["ewa", "boa"]
    share: float = _SHARE
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _grounding: AffineGrounding = field(default_factory=AffineGrounding)
    _states: dict[str, _BucketState] = field(default_factory=dict)
    _watermark: str | None = None
    _sources: tuple[str, ...] = ()

    def _update_factor(
        self,
        state: _BucketState,
        normalized: FloatArray,
        mixture_loss: float,
        awake: BoolArray,
    ) -> FloatArray:
        k = int(awake.sum())
        match self.scheme:
            case "ewa":
                eta = min(_MAX_ETA, float(np.sqrt(8.0 * np.log(k) / state.horizon)))
                exponent = -eta * (normalized[awake] - mixture_loss)
            case "boa":
                regret = mixture_loss - normalized[awake]
                state.regret_variance[awake] += regret**2
                eta = np.minimum(
                    _MAX_ETA, np.sqrt(np.log(k) / state.regret_variance[awake])
                )
                exponent = eta * regret - (eta * regret) ** 2
        return np.exp(np.clip(exponent, -50.0, 50.0))

    def _step(self, state: _BucketState, losses: FloatArray, awake: BoolArray) -> None:
        """One sequential round; ``losses`` is finite on the awake set.

        The loss scale, the expert count in the learning rate, and the mixture
        all come from the awake set alone, so a source that is asleep (or never
        present at all) cannot perturb the update.
        """
        if int(awake.sum()) < _MIN_AWAKE:
            return
        weights = _awake_weights(state.weights, awake)
        if weights is None:
            return
        state.steps += 1
        # Per-round range normalization keeps losses in [0, 1], the precondition
        # for both regret bounds, without assuming a global loss scale.
        scale = max(float(losses[awake].max()), 1e-9)
        normalized = losses / scale
        mixture_loss = float((weights * normalized[awake]).sum())

        updated = state.weights[awake] * self._update_factor(
            state, normalized, mixture_loss, awake
        )
        mass = float(updated.sum())
        if not np.isfinite(mass) or mass <= 0.0:
            updated = np.full(updated.shape[0], 1.0 / updated.shape[0])
            mass = 1.0
        # Fixed share, applied inside the awake block so the mass relationship
        # to sleeping experts is left untouched.
        shared = (1.0 - self.share) * updated + self.share * mass / updated.shape[0]
        state.weights[awake] = shared
        total = float(state.weights.sum())
        if total > 0.0:  # uniform rescale; relative weights are unchanged
            state.weights /= total

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._grounding = AffineGrounding().fit(train)
        self._replay(train)
        return self

    def advance(self, train: SupervisedSlice) -> Self:
        """True online continuation: consume only rows past the watermark.

        The training matrix IS the pending-loss queue — rows whose truth
        resolved since the last serve are exactly the rows with a newer
        ``issue_time``. Weights carry over; the batch replay of ``fit`` is
        never re-run, so a serve costs O(new rows), not O(history). The
        grounding is re-fit each call (a cheap batch OLS): expert weights are
        the drift-adaptive state worth persisting, grounded corrections are
        not — matching how the online-aggregation literature grounds
        per round.
        """
        self._kind = train.variable.kind
        self._variable = train.variable
        self._grounding = AffineGrounding().fit(train)
        self._replay(train, after=self._watermark)
        return self

    def _replay(self, train: SupervisedSlice, after: str | None = None) -> None:
        self._sources = train.x.sources
        corrected = self._grounding.transform(train.x)
        n_experts = len(train.x.sources)
        buckets = [
            bucket_for_product(train.x.product, lead) or _GLOBAL_BUCKET
            for lead in train.x.lead_hours
        ]
        horizons = Counter(buckets)
        issue = (
            train.x.features["issue_time"].cast(pl.Datetime("us", "UTC")).to_numpy()
            if "issue_time" in train.x.features.columns
            else None
        )
        threshold = np.datetime64(after) if after and issue is not None else None
        order = np.argsort(issue, kind="stable") if issue is not None else None
        rows = order if order is not None else np.arange(train.x.n_rows)
        for row in rows:
            if threshold is not None and issue is not None and issue[row] <= threshold:
                continue
            bucket = buckets[row]
            if bucket not in self._states:
                fresh = _uniform(n_experts)
                fresh.horizon = horizons[bucket]
                self._states[bucket] = fresh
            awake = train.x.availability[row]
            losses = np.where(awake, (corrected[row] - train.y[row]) ** 2, 0.0)
            self._step(self._states[bucket], losses, awake)
        if issue is not None and issue.shape[0]:
            newest = str(np.datetime_as_string(issue.max(), unit="us"))
            if self._watermark is None or newest > self._watermark:
                self._watermark = newest

    def _state_for(self, product: Product, lead: float, n_experts: int) -> _BucketState:
        """Fitted state for a lead, or a uniform default for unseen buckets."""
        state = self._states.get(bucket_for_product(product, lead) or _GLOBAL_BUCKET)
        if state is None or state.weights.shape[0] != n_experts:
            return _uniform(n_experts)
        return state

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        n_experts = len(x.sources)
        point = np.full(x.n_rows, np.nan)
        for row in range(x.n_rows):
            awake = x.availability[row]
            state = self._state_for(x.product, float(x.lead_hours[row]), n_experts)
            weights = _awake_weights(state.weights, awake)
            if weights is None:
                continue
            point[row] = float((weights * corrected[row][awake]).sum())
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def bucket_weights(self, bucket_label: str) -> FloatArray | None:
        """Final normalized weights for one bucket (inspection and tests)."""
        state = self._states.get(bucket_label)
        if state is None:
            return None
        return _awake_weights(
            state.weights, np.ones(state.weights.shape[0], dtype=bool)
        )

    def to_state(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "share": self.share,
            "watermark": self._watermark,
            "sources": list(self._sources),
            "buckets": {
                label: {
                    "weights": state.weights.tolist(),
                    "regret_variance": state.regret_variance.tolist(),
                    "horizon": state.horizon,
                    "steps": state.steps,
                }
                for label, state in self._states.items()
            },
            "grounding": self._grounding.to_state(),
        }

    @classmethod
    def from_state(cls, state: dict[str, object], method_id: str) -> "OnlineExperts":
        """Rehydrate persisted weights; grounding re-fits on the next advance."""
        match state.get("scheme"):
            case "ewa":
                scheme: Literal["ewa", "boa"] = "ewa"
            case "boa":
                scheme = "boa"
            case other:
                msg = f"unknown expert scheme in state: {other!r}"
                raise ValueError(msg)
        experts = cls(method_id=method_id, scheme=scheme)
        share = state.get("share")
        if isinstance(share, (int, float)):
            experts.share = float(share)
        watermark = state.get("watermark")
        if isinstance(watermark, str):
            experts._watermark = watermark
        sources = state.get("sources")
        if isinstance(sources, list):
            experts._sources = tuple(str(s) for s in sources)
        buckets = state.get("buckets")
        if isinstance(buckets, dict):
            for label, raw in buckets.items():
                if not isinstance(raw, dict):
                    continue
                horizon = raw.get("horizon", 1)
                steps = raw.get("steps", 0)
                experts._states[str(label)] = _BucketState(
                    weights=np.asarray(raw.get("weights", []), dtype=np.float64),
                    regret_variance=np.asarray(
                        raw.get("regret_variance", []), dtype=np.float64
                    ),
                    horizon=int(horizon) if isinstance(horizon, (int, float)) else 1,
                    steps=int(steps) if isinstance(steps, (int, float)) else 0,
                )
        return experts


def _ewa() -> OnlineExperts:
    return OnlineExperts(method_id="ewa", scheme="ewa")


def _boa() -> OnlineExperts:
    return OnlineExperts(method_id="boa", scheme="boa")


register("ewa", _ewa)
register("boa", _boa)
