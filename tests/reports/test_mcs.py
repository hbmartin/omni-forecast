from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.reports.leaderboard import leaderboard, slice_winners
from grounded_weather_forecast.reports.mcs import (
    collapsed_loss_matrix,
    model_confidence_set,
)
from grounded_weather_forecast.serve.selection import Selection, apply_live_gate
from grounded_weather_forecast.timeutil import utc


def losses_matrix(n=120, n_methods=10, best=None, seed=5):
    rng = np.random.default_rng(seed)
    losses = 1.0 + rng.normal(0.0, 0.3, size=(n, n_methods))
    if best is not None:
        losses[:, best] -= 0.5  # genuinely dominant
    return np.abs(losses)


class TestModelConfidenceSet:
    def test_equal_methods_mostly_survive(self):
        """Winner's-curse guard: near-equal methods stay indistinguishable."""
        ids = tuple(f"m{i}" for i in range(10))
        result = model_confidence_set(losses_matrix(), ids, alpha=0.1)
        assert len(result.survivors) >= 8

    def test_dominant_method_prunes_the_field(self):
        ids = tuple(f"m{i}" for i in range(10))
        result = model_confidence_set(losses_matrix(best=3), ids, alpha=0.1)
        assert "m3" in result.survivors
        assert len(result.survivors) < 5

    def test_thin_data_eliminates_nothing(self):
        ids = ("a", "b", "c")
        result = model_confidence_set(losses_matrix(n=5)[:, :3], ids)
        assert result.survivors == ids


def scores_frame(gap: float, n_times=80, seed=9):
    """Two methods with independent errors; ``gap`` shrinks the challenger's.

    Independent noise streams matter: a deterministic per-case edge would be
    genuinely significant at any size, which is not a "near tie".
    """
    start = utc(2026, 3, 1)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_times):
        valid = start + timedelta(hours=i)
        issue = valid - timedelta(hours=30)
        truth = 10.0
        challenger_noise = rng.normal(0.0, 1.0 - gap)
        reference_noise = rng.normal(0.0, 1.0)
        rows.append(("challenger", issue, valid, truth + challenger_noise, truth))
        rows.append(("equal_weight", issue, valid, truth + reference_noise, truth))
    return pl.DataFrame(
        {
            "method_id": [r[0] for r in rows],
            "issue_time": [r[1] for r in rows],
            "valid_time": [r[2] for r in rows],
            "y_pred": [r[3] for r in rows],
            "y_true": [r[4] for r in rows],
            "product": ["hourly"] * len(rows),
            "variable": ["temp_c"] * len(rows),
            "lead_bucket": ["24-48h"] * len(rows),
            "lead_hours": [30.0] * len(rows),
        }
    )


class TestMcsPromotionGate:
    def test_clear_winner_promotes(self):
        scores = scores_frame(gap=0.5)
        board = leaderboard(scores)
        winners = slice_winners(board, scores=scores, rule="mcs", alpha=0.1)
        assert winners["method_id"][0] == "challenger"

    def test_near_tie_keeps_the_reference(self):
        # seed 10: the challenger leads on sample MAE (the winner's-curse
        # configuration) yet the MCS keeps both, so the reference serves
        scores = scores_frame(gap=0.0, seed=10)
        board = leaderboard(scores)
        ranked = board.sort("mae")
        assert ranked["method_id"][0] == "challenger"  # lucky on the sample
        winners = slice_winners(board, scores=scores, rule="mcs", alpha=0.1)
        assert winners["method_id"][0] == "equal_weight"

    def test_collapsed_matrix_shape(self):
        scores = scores_frame(gap=0.1, n_times=20)
        built = collapsed_loss_matrix(scores)
        assert built is not None
        matrix, methods = built
        assert matrix.shape == (20, 2)
        assert methods == ("challenger", "equal_weight")


class TestLiveGate:
    def make_selection(self, mae=1.0):
        return {
            ("hourly", "temp_c", "24-48h"): Selection(
                method_id="gbm", reason="won", n=100, mae=mae
            )
        }

    def live_frame(self, live_mae, n=50):
        return pl.DataFrame(
            {
                "product": ["hourly"],
                "variable": ["temp_c"],
                "method_id": ["gbm"],
                "n": [n],
                "live_mae": [live_mae],
                "live_rmse": [live_mae],
                "live_bias": [0.0],
            }
        )

    def test_materially_worse_live_mae_demotes(self):
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            self.live_frame(live_mae=2.0),
            factor=1.5,
            min_n=24,
        )
        selected = gated[("hourly", "temp_c", "24-48h")]
        assert selected.method_id == "equal_weight"
        assert "demoted gbm" in selected.reason

    def test_delivering_method_survives(self):
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            self.live_frame(live_mae=1.1),
            factor=1.5,
            min_n=24,
        )
        assert gated[("hourly", "temp_c", "24-48h")].method_id == "gbm"

    def test_thin_live_evidence_is_ignored(self):
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            self.live_frame(live_mae=5.0, n=5),
            factor=1.5,
            min_n=24,
        )
        assert gated[("hourly", "temp_c", "24-48h")].method_id == "gbm"

    def test_empty_live_frame_is_noop(self):
        selections = self.make_selection()
        assert (
            apply_live_gate(selections, pl.DataFrame(), factor=1.5, min_n=24)
            == selections
        )


class TestPValuesMonotone:
    def test_elimination_p_values_recorded(self):
        ids = tuple(f"m{i}" for i in range(6))
        result = model_confidence_set(losses_matrix(n_methods=6, best=0), ids)
        eliminated = set(ids) - set(result.survivors)
        assert set(result.p_values) == eliminated
        assert all(0.0 <= p < 0.1 for p in result.p_values.values())
