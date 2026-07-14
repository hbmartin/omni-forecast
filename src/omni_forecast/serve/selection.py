"""Method selection: which blender serves which (product, variable, lead bucket).

The backtest leaderboard is the only thing allowed to declare winners, so
selection reads the persisted scores rather than re-deciding anything. Config
pins override, and any slice with no evidence falls back to a named default —
never to silence.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from omni_forecast.backtest.scores import load_scores
from omni_forecast.config import Config
from omni_forecast.reports.leaderboard import leaderboard, slice_winners

FALLBACK_METHOD = "grounded_equal_weight"


@dataclass(frozen=True, slots=True)
class Selection:
    """The chosen method for one slice, and why."""

    method_id: str
    reason: str
    n: int = 0
    mae: float | None = None


type SelectionMap = Mapping[tuple[str, str, str], Selection]


def _pins(config: Config) -> dict[tuple[str, str], str]:
    """``[predict.methods]`` keys are ``"<product>.<variable>"``."""
    pinned: dict[tuple[str, str], str] = {}
    for key, method_id in config.predict.methods.items():
        product, _, variable = key.partition(".")
        if product and variable:
            pinned[(product, variable)] = method_id
    return pinned


def select_methods(config: Config, scores_dir: Path) -> SelectionMap:
    """Per (product, variable, lead bucket): the method that won its slice."""
    pinned = _pins(config)
    selections: dict[tuple[str, str, str], Selection] = {}
    for path in sorted(scores_dir.glob("scores_*.parquet")):
        scores = load_scores(path)
        if scores.is_empty():
            continue
        winners = slice_winners(leaderboard(scores))
        for row in winners.iter_rows(named=True):
            key = (row["product"], row["variable"], row["lead_bucket"])
            selections[key] = Selection(
                method_id=row["method_id"],
                reason=f"lowest backtest MAE in {path.stem}",
                n=int(row["n"]),
                mae=float(row["mae"]),
            )
    for (product, variable), method_id in pinned.items():
        for key in [k for k in selections if k[:2] == (product, variable)]:
            selections[key] = Selection(method_id, reason="pinned in config")
    return selections


def method_for(
    selections: SelectionMap,
    product: str,
    variable: str,
    lead_bucket: str | None,
    config: Config | None = None,
) -> Selection:
    """The selected method, falling back explicitly when a slice has no scores."""
    if config is not None:
        pinned = _pins(config).get((product, variable))
        if pinned is not None:
            return Selection(pinned, reason="pinned in config")
    if lead_bucket is not None:
        found = selections.get((product, variable, lead_bucket))
        if found is not None:
            return found
    return Selection(FALLBACK_METHOD, reason="no backtest evidence for this slice")


def selection_report(selections: SelectionMap) -> pl.DataFrame:
    rows = [
        {
            "product": product,
            "variable": variable,
            "lead_bucket": bucket,
            "method_id": chosen.method_id,
            "n": chosen.n,
            "mae": chosen.mae,
            "reason": chosen.reason,
        }
        for (product, variable, bucket), chosen in sorted(selections.items())
    ]
    return pl.DataFrame(rows)
