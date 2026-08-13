"""Composite scoring and ranking.

Each signal family produces a [0, 1] score; the composite is their weighted
mean (weights in config.yaml, auto-normalized). Because this is a *growth*
scanner, tickers are gated on positive price action — a crash is also an
anomaly, but not one you buy.
"""
from __future__ import annotations

import pandas as pd

from .signals import (
    ml_outlier_score,
    momentum_score,
    relative_strength_score,
    volume_score,
)

SIGNAL_FUNCS = {
    "volume": volume_score,
    "momentum": momentum_score,
    "relative_strength": relative_strength_score,
    "ml_outlier": ml_outlier_score,
}


def score_universe(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    weights = cfg["signals"]["weights"]
    total_w = sum(weights.values())

    scores = pd.DataFrame(index=features.index)
    for name, func in SIGNAL_FUNCS.items():
        scores[f"{name}_score"] = func(features, cfg)

    scores["composite"] = sum(
        scores[f"{name}_score"] * (w / total_w) for name, w in weights.items()
    )

    result = features.join(scores)

    # Growth gate: today green, or a strong week with today at worst flat-ish.
    upward = (result["ret_1d"] > 0) | (
        (result["ret_5d"] > 0.03) & (result["ret_1d"] > -0.01)
    )
    result = result[upward]

    return result.sort_values("composite", ascending=False)
