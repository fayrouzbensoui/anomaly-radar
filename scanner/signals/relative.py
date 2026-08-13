"""Relative strength vs the market benchmark.

A ticker grinding up while SPY is flat or red is showing genuine demand
rather than beta. We blend benchmark-relative returns over several horizons
(default 1/5/20 days, front-weighted) and convert to a cross-sectional
percentile so the score is comparable day to day regardless of market
regime.
"""
from __future__ import annotations

import pandas as pd


def relative_strength_score(features: pd.DataFrame, cfg: dict) -> pd.Series:
    rcfg = cfg["signals"]["relative_strength"]
    horizons, weights = rcfg["horizons"], rcfg["horizon_weights"]
    total_w = sum(weights)

    blended = sum(
        (features[f"rs_{h}d"].fillna(0)) * (w / total_w)
        for h, w in zip(horizons, weights)
    )
    # Percentile rank across the universe -> already in [0, 1].
    return blended.rank(pct=True).rename("relative_strength_score")
