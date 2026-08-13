"""Price momentum / breakout signal.

Four ingredients, each mapped to [0, 1]:

  * ATR-normalized daily move — a +6% day means little for a biotech that
    swings 5% daily, but a lot for a utility; normalizing by the ticker's
    own ATR makes moves comparable across the whole universe.
  * 52-week-high breakout — price within the configured proximity of its
    52w high (or making a new one) with a positive day.
  * Momentum acceleration — this week's 5-day return exceeding the prior
    week's (trend igniting rather than fading).
  * Gap-and-hold — an overnight gap up that closed in the upper part of the
    day's range (gap-and-fade is deliberately not rewarded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_score(features: pd.DataFrame, cfg: dict) -> pd.Series:
    mcfg = cfg["signals"]["momentum"]

    directional = np.sign(features["ret_1d"]).clip(lower=0)  # 1 if up day else 0

    atr_component = (features["atr_move"].clip(upper=6) / 6.0) * directional

    near_high = features["dist_52w_high"] >= -mcfg["high_proximity_pct"]
    breakout_component = (near_high & (features["ret_1d"] > 0)).astype(float)

    accel = features["ret_accel"].clip(lower=0)
    accel_component = (accel / (accel + 0.05)).fillna(0)  # saturating ramp

    gap = features["gap_pct"]
    gap_up = (gap >= mcfg["gap_floor_pct"]).astype(float)
    held = (features["range_pos"] >= 0.6).astype(float)
    gap_component = gap_up * held

    score = (
        0.35 * atr_component
        + 0.25 * breakout_component
        + 0.25 * accel_component
        + 0.15 * gap_component
    )
    return score.fillna(0.0).rename("momentum_score")
