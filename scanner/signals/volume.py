"""Volume anomaly signal.

A stock quietly trading 400k shares that suddenly prints 5M is the single
most reliable precursor of a large move in either direction; combined with
positive price action it is a classic accumulation footprint. We score:

  * z-score of today's log-volume vs the ticker's own 60-day baseline
  * today's dollar volume as a multiple of its 60-day median

Both are squashed to [0, 1] with a logistic so one wild print can't dominate
the composite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _logistic(x: pd.Series, midpoint: float, steepness: float) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def volume_score(features: pd.DataFrame, cfg: dict) -> pd.Series:
    vcfg = cfg["signals"]["volume"]
    clip = vcfg["zscore_clip"]

    vol_z = features["vol_z"].clip(-clip, clip)
    # z=2 (≈97.7th percentile of the ticker's own history) maps to ~0.5.
    z_component = _logistic(vol_z, midpoint=2.0, steepness=1.2)

    ratio = features["dollar_vol_ratio"].clip(upper=50)
    # 3x median dollar volume maps to ~0.5.
    ratio_component = _logistic(np.log1p(ratio), midpoint=np.log1p(3.0), steepness=2.5)

    score = 0.65 * z_component + 0.35 * ratio_component
    return score.fillna(0.0).rename("volume_score")
