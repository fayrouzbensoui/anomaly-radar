"""Multivariate statistical outlier detection (Isolation Forest).

The rule-based signals each look at one dimension. The ML pass looks at the
joint distribution: a ticker whose volume, volatility-adjusted move, relative
strength AND breakout position are simultaneously unusual gets isolated in
very few random splits even if no single dimension is extreme. Scores are
converted to cross-sectional percentiles (higher = more anomalous).

Direction-agnostic by design — the composite layer gates on positive price
action, so crash-anomalies don't pollute the growth list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from ..utils import log

ML_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_accel",
    "vol_z",
    "dollar_vol_ratio",
    "atr_move",
    "dist_52w_high",
    "gap_pct",
    "rs_5d",
    "rs_20d",
]


def ml_outlier_score(features: pd.DataFrame, cfg: dict) -> pd.Series:
    ocfg = cfg["signals"]["ml_outlier"]
    cols = [c for c in ML_FEATURES if c in features.columns]
    X = features[cols].replace([np.inf, -np.inf], np.nan)

    valid = X.dropna()
    if len(valid) < 50:
        log.warning("Too few complete rows (%d) for IsolationForest; skipping", len(valid))
        return pd.Series(0.0, index=features.index, name="ml_outlier_score")

    # RobustScaler (median/IQR) — the whole point is that outliers exist,
    # so mean/std standardization would let them distort the scale.
    Xs = RobustScaler().fit_transform(valid)

    model = IsolationForest(
        n_estimators=ocfg["n_estimators"],
        contamination=ocfg["contamination"],
        random_state=ocfg["random_state"],
        n_jobs=-1,
    )
    model.fit(Xs)
    # score_samples: higher = more normal; negate so higher = more anomalous.
    raw = -model.score_samples(Xs)

    score = pd.Series(0.0, index=features.index, name="ml_outlier_score")
    score.loc[valid.index] = pd.Series(raw, index=valid.index).rank(pct=True)
    return score
