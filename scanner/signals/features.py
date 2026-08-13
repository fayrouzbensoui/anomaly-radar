"""Per-ticker feature extraction from OHLCV history.

Everything downstream (rule-based signals and the ML outlier model) works
from this single cross-sectional feature table, computed once per scan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import log

EPS = 1e-12


def _atr(df: pd.DataFrame, window: int = 14) -> float:
    """Average True Range (as a fraction of close) over `window` days."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    last_close = close.iloc[-1]
    if pd.isna(atr) or last_close <= 0:
        return np.nan
    return float(atr / last_close)


def _ticker_features(df: pd.DataFrame, vol_baseline_days: int) -> dict | None:
    close, volume = df["Close"], df["Volume"]
    if len(close) < 21 or close.iloc[-1] <= 0:
        return None

    ret_1d = close.iloc[-1] / close.iloc[-2] - 1
    ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(close) > 5 else np.nan
    ret_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) > 20 else np.nan
    # Momentum acceleration: this week's 5d return vs the prior week's.
    prior_5d = (
        close.iloc[-6] / close.iloc[-11] - 1 if len(close) > 10 else np.nan
    )

    # Volume: z-score of log-volume against its own trailing baseline
    # (log tames the heavy right tail; excludes today from the baseline).
    log_vol = np.log(volume.clip(lower=1).astype(float))
    baseline = log_vol.iloc[:-1].tail(vol_baseline_days)
    std = baseline.std()
    vol_z = float((log_vol.iloc[-1] - baseline.mean()) / (std + EPS)) if len(baseline) >= 20 else np.nan

    dollar_vol = close * volume
    base_dv = dollar_vol.iloc[:-1].tail(vol_baseline_days).median()
    dollar_vol_ratio = float(dollar_vol.iloc[-1] / (base_dv + EPS)) if base_dv and base_dv > 0 else np.nan

    atr_frac = _atr(df)
    atr_move = float(abs(ret_1d) / (atr_frac + EPS)) if atr_frac and not np.isnan(atr_frac) else np.nan

    high_52w = close.max()
    dist_52w_high = float(close.iloc[-1] / high_52w - 1)

    gap_pct = float(df["Open"].iloc[-1] / close.iloc[-2] - 1)
    day_range = df["High"].iloc[-1] - df["Low"].iloc[-1]
    range_pos = (
        float((close.iloc[-1] - df["Low"].iloc[-1]) / day_range) if day_range > 0 else 0.5
    )

    return {
        "close": float(close.iloc[-1]),
        "ret_1d": float(ret_1d),
        "ret_5d": float(ret_5d),
        "ret_20d": float(ret_20d),
        "ret_accel": float(ret_5d - prior_5d) if not (np.isnan(ret_5d) or np.isnan(prior_5d)) else np.nan,
        "vol_z": vol_z,
        "dollar_vol_ratio": dollar_vol_ratio,
        "dollar_volume": float(dollar_vol.iloc[-1]),
        "atr_move": atr_move,
        "dist_52w_high": dist_52w_high,
        "gap_pct": gap_pct,
        "range_pos": range_pos,
    }


def compute_features(
    history: dict[str, pd.DataFrame], cfg: dict
) -> pd.DataFrame:
    """Build the cross-sectional feature table plus benchmark-relative columns."""
    scfg = cfg["signals"]
    rows: dict[str, dict] = {}
    for ticker, df in history.items():
        try:
            feats = _ticker_features(df, scfg["volume"]["baseline_days"])
        except Exception:  # noqa: BLE001 - one bad ticker must not kill the scan
            log.debug("Feature extraction failed for %s", ticker, exc_info=True)
            continue
        if feats is not None:
            rows[ticker] = feats

    features = pd.DataFrame.from_dict(rows, orient="index")

    # Benchmark-relative returns (relative strength raw material).
    bench = scfg["relative_strength"]["benchmark"]
    if bench in features.index:
        for horizon in scfg["relative_strength"]["horizons"]:
            col = f"ret_{horizon}d"
            features[f"rs_{horizon}d"] = features[col] - features.at[bench, col]
    else:
        log.warning("Benchmark %s missing; relative strength vs cross-median", bench)
        for horizon in scfg["relative_strength"]["horizons"]:
            col = f"ret_{horizon}d"
            features[f"rs_{horizon}d"] = features[col] - features[col].median()

    log.info("Features computed for %d tickers", len(features))
    return features
