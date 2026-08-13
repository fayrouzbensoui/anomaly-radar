"""Synthetic OHLCV fixtures — deterministic, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    days: int = 200,
    start_price: float = 20.0,
    daily_vol: float = 0.015,
    base_volume: int = 800_000,
    seed: int = 0,
    spike_last_day: bool = False,
    spike_return: float = 0.08,
    spike_volume_mult: float = 8.0,
    crash_last_day: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, daily_vol, days)
    if spike_last_day:
        rets[-1] = spike_return
    if crash_last_day:
        rets[-1] = -abs(spike_return)
    close = start_price * np.cumprod(1 + rets)
    open_ = np.empty_like(close)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.002, days - 1))
    if spike_last_day:
        open_[-1] = close[-2] * 1.03  # gap up on the anomaly day
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, days)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, days)))
    volume = rng.lognormal(np.log(base_volume), 0.25, days).astype(int)
    if spike_last_day or crash_last_day:
        volume[-1] = int(volume[:-1].mean() * spike_volume_mult)
    idx = pd.bdate_range(end="2026-08-12", periods=days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def cfg():
    from scanner.utils import load_config

    return load_config()


@pytest.fixture
def synthetic_history():
    """60 quiet tickers + SPY + one planted upward anomaly + one crasher."""
    history = {f"QUIET{i:02d}": make_ohlcv(seed=i) for i in range(60)}
    history["SPY"] = make_ohlcv(seed=999, daily_vol=0.007, base_volume=80_000_000)
    history["ROCKT"] = make_ohlcv(seed=77, spike_last_day=True)
    history["CRASH"] = make_ohlcv(seed=78, crash_last_day=True)
    return history
