"""EOD price/volume ingestion via yfinance, batched with an on-disk cache.

The scan needs ~1 year of daily bars for the full US universe (~9-10k
symbols). yfinance handles multi-ticker batches well; we chunk requests,
persist each chunk to parquet keyed by (chunk-hash, as-of date), and skip
re-downloading on reruns the same day - which makes CI retries cheap.
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from .utils import PROJECT_ROOT, log

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _chunk_key(tickers: list[str], asof: date) -> str:
    digest = hashlib.sha1(",".join(tickers).encode()).hexdigest()[:12]
    return f"{asof.isoformat()}_{digest}"


def fetch_history(
    tickers: list[str],
    cfg: dict,
    asof: date | None = None,
) -> dict[str, pd.DataFrame]:
    """Return {ticker: OHLCV DataFrame indexed by date} for tickers with data."""
    dcfg = cfg["data"]
    asof = asof or date.today()
    cache_dir = PROJECT_ROOT / dcfg["cache_dir"]
    cache_dir.mkdir(exist_ok=True)

    chunk_size = dcfg["chunk_size"]
    period_days = dcfg["lookback_days"]
    out: dict[str, pd.DataFrame] = {}

    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for n, chunk in enumerate(chunks, 1):
        cache_file = cache_dir / f"{_chunk_key(chunk, asof)}.parquet"
        if cache_file.exists():
            wide = pd.read_parquet(cache_file)
        else:
            log.info("Downloading chunk %d/%d (%d tickers)", n, len(chunks), len(chunk))
            wide = yf.download(
                tickers=" ".join(chunk),
                period=f"{period_days}d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if wide is None or wide.empty:
                log.warning("Chunk %d returned no data", n)
                continue
            wide.to_parquet(cache_file)
        out.update(_split_wide(wide, chunk))

    log.info("History fetched for %d/%d tickers", len(out), len(tickers))
    return out


def _split_wide(wide: pd.DataFrame, chunk: list[str]) -> dict[str, pd.DataFrame]:
    """Split yfinance's multi-ticker wide frame into per-ticker OHLCV frames."""
    result: dict[str, pd.DataFrame] = {}
    if not isinstance(wide.columns, pd.MultiIndex):
        # Single-ticker chunk: columns are flat.
        if len(chunk) == 1:
            df = wide[[c for c in FIELDS if c in wide.columns]].dropna(how="all")
            if not df.empty:
                result[chunk[0]] = df
        return result

    available = wide.columns.get_level_values(0).unique()
    for ticker in chunk:
        if ticker not in available:
            continue
        df = wide[ticker]
        df = df[[c for c in FIELDS if c in df.columns]].dropna(how="all")
        # Require a valid latest close - dead/halted listings produce all-NaN tails.
        if df.empty or pd.isna(df["Close"].iloc[-1]):
            continue
        result[ticker] = df
    return result


def apply_liquidity_filter(
    history: dict[str, pd.DataFrame], cfg: dict
) -> dict[str, pd.DataFrame]:
    """Drop illiquid, sub-minimum-price and short-history tickers."""
    ucfg, dcfg = cfg["universe"], cfg["data"]
    kept: dict[str, pd.DataFrame] = {}
    for ticker, df in history.items():
        if len(df) < dcfg["min_history_days"]:
            continue
        close = df["Close"].iloc[-1]
        if pd.isna(close) or close < ucfg["min_price"]:
            continue
        dollar_vol = (df["Close"] * df["Volume"]).tail(60).median()
        if pd.isna(dollar_vol) or dollar_vol < ucfg["min_median_dollar_volume"]:
            continue
        kept[ticker] = df
    log.info("Liquidity filter: %d -> %d tickers", len(history), len(kept))
    return kept
