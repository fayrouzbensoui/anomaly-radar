"""Unusual options activity for shortlisted candidates.

Full-chain scanning of the whole market needs paid data; instead we inspect
chains only for the top pre-options candidates, where confirmation matters.
Signals of speculative/informed call buying:

  * call volume today greatly exceeding existing open interest (fresh
    positioning, not closing trades)
  * call/put volume ratio heavily call-tilted
  * activity concentrated in near-dated, out-of-the-money strikes

The result is a bounded bonus added to the composite - options flow
confirms an anomaly; it never creates one on its own.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from .utils import log

EPS = 1e-9


def _chain_stats(tkr: yf.Ticker, expiries: list[str], spot: float) -> dict | None:
    call_vol = put_vol = call_oi = otm_call_vol = 0.0
    for expiry in expiries:
        try:
            chain = tkr.option_chain(expiry)
        except Exception:  # noqa: BLE001 - individual expiries fail routinely
            continue
        calls, puts = chain.calls, chain.puts
        cv = calls["volume"].fillna(0).sum()
        call_vol += cv
        put_vol += puts["volume"].fillna(0).sum()
        call_oi += calls["openInterest"].fillna(0).sum()
        if spot > 0:
            otm = calls[calls["strike"] > spot]
            otm_call_vol += otm["volume"].fillna(0).sum()
    if call_vol <= 0:
        return None
    return {
        "call_volume": float(call_vol),
        "put_volume": float(put_vol),
        "call_vol_oi_ratio": float(call_vol / (call_oi + EPS)),
        "call_put_ratio": float(call_vol / (put_vol + EPS)),
        "otm_call_share": float(otm_call_vol / (call_vol + EPS)),
    }


def options_flow_bonus(
    ranked: pd.DataFrame, cfg: dict
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (bonus series aligned to ranked.index, per-ticker flow details)."""
    ocfg = cfg["options_flow"]
    bonus = pd.Series(0.0, index=ranked.index, name="options_bonus")
    details: dict[str, dict] = {}

    if not ocfg["enabled"]:
        return bonus, pd.DataFrame()

    shortlist = ranked.head(ocfg["shortlist_size"]).index
    log.info("Scanning option chains for %d shortlisted tickers", len(shortlist))

    for ticker in shortlist:
        try:
            tkr = yf.Ticker(ticker)
            expiries = list(tkr.options or [])[: ocfg["max_expiries"]]
            if not expiries:
                continue
            stats = _chain_stats(tkr, expiries, float(ranked.at[ticker, "close"]))
        except Exception:  # noqa: BLE001 - options data is best-effort
            log.debug("Options scan failed for %s", ticker, exc_info=True)
            continue
        if stats is None:
            continue
        details[ticker] = stats

        b = 0.0
        if stats["call_vol_oi_ratio"] >= ocfg["min_call_vol_oi_ratio"]:
            b += 0.6
        if stats["call_put_ratio"] >= 3.0:
            b += 0.25
        if stats["otm_call_share"] >= 0.6:
            b += 0.15
        bonus.at[ticker] = min(b, 1.0) * ocfg["score_bonus_cap"]

    flow_df = pd.DataFrame.from_dict(details, orient="index")
    log.info("Options flow: %d tickers with usable chains, %d flagged",
             len(flow_df), int((bonus > 0).sum()))
    return bonus, flow_df
