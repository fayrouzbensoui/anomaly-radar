"""Build the scan universe from the official Nasdaq Trader symbol directory.

Two pipe-delimited files cover every US-listed instrument:
  * nasdaqlisted.txt - all Nasdaq listings
  * otherlisted.txt  - NYSE / NYSE American / NYSE Arca / BATS / IEX listings
Both are refreshed nightly by Nasdaq and include an ETF flag and a test-issue
flag, which lets us build a clean common-stock + ETF universe without a paid
reference-data feed.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

from .utils import http_get, log

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


@dataclass(frozen=True)
class Instrument:
    symbol: str        # Yahoo-compatible symbol (BRK.B -> BRK-B)
    name: str
    exchange: str
    is_etf: bool


def _parse_symbol_file(text: str, symbol_col: str, exchange: str | None) -> pd.DataFrame:
    """Parse a Nasdaq Trader pipe file, dropping the 'File Creation Time' footer."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
    df = df.rename(columns={symbol_col: "Symbol"})
    if exchange is not None:
        df["Exchange"] = exchange
    return df


def _clean(df: pd.DataFrame, exclude_patterns: list[str]) -> pd.DataFrame:
    df = df[df["Test Issue"].str.strip() == "N"]
    df = df.dropna(subset=["Symbol"])
    sym = df["Symbol"].str.strip()
    # Drop symbols with 5th-letter suffixes for warrants/rights/units and any
    # symbol containing $ (preferred series) — not growth-scan candidates.
    bad_suffix = sym.str.len().ge(5) & sym.str[4:].str.contains(
        "|".join(exclude_patterns), regex=True
    )
    df = df[~bad_suffix & ~sym.str.contains(r"[$]", regex=True)]
    return df


def build_universe(cfg: dict) -> list[Instrument]:
    ucfg = cfg["universe"]
    nasdaq = _clean(
        _parse_symbol_file(http_get(NASDAQ_LISTED), "Symbol", "NASDAQ"),
        ucfg["exclude_patterns"],
    )
    other = _clean(
        _parse_symbol_file(http_get(OTHER_LISTED), "ACT Symbol", None),
        ucfg["exclude_patterns"],
    )
    other["Exchange"] = other["Exchange"].map(
        {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "BATS", "V": "IEX"}
    ).fillna("OTHER")

    instruments: dict[str, Instrument] = {}
    for _, row in pd.concat(
        [
            nasdaq[["Symbol", "Security Name", "Exchange", "ETF"]],
            other[["Symbol", "Security Name", "Exchange", "ETF"]],
        ]
    ).iterrows():
        raw = str(row["Symbol"]).strip()
        # Yahoo uses '-' where Nasdaq Trader uses '.' for share classes.
        yahoo = raw.replace(".", "-")
        is_etf = str(row["ETF"]).strip().upper() == "Y"
        if not ucfg["include_etfs"] and is_etf:
            continue
        instruments[yahoo] = Instrument(
            symbol=yahoo,
            name=str(row["Security Name"]).strip(),
            exchange=str(row["Exchange"]).strip(),
            is_etf=is_etf,
        )

    out = sorted(instruments.values(), key=lambda i: i.symbol)
    log.info(
        "Universe: %d instruments (%d ETFs)",
        len(out),
        sum(i.is_etf for i in out),
    )
    return out
