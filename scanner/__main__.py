"""Entry point: `python -m scanner [--config path] [--verbose] [--limit N]`."""
from __future__ import annotations

import argparse
import sys
from datetime import date

from . import __version__
from .data import apply_liquidity_filter, fetch_history
from .options_flow import options_flow_bonus
from .report import build_outputs
from .scoring import score_universe
from .signals import compute_features
from .universe import build_universe
from .utils import load_config, log, setup_logging


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scanner", description=__doc__)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--limit", type=int, default=None,
        help="scan only the first N universe symbols (smoke tests)",
    )
    args = ap.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config)
    log.info("Market Anomaly Scanner v%s", __version__)

    instruments = build_universe(cfg)
    names = {i.symbol: i.name for i in instruments}
    etf_flags = {i.symbol: i.is_etf for i in instruments}

    tickers = [i.symbol for i in instruments]
    if args.limit:
        tickers = tickers[: args.limit]
    bench = cfg["signals"]["relative_strength"]["benchmark"]
    if bench not in tickers:
        tickers.append(bench)

    history = fetch_history(tickers, cfg)
    if not history:
        log.error("No price data retrieved — aborting")
        return 1
    history = apply_liquidity_filter(history, cfg)

    features = compute_features(history, cfg)
    ranked = score_universe(features, cfg)
    # The benchmark itself is not a candidate.
    ranked = ranked.drop(index=[bench], errors="ignore")

    bonus, flow = options_flow_bonus(ranked, cfg)
    ranked["options_bonus"] = bonus
    ranked["final_score"] = ranked["composite"] + ranked["options_bonus"]
    ranked = ranked.sort_values("final_score", ascending=False)

    build_outputs(
        ranked=ranked,
        flow=flow,
        names=names,
        etf_flags=etf_flags,
        cfg=cfg,
        universe_size=len(instruments),
        scanned=len(features),
        asof=date.today(),
    )
    top = ranked.head(5)
    log.info(
        "Top candidates: %s",
        ", ".join(f"{t} ({row.final_score:.2f})" for t, row in top.iterrows()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
