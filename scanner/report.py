"""Report generation: results.json / results.csv / self-contained report.html.

The HTML dashboard is a single file with inlined CSS, light + dark mode from
one token block, and the ranked table as the primary (accessible) view —
score bars are decoration on top of visible numbers, never the only encoding.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from .utils import PROJECT_ROOT, log

PCT_COLS = ["ret_1d", "ret_5d", "ret_20d", "gap_pct", "dist_52w_high"]


def build_outputs(
    ranked: pd.DataFrame,
    flow: pd.DataFrame,
    names: dict[str, str],
    etf_flags: dict[str, bool],
    cfg: dict,
    universe_size: int,
    scanned: int,
    asof: date,
) -> None:
    rcfg = cfg["report"]
    out_dir = PROJECT_ROOT / rcfg["output_dir"]
    hist_dir = PROJECT_ROOT / rcfg["history_dir"]
    out_dir.mkdir(exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    top = ranked.head(rcfg["top_n"]).copy()
    top["name"] = [names.get(t, "") for t in top.index]
    top["is_etf"] = [etf_flags.get(t, False) for t in top.index]

    records = []
    for ticker, row in top.iterrows():
        rec = {
            "ticker": ticker,
            "name": row["name"],
            "is_etf": bool(row["is_etf"]),
            "close": round(float(row["close"]), 4),
            "final_score": round(float(row["final_score"]), 4),
            "composite": round(float(row["composite"]), 4),
            "options_bonus": round(float(row.get("options_bonus", 0.0)), 4),
            "signals": {
                "volume": round(float(row["volume_score"]), 3),
                "momentum": round(float(row["momentum_score"]), 3),
                "relative_strength": round(float(row["relative_strength_score"]), 3),
                "ml_outlier": round(float(row["ml_outlier_score"]), 3),
            },
            "stats": {
                c: (None if pd.isna(row[c]) else round(float(row[c]), 4))
                for c in PCT_COLS + ["vol_z", "dollar_vol_ratio", "dollar_volume"]
            },
        }
        if not flow.empty and ticker in flow.index:
            rec["options_flow"] = {
                k: round(float(v), 3) for k, v in flow.loc[ticker].items()
            }
        records.append(rec)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": asof.isoformat(),
        "universe_size": universe_size,
        "scanned": scanned,
        "candidates": int(len(ranked)),
        "flagged_options_flow": int((top.get("options_bonus", pd.Series(dtype=float)) > 0).sum()),
        "disclaimer": (
            "Statistical anomaly screen, not investment advice. Anomalies mark "
            "unusual behavior, not guaranteed appreciation. Do your own research."
        ),
        "results": records,
    }

    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    (hist_dir / f"{asof.isoformat()}.json").write_text(json.dumps(payload, indent=1))
    top.reset_index(names="ticker").to_csv(out_dir / "results.csv", index=False)
    (out_dir / "report.html").write_text(_render_html(payload))
    log.info("Outputs written to %s", out_dir)


# ── HTML rendering ────────────────────────────────────────────────────


def _pct(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "–"
    cls = "up" if x > 0 else ("down" if x < 0 else "")
    sign = "+" if (signed and x > 0) else ""
    return f'<span class="{cls}">{sign}{x * 100:.1f}%</span>'


def _bar(value: float, maximum: float = 1.0) -> str:
    width = max(0.0, min(1.0, value / maximum)) * 100
    return (
        f'<div class="bar"><div class="bar-fill" style="width:{width:.1f}%"></div>'
        f'</div><span class="bar-num">{value:.2f}</span>'
    )


def _render_html(p: dict) -> str:
    rows = []
    for i, r in enumerate(p["results"], 1):
        s = r["stats"]
        flow_badge = ""
        if r.get("options_bonus", 0) > 0:
            flow_badge = '<span class="badge">▲ unusual call flow</span>'
        etf_badge = '<span class="tag">ETF</span>' if r["is_etf"] else ""
        rows.append(f"""<tr>
<td class="rank">{i}</td>
<td class="tk"><strong>{escape(r["ticker"])}</strong>{etf_badge}{flow_badge}
  <div class="nm">{escape(r["name"][:60])}</div></td>
<td class="num">${r["close"]:,.2f}</td>
<td class="score">{_bar(r["final_score"])}</td>
<td class="num">{_pct(s["ret_1d"])}</td>
<td class="num">{_pct(s["ret_5d"])}</td>
<td class="num">{_pct(s["ret_20d"])}</td>
<td class="num">{s["vol_z"]:.1f}σ</td>
<td class="num">{s["dollar_vol_ratio"]:.1f}×</td>
<td class="num">{_pct(s["dist_52w_high"], signed=False)}</td>
<td class="mini">{r["signals"]["volume"]:.2f} · {r["signals"]["momentum"]:.2f} · {r["signals"]["relative_strength"]:.2f} · {r["signals"]["ml_outlier"]:.2f}</td>
</tr>""")

    tiles = [
        ("Universe", f"{p['universe_size']:,}", "listed instruments"),
        ("Scanned", f"{p['scanned']:,}", "passed liquidity filter"),
        ("Upward anomalies", f"{p['candidates']:,}", "positive-direction outliers"),
        ("Unusual options flow", str(p["flagged_options_flow"]), "of top candidates"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tile-label">{t}</div>'
        f'<div class="tile-value">{v}</div><div class="tile-sub">{s}</div></div>'
        for t, v, s in tiles
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anomaly Scan — {p["asof"]}</title>
<style>
:root {{
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,.10);
  --accent: #2a78d6; --accent-soft: #cde2fb;
  --up: #006300; --down: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,.10);
    --accent: #3987e5; --accent-soft: #184f95;
    --up: #0ca30c; --down: #e66767;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }}
h1 {{ font-size: 22px; margin: 0 0 2px; }}
.sub {{ color: var(--ink-2); margin-bottom: 22px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
  gap: 12px; margin-bottom: 24px; }}
.tile {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; }}
.tile-label {{ color: var(--ink-2); font-size: 12px; }}
.tile-value {{ font-size: 26px; font-weight: 650; margin: 2px 0; }}
.tile-sub {{ color: var(--muted); font-size: 12px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--surface);
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
th {{ text-align: left; font-size: 11px; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted); padding: 10px 10px;
  border-bottom: 1px solid var(--grid); }}
td {{ padding: 9px 10px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: color-mix(in srgb, var(--accent) 6%, transparent); }}
.rank {{ color: var(--muted); width: 30px; }}
.nm {{ color: var(--ink-2); font-size: 12px; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.up {{ color: var(--up); }} .down {{ color: var(--down); }}
.score {{ min-width: 130px; }}
.bar {{ display: inline-block; width: 80px; height: 8px; background: var(--grid);
  border-radius: 4px; vertical-align: middle; }}
.bar-fill {{ height: 100%; background: var(--accent); border-radius: 4px; }}
.bar-num {{ margin-left: 8px; font-variant-numeric: tabular-nums; }}
.mini {{ color: var(--ink-2); font-size: 12px; white-space: nowrap; }}
.badge {{ margin-left: 8px; font-size: 11px; color: var(--up);
  border: 1px solid currentColor; border-radius: 999px; padding: 1px 8px; }}
.tag {{ margin-left: 8px; font-size: 11px; color: var(--muted);
  border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }}
.note {{ margin-top: 18px; color: var(--muted); font-size: 12px; }}
</style></head><body><div class="wrap">
<h1>Daily Anomaly Scan</h1>
<div class="sub">As of {p["asof"]} · generated {p["generated_at"]} UTC</div>
<div class="tiles">{tile_html}</div>
<table>
<thead><tr><th></th><th>Ticker</th><th style="text-align:right">Close</th>
<th>Score</th><th style="text-align:right">1d</th><th style="text-align:right">5d</th>
<th style="text-align:right">20d</th><th style="text-align:right">Vol z</th>
<th style="text-align:right">$Vol ×</th><th style="text-align:right">vs 52w-hi</th>
<th>Vol · Mom · RS · ML</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="note">{escape(p["disclaimer"])}</p>
</div></body></html>"""
