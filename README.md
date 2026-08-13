# Market Anomaly Scanner

Daily statistical anomaly screen over every US-listed stock and ETF
(~9–10k instruments from the official Nasdaq Trader symbol directory),
surfacing tickers whose **volume, momentum, relative strength, and joint
statistical profile** look like the early footprint of a rapid move — with an
**unusual options activity** confirmation layer on the top candidates.

> **Not investment advice.** This tool finds statistical anomalies. Anomalies
> mark unusual behavior — pumps, squeezes, and news spikes included — not
> guaranteed appreciation. Always do your own research.

## How it works

```
Nasdaq Trader symbol directory  ──▶  universe (stocks + ETFs, junk filtered)
yfinance (batched, cached)      ──▶  ~1y daily OHLCV per ticker
liquidity filter                ──▶  price ≥ $1, median $vol ≥ $500k
feature extraction              ──▶  returns, gaps, vol z-score, ATR-moves,
                                     52w-high distance, RS vs SPY
4 signal engines                ──▶  volume anomaly · momentum/breakout ·
                                     relative strength · IsolationForest
composite score + growth gate   ──▶  upward anomalies only, ranked
options-flow confirmation       ──▶  call vol/OI, call/put ratio, OTM share
                                     on the top-40 shortlist (bounded bonus)
outputs                         ──▶  output/report.html · results.json ·
                                     results.csv · history/YYYY-MM-DD.json
```

Every run commits its outputs back to the repo, so `output/history/` becomes a
free time series of your daily scans.

## 5-minute setup (recommended: run on GitHub Actions)

1. Create a **new GitHub repository** (public or private) — e.g.
   `market-anomaly-scanner`.
2. Upload the contents of this folder to the repo root (drag-and-drop on
   github.com works: *Add file → Upload files*). Make sure the
   `.github/workflows/daily_scan.yml` path is preserved.
   - If you use the web uploader, hidden folders can't be dragged; create the
     workflow file manually instead: *Add file → Create new file*, name it
     `.github/workflows/daily_scan.yml`, and paste the file's contents.
3. In the repo: **Settings → Actions → General → Workflow permissions** →
   select **Read and write permissions** → Save.
4. Test it: **Actions → Daily Anomaly Scan → Run workflow**. A run takes
   roughly 20–40 minutes (the whole US market is downloaded in batches).
5. Done. The workflow runs automatically every weekday at 22:15 UTC
   (after the US close) and commits fresh results to `output/`.

The latest report is always at:

```
https://raw.githubusercontent.com/<you>/<repo>/main/output/report.html
https://raw.githubusercontent.com/<you>/<repo>/main/output/results.json
```

Tip: enable GitHub Pages (Settings → Pages → deploy from branch, `/ (root)`)
and the report becomes a browsable dashboard at
`https://<you>.github.io/<repo>/output/report.html`.

## Running locally instead

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scanner                 # full market scan
python -m scanner --limit 500     # quick smoke test
python -m scanner --verbose       # debug logging
```

Outputs land in `output/`. Downloads are cached per day in `.cache/`, so
re-runs on the same day are fast.

## Tuning

Everything lives in `config.yaml`:

- `signals.weights` — the blend of the four signal families.
- `universe.min_price`, `min_median_dollar_volume` — junk/liquidity floor.
- `options_flow.shortlist_size`, `min_call_vol_oi_ratio` — options layer.
- `report.top_n` — how many candidates the report shows.

## Signal design notes

- **Volume** — z-score of log-volume vs the ticker's own 60-day baseline plus
  dollar-volume multiple, logistic-squashed so one wild print can't dominate.
- **Momentum** — ATR-normalized move (a +6% day means more for a utility than
  a biotech), 52-week-high breakouts, week-over-week acceleration, and
  gap-and-hold (gap-and-fade earns nothing).
- **Relative strength** — benchmark-relative returns over 1/5/20 days,
  front-weighted, expressed as a cross-sectional percentile.
- **ML outlier** — IsolationForest over the joint feature vector with robust
  scaling; catches tickers that are moderately unusual on *many* axes at once.
- **Growth gate** — anomalies are direction-agnostic; the ranking keeps only
  upward ones (crashes are anomalies you don't buy).
- **Options flow** — call volume swamping open interest, heavy call/put tilt,
  and OTM concentration add a *bounded* confirmation bonus (cap 0.15) to the
  composite; flow confirms an anomaly but never creates one.

## Tests

```bash
pip install pytest && python -m pytest tests/ -q
```

Unit tests run on synthetic fixtures — no network required.
