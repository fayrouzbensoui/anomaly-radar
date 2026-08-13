from __future__ import annotations

import json
from datetime import date

import pandas as pd

import scanner.report as report_mod
from scanner.report import build_outputs
from scanner.scoring import score_universe
from scanner.signals import compute_features


def test_end_to_end_outputs(cfg, synthetic_history, tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod, "PROJECT_ROOT", tmp_path)

    feats = compute_features(synthetic_history, cfg)
    ranked = score_universe(feats, cfg).drop(index=["SPY"], errors="ignore")
    ranked["options_bonus"] = 0.0
    ranked["final_score"] = ranked["composite"]

    build_outputs(
        ranked=ranked,
        flow=pd.DataFrame(),
        names={t: f"{t} Corp" for t in ranked.index},
        etf_flags={t: False for t in ranked.index},
        cfg=cfg,
        universe_size=len(synthetic_history),
        scanned=len(feats),
        asof=date(2026, 8, 12),
    )

    out = tmp_path / "output"
    payload = json.loads((out / "results.json").read_text())
    assert payload["results"][0]["ticker"] == "ROCKT"
    assert payload["asof"] == "2026-08-12"
    assert "disclaimer" in payload
    assert (out / "history" / "2026-08-12.json").exists()
    assert (out / "results.csv").exists()

    html = (out / "report.html").read_text()
    assert "ROCKT" in html
    assert "CRASH" not in html
    assert "not investment advice" in html.lower()
