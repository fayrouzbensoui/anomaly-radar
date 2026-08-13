from __future__ import annotations

import numpy as np

from scanner.scoring import score_universe
from scanner.signals import (
    compute_features,
    ml_outlier_score,
    momentum_score,
    volume_score,
)


def test_features_detect_planted_volume_spike(cfg, synthetic_history):
    feats = compute_features(synthetic_history, cfg)
    assert feats.at["ROCKT", "vol_z"] > 3, "8x volume day must be a big z-score"
    assert feats.at["ROCKT", "ret_1d"] > 0.07
    assert abs(feats.at["QUIET00", "vol_z"]) < 3


def test_volume_score_ranks_spike_first(cfg, synthetic_history):
    feats = compute_features(synthetic_history, cfg)
    scores = volume_score(feats, cfg)
    assert scores.idxmax() in ("ROCKT", "CRASH")  # both spiked volume
    assert scores["ROCKT"] > scores["QUIET05"]
    assert scores.between(0, 1).all()


def test_momentum_score_rewards_up_not_down(cfg, synthetic_history):
    feats = compute_features(synthetic_history, cfg)
    scores = momentum_score(feats, cfg)
    assert scores["ROCKT"] > scores["CRASH"], "down move must not earn momentum"
    assert scores.between(0, 1).all()


def test_ml_outlier_flags_planted_anomaly(cfg, synthetic_history):
    feats = compute_features(synthetic_history, cfg)
    scores = ml_outlier_score(feats, cfg)
    assert scores["ROCKT"] > 0.9, "planted anomaly should be a top-decile outlier"
    assert scores.between(0, 1).all()


def test_growth_gate_excludes_crash(cfg, synthetic_history):
    feats = compute_features(synthetic_history, cfg)
    ranked = score_universe(feats, cfg)
    assert "ROCKT" in ranked.index
    assert "CRASH" not in ranked.index, "crash anomalies must be gated out"
    assert ranked.index[0] == "ROCKT", "planted anomaly should rank #1"


def test_composite_bounded(cfg, synthetic_history):
    feats = compute_features(synthetic_history, cfg)
    ranked = score_universe(feats, cfg)
    assert np.isfinite(ranked["composite"]).all()
    assert ranked["composite"].between(0, 1).all()
