"""Tests for the substantial RBA-derived research modules."""
import numpy as np
import pandas as pd
import pytest

from cds.physics.rba import RBAModel
from cds.evaluation.comparison import ModelComparison, compute_delta_bic_strength
from cds.verification.theory import TheoryVerifier


def test_rba_model_basic():
    model = RBAModel(a0_mps2=1.14e-10)
    r = np.array([5.0, 10.0, 20.0])
    vgas = np.array([30.0, 40.0, 35.0])
    vdisk = np.array([60.0, 80.0, 70.0])
    vbul = np.zeros(3)

    v = model.predict_velocity(r, vgas, vdisk, vbul)
    assert len(v) == 3
    assert np.all(v > 0)
    assert np.all(np.isfinite(v))


def test_rba_bic_computation():
    model = RBAModel()
    r = np.linspace(2, 15, 15)
    vobs = 120 + 10 * np.sin(r / 3)
    verr = np.full_like(r, 5.0)
    vgas = 35 * np.ones_like(r)
    vdisk = 75 * np.exp(-r / 10)
    vbul = np.zeros_like(r)

    metrics = model.compute_bic(r, vobs, verr, vgas, vdisk, vbul, k=0)
    assert "BIC" in metrics
    assert "lnL" in metrics
    assert metrics["k"] == 0


def test_model_comparison():
    data = pd.DataFrame({
        "galaxy": ["G1"] * 3 + ["G2"] * 3,
        "model": ["RBA", "EIN", "NFW"] * 2,
        "BIC": [210, 235, 250, 180, 195, 220],
    })
    comp = ModelComparison(data)
    winners = comp.per_dataset_winners("BIC")
    assert len(winners) == 2
    assert "RBA" in winners["best_model_BIC"].values


def test_delta_bic_strength():
    deltas = [0.5, 3.2, 7.1, 12.0, np.nan]
    buckets = compute_delta_bic_strength(deltas)
    assert "<2" in buckets.values
    assert ">=10" in buckets.values


def test_theory_verifier_runs():
    verifier = TheoryVerifier()
    results = verifier.run_rba_style_checks()
    assert "L_definition_conflict" in results
    assert "path_independence_phi_over_L" in results
    assert "w_degeneracy" in results
