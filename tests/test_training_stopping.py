"""Stopping-criterion tests: convergence, divergence, and JSON-safe output."""
from __future__ import annotations

import json

from api.training import fit, iter_fit
from shared.data import generate_linear

XS, YS = generate_linear(slope=2.5, intercept=1.0, noise=2.0, n_points=500)


def test_small_lr_converges_early():
    r = fit(XS, YS, lr=0.01, batch_size=32, epochs=500)
    assert r.status == "converged"
    assert r.epochs_run < 500
    assert abs(r.weights["slope"] - 2.5) < 0.2


def test_large_lr_diverges_and_stays_json_safe():
    r = fit(XS, YS, lr=1.5, batch_size=32, epochs=100)
    assert r.status == "diverged"
    assert r.epochs_run < 100
    # NaN/inf must have been mapped to None, or the Supabase insert and the
    # FastAPI response would both fail.
    json.dumps({"m": r.metrics, "w": r.weights, "l": r.loss_history}, allow_nan=False)


def test_no_early_stopping_runs_full_budget():
    r = fit(XS, YS, lr=0.01, batch_size=32, epochs=40, early_stopping=False)
    assert r.status == "max_epochs"
    assert r.epochs_run == 40
    assert len(r.loss_history) == 40


def test_iter_fit_emits_one_event_per_epoch_then_done():
    events = list(iter_fit(XS, YS, lr=0.01, batch_size=32, epochs=30))
    epoch_events = [e for e in events if e["type"] == "epoch"]
    assert events[-1]["type"] == "done"
    assert len(epoch_events) == events[-1]["result"].epochs_run


def test_training_is_deterministic():
    a = fit(XS, YS, lr=0.01, batch_size=32, epochs=100)
    b = fit(XS, YS, lr=0.01, batch_size=32, epochs=100)
    assert a.weights == b.weights and a.loss_history == b.loss_history
