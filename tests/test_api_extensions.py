"""Tests for the endpoints and behaviors added on top of the template:
SSE telemetry, dataset points, diverged-run handling, and /healthz 503."""
from __future__ import annotations

import json

from api import db

DATASET = {"name": "t", "slope": 2.5, "intercept": 1.0, "noise": 2.0, "n_points": 500}


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_train_stream_emits_epochs_then_done_and_persists(client):
    ds = client.post("/datasets", json=DATASET).json()
    resp = client.post("/train/stream", json={"dataset_id": ds["id"], "lr": 0.01, "epochs": 200})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and set(kinds[:-1]) == {"epoch"}

    done = events[-1][1]
    assert done["status"] == "converged"
    assert len(kinds) - 1 == done["epochs_run"] == len(done["loss_history"])
    # The run row was written, with the same loss curve the client saw.
    assert client._store["runs"][done["run_id"]]["loss_history"] == done["loss_history"]


def test_train_stream_unknown_dataset_is_404(client):
    assert client.post("/train/stream", json={"dataset_id": 999}).status_code == 404


def test_diverged_run_is_persisted_with_null_metrics(client):
    ds = client.post("/datasets", json=DATASET).json()
    resp = client.post("/train", json={"dataset_id": ds["id"], "lr": 1.5, "epochs": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "diverged"
    assert body["metrics"]["mse"] is None
    run = client.get(f"/runs/{body['run_id']}").json()
    assert run["status"] == "diverged"


def test_predict_refuses_diverged_run(client):
    ds = client.post("/datasets", json=DATASET).json()
    run_id = client.post("/train", json={"dataset_id": ds["id"], "lr": 1.5}).json()["run_id"]
    resp = client.post("/predict", json={"run_id": run_id, "x": 3.0})
    assert resp.status_code == 409
    assert client._store["predictions"] == []  # nothing logged


def test_predict_logs_converged_run(client):
    ds = client.post("/datasets", json=DATASET).json()
    run_id = client.post("/train", json={"dataset_id": ds["id"], "lr": 0.01}).json()["run_id"]
    resp = client.post("/predict", json={"run_id": run_id, "x": 3.0})
    assert resp.status_code == 200
    assert abs(resp.json()["yhat"] - (2.5 * 3.0 + 1.0)) < 1.0
    assert len(client._store["predictions"]) == 1


def test_get_dataset_returns_points(client):
    ds = client.post("/datasets", json=DATASET).json()
    body = client.get(f"/datasets/{ds['id']}").json()
    assert len(body["xs"]) == len(body["ys"]) == 500


def test_healthz_is_503_when_supabase_down(client, monkeypatch):
    monkeypatch.setattr(db, "ping", lambda: False)
    resp = client.get("/healthz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
