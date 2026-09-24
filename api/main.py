"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the model behind HTTP so the Streamlit UI can call it over HTTPS,
  * read datasets from Supabase before training,
  * write run rows and prediction rows back to Supabase.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds.
"""
from __future__ import annotations

import math
import os
import subprocess
from typing import Iterator

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api import db
from api.training import TrainResult, fit, iter_fit
from api.training import predict as run_predict
from shared.data import generate_linear
from shared.schemas import (
    Dataset,
    DatasetCreate,
    DatasetWithPoints,
    Health,
    Metrics,
    PredictRequest,
    PredictResponse,
    Run,
    TrainEpochEvent,
    TrainRequest,
    TrainResponse,
    Version,
)

app = FastAPI(
    title="Regress-It API",
    description="Linear-regression model service for the three-cloud stack.",
    version="1.1.0",
)

# The UI lives on a different origin (Streamlit Cloud), so CORS must allow it.
# "*" is fine for a teaching demo; tighten to your Streamlit URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _project_ref() -> str | None:
    url = os.environ.get("SUPABASE_URL", "")
    # https://<ref>.supabase.co -> <ref>
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["datasets"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Generate a synthetic dataset and persist it to Supabase."""
    xs, ys = generate_linear(req.slope, req.intercept, req.noise, req.n_points)
    row = db.insert_dataset(
        req.name, req.slope, req.intercept, req.noise, req.n_points, xs, ys
    )
    return Dataset.model_validate(row)


@app.get("/datasets/{dataset_id}", response_model=DatasetWithPoints, tags=["datasets"])
def get_dataset(dataset_id: int) -> DatasetWithPoints:
    """Return a dataset with its points (for the UI's fitted-line overlay)."""
    row = db.get_dataset(dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    return DatasetWithPoints.model_validate(row)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _load_dataset_or_404(dataset_id: int) -> dict:
    dataset = db.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    return dataset


def _fit_kwargs(req: TrainRequest) -> dict:
    return dict(
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        test_size=req.test_size,
        early_stopping=req.early_stopping,
        patience=req.patience,
        min_delta=req.min_delta,
    )


def _persist_run(req: TrainRequest, result: TrainResult) -> TrainResponse:
    """Write the run row to Supabase and build the API response."""
    run = db.insert_run(
        dataset_id=req.dataset_id,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        status=result.status,
        epochs_run=result.epochs_run,
        patience=req.patience if req.early_stopping else None,
        min_delta=req.min_delta if req.early_stopping else None,
        mse=result.metrics["mse"],
        mae=result.metrics["mae"],
        r2=result.metrics["r2"],
        weights_json=result.weights,
        loss_history=result.loss_history,
    )
    return TrainResponse(
        run_id=run["id"],
        status=result.status,
        epochs_run=result.epochs_run,
        metrics=Metrics(**result.metrics),
        weights=result.weights,
        loss_history=result.loss_history,
    )


@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest) -> TrainResponse:
    """Read a dataset from Supabase, train, write the run row, return metrics."""
    dataset = _load_dataset_or_404(req.dataset_id)
    result = fit(dataset["xs"], dataset["ys"], **_fit_kwargs(req))
    return _persist_run(req, result)


def _sse(event: str, payload: str) -> str:
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/train/stream", tags=["training"])
def train_stream(req: TrainRequest) -> StreamingResponse:
    """Same as POST /train, but streams per-epoch loss as Server-Sent Events.

    Event stream:
        event: epoch  data: TrainEpochEvent   (one per epoch)
        event: done   data: TrainResponse     (after the run row is written)
        event: error  data: {"detail": ...}   (if persisting fails)
    """
    # Look the dataset up BEFORE streaming starts so a bad id is a real 404.
    dataset = _load_dataset_or_404(req.dataset_id)

    def events() -> Iterator[str]:
        for ev in iter_fit(dataset["xs"], dataset["ys"], **_fit_kwargs(req)):
            if ev["type"] == "epoch":
                yield _sse("epoch", TrainEpochEvent(epoch=ev["epoch"], loss=ev["loss"]).model_dump_json())
            else:
                try:
                    resp = _persist_run(req, ev["result"])
                    yield _sse("done", resp.model_dump_json())
                except Exception as exc:  # noqa: BLE001
                    yield _sse("error", f'{{"detail": "failed to persist run: {type(exc).__name__}"}}')

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Stop proxies (including Render's) from buffering the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/runs/{run_id}", response_model=Run, tags=["training"])
def get_run(run_id: int) -> Run:
    """One run from Supabase, including its per-epoch loss history."""
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return Run.model_validate(row)


@app.get("/runs", response_model=list[Run], tags=["training"])
def list_runs() -> list[Run]:
    """Return the latest 50 runs from Supabase."""
    return [Run.model_validate(r) for r in db.latest_runs(limit=50)]


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Predict yhat for a feature value using a stored run; log to Supabase."""
    run = db.get_run(req.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    weights = run["weights_json"] or {}
    if (
        run.get("status") == "diverged"
        or weights.get("slope") is None
        or weights.get("intercept") is None
    ):
        raise HTTPException(
            status_code=409,
            detail="This run diverged; its weights are not a usable model. Pick a converged run.",
        )
    yhat = run_predict(weights, req.x)
    if not math.isfinite(yhat):
        raise HTTPException(status_code=422, detail="Prediction is not a finite number.")
    db.insert_prediction(req.run_id, req.x, yhat)
    return PredictResponse(run_id=req.run_id, x=req.x, yhat=yhat)


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz(response: Response) -> Health:
    """200 when the model loader (torch) and the Supabase client are both
    reachable; 503 otherwise, so Render's health check reflects real state."""
    supabase_ok = db.ping()
    try:
        model_ok = torch.tensor([1.0]).sum().item() == 1.0
    except Exception:  # noqa: BLE001
        model_ok = False
    healthy = supabase_ok and model_ok
    if not healthy:
        response.status_code = 503
    return Health(status="ok" if healthy else "degraded", model_loader=model_ok, supabase=supabase_ok)


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        supabase_project_ref=_project_ref(),
    )
