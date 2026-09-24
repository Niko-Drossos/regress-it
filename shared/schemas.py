"""Pydantic request/response models shared by the API and (optionally) the UI.

Keeping every wire-format type in one module is the contract between the three
clouds. The Streamlit UI never imports model or SQL code -- it only imports (or
mirrors) these schemas so that the payloads it sends match what FastAPI expects.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["converged", "max_epochs", "diverged"]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets."""

    name: str = Field(..., min_length=1, max_length=100)
    slope: float = Field(2.0, description="True slope of the generating line.")
    intercept: float = Field(1.0, description="True intercept of the generating line.")
    noise: float = Field(1.0, ge=0.0, description="Std-dev of the Gaussian noise.")
    n_points: int = Field(500, ge=50, le=100_000)


class Dataset(BaseModel):
    """A dataset row as stored in Supabase (metadata only)."""

    id: int
    name: str
    slope: float
    intercept: float
    noise: float
    n_points: int
    created_at: datetime


class DatasetWithPoints(Dataset):
    """Response for GET /datasets/{id}: metadata plus the raw points, used by
    the UI for the fitted-line overlay (the UI's anon key cannot read datasets)."""

    xs: List[float]
    ys: List[float]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train and POST /train/stream."""

    dataset_id: int
    lr: float = Field(0.01, gt=0.0, description="Learning rate for gradient descent.")
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(100, ge=1, le=5000, description="Maximum epoch budget.")
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Held-out fraction.")
    early_stopping: bool = Field(True, description="Stop when the training loss plateaus.")
    patience: int = Field(20, ge=1, le=1000, description="Epochs without improvement before stopping.")
    min_delta: float = Field(1e-4, ge=0.0, lt=1.0, description="Relative loss improvement that counts.")


class Metrics(BaseModel):
    """Held-out metrics. None when a diverged run produced NaN/inf."""

    mse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]


class Run(BaseModel):
    """A training-run row as stored in Supabase."""

    id: int
    dataset_id: int
    lr: float
    batch_size: int
    epochs: int
    status: RunStatus
    epochs_run: Optional[int] = None
    patience: Optional[int] = None
    min_delta: Optional[float] = None
    mse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]
    weights_json: dict
    loss_history: List[Optional[float]] = Field(default_factory=list)
    created_at: datetime


class TrainResponse(BaseModel):
    """Final result of a training job (also the last SSE event of /train/stream)."""

    run_id: int
    status: RunStatus
    epochs_run: int
    metrics: Metrics
    weights: dict = Field(..., description="Fitted {'slope': .., 'intercept': ..}.")
    loss_history: List[Optional[float]]


class TrainEpochEvent(BaseModel):
    """One per-epoch telemetry event streamed by POST /train/stream."""

    epoch: int
    loss: Optional[float] = Field(..., description="Mean training MSE; None if non-finite.")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    run_id: int
    x: float = Field(..., description="Single feature value to predict on.")


class PredictResponse(BaseModel):
    run_id: int
    x: float
    yhat: float


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
class Health(BaseModel):
    # "model_loader" is the field name the brief's /healthz check describes;
    # allow it despite Pydantic reserving the "model_" prefix.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loader: bool
    supabase: bool


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]
