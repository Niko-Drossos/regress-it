"""PyTorch linear-regression training loop.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. Given a dataset's (xs, ys) and
hyperparameters, it fits ``y = slope*x + intercept`` with mini-batch gradient
descent and returns held-out metrics plus the fitted weights.

Three entry points share one loop:

    iter_fit(...)                -> generator of per-epoch events, then a final
                                    "done" event (used by the SSE endpoint)
    fit(...)                     -> runs iter_fit to completion, returns a
                                    TrainResult (used by POST /train)
    train_linear_regression(...) -> legacy (metrics, weights, loss_history)
                                    tuple, kept so existing callers still work

Stopping criterion (checked after every epoch, in this order):

    1. Divergence guard -- stop immediately if the epoch loss or either
       parameter is non-finite, or if the loss exceeds ``divergence_factor``
       times the loss of the untrained model. status = "diverged".
    2. Plateau early stopping -- track the best training loss seen so far. If
       it has not improved by at least ``min_delta`` (relative) for
       ``patience`` consecutive epochs, training has converged.
       status = "converged".
    3. Epoch budget -- otherwise stop after ``epochs``. status = "max_epochs".

Early stopping watches the TRAINING loss, not the 20% test split. Using the
test split to decide when to stop would leak it into training and make the
reported MSE/MAE/R² optimistic. With two parameters, overfitting is not the
concern; the question is only "has the optimizer stopped making progress?"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# Defaults also documented in api/configs/default.yaml.
DEFAULT_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_DIVERGENCE_FACTOR = 100.0


@dataclass
class TrainResult:
    """Everything the API needs to persist a run and answer the client."""

    status: str  # "converged" | "max_epochs" | "diverged"
    epochs_run: int
    metrics: Dict[str, Optional[float]]  # mse/mae/r2 on the test split
    weights: Dict[str, Optional[float]]  # slope/intercept
    loss_history: List[Optional[float]] = field(default_factory=list)


def _finite(value: float) -> Optional[float]:
    """JSON and Postgres (via PostgREST) cannot carry NaN/inf -> map to None."""
    return float(value) if math.isfinite(value) else None


def _train_test_split(
    xs: np.ndarray, ys: np.ndarray, test_size: float, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xs))
    n_test = int(len(xs) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return xs[train_idx], xs[test_idx], ys[train_idx], ys[test_idx]


def _metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, Optional[float]]:
    err = y_pred - y_true
    mse = torch.mean(err**2).item()
    mae = torch.mean(torch.abs(err)).item()
    ss_res = torch.sum(err**2).item()
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": _finite(mse), "mae": _finite(mae), "r2": _finite(r2)}


def iter_fit(
    xs: List[float],
    ys: List[float],
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
    early_stopping: bool = True,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    divergence_factor: float = DEFAULT_DIVERGENCE_FACTOR,
    seed: int = 0,
) -> Iterator[dict]:
    """Train and yield telemetry.

    Yields ``{"type": "epoch", "epoch": i, "loss": float|None}`` after every
    epoch, then exactly one ``{"type": "done", "result": TrainResult}``.
    """
    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    x_train, x_test, y_train, y_test = _train_test_split(x_arr, y_arr, test_size, seed)

    x_train_t = torch.from_numpy(x_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    x_test_t = torch.from_numpy(x_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).unsqueeze(1)

    # A local generator instead of torch.manual_seed(): FastAPI runs sync
    # endpoints in a thread pool, so global RNG state would be shared between
    # concurrent training requests.
    gen = torch.Generator().manual_seed(seed)
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.uniform_(-1.0, 1.0, generator=gen)
        model.bias.uniform_(-1.0, 1.0, generator=gen)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = x_train_t.shape[0]

    # Loss of the untrained model: the baseline for the divergence guard.
    with torch.no_grad():
        initial_loss = loss_fn(model(x_train_t), y_train_t).item()

    loss_history: List[Optional[float]] = []
    best_loss = math.inf
    epochs_since_best = 0
    status = "max_epochs"
    epoch = 0

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb = x_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n

        loss_history.append(_finite(epoch_loss))
        yield {"type": "epoch", "epoch": epoch, "loss": _finite(epoch_loss)}

        # 1. Divergence guard
        params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
        if (
            not math.isfinite(epoch_loss)
            or not params_finite
            or epoch_loss > divergence_factor * initial_loss
        ):
            status = "diverged"
            break

        # 2. Plateau early stopping (relative improvement on the training loss)
        if epoch_loss < best_loss * (1.0 - min_delta):
            best_loss = epoch_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if early_stopping and epochs_since_best >= patience:
                status = "converged"
                break

    model.eval()
    with torch.no_grad():
        metrics = _metrics(y_test_t, model(x_test_t))

    result = TrainResult(
        status=status,
        epochs_run=epoch,
        metrics=metrics,
        weights={
            "slope": _finite(model.weight.item()),
            "intercept": _finite(model.bias.item()),
        },
        loss_history=loss_history,
    )
    yield {"type": "done", "result": result}


def fit(xs: List[float], ys: List[float], **kwargs) -> TrainResult:
    """Run training to completion and return only the final result."""
    for event in iter_fit(xs, ys, **kwargs):
        if event["type"] == "done":
            return event["result"]
    raise RuntimeError("iter_fit ended without a 'done' event")  # unreachable


def train_linear_regression(
    xs: List[float],
    ys: List[float],
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], List[Optional[float]]]:
    """Legacy wrapper: fixed epoch budget, returns (metrics, weights, loss_history)."""
    r = fit(
        xs, ys, lr=lr, batch_size=batch_size, epochs=epochs,
        test_size=test_size, early_stopping=False,
    )
    return r.metrics, r.weights, r.loss_history


def predict(weights: Dict[str, float], x: float) -> float:
    """Apply fitted weights to a single feature value."""
    return weights["slope"] * x + weights["intercept"]
