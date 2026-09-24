"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)

Locally, Streamlit reads .streamlit/secrets.toml from the CURRENT directory, so
run it from inside ui/:   cd ui && streamlit run app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")
MODEL_CARD = Path(__file__).resolve().parent.parent / "MODEL_CARD.md"

# (connect, read) timeouts. Render's free tier sleeps when idle and can take
# ~1 minute to wake, so reads get a generous budget.
TIMEOUT = (15, 120)

st.set_page_config(page_title="Regress-It", page_icon="📈", layout="wide")
st.title("📈 Regress-It — A Live Linear-Regression Service")
st.caption("Streamlit (this UI) → FastAPI (model) → Supabase (data). Three clouds.")


# ---------------------------------------------------------------------------
# API + Supabase helpers
# ---------------------------------------------------------------------------
class ApiError(Exception):
    """An API call failed; the message is safe to show the user."""


@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def _raise_for_api(r: requests.Response) -> None:
    if r.ok:
        return
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    raise ApiError(f"API returned {r.status_code}: {detail}")


def _request(method: str, path: str, **kwargs) -> requests.Response:
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ApiError(
            "Could not reach the API. If it has been idle, Render's free tier may "
            "still be waking up. Wait a minute and try again."
        ) from exc
    _raise_for_api(r)
    return r


def api_get(path: str):
    return _request("GET", path).json()


def api_post(path: str, payload: dict):
    return _request("POST", path, json=payload).json()


def api_train_stream(payload: dict) -> Iterator[tuple[str, dict]]:
    """POST /train/stream and yield (event, data) pairs as they arrive."""
    try:
        r = requests.post(
            f"{API_URL}/train/stream", json=payload, stream=True, timeout=TIMEOUT
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ApiError("Could not reach the API (it may be waking up).") from exc
    with r:
        _raise_for_api(r)
        event: Optional[str] = None
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[len("data: "):])
                event = None


@st.cache_data(ttl=30, show_spinner=False)
def api_status() -> dict:
    """Health + version for the sidebar. /healthz returns 503 when degraded,
    so read the body instead of raising on the status code."""
    try:
        health = requests.get(f"{API_URL}/healthz", timeout=TIMEOUT).json()
        version = requests.get(f"{API_URL}/version", timeout=TIMEOUT).json()
        return {"ok": health.get("status") == "ok", "health": health, "version": version}
    except (requests.RequestException, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


@st.cache_data(ttl=15, show_spinner=False)
def api_runs() -> list[dict]:
    return api_get("/runs")


def fmt(value: Optional[float], spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def loss_chart(losses: list[Optional[float]]) -> alt.Chart:
    """Training loss per epoch on a log scale (a diverging run spans many
    orders of magnitude). Non-finite epochs arrive as None and are dropped."""
    df = pd.DataFrame(
        [{"epoch": i + 1, "loss": v} for i, v in enumerate(losses) if v is not None and v > 0]
    )
    if df.empty:
        df = pd.DataFrame({"epoch": [], "loss": []})
    return (
        alt.Chart(df)
        .mark_line(point=len(df) <= 60)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="Training MSE (log scale)", scale=alt.Scale(type="log")),
            tooltip=["epoch", alt.Tooltip("loss:Q", format=".4g")],
        )
        .properties(height=280)
    )


def overlay_chart(dataset: dict, weights: dict, max_points: int = 2000) -> alt.LayerChart:
    """Dataset scatter with the true line (dashed) and the fitted line."""
    pts = pd.DataFrame({"x": dataset["xs"], "y": dataset["ys"]})
    if len(pts) > max_points:
        pts = pts.sample(max_points, random_state=0)
    x_lo, x_hi = min(dataset["xs"]), max(dataset["xs"])
    lines = pd.DataFrame(
        [
            {"line": "True line", "x": x, "y": dataset["slope"] * x + dataset["intercept"]}
            for x in (x_lo, x_hi)
        ]
        + [
            {"line": "Fitted line", "x": x, "y": weights["slope"] * x + weights["intercept"]}
            for x in (x_lo, x_hi)
        ]
    )
    scatter = (
        alt.Chart(pts)
        .mark_circle(size=18, opacity=0.35)
        .encode(x=alt.X("x:Q", title="x"), y=alt.Y("y:Q", title="y"))
    )
    fitted = (
        alt.Chart(lines)
        .mark_line(strokeWidth=3)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("line:N", title=None),
            strokeDash=alt.StrokeDash(
                "line:N",
                scale=alt.Scale(domain=["Fitted line", "True line"], range=[[1, 0], [6, 4]]),
                legend=None,
            ),
        )
    )
    return (scatter + fitted).properties(height=380)


# ---------------------------------------------------------------------------
# Sidebar: API status (handy for debugging cold starts and for screenshots)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Service status")
    status = api_status()
    if status["ok"]:
        st.success("API healthy")
    elif "health" in status:
        h = status["health"]
        st.warning(f"API degraded — model loader: {h['model_loader']}, Supabase: {h['supabase']}")
    else:
        st.error("API unreachable (it may be waking up).")
    if "version" in status:
        v = status["version"]
        st.caption(
            f"git `{v['git_sha']}` · torch {v['torch_version']} · "
            f"Supabase `{v['supabase_project_ref']}`"
        )
    st.caption(f"API: {API_URL}")
    if st.button("Recheck"):
        api_status.clear()
        st.rerun()


concepts, train_tab, predict_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Train", "Predict", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Gradient Descent & the Chain Rule")
    st.markdown(
        "We fit $y = w x + b$ by minimizing the mean squared error over the "
        "training set:"
    )
    st.latex(r"\mathcal{L}(w, b) = \frac{1}{n}\sum_{i=1}^{n}\left(w x_i + b - y_i\right)^2")
    st.markdown("Gradient descent updates the parameters against the gradient:")
    st.latex(
        r"\frac{\partial \mathcal{L}}{\partial w} = "
        r"\frac{2}{n}\sum_i x_i\,(w x_i + b - y_i), \qquad "
        r"\frac{\partial \mathcal{L}}{\partial b} = "
        r"\frac{2}{n}\sum_i (w x_i + b - y_i)"
    )
    st.latex(r"w \leftarrow w - \eta\,\frac{\partial \mathcal{L}}{\partial w}, \qquad "
             r"b \leftarrow b - \eta\,\frac{\partial \mathcal{L}}{\partial b}")
    st.markdown(
        "The **chain rule** is what lets us push the error at the output back to "
        "each parameter — the same engine that powers backpropagation in deeper "
        "networks later in the course. The learning rate $\\eta$ controls the step "
        "size: too small and training crawls; too large and the loss diverges."
    )

    st.subheader("How large can the learning rate be?")
    st.markdown(
        "The loss is a quadratic bowl, so its curvature is constant. When $x$ is "
        "roughly centered at zero, the curvature along $w$ is"
    )
    st.latex(r"\frac{\partial^2 \mathcal{L}}{\partial w^2} = \frac{2}{n}\sum_i x_i^2 = 2\,\mathbb{E}[x^2]")
    st.markdown(
        "and gradient descent on a quadratic is stable only when the step is below "
        "$2$ divided by the curvature:"
    )
    st.latex(r"\eta < \frac{2}{2\,\mathbb{E}[x^2]} = \frac{1}{\mathbb{E}[x^2]}")
    st.markdown(
        "For $x$ uniform on $[-10, 10]$, $\\mathbb{E}[x^2] = 100/3 \\approx 33$, so "
        "the limit is $\\eta \\approx 0.03$. Above it, each step overshoots the "
        "minimum by more than it corrects, and the loss grows geometrically."
    )

    st.subheader("When does training stop?")
    st.markdown(
        "After every epoch the API checks, in order: **(1) divergence** — the loss "
        "or a weight is non-finite, or the loss exceeds 100× the untrained model's "
        "loss; **(2) plateau** — the best training loss has not improved by the "
        "relative tolerance for *patience* epochs; **(3) budget** — the maximum "
        "epoch count is reached. The plateau check watches the *training* loss so "
        "the held-out 20% test split never influences training."
    )

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model")

    with st.expander("1) Create a synthetic dataset", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Name", value="demo")
        slope = c2.number_input("True slope", value=2.5)
        intercept = c3.number_input("True intercept", value=1.0)
        noise = c4.number_input("Noise (std)", value=2.0, min_value=0.0)
        n_points = st.slider("Number of points", 100, 5000, 500, step=100)
        if st.button("Create dataset"):
            try:
                ds = api_post(
                    "/datasets",
                    {
                        "name": name,
                        "slope": slope,
                        "intercept": intercept,
                        "noise": noise,
                        "n_points": n_points,
                    },
                )
                st.session_state["dataset_id"] = ds["id"]
                st.success(f"Created dataset id={ds['id']}")
            except ApiError as exc:
                st.error(str(exc))

    st.subheader("2) Train")
    dataset_id = st.number_input(
        "dataset_id", value=int(st.session_state.get("dataset_id", 1)), min_value=1, step=1
    )
    c1, c2, c3 = st.columns(3)
    lr = c1.number_input("Learning rate", value=0.01, min_value=0.0, format="%.4f")
    batch_size = c2.number_input("Batch size", value=32, min_value=1, step=1)
    epochs = c3.number_input("Max epochs", value=200, min_value=1, max_value=5000, step=10)

    with st.expander("Stopping criterion"):
        early_stopping = st.checkbox("Stop early when the training loss plateaus", value=True)
        s1, s2 = st.columns(2)
        patience = s1.number_input(
            "Patience (epochs)", value=20, min_value=1, max_value=1000, disabled=not early_stopping
        )
        min_delta = s2.number_input(
            "Min relative improvement", value=0.0001, min_value=0.0, max_value=0.99,
            format="%.5f", disabled=not early_stopping,
        )

    run_clicked = st.button("Run training", type="primary", disabled=lr <= 0)
    loss_slot = st.empty()

    if run_clicked:
        payload = {
            "dataset_id": int(dataset_id),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "early_stopping": bool(early_stopping),
            "patience": int(patience),
            "min_delta": float(min_delta),
        }
        losses: list[Optional[float]] = []
        last_draw = 0.0
        try:
            with st.spinner("Training on the FastAPI service..."):
                for event, data in api_train_stream(payload):
                    if event == "epoch":
                        losses.append(data["loss"])
                        # Redraw at most ~10x/second so long runs stay smooth.
                        if time.monotonic() - last_draw > 0.1:
                            loss_slot.altair_chart(loss_chart(losses), use_container_width=True)
                            last_draw = time.monotonic()
                    elif event == "done":
                        st.session_state["last_result"] = data
                        st.session_state["last_dataset"] = api_get(f"/datasets/{int(dataset_id)}")
                        if data["status"] != "diverged":
                            st.session_state["last_run_id"] = data["run_id"]
                        api_runs.clear()
                    elif event == "error":
                        raise ApiError(data.get("detail", "training failed"))
        except ApiError as exc:
            st.error(str(exc))

    # Results persist in session_state so they survive Streamlit reruns.
    result = st.session_state.get("last_result")
    if result:
        if any(v is not None for v in result["loss_history"]):
            loss_slot.altair_chart(loss_chart(result["loss_history"]), use_container_width=True)
        else:
            loss_slot.caption(
                "The training loss overflowed to infinity in the first epoch, so there "
                "is no curve to draw. Try a learning rate just above the limit (e.g. "
                "0.035) to watch the loss grow instead."
            )
        rid, status_, n_ep = result["run_id"], result["status"], result["epochs_run"]
        if status_ == "converged":
            st.success(f"Run {rid} converged after {n_ep} epochs (training loss plateaued).")
        elif status_ == "max_epochs":
            st.info(f"Run {rid} used the full budget of {n_ep} epochs without plateauing.")
        else:
            st.error(
                f"Run {rid} diverged after {n_ep} epoch(s): the learning rate is too "
                "large for this data. See Concepts for the stability limit."
            )

        m = result["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test MSE", fmt(m["mse"], ".4g"))
        mc2.metric("Test MAE", fmt(m["mae"], ".4g"))
        mc3.metric("Test R²", fmt(m["r2"], ".3f"))
        mc4.metric("Epochs run", n_ep)

        ds = st.session_state.get("last_dataset")
        w = result["weights"]
        if status_ != "diverged" and ds and w.get("slope") is not None:
            st.markdown(
                f"Fitted line: **y = {w['slope']:.3f}·x + {w['intercept']:.3f}** "
                f"(true: y = {ds['slope']:.3f}·x + {ds['intercept']:.3f})"
            )
            st.altair_chart(overlay_chart(ds, w), use_container_width=True)
        elif status_ == "diverged":
            st.caption("No fitted-line overlay: a diverged run's weights are not a usable model.")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
with predict_tab:
    st.header("Predict")
    try:
        usable = [r for r in api_runs() if r["status"] != "diverged"]
    except ApiError as exc:
        usable = []
        st.error(str(exc))

    if not usable:
        st.info("No usable runs yet. Train a model on the Train tab.")
    else:
        labels = {
            r["id"]: f"Run {r['id']} — {r['status']}, lr={r['lr']:g}, R²={fmt(r['r2'])}"
            for r in usable
        }
        ids = list(labels)  # newest first, from GET /runs
        preferred = st.session_state.get("last_run_id")
        run_id = st.selectbox(
            "Model (run)",
            ids,
            index=ids.index(preferred) if preferred in ids else 0,
            format_func=labels.get,
        )
        x = st.number_input("x", value=3.0)
        if st.button("Predict"):
            try:
                resp = api_post("/predict", {"run_id": int(run_id), "x": float(x)})
                st.metric(f"ŷ for x={resp['x']}", f"{resp['yhat']:.4f}")
                st.caption("This prediction was logged to Supabase by the API.")
            except ApiError as exc:
                st.error(str(exc))

# ---------------------------------------------------------------------------
# Run History  (read-only anon-key query against Supabase)
# ---------------------------------------------------------------------------
with history_tab:
    st.header("Run History")
    st.caption("Read directly from Supabase with the anon key — no API call. Click a column header to sort.")
    if st.button("Refresh"):
        st.rerun()
    try:
        rows = (
            supabase_anon()
            .table("runs")
            .select(
                "id,dataset_id,status,lr,batch_size,epochs,epochs_run,patience,"
                "min_delta,mse,mae,r2,loss_history,created_at"
            )
            .order("created_at", desc=True)
            .limit(50)
            .execute()
            .data
        )
    except Exception as exc:  # noqa: BLE001
        rows = []
        st.error(f"Could not read runs: {exc}")

    if not rows:
        st.info("No runs yet. Train a model on the Train tab.")
    else:
        df = pd.DataFrame(rows)
        statuses = st.multiselect(
            "Status", ["converged", "max_epochs", "diverged"],
            default=["converged", "max_epochs", "diverged"],
        )
        shown = df[df["status"].isin(statuses)]
        st.dataframe(
            shown.drop(columns=["loss_history"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("Run", format="%d"),
                "dataset_id": st.column_config.NumberColumn("Dataset", format="%d"),
                "lr": st.column_config.NumberColumn("LR", format="%.4g"),
                "epochs": st.column_config.NumberColumn("Max epochs", format="%d"),
                "epochs_run": st.column_config.NumberColumn("Epochs run", format="%d"),
                "min_delta": st.column_config.NumberColumn("Min delta", format="%.0e"),
                "mse": st.column_config.NumberColumn("Test MSE", format="%.4g"),
                "mae": st.column_config.NumberColumn("Test MAE", format="%.4g"),
                "r2": st.column_config.NumberColumn("Test R²", format="%.4f"),
                "created_at": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD HH:mm"),
            },
        )

        st.subheader("Compare loss curves")
        pick = st.multiselect(
            "Runs to overlay",
            shown["id"].tolist(),
            default=shown["id"].tolist()[:3],
            format_func=lambda i: f"Run {i} (lr={df.loc[df['id'] == i, 'lr'].iloc[0]:g})",
        )
        curves = pd.DataFrame(
            [
                {"run": f"Run {r['id']} · lr={r['lr']:g}", "epoch": e + 1, "loss": v}
                for r in rows if r["id"] in pick
                for e, v in enumerate(r.get("loss_history") or [])
                if v is not None and v > 0
            ]
        )
        if curves.empty:
            st.caption("Pick one or more runs to compare.")
        else:
            st.altair_chart(
                alt.Chart(curves)
                .mark_line()
                .encode(
                    x=alt.X("epoch:Q", title="Epoch"),
                    y=alt.Y("loss:Q", title="Training MSE (log scale)", scale=alt.Scale(type="log")),
                    color=alt.Color("run:N", title=None),
                    tooltip=["run", "epoch", alt.Tooltip("loss:Q", format=".4g")],
                )
                .properties(height=320),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Model Card  (rendered from a markdown file in the repo)
# ---------------------------------------------------------------------------
with card_tab:
    st.header("Model Card")
    try:
        st.markdown(MODEL_CARD.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.warning("MODEL_CARD.md not found.")
