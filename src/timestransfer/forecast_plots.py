from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-8


def _safe_filename(text: str) -> str:
    """Make a unique_id safe for a filename (Weather series names contain '/', '%', etc.)."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")


def select_plot_series(
    forecasts: pd.DataFrame,
    *,
    reference_model: str,
    first_k: int = 2,
) -> list[str]:
    """First k sorted unique_ids plus {best, median, worst} by per-series sMAPE.

    The best/median/worst ranking uses the reference model's forecasts; when the
    reference model is absent from the frame, the alphabetically first model is used.
    """
    if forecasts.empty:
        return []

    all_ids = sorted(forecasts["unique_id"].astype(str).unique())
    selected = list(all_ids[:first_k])

    available_models = sorted(forecasts["model"].astype(str).unique())
    if reference_model not in available_models:
        reference_model = available_models[0]
    reference = forecasts[forecasts["model"] == reference_model]

    scores: dict[str, float] = {}
    for unique_id, group in reference.groupby("unique_id", sort=True):
        scores[str(unique_id)] = _series_smape(group)
    ordered = sorted(scores, key=lambda unique_id: scores[unique_id])
    if ordered:
        for unique_id in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
            if unique_id not in selected:
                selected.append(unique_id)
    return selected


def write_forecast_plots(
    train_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    dataset_name: str,
    output_dir: str | Path,
    history_points: int = 168,
    series_ids: list[str] | None = None,
    models: list[str] | None = None,
    reference_model: str = "timesfm_2p5",
    first_k: int = 2,
) -> list[Path]:
    """One PNG per series: train-history tail, actual continuation, model forecasts.

    The forecast origin is marked with a vertical line; the actual test values are a
    solid black line and every model's prediction is a dashed line.
    """
    frame = forecasts[forecasts["dataset"].astype(str) == dataset_name].copy()
    if models:
        frame = frame[frame["model"].isin(models)]
    if frame.empty:
        return []
    if series_ids is None:
        series_ids = select_plot_series(frame, reference_model=reference_model, first_k=first_k)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for unique_id in series_ids:
        series_forecasts = frame[frame["unique_id"].astype(str) == str(unique_id)]
        if series_forecasts.empty:
            continue
        history = (
            train_df[train_df["unique_id"].astype(str) == str(unique_id)]
            .sort_values("ds")
            .tail(history_points)
        )
        first_model = sorted(series_forecasts["model"].unique())[0]
        actual = series_forecasts[series_forecasts["model"] == first_model].sort_values("horizon")

        fig, ax = plt.subplots(figsize=(12, 5))
        if not history.empty:
            ax.plot(history["ds"], history["y"], color="0.45", linewidth=1.2, label="history (train)")
            ax.axvline(history["ds"].iloc[-1], color="0.7", linestyle=":", linewidth=1.0)
        ax.plot(
            actual["ds"],
            actual["y_true"],
            color="black",
            linewidth=2.0,
            label="actual (test)",
        )
        for model, model_group in series_forecasts.groupby("model", sort=True):
            model_group = model_group.sort_values("horizon")
            ax.plot(model_group["ds"], model_group["y_pred"], linestyle="--", linewidth=1.2, label=model)
        ax.set_title(f"{dataset_name} / {unique_id}")
        ax.set_xlabel("ds")
        ax.set_ylabel("y")
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
        plt.tight_layout()
        path = output_dir / f"{dataset_name}_{_safe_filename(unique_id)}.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _series_smape(group: pd.DataFrame) -> float:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true
    return float(np.mean(200.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + EPSILON)))
