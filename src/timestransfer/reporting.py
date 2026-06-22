from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_reporting_outputs(metrics: pd.DataFrame, outputs_dir: str | Path) -> dict[str, Path | None]:
    outputs_dir = Path(outputs_dir)
    metrics_dir = outputs_dir / "metrics"
    figures_dir = outputs_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    overall = metrics[(metrics["scope"] == "overall") & (metrics["status"] == "ok")].copy()
    horizon_metrics = metrics[(metrics["scope"] == "horizon") & (metrics["status"] == "ok")].copy()

    overall_path = metrics_dir / "overall_by_dataset.csv"
    horizon_path = metrics_dir / "horizon_metrics.csv"
    wide_path = metrics_dir / "model_comparison_wide.csv"
    overall.to_csv(overall_path, index=False)
    horizon_metrics.to_csv(horizon_path, index=False)
    _wide_comparison(overall).to_csv(wide_path, index=False)

    figure_paths = _write_plots(overall=overall, horizon_metrics=horizon_metrics, figures_dir=figures_dir)
    return {
        "overall_by_dataset": overall_path,
        "horizon_metrics": horizon_path,
        "model_comparison_wide": wide_path,
        **figure_paths,
    }


def _wide_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    wide = overall.pivot_table(
        index=["dataset", "forecast_horizon"],
        columns="model",
        values=["mae", "rmse", "smape"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    return wide.reset_index()


def _write_plots(
    *,
    overall: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Path | None]:
    if overall.empty and horizon_metrics.empty:
        return {"overall_smape_plot": None, "horizon_smape_plot": None}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall_path = figures_dir / "overall_smape_by_dataset.png"
    horizon_path = figures_dir / "horizon_smape_curves.png"

    if not overall.empty:
        pivot = overall.pivot_table(index="dataset", columns="model", values="smape", aggfunc="first")
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("sMAPE")
        ax.set_xlabel("Dataset")
        ax.set_title("Overall sMAPE by Dataset and Model")
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
        plt.tight_layout()
        plt.savefig(overall_path, dpi=150)
        plt.close()

    if not horizon_metrics.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for (dataset, model), group in horizon_metrics.groupby(["dataset", "model"], sort=True):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group["smape"], label=f"{dataset} / {model}")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("sMAPE")
        ax.set_title("Horizon-wise sMAPE")
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
        plt.tight_layout()
        plt.savefig(horizon_path, dpi=150)
        plt.close(fig)

    return {"overall_smape_plot": overall_path, "horizon_smape_plot": horizon_path}
