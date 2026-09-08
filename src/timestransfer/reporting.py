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
    series_metrics = metrics[(metrics["scope"] == "series") & (metrics["status"] == "ok")].copy()

    overall_path = metrics_dir / "overall_by_dataset.csv"
    horizon_path = metrics_dir / "horizon_metrics.csv"
    series_path = metrics_dir / "series_metrics.csv"
    wide_path = metrics_dir / "model_comparison_wide.csv"
    overall.to_csv(overall_path, index=False)
    horizon_metrics.to_csv(horizon_path, index=False)
    series_metrics.to_csv(series_path, index=False)
    _wide_comparison(overall).to_csv(wide_path, index=False)

    figure_paths = _write_plots(overall=overall, horizon_metrics=horizon_metrics, figures_dir=figures_dir)
    return {
        "overall_by_dataset": overall_path,
        "horizon_metrics": horizon_path,
        "series_metrics": series_path,
        "model_comparison_wide": wide_path,
        **figure_paths,
    }


def _wide_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    value_columns = [
        column for column in ("mae", "rmse", "smape", "mase") if column in overall.columns
    ]
    wide = overall.pivot_table(
        index=["dataset", "forecast_horizon"],
        columns="model",
        values=value_columns,
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
    plot_specs = [
        ("smape", "sMAPE", "sMAPE"),
        ("rmse", "RMSE", "RMSE"),
        ("mase", "MASE", "MASE"),
    ]

    if overall.empty and horizon_metrics.empty:
        return {
            key: None
            for metric, _, _ in plot_specs
            for key in (f"overall_{metric}_plot", f"horizon_{metric}_plot")
        }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths: dict[str, Path | None] = {}

    for metric, label, title_metric in plot_specs:
        overall_available = not overall.empty and metric in overall.columns
        horizon_available = not horizon_metrics.empty and metric in horizon_metrics.columns
        overall_path = figures_dir / f"overall_{metric}_by_dataset.png"
        horizon_path = figures_dir / f"horizon_{metric}_curves.png"
        paths[f"overall_{metric}_plot"] = overall_path if overall_available else None
        paths[f"horizon_{metric}_plot"] = horizon_path if horizon_available else None

        if overall_available:
            pivot = overall.pivot_table(index="dataset", columns="model", values=metric, aggfunc="first")
            ax = pivot.plot(kind="bar", figsize=(10, 5))
            ax.set_ylabel(label)
            ax.set_xlabel("Dataset")
            ax.set_title(f"Overall {title_metric} by Dataset and Model")
            ax.legend(title="Model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
            plt.tight_layout()
            plt.savefig(overall_path, dpi=150)
            plt.close()

        if horizon_available:
            fig, ax = plt.subplots(figsize=(10, 5))
            for (dataset, model), group in horizon_metrics.groupby(["dataset", "model"], sort=True):
                group = group.sort_values("horizon")
                ax.plot(group["horizon"], group[metric], label=f"{dataset} / {model}")
            ax.set_xlabel("Forecast step")
            ax.set_ylabel(label)
            ax.set_title(f"Horizon-wise {title_metric}")
            ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
            plt.tight_layout()
            plt.savefig(horizon_path, dpi=150)
            plt.close(fig)

    return paths
