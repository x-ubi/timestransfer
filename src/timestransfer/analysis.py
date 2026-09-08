from __future__ import annotations

from pathlib import Path

import pandas as pd

from timestransfer.series_features import FEATURE_COLUMNS


def join_features_and_metrics(
    features: pd.DataFrame, series_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Join per-series features (with a dataset column) onto scope="series" metric rows."""
    return series_metrics.merge(features, on=["dataset", "unique_id"], how="inner")


def spearman_correlations(
    joined: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    metric: str = "smape",
    min_series: int = 3,
) -> pd.DataFrame:
    """Spearman rho of each series feature vs the per-series metric, per model.

    Computed pooled across datasets (dataset="all") and per dataset. Rows with fewer
    than min_series valid (feature, metric) pairs get NaN rho.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    rows: list[dict[str, object]] = []
    datasets = ["all", *sorted(joined["dataset"].astype(str).unique())]
    for dataset in datasets:
        subset = joined if dataset == "all" else joined[joined["dataset"] == dataset]
        for model, model_group in subset.groupby("model", sort=True):
            for feature in feature_columns:
                valid = model_group.loc[:, [feature, metric]].dropna()
                if len(valid) >= min_series and valid[feature].nunique() > 1:
                    rho = float(valid[feature].corr(valid[metric], method="spearman"))
                else:
                    rho = float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "feature": feature,
                        "metric": metric,
                        "spearman_rho": rho,
                        "n_series": int(len(valid)),
                    }
                )
    return pd.DataFrame(rows)


def write_analysis_outputs(
    joined: pd.DataFrame,
    *,
    output_dir: str | Path,
    feature_columns: list[str] | None = None,
    metric: str = "smape",
) -> dict[str, Path]:
    """Write the joined table, the correlation table, a heatmap, and scatter figures."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    joined_path = output_dir / "series_features_with_metrics.csv"
    joined.to_csv(joined_path, index=False)

    correlations = spearman_correlations(
        joined, feature_columns=feature_columns, metric=metric
    )
    correlations_path = output_dir / "feature_metric_correlations.csv"
    correlations.to_csv(correlations_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {
        "joined": joined_path,
        "correlations": correlations_path,
    }

    pooled = correlations[correlations["dataset"] == "all"]
    heatmap = pooled.pivot_table(index="model", columns="feature", values="spearman_rho")
    heatmap = heatmap.reindex(columns=[c for c in feature_columns if c in heatmap.columns])
    if not heatmap.empty:
        fig, ax = plt.subplots(
            figsize=(1.2 * len(heatmap.columns) + 3, 0.5 * len(heatmap) + 2)
        )
        image = ax.imshow(heatmap.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap.index)), heatmap.index)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                value = heatmap.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Spearman rho: series feature vs per-series {metric} (all datasets)")
        fig.colorbar(image, ax=ax, shrink=0.8)
        plt.tight_layout()
        heatmap_path = figures_dir / f"correlation_heatmap_{metric}.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        paths["heatmap"] = heatmap_path

    for feature in feature_columns:
        if feature not in joined.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, model_group in joined.groupby("model", sort=True):
            valid = model_group.loc[:, [feature, metric]].dropna()
            ax.scatter(valid[feature], valid[metric], s=12, alpha=0.5, label=model)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"per-series {metric}")
        ax.set_title(f"{feature} vs per-series {metric}")
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
        plt.tight_layout()
        figure_path = figures_dir / f"{feature}_vs_{metric}.png"
        plt.savefig(figure_path, dpi=150)
        plt.close(fig)
        paths[f"scatter_{feature}"] = figure_path

    return paths
