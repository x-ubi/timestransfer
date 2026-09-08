from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timestransfer.analysis import join_features_and_metrics, write_analysis_outputs
from timestransfer.config import dataset_configs, load_config
from timestransfer.data import fixed_train_test_split, load_dataset, select_series
from timestransfer.series_features import compute_series_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate per-series characteristics (ACF, STL strengths, ...) "
        "with per-series model performance from a finished benchmark run."
    )
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Metrics CSV with scope='series' rows (default: <outputs_dir>/metrics/metrics.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs (default: <outputs_dir>/analysis).",
    )
    parser.add_argument("--datasets", nargs="+", help="Optional subset of configured dataset names.")
    parser.add_argument(
        "--metric",
        default="smape",
        choices=["smape", "mase", "mae", "rmse"],
        help="Per-series metric to correlate against (scale-free smape/mase recommended).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="Tail cap per series for feature computation (STL cost control).",
    )
    parser.add_argument(
        "--series-limit",
        type=int,
        default=None,
        help="Override config dataset.series_limit (must match the benchmark run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project = config.get("project", {})
    data_dir = Path(project.get("data_dir", "data"))
    outputs_dir = Path(project.get("outputs_dir", "outputs"))
    metrics_csv = (
        Path(args.metrics_csv) if args.metrics_csv else outputs_dir / "metrics" / "metrics.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else outputs_dir / "analysis"

    metrics = pd.read_csv(metrics_csv, keep_default_na=True)
    series_metrics = metrics[
        (metrics["scope"] == "series") & (metrics["status"] == "ok")
    ].copy()
    if series_metrics.empty:
        raise ValueError(
            f"No scope='series' rows in {metrics_csv}; run the benchmark first "
            "(per-series metrics are produced by scripts/run_benchmark.py)."
        )
    series_metrics["dataset"] = series_metrics["dataset"].astype(str)
    series_metrics["unique_id"] = series_metrics["unique_id"].astype(str)

    configured = dataset_configs(config)
    if args.datasets:
        requested = set(args.datasets)
        configured = [dataset for dataset in configured if str(dataset.get("name")) in requested]
        missing = requested - {str(dataset.get("name")) for dataset in configured}
        if missing:
            raise ValueError(f"Unknown dataset name(s): {sorted(missing)}")

    evaluated_datasets = set(series_metrics["dataset"].unique())
    feature_frames: list[pd.DataFrame] = []
    for dataset_config in configured:
        dataset_name = str(dataset_config["name"])
        if dataset_name not in evaluated_datasets:
            print(f"[skip] {dataset_name}: no per-series metrics in {metrics_csv}")
            continue
        horizon = int(dataset_config.get("horizon", 48))
        seasonality = int(dataset_config.get("seasonality", 24))
        configured_series_limit = dataset_config.get("series_limit", 32)
        effective_series_limit = (
            configured_series_limit if args.series_limit is None else args.series_limit
        )
        full_df = load_dataset(data_dir=data_dir, dataset_config=dataset_config)
        df = select_series(full_df, effective_series_limit)
        split = fixed_train_test_split(df, horizon=horizon)
        features = compute_series_features(
            split.train, seasonality=seasonality, max_points=args.max_points
        )
        features.insert(0, "dataset", dataset_name)
        feature_frames.append(features)
        print(f"[ok] {dataset_name}: features for {len(features)} series")

    if not feature_frames:
        raise ValueError("No datasets produced features; nothing to analyze.")
    all_features = pd.concat(feature_frames, ignore_index=True)

    features_path = output_dir / "series_features.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_features.to_csv(features_path, index=False)

    joined = join_features_and_metrics(all_features, series_metrics)
    paths = write_analysis_outputs(joined, output_dir=output_dir, metric=args.metric)

    print(f"series features: {features_path}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
