from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timestransfer.config import dataset_configs, load_config
from timestransfer.data import fixed_train_test_split, load_dataset, select_series
from timestransfer.forecast_plots import write_forecast_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-series train history + actual continuation vs model forecasts."
    )
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to YAML config.")
    parser.add_argument("--datasets", nargs="+", help="Optional subset of configured dataset names.")
    parser.add_argument(
        "--forecasts-dir",
        default=None,
        help="Directory with per-model forecast parquets (default: <outputs_dir>/forecasts).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figures (default: <outputs_dir>/figures/forecasts).",
    )
    parser.add_argument("--history-points", type=int, default=168, help="Train tail length to draw.")
    parser.add_argument(
        "--reference-model",
        default="timesfm_2p5",
        help="Model whose per-series sMAPE picks the best/median/worst series.",
    )
    parser.add_argument("--series", nargs="+", default=None, help="Explicit unique_ids to plot.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of models to draw.")
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
    forecasts_dir = Path(args.forecasts_dir) if args.forecasts_dir else outputs_dir / "forecasts"
    output_dir = Path(args.output_dir) if args.output_dir else outputs_dir / "figures" / "forecasts"

    configured = dataset_configs(config)
    if args.datasets:
        requested = set(args.datasets)
        configured = [dataset for dataset in configured if str(dataset.get("name")) in requested]
        missing = requested - {str(dataset.get("name")) for dataset in configured}
        if missing:
            raise ValueError(f"Unknown dataset name(s): {sorted(missing)}")

    for dataset_config in configured:
        dataset_name = str(dataset_config["name"])
        parquet_paths = sorted(forecasts_dir.glob(f"{dataset_name}_*.parquet"))
        if not parquet_paths:
            print(f"[skip] {dataset_name}: no forecast parquets in {forecasts_dir}")
            continue
        forecasts = pd.concat(
            [pd.read_parquet(path) for path in parquet_paths], ignore_index=True
        )
        forecasts = forecasts[forecasts["dataset"].astype(str) == dataset_name]
        forecasts = forecasts.drop_duplicates(
            subset=["dataset", "model", "unique_id", "horizon"], keep="last"
        )

        horizon = int(dataset_config.get("horizon", 48))
        configured_series_limit = dataset_config.get("series_limit", 32)
        effective_series_limit = (
            configured_series_limit if args.series_limit is None else args.series_limit
        )
        full_df = load_dataset(data_dir=data_dir, dataset_config=dataset_config)
        df = select_series(full_df, effective_series_limit)
        split = fixed_train_test_split(df, horizon=horizon)

        paths = write_forecast_plots(
            split.train,
            forecasts,
            dataset_name=dataset_name,
            output_dir=output_dir,
            history_points=args.history_points,
            series_ids=args.series,
            models=args.models,
            reference_model=args.reference_model,
        )
        print(f"[ok] {dataset_name}: wrote {len(paths)} figure(s) under {output_dir}")


if __name__ == "__main__":
    main()
