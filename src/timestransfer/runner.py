from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from timestransfer.config import load_config, model_enabled
from timestransfer.data import fixed_train_test_split, load_m4_hourly, select_series
from timestransfer.features import build_lag_feature_bundle
from timestransfer.metadata import build_environment_metadata, write_json
from timestransfer.metrics import compute_metric_rows, failure_metric_row
from timestransfer.models import run_linear_regression, run_tabpfn, run_timesfm_2p5


def run_benchmark(config_path: str | Path, model_names: list[str] | None = None, series_limit: int | None = None) -> dict[str, Path]:
    config = load_config(config_path)
    project = config.get("project", {})
    dataset_config = config.get("dataset", {})
    feature_config = config.get("features", {})
    model_config = config.get("models", {})

    data_dir = Path(project.get("data_dir", "data"))
    outputs_dir = Path(project.get("outputs_dir", "outputs"))
    forecasts_dir = outputs_dir / "forecasts"
    metrics_dir = outputs_dir / "metrics"
    metadata_dir = outputs_dir / "metadata"
    for path in (forecasts_dir, metrics_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset_name = str(dataset_config.get("name", "m4_hourly"))
    horizon = int(dataset_config.get("horizon", 48))
    seasonality = int(dataset_config.get("seasonality", 24))
    expected_n_series = dataset_config.get("expected_n_series", 414)
    configured_series_limit = dataset_config.get("series_limit", 32)
    effective_series_limit = configured_series_limit if series_limit is None else series_limit
    seed = int(project.get("seed", 42))

    full_df = load_m4_hourly(data_dir=data_dir, expected_n_series=expected_n_series)
    df = select_series(full_df, effective_series_limit)
    split = fixed_train_test_split(df, horizon=horizon)

    bundle = build_lag_feature_bundle(
        split.train,
        split.test,
        horizon=horizon,
        seasonality=seasonality,
        lags=[int(lag) for lag in feature_config["lags"]],
        train_row_cap=feature_config.get("train_row_cap"),
        seed=seed,
    )

    requested_models = model_names or [
        name for name in ("linear_regression", "tabpfn", "timesfm_2p5") if model_enabled(config, name)
    ]

    model_results = []
    for model_name in requested_models:
        if model_name == "linear_regression":
            model_results.append(run_linear_regression(bundle, dataset_name))
        elif model_name == "tabpfn":
            tabpfn_config = model_config.get("tabpfn", {})
            model_results.append(
                run_tabpfn(
                    bundle,
                    dataset_name,
                    prediction_batch_size=int(tabpfn_config.get("prediction_batch_size", 1024)),
                )
            )
        elif model_name == "timesfm_2p5":
            timesfm_config = model_config.get("timesfm_2p5", {})
            model_results.append(
                run_timesfm_2p5(
                    split.train,
                    split.test,
                    dataset_name,
                    model_id=str(timesfm_config.get("model_id", "google/timesfm-2.5-200m-pytorch")),
                    max_context=int(timesfm_config.get("max_context", 1024)),
                    max_horizon=int(timesfm_config.get("max_horizon", horizon)),
                    horizon=horizon,
                )
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    forecast_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    model_statuses: dict[str, dict[str, Any]] = {}

    for result in model_results:
        model_statuses[result.model] = {"status": result.status, **result.details}
        if result.forecasts is not None and not result.forecasts.empty:
            forecast_frames.append(result.forecasts)
            forecast_path = forecasts_dir / f"{dataset_name}_{result.model}.parquet"
            result.forecasts.to_parquet(forecast_path, index=False)
        else:
            metric_frames.append(
                pd.DataFrame(
                    [
                        failure_metric_row(
                            dataset=dataset_name,
                            model=result.model,
                            error=result.details.get("error", result.details.get("message", "")),
                        )
                    ]
                )
            )

    if forecast_frames:
        all_forecasts = pd.concat(forecast_frames, ignore_index=True)
        all_forecasts.to_parquet(forecasts_dir / f"{dataset_name}_all_models.parquet", index=False)
        metric_frames.insert(0, compute_metric_rows(all_forecasts))

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    metrics_path = metrics_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    dataset_summary = {
        "name": dataset_name,
        "group": dataset_config.get("group", "Hourly"),
        "horizon": horizon,
        "seasonality": seasonality,
        "full_series_count": int(full_df["unique_id"].nunique()),
        "evaluated_series_count": int(df["unique_id"].nunique()),
        "series_limit": effective_series_limit,
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "lag_feature_train_rows": int(len(bundle.X_train)),
        "lag_feature_columns": bundle.feature_columns,
    }
    metadata = build_environment_metadata(
        config=config,
        dataset_summary=dataset_summary,
        model_statuses=model_statuses,
    )
    metadata_path = metadata_dir / "environment.json"
    write_json(metadata_path, metadata)

    return {
        "metrics": metrics_path,
        "metadata": metadata_path,
        "forecasts": forecasts_dir,
    }
