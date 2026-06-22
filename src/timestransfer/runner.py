from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from timestransfer.config import dataset_configs, load_config, model_enabled
from timestransfer.data import fixed_train_test_split, load_dataset, select_series
from timestransfer.features import build_lag_feature_bundle
from timestransfer.metadata import build_environment_metadata, write_json
from timestransfer.metrics import compute_metric_rows, failure_metric_row
from timestransfer.models import run_auto_arima, run_linear_regression, run_tabpfn, run_timesfm_2p5
from timestransfer.reporting import write_reporting_outputs


def run_benchmark(
    config_path: str | Path,
    model_names: list[str] | None = None,
    series_limit: int | None = None,
    dataset_names: list[str] | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    project = config.get("project", {})
    model_config = config.get("models", {})

    data_dir = Path(project.get("data_dir", "data"))
    outputs_dir = Path(project.get("outputs_dir", "outputs"))
    forecasts_dir = outputs_dir / "forecasts"
    metrics_dir = outputs_dir / "metrics"
    metadata_dir = outputs_dir / "metadata"
    for path in (forecasts_dir, metrics_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    seed = int(project.get("seed", 42))

    configured_datasets = dataset_configs(config)
    if dataset_names:
        requested = set(dataset_names)
        configured_datasets = [
            dataset for dataset in configured_datasets if str(dataset.get("name")) in requested
        ]
        missing = requested - {str(dataset.get("name")) for dataset in configured_datasets}
        if missing:
            raise ValueError(f"Unknown dataset name(s): {sorted(missing)}")

    requested_models = model_names or [
        name
        for name in ("linear_regression", "tabpfn", "timesfm_2p5", "auto_arima")
        if model_enabled(config, name)
    ]

    forecast_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    model_statuses: dict[str, dict[str, dict[str, Any]]] = {}
    dataset_summaries: list[dict[str, Any]] = []

    for dataset_config in configured_datasets:
        dataset_name = str(dataset_config.get("name", "m4_hourly"))
        horizon = int(dataset_config.get("horizon", 48))
        seasonality = int(dataset_config.get("seasonality", 24))
        configured_series_limit = dataset_config.get("series_limit", 32)
        effective_series_limit = configured_series_limit if series_limit is None else series_limit
        feature_config = dataset_config["features"]

        full_df = load_dataset(data_dir=data_dir, dataset_config=dataset_config)
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
            elif model_name == "auto_arima":
                auto_arima_config = model_config.get("auto_arima", {})
                auto_arima_kwargs = {
                    key: value
                    for key, value in auto_arima_config.items()
                    if key not in {"enabled", "n_jobs", "season_length"}
                }
                model_results.append(
                    run_auto_arima(
                        split.train,
                        split.test,
                        dataset_name,
                        horizon=horizon,
                        seasonality=int(auto_arima_config.get("season_length", seasonality)),
                        freq=dataset_config.get("freq", 1),
                        n_jobs=int(auto_arima_config.get("n_jobs", -1)),
                        model_kwargs=auto_arima_kwargs,
                    )
                )
            else:
                raise ValueError(f"Unknown model name: {model_name}")

        model_statuses[dataset_name] = {}
        for result in model_results:
            model_statuses[dataset_name][result.model] = {"status": result.status, **result.details}
            if result.forecasts is not None and not result.forecasts.empty:
                result_forecasts = result.forecasts.copy()
                result_forecasts["forecast_horizon"] = horizon
                forecast_frames.append(result_forecasts)
                forecast_path = forecasts_dir / f"{dataset_name}_{result.model}.parquet"
                result_forecasts.to_parquet(forecast_path, index=False)
            else:
                metric_frames.append(
                    pd.DataFrame(
                        [
                            failure_metric_row(
                                dataset=dataset_name,
                                model=result.model,
                                error=result.details.get("error", result.details.get("message", "")),
                                forecast_horizon=horizon,
                            )
                        ]
                    )
                )

        dataset_summaries.append(
            {
                "name": dataset_name,
                "loader": dataset_config.get("loader", dataset_config.get("name")),
                "group": dataset_config.get("group"),
                "horizon": horizon,
                "seasonality": seasonality,
                "freq": dataset_config.get("freq", 1),
                "full_series_count": int(full_df["unique_id"].nunique()),
                "evaluated_series_count": int(df["unique_id"].nunique()),
                "series_limit": effective_series_limit,
                "train_rows": int(len(split.train)),
                "test_rows": int(len(split.test)),
                "lag_feature_train_rows": int(len(bundle.X_train)),
                "lag_feature_columns": bundle.feature_columns,
            }
        )

    if forecast_frames:
        all_forecasts = pd.concat(forecast_frames, ignore_index=True)
        all_forecasts_for_write = all_forecasts.copy()
        all_forecasts_for_write["ds"] = all_forecasts_for_write["ds"].astype(str)
        all_forecasts_for_write.to_parquet(
            forecasts_dir / "all_datasets_all_models.parquet",
            index=False,
        )
        metric_frames.insert(0, compute_metric_rows(all_forecasts))

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    metrics_path = metrics_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    report_paths = write_reporting_outputs(metrics, outputs_dir)

    metadata = build_environment_metadata(
        config=config,
        dataset_summary=dataset_summaries,
        model_statuses=model_statuses,
    )
    metadata_path = metadata_dir / "environment.json"
    write_json(metadata_path, metadata)

    return {
        "metrics": metrics_path,
        "metadata": metadata_path,
        "forecasts": forecasts_dir,
        **{name: path for name, path in report_paths.items() if path is not None},
    }
