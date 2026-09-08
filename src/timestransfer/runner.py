from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from timestransfer.config import dataset_configs, enabled_model_names, load_config
from timestransfer.data import DatasetSplit, fixed_train_test_split, load_dataset, select_series
from timestransfer.features import FeatureBundle, build_lag_feature_bundle
from timestransfer.metadata import build_environment_metadata, write_json
from timestransfer.metrics import compute_metric_rows, failure_metric_row, seasonal_naive_scales
from timestransfer.models import (
    ModelRunResult,
    run_chronos_bolt,
    run_linear_regression,
    run_prophet,
    run_statsforecast_model,
    run_tabpfn,
    run_timesfm_2p5,
)
from timestransfer.reporting import write_reporting_outputs


@dataclass(frozen=True)
class DatasetRunContext:
    dataset_name: str
    dataset_config: dict[str, Any]
    horizon: int
    seasonality: int
    freq: str | int
    split: DatasetSplit
    bundle: FeatureBundle
    seed: int


ModelEntryRunner = Callable[[str, dict[str, Any], DatasetRunContext], ModelRunResult]

_TIMESFM_FLAG_KEYS = (
    "normalize_inputs",
    "use_continuous_quantile_head",
    "force_flip_invariance",
    "infer_is_positive",
    "fix_quantile_crossing",
)

_STATSFORECAST_RESERVED_KEYS = {
    "enabled",
    "runner",
    "estimator",
    "n_jobs",
    "season_length",
    "max_train_length",
}


def _run_linear_regression_entry(
    model_name: str, model_cfg: dict[str, Any], ctx: DatasetRunContext
) -> ModelRunResult:
    return run_linear_regression(ctx.bundle, ctx.dataset_name, model_name=model_name)


def _run_tabpfn_entry(
    model_name: str, model_cfg: dict[str, Any], ctx: DatasetRunContext
) -> ModelRunResult:
    return run_tabpfn(
        ctx.bundle,
        ctx.dataset_name,
        model_name=model_name,
        prediction_batch_size=int(model_cfg.get("prediction_batch_size", 1024)),
    )


def _run_timesfm_entry(
    model_name: str, model_cfg: dict[str, Any], ctx: DatasetRunContext
) -> ModelRunResult:
    overrides = {key: bool(model_cfg[key]) for key in _TIMESFM_FLAG_KEYS if key in model_cfg}
    return run_timesfm_2p5(
        ctx.split.train,
        ctx.split.test,
        ctx.dataset_name,
        model_name=model_name,
        model_id=str(model_cfg.get("model_id", "google/timesfm-2.5-200m-pytorch")),
        max_context=int(model_cfg.get("max_context", 1024)),
        max_horizon=int(model_cfg.get("max_horizon", ctx.horizon)),
        horizon=ctx.horizon,
        forecast_config_overrides=overrides or None,
    )


_PROPHET_RESERVED_KEYS = {"enabled", "runner", "n_jobs", "max_train_length"}


def _run_prophet_entry(
    model_name: str, model_cfg: dict[str, Any], ctx: DatasetRunContext
) -> ModelRunResult:
    max_train_length = model_cfg.get("max_train_length", 4320)
    return run_prophet(
        ctx.split.train,
        ctx.split.test,
        ctx.dataset_name,
        model_name=model_name,
        horizon=ctx.horizon,
        time_freq=str(ctx.dataset_config.get("time_freq", ctx.freq)),
        max_train_length=int(max_train_length) if max_train_length is not None else None,
        n_jobs=int(model_cfg.get("n_jobs", 1)),
        model_kwargs={
            key: value for key, value in model_cfg.items() if key not in _PROPHET_RESERVED_KEYS
        },
    )


def _run_chronos_entry(
    model_name: str, model_cfg: dict[str, Any], ctx: DatasetRunContext
) -> ModelRunResult:
    return run_chronos_bolt(
        ctx.split.train,
        ctx.split.test,
        ctx.dataset_name,
        model_name=model_name,
        horizon=ctx.horizon,
        model_id=str(model_cfg.get("model_id", "amazon/chronos-bolt-base")),
        context_length=int(model_cfg.get("context_length", 2048)),
        batch_size=int(model_cfg.get("batch_size", 64)),
        device_map=str(model_cfg.get("device_map", "cuda")),
    )


def _make_statsforecast_entry(estimator: str) -> ModelEntryRunner:
    def _entry(model_name: str, model_cfg: dict[str, Any], ctx: DatasetRunContext) -> ModelRunResult:
        max_train_length = model_cfg.get("max_train_length")
        return run_statsforecast_model(
            ctx.split.train,
            ctx.split.test,
            ctx.dataset_name,
            model_name=model_name,
            estimator=estimator,
            horizon=ctx.horizon,
            seasonality=int(model_cfg.get("season_length", ctx.seasonality)),
            freq=ctx.freq,
            n_jobs=int(model_cfg.get("n_jobs", -1)),
            max_train_length=int(max_train_length) if max_train_length is not None else None,
            model_kwargs={
                key: value
                for key, value in model_cfg.items()
                if key not in _STATSFORECAST_RESERVED_KEYS
            },
        )

    return _entry


MODEL_RUNNERS: dict[str, ModelEntryRunner] = {
    "linear_regression": _run_linear_regression_entry,
    "tabpfn": _run_tabpfn_entry,
    "timesfm_2p5": _run_timesfm_entry,
    "auto_arima": _make_statsforecast_entry("AutoARIMA"),
    "seasonal_naive": _make_statsforecast_entry("SeasonalNaive"),
    "auto_ets": _make_statsforecast_entry("AutoETS"),
    "auto_theta": _make_statsforecast_entry("AutoTheta"),
    "prophet": _run_prophet_entry,
    "chronos_bolt": _run_chronos_entry,
}


def _dispatch_model(
    model_name: str, model_cfg: dict[str, Any], ctx: DatasetRunContext
) -> ModelRunResult:
    runner_key = str(model_cfg.get("runner", model_name))
    if runner_key not in MODEL_RUNNERS:
        raise ValueError(
            f"Unknown model runner {runner_key!r} for model {model_name!r}. "
            f"Known runners: {sorted(MODEL_RUNNERS)}"
        )
    return MODEL_RUNNERS[runner_key](model_name, model_cfg, ctx)


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

    requested_models = model_names or enabled_model_names(config)
    unknown_models = [name for name in requested_models if name not in model_config]
    if unknown_models:
        raise ValueError(
            f"Unknown model name(s): {sorted(unknown_models)}. "
            f"Configured models: {sorted(model_config)}"
        )

    forecast_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    mase_scale_frames: list[pd.DataFrame] = []
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

        mase_seasonality = int(dataset_config.get("mase_seasonality", seasonality))
        dataset_mase_scales = seasonal_naive_scales(split.train, mase_seasonality)
        dataset_mase_scales.insert(0, "dataset", dataset_name)
        mase_scale_frames.append(dataset_mase_scales)

        bundle = build_lag_feature_bundle(
            split.train,
            split.test,
            horizon=horizon,
            seasonality=seasonality,
            lags=[int(lag) for lag in feature_config["lags"]],
            train_row_cap=feature_config.get("train_row_cap"),
            seed=seed,
        )

        ctx = DatasetRunContext(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            horizon=horizon,
            seasonality=seasonality,
            freq=dataset_config.get("freq", 1),
            split=split,
            bundle=bundle,
            seed=seed,
        )
        model_results = [
            _dispatch_model(model_name, model_config.get(model_name, {}), ctx)
            for model_name in requested_models
        ]

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

    mase_scales = (
        pd.concat(mase_scale_frames, ignore_index=True) if mase_scale_frames else None
    )
    if mase_scales is not None:
        mase_scales.to_csv(metrics_dir / "mase_scales.csv", index=False)

    if forecast_frames:
        all_forecasts = pd.concat(forecast_frames, ignore_index=True)
        all_forecasts_for_write = all_forecasts.copy()
        all_forecasts_for_write["ds"] = all_forecasts_for_write["ds"].astype(str)
        all_forecasts_for_write.to_parquet(
            forecasts_dir / "all_datasets_all_models.parquet",
            index=False,
        )
        metric_frames.insert(0, compute_metric_rows(all_forecasts, mase_scales=mase_scales))

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
