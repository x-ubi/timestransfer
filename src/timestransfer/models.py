from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from timestransfer.features import FeatureBundle


@dataclass(frozen=True)
class ModelRunResult:
    model: str
    status: str
    forecasts: pd.DataFrame | None
    details: dict[str, Any]


def run_linear_regression(
    bundle: FeatureBundle,
    dataset_name: str,
    *,
    model_name: str = "linear_regression",
) -> ModelRunResult:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(bundle.X_train, bundle.y_train)
    y_pred_scaled = pipeline.predict(bundle.X_test)
    forecasts = _forecast_frame(
        dataset_name=dataset_name,
        model=model_name,
        test_index=bundle.test_index,
        y_pred_scaled=y_pred_scaled,
    )
    return ModelRunResult(
        model=model_name,
        status="ok",
        forecasts=forecasts,
        details={"estimator": "sklearn.linear_model.LinearRegression"},
    )


def run_tabpfn(
    bundle: FeatureBundle,
    dataset_name: str,
    *,
    model_name: str = "tabpfn",
    prediction_batch_size: int = 1024,
) -> ModelRunResult:
    try:
        import tabpfn
        from tabpfn import TabPFNRegressor
    except Exception as exc:
        return _failed_result(model_name, exc, "TabPFN import failed.")

    try:
        regressor = TabPFNRegressor()
        regressor.fit(bundle.X_train.to_numpy(dtype=np.float32), bundle.y_train.to_numpy(dtype=np.float32))
        predictions: list[np.ndarray] = []
        X_test = bundle.X_test.to_numpy(dtype=np.float32)
        for start in range(0, len(X_test), prediction_batch_size):
            stop = start + prediction_batch_size
            predictions.append(np.asarray(regressor.predict(X_test[start:stop]), dtype=float))
        y_pred_scaled = np.concatenate(predictions)
        forecasts = _forecast_frame(
            dataset_name=dataset_name,
            model=model_name,
            test_index=bundle.test_index,
            y_pred_scaled=y_pred_scaled,
        )
        return ModelRunResult(
            model=model_name,
            status="ok",
            forecasts=forecasts,
            details={
                "estimator": "tabpfn.TabPFNRegressor",
                "tabpfn_version": getattr(tabpfn, "__version__", "unknown"),
                "prediction_batch_size": prediction_batch_size,
            },
        )
    except Exception as exc:
        return _failed_result(model_name, exc, "TabPFN fit/predict failed.")


_TIMESFM_CHECKPOINT_CACHE: dict[str, Any] = {}


def run_timesfm_2p5(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    model_id: str,
    max_context: int,
    max_horizon: int,
    horizon: int,
    model_name: str = "timesfm_2p5",
    forecast_config_overrides: dict[str, Any] | None = None,
) -> ModelRunResult:
    try:
        import timesfm
    except Exception as exc:
        return _failed_result(model_name, exc, "TimesFM import failed.")

    try:
        if horizon > max_horizon:
            raise ValueError(f"Requested horizon {horizon} exceeds configured max_horizon {max_horizon}.")

        if model_id not in _TIMESFM_CHECKPOINT_CACHE:
            _TIMESFM_CHECKPOINT_CACHE[model_id] = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        model = _TIMESFM_CHECKPOINT_CACHE[model_id]
        config = _make_timesfm_forecast_config(
            timesfm,
            max_context=max_context,
            max_horizon=max_horizon,
            overrides=forecast_config_overrides,
        )
        model.compile(config)

        inputs: list[np.ndarray] = []
        unique_ids: list[str] = []
        train_nonnegative: dict[str, bool] = {}
        for unique_id, group in train_df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True):
            values = group["y"].to_numpy(dtype=np.float32)
            inputs.append(values[-max_context:])
            unique_ids.append(str(unique_id))
            train_nonnegative[str(unique_id)] = bool(np.nanmin(values) >= 0)

        forecast_output = model.forecast(horizon=horizon, inputs=inputs)
        point_forecast = forecast_output[0] if isinstance(forecast_output, tuple) else forecast_output
        point_forecast = np.asarray(point_forecast, dtype=float)
        if point_forecast.shape != (len(unique_ids), horizon):
            raise ValueError(
                "TimesFM returned unexpected forecast shape "
                f"{point_forecast.shape}; expected {(len(unique_ids), horizon)}."
            )

        rows: list[dict[str, object]] = []
        test_groups = {
            str(unique_id): group.sort_values("ds").reset_index(drop=True)
            for unique_id, group in test_df.groupby("unique_id", sort=True)
        }
        for series_idx, unique_id in enumerate(unique_ids):
            test_group = test_groups[unique_id]
            for h in range(1, horizon + 1):
                y_pred = float(point_forecast[series_idx, h - 1])
                if train_nonnegative[unique_id]:
                    y_pred = max(0.0, y_pred)
                test_row = test_group.iloc[h - 1]
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "unique_id": unique_id,
                        "horizon": h,
                        "ds": test_row["ds"],
                        "y_true": float(test_row["y"]),
                        "y_pred": y_pred,
                    }
                )

        return ModelRunResult(
            model=model_name,
            status="ok",
            forecasts=pd.DataFrame(rows),
            details={
                "estimator": "timesfm.TimesFM_2p5_200M_torch",
                "model_id": model_id,
                "max_context": max_context,
                "max_horizon": max_horizon,
                "forecast_config": _timesfm_forecast_config_kwargs(
                    max_context=max_context,
                    max_horizon=max_horizon,
                    overrides=forecast_config_overrides,
                ),
            },
        )
    except Exception as exc:
        return _failed_result(model_name, exc, "TimesFM load/forecast failed.")


_STATSFORECAST_ESTIMATORS = ("AutoARIMA", "AutoETS", "AutoTheta", "SeasonalNaive")


def run_statsforecast_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    model_name: str,
    estimator: str,
    horizon: int,
    seasonality: int,
    freq: str | int,
    n_jobs: int,
    max_train_length: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> ModelRunResult:
    try:
        import statsforecast
        from statsforecast import StatsForecast
        from statsforecast import models as sf_models
    except Exception as exc:
        return _failed_result(model_name, exc, "StatsForecast import failed.")

    try:
        if estimator not in _STATSFORECAST_ESTIMATORS:
            raise ValueError(
                f"Unsupported statsforecast estimator {estimator!r}; "
                f"expected one of {_STATSFORECAST_ESTIMATORS}."
            )
        estimator_kwargs = {"season_length": seasonality}
        estimator_kwargs.update(model_kwargs or {})
        model = getattr(sf_models, estimator)(**estimator_kwargs)
        sf = StatsForecast(models=[model], freq=freq, n_jobs=n_jobs)
        train = train_df.loc[:, ["unique_id", "ds", "y"]].copy()
        train["unique_id"] = train["unique_id"].astype(str)
        if max_train_length is not None:
            train = (
                train.sort_values(["unique_id", "ds"], kind="mergesort")
                .groupby("unique_id", sort=True)
                .tail(int(max_train_length))
                .reset_index(drop=True)
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="statsforecast")
            forecast_df = sf.forecast(df=train, h=horizon)
        prediction_columns = [col for col in forecast_df.columns if col not in {"unique_id", "ds"}]
        if not prediction_columns:
            raise ValueError("StatsForecast returned no prediction column.")
        prediction_column = prediction_columns[0]

        test = test_df.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()
        forecast_df = forecast_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        if len(test) != len(forecast_df):
            raise ValueError(
                f"{estimator} returned {len(forecast_df)} rows; expected {len(test)} test rows."
            )
        if test["unique_id"].astype(str).tolist() != forecast_df["unique_id"].astype(str).tolist():
            raise ValueError(f"{estimator} forecast unique_id order does not match test order.")

        y_pred = forecast_df[prediction_column].to_numpy(dtype=float)
        train_nonnegative = (
            train_df.groupby("unique_id", sort=True)["y"].min().astype(float).ge(0.0).to_dict()
        )
        rows: list[dict[str, object]] = []
        for row, pred in zip(test.to_dict("records"), y_pred, strict=True):
            unique_id = str(row["unique_id"])
            y_hat = float(pred)
            if train_nonnegative.get(unique_id, False):
                y_hat = max(0.0, y_hat)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "unique_id": unique_id,
                    "horizon": int(row.get("horizon", 0)) if "horizon" in row else None,
                    "ds": row["ds"],
                    "y_true": float(row["y"]),
                    "y_pred": y_hat,
                }
            )

        forecasts = pd.DataFrame(rows)
        forecasts["horizon"] = forecasts.groupby("unique_id").cumcount() + 1
        return ModelRunResult(
            model=model_name,
            status="ok",
            forecasts=forecasts.loc[
                :, ["dataset", "model", "unique_id", "horizon", "ds", "y_true", "y_pred"]
            ],
            details={
                "estimator": f"statsforecast.models.{estimator}",
                "statsforecast_version": getattr(statsforecast, "__version__", "unknown"),
                "season_length": seasonality,
                "freq": freq,
                "n_jobs": n_jobs,
                "max_train_length": max_train_length,
                "model_kwargs": estimator_kwargs,
            },
        )
    except Exception as exc:
        return _failed_result(model_name, exc, f"{estimator} fit/forecast failed.")


def run_auto_arima(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    horizon: int,
    seasonality: int,
    freq: str | int,
    n_jobs: int,
    model_kwargs: dict[str, Any] | None = None,
    model_name: str = "auto_arima",
    max_train_length: int | None = None,
) -> ModelRunResult:
    return run_statsforecast_model(
        train_df,
        test_df,
        dataset_name,
        model_name=model_name,
        estimator="AutoARIMA",
        horizon=horizon,
        seasonality=seasonality,
        freq=freq,
        n_jobs=n_jobs,
        max_train_length=max_train_length,
        model_kwargs=model_kwargs,
    )


def run_seasonal_naive(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    horizon: int,
    seasonality: int,
    freq: str | int,
    n_jobs: int,
    model_kwargs: dict[str, Any] | None = None,
    model_name: str = "seasonal_naive",
    max_train_length: int | None = None,
) -> ModelRunResult:
    return run_statsforecast_model(
        train_df,
        test_df,
        dataset_name,
        model_name=model_name,
        estimator="SeasonalNaive",
        horizon=horizon,
        seasonality=seasonality,
        freq=freq,
        n_jobs=n_jobs,
        max_train_length=max_train_length,
        model_kwargs=model_kwargs,
    )


def run_auto_ets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    horizon: int,
    seasonality: int,
    freq: str | int,
    n_jobs: int,
    model_kwargs: dict[str, Any] | None = None,
    model_name: str = "auto_ets",
    max_train_length: int | None = None,
) -> ModelRunResult:
    return run_statsforecast_model(
        train_df,
        test_df,
        dataset_name,
        model_name=model_name,
        estimator="AutoETS",
        horizon=horizon,
        seasonality=seasonality,
        freq=freq,
        n_jobs=n_jobs,
        max_train_length=max_train_length,
        model_kwargs=model_kwargs,
    )


def run_auto_theta(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    horizon: int,
    seasonality: int,
    freq: str | int,
    n_jobs: int,
    model_kwargs: dict[str, Any] | None = None,
    model_name: str = "auto_theta",
    max_train_length: int | None = None,
) -> ModelRunResult:
    return run_statsforecast_model(
        train_df,
        test_df,
        dataset_name,
        model_name=model_name,
        estimator="AutoTheta",
        horizon=horizon,
        seasonality=seasonality,
        freq=freq,
        n_jobs=n_jobs,
        max_train_length=max_train_length,
        model_kwargs=model_kwargs,
    )


def _prophet_fit_predict_worker(
    payload: tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
) -> tuple[str, np.ndarray]:
    """Fit Prophet on one series and predict the future timestamps.

    Top-level function with a plain-array payload so it pickles under the
    multiprocessing "spawn" start method.
    """
    unique_id, train_ds, train_y, future_ds, prophet_kwargs = payload

    import logging

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    import pandas as pd
    from prophet import Prophet

    model_kwargs = {"uncertainty_samples": 0}
    model_kwargs.update(prophet_kwargs)
    model = Prophet(**model_kwargs)
    model.fit(pd.DataFrame({"ds": pd.to_datetime(train_ds), "y": train_y}))
    prediction = model.predict(pd.DataFrame({"ds": pd.to_datetime(future_ds)}))
    return unique_id, prediction["yhat"].to_numpy(dtype=float)


def run_prophet(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    horizon: int,
    time_freq: str,
    model_name: str = "prophet",
    max_train_length: int | None = 4320,
    n_jobs: int = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> ModelRunResult:
    try:
        import prophet
    except Exception as exc:
        return _failed_result(model_name, exc, "Prophet import failed.")

    try:
        from timestransfer.data import ensure_datetime_ds

        train_dt, test_dt = ensure_datetime_ds(train_df, test_df, time_freq=time_freq)

        test_dt_groups = {
            str(unique_id): group.sort_values("ds")
            for unique_id, group in test_dt.groupby("unique_id", sort=True)
        }
        payloads: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
        train_nonnegative: dict[str, bool] = {}
        sorted_train = train_dt.sort_values(["unique_id", "ds"], kind="mergesort")
        for unique_id, group in sorted_train.groupby("unique_id", sort=True):
            unique_id = str(unique_id)
            if max_train_length is not None:
                group = group.tail(int(max_train_length))
            values = group["y"].to_numpy(dtype=float)
            train_nonnegative[unique_id] = bool(np.nanmin(values) >= 0.0)
            payloads.append(
                (
                    unique_id,
                    group["ds"].to_numpy(),
                    values,
                    test_dt_groups[unique_id]["ds"].to_numpy(),
                    dict(model_kwargs or {}),
                )
            )

        if n_jobs > 1:
            import multiprocessing

            with multiprocessing.get_context("spawn").Pool(processes=n_jobs) as pool:
                worker_results = pool.map(_prophet_fit_predict_worker, payloads)
        else:
            worker_results = [_prophet_fit_predict_worker(payload) for payload in payloads]

        predictions = {unique_id: yhat for unique_id, yhat in worker_results}

        rows: list[dict[str, object]] = []
        test_groups = {
            str(unique_id): group.sort_values("ds").reset_index(drop=True)
            for unique_id, group in test_df.groupby("unique_id", sort=True)
        }
        for unique_id, test_group in test_groups.items():
            yhat = predictions[unique_id]
            if len(yhat) != horizon:
                raise ValueError(
                    f"Prophet returned {len(yhat)} predictions for series {unique_id!r}; "
                    f"expected {horizon}."
                )
            for h in range(1, horizon + 1):
                y_pred = float(yhat[h - 1])
                if train_nonnegative[unique_id]:
                    y_pred = max(0.0, y_pred)
                test_row = test_group.iloc[h - 1]
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "unique_id": unique_id,
                        "horizon": h,
                        "ds": test_row["ds"],
                        "y_true": float(test_row["y"]),
                        "y_pred": y_pred,
                    }
                )

        return ModelRunResult(
            model=model_name,
            status="ok",
            forecasts=pd.DataFrame(rows),
            details={
                "estimator": "prophet.Prophet",
                "prophet_version": getattr(prophet, "__version__", "unknown"),
                "time_freq": time_freq,
                "max_train_length": max_train_length,
                "n_jobs": n_jobs,
                "uncertainty_samples": 0,
                "model_kwargs": dict(model_kwargs or {}),
            },
        )
    except Exception as exc:
        return _failed_result(model_name, exc, "Prophet fit/forecast failed.")


def run_chronos_bolt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    horizon: int,
    model_name: str = "chronos_bolt",
    model_id: str = "amazon/chronos-bolt-base",
    context_length: int = 2048,
    batch_size: int = 64,
    device_map: str = "cuda",
) -> ModelRunResult:
    try:
        import chronos
        from chronos import BaseChronosPipeline
    except Exception as exc:
        return _failed_result(model_name, exc, "Chronos import failed.")

    try:
        import torch

        pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        contexts: list[Any] = []
        unique_ids: list[str] = []
        train_nonnegative: dict[str, bool] = {}
        sorted_train = train_df.sort_values(["unique_id", "ds"], kind="mergesort")
        for unique_id, group in sorted_train.groupby("unique_id", sort=True):
            values = group["y"].to_numpy(dtype=np.float32)
            contexts.append(torch.tensor(values[-context_length:]))
            unique_ids.append(str(unique_id))
            train_nonnegative[str(unique_id)] = bool(np.nanmin(values) >= 0.0)

        batches: list[np.ndarray] = []
        for start in range(0, len(contexts), batch_size):
            quantiles, _ = pipeline.predict_quantiles(
                context=contexts[start : start + batch_size],
                prediction_length=horizon,
                quantile_levels=[0.5],
                limit_prediction_length=False,
            )
            batches.append(quantiles[..., 0].to(torch.float32).cpu().numpy())
        point_forecast = np.concatenate(batches, axis=0).astype(float)
        if point_forecast.shape != (len(unique_ids), horizon):
            raise ValueError(
                "Chronos returned unexpected forecast shape "
                f"{point_forecast.shape}; expected {(len(unique_ids), horizon)}."
            )

        rows: list[dict[str, object]] = []
        test_groups = {
            str(unique_id): group.sort_values("ds").reset_index(drop=True)
            for unique_id, group in test_df.groupby("unique_id", sort=True)
        }
        for series_idx, unique_id in enumerate(unique_ids):
            test_group = test_groups[unique_id]
            for h in range(1, horizon + 1):
                y_pred = float(point_forecast[series_idx, h - 1])
                if train_nonnegative[unique_id]:
                    y_pred = max(0.0, y_pred)
                test_row = test_group.iloc[h - 1]
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "unique_id": unique_id,
                        "horizon": h,
                        "ds": test_row["ds"],
                        "y_true": float(test_row["y"]),
                        "y_pred": y_pred,
                    }
                )

        return ModelRunResult(
            model=model_name,
            status="ok",
            forecasts=pd.DataFrame(rows),
            details={
                "estimator": "chronos.BaseChronosPipeline",
                "chronos_version": getattr(chronos, "__version__", "unknown"),
                "model_id": model_id,
                "context_length": context_length,
                "batch_size": batch_size,
                "point_forecast": "median",
                "autoregressive_beyond_64": horizon > 64,
            },
        )
    except Exception as exc:
        return _failed_result(model_name, exc, "Chronos load/forecast failed.")


def _timesfm_forecast_config_kwargs(
    *,
    max_context: int,
    max_horizon: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = {
        "max_context": max_context,
        "max_horizon": max_horizon,
        "normalize_inputs": True,
        "use_continuous_quantile_head": True,
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "fix_quantile_crossing": True,
    }
    kwargs.update(overrides or {})
    return kwargs


def _make_timesfm_forecast_config(
    timesfm_module: Any,
    *,
    max_context: int,
    max_horizon: int,
    overrides: dict[str, Any] | None = None,
) -> Any:
    kwargs = _timesfm_forecast_config_kwargs(
        max_context=max_context,
        max_horizon=max_horizon,
        overrides=overrides,
    )
    try:
        return timesfm_module.ForecastConfig(**kwargs)
    except TypeError:
        return timesfm_module.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
        )


def _forecast_frame(
    *,
    dataset_name: str,
    model: str,
    test_index: pd.DataFrame,
    y_pred_scaled: np.ndarray,
) -> pd.DataFrame:
    frame = test_index.loc[:, ["unique_id", "horizon", "ds", "y_true", "scale", "train_nonnegative"]].copy()
    frame["dataset"] = dataset_name
    frame["model"] = model
    frame["y_pred"] = y_pred_scaled.astype(float) * frame["scale"].astype(float).to_numpy()
    nonnegative_mask = frame["train_nonnegative"].astype(bool)
    frame.loc[nonnegative_mask, "y_pred"] = frame.loc[nonnegative_mask, "y_pred"].clip(lower=0.0)
    return frame.loc[:, ["dataset", "model", "unique_id", "horizon", "ds", "y_true", "y_pred"]]


def _failed_result(model: str, exc: Exception, message: str) -> ModelRunResult:
    return ModelRunResult(
        model=model,
        status="failed",
        forecasts=None,
        details={
            "message": message,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )
