from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def run_linear_regression(bundle: FeatureBundle, dataset_name: str) -> ModelRunResult:
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
        model="linear_regression",
        test_index=bundle.test_index,
        y_pred_scaled=y_pred_scaled,
    )
    return ModelRunResult(
        model="linear_regression",
        status="ok",
        forecasts=forecasts,
        details={"estimator": "sklearn.linear_model.LinearRegression"},
    )


def run_tabpfn(
    bundle: FeatureBundle,
    dataset_name: str,
    *,
    prediction_batch_size: int = 1024,
) -> ModelRunResult:
    try:
        import tabpfn
        from tabpfn import TabPFNRegressor
    except Exception as exc:
        return _failed_result("tabpfn", exc, "TabPFN import failed.")

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
            model="tabpfn",
            test_index=bundle.test_index,
            y_pred_scaled=y_pred_scaled,
        )
        return ModelRunResult(
            model="tabpfn",
            status="ok",
            forecasts=forecasts,
            details={
                "estimator": "tabpfn.TabPFNRegressor",
                "tabpfn_version": getattr(tabpfn, "__version__", "unknown"),
                "prediction_batch_size": prediction_batch_size,
            },
        )
    except Exception as exc:
        return _failed_result("tabpfn", exc, "TabPFN fit/predict failed.")


def run_timesfm_2p5(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    *,
    model_id: str,
    max_context: int,
    max_horizon: int,
    horizon: int,
) -> ModelRunResult:
    try:
        import timesfm
    except Exception as exc:
        return _failed_result("timesfm_2p5", exc, "TimesFM import failed.")

    try:
        if horizon > max_horizon:
            raise ValueError(f"Requested horizon {horizon} exceeds configured max_horizon {max_horizon}.")

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        config = _make_timesfm_forecast_config(timesfm, max_context=max_context, max_horizon=max_horizon)
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
                        "model": "timesfm_2p5",
                        "unique_id": unique_id,
                        "horizon": h,
                        "ds": int(test_row["ds"]),
                        "y_true": float(test_row["y"]),
                        "y_pred": y_pred,
                    }
                )

        return ModelRunResult(
            model="timesfm_2p5",
            status="ok",
            forecasts=pd.DataFrame(rows),
            details={
                "estimator": "timesfm.TimesFM_2p5_200M_torch",
                "model_id": model_id,
                "max_context": max_context,
                "max_horizon": max_horizon,
            },
        )
    except Exception as exc:
        return _failed_result("timesfm_2p5", exc, "TimesFM load/forecast failed.")


def _make_timesfm_forecast_config(timesfm_module: Any, *, max_context: int, max_horizon: int) -> Any:
    kwargs = {
        "max_context": max_context,
        "max_horizon": max_horizon,
        "normalize_inputs": True,
        "use_continuous_quantile_head": True,
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "fix_quantile_crossing": True,
    }
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
