"""Post-hoc combination of saved member forecasts under a rolling-origin protocol.

Window ``k`` is the k-th most recent evaluation window (0 = final holdout; windows are
non-overlapping and window k + 1 ends where window k starts). Weights or selections used
for window ``k`` are fitted only on windows ``> k``: their targets were observed and
their MASE scales were computed before window k's forecast origin.

Learned weights live in MASE-scaled space: member forecasts and targets are divided by
the per-series, per-window seasonal-naive scale before fitting, so every series counts
on the same footing as in panel MASE. Rows without a valid scale are not used for
fitting but are still combined.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

KEY_COLUMNS = ["dataset", "unique_id", "window", "horizon"]
FORECAST_COLUMNS = [
    "dataset",
    "model",
    "unique_id",
    "horizon",
    "ds",
    "y_true",
    "y_pred",
    "window",
    "forecast_horizon",
]
LEVELS = ("global", "task", "series")
META_FEATURE_COLUMNS = (
    "length",
    "cv",
    "acf1",
    "acf_seasonal",
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
)


class CombinationTable:
    """Member point forecasts in wide form, one row per (dataset, unique_id, window, horizon).

    ``frame`` holds KEY_COLUMNS, ``ds``, ``forecast_horizon``, ``y_true``, ``mase_scale``
    and one column per member; the array attributes are aligned with its rows.
    """

    def __init__(self, frame: pd.DataFrame, members: Sequence[str]):
        if not members:
            raise ValueError("A combination table needs at least one member.")
        frame = frame.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
        self.frame = frame
        self.members = tuple(members)
        self.predictions = frame.loc[:, list(self.members)].to_numpy(dtype=float)
        self.y = frame["y_true"].to_numpy(dtype=float)
        self.scale = frame["mase_scale"].to_numpy(dtype=float)
        self.window = frame["window"].to_numpy(dtype=int)
        self.horizon = frame["horizon"].to_numpy(dtype=int)
        self.forecast_horizon = frame["forecast_horizon"].to_numpy(dtype=int)
        self.valid_scale = np.isfinite(self.scale) & (self.scale > 0) & np.isfinite(self.y)
        safe_scale = np.where(self.valid_scale, self.scale, 1.0)
        self.scaled_predictions = self.predictions / safe_scale[:, None]
        self.scaled_y = self.y / safe_scale

        dataset_codes, datasets = pd.factorize(frame["dataset"], sort=True)
        self.dataset_codes = dataset_codes.astype(int)
        self.datasets = tuple(str(name) for name in datasets)
        series_keys = frame["dataset"].astype(str) + "\x1f" + frame["unique_id"].astype(str)
        self.series_codes = pd.factorize(series_keys, sort=True)[0].astype(int)
        self.series_window_codes = pd.factorize(
            series_keys + "\x1f" + frame["window"].astype(str), sort=False
        )[0].astype(int)

        self.n_windows: dict[str, int] = {}
        for dataset, windows in frame.groupby("dataset", sort=True)["window"]:
            observed = np.unique(windows.to_numpy(dtype=int))
            if not np.array_equal(observed, np.arange(len(observed))):
                raise ValueError(f"Windows of {dataset!r} are not contiguous from 0: {observed}.")
            self.n_windows[str(dataset)] = int(len(observed))
        self._group_code_cache: dict[tuple[str, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def group_codes(self, level: str, n_bins: int = 1) -> np.ndarray:
        """Integer weight-group id per row: level (global/task/series) x horizon bin."""
        key = (level, n_bins)
        if key not in self._group_code_cache:
            if level == "global":
                base = np.zeros(len(self), dtype=int)
            elif level == "task":
                base = self.dataset_codes
            elif level == "series":
                base = self.series_codes
            else:
                raise ValueError(f"Unknown level {level!r}; expected one of {LEVELS}.")
            self._group_code_cache[key] = base * n_bins + horizon_bins(
                self.horizon, self.forecast_horizon, n_bins
            )
        return self._group_code_cache[key]


def horizon_bins(horizon: np.ndarray, forecast_horizon: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin of each step when 1..H is cut into ``n_bins`` equal consecutive parts."""
    horizon = np.asarray(horizon, dtype=int)
    forecast_horizon = np.asarray(forecast_horizon, dtype=int)
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    if n_bins > int(np.min(forecast_horizon, initial=n_bins)):
        raise ValueError(f"n_bins {n_bins} exceeds the shortest forecast horizon.")
    return ((horizon - 1) * n_bins) // forecast_horizon


def build_combination_table(
    forecasts: pd.DataFrame,
    mase_scales: pd.DataFrame,
    members: Sequence[str],
) -> CombinationTable:
    """Pivot long benchmark forecasts (FORECAST_COLUMNS) to a CombinationTable.

    Every member must cover every row of every dataset; targets must agree across members.
    """
    members = list(dict.fromkeys(members))
    missing = sorted(set(members) - set(forecasts["model"].astype(str)))
    if missing:
        raise ValueError(f"No forecasts for member(s) {missing}.")
    long = forecasts.loc[forecasts["model"].isin(members), FORECAST_COLUMNS].copy()
    long["unique_id"] = long["unique_id"].astype(str)
    long["ds"] = long["ds"].astype(str)
    if long.duplicated([*KEY_COLUMNS, "model"]).any():
        raise ValueError("Duplicate (dataset, model, unique_id, window, horizon) forecasts.")

    wide = long.pivot(index=KEY_COLUMNS, columns="model", values="y_pred")
    incomplete = wide.isna().groupby(level="dataset").any()
    gaps = [
        f"{dataset}: {sorted(incomplete.columns[row].tolist())}"
        for dataset, row in incomplete.iterrows()
        if row.any()
    ]
    if gaps:
        raise ValueError("Members do not cover every row of their datasets: " + "; ".join(gaps))

    targets = long.pivot(index=KEY_COLUMNS, columns="model", values="y_true")
    spread = (targets.max(axis=1) - targets.min(axis=1)).abs()
    if (spread > 0).any():
        raise ValueError("Members disagree on y_true; forecasts come from different data.")

    meta = long.groupby(KEY_COLUMNS, sort=False)[["ds", "forecast_horizon"]].first()
    frame = wide.loc[:, members].join(meta).join(targets.iloc[:, 0].rename("y_true"))
    frame = frame.reset_index()
    scales = mase_scales.loc[:, ["dataset", "window", "unique_id", "mase_scale"]].copy()
    scales["unique_id"] = scales["unique_id"].astype(str)
    frame = frame.merge(
        scales, on=["dataset", "window", "unique_id"], how="left", validate="many_to_one"
    )
    columns = [*KEY_COLUMNS, "ds", "forecast_horizon", "y_true", "mase_scale", *members]
    return CombinationTable(frame.loc[:, columns], members)


def load_forecasts(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate long forecast parquet files (point columns only)."""
    frames = [pd.read_parquet(path, columns=FORECAST_COLUMNS) for path in paths]
    if not frames:
        raise ValueError("No forecast files given.")
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts["unique_id"] = forecasts["unique_id"].astype(str)
    forecasts["ds"] = forecasts["ds"].astype(str)
    return forecasts


def load_mase_scales(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate mase_scales.csv files; identical repeated rows are merged."""
    frames = [pd.read_csv(path, dtype={"unique_id": str}) for path in paths]
    if not frames:
        raise ValueError("No MASE scale files given.")
    scales = pd.concat(frames, ignore_index=True).drop_duplicates()
    if scales.duplicated(["dataset", "window", "unique_id"]).any():
        raise ValueError("Conflicting MASE scales for the same (dataset, window, unique_id).")
    return scales


@dataclass(frozen=True)
class RollingProtocol:
    """Which past windows may inform window k: k+1 .. k+max_past_windows (all if None).

    Window k is evaluated only if at least ``min_past_windows`` such windows exist.
    """

    min_past_windows: int = 3
    max_past_windows: int | None = None

    def __post_init__(self) -> None:
        if self.min_past_windows < 1:
            raise ValueError("min_past_windows must be at least 1.")
        if self.max_past_windows is not None and self.max_past_windows < self.min_past_windows:
            raise ValueError("max_past_windows must be at least min_past_windows.")

    def training_windows(self, window: int, n_windows: int) -> list[int]:
        last = n_windows - 1
        if self.max_past_windows is not None:
            last = min(last, window + self.max_past_windows)
        return list(range(window + 1, last + 1))

    def evaluation_windows(self, n_windows: int) -> list[int]:
        return [
            window
            for window in range(n_windows)
            if len(self.training_windows(window, n_windows)) >= self.min_past_windows
        ]

    def evaluation_mask(self, table: CombinationTable) -> np.ndarray:
        mask = np.zeros(len(table), dtype=bool)
        for code, dataset in enumerate(table.datasets):
            windows = self.evaluation_windows(table.n_windows[dataset])
            mask |= (table.dataset_codes == code) & np.isin(table.window, windows)
        return mask

    def training_mask(self, table: CombinationTable, window: int) -> np.ndarray:
        mask = table.window > window
        if self.max_past_windows is not None:
            mask &= table.window <= window + self.max_past_windows
        return mask


@dataclass(frozen=True)
class CombineOutput:
    predictions: np.ndarray
    weights: pd.DataFrame | None = None


class Combiner(Protocol):
    name: str

    def combine(
        self, table: CombinationTable, train_idx: np.ndarray, target_idx: np.ndarray
    ) -> CombineOutput:
        """Combined forecasts for ``target_idx`` (one window) using only ``train_idx``."""
        ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CombinationResult:
    forecasts: pd.DataFrame
    weights: pd.DataFrame


def rolling_combine(
    table: CombinationTable, combiner: Combiner, protocol: RollingProtocol
) -> CombinationResult:
    """Apply ``combiner`` to every evaluated window, fitting only on that window's past."""
    evaluated = protocol.evaluation_mask(table)
    predictions = np.full(len(table), np.nan)
    weight_frames: list[pd.DataFrame] = []
    for window in np.unique(table.window[evaluated]):
        target_idx = np.flatnonzero(evaluated & (table.window == window))
        train_idx = np.flatnonzero(protocol.training_mask(table, int(window)))
        output = combiner.combine(table, train_idx, target_idx)
        if output.predictions.shape != target_idx.shape:
            raise ValueError(f"{combiner.name} returned {output.predictions.shape} predictions.")
        predictions[target_idx] = output.predictions
        if output.weights is not None:
            weight_frames.append(output.weights.assign(window=int(window)))

    rows = table.frame.loc[evaluated, ["dataset", "unique_id", "horizon", "ds", "y_true"]]
    forecasts = rows.assign(
        model=combiner.name,
        y_pred=predictions[evaluated],
        window=table.window[evaluated],
        forecast_horizon=table.forecast_horizon[evaluated],
    ).loc[:, FORECAST_COLUMNS]
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    if not weights.empty:
        weights.insert(0, "method", combiner.name)
    return CombinationResult(forecasts=forecasts.reset_index(drop=True), weights=weights)


# --- weight fitters: (scaled member forecasts (n, M), scaled targets (n,)) -> weights (M,)

WeightFitter = Callable[[np.ndarray, np.ndarray], np.ndarray]


def select_best(scaled_predictions: np.ndarray, scaled_y: np.ndarray) -> np.ndarray:
    """One-hot weight on the member with the lowest mean scaled absolute error."""
    errors = np.mean(np.abs(scaled_predictions - scaled_y[:, None]), axis=0)
    weights = np.zeros(scaled_predictions.shape[1])
    weights[int(np.argmin(errors))] = 1.0
    return weights


@dataclass(frozen=True)
class InverseErrorWeights:
    """w_m proportional to (mean scaled absolute error of m) ** -power, optionally top-k only."""

    power: float
    top_k: int | None = None

    def __call__(self, scaled_predictions: np.ndarray, scaled_y: np.ndarray) -> np.ndarray:
        errors = np.mean(np.abs(scaled_predictions - scaled_y[:, None]), axis=0)
        errors = np.maximum(errors, 1e-12)
        weights = errors ** (-self.power)
        if self.top_k is not None and self.top_k < len(weights):
            weights[np.argsort(errors, kind="stable")[self.top_k :]] = 0.0
        return weights / weights.sum()


def nnls_weights(scaled_predictions: np.ndarray, scaled_y: np.ndarray) -> np.ndarray:
    """Nonnegative least squares, renormalized to sum to 1 (equal weights if all zero)."""
    weights, _ = nnls(scaled_predictions, scaled_y)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(scaled_predictions.shape[1], 1.0 / scaled_predictions.shape[1])
    return weights / total


# --- combiners


def _equal_combiner_output(table: CombinationTable, target_idx: np.ndarray, how: str):
    values = table.predictions[target_idx]
    return values.mean(axis=1) if how == "mean" else np.median(values, axis=1)


@dataclass(frozen=True)
class EqualMean:
    name: str = "equal_mean"

    def combine(self, table, train_idx, target_idx) -> CombineOutput:
        return CombineOutput(_equal_combiner_output(table, target_idx, "mean"))

    def describe(self) -> dict[str, Any]:
        return {"kind": "equal_mean"}


@dataclass(frozen=True)
class Median:
    name: str = "median"

    def combine(self, table, train_idx, target_idx) -> CombineOutput:
        return CombineOutput(_equal_combiner_output(table, target_idx, "median"))

    def describe(self) -> dict[str, Any]:
        return {"kind": "median"}


@dataclass(frozen=True)
class LinearCombiner:
    """Per-group convex weights from ``fitter`` on the group's past rows.

    Groups are ``level`` (global, task, series) x ``n_bins`` horizon bins. Groups with no
    past rows get equal weights. ``recent_windows`` keeps only the most recent past
    windows; ``shrinkage`` moves the fitted weights toward equal weights.
    """

    name: str
    fitter: WeightFitter
    level: str = "task"
    n_bins: int = 1
    recent_windows: int | None = None
    shrinkage: float = 0.0

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"Unknown level {self.level!r}; expected one of {LEVELS}.")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1].")

    def combine(self, table, train_idx, target_idx) -> CombineOutput:
        if self.recent_windows is not None:
            window = int(table.window[target_idx[0]])
            train_idx = train_idx[table.window[train_idx] <= window + self.recent_windows]
        train_idx = train_idx[table.valid_scale[train_idx]]

        codes = table.group_codes(self.level, self.n_bins)
        groups, first_target, target_group = np.unique(
            codes[target_idx], return_index=True, return_inverse=True
        )
        train_codes = codes[train_idx]
        order = np.argsort(train_codes, kind="stable")
        sorted_codes = train_codes[order]
        starts = np.searchsorted(sorted_codes, groups, side="left")
        ends = np.searchsorted(sorted_codes, groups, side="right")

        n_members = len(table.members)
        weights = np.empty((len(groups), n_members))
        for group, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if end == start:
                weights[group] = 1.0 / n_members
                continue
            rows = train_idx[order[start:end]]
            fitted = self.fitter(table.scaled_predictions[rows], table.scaled_y[rows])
            weights[group] = (1.0 - self.shrinkage) * fitted + self.shrinkage / n_members

        predictions = np.einsum(
            "ij,ij->i", table.predictions[target_idx], weights[target_group.ravel()]
        )
        described = _describe_groups(table, target_idx[first_target], self.level, self.n_bins)
        described["n_train_rows"] = ends - starts
        described["fallback"] = ends == starts
        for column, member in enumerate(table.members):
            described[f"w_{member}"] = weights[:, column]
        return CombineOutput(predictions, described)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "fitter": _fitter_name(self.fitter),
            "level": self.level,
            "n_bins": self.n_bins,
            "recent_windows": self.recent_windows,
            "shrinkage": self.shrinkage,
        }


def _fitter_name(fitter: WeightFitter) -> str | dict[str, Any]:
    if isinstance(fitter, InverseErrorWeights):
        return {"inverse_error": {"power": fitter.power, "top_k": fitter.top_k}}
    return getattr(fitter, "__name__", type(fitter).__name__)


def _describe_groups(
    table: CombinationTable, rows: np.ndarray, level: str, n_bins: int
) -> pd.DataFrame:
    frame = table.frame
    return pd.DataFrame(
        {
            "level": level,
            "dataset": "all" if level == "global" else frame["dataset"].to_numpy()[rows],
            "unique_id": frame["unique_id"].to_numpy()[rows] if level == "series" else "",
            "horizon_bin": horizon_bins(table.horizon[rows], table.forecast_horizon[rows], n_bins),
            "n_bins": n_bins,
        }
    )


@dataclass
class FeatureWeightedCombiner:
    """FFORMA-style meta-learner (Montero-Manso et al., 2020) with a linear softmax model.

    Each past (series, window) is a training example: features of the series history
    before that window, and each member's series-window MASE divided by the mean over
    members. The model outputs softmax weights over members and is fitted by minimizing
    the weight-averaged relative error plus an L2 penalty, as in FFORMA's objective. With
    ``leave_one_dataset_out`` a target dataset's own examples are never used, so weights
    for a task come only from the other tasks.
    """

    name: str
    features: pd.DataFrame
    feature_columns: tuple[str, ...] = META_FEATURE_COLUMNS
    l2: float = 1e-2
    leave_one_dataset_out: bool = True
    _feature_cache: tuple[CombinationTable, np.ndarray] | None = field(default=None, repr=False)

    def combine(self, table, train_idx, target_idx) -> CombineOutput:
        train_idx = train_idx[table.valid_scale[train_idx]]
        features = self._series_window_features(table)

        example_codes, example_rows = np.unique(
            table.series_window_codes[train_idx], return_inverse=True
        )
        n_members = len(table.members)
        abs_errors = np.abs(table.scaled_predictions[train_idx] - table.scaled_y[train_idx, None])
        counts = np.bincount(example_rows, minlength=len(example_codes)).astype(float)
        errors = (
            np.column_stack(
                [
                    np.bincount(
                        example_rows, weights=abs_errors[:, m], minlength=len(example_codes)
                    )
                    for m in range(n_members)
                ]
            )
            / counts[:, None]
        )
        example_dataset = table.dataset_codes[train_idx][
            np.unique(example_rows, return_index=True)[1]
        ]
        relative = errors / np.maximum(errors.mean(axis=1, keepdims=True), 1e-12)

        target_codes, first_target, target_group = np.unique(
            table.series_window_codes[target_idx], return_index=True, return_inverse=True
        )
        target_dataset = table.dataset_codes[target_idx][first_target]
        weights = np.full((len(target_codes), n_members), 1.0 / n_members)
        n_examples = np.zeros(len(target_codes), dtype=int)
        for dataset in np.unique(target_dataset):
            use = (
                example_dataset != dataset
                if self.leave_one_dataset_out
                else np.ones(len(example_codes), dtype=bool)
            )
            selected = target_dataset == dataset
            n_examples[selected] = int(use.sum())
            if not use.any():
                continue
            model = _fit_softmax_weights(features[example_codes[use]], relative[use], l2=self.l2)
            weights[selected] = model.predict(features[target_codes[selected]])

        predictions = np.einsum(
            "ij,ij->i", table.predictions[target_idx], weights[target_group.ravel()]
        )
        rows = target_idx[first_target]
        described = pd.DataFrame(
            {
                "level": "series",
                "dataset": table.frame["dataset"].to_numpy()[rows],
                "unique_id": table.frame["unique_id"].to_numpy()[rows],
                "horizon_bin": 0,
                "n_bins": 1,
                "n_train_rows": n_examples,
                "fallback": n_examples == 0,
            }
        )
        for column, member in enumerate(table.members):
            described[f"w_{member}"] = weights[:, column]
        return CombineOutput(predictions, described)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "fforma_linear_softmax",
            "feature_columns": list(self.feature_columns),
            "l2": self.l2,
            "leave_one_dataset_out": self.leave_one_dataset_out,
        }

    def _series_window_features(self, table: CombinationTable) -> np.ndarray:
        """Raw feature matrix indexed by the table's series-window codes (cached per table)."""
        if self._feature_cache is None or self._feature_cache[0] is not table:
            frame = table.frame
            first = np.unique(table.series_window_codes, return_index=True)[1]
            index = frame.loc[first, ["dataset", "unique_id", "window"]].reset_index(drop=True)
            features = self.features.copy()
            features["unique_id"] = features["unique_id"].astype(str)
            joined = index.merge(
                features.loc[:, ["dataset", "unique_id", "window", *self.feature_columns]],
                on=["dataset", "unique_id", "window"],
                how="left",
                validate="one_to_one",
            )
            missing = joined.loc[:, self.feature_columns].isna().all(axis=1)
            if missing.any():
                example = joined.loc[missing, ["dataset", "unique_id", "window"]].iloc[0].tolist()
                raise ValueError(
                    f"No series features for {int(missing.sum())} series-windows, e.g. {example}."
                )
            matrix = joined.loc[:, self.feature_columns].to_numpy(dtype=float)
            self._feature_cache = (table, matrix)
        return self._feature_cache[1]


@dataclass(frozen=True)
class _SoftmaxModel:
    weights: np.ndarray
    bias: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    fill: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return _softmax(_standardize(features, self) @ self.weights + self.bias)


def _transform_features(features: np.ndarray) -> np.ndarray:
    # Heavy-tailed, nonnegative-in-practice features enter on a signed log scale.
    return np.sign(features) * np.log1p(np.abs(features))


def _standardize(features: np.ndarray, model: _SoftmaxModel) -> np.ndarray:
    values = _transform_features(features)
    values = np.where(np.isnan(values), model.fill, values)
    return (values - model.center) / model.spread


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _fit_softmax_weights(features: np.ndarray, errors: np.ndarray, *, l2: float) -> _SoftmaxModel:
    """Minimize mean_i sum_m softmax(x_i W + b)_m e_im + l2 ||W||^2 (start: equal weights)."""
    values = _transform_features(features)
    fill = np.nanmedian(values, axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    values = np.where(np.isnan(values), fill, values)
    center = values.mean(axis=0)
    spread = values.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    x = (values - center) / spread
    n_examples, n_features = x.shape
    n_members = errors.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        weights = theta[: n_features * n_members].reshape(n_features, n_members)
        bias = theta[n_features * n_members :]
        probabilities = _softmax(x @ weights + bias)
        expected = (probabilities * errors).sum(axis=1)
        loss = expected.mean() + l2 * np.sum(weights**2)
        grad_logits = probabilities * (errors - expected[:, None]) / n_examples
        grad = np.concatenate([(x.T @ grad_logits + 2 * l2 * weights).ravel(), grad_logits.sum(0)])
        return float(loss), grad

    theta0 = np.zeros(n_features * n_members + n_members)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    theta = result.x
    return _SoftmaxModel(
        weights=theta[: n_features * n_members].reshape(n_features, n_members),
        bias=theta[n_features * n_members :],
        center=center,
        spread=spread,
        fill=fill,
    )


def default_combiners(
    members: Sequence[str], features: pd.DataFrame | None = None
) -> list[Combiner]:
    """The Phase-B method grid; the meta-learner is included when features are given."""
    combiners: list[Combiner] = [
        EqualMean(),
        Median(),
        LinearCombiner("select_task", select_best, level="task"),
        LinearCombiner("select_series", select_best, level="series"),
        LinearCombiner("select_series_last", select_best, level="series", recent_windows=1),
    ]
    for power in (1, 2, 4):
        combiners.append(
            LinearCombiner(f"inv_mase_p{power}", InverseErrorWeights(power), level="series")
        )
    if len(members) > 2:
        combiners.append(
            LinearCombiner("inv_mase_p2_top2", InverseErrorWeights(2, top_k=2), level="series")
        )
    combiners += [
        LinearCombiner("nnls_global", nnls_weights, level="global"),
        LinearCombiner("nnls_task", nnls_weights, level="task"),
        LinearCombiner("nnls_task_h3", nnls_weights, level="task", n_bins=3),
        LinearCombiner("nnls_task_h6", nnls_weights, level="task", n_bins=6),
        LinearCombiner("nnls_series", nnls_weights, level="series"),
        LinearCombiner(
            "nnls_task_h3_shrink50", nnls_weights, level="task", n_bins=3, shrinkage=0.5
        ),
    ]
    if features is not None:
        combiners.append(FeatureWeightedCombiner("fforma_lodo", features))
    return combiners


# --- series features per (dataset, window, unique_id) for the meta-learner


def rolling_series_features(
    config_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    datasets: Sequence[str] | None = None,
    max_points: int = 2048,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Series features of each benchmark window's training history.

    Mirrors the benchmark's series selection and rolling-origin splits: the training
    history of window k is the series up to ``k * stride + horizon`` steps before its
    end. Features use the last ``max_points`` of that history (``length`` is its full
    length).
    """
    from timestransfer.config import dataset_configs, load_config
    from timestransfer.data import load_dataset, select_series
    from timestransfer.series_features import FEATURE_COLUMNS

    config = load_config(config_path)
    project = config.get("project", {})
    data_dir = Path(data_dir if data_dir is not None else project.get("data_dir", "data"))
    frames: list[pd.DataFrame] = []
    for dataset_config in dataset_configs(config):
        name = str(dataset_config["name"])
        if datasets is not None and name not in datasets:
            continue
        horizon = int(dataset_config.get("horizon", 48))
        seasonality = int(dataset_config.get("seasonality", 24))
        n_windows = int(dataset_config.get("n_windows", project.get("n_windows", 1)))
        stride = dataset_config.get("window_stride", project.get("window_stride"))
        stride = horizon if stride is None else int(stride)
        df = select_series(
            load_dataset(data_dir=data_dir, dataset_config=dataset_config),
            dataset_config.get("series_limit", 32),
        )
        payloads = []
        for unique_id, group in df.groupby("unique_id", sort=True):
            values = group["y"].to_numpy(dtype=float)
            for window in range(n_windows):
                end = len(values) - window * stride - horizon
                payloads.append(
                    (
                        str(unique_id),
                        window,
                        values[max(0, end - max_points) : end],
                        end,
                        seasonality,
                    )
                )
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                results = list(pool.map(_features_worker, payloads, chunksize=64))
        else:
            results = [_features_worker(payload) for payload in payloads]
        frame = pd.DataFrame(results)
        frame.insert(0, "dataset", name)
        frames.append(frame.loc[:, ["dataset", "window", "unique_id", *FEATURE_COLUMNS]])
    features = pd.concat(frames, ignore_index=True)
    features["max_points"] = max_points
    return features


def _features_worker(payload: tuple[str, int, np.ndarray, int, int]) -> dict[str, object]:
    from timestransfer.series_features import features_from_values

    unique_id, window, values, length, seasonality = payload
    features = features_from_values(values, seasonality=seasonality, length=length)
    return {"unique_id": unique_id, "window": window, **features}
