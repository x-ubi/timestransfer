"""Phase C analyses of member and combined forecasts (diversity, granularity, data needs,
robustness and inference, failure modes).

Everything works on a CombinationTable whose "members" are the models being analysed,
so all models are compared on identical rows. Errors are in MASE-scaled space: rows
without a valid seasonal-naive scale are excluded, exactly as in panel MASE.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from timestransfer.combination import (
    CombinationTable,
    Combiner,
    RollingProtocol,
    horizon_bins,
    rolling_combine,
)

# --- accuracy at task level


def task_mase(table: CombinationTable) -> pd.DataFrame:
    """Panel MASE per (dataset, model) over the table's rows with a valid scale."""
    valid = table.valid_scale
    errors = np.abs(table.scaled_predictions - table.scaled_y[:, None])[valid]
    codes = table.dataset_codes[valid]
    counts = np.bincount(codes, minlength=len(table.datasets)).astype(float)
    rows = []
    for column, model in enumerate(table.members):
        sums = np.bincount(codes, weights=errors[:, column], minlength=len(table.datasets))
        for code, dataset in enumerate(table.datasets):
            mase = sums[code] / counts[code] if counts[code] else np.nan
            rows.append({"dataset": dataset, "model": model, "mase": mase})
    return pd.DataFrame(rows)


def relative_to_baseline(scores: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Add ``relative_mase`` = task MASE / baseline task MASE on the same rows."""
    base = scores[scores["model"] == baseline].set_index("dataset")["mase"]
    if base.empty:
        raise ValueError(f"Baseline {baseline!r} is not among the scored models.")
    out = scores.copy()
    out["relative_mase"] = out["mase"] / out["dataset"].map(base)
    return out


def aggregate_relative(relative: pd.DataFrame) -> pd.DataFrame:
    """Geometric mean of relative MASE over tasks, per model (sorted, best first)."""
    grouped = relative.groupby("model")["relative_mase"]
    out = pd.DataFrame(
        {
            "n_tasks": grouped.size(),
            "geomean_relative_mase": grouped.apply(lambda v: float(np.exp(np.log(v).mean()))),
        }
    )
    return out.sort_values("geomean_relative_mase").reset_index()


def regret_table(relative: pd.DataFrame, reference_models: Sequence[str]) -> pd.DataFrame:
    """Per task: relative MASE / that of the ex-post best of ``reference_models``."""
    pivot = relative.pivot(index="dataset", columns="model", values="relative_mase")
    best = pivot.loc[:, list(reference_models)].min(axis=1)
    regret = pivot.div(best, axis=0)
    return regret.reset_index().melt(id_vars="dataset", var_name="model", value_name="regret")


def regret_summary(regret: pd.DataFrame, threshold: float = 1.10) -> pd.DataFrame:
    grouped = regret.groupby("model")["regret"]
    out = pd.DataFrame(
        {
            "max_regret": grouped.max(),
            "mean_regret": grouped.mean(),
            f"tasks_above_{threshold:.2f}": grouped.apply(lambda v: int((v > threshold).sum())),
        }
    )
    return out.sort_values(["max_regret", "mean_regret"]).reset_index()


# --- inference


def window_losses(table: CombinationTable) -> pd.DataFrame:
    """Series-averaged MASE per (dataset, window, model): the per-origin loss for DM tests."""
    valid = table.valid_scale
    errors = np.abs(table.scaled_predictions - table.scaled_y[:, None])[valid]
    n_windows = int(table.window.max()) + 1
    cell = table.dataset_codes[valid] * n_windows + table.window[valid]
    size = len(table.datasets) * n_windows
    counts = np.bincount(cell, minlength=size).astype(float)
    present = np.flatnonzero(counts)
    frames = []
    for column, model in enumerate(table.members):
        sums = np.bincount(cell, weights=errors[:, column], minlength=size)
        frames.append(
            pd.DataFrame(
                {
                    "dataset": [table.datasets[c // n_windows] for c in present],
                    "window": present % n_windows,
                    "model": model,
                    "loss": sums[present] / counts[present],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def diebold_mariano(loss_differences: np.ndarray, max_lag: int | None = None) -> dict[str, float]:
    """Diebold-Mariano test of zero mean loss difference with a Newey-West long-run variance.

    Uses the Harvey-Leybourne-Newbold small-sample factor for one-step comparisons and a
    Student-t reference with T-1 degrees of freedom; ``max_lag`` defaults to floor(T^(1/3)).
    Negative statistics mean the first model has the lower loss.
    """
    d = np.asarray(loss_differences, dtype=float)
    n = len(d)
    if n < 3:
        return {
            "n": n,
            "mean_diff": float(np.mean(d)) if n else np.nan,
            "stat": np.nan,
            "p": np.nan,
        }
    lag = int(np.floor(n ** (1 / 3))) if max_lag is None else int(max_lag)
    centered = d - d.mean()
    variance = float(centered @ centered) / n
    for k in range(1, min(lag, n - 1) + 1):
        autocovariance = float(centered[k:] @ centered[:-k]) / n
        variance += 2 * (1 - k / (lag + 1)) * autocovariance
    if variance <= 0:
        variance = float(centered @ centered) / n
    if variance <= 0:
        return {"n": n, "mean_diff": float(d.mean()), "stat": np.nan, "p": np.nan}
    stat = d.mean() / np.sqrt(variance / n) * np.sqrt((n - 1) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"n": n, "mean_diff": float(d.mean()), "stat": float(stat), "p": float(p)}


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are left out and kept as NaN)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def dm_tests(
    losses: pd.DataFrame,
    models: Sequence[str],
    references: Sequence[str],
    best_member_pool: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-task DM tests of each model against each reference, Holm-adjusted across tasks.

    The reference "best_member" is, per task, the member of ``best_member_pool`` with the
    lowest mean window loss (chosen ex post, so the comparison is deliberately harsh).
    """
    pivot = losses.pivot_table(index=["dataset", "window"], columns="model", values="loss")
    rows = []
    for dataset, frame in pivot.groupby(level="dataset"):
        frame = frame.droplevel("dataset").sort_index()
        for reference in references:
            if reference == "best_member":
                if not best_member_pool:
                    raise ValueError("best_member needs best_member_pool.")
                ref_model = str(frame.loc[:, list(best_member_pool)].mean().idxmin())
            else:
                ref_model = reference
            for model in models:
                if model == ref_model:
                    continue
                d = (frame[model] - frame[ref_model]).to_numpy(dtype=float)
                result = diebold_mariano(d)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "reference": reference,
                        "reference_model": ref_model,
                        **result,
                        "relative_change": float(frame[model].mean() / frame[ref_model].mean() - 1),
                        "window_win_share": float(np.mean(d < 0)),
                    }
                )
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, index in out.groupby(["model", "reference"]).groups.items():
        out.loc[index, "p_holm"] = holm_adjust(out.loc[index, "p"].to_numpy())
    return out


def bootstrap_aggregate(
    table: CombinationTable,
    baseline: str,
    references: Sequence[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    resample: str = "series",
) -> pd.DataFrame:
    """Bootstrap of the aggregate relative MASE, stratified by task.

    ``resample`` picks the units drawn with replacement within each task: "series" (whole
    series with all their windows; conditional on the observed windows), "windows" (whole
    windows with all their series; the forecast origins are the uncertain part), or
    "both" (independent series and window draws). Reports each model's aggregate and its
    ratio to each reference.
    """
    if resample not in ("series", "windows", "both"):
        raise ValueError("resample must be 'series', 'windows' or 'both'.")
    rng = np.random.default_rng(seed)
    valid = table.valid_scale
    errors = np.abs(table.scaled_predictions - table.scaled_y[:, None])[valid]
    series = table.series_codes[valid]
    windows = table.window[valid]
    datasets = table.dataset_codes[valid]
    n_models = len(table.members)
    base = table.members.index(baseline)
    log_relative = np.zeros((n_boot + 1, n_models))
    for code in range(len(table.datasets)):
        in_task = datasets == code
        series_ids, series_index = np.unique(series[in_task], return_inverse=True)
        window_ids, window_index = np.unique(windows[in_task], return_inverse=True)
        n_series, n_windows = len(series_ids), len(window_ids)
        cell = series_index * n_windows + window_index
        size = n_series * n_windows
        counts = np.bincount(cell, minlength=size).astype(float).reshape(n_series, n_windows)
        sums = np.stack(
            [
                np.bincount(cell, weights=errors[in_task, m], minlength=size).reshape(
                    n_series, n_windows
                )
                for m in range(n_models)
            ],
            axis=-1,
        )
        series_weights = np.ones((n_boot + 1, n_series))
        window_weights = np.ones((n_boot + 1, n_windows))
        if resample in ("series", "both"):
            series_weights[1:] = rng.multinomial(
                n_series, np.full(n_series, 1.0 / n_series), size=n_boot
            )
        if resample in ("windows", "both"):
            window_weights[1:] = rng.multinomial(
                n_windows, np.full(n_windows, 1.0 / n_windows), size=n_boot
            )
        weighted_sums = (series_weights @ sums.reshape(n_series, -1)).reshape(
            n_boot + 1, n_windows, n_models
        )
        numerator = np.einsum("bwm,bw->bm", weighted_sums, window_weights)
        denominator = np.einsum("bw,bw->b", series_weights @ counts, window_weights)
        mase = numerator / denominator[:, None]
        log_relative += np.log(mase / mase[:, [base]])
    aggregate = np.exp(log_relative / len(table.datasets))
    alpha = (1 - level) / 2
    rows = []
    for column, model in enumerate(table.members):
        values = aggregate[1:, column]
        rows.append(
            {
                "model": model,
                "comparison": "aggregate",
                "estimate": aggregate[0, column],
                "ci_low": np.quantile(values, alpha),
                "ci_high": np.quantile(values, 1 - alpha),
            }
        )
        for reference in references:
            ref = table.members.index(reference)
            if ref == column:
                continue
            ratio = aggregate[:, column] / aggregate[:, ref]
            rows.append(
                {
                    "model": model,
                    "comparison": f"ratio_to_{reference}",
                    "estimate": ratio[0],
                    "ci_low": np.quantile(ratio[1:], alpha),
                    "ci_high": np.quantile(ratio[1:], 1 - alpha),
                    "share_below_1": float(np.mean(ratio[1:] < 1)),
                }
            )
    return pd.DataFrame(rows)


# --- diversity


def error_correlations(table: CombinationTable, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Pearson correlation of signed scaled errors between member pairs, pooled per task."""
    use = table.valid_scale if mask is None else table.valid_scale & mask
    signed = table.scaled_predictions - table.scaled_y[:, None]
    rows = []
    for code, dataset in enumerate(table.datasets):
        errors = signed[use & (table.dataset_codes == code)]
        corr = np.corrcoef(errors, rowvar=False)
        for i, a in enumerate(table.members):
            for j in range(i + 1, len(table.members)):
                rows.append(
                    {
                        "dataset": dataset,
                        "member_a": a,
                        "member_b": table.members[j],
                        "corr": corr[i, j],
                    }
                )
    return pd.DataFrame(rows)


def ambiguity_decomposition(
    table: CombinationTable, mask: np.ndarray | None = None
) -> pd.DataFrame:
    """Equal-weight ambiguity decomposition of squared scaled error, per task.

    ensemble MSE = mean member MSE - ambiguity, where ambiguity is the mean squared
    deviation of members from the ensemble (Krogh & Vedelsby, 1995); exact for MSE.
    """
    use = table.valid_scale if mask is None else table.valid_scale & mask
    rows = []
    for code, dataset in enumerate(table.datasets):
        rows_in = use & (table.dataset_codes == code)
        predictions = table.scaled_predictions[rows_in]
        y = table.scaled_y[rows_in]
        ensemble = predictions.mean(axis=1)
        member_mse = np.mean((predictions - y[:, None]) ** 2, axis=0)
        ambiguity = float(np.mean((predictions - ensemble[:, None]) ** 2))
        ensemble_mse = float(np.mean((ensemble - y) ** 2))
        rows.append(
            {
                "dataset": dataset,
                "mean_member_mse": float(member_mse.mean()),
                "best_member_mse": float(member_mse.min()),
                "ambiguity": ambiguity,
                "ensemble_mse": ensemble_mse,
                "diversity_ratio": ambiguity / float(member_mse.mean()),
                "ensemble_vs_best_mse": ensemble_mse / float(member_mse.min()),
            }
        )
    return pd.DataFrame(rows)


# --- granularity


def series_window_mase(table: CombinationTable) -> pd.DataFrame:
    """MASE of each member per (dataset, unique_id, window)."""
    valid = table.valid_scale
    errors = np.abs(table.scaled_predictions - table.scaled_y[:, None])[valid]
    codes, inverse = np.unique(table.series_window_codes[valid], return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    first = np.flatnonzero(valid)[np.unique(inverse, return_index=True)[1]]
    out = table.frame.loc[first, ["dataset", "unique_id", "window"]].reset_index(drop=True)
    for column, member in enumerate(table.members):
        out[member] = np.bincount(inverse, weights=errors[:, column]) / counts
    return out


def winner_persistence(table: CombinationTable) -> pd.DataFrame:
    """P(same member wins a series in windows k and k+1) vs chance from marginal win shares."""
    scores = series_window_mase(table)
    members = list(table.members)
    scores["winner"] = scores.loc[:, members].to_numpy().argmin(axis=1)
    rows = []
    for dataset, frame in scores.groupby("dataset"):
        wide = frame.pivot(index="unique_id", columns="window", values="winner").sort_index(axis=1)
        same = []
        for window in wide.columns[:-1]:
            pair = wide.loc[:, [window, window + 1]].dropna()
            if len(pair):
                same.append(float((pair[window] == pair[window + 1]).mean()))
        shares = np.bincount(frame["winner"], minlength=len(members)) / len(frame)
        rows.append(
            {
                "dataset": dataset,
                "persistence": float(np.mean(same)),
                "chance": float(np.sum(shares**2)),
                "n_window_pairs": len(same),
                **{f"win_share_{m}": shares[i] for i, m in enumerate(members)},
            }
        )
    return pd.DataFrame(rows)


# --- data requirements (learning curve)


def task_mase_long(forecasts: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """Panel MASE per (dataset, model) straight from long forecasts (valid scales only)."""
    scored = forecasts.loc[
        :, ["dataset", "model", "unique_id", "window", "y_true", "y_pred"]
    ].merge(
        scales.loc[:, ["dataset", "window", "unique_id", "mase_scale"]].astype({"unique_id": str}),
        on=["dataset", "window", "unique_id"],
        how="left",
        validate="many_to_one",
    )
    valid = np.isfinite(scored["mase_scale"]) & (scored["mase_scale"] > 0)
    scored = scored[valid]
    scored = scored.assign(error=(scored["y_pred"] - scored["y_true"]).abs() / scored["mase_scale"])
    return (
        scored.groupby(["dataset", "model"], as_index=False)["error"]
        .mean()
        .rename(columns={"error": "mase"})
    )


def learning_curve(
    table: CombinationTable,
    combiners: Callable[[], Sequence[Combiner]],
    reference_forecasts: pd.DataFrame,
    scales: pd.DataFrame,
    *,
    past_windows: Sequence[int | None],
    baseline: str,
) -> pd.DataFrame:
    """Aggregate relative MASE of each combiner when fitted on the last L past windows.

    ``None`` means all past windows. Every L is scored on the same windows: those with at
    least max(L) past windows. ``reference_forecasts`` (long, e.g. members and baseline)
    are scored on the same windows.
    """
    finite = [value for value in past_windows if value is not None]
    common = RollingProtocol(min_past_windows=max(finite)).evaluation_mask(table)
    pairs = table.frame.loc[common, ["dataset", "window"]].drop_duplicates()
    reference = reference_forecasts.merge(pairs, on=["dataset", "window"], how="inner")
    frames = [task_mase_long(reference, scales)]
    for value in past_windows:
        protocol = RollingProtocol(
            min_past_windows=1 if value is None else value, max_past_windows=value
        )
        for combiner in combiners():
            result = rolling_combine(table, combiner, protocol).forecasts
            result = result.merge(pairs, on=["dataset", "window"], how="inner")
            label = "all" if value is None else str(value)
            frames.append(task_mase_long(result.assign(model=f"{combiner.name}@L{label}"), scales))
    relative = relative_to_baseline(pd.concat(frames, ignore_index=True), baseline)
    aggregate = aggregate_relative(relative)
    split = aggregate["model"].str.split("@L", n=1, expand=True)
    aggregate["method"] = split[0]
    aggregate["past_windows"] = split[1] if split.shape[1] > 1 else None
    return aggregate


# --- failure modes


def _phase_of_day(ds: pd.Series) -> np.ndarray | None:
    """Fraction of the day for datetime stamps; None if ds is not sub-daily datetime."""
    parsed = pd.to_datetime(ds, errors="coerce", format="ISO8601")
    if parsed.isna().any():
        return None
    minutes = (parsed.dt.hour * 60 + parsed.dt.minute).to_numpy(dtype=float)
    if np.all(minutes == 0):
        return None
    return minutes / (24 * 60)


def horizon_phase_ratios(
    table: CombinationTable,
    baseline: str,
    *,
    n_horizon_bins: int = 3,
    n_phase_bins: int = 2,
) -> pd.DataFrame:
    """Scaled MAE relative to the baseline by horizon bin, overall and within time-of-day bins.

    Separates degradation along the horizon from time-of-day effects: with stride equal
    to the horizon, horizon position and time of day are confounded on sub-daily data.
    Datasets without sub-daily timestamps are skipped.
    """
    errors = np.abs(table.scaled_predictions - table.scaled_y[:, None])
    base = table.members.index(baseline)
    hbin = horizon_bins(table.horizon, table.forecast_horizon, n_horizon_bins)
    rows = []
    for code, dataset in enumerate(table.datasets):
        in_task = (table.dataset_codes == code) & table.valid_scale
        phase = _phase_of_day(table.frame.loc[in_task, "ds"])
        if phase is None:
            continue
        pbin = np.minimum((phase * n_phase_bins).astype(int), n_phase_bins - 1)
        task_errors = errors[in_task]
        task_hbin = hbin[in_task]
        for phase_bin in [-1, *range(n_phase_bins)]:
            in_phase = np.ones(len(pbin), dtype=bool) if phase_bin < 0 else pbin == phase_bin
            for horizon_bin in range(n_horizon_bins):
                cell = in_phase & (task_hbin == horizon_bin)
                if not cell.any():
                    continue
                means = task_errors[cell].mean(axis=0)
                for column, model in enumerate(table.members):
                    rows.append(
                        {
                            "dataset": dataset,
                            "phase_bin": "all" if phase_bin < 0 else phase_bin,
                            "horizon_bin": horizon_bin,
                            "model": model,
                            "scaled_mae": means[column],
                            "ratio_to_baseline": means[column] / means[base],
                            "n_rows": int(cell.sum()),
                        }
                    )
    return pd.DataFrame(rows)


def amplitude_ratios(table: CombinationTable) -> pd.DataFrame:
    """Median over series-windows of std(forecast) / std(actual) within the window."""
    codes, inverse = np.unique(table.series_window_codes, return_inverse=True)
    counts = np.bincount(inverse).astype(float)

    def window_std(values: np.ndarray) -> np.ndarray:
        mean = np.bincount(inverse, weights=values) / counts
        square = np.bincount(inverse, weights=values**2) / counts
        return np.sqrt(np.maximum(square - mean**2, 0.0))

    actual = window_std(table.y)
    first = np.unique(inverse, return_index=True)[1]
    dataset = np.asarray(table.datasets)[table.dataset_codes[first]]
    rows = []
    for column, model in enumerate(table.members):
        ratio = np.where(
            actual > 0,
            window_std(table.predictions[:, column]) / np.where(actual > 0, actual, 1),
            np.nan,
        )
        frame = pd.DataFrame({"dataset": dataset, "ratio": ratio})
        medians = frame.groupby("dataset")["ratio"].median()
        for name, value in medians.items():
            rows.append({"dataset": name, "model": model, "median_amplitude_ratio": float(value)})
    return pd.DataFrame(rows)
