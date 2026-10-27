import numpy as np
import pandas as pd
import pytest
from scipy import stats

from timestransfer.combination import (
    EqualMean,
    LinearCombiner,
    build_combination_table,
    nnls_weights,
)
from timestransfer.combination_analysis import (
    _block_window_counts,
    _simplex_mse_weights,
    aggregate_relative,
    ambiguity_decomposition,
    amplitude_ratios,
    bootstrap_aggregate,
    bootstrap_contrasts,
    bootstrap_replicates,
    combination_bound,
    diebold_mariano,
    dm_tests,
    error_correlations,
    holm_adjust,
    horizon_phase_ratios,
    learning_curve,
    pairwise_combination_theory,
    regret_summary,
    regret_table,
    relative_to_baseline,
    shared_error_decomposition,
    task_mase,
    task_mase_long,
    window_losses,
    winner_persistence,
)


def _long(predict, *, datasets=("d1", "d2"), n_series=4, n_windows=6, horizon=4, ds=None, seed=0):
    """Long forecasts + scales; predict(y, rng, dataset, series, window) -> {model: values}."""
    rng = np.random.default_rng(seed)
    rows, scale_rows = [], []
    for dataset in datasets:
        for series in range(n_series):
            for window in range(n_windows):
                y = 10 + rng.normal(size=horizon)
                scale_rows.append(
                    {
                        "dataset": dataset,
                        "window": window,
                        "unique_id": f"s{series}",
                        "mase_scale": 1.0,
                    }
                )
                for model, values in predict(y, rng, dataset, series, window).items():
                    for h in range(horizon):
                        stamp = ds(window, h) if ds else f"{window}-{h}"
                        rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "unique_id": f"s{series}",
                                "horizon": h + 1,
                                "ds": stamp,
                                "y_true": float(y[h]),
                                "y_pred": float(values[h]),
                                "window": window,
                                "forecast_horizon": horizon,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(scale_rows)


def _offsets(y, rng, *_):
    return {"base": y + 2.0, "good": y + 0.5, "bad": y - 1.0}


def test_task_scores_relative_aggregate_and_regret():
    forecasts, scales = _long(_offsets)
    table = build_combination_table(forecasts, scales, ["base", "good", "bad"])

    scores = task_mase(table)
    relative = relative_to_baseline(scores, "base")
    aggregate = aggregate_relative(relative).set_index("model")

    assert np.allclose(scores.set_index(["dataset", "model"]).loc[("d1", "good"), "mase"], 0.5)
    assert aggregate.loc["good", "geomean_relative_mase"] == pytest.approx(0.25)
    assert aggregate.loc["bad", "geomean_relative_mase"] == pytest.approx(0.5)
    regret = regret_table(relative, ["good", "bad"])
    summary = regret_summary(regret).set_index("model")
    assert summary.loc["bad", "max_regret"] == pytest.approx(2.0)
    assert summary.loc["good", "max_regret"] == pytest.approx(1.0)
    assert summary.loc["bad", "tasks_above_1.10"] == 2
    assert np.allclose(
        task_mase_long(forecasts, scales).sort_values(["dataset", "model"])["mase"],
        scores.sort_values(["dataset", "model"])["mase"],
    )


def test_diebold_mariano_without_lags_is_the_one_sample_t_test():
    rng = np.random.default_rng(1)
    d = rng.normal(loc=0.3, size=20)

    result = diebold_mariano(d, max_lag=0)

    expected = stats.ttest_1samp(d, 0.0)
    assert result["stat"] == pytest.approx(expected.statistic)
    assert result["p"] == pytest.approx(expected.pvalue)
    assert diebold_mariano(rng.normal(loc=-2.0, size=17))["p"] < 0.01
    assert np.isnan(diebold_mariano(np.array([0.1, 0.2]))["p"])


def test_holm_adjust_known_values():
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    adjusted = holm_adjust([0.2, np.nan, 0.01])
    assert np.isnan(adjusted[1]) and np.allclose(adjusted[[0, 2]], [0.2, 0.02])


def test_dm_tests_resolves_best_member_and_signs():
    forecasts, scales = _long(_offsets, n_windows=8)
    table = build_combination_table(forecasts, scales, ["base", "good", "bad"])
    losses = window_losses(table)

    tests = dm_tests(losses, ["bad", "good"], ["base", "best_member"], ["good", "bad"])

    against_best = tests[tests["reference"] == "best_member"]
    assert set(against_best["reference_model"]) == {"good"}
    assert set(against_best["model"]) == {"bad"}  # a model is never tested against itself
    bad_vs_base = tests[(tests["model"] == "bad") & (tests["reference"] == "base")]
    assert np.allclose(bad_vs_base["relative_change"], -0.5)
    assert (bad_vs_base["window_win_share"] == 1.0).all()
    assert "p_holm" in tests.columns


def test_bootstrap_estimate_matches_point_aggregate_and_is_seeded():
    def predict(y, rng, *_):
        return {"base": y + rng.normal(size=y.shape), "m": y + 0.5 * rng.normal(size=y.shape)}

    forecasts, scales = _long(predict, n_series=10)
    table = build_combination_table(forecasts, scales, ["base", "m"])

    first = bootstrap_aggregate(table, "base", ["base"], n_boot=200, seed=3)
    second = bootstrap_aggregate(table, "base", ["base"], n_boot=200, seed=3)

    point = aggregate_relative(relative_to_baseline(task_mase(table), "base")).set_index("model")
    row = first[(first["model"] == "m") & (first["comparison"] == "aggregate")].iloc[0]
    assert row["estimate"] == pytest.approx(point.loc["m", "geomean_relative_mase"])
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    pd.testing.assert_frame_equal(first, second)
    for resample in ("windows", "both", "blocks", "series_blocks"):
        other = bootstrap_aggregate(table, "base", ["base"], n_boot=200, seed=3, resample=resample)
        other_row = other[(other["model"] == "m") & (other["comparison"] == "aggregate")].iloc[0]
        assert other_row["estimate"] == pytest.approx(row["estimate"])
        assert other_row["ci_low"] < other_row["estimate"] < other_row["ci_high"]
    with pytest.raises(ValueError):
        bootstrap_aggregate(table, "base", ["base"], resample="tasks")


def test_block_window_counts_draw_consecutive_windows():
    rng = np.random.default_rng(0)
    counts = _block_window_counts(rng, n_windows=7, n_boot=500, block_length=3)
    assert counts.shape == (500, 7)
    assert (counts.sum(axis=1) == 7).all()
    # every window is reachable, including across the circular wrap
    assert (counts.sum(axis=0) > 0).all()
    single = _block_window_counts(np.random.default_rng(1), n_windows=5, n_boot=50, block_length=1)
    assert (single.sum(axis=1) == 5).all()
    # block length 5 on 5 windows: every replicate is a rotation, so each window once
    full = _block_window_counts(np.random.default_rng(2), n_windows=5, n_boot=50, block_length=5)
    assert (full == 1).all()


def test_bootstrap_contrasts_ratio_of_ratios():
    def predict(y, rng, *_):
        noise = rng.normal(size=y.shape)
        return {"base": y + 2 * noise, "a": y + noise, "b": y + noise, "c": y + 0.5 * noise}

    forecasts, scales = _long(predict, n_series=8)
    table = build_combination_table(forecasts, scales, ["base", "a", "b", "c"])
    replicates = bootstrap_replicates(table, "base", n_boot=300, seed=0, resample="both")
    assert replicates.shape == (301, 4)
    out = bootstrap_contrasts(
        replicates,
        table.members,
        {"c_vs_a": {"c": 1, "a": -1}, "identical": {"a": 1, "b": -1}},
    ).set_index("contrast")
    point = aggregate_relative(relative_to_baseline(task_mase(table), "base")).set_index("model")
    expected = point.loc["c", "geomean_relative_mase"] / point.loc["a", "geomean_relative_mase"]
    assert out.loc["c_vs_a", "estimate"] == pytest.approx(expected)
    assert out.loc["c_vs_a", "ci_high"] < 1
    assert out.loc["c_vs_a", "share_below_1"] == 1.0
    # a and b are the same forecasts: their ratio is exactly 1 in every replicate
    assert out.loc["identical", ["estimate", "ci_low", "ci_high"]].tolist() == pytest.approx([1, 1, 1])


def test_diversity_measures():
    def predict(y, rng, *_):
        noise = rng.normal(size=y.shape)
        return {"a": y + noise, "b": y + 2 * noise, "c": y - noise}

    forecasts, scales = _long(predict)
    table = build_combination_table(forecasts, scales, ["a", "b", "c"])

    corr = error_correlations(table).set_index(["dataset", "member_a", "member_b"])["corr"]
    assert corr.loc[("d1", "a", "b")] == pytest.approx(1.0)
    assert corr.loc[("d1", "a", "c")] == pytest.approx(-1.0)

    ambiguity = ambiguity_decomposition(table)
    assert np.allclose(
        ambiguity["ensemble_mse"], ambiguity["mean_member_mse"] - ambiguity["ambiguity"]
    )


def test_shared_error_decomposition_is_exact_and_matches_ambiguity():
    def predict(y, rng, *_):
        common = rng.normal(size=y.shape)
        return {m: y + common + 0.3 * rng.normal(size=y.shape) for m in ("a", "b", "c")}

    forecasts, scales = _long(predict)
    table = build_combination_table(forecasts, scales, ["a", "b", "c"])
    shared = shared_error_decomposition(table).set_index("dataset")
    ambiguity = ambiguity_decomposition(table).set_index("dataset")

    assert np.allclose(shared["shared_mse"] + shared["specific_mse"], shared["mean_member_mse"])
    assert np.allclose(shared["shared_share"], 1 - ambiguity["diversity_ratio"])
    assert (shared["shared_share"] > 0.7).all()
    assert ((shared["first_pc_share"] > 1 / 3) & (shared["first_pc_share"] <= 1)).all()


# Orthogonal errors within every series-window: E[e_a^2] = 1, E[e_b^2] = 4, E[e_a e_b] = 0.
_PATTERN = {"a": np.array([1.0, -1.0, 1.0, -1.0]), "b": np.array([2.0, 2.0, -2.0, -2.0])}


def _orthogonal(y, *_):
    return {m: y + e for m, e in _PATTERN.items()}


def test_combination_bound_on_orthogonal_errors():
    forecasts, scales = _long(_orthogonal)
    table = build_combination_table(forecasts, scales, ["a", "b"])
    bound = combination_bound(table).set_index("dataset").loc["d1"]

    assert bound["best_member"] == "a"
    assert bound["best_member_mse"] == pytest.approx(1.0)
    # Optimum (ab - c^2) / (a + b - 2c) = 4 / 5 with weights (0.8, 0.2); equal = 5/4.
    assert bound["unconstrained_vs_best"] == pytest.approx(0.8)
    assert bound["simplex_vs_best"] == pytest.approx(0.8)
    assert bound["equal_vs_best"] == pytest.approx(1.25)
    assert bound["simplex_w_a"] == pytest.approx(0.8)

    theory = pairwise_combination_theory(table).set_index("dataset").loc["d1"]
    assert theory["better_member"] == "a"
    assert theory["rho"] == pytest.approx(0.0, abs=1e-12)
    assert theory["accuracy_ratio_k"] == pytest.approx(0.5)
    assert theory["optimal_vs_better"] == pytest.approx(0.8)


def test_simplex_weights_drop_a_member_that_needs_a_negative_weight():
    moments = np.array([[1.0, 1.9], [1.9, 4.0]])
    free = np.linalg.solve(moments, np.ones(2))
    free /= free.sum()
    assert free[1] < 0
    assert free @ moments @ free == pytest.approx(0.39 / 1.2)

    assert np.allclose(_simplex_mse_weights(moments), [1.0, 0.0])
    three = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(_simplex_mse_weights(three), [1 / 3] * 3)


def test_winner_persistence_and_chance():
    def predict(y, rng, dataset, series, window):
        # Member a always wins on d1; on d2 the winner alternates by window.
        if dataset == "d1" or window % 2 == 0:
            return {"a": y + 0.1, "b": y + 1.0}
        return {"a": y + 1.0, "b": y + 0.1}

    forecasts, scales = _long(predict)
    table = build_combination_table(forecasts, scales, ["a", "b"])

    persistence = winner_persistence(table).set_index("dataset")

    assert persistence.loc["d1", "persistence"] == 1.0
    assert persistence.loc["d1", "chance"] == 1.0
    assert persistence.loc["d2", "persistence"] == 0.0
    assert persistence.loc["d2", "chance"] == pytest.approx(0.5)
    assert persistence.loc["d2", "n_window_pairs"] == 5


def test_horizon_phase_ratios_and_amplitude():
    def stamp(window, h):
        # 6-hour steps: horizon 1..4 covers one day; windows start at midnight.
        return str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=window, hours=6 * h))

    def predict(y, rng, *_):
        mean = y.mean()
        return {"base": y + 1.0, "damped": mean + 0.5 * (y - mean)}

    forecasts, scales = _long(predict, ds=stamp, horizon=4)
    table = build_combination_table(forecasts, scales, ["base", "damped"])

    ratios = horizon_phase_ratios(table, "base", n_horizon_bins=2, n_phase_bins=2)
    assert set(ratios["phase_bin"].astype(str)) == {"all", "0", "1"}
    assert (ratios.loc[ratios["model"] == "base", "ratio_to_baseline"] == 1.0).all()

    amplitude = amplitude_ratios(table).set_index(["dataset", "model"])["median_amplitude_ratio"]
    assert amplitude.loc[("d1", "damped")] == pytest.approx(0.5)
    assert amplitude.loc[("d1", "base")] == pytest.approx(1.0)

    integer_ds, scales2 = _long(predict)
    skipped = horizon_phase_ratios(
        build_combination_table(integer_ds, scales2, ["base", "damped"]), "base"
    )
    assert skipped.empty


def test_learning_curve_scores_every_window_cap_on_common_windows():
    def predict(y, rng, *_):
        noise = 0.05 * rng.normal(size=(2, len(y)))
        return {"base": y + 2.0, "good": y + 0.5 + noise[0], "bad": y - 1.0 + noise[1]}

    forecasts, scales = _long(predict, n_windows=8)
    table = build_combination_table(forecasts, scales, ["good", "bad"])
    reference = forecasts[forecasts["model"].isin(["base", "good", "bad"])]

    curve = learning_curve(
        table,
        lambda: [EqualMean(), LinearCombiner("nnls_task", nnls_weights, level="task")],
        reference,
        scales,
        past_windows=[1, 3, None],
        baseline="base",
    )

    combos = curve.dropna(subset=["past_windows"])
    assert set(combos["past_windows"]) == {"1", "3", "all"}
    assert set(combos["method"]) == {"equal_mean", "nnls_task"}
    equal = combos[combos["method"] == "equal_mean"]["geomean_relative_mase"]
    assert np.allclose(equal, equal.iloc[0])  # equal weights do not depend on L
    nnls = combos[combos["method"] == "nnls_task"]["geomean_relative_mase"]
    good = curve.set_index("model").loc["good", "geomean_relative_mase"]
    # Offsets +0.5 and -1.0 cancel at weights (2/3, 1/3), so learned weights beat "good".
    assert (nnls < 0.5 * good).all()
    assert {"base", "good", "bad"} <= set(curve["model"])
