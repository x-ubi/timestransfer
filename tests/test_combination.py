import numpy as np
import pandas as pd
import pytest

from timestransfer.combination import (
    EqualMean,
    FeatureWeightedCombiner,
    InverseErrorWeights,
    LinearCombiner,
    Median,
    RollingProtocol,
    build_combination_table,
    default_combiners,
    horizon_bins,
    nnls_weights,
    rolling_combine,
    rolling_series_features,
    select_best,
)

MEMBERS = ["a", "b", "c"]


def _toy_forecasts(
    *,
    datasets=("d1", "d2"),
    n_series=3,
    n_windows=6,
    horizon=4,
    predict=None,
    seed=0,
):
    """Long forecasts and MASE scales; ``predict(y, rng, dataset, series, window)`` -> dict."""
    rng = np.random.default_rng(seed)
    predict = predict or (lambda y, rng, *_: {m: y + rng.normal(size=y.shape) for m in MEMBERS})
    forecast_rows, scale_rows = [], []
    for dataset in datasets:
        for series in range(n_series):
            unique_id = f"s{series}"
            for window in range(n_windows):
                y = 10.0 + rng.normal(size=horizon)
                member_predictions = predict(y, rng, dataset, series, window)
                scale_rows.append(
                    {
                        "dataset": dataset,
                        "window": window,
                        "unique_id": unique_id,
                        "mase_scale": 1.0 + series,
                    }
                )
                for model, y_pred in member_predictions.items():
                    for h in range(horizon):
                        forecast_rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "unique_id": unique_id,
                                "horizon": h + 1,
                                "ds": f"{window}-{h}",
                                "y_true": float(y[h]),
                                "y_pred": float(y_pred[h]),
                                "window": window,
                                "forecast_horizon": horizon,
                            }
                        )
    return pd.DataFrame(forecast_rows), pd.DataFrame(scale_rows)


def _toy_table(**kwargs):
    forecasts, scales = _toy_forecasts(**kwargs)
    return build_combination_table(forecasts, scales, MEMBERS)


def test_build_table_pivots_members_and_scales():
    forecasts, scales = _toy_forecasts()

    table = build_combination_table(forecasts, scales, MEMBERS)

    assert len(table) == 2 * 3 * 6 * 4
    assert table.members == ("a", "b", "c")
    assert table.n_windows == {"d1": 6, "d2": 6}
    row = table.frame.iloc[0]
    source = forecasts[
        (forecasts["dataset"] == row["dataset"])
        & (forecasts["unique_id"] == row["unique_id"])
        & (forecasts["window"] == row["window"])
        & (forecasts["horizon"] == row["horizon"])
    ].set_index("model")
    assert [row[m] for m in MEMBERS] == source.loc[MEMBERS, "y_pred"].tolist()
    assert row["y_true"] == source["y_true"].iloc[0]
    assert row["mase_scale"] == 1.0


def test_build_table_rejects_incomplete_or_inconsistent_members():
    forecasts, scales = _toy_forecasts()

    gap = forecasts.drop(
        forecasts[(forecasts["model"] == "c") & (forecasts["dataset"] == "d2")].index[:1]
    )
    with pytest.raises(ValueError, match="d2"):
        build_combination_table(gap, scales, MEMBERS)

    shifted = forecasts.copy()
    shifted.loc[shifted["model"] == "b", "y_true"] += 1.0
    with pytest.raises(ValueError, match="disagree"):
        build_combination_table(shifted, scales, MEMBERS)

    with pytest.raises(ValueError, match="No forecasts"):
        build_combination_table(forecasts, scales, [*MEMBERS, "missing"])


def test_rolling_protocol_windows():
    protocol = RollingProtocol(min_past_windows=3)
    assert protocol.training_windows(0, 6) == [1, 2, 3, 4, 5]
    assert protocol.evaluation_windows(6) == [0, 1, 2]

    capped = RollingProtocol(min_past_windows=2, max_past_windows=3)
    assert capped.training_windows(0, 6) == [1, 2, 3]
    assert capped.evaluation_windows(6) == [0, 1, 2, 3]

    with pytest.raises(ValueError):
        RollingProtocol(min_past_windows=0)
    with pytest.raises(ValueError):
        RollingProtocol(min_past_windows=3, max_past_windows=2)


def test_horizon_bins_are_equal_consecutive_parts():
    bins = horizon_bins(np.arange(1, 49), np.full(48, 48), 3)
    assert np.bincount(bins).tolist() == [16, 16, 16]
    assert bins[15] == 0 and bins[16] == 1
    assert np.bincount(horizon_bins(np.arange(1, 97), np.full(96, 96), 6)).tolist() == [16] * 6
    with pytest.raises(ValueError):
        horizon_bins(np.arange(1, 5), np.full(4, 4), 5)


def test_equal_mean_and_median_reproduce_member_statistics():
    table = _toy_table()
    protocol = RollingProtocol(min_past_windows=3)

    mean = rolling_combine(table, EqualMean(), protocol).forecasts
    median = rolling_combine(table, Median(), protocol).forecasts

    evaluated = protocol.evaluation_mask(table)
    members = table.predictions[evaluated]
    assert sorted(mean["window"].unique()) == [0, 1, 2]
    assert np.array_equal(mean["y_pred"].to_numpy(), members.mean(axis=1))
    assert np.array_equal(median["y_pred"].to_numpy(), np.median(members, axis=1))
    assert set(mean["model"]) == {"equal_mean"}
    assert list(mean.columns) == [
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


def test_nnls_recovers_mixing_weights():
    def predict(y, rng, *_):
        a = y + rng.normal(size=y.shape)
        c = rng.normal(size=y.shape)
        # y = 0.7 a + 0.3 b exactly; c is noise.
        return {"a": a, "b": (y - 0.7 * a) / 0.3, "c": c}

    table = _toy_table(predict=predict)
    result = rolling_combine(
        table, LinearCombiner("nnls_task", nnls_weights, level="task"), RollingProtocol(3)
    )

    weights = result.weights
    assert len(weights) == 2 * 3  # 2 tasks x 3 evaluated windows
    assert np.allclose(weights[["w_a", "w_b", "w_c"]].to_numpy(), [0.7, 0.3, 0.0], atol=1e-8)
    assert np.allclose(result.forecasts["y_pred"], result.forecasts["y_true"])


def test_series_selection_expanding_and_last_window():
    # Member a is best in window 1 only; b is best in windows 2..5.
    def predict(y, rng, dataset, series, window):
        good, bad = y + 0.1, y + 3.0
        if window == 1:
            return {"a": good, "b": bad, "c": bad}
        return {"a": bad, "b": good, "c": bad}

    table = _toy_table(predict=predict)
    protocol = RollingProtocol(3)
    expanding = rolling_combine(
        table, LinearCombiner("sel", select_best, level="series"), protocol
    ).weights
    last = rolling_combine(
        table,
        LinearCombiner("sel_last", select_best, level="series", recent_windows=1),
        protocol,
    ).weights

    at_window_0 = expanding[expanding["window"] == 0]
    assert (at_window_0[["w_a", "w_b", "w_c"]].to_numpy() == [0, 1, 0]).all()
    last_at_0 = last[last["window"] == 0]
    assert (last_at_0[["w_a", "w_b", "w_c"]].to_numpy() == [1, 0, 0]).all()
    assert (last_at_0["n_train_rows"] == 4).all()  # one window x 4 steps


def test_inverse_error_weights_hand_computed():
    scaled_y = np.zeros(2)
    scaled_predictions = np.array([[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]])  # MAE 1, 2, 4

    weights = InverseErrorWeights(power=2)(scaled_predictions, scaled_y)
    expected = np.array([1.0, 1 / 4, 1 / 16])
    assert np.allclose(weights, expected / expected.sum())

    top2 = InverseErrorWeights(power=1, top_k=2)(scaled_predictions, scaled_y)
    assert np.allclose(top2, [2 / 3, 1 / 3, 0.0])

    assert np.array_equal(select_best(scaled_predictions, scaled_y), [1.0, 0.0, 0.0])


def test_shrinkage_and_fallback_to_equal_weights():
    def predict(y, rng, *_):
        return {"a": y.copy(), "b": y + 5.0, "c": y - 5.0}

    forecasts, scales = _toy_forecasts(predict=predict)
    # Series s0 of d1 has no valid scale in any past window -> no training rows.
    invalid = (scales["dataset"] == "d1") & (scales["unique_id"] == "s0") & (scales["window"] > 0)
    scales.loc[invalid, "mase_scale"] = np.nan
    table = build_combination_table(forecasts, scales, MEMBERS)

    shrunk = rolling_combine(
        table,
        LinearCombiner("sel", select_best, level="series", shrinkage=0.5),
        RollingProtocol(3),
    ).weights
    window_0 = shrunk[shrunk["window"] == 0].set_index(["dataset", "unique_id"])
    assert np.allclose(window_0.loc[("d1", "s0"), ["w_a", "w_b", "w_c"]], 1 / 3)
    assert bool(window_0.loc[("d1", "s0"), "fallback"])
    assert np.allclose(window_0.loc[("d2", "s1"), ["w_a", "w_b", "w_c"]], [2 / 3, 1 / 6, 1 / 6])


def _toy_features(table, seed=1):
    rng = np.random.default_rng(seed)
    index = table.frame.loc[:, ["dataset", "unique_id", "window"]].drop_duplicates()
    features = index.copy()
    for column in ("length", "cv", "acf1", "acf_seasonal"):
        features[column] = rng.uniform(0.1, 1.0, size=len(index))
    for column in ("trend_strength", "seasonal_strength", "spectral_entropy"):
        features[column] = rng.uniform(0.0, 1.0, size=len(index))
    return features.reset_index(drop=True)


def test_combiners_do_not_use_information_from_the_target_window_or_later():
    window = 1
    # 12 steps so the 6-bin horizon combiner is valid.
    forecasts, scales = _toy_forecasts(seed=3, horizon=12)
    table = build_combination_table(forecasts, scales, MEMBERS)
    features = _toy_features(table)

    # Everything observed at or after window 1's forecast origin is perturbed: targets and
    # scales of windows <= 1, member forecasts and features of windows < 1.
    rng = np.random.default_rng(99)
    late = forecasts["window"] <= window
    perturbed = forecasts.copy()
    # Same shift for every member's copy of a target, so the table stays consistent.
    perturbed.loc[late, "y_true"] += 7.0 + 0.5 * perturbed.loc[late, "horizon"]
    later = perturbed["window"] < window
    perturbed.loc[later, "y_pred"] += rng.normal(scale=5.0, size=later.sum())
    perturbed_scales = scales.copy()
    perturbed_scales.loc[perturbed_scales["window"] <= window, "mase_scale"] *= 3.0
    perturbed_table = build_combination_table(perturbed, perturbed_scales, MEMBERS)
    perturbed_features = features.copy()
    later_features = perturbed_features["window"] < window
    perturbed_features.loc[later_features, ["cv", "acf1"]] *= -4.0

    protocol = RollingProtocol(min_past_windows=3)
    combiners = default_combiners(MEMBERS, features)
    perturbed_combiners = default_combiners(MEMBERS, perturbed_features)
    assert any(isinstance(c, FeatureWeightedCombiner) for c in combiners)
    for combiner, perturbed_combiner in zip(combiners, perturbed_combiners, strict=True):
        original = rolling_combine(table, combiner, protocol)
        changed = rolling_combine(perturbed_table, perturbed_combiner, protocol)
        at_window = original.forecasts["window"] == window
        assert np.array_equal(
            original.forecasts.loc[at_window, "y_pred"].to_numpy(),
            changed.forecasts.loc[at_window, "y_pred"].to_numpy(),
        ), combiner.name
        if not original.weights.empty:
            columns = [f"w_{m}" for m in MEMBERS]
            assert np.array_equal(
                original.weights.loc[original.weights["window"] == window, columns].to_numpy(),
                changed.weights.loc[changed.weights["window"] == window, columns].to_numpy(),
            ), combiner.name


def test_learned_weights_do_depend_on_past_windows():
    # Sensitivity check for the leakage test: changing window 2 must move window-1 weights.
    forecasts, scales = _toy_forecasts(seed=3)
    table = build_combination_table(forecasts, scales, MEMBERS)
    shifted = forecasts.copy()
    past = (shifted["window"] == 2) & (shifted["model"] == "a")
    shifted.loc[past, "y_pred"] += 4.0
    shifted_table = build_combination_table(shifted, scales, MEMBERS)
    combiner = LinearCombiner("nnls_task", nnls_weights, level="task")

    original = rolling_combine(table, combiner, RollingProtocol(3)).weights
    changed = rolling_combine(shifted_table, combiner, RollingProtocol(3)).weights

    window_1 = original["window"] == 1
    assert not np.allclose(
        original.loc[window_1, ["w_a", "w_b", "w_c"]], changed.loc[window_1, ["w_a", "w_b", "w_c"]]
    )


def test_feature_weighted_combiner_transfers_across_datasets():
    # A high cv feature means member a is accurate, a low one means member b is.
    datasets = ("d1", "d2", "d3")
    cv = {(d, s): (2.0 if s % 2 == 0 else 0.1) for d in datasets for s in range(6)}

    def predict(y, rng, dataset, series, window):
        good = y + rng.normal(scale=0.1, size=y.shape)
        bad = y + rng.normal(scale=2.0, size=y.shape)
        if cv[(dataset, series)] > 1:
            return {"a": good, "b": bad, "c": bad}
        return {"a": bad, "b": good, "c": bad}

    forecasts, scales = _toy_forecasts(datasets=datasets, n_series=6, predict=predict)
    table = build_combination_table(forecasts, scales, MEMBERS)
    features = table.frame.loc[:, ["dataset", "unique_id", "window"]].drop_duplicates()
    features["cv"] = [
        cv[(d, int(u[1:]))] for d, u in zip(features["dataset"], features["unique_id"])
    ]
    for column in ("length", "acf1", "acf_seasonal", "trend_strength"):
        features[column] = 0.5
    features["seasonal_strength"] = np.nan  # must be imputed
    features["spectral_entropy"] = 0.5

    combiner = FeatureWeightedCombiner("fforma", features, l2=1e-4)
    result = rolling_combine(table, combiner, RollingProtocol(3))

    weights = result.weights
    high = weights["unique_id"].isin(["s0", "s2", "s4"])
    assert (weights.loc[high, "w_a"] > 0.6).all()
    assert (weights.loc[~high, "w_b"] > 0.6).all()
    # Leave-one-dataset-out: window k learns from 2 other tasks x 6 series x (5 - k) windows.
    assert (weights["n_train_rows"] == 2 * 6 * (5 - weights["window"])).all()
    assert set(weights["dataset"]) == set(datasets)


def test_rolling_series_features_match_benchmark_splits(monkeypatch, tmp_path):
    from timestransfer import data
    from timestransfer.data import rolling_origin_splits
    from timestransfer.series_features import compute_series_features

    rng = np.random.default_rng(0)
    toy = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": np.arange(n),
                    "y": np.sin(np.arange(n) * 2 * np.pi / 6) + rng.normal(size=n) * 0.1 + 5,
                }
            )
            for unique_id, n in (("u1", 120), ("u2", 100))
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(data, "load_dataset", lambda data_dir, dataset_config: toy)
    config = tmp_path / "config.yaml"
    config.write_text(
        "project: {n_windows: 3}\n"
        "datasets:\n"
        "  - {name: toy, loader: fake, horizon: 8, seasonality: 6, series_limit: null,\n"
        "     features: {lags: [1]}}\n"
    )

    features = rolling_series_features(config, max_points=40)

    splits = rolling_origin_splits(toy, horizon=8, n_windows=3)
    for window, split in enumerate(splits):
        expected = compute_series_features(split.train, seasonality=6, max_points=40)
        got = features[features["window"] == window].drop(columns=["dataset", "window"])
        got = got.drop(columns="max_points").reset_index(drop=True)
        pd.testing.assert_frame_equal(got, expected)
