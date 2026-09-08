import pandas as pd

from timestransfer.analysis import (
    join_features_and_metrics,
    spearman_correlations,
    write_analysis_outputs,
)


def _joined_toy() -> pd.DataFrame:
    # Four series, one model: smape increases monotonically with cv, decreases with acf1.
    return pd.DataFrame(
        {
            "dataset": ["toy"] * 4,
            "model": ["m"] * 4,
            "unique_id": ["s1", "s2", "s3", "s4"],
            "smape": [10.0, 20.0, 30.0, 40.0],
            "cv": [0.1, 0.2, 0.3, 0.4],
            "acf1": [0.9, 0.7, 0.5, 0.3],
        }
    )


def test_join_features_and_metrics_joins_on_dataset_and_series():
    features = pd.DataFrame(
        {"dataset": ["toy", "toy"], "unique_id": ["s1", "s2"], "cv": [0.1, 0.2]}
    )
    series_metrics = pd.DataFrame(
        {
            "dataset": ["toy", "toy", "toy"],
            "model": ["m", "m", "m"],
            "unique_id": ["s1", "s2", "s3"],
            "smape": [10.0, 20.0, 30.0],
        }
    )

    joined = join_features_and_metrics(features, series_metrics)

    assert len(joined) == 2
    assert set(joined["unique_id"]) == {"s1", "s2"}
    assert "cv" in joined.columns


def test_spearman_correlations_monotone_toy():
    correlations = spearman_correlations(
        _joined_toy(), feature_columns=["cv", "acf1"], metric="smape"
    )

    pooled = correlations[correlations["dataset"] == "all"].set_index("feature")
    assert pooled.loc["cv", "spearman_rho"] == 1.0
    assert pooled.loc["acf1", "spearman_rho"] == -1.0
    assert set(correlations["dataset"]) == {"all", "toy"}


def test_write_analysis_outputs_creates_tables_and_figures(tmp_path):
    paths = write_analysis_outputs(
        _joined_toy(), output_dir=tmp_path, feature_columns=["cv", "acf1"], metric="smape"
    )

    assert paths["joined"].exists()
    assert paths["correlations"].exists()
    assert paths["heatmap"].exists()
    assert paths["scatter_cv"].exists()
    assert paths["scatter_acf1"].exists()
