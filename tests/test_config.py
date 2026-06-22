from timestransfer.config import dataset_configs


def test_dataset_configs_supports_old_single_dataset_shape():
    config = {
        "dataset": {"name": "m4_hourly", "horizon": 48},
        "features": {"lags": [1, 2, 24], "train_row_cap": 100},
    }

    datasets = dataset_configs(config)

    assert len(datasets) == 1
    assert datasets[0]["name"] == "m4_hourly"
    assert datasets[0]["features"]["lags"] == [1, 2, 24]
    assert datasets[0]["features"]["train_row_cap"] == 100


def test_dataset_configs_supports_multi_dataset_shape():
    config = {
        "features": {"lags": [1], "train_row_cap": 10},
        "datasets": [
            {"name": "m4_hourly", "features": {"lags": [1, 24]}},
            {"name": "ett_h1_h48", "features": {"lags": [1, 48]}},
        ],
    }

    datasets = dataset_configs(config)

    assert [dataset["name"] for dataset in datasets] == ["m4_hourly", "ett_h1_h48"]
    assert datasets[0]["features"]["lags"] == [1, 24]
    assert datasets[1]["features"]["lags"] == [1, 48]
    assert datasets[1]["features"]["train_row_cap"] == 10
