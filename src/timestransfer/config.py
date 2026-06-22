from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config as a plain dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must contain a YAML mapping.")
    return config


def model_enabled(config: dict[str, Any], model_name: str) -> bool:
    model_config = config.get("models", {}).get(model_name, {})
    return bool(model_config.get("enabled", False))


def dataset_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize old single-dataset and new multi-dataset config shapes."""
    if "datasets" in config:
        raw_datasets = config["datasets"]
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError("config['datasets'] must be a non-empty list.")
        return [_with_feature_defaults(dataset, config.get("features", {})) for dataset in raw_datasets]

    if "dataset" not in config:
        raise ValueError("Config must include either 'dataset' or 'datasets'.")
    return [_with_feature_defaults(config["dataset"], config.get("features", {}))]


def _with_feature_defaults(dataset: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(dataset)
    features = dict(defaults)
    features.update(dataset.get("features", {}))
    if "lags" not in features:
        raise ValueError(f"Dataset {dataset.get('name', '<unknown>')} must define feature lags.")
    normalized["features"] = features
    return normalized
