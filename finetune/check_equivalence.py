"""Check that the HF Transformers TimesFM 2.5 checkpoint reproduces the benchmark's TimesFM.

Runs `TimesFm2_5ModelForPrediction` zero-shot on the same contexts as the 20-origin
combination run (configs/combination.yaml, context 1024, all forecast flags on) and
compares its median and deciles with the stored `timesfm_2p5` forecasts in
outputs/combination/forecasts, in MASE units. Inference goes through
`timesfm_hf.forecast`, which applies the benchmark's quantile-crossing repair and
nonnegative clipping (the HF model has neither in the same form).

    finetune/.venv/bin/python finetune/check_equivalence.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timestransfer.config import dataset_configs, load_config  # noqa: E402
from timestransfer.data import load_dataset, rolling_origin_splits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timesfm_hf import MEDIAN, forecast, load_model  # noqa: E402

QUANTILES = [f"q{i}0" for i in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs/combination.yaml"))
    parser.add_argument(
        "--forecasts",
        default=str(ROOT / "outputs/combination/forecasts/all_datasets_all_models.parquet"),
    )
    parser.add_argument(
        "--mase-scales", default=str(ROOT / "outputs/combination/metrics/mase_scales.csv")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["m4_hourly", "ett_h1_h48", "traffic_h48", "weather_h48", "ili_h48"],
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[0, 5])
    parser.add_argument("--series-per-dataset", type=int, default=4)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.device)

    config = load_config(args.config)
    data_dir = ROOT / config["project"]["data_dir"]
    datasets = {d["name"]: d for d in dataset_configs(config)}
    stored = pd.read_parquet(
        args.forecasts,
        columns=["dataset", "model", "unique_id", "window", "horizon", "y_pred", *QUANTILES],
        filters=[("model", "==", "timesfm_2p5"), ("dataset", "in", args.datasets)],
    )
    stored["unique_id"] = stored["unique_id"].astype(str)
    scales = pd.read_csv(args.mase_scales).astype({"unique_id": str})

    rows = []
    for name in args.datasets:
        dataset = datasets[name]
        horizon = int(dataset["horizon"])
        n_windows = int(dataset.get("n_windows", config["project"]["n_windows"]))
        frame = load_dataset(data_dir, dataset)
        frame["unique_id"] = frame["unique_id"].astype(str)
        ids = sorted(frame["unique_id"].unique())[: args.series_per_dataset]
        splits = rolling_origin_splits(frame[frame["unique_id"].isin(ids)], horizon, n_windows)
        for window in args.windows:
            train = splits[window].train.sort_values(["unique_id", "ds"])
            histories = {u: g["y"].to_numpy(np.float32) for u, g in train.groupby("unique_id")}
            full = forecast(model, [histories[u] for u in ids], horizon, args.context)
            for i, unique_id in enumerate(ids):
                predicted = full[i]
                reference = stored[
                    (stored["dataset"] == name)
                    & (stored["unique_id"] == unique_id)
                    & (stored["window"] == window)
                ].sort_values("horizon")
                scale = scales[
                    (scales["dataset"] == name)
                    & (scales["unique_id"] == unique_id)
                    & (scales["window"] == window)
                ]["mase_scale"].iloc[0]
                median_diff = np.abs(predicted[:, MEDIAN] - reference["y_pred"].to_numpy()) / scale
                quantile_diff = np.abs(predicted[:, 1:] - reference[QUANTILES].to_numpy()) / scale
                rows.append(
                    {
                        "dataset": name,
                        "window": window,
                        "unique_id": unique_id,
                        "context_used": min(len(histories[unique_id]), args.context),
                        "max_median_diff_mase": float(median_diff.max()),
                        "max_quantile_diff_mase": float(quantile_diff.max()),
                    }
                )
    result = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(result.to_string(index=False))
    print("\nWorst median difference per dataset (MASE units):")
    print(result.groupby("dataset")[["max_median_diff_mase", "max_quantile_diff_mase"]].max())


if __name__ == "__main__":
    main()
