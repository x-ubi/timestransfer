"""Chronos-2 LoRA fine-tuning per task via `Chronos2Pipeline.fit`, on the same split as
train_lora.py (train on history before the 3 oldest windows, validate on those windows,
score the combination-study windows).

Inference mirrors the benchmark's `run_chronos2` runner: univariate (no cross learning),
context 8192, batch 100, the raw 0.5 quantile as the point forecast, deciles sorted
(monotone rearrangement) and nonnegative clipping when a series' whole training history is
nonnegative. `fit` keeps the checkpoint with the lowest validation loss (evaluated every
100 steps); it trains with bf16/TF32 autocast on Ampere or newer GPUs, while inference
runs in float32 as in the benchmark.

    finetune/.venv/bin/python finetune/chronos2_ft.py --datasets ett_h1_h48 --zero-shot-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from timestransfer.config import dataset_configs, load_config  # noqa: E402
from timestransfer.data import load_dataset, rolling_origin_splits  # noqa: E402
from train_lora import long_forecasts, window_roles  # noqa: E402

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"  # same pin as the benchmark
DECILES = [round(0.1 * i, 1) for i in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs/combination.yaml"))
    parser.add_argument("--datasets", nargs="+", help="Default: every dataset in the config.")
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs/finetune/chronos2_lora"))
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5, help="fit() docstring advice for LoRA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--zero-shot-only", action="store_true", help="Score the base model; no training."
    )
    return parser.parse_args()


def load_pipeline(device: str):
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(MODEL_ID, revision=REVISION, device_map=device)


def chronos_forecast(
    pipeline, histories: list[np.ndarray], horizon: int, context: int
) -> tuple[np.ndarray, np.ndarray]:
    """(point (n, horizon), deciles (n, horizon, 9)) exactly as the benchmark runner."""
    levels = sorted({0.5, *DECILES})
    quantiles, _ = pipeline.predict_quantiles(
        [torch.tensor(h[-context:]) for h in histories],
        prediction_length=horizon,
        quantile_levels=levels,
        batch_size=100,
        context_length=context,
        cross_learning=False,
    )
    values = np.stack([q[0].to(torch.float32).cpu().numpy() for q in quantiles]).astype(float)
    point = values[..., levels.index(0.5)]
    deciles = np.sort(values[..., [levels.index(level) for level in DECILES]], axis=-1)
    nonnegative = np.array([np.nanmin(h) >= 0 for h in histories])
    point[nonnegative] = np.maximum(point[nonnegative], 0.0)
    deciles[nonnegative] = np.maximum(deciles[nonnegative], 0.0)
    return point, deciles


def run_task(args, config, dataset: dict, model_name: str, out: Path) -> tuple[pd.DataFrame, dict]:
    horizon = int(dataset["horizon"])
    n_windows = int(dataset.get("n_windows", config["project"]["n_windows"]))
    evaluated, validation = window_roles(n_windows)
    frame = load_dataset(ROOT / config["project"]["data_dir"], dataset)
    frame["unique_id"] = frame["unique_id"].astype(str)
    splits = rolling_origin_splits(frame, horizon, n_windows)

    def series_of(part: pd.DataFrame) -> list[np.ndarray]:
        part = part.sort_values(["unique_id", "ds"])
        return [g["y"].to_numpy(np.float32) for _, g in part.groupby("unique_id", sort=True)]

    record: dict = {"dataset": dataset["name"], "evaluated_windows": evaluated,
                    "validation_windows": validation}
    pipeline = load_pipeline(args.device)
    if not args.zero_shot_only:
        train_inputs = series_of(splits[validation[-1]].train)
        # Validation series end with their window's test span; fit() scores the last
        # `horizon` points of each.
        validation_inputs = [
            np.concatenate([history, target])
            for k in validation
            for history, target in zip(
                series_of(splits[k].train), series_of(splits[k].test), strict=True
            )
        ]
        start = time.monotonic()
        pipeline = pipeline.fit(
            train_inputs,
            prediction_length=horizon,
            validation_inputs=validation_inputs,
            finetune_mode="lora",
            context_length=args.context,
            learning_rate=args.lr,
            num_steps=args.steps,
            batch_size=args.batch_size,
            output_dir=out / "trainer" / dataset["name"],
            remove_printer_callback=True,
        )
        record["train_seconds"] = round(time.monotonic() - start, 1)

    start = time.monotonic()
    frames = []
    for k in evaluated:
        histories = series_of(splits[k].train)
        point, deciles = chronos_forecast(pipeline, histories, horizon, args.context)
        frames.append(long_forecasts(dataset["name"], model_name, splits[k], horizon, point, deciles, k))
    record["inference_seconds"] = round(time.monotonic() - start, 1)
    return pd.concat(frames, ignore_index=True), record


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = [d for d in dataset_configs(config) if not args.datasets or d["name"] in args.datasets]
    model_name = "chronos2_zeroshot_ft_env" if args.zero_shot_only else "chronos2_lora"
    out = Path(args.outputs_dir)
    (out / "forecasts").mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in datasets:
        forecasts, record = run_task(args, config, dataset, model_name, out)
        forecasts.to_parquet(out / "forecasts" / f"{dataset['name']}.parquet", index=False)
        records.append(record)
        print(json.dumps(record), flush=True)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "model_name": model_name,
        "revision": REVISION,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tasks": records,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with (out / f"run_{stamp}.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
