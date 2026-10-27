"""Chronos-2 LoRA fine-tuning per task via `Chronos2Pipeline.fit`, on the same split as
train_lora.py (train on history before the 3 oldest windows, validate on those windows,
score the combination-study windows).

Inference mirrors the benchmark's `run_chronos2` runner: univariate (no cross learning),
context 8192, batch 100, the raw 0.5 quantile as the point forecast, deciles sorted
(monotone rearrangement) and nonnegative clipping when a series' whole training history is
nonnegative. `fit` keeps the checkpoint with the lowest validation loss (evaluated every
100 steps); it trains with bf16/TF32 autocast on Ampere or newer GPUs, while inference
runs in float32 as in the benchmark.

`fit` never scores the untrained model, so after training the fine-tuned and the zero-shot
model are compared on the validation windows with the pinball loss used by train_lora.py
(nine deciles, per-sample normalised by the context mean and std, at the training
context); the fine-tuned model is used only if it is better, as with step-0 early stopping
in train_lora.py. Forecasts cover all windows (score only windows 0..K-4) at each
``--inference-contexts`` value (model name suffix ``_inf<context>`` for contexts other
than ``--context``); `fit` keeps the model's maximum context, so a model fine-tuned at a
short context can forecast at a longer one. ``--load-finetuned`` reuses saved models.

The pipeline returned by `fit` is not used for forecasting: its model is still in
training mode (dropout 0.1 active), Accelerate leaves a bf16 autocast wrapper on its
forward, and `fit` leaves TF32 matmuls enabled for the rest of the process. Instead, the
precision settings are restored and the saved adapter is applied to a freshly loaded
pinned base model (saved adapters reference the base model without a revision), in fp32
and eval mode, exactly as ``--load-finetuned`` does.

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
    parser.add_argument("--batch-size", type=int, default=256, help="Effective batch size.")
    parser.add_argument(
        "--grad-accum", type=int, default=8,
        help="Gradient accumulation steps; per-step batch = batch-size / grad-accum "
        "(256 at context 8192 does not fit in 24 GB).",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="fit() docstring advice for LoRA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--zero-shot-only", action="store_true", help="Score the base model; no training."
    )
    parser.add_argument(
        "--inference-contexts", nargs="+", type=int,
        help="Contexts to forecast at (default: --context only).",
    )
    parser.add_argument(
        "--load-finetuned",
        help="Trainer directory of an earlier run (<dir>/<dataset>/finetuned-ckpt); skips training.",
    )
    return parser.parse_args()


def load_pipeline(device: str):
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(MODEL_ID, revision=REVISION, device_map=device)


def load_finetuned(path: Path, device: str):
    """Saved LoRA adapter merged into the pinned base model, fp32, eval mode."""
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    base = load_pipeline(device).model
    model = PeftModel.from_pretrained(base, path).merge_and_unload()
    return Chronos2Pipeline(model=model.eval())


def _restore_fp32_matmul() -> None:
    """Undo fit()'s TF32: "none" would defer to the legacy flag fit() leaves set."""
    torch.backends.cuda.matmul.fp32_precision = "ieee"


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


def validation_loss(
    pipeline, windows: list[tuple[list[np.ndarray], np.ndarray]], horizon: int, context: int
) -> float:
    """train_lora.pinball_loss in numpy: deciles vs targets, per-sample normalised by the
    mean and std of the context the model sees."""
    levels = np.array(DECILES)
    total, count = 0.0, 0
    for histories, targets in windows:
        _, deciles = chronos_forecast(pipeline, histories, horizon, context)
        contexts = [h[-context:] for h in histories]
        location = np.array([c.mean() for c in contexts])[:, None]
        spread = np.array([c.std(ddof=1) if len(c) > 1 else 1.0 for c in contexts])
        spread = np.where(spread > 1e-6, spread, 1.0)[:, None]
        errors = ((targets - location) / spread)[..., None] - (deciles - location[..., None]) / spread[..., None]
        loss = np.maximum(levels * errors, (levels - 1) * errors)
        total += float(loss.mean(axis=(1, 2)).sum())
        count += len(histories)
    return total / count


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
        val_windows = [
            (series_of(splits[k].train),
             np.stack(series_of(splits[k].test)).astype(float))
            for k in validation
        ]
        record["zero_shot_val_loss"] = validation_loss(pipeline, val_windows, horizon, args.context)
    if args.load_finetuned:
        path = Path(args.load_finetuned) / dataset["name"] / "finetuned-ckpt"
        finetuned = load_finetuned(path, args.device)
        record["finetuned_from"] = str(path)
    elif not args.zero_shot_only:
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
        pipeline.fit(
            train_inputs,
            prediction_length=horizon,
            validation_inputs=validation_inputs,
            finetune_mode="lora",
            context_length=args.context,
            learning_rate=args.lr,
            num_steps=args.steps,
            batch_size=args.batch_size // args.grad_accum,
            gradient_accumulation_steps=args.grad_accum,
            output_dir=out / "trainer" / dataset["name"],
            remove_printer_callback=True,
        )
        record["train_seconds"] = round(time.monotonic() - start, 1)
        _restore_fp32_matmul()
        finetuned = load_finetuned(out / "trainer" / dataset["name"] / "finetuned-ckpt", args.device)
    if not args.zero_shot_only:
        record["finetuned_val_loss"] = validation_loss(finetuned, val_windows, horizon, args.context)
        if record["finetuned_val_loss"] < record["zero_shot_val_loss"]:
            pipeline = finetuned
            record["selected"] = "finetuned"
        else:
            record["selected"] = "zero_shot"

    frames = []
    record["inference_seconds"] = {}
    for context in args.inference_contexts or [args.context]:
        name = model_name if context == args.context else f"{model_name}_inf{context}"
        start = time.monotonic()
        for k in [*evaluated, *validation]:
            histories = series_of(splits[k].train)
            point, deciles = chronos_forecast(pipeline, histories, horizon, context)
            frames.append(
                long_forecasts(dataset["name"], name, splits[k], horizon, point, deciles, k)
            )
        record["inference_seconds"][name] = round(time.monotonic() - start, 1)
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
