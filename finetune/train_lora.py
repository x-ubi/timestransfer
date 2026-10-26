"""LoRA fine-tuning of TimesFM 2.5 per task, scored on the combination-study windows.

Protocol (one adapter per task):
- Windows follow configs/combination.yaml: K rolling origins with stride = horizon. The
  combination study evaluates windows 0..K-4 (they have >= 3 past windows); the 3 oldest
  windows K-3..K-1 are the validation windows here, and training samples come only from
  the history before the oldest validation window. No evaluated test value is seen in
  training or model selection.
- Training samples: random (series, cutoff) pairs inside that history; the context is the
  last ``context`` points before the cutoff (shorter when the history is shorter; the model
  pads), the target is the next ``horizon`` points.
- Loss: pinball loss over the nine decile channels (q0.1..q0.9, output channels 1..9), in
  each sample's context-normalised units. The built-in HF loss is not used: it applies MSE
  to the median channel and pairs the mean head and q0.1..q0.4 with the wrong levels.
- Inference matches the benchmark (all flags on, crossing repair, nonnegative clipping) via
  timesfm_hf.forecast; the adapter with the lowest validation pinball loss is kept.

    finetune/.venv/bin/python finetune/train_lora.py --datasets ett_h1_h48 --device cuda
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
from timesfm_hf import MEDIAN, REVISION, forecast, load_model  # noqa: E402

LEVELS = torch.arange(1, 10, dtype=torch.float32) / 10  # output channels 1..9
QUANTILES = [f"q{i}0" for i in range(1, 10)]
N_VALIDATION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs/combination.yaml"))
    parser.add_argument("--datasets", nargs="+", help="Default: every dataset in the config.")
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs/finetune/timesfm_lora"))
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--min-context", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--zero-shot-only", action="store_true", help="Score the base model; no training."
    )
    return parser.parse_args()


def window_roles(n_windows: int) -> tuple[list[int], list[int]]:
    """(evaluated, validation) window indices; validation are the 3 oldest windows."""
    if n_windows <= N_VALIDATION:
        raise ValueError(f"Need more than {N_VALIDATION} windows, got {n_windows}.")
    return list(range(n_windows - N_VALIDATION)), list(range(n_windows - N_VALIDATION, n_windows))


def sample_batch(
    histories: list[np.ndarray],
    batch_size: int,
    context: int,
    horizon: int,
    min_context: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Random (context, target) pairs strictly inside the given training histories."""
    eligible = [h for h in histories if len(h) >= min_context + horizon]
    if not eligible:
        raise ValueError("No training history is long enough for one sample.")
    contexts, targets = [], []
    for index in rng.integers(len(eligible), size=batch_size):
        history = eligible[index]
        cutoff = int(rng.integers(min_context, len(history) - horizon + 1))
        contexts.append(history[max(0, cutoff - context) : cutoff])
        targets.append(history[cutoff : cutoff + horizon])
    return contexts, np.stack(targets)


def pinball_loss(
    predictions: torch.Tensor, targets: torch.Tensor, contexts: list[torch.Tensor]
) -> torch.Tensor:
    """Mean pinball loss of channels 1..9 against levels 0.1..0.9, per-sample normalised.

    ``predictions`` is (batch, >= horizon, 10) in data units; ``targets`` (batch, horizon).
    """
    horizon = targets.shape[1]
    location = torch.stack([c.mean() for c in contexts])[:, None]
    spread = torch.stack([c.std() if len(c) > 1 else c.new_tensor(1.0) for c in contexts])
    spread = torch.where(spread > 1e-6, spread, torch.ones_like(spread))[:, None]
    quantiles = (predictions[:, :horizon, 1:10] - location[..., None]) / spread[..., None]
    target = ((targets - location) / spread)[..., None]
    errors = target - quantiles
    levels = LEVELS.to(errors.device)
    return torch.maximum(levels * errors, (levels - 1) * errors).mean()


def validation_loss(model, windows: list[tuple[list[np.ndarray], np.ndarray]], context: int) -> float:
    device = next(model.parameters()).device
    total, count = 0.0, 0
    with torch.no_grad():
        for histories, targets in windows:
            for start in range(0, len(histories), 64):
                batch = [
                    torch.from_numpy(h[-context:].astype(np.float32)).to(device)
                    for h in histories[start : start + 64]
                ]
                target = torch.from_numpy(targets[start : start + 64].astype(np.float32)).to(device)
                out = model(past_values=batch, forecast_context_len=context, truncate_negative=False)
                total += float(pinball_loss(out.full_predictions, target, batch)) * len(batch)
                count += len(batch)
    return total / count


def long_forecasts(
    dataset: str,
    model_name: str,
    split,
    horizon: int,
    point: np.ndarray,
    quantiles: np.ndarray,
    window: int,
) -> pd.DataFrame:
    """Benchmark long format for one window from (n_series, horizon) point forecasts and
    (n_series, horizon, 9) deciles, series in sorted-id order."""
    test = split.test.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    ids = sorted(test["unique_id"].unique())
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model_name,
            "unique_id": np.repeat(ids, horizon),
            "horizon": np.tile(np.arange(1, horizon + 1), len(ids)),
            "ds": test["ds"].astype(str).to_numpy(),
            "y_true": test["y"].to_numpy(dtype=float),
            "y_pred": point.reshape(-1),
            "window": window,
            "forecast_horizon": horizon,
        }
    )
    for column, values in zip(QUANTILES, np.moveaxis(quantiles, -1, 0), strict=True):
        frame[column] = values.reshape(-1)
    return frame


def run_task(args, config, dataset: dict, model_name: str) -> tuple[pd.DataFrame, dict]:
    from peft import LoraConfig, get_peft_model

    horizon = int(dataset["horizon"])
    n_windows = int(dataset.get("n_windows", config["project"]["n_windows"]))
    evaluated, validation = window_roles(n_windows)
    frame = load_dataset(ROOT / config["project"]["data_dir"], dataset)
    frame["unique_id"] = frame["unique_id"].astype(str)
    splits = rolling_origin_splits(frame, horizon, n_windows)

    def histories_of(split) -> tuple[list[str], list[np.ndarray]]:
        train = split.train.sort_values(["unique_id", "ds"])
        groups = {u: g["y"].to_numpy(np.float32) for u, g in train.groupby("unique_id")}
        ids = sorted(groups)
        return ids, [groups[u] for u in ids]

    record: dict = {"dataset": dataset["name"], "evaluated_windows": evaluated,
                    "validation_windows": validation}
    model = load_model(args.device)
    if not args.zero_shot_only:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        _, train_histories = histories_of(splits[validation[-1]])
        val_windows = []
        for k in validation:
            _, histories = histories_of(splits[k])
            targets = splits[k].test.sort_values(["unique_id", "ds"])["y"].to_numpy(np.float32)
            val_windows.append((histories, targets.reshape(len(histories), horizon)))

        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
                          lora_dropout=args.lora_dropout, bias="none")
        model = get_peft_model(model, lora)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
        model.eval()
        best = {"step": 0, "loss": validation_loss(model, val_windows, args.context)}
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
        history = [{"step": 0, "val_loss": best["loss"]}]
        start = time.monotonic()
        for step in range(1, args.steps + 1):
            model.train()
            contexts, targets = sample_batch(train_histories, args.batch_size, args.context,
                                             horizon, args.min_context, rng)
            batch = [torch.from_numpy(c).to(args.device) for c in contexts]
            target = torch.from_numpy(targets).to(args.device)
            out = model(past_values=batch, forecast_context_len=args.context,
                        truncate_negative=False)
            loss = pinball_loss(out.full_predictions, target, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                val = validation_loss(model, val_windows, args.context)
                history.append({"step": step, "train_loss": loss.item(), "val_loss": val})
                if val < best["loss"]:
                    best = {"step": step, "loss": val}
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items() if "lora_" in k}
        model.load_state_dict(best_state, strict=False)
        model.eval()
        record.update(train_seconds=round(time.monotonic() - start, 1), best_step=best["step"],
                      best_val_loss=best["loss"], zero_shot_val_loss=history[0]["val_loss"],
                      history=history)

    start = time.monotonic()
    frames = []
    for k in evaluated:
        _, histories = histories_of(splits[k])
        full = forecast(model, histories, horizon, args.context)
        frames.append(
            long_forecasts(
                dataset["name"], model_name, splits[k], horizon, full[:, :, MEDIAN], full[:, :, 1:10], k
            )
        )
    record["inference_seconds"] = round(time.monotonic() - start, 1)
    return pd.concat(frames, ignore_index=True), record


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = [d for d in dataset_configs(config) if not args.datasets or d["name"] in args.datasets]
    model_name = "timesfm_2p5_hf_zeroshot" if args.zero_shot_only else "timesfm_2p5_lora"
    out = Path(args.outputs_dir)
    (out / "forecasts").mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in datasets:
        forecasts, record = run_task(args, config, dataset, model_name)
        forecasts.to_parquet(out / "forecasts" / f"{dataset['name']}.parquet", index=False)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "history"}), flush=True)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "model_name": model_name,
        "hf_revision": REVISION,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tasks": records,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with (out / f"run_{stamp}.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
