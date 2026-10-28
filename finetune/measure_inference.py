"""Measure GPU inference time per forecast origin for the step 4 cost table.

Every TimesFM variant runs through the same HF implementation (timesfm_hf.forecast) and
every Chronos-2 variant through chronos2_ft.chronos_forecast, with LoRA adapters merged
into the base weights (as a deployment would), so differences reflect context length
and model, not implementation or logging overhead. For each task, a few windows are
timed after one warm-up window, with CUDA synchronisation; the mean is reported in
seconds per window (one forecast origin, all series of the task). Run with the GPU
otherwise idle.

The HF TimesFM pads every input to the requested context, so a context longer than the
longest history only adds padding (forecasts agree to ~1e-5 relative, checked on M4, ILI
and Exchange). TimesFM contexts are therefore capped at the longest history in the call,
rounded up to the 32-step patch, as a deployment would do. Chronos-2 pads only to the
longest input already.

    finetune/.venv/bin/python finetune/measure_inference.py --out outputs/strategy_comparison/timings.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from timestransfer.config import dataset_configs, load_config  # noqa: E402
from timestransfer.data import load_dataset, rolling_origin_splits  # noqa: E402
from chronos2_ft import chronos_forecast, load_finetuned, load_pipeline  # noqa: E402
from timesfm_hf import forecast, load_model  # noqa: E402

# strategy -> (family, context, adapter directory or None)
STRATEGIES = {
    "tfm_default": ("timesfm", 1024, None),
    "tfm_ctx8192": ("timesfm", 8192, None),
    "tfm_long": ("timesfm", 16256, None),
    "tfm_lora": ("timesfm", 1024, "outputs/finetune/timesfm_lora_v2/adapters"),
    "tfm_lora_long": ("timesfm", 16256, "outputs/finetune/timesfm_lora_v2/adapters"),
    "chr_ctx2048": ("chronos", 2048, None),
    "chr_default": ("chronos", 8192, None),
    "chr_lora_c2048": ("chronos", 2048, "outputs/finetune/chronos2_lora_c2048/trainer"),
    "chr_lora_long": ("chronos", 8192, "outputs/finetune/chronos2_lora_c2048/trainer"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs/combination.yaml"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--windows", nargs="+", type=int, default=[0, 3, 6])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(ROOT / "outputs/strategy_comparison/timings.csv"))
    return parser.parse_args()


def capped_context(histories: list[np.ndarray], context: int, patch: int = 32) -> int:
    return min(context, -(-max(len(h) for h in histories) // patch) * patch)


def timesfm_with_adapter(adapter: Path, device: str, lora_r: int = 4, lora_alpha: int = 8):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        load_model(device),
        LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules="all-linear", bias="none"),
    )
    model.load_state_dict(torch.load(adapter, map_location=device), strict=False)
    return model.merge_and_unload().eval()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = [d for d in dataset_configs(config) if not args.datasets or d["name"] in args.datasets]
    rows = []
    for dataset in datasets:
        horizon = int(dataset["horizon"])
        n_windows = int(dataset.get("n_windows", config["project"]["n_windows"]))
        frame = load_dataset(ROOT / config["project"]["data_dir"], dataset)
        frame["unique_id"] = frame["unique_id"].astype(str)
        splits = rolling_origin_splits(frame, horizon, n_windows)
        windows = [w for w in args.windows if w < n_windows]
        histories = {
            k: [
                g["y"].to_numpy(np.float32)
                for _, g in splits[k].train.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True)
            ]
            for k in [n_windows - 1, *windows]
        }
        for strategy in args.strategies:
            family, context, adapters = STRATEGIES[strategy]
            if family == "timesfm":
                model = (
                    timesfm_with_adapter(ROOT / adapters / f"{dataset['name']}.pt", args.device)
                    if adapters else load_model(args.device)
                )
                run = lambda h: forecast(model, h, horizon, capped_context(h, context))  # noqa: E731
            else:
                model = (
                    load_finetuned(ROOT / adapters / dataset["name"] / "finetuned-ckpt", args.device)
                    if adapters else load_pipeline(args.device)
                )
                run = lambda h: chronos_forecast(model, h, horizon, context)  # noqa: E731
            run(histories[n_windows - 1])  # warm-up
            seconds = []
            for k in windows:
                torch.cuda.synchronize()
                start = time.perf_counter()
                run(histories[k])
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
            rows.append(
                {
                    "strategy": strategy,
                    "dataset": dataset["name"],
                    "seconds_per_window": float(np.mean(seconds)),
                    "n_series": len(histories[windows[0]]),
                    "windows_timed": len(seconds),
                    "gpu": torch.cuda.get_device_name(0),
                }
            )
            print(rows[-1], flush=True)
            del model
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
