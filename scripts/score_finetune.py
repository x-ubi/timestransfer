"""Score fine-tuned forecasts (finetune/*.py outputs) against zero-shot references.

Joins the fine-tuned forecasts with reference models from the configuration grid
(outputs/timesfm_config) on the same series, windows and horizons, and prints per-task
MASE relative to SeasonalNaive plus each model's ratio to the references. Only the tasks
present in the fine-tuned forecasts and the evaluated windows 0..K-4 are scored.

    uv run python scripts/score_finetune.py outputs/finetune/timesfm_lora/forecasts/*.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from timestransfer.combination import build_combination_table, load_forecasts, load_mase_scales
from timestransfer.combination_analysis import relative_to_baseline, task_mase

REFERENCES = ["timesfm_2p5_ctx1024", "timesfm_2p5_ctx16256", "chronos2_ctx8192"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("forecasts", nargs="+")
    parser.add_argument(
        "--references-forecasts",
        default="outputs/timesfm_config/forecasts/all_datasets_all_models.parquet",
    )
    parser.add_argument("--mase-scales", default="outputs/timesfm_config/metrics/mase_scales.csv")
    parser.add_argument("--references", nargs="+", default=REFERENCES)
    parser.add_argument("--baseline", default="seasonal_naive")
    parser.add_argument("--out", help="Optional CSV of per-task relative MASE.")
    args = parser.parse_args()

    tuned = load_forecasts(args.forecasts)
    keys = ["dataset", "unique_id", "window", "horizon"]
    references = load_forecasts([args.references_forecasts])
    references = references[
        references["model"].isin([*args.references, args.baseline])
        & references["dataset"].isin(tuned["dataset"].unique())
    ]
    # Evaluated windows only: the 3 oldest windows are the fine-tuning validation windows.
    n_windows = references.groupby("dataset")["window"].max() + 1
    tuned = tuned[tuned["window"] < tuned["dataset"].map(n_windows) - 3]
    references = references.merge(tuned[keys].drop_duplicates(), on=keys)
    forecasts = pd.concat([tuned, references], ignore_index=True)
    models = sorted(forecasts["model"].unique())
    table = build_combination_table(forecasts, load_mase_scales([args.mase_scales]), models)
    relative = relative_to_baseline(task_mase(table), args.baseline)
    pivot = relative.pivot(index="dataset", columns="model", values="relative_mase")
    pivot = pivot.drop(columns=args.baseline)
    tuned_models = sorted(tuned["model"].unique())
    for model in tuned_models:
        for reference in args.references:
            pivot[f"{model}/{reference}"] = pivot[model] / pivot[reference]
    geomean = np.exp(np.log(pivot).mean()).rename("geomean")
    pd.set_option("display.width", 250)
    print(pd.concat([pivot, geomean.to_frame().T]).round(4).to_string())
    if args.out:
        pivot.to_csv(args.out)


if __name__ == "__main__":
    main()
