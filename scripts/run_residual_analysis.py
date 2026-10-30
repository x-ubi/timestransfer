"""Residuals of the zero-shot foundation models and their pairwise correlations.

For every forecast point (series x window x horizon step) of the combination study's
evaluated windows, each model's residual is e = (forecast - actual) / MASE scale. Writes,
per task: residual bias, spread and size per model; pairwise residual correlations
(pooled Pearson, pooled Spearman, median within-series Pearson); and a fixed random sample
of residual triples for scatter plots.

    uv run python scripts/run_residual_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from timestransfer.combination import (
    RollingProtocol,
    build_combination_table,
    load_forecasts,
    load_mase_scales,
)
from timestransfer.combination_analysis import residual_correlations, residual_summary

MEMBERS = ["timesfm_2p5", "chronos2", "tabpfn_ts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecasts", default="outputs/combination/forecasts/all_datasets_all_models.parquet")
    parser.add_argument("--mase-scales", default="outputs/combination/metrics/mase_scales.csv")
    parser.add_argument("--outputs-dir", default="outputs/combination/analysis")
    parser.add_argument("--sample-per-task", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    forecasts = load_forecasts([args.forecasts])
    table = build_combination_table(forecasts, load_mase_scales([args.mase_scales]), MEMBERS)
    mask = RollingProtocol().evaluation_mask(table)

    residual_summary(table, mask).to_csv(out / "residual_summary.csv", index=False)
    residual_correlations(table, mask).to_csv(out / "residual_correlations.csv", index=False)

    rng = np.random.default_rng(args.seed)
    use = mask & table.valid_scale
    signed = table.scaled_predictions - table.scaled_y[:, None]
    frames = []
    for code, dataset in enumerate(table.datasets):
        rows = np.flatnonzero(use & (table.dataset_codes == code))
        pick = rng.choice(rows, size=min(args.sample_per_task, len(rows)), replace=False)
        frames.append(pd.DataFrame(signed[pick], columns=MEMBERS).assign(dataset=dataset))
    pd.concat(frames, ignore_index=True).to_csv(out / "residual_sample.csv", index=False)
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
