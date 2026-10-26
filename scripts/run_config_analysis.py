"""Step 2 analyses: TimesFM 2.5 inference configuration (context length, flip invariance).

Reads the configuration grid (configs/timesfm_config.yaml, outputs/timesfm_config) and the
post-hoc selection/combination runs of scripts/run_timesfm_config_selection.sh, and writes
CSV tables: per-task scores, context profiles, regret against the ex-post best
configuration, Diebold-Mariano tests and bootstrap intervals against the default
(context 1024) and the longest context, the configurations select_task picks, and the
two-member combination theory before and after the context change. All scores use the
rolling protocol's evaluated windows (min 3 past windows).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from timestransfer.combination import (
    RollingProtocol,
    build_combination_table,
    load_forecasts,
    load_mase_scales,
)
from timestransfer.combination_analysis import (
    aggregate_relative,
    bootstrap_aggregate,
    dm_tests,
    pairwise_combination_theory,
    regret_summary,
    regret_table,
    relative_to_baseline,
    shared_error_decomposition,
    task_mase,
    window_losses,
)
from timestransfer.metadata import _git_metadata

CONTEXTS = [512, 1024, 2048, 4096, 8192, 16256]
TFM = [f"timesfm_2p5_ctx{c}" for c in CONTEXTS] + [
    f"timesfm_2p5_ctx{c}_noflip" for c in CONTEXTS
]
CHRONOS = [f"chronos2_ctx{c}" for c in (1024, 2048, 4096, 8192)]
DEFAULT = "timesfm_2p5_ctx1024"
LONG = "timesfm_2p5_ctx16256"
COMBINERS = ["equal_mean", "median", "select_task", "select_series", "nnls_task"]
RUNS = [
    "tfm_ctx",
    "tfm_cfg",
    "chr_ctx",
    "default_pair",
    "long_pair",
    "all_cfg",
    "default_trio",
    "long_trio",
]
TESTED = [
    "timesfm_2p5_ctx8192",
    LONG,
    "timesfm_2p5_ctx16256_noflip",
    "tfm_ctx:select_task",
    "tfm_cfg:select_task",
    "tfm_cfg:nnls_task",
    "chronos2_ctx8192",
    "default_pair:nnls_task",
    "long_pair:nnls_task",
    "long_pair:select_task",
    "long_trio:nnls_task",
    "all_cfg:nnls_task",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default="outputs/timesfm_config")
    parser.add_argument("--baseline", default="seasonal_naive")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    record: dict = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args)}

    scales = load_mase_scales([run_dir / "metrics/mase_scales.csv"])
    frames = []
    for run in RUNS:
        combined = load_forecasts([run_dir / "selection" / run / "forecasts/combined_forecasts.parquet"])
        is_combiner = combined["model"].isin(COMBINERS)
        combined.loc[is_combiner, "model"] = run + ":" + combined.loc[is_combiner, "model"]
        frames.append(combined)
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "model", "unique_id", "window", "horizon"]
    )
    models = sorted(combined["model"].unique())
    table = build_combination_table(combined, scales, models)
    record["n_rows"] = int(len(table.frame))

    relative = relative_to_baseline(task_mase(table), args.baseline)
    relative.to_csv(out / "task_scores.csv", index=False)
    aggregate_relative(relative).to_csv(out / "aggregate.csv", index=False)

    pivot = relative.pivot(index="dataset", columns="model", values="relative_mase")
    profile = pivot[[*TFM, *CHRONOS]].div(pivot[DEFAULT], axis=0)
    profile.to_csv(out / "context_profile.csv")

    regret = regret_table(relative, TFM)
    regret.to_csv(out / "regret.csv", index=False)
    regret_summary(regret).to_csv(out / "regret_summary.csv", index=False)

    losses = window_losses(table)
    tests = dm_tests(losses, TESTED, [DEFAULT, LONG])
    tests.to_csv(out / "dm_tests.csv", index=False)
    (
        tests.assign(
            better=tests["mean_diff"] < 0,
            sig_better=(tests["p_holm"] < 0.05) & (tests["mean_diff"] < 0),
            sig_worse=(tests["p_holm"] < 0.05) & (tests["mean_diff"] > 0),
        )
        .groupby(["model", "reference"])[["better", "sig_better", "sig_worse"]]
        .sum()
        .reset_index()
        .to_csv(out / "dm_summary.csv", index=False)
    )

    bootstrap = pd.concat(
        [
            bootstrap_aggregate(
                table,
                args.baseline,
                [DEFAULT, LONG],
                n_boot=args.n_boot,
                seed=args.seed,
                resample=resample,
            ).assign(resample=resample)
            for resample in ("series", "windows", "both")
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(out / "bootstrap.csv", index=False)

    # Which configuration select_task picks, per task (share of evaluated windows).
    weights = pd.read_parquet(run_dir / "selection/tfm_cfg/weights/weights.parquet")
    picks = weights[weights["method"] == "select_task"]
    columns = [c for c in picks.columns if c.startswith("w_")]
    picks = picks.assign(pick=picks[columns].idxmax(axis=1).str.removeprefix("w_timesfm_2p5_"))
    (
        picks.groupby("dataset")["pick"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .to_csv(out / "select_task_picks.csv", index=False)
    )

    # Two-member theory and shared error, before and after the context change.
    protocol = RollingProtocol()
    pair_frames, shared_frames = [], []
    for name, pool in {"default_pair": [DEFAULT, "chronos2_ctx8192"], "long_pair": [LONG, "chronos2_ctx8192"]}.items():
        pool_table = build_combination_table(combined, scales, pool)
        mask = protocol.evaluation_mask(pool_table)
        pair_frames.append(pairwise_combination_theory(pool_table, mask).assign(pool=name))
        shared_frames.append(shared_error_decomposition(pool_table, mask).assign(pool=name))
    pd.concat(pair_frames, ignore_index=True).to_csv(out / "pairwise_theory.csv", index=False)
    pd.concat(shared_frames, ignore_index=True).to_csv(out / "shared_error.csv", index=False)

    record["git"] = _git_metadata()
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
