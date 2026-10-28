"""Step 4: combination vs configuration vs fine-tuning on one common evaluation set.

Collects zero-shot forecasts (configuration grid, TabPFN-TS from the combination run) and
fine-tuned forecasts (finetune/train_lora.py, finetune/chronos2_ft.py), renames them to
short strategy names, and drops each task's 3 oldest windows: they are the fine-tuning
validation (early-stopping) windows, so forecasts there are not out of sample. Combiners
are refitted under the rolling protocol on the remaining windows only, which leaves the
windows 0..K-7 as the common evaluation set for every strategy (14 windows, 4 for M4 and
ILI). Optional inputs that do not exist yet are skipped.

Writes to --outputs-dir: per-task scores and aggregates, bootstrap intervals (five
resampling schemes, including moving blocks of windows), bootstrap intervals of
combination gains and their differences, DM tests, regret, seed robustness, and an
accuracy-cost table (one-time training vs recurring inference GPU time).

    nice -n 19 uv run python scripts/run_strategy_comparison.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from timestransfer.combination import (
    FORECAST_COLUMNS,
    CombinationTable,
    EqualMean,
    LinearCombiner,
    RollingProtocol,
    build_combination_table,
    load_mase_scales,
    nnls_weights,
    rolling_combine,
    select_best,
)
from timestransfer.combination_analysis import (
    RESAMPLE_SCHEMES,
    aggregate_relative,
    bootstrap_aggregate,
    bootstrap_contrasts,
    bootstrap_replicates,
    dm_tests,
    regret_summary,
    regret_table,
    relative_to_baseline,
    task_mase,
    window_losses,
)
from timestransfer.metadata import _git_metadata

GRID = "outputs/timesfm_config/forecasts/all_datasets_all_models.parquet"
CONTEXTS = [512, 1024, 2048, 4096, 8192, 16256]
GRID_ALIASES = {
    "seasonal_naive": "seasonal_naive",
    "timesfm_2p5_ctx1024": "tfm_default",
    "timesfm_2p5_ctx8192": "tfm_ctx8192",
    "timesfm_2p5_ctx16256": "tfm_long",
    **{f"timesfm_2p5_ctx{c}": f"tfm_ctx{c}" for c in (512, 2048, 4096)},
    "chronos2_ctx8192": "chr_default",
    "chronos2_ctx2048": "chr_ctx2048",
}
CONTEXT_POOL = ["tfm_ctx512", "tfm_default", "tfm_ctx2048", "tfm_ctx4096", "tfm_ctx8192", "tfm_long"]
# (glob, {source model: alias}); missing sources are skipped.
OPTIONAL_SOURCES = [
    ("outputs/combination/forecasts/*_tabpfn_ts.parquet", {"tabpfn_ts": "tabpfn"}),
    (
        "outputs/finetune/timesfm_lora_v2/forecasts/*.parquet",
        {"timesfm_2p5_lora": "tfm_lora", "timesfm_2p5_lora_inf16256": "tfm_lora_long"},
    ),
    *[
        (
            f"outputs/finetune/timesfm_lora_seed{s}/forecasts/*.parquet",
            {
                f"timesfm_2p5_lora_s{s}": f"tfm_lora_s{s}",
                f"timesfm_2p5_lora_s{s}_inf16256": f"tfm_lora_long_s{s}",
            },
        )
        for s in (1, 2)
    ],
    (
        "outputs/finetune/chronos2_lora_c2048/forecasts/*.parquet",
        {"chronos2_lora": "chr_lora_c2048", "chronos2_lora_inf8192": "chr_lora_long"},
    ),
]
POOLS = {
    "default_pair": ["tfm_default", "chr_default"],
    "cheap_pair": ["tfm_default", "chr_ctx2048"],
    "default_trio": ["tfm_default", "chr_default", "tabpfn"],
    "long_pair": ["tfm_long", "chr_default"],
    "long_trio": ["tfm_long", "chr_default", "tabpfn"],
    "lora_long_pair": ["tfm_lora_long", "chr_default"],
    "lora_long_trio": ["tfm_lora_long", "chr_default", "tabpfn"],
    "both_ft_pair": ["tfm_lora_long", "chr_lora_long"],
}
SELECTION_POOLS = {"ctx_select": CONTEXT_POOL}
PRIMARY = "nnls_task"
REFERENCES = ["tfm_default", "tfm_long"]
# Combination gain = combination / its TimesFM member; contrasts compare the gains.
GAINS = {
    "gain_default": ("default_pair", "tfm_default"),
    "gain_long": ("long_pair", "tfm_long"),
    "gain_lora_long": ("lora_long_pair", "tfm_lora_long"),
    "gain_both_ft": ("both_ft_pair", "tfm_lora_long"),
}
GAIN_DIFFERENCES = {
    "gain_long_vs_default": ("gain_long", "gain_default"),
    "gain_lora_long_vs_long": ("gain_lora_long", "gain_long"),
    "gain_lora_long_vs_default": ("gain_lora_long", "gain_default"),
    "gain_both_ft_vs_lora_long": ("gain_both_ft", "gain_lora_long"),
}
# Fine-tuned strategy -> (training run records, the run's model name for its inference time)
TRAINING_RUNS = {
    "tfm_lora": ("outputs/finetune/timesfm_lora_v2/run_*.json", "timesfm_2p5_lora"),
    "tfm_lora_long": ("outputs/finetune/timesfm_lora_v2/run_*.json", "timesfm_2p5_lora_inf16256"),
    "chr_lora_c2048": ("outputs/finetune/chronos2_lora_c2048/run_*.json", "chronos2_lora"),
    "chr_lora_long": ("outputs/finetune/chronos2_lora_c2048/run_*.json", "chronos2_lora_inf8192"),
}
LOG_TIMINGS = {  # zero-shot strategy -> (progress log, model name in the log)
    **{alias: ("outputs/timesfm_config/progress.log", model) for model, alias in GRID_ALIASES.items()},
    "tabpfn": ("outputs/combination/progress.log", "tabpfn_ts"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mase-scales", default="outputs/timesfm_config/metrics/mase_scales.csv")
    parser.add_argument("--outputs-dir", default="outputs/strategy_comparison")
    parser.add_argument("--baseline", default="seasonal_naive")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-length", type=int, default=3)
    parser.add_argument(
        "--timings",
        help="CSV (strategy, dataset, seconds_per_window) of measured inference times; "
        "progress logs and run records are used for strategies it does not cover.",
    )
    return parser.parse_args()


# --- data


def _read(paths: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path, columns=FORECAST_COLUMNS, filters=[("model", "in", list(aliases))])
        for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["model"] = frame["model"].map(aliases)
    frame["unique_id"] = frame["unique_id"].astype(str)
    frame["ds"] = frame["ds"].astype(str)
    return frame


def load_strategies() -> tuple[pd.DataFrame, list[str]]:
    frames = [_read([GRID], GRID_ALIASES)]
    skipped = []
    for pattern, aliases in OPTIONAL_SOURCES:
        paths = sorted(glob.glob(pattern))
        if not paths:
            skipped.append(pattern)
            continue
        frames.append(_read(paths, aliases))
    forecasts = pd.concat(frames, ignore_index=True)
    # Drop the 3 oldest windows of every task (fine-tuning validation windows).
    n_windows = forecasts.groupby("dataset")["window"].max() + 1
    forecasts = forecasts[forecasts["window"] < forecasts["dataset"].map(n_windows) - 3]
    return forecasts.reset_index(drop=True), skipped


def subset_table(table: CombinationTable, models: list[str]) -> CombinationTable:
    columns = [c for c in table.frame.columns if c not in table.members]
    return CombinationTable(table.frame.loc[:, [*columns, *models]], models)


def combiners() -> list:
    return [
        EqualMean(),
        LinearCombiner("select_task", select_best, level="task"),
        LinearCombiner("nnls_task", nnls_weights, level="task"),
    ]


# --- costs


def _log_seconds_per_window(path: str, model: str) -> pd.Series:
    """Mean logged seconds per window and dataset for one model of a benchmark run."""
    text = Path(path).read_text().replace("\r", "\n")
    pattern = re.compile(r"^\[(\w+)\] window \d+/\d+ \|.*?\b" + re.escape(model) + r" ok ([\d.]+)s", re.M)
    rows = [(m.group(1), float(m.group(2))) for m in pattern.finditer(text)]
    return pd.DataFrame(rows, columns=["dataset", "seconds"]).groupby("dataset")["seconds"].mean()


def cost_table(strategies: list[str], timings_path: str | None, n_windows: pd.Series) -> pd.DataFrame:
    """GPU seconds per strategy and task: one-time training and inference per forecast origin."""
    measured = pd.read_csv(timings_path) if timings_path and Path(timings_path).exists() else None
    rows = []
    for strategy in strategies:
        train, infer, source = {}, {}, None
        if strategy in TRAINING_RUNS:
            pattern, run_model = TRAINING_RUNS[strategy]
            paths = sorted(glob.glob(pattern))
            if not paths:
                continue
            tasks = json.loads(Path(paths[-1]).read_text())["tasks"]
            for task in tasks:
                train[task["dataset"]] = task.get("train_seconds", 0.0)
                infer[task["dataset"]] = task["inference_seconds"][run_model] / n_windows[task["dataset"]]
            source = "run record"
        elif strategy in LOG_TIMINGS:
            path, log_model = LOG_TIMINGS[strategy]
            infer = _log_seconds_per_window(path, log_model).to_dict()
            source = "progress log (includes data handling)"
        if measured is not None and strategy in set(measured["strategy"]):
            rows_measured = measured[measured["strategy"] == strategy]
            infer = rows_measured.set_index("dataset")["seconds_per_window"].to_dict()
            source = "measured"
        for dataset, seconds in infer.items():
            rows.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    "train_seconds": train.get(dataset, 0.0),
                    "inference_seconds_per_origin": seconds,
                    "timing_source": source,
                }
            )
    return pd.DataFrame(rows)


def accuracy_cost(aggregate: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    totals = costs.groupby("strategy").agg(
        one_time_gpu_minutes=("train_seconds", lambda s: s.sum() / 60),
        recurring_gpu_seconds_per_origin=("inference_seconds_per_origin", "sum"),
        timing_source=("timing_source", "first"),
    )
    rows = []
    for model, score in aggregate.set_index("model")["geomean_relative_mase"].items():
        if ":" in model:  # combination: sum of its members
            pool = model.split(":")[0]
            members = POOLS.get(pool) or SELECTION_POOLS.get(pool, [])
            if not set(members) <= set(totals.index):
                continue
            part = totals.loc[members]
            rows.append(
                {
                    "strategy": model,
                    "geomean_relative_mase": score,
                    "one_time_gpu_minutes": part["one_time_gpu_minutes"].sum(),
                    "recurring_gpu_seconds_per_origin": part["recurring_gpu_seconds_per_origin"].sum(),
                    "timing_source": "sum of members",
                }
            )
        elif model in totals.index:
            rows.append({"strategy": model, "geomean_relative_mase": score, **totals.loc[model].to_dict()})
    out = pd.DataFrame(rows)
    default = aggregate.set_index("model").loc["tfm_default", "geomean_relative_mase"]
    out["ratio_to_tfm_default"] = out["geomean_relative_mase"] / default
    return out.sort_values("geomean_relative_mase").reset_index(drop=True)


def break_even(costs: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Forecast origins after which strategy a (more one-time cost) becomes cheaper than b."""
    index = costs.set_index("strategy")
    rows = []
    for a, b in pairs:
        if a not in index.index or b not in index.index:
            continue
        extra_once = (index.loc[a, "one_time_gpu_minutes"] - index.loc[b, "one_time_gpu_minutes"]) * 60
        saving = index.loc[b, "recurring_gpu_seconds_per_origin"] - index.loc[a, "recurring_gpu_seconds_per_origin"]
        rows.append(
            {
                "a": a,
                "b": b,
                "extra_one_time_gpu_seconds": extra_once,
                "recurring_saving_gpu_seconds_per_origin": saving,
                "break_even_origins": extra_once / saving if saving > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


# --- main


def main() -> None:
    args = parse_args()
    out = Path(args.outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    record: dict = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args)}

    forecasts, skipped = load_strategies()
    record["skipped_sources"] = skipped
    scales = load_mase_scales([args.mase_scales])
    singles = sorted(forecasts["model"].unique())
    full = build_combination_table(forecasts, scales, singles)
    protocol = RollingProtocol()
    evaluated = protocol.evaluation_mask(full)
    record["evaluated_windows"] = {
        d: protocol.evaluation_windows(n) for d, n in full.n_windows.items()
    }

    combined = []
    pools = {**POOLS, **SELECTION_POOLS}
    record["pools"] = {}
    for pool, members in pools.items():
        if not set(members) <= set(singles):
            continue
        record["pools"][pool] = members
        table = subset_table(full, members)
        methods = combiners() if pool in POOLS else [combiners()[1]]
        for combiner in methods:
            result = rolling_combine(table, combiner, protocol)
            combined.append(result.forecasts.assign(model=f"{pool}:{combiner.name}"))
    rows = full.frame.loc[evaluated]
    long_singles = rows.melt(
        id_vars=["dataset", "unique_id", "window", "horizon", "ds", "forecast_horizon", "y_true"],
        value_vars=singles,
        var_name="model",
        value_name="y_pred",
    )[FORECAST_COLUMNS]
    evaluation = pd.concat([long_singles, *combined], ignore_index=True)
    models = sorted(evaluation["model"].unique())
    table = build_combination_table(evaluation, scales, models)

    relative = relative_to_baseline(task_mase(table), args.baseline)
    relative.to_csv(out / "task_scores.csv", index=False)
    aggregate = aggregate_relative(relative)
    aggregate.to_csv(out / "aggregate.csv", index=False)

    main_singles = [m for m in singles if m != args.baseline and not m.startswith("tfm_ctx")]
    regret = regret_table(relative, main_singles)
    regret.to_csv(out / "regret.csv", index=False)
    regret_summary(regret).to_csv(out / "regret_summary.csv", index=False)

    tested = [m for m in models if m not in (args.baseline, *REFERENCES)]
    tests = dm_tests(window_losses(table), tested, REFERENCES)
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

    gains = {
        name: {f"{pool}:{PRIMARY}": 1, member: -1}
        for name, (pool, member) in GAINS.items()
        if pool in record["pools"]
    }
    contrasts = dict(gains)
    for name, (a, b) in GAIN_DIFFERENCES.items():
        if a in gains and b in gains:
            terms: dict[str, float] = {}
            for model, power in [*gains[a].items(), *((m, -p) for m, p in gains[b].items())]:
                terms[model] = terms.get(model, 0) + power
            contrasts[name] = {m: p for m, p in terms.items() if p}
    boot_frames, contrast_frames = [], []
    for resample in RESAMPLE_SCHEMES:
        replicates = bootstrap_replicates(
            table, args.baseline, n_boot=args.n_boot, seed=args.seed, resample=resample,
            block_length=args.block_length,
        )
        boot_frames.append(
            bootstrap_aggregate(
                table, args.baseline, REFERENCES, n_boot=args.n_boot, seed=args.seed,
                resample=resample, block_length=args.block_length,
            ).assign(resample=resample)
        )
        contrast_frames.append(
            bootstrap_contrasts(replicates, table.members, contrasts).assign(resample=resample)
        )
    pd.concat(boot_frames, ignore_index=True).to_csv(out / "bootstrap.csv", index=False)
    pd.concat(contrast_frames, ignore_index=True).to_csv(out / "gain_contrasts.csv", index=False)

    seeds = [m for m in models if m.startswith("tfm_lora_long")]
    scores = aggregate.set_index("model")["geomean_relative_mase"]
    pd.DataFrame(
        {
            "model": seeds,
            "geomean_relative_mase": [scores[m] for m in seeds],
            "ratio_to_tfm_long": [scores[m] / scores["tfm_long"] for m in seeds],
        }
    ).to_csv(out / "seeds.csv", index=False)

    costs = cost_table(singles, args.timings, pd.Series(full.n_windows) + 3)
    costs.to_csv(out / "costs_by_task.csv", index=False)
    accuracy = accuracy_cost(aggregate, costs)
    accuracy.to_csv(out / "accuracy_cost.csv", index=False)
    break_even(
        accuracy,
        [("tfm_lora", "tfm_long"), ("tfm_lora", "tfm_ctx8192"), ("tfm_lora", "tfm_default"),
         ("tfm_lora_long", "tfm_long")],
    ).to_csv(out / "break_even.csv", index=False)

    record["git"] = _git_metadata()
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    print(aggregate.head(25).to_string(index=False))
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
