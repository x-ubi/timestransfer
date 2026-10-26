"""Phase C analyses on the saved member forecasts (R1) and combined forecasts (R2).

C1 diversity, C2 granularity, C3 combination puzzle (learning curves and pools), C4
robustness and inference, C5 failure modes. Statistical members (AutoETS, AutoTheta)
join the C1/C3 pools when their run's forecasts exist. Writes CSV tables and a run
record; nothing here produces new model forecasts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from timestransfer.combination import (
    CombinationTable,
    EqualMean,
    InverseErrorWeights,
    LinearCombiner,
    RollingProtocol,
    build_combination_table,
    load_forecasts,
    load_mase_scales,
    nnls_weights,
    select_best,
)
from scipy.stats import spearmanr

from timestransfer.combination_analysis import (
    aggregate_relative,
    ambiguity_decomposition,
    amplitude_ratios,
    bootstrap_aggregate,
    dm_tests,
    error_correlations,
    horizon_phase_ratios,
    learning_curve,
    regret_summary,
    regret_table,
    relative_to_baseline,
    task_mase,
    window_losses,
    winner_persistence,
)
from timestransfer.metadata import _git_metadata

FM = ["timesfm_2p5", "chronos2", "tabpfn_ts"]
STATS = ["auto_ets", "auto_theta"]
TESTED = [
    "nnls_task",
    "nnls_task_h3",
    "nnls_task_h6",
    "inv_mase_p4",
    "select_task",
    "nnls_series",
    "fforma_lodo",
    "equal_mean",
]
FAILURE_MODELS = [*FM, "equal_mean", "nnls_task", "nnls_task_h3", "nnls_task_h6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasts", default="outputs/combination/forecasts/all_datasets_all_models.parquet"
    )
    parser.add_argument("--mase-scales", default="outputs/combination/metrics/mase_scales.csv")
    parser.add_argument(
        "--stats-forecasts",
        default="outputs/combination_stats/forecasts/all_datasets_all_models.parquet",
        help="AutoETS/AutoTheta run; used if the file exists.",
    )
    parser.add_argument(
        "--combined",
        default="outputs/combination/combiners/fm/forecasts/combined_forecasts.parquet",
        help="R2 combined forecasts (members, baseline and combiners on evaluated windows).",
    )
    parser.add_argument("--baseline", default="seasonal_naive")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outputs-dir", default="outputs/combination/analysis")
    return parser.parse_args()


def curve_combiners() -> list:
    return [
        EqualMean(),
        LinearCombiner("select_task", select_best, level="task"),
        LinearCombiner("inv_mase_p4", InverseErrorWeights(4), level="series"),
        LinearCombiner("nnls_task", nnls_weights, level="task"),
        LinearCombiner("nnls_task_h3", nnls_weights, level="task", n_bins=3),
        LinearCombiner("nnls_series", nnls_weights, level="series"),
    ]


def subset_table(table: CombinationTable, models: list[str]) -> CombinationTable:
    columns = [c for c in table.frame.columns if c not in table.members]
    return CombinationTable(table.frame.loc[:, [*columns, *models]], models)


def main() -> None:
    args = parse_args()
    out = Path(args.outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    record: dict = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args)}

    forecasts = load_forecasts([args.forecasts])
    scales = load_mase_scales([args.mase_scales])
    stats_path = Path(args.stats_forecasts)
    pools = {"fm": FM, "fm_sn": [*FM, args.baseline]}
    if stats_path.exists():
        extra = load_forecasts([stats_path])
        # Acceptance: the stats run's SeasonalNaive must equal R1's.
        check = extra[extra["model"] == args.baseline].merge(
            forecasts[forecasts["model"] == args.baseline],
            on=["dataset", "unique_id", "window", "horizon"],
            suffixes=("", "_r1"),
        )
        record["stats_run_seasonal_naive_max_abs_diff"] = float(
            (check["y_pred"] - check["y_pred_r1"]).abs().max()
        )
        record["stats_run_rows_matched"] = int(len(check))
        forecasts = pd.concat([forecasts, extra[extra["model"].isin(STATS)]], ignore_index=True)
        pools.update({"fm_stats": [*FM, *STATS], "fm_sn_stats": [*FM, args.baseline, *STATS]})
    record["pools"] = pools

    # --- C4 and C2 on the R2 combined forecasts
    combined = load_forecasts([args.combined])
    models = sorted(combined["model"].unique())
    table = build_combination_table(combined, scales, models)
    relative = relative_to_baseline(task_mase(table), args.baseline)
    relative.to_csv(out / "task_scores.csv", index=False)
    aggregate = aggregate_relative(relative)
    aggregate.to_csv(out / "aggregate.csv", index=False)

    regret = regret_table(relative, FM)
    regret.to_csv(out / "regret.csv", index=False)
    regrets = regret_summary(regret)
    regrets.to_csv(out / "regret_summary.csv", index=False)

    losses = window_losses(table)
    tests = dm_tests(losses, TESTED, ["timesfm_2p5", "equal_mean", "best_member"], FM)
    tests.to_csv(out / "dm_tests.csv", index=False)
    dm_summary = (
        tests.assign(
            better=tests["mean_diff"] < 0,
            sig_better=(tests["p_holm"] < 0.05) & (tests["mean_diff"] < 0),
            sig_worse=(tests["p_holm"] < 0.05) & (tests["mean_diff"] > 0),
            sig_better_raw=(tests["p"] < 0.05) & (tests["mean_diff"] < 0),
        )
        .groupby(["model", "reference"])[["better", "sig_better", "sig_worse", "sig_better_raw"]]
        .sum()
        .reset_index()
    )
    dm_summary.to_csv(out / "dm_summary.csv", index=False)

    bootstrap = pd.concat(
        [
            bootstrap_aggregate(
                table,
                args.baseline,
                ["timesfm_2p5", "equal_mean"],
                n_boot=args.n_boot,
                seed=args.seed,
                resample=resample,
            ).assign(resample=resample)
            for resample in ("series", "windows", "both")
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(out / "bootstrap.csv", index=False)

    # --- C1 diversity (member tables on the R2 evaluated windows)
    protocol = RollingProtocol()
    correlation_frames, ambiguity = [], None
    for name, members in pools.items():
        pool_table = build_combination_table(forecasts, scales, members)
        mask = protocol.evaluation_mask(pool_table)
        correlation_frames.append(error_correlations(pool_table, mask).assign(pool=name))
        if name == "fm":
            ambiguity = ambiguity_decomposition(pool_table, mask)
            persistence = winner_persistence(pool_table)
            fm_table = pool_table
    correlations = pd.concat(correlation_frames, ignore_index=True).drop_duplicates(
        ["dataset", "member_a", "member_b"]
    )
    correlations.to_csv(out / "error_correlations.csv", index=False)
    fm_corr = (
        correlations[correlations["member_a"].isin(FM) & correlations["member_b"].isin(FM)]
        .groupby("dataset")["corr"]
        .mean()
    )
    regret_wide = regret.pivot(index="dataset", columns="model", values="regret")
    diversity = ambiguity.set_index("dataset").assign(
        mean_fm_error_corr=fm_corr,
        regret_equal_mean=regret_wide["equal_mean"],
        regret_nnls_task_h3=regret_wide["nnls_task_h3"],
        regret_timesfm=regret_wide["timesfm_2p5"],
    )
    diversity.reset_index().to_csv(out / "diversity_vs_gain.csv", index=False)
    record["spearman_diversity_vs_regret"] = {
        target: dict(
            zip(
                ("rho", "p"),
                map(float, spearmanr(diversity["diversity_ratio"], diversity[target])),
                strict=True,
            )
        )
        for target in ("regret_equal_mean", "regret_nnls_task_h3", "ensemble_vs_best_mse")
    }

    # --- C2 granularity
    persistence.to_csv(out / "winner_persistence.csv", index=False)
    levels = aggregate[
        aggregate["model"].isin(
            [
                "nnls_global",
                "nnls_task",
                "nnls_task_h3",
                "nnls_task_h6",
                "nnls_series",
                "select_task",
                "select_series",
                "select_series_last",
                "fforma_lodo",
                "equal_mean",
                "timesfm_2p5",
            ]
        )
    ]
    levels.to_csv(out / "levels.csv", index=False)

    # --- C3 learning curves: all tasks (L <= 5) and 20-window tasks (L <= 10)
    curve_frames = []
    long_tasks = [d for d, n in fm_table.n_windows.items() if n > 10]
    for name, members in pools.items():
        reference_models = list(dict.fromkeys([*members, args.baseline]))
        reference = forecasts[forecasts["model"].isin(reference_models)]
        for variant, datasets, past in (
            ("all_tasks", None, [1, 2, 3, 5, None]),
            ("long_tasks", long_tasks, [1, 2, 3, 5, 10, None]),
        ):
            pool_forecasts = (
                forecasts if datasets is None else forecasts[forecasts["dataset"].isin(datasets)]
            )
            pool_scales = scales if datasets is None else scales[scales["dataset"].isin(datasets)]
            pool_reference = (
                reference if datasets is None else reference[reference["dataset"].isin(datasets)]
            )
            pool_table = build_combination_table(pool_forecasts, pool_scales, members)
            curve = learning_curve(
                pool_table,
                curve_combiners,
                pool_reference,
                pool_scales,
                past_windows=past,
                baseline=args.baseline,
            )
            curve_frames.append(curve.assign(pool=name, variant=variant))
            print(f"learning curve {name}/{variant} done")
    curves = pd.concat(curve_frames, ignore_index=True)
    curves.to_csv(out / "learning_curves.csv", index=False)

    # --- C5 failure modes
    failure_table = subset_table(table, [*FAILURE_MODELS, args.baseline])
    horizon_phase_ratios(failure_table, args.baseline).to_csv(
        out / "horizon_phase.csv", index=False
    )
    amplitude_ratios(failure_table).to_csv(out / "amplitude.csv", index=False)

    record["git"] = _git_metadata()
    with (out / "analysis_run.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)

    print(aggregate.to_string(index=False))
    print(regrets.to_string(index=False))
    print(dm_summary.to_string(index=False))
    print(json.dumps({k: v for k, v in record.items() if k not in ("args", "git")}, indent=2))
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
