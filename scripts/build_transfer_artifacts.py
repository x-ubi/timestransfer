"""Thesis tables and figures for the transfer-strategy studies (combination, configuration,
fine-tuning, joint comparison); reads saved analysis CSVs and run records only.

Run from the code checkout after the analyses (or `archive_results.py restore`):

    uv run python scripts/build_transfer_artifacts.py --preview

Tables go to <thesis>/tables/transfer, figures to <thesis>/img/transfer. ``--preview``
also compiles every table and figure into one PDF (docs/transfer-results-preview.pdf).
Window sets: the combination and configuration studies score windows 0..K-4 (17 per task,
7 for M4 and ILI); the joint comparison uses the common set 0..K-7 (14 and 4), which
excludes the fine-tuning validation windows. Intervals are 95% task-stratified bootstrap
intervals resampling series and windows.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_thesis_artifacts import NAMES, TASKS, esc, table  # noqa: E402

COMBINATION = ROOT / "outputs/combination/analysis"
CONFIG = ROOT / "outputs/timesfm_config/analysis"
STRATEGY = ROOT / "outputs/strategy_comparison"
FINETUNE = ROOT / "outputs/finetune"
CONTEXTS = [512, 1024, 2048, 4096, 8192, 16256]
MEMBER_NAMES = {"timesfm_2p5": "TimesFM", "chronos2": "Chronos-2", "tabpfn_ts": "TabPFN-TS"}
COMBINER_NAMES = {
    "nnls_task": "NNLS, per task",
    "inv_mase_p4": "Inverse MASE$^4$, per series",
    "nnls_task_h3": "NNLS, per task $\\times$ 3 horizon bins",
    "nnls_task_h6": "NNLS, per task $\\times$ 6 horizon bins",
    "inv_mase_p2_top2": "Inverse MASE$^2$, top 2, per series",
    "nnls_series": "NNLS, per series",
    "select_task": "Best member, per task",
    "select_series": "Best member, per series",
    "inv_mase_p2": "Inverse MASE$^2$, per series",
    "nnls_task_h3_shrink50": "NNLS, per task $\\times$ 3 bins, 50\\% shrinkage",
    "select_series_last": "Best member, per series, last window",
    "nnls_global": "NNLS, global",
    "inv_mase_p1": "Inverse MASE, per series",
    "fforma_lodo": "FFORMA-style meta-learner (LODO)",
    "equal_mean": "Equal mean",
    "median": "Median",
    **MEMBER_NAMES,
}
STRATEGY_NAMES = {
    "tfm_default": "TimesFM, ctx 1024 (default)",
    "tfm_ctx8192": "TimesFM, ctx 8192",
    "tfm_long": "TimesFM, ctx 16256",
    "ctx_select:select_task": "TimesFM, context selected per task",
    "tfm_lora": "TimesFM LoRA, inference ctx 1024",
    "tfm_lora_long": "TimesFM LoRA, inference ctx 16256 (seed 0)",
    "tfm_lora_long_s1": "TimesFM LoRA, inference ctx 16256 (seed 1)",
    "tfm_lora_long_s2": "TimesFM LoRA, inference ctx 16256 (seed 2)",
    "chr_ctx2048": "Chronos-2, ctx 2048",
    "chr_default": "Chronos-2, ctx 8192",
    "chr_lora_c2048": "Chronos-2 LoRA, ctx 2048",
    "chr_lora_long": "Chronos-2 LoRA, inference ctx 8192",
    "tabpfn": "TabPFN-TS",
    "cheap_pair:nnls_task": "TimesFM 1024 + Chronos-2 2048",
    "default_pair:nnls_task": "TimesFM 1024 + Chronos-2 8192",
    "default_trio:nnls_task": "TimesFM 1024 + Chronos-2 8192 + TabPFN-TS",
    "long_pair:nnls_task": "TimesFM 16256 + Chronos-2 8192",
    "lora_long_pair:nnls_task": "TimesFM LoRA@16256 + Chronos-2 8192",
    "both_ft_pair:nnls_task": "TimesFM LoRA@16256 + Chronos-2 LoRA@8192",
}
SHORT = {
    "tfm_default": "TFM",
    "tfm_long": "TFM-L",
    "tfm_ctx8192": "TFM-8k",
    "ctx_select:select_task": "TFM-sel",
    "tfm_lora": "TFM-LoRA",
    "tfm_lora_long": "TFM-L-LoRA",
    "chr_default": "Chr",
    "chr_lora_long": "Chr-LoRA",
    "cheap_pair:nnls_task": "TFM+Chr2k",
    "default_pair:nnls_task": "TFM+Chr",
    "long_pair:nnls_task": "TFM-L+Chr",
    "lora_long_pair:nnls_task": "TFM-L-LoRA+Chr",
}
CI_NOTE = "Intervals: 95\\% task-stratified bootstrap (2,000 replicates) resampling series and windows."
COMMON_NOTE = (
    "Common evaluation set: windows 0--13 (0--3 for M4 and ILI); the three oldest windows, used "
    "for fine-tuning early stopping, are excluded from all fitting and scoring."
)
STUDY_NOTE = "Evaluated windows 0--16 (0--6 for M4 and ILI); weights and selections use earlier windows only."


def pct(ratio: float, places: int = 1) -> str:
    """(ratio - 1) in percent with a sign."""
    value = round(100 * (ratio - 1), places)
    return f"{0:.{places}f}" if value == 0 else f"${value:+.{places}f}$"


def pct_ci(estimate: float, low: float, high: float) -> str:
    return f"{pct(estimate)} [{pct(low)}, {pct(high)}]"


def bold(text: str, condition: bool) -> str:
    if not condition:
        return text
    if text.startswith("$"):  # \textbf does not reach math mode
        return r"$\mathbf{" + text.strip("$") + "}$"
    return r"\textbf{" + text + "}"


def footnotesize_table(path, caption, label, header, rows, spec=None, sep="3pt"):
    table(path, caption, label, header, rows, spec=spec, blue=False, size=r"\footnotesize", sep=sep)


def geomean(values: pd.Series) -> float:
    return float(np.exp(np.log(values.astype(float)).mean()))


# --- tables


def table_combiners(out: Path) -> None:
    aggregate = pd.read_csv(COMBINATION / "aggregate.csv").set_index("model")["geomean_relative_mase"]
    regret = pd.read_csv(COMBINATION / "regret_summary.csv").set_index("model")
    boot = pd.read_csv(COMBINATION / "bootstrap.csv")
    boot = boot[(boot["resample"] == "both") & (boot["comparison"] == "ratio_to_timesfm_2p5")].set_index("model")
    rows = []
    for model in aggregate.drop("seasonal_naive").index:
        ci = (
            pct_ci(*boot.loc[model, ["estimate", "ci_low", "ci_high"]])
            if model in boot.index
            else "--"
        )
        rows.append(
            [
                COMBINER_NAMES.get(model, esc(model)),
                f"{aggregate[model]:.3f}",
                ci,
                f"{regret.loc[model, 'mean_regret']:.3f}",
                f"{regret.loc[model, 'max_regret']:.3f}",
            ]
        )
    footnotesize_table(
        out / "combiners.tex",
        "Combination of TimesFM (ctx 1024), Chronos-2 (ctx 8192) and TabPFN-TS: aggregate relative MASE "
        "($\\downarrow$), change vs TimesFM with 95\\% interval, and regret vs the ex-post best member per task. "
        + STUDY_NOTE + " " + CI_NOTE,
        "tab:transfer-combiners",
        ["Method", "Score", "\\% vs TimesFM [CI]", "Mean regret", "Max regret"],
        rows,
        spec="lrrrr",
    )


def table_redundancy(out: Path) -> None:
    shared = pd.read_csv(COMBINATION / "shared_error.csv")
    shared = shared[shared["pool"] == "fm"].set_index("dataset")
    bound = pd.read_csv(COMBINATION / "combination_bound.csv")
    stats_bound = bound[bound["pool"] == "fm_sn_stats"].set_index("dataset")
    bound = bound[bound["pool"] == "fm"].set_index("dataset")
    pairs = pd.read_csv(COMBINATION / "pairwise_theory.csv")
    pairs = pairs[
        (pairs["pool"] == "fm")
        & pairs[["better_member", "worse_member"]].isin(["timesfm_2p5", "chronos2"]).all(axis=1)
    ].set_index("dataset")
    rows = []
    for task in TASKS:
        rows.append(
            [
                NAMES[task],
                f"{shared.loc[task, 'shared_share']:.2f}",
                f"{shared.loc[task, 'first_pc_share']:.2f}",
                MEMBER_NAMES[bound.loc[task, "best_member"]],
                pct(bound.loc[task, "simplex_vs_best"]),
                pct(bound.loc[task, "equal_vs_best"]),
                pct(stats_bound.loc[task, "simplex_vs_best"]) if task in stats_bound.index else "--",
                f"{pairs.loc[task, 'rho']:.2f}",
                f"{pairs.loc[task, 'accuracy_ratio_k']:.2f}",
                pct(pairs.loc[task, "optimal_vs_better"]),
            ]
        )
    rows.append(
        [
            "Median",
            f"{shared['shared_share'].median():.2f}",
            f"{shared['first_pc_share'].median():.2f}",
            "",
            pct(bound["simplex_vs_best"].median()),
            pct(bound["equal_vs_best"].median()),
            pct(stats_bound["simplex_vs_best"].median()),
            f"{pairs['rho'].median():.2f}",
            f"{pairs['accuracy_ratio_k'].median():.2f}",
            pct(pairs["optimal_vs_better"].median()),
        ]
    )
    footnotesize_table(
        out / "redundancy.tex",
        "Error redundancy of the zero-shot foundation models (squared scaled errors). Shared: share of member "
        "error explained by the members' mean error; 1st PC: share of the first principal component. Change in "
        "MSE vs the best member for ex-post optimal simplex weights (ideal), equal weights, and ideal weights with "
        "SeasonalNaive, AutoETS and AutoTheta added (+stats). TimesFM--Chronos-2 pair: uncentred error correlation "
        "$\\rho$, accuracy ratio $k=\\sigma_1/\\sigma_2$ and the Bates--Granger optimum $(1-\\rho^2)/(1+k^2-2\\rho k)$, "
        "which is below one only if $\\rho<k$. " + STUDY_NOTE,
        "tab:transfer-redundancy",
        ["Task", "Shared", "1st PC", "Best", "Ideal", "Equal", "Ideal +stats", "$\\rho$", "$k$", "Pair opt."],
        rows,
        spec="lrrlrrrrrr",
    )


def table_residuals(out: Path) -> None:
    corr = pd.read_csv(COMBINATION / "residual_correlations.csv")
    summary = pd.read_csv(COMBINATION / "residual_summary.csv").pivot(index="dataset", columns="model", values="bias")
    pairs = [("timesfm_2p5", "chronos2"), ("timesfm_2p5", "tabpfn_ts"), ("chronos2", "tabpfn_ts")]
    rows = []
    for task in TASKS:
        part = corr[corr["dataset"] == task].set_index(["member_a", "member_b"])
        cells = []
        for pair in pairs:
            cells += [f"{part.loc[pair, c]:.2f}".replace("-", "$-$") for c in ("pearson", "series_median")]
        rows.append([NAMES[task], *cells, *[f"{summary.loc[task, m]:+.2f}".replace("+", "$+$").replace("-", "$-$")
                                          for m in ("timesfm_2p5", "chronos2", "tabpfn_ts")]])
    medians = []
    for pair in pairs:
        part = corr.set_index(["member_a", "member_b"]).loc[pair]
        medians += [f"{part['pearson'].median():.2f}", f"{part['series_median'].median():.2f}"]
    rows.append(["Median", *medians, "", "", ""])
    footnotesize_table(
        out / "residual_correlations.tex",
        "Residuals $e=(\\hat y-y)/s$ of the zero-shot models at every forecast point ($s$: the series' MASE scale). "
        "Pairwise Pearson correlation of residuals, pooled over the task's points (all) and the median over series "
        "of within-series correlations (ser.); mean residual (bias; positive: over-forecasting). " + STUDY_NOTE,
        "tab:transfer-residuals",
        ["Task", "TFM--Chr all", "ser.", "TFM--TabPFN all", "ser.", "Chr--TabPFN all", "ser.",
         "Bias TFM", "Bias Chr", "Bias TabPFN"],
        rows,
        spec="l" + "r" * 9,
        sep="2.5pt",
    )


def table_context(out: Path) -> None:
    profile = pd.read_csv(CONFIG / "context_profile.csv").set_index("dataset")
    scores = pd.read_csv(CONFIG / "task_scores.csv").pivot(index="dataset", columns="model", values="relative_mase")
    flip_on = [f"timesfm_2p5_ctx{c}" for c in CONTEXTS]
    rows = []
    for task in TASKS:
        values = profile.loc[task, flip_on]
        best = values.min()
        selected = scores.loc[task, "tfm_cfg:select_task"] / scores.loc[task, "timesfm_2p5_ctx1024"]
        rows.append(
            [
                NAMES[task],
                f"{scores.loc[task, 'timesfm_2p5_ctx1024']:.3f}",
                *[bold(pct(v), v == best and round(100 * (v - 1), 1) < 0) if c != 1024 else "0"
                  for c, v in zip(CONTEXTS, values)],
                pct(profile.loc[task, "timesfm_2p5_ctx16256_noflip"]),
                pct(selected),
                pct(profile.loc[task, "chronos2_ctx8192"]),
            ]
        )
    aggregate = {m: geomean(scores[m]) for m in [*flip_on, "timesfm_2p5_ctx16256_noflip", "tfm_cfg:select_task", "chronos2_ctx8192"]}
    base = aggregate["timesfm_2p5_ctx1024"]
    rows.append(
        [
            "Aggregate",
            f"{base:.3f}",
            *[pct(aggregate[m] / base) if m != "timesfm_2p5_ctx1024" else "0" for m in flip_on],
            pct(aggregate["timesfm_2p5_ctx16256_noflip"] / base),
            pct(aggregate["tfm_cfg:select_task"] / base),
            pct(aggregate["chronos2_ctx8192"] / base),
        ]
    )
    footnotesize_table(
        out / "context_grid.tex",
        "TimesFM context length: change in relative MASE vs ctx 1024 (\\%, $\\downarrow$), flip invariance on. "
        "First column: relative MASE at ctx 1024. No flip: ctx 16256 without flip invariance. Selected: context "
        "and flip setting chosen per task from earlier windows (12 variants). Chronos-2: ctx 8192, for reference. "
        "Bold: best context per task. " + STUDY_NOTE,
        "tab:transfer-context",
        ["Task", "1024 (rel.)", "512", "1024", "2048", "4096", "8192", "16256", "No flip", "Selected", "Chronos-2"],
        rows,
        spec="lrrrrrrrrrr",
    )


def table_flip(out: Path) -> None:
    profile = pd.read_csv(CONFIG / "context_profile.csv").set_index("dataset")
    rows = []
    for task in TASKS:
        rows.append(
            [NAMES[task]]
            + [
                pct(profile.loc[task, f"timesfm_2p5_ctx{c}_noflip"] / profile.loc[task, f"timesfm_2p5_ctx{c}"])
                for c in CONTEXTS
            ]
        )
    footnotesize_table(
        out / "flip_grid.tex",
        "Effect of turning flip invariance off: change in TimesFM relative MASE (\\%) vs flip on at the same "
        "context. " + STUDY_NOTE,
        "tab:transfer-flip",
        ["Task", *[str(c) for c in CONTEXTS]],
        rows,
    )


def table_chronos_context(out: Path) -> None:
    scores = pd.read_csv(CONFIG / "task_scores.csv").pivot(index="dataset", columns="model", values="relative_mase")
    columns = [f"chronos2_ctx{c}" for c in (1024, 2048, 4096, 8192)]
    rows = []
    for task in TASKS:
        values = scores.loc[task, columns]
        rows.append([NAMES[task], *[bold(f"{v:.3f}", v == values.min()) for v in values]])
    rows.append(["Aggregate", *[f"{geomean(scores[c]):.3f}" for c in columns]])
    footnotesize_table(
        out / "chronos_context.tex",
        "Chronos-2 relative MASE ($\\downarrow$) by context length. " + STUDY_NOTE,
        "tab:transfer-chronos-context",
        ["Task", "1024", "2048", "4096", "8192"],
        rows,
    )


def _timesfm_kept_zero_shot(pattern: str) -> set[str]:
    path = sorted(glob.glob(str(FINETUNE / pattern)))[-1]
    return {t["dataset"] for t in json.loads(Path(path).read_text())["tasks"] if t.get("best_step") == 0}


def _chronos_kept_zero_shot() -> set[str]:
    path = sorted(glob.glob(str(FINETUNE / "chronos2_lora_c2048/run_*.json")))[-1]
    return {t["dataset"] for t in json.loads(Path(path).read_text())["tasks"] if t.get("selected") == "zero_shot"}


def table_finetune(out: Path) -> None:
    scores = pd.read_csv(STRATEGY / "task_scores.csv").pivot(index="dataset", columns="model", values="relative_mase")
    kept = {
        "tfm_lora_long": _timesfm_kept_zero_shot("timesfm_lora_v2/run_*.json"),
        "tfm_lora_long_s1": _timesfm_kept_zero_shot("timesfm_lora_seed1/run_*.json"),
        "tfm_lora_long_s2": _timesfm_kept_zero_shot("timesfm_lora_seed2/run_*.json"),
        "chr": _chronos_kept_zero_shot(),
    }
    mark = "$^{\\ast}$"
    rows = []
    for task in TASKS:
        row = [NAMES[task], f"{scores.loc[task, 'tfm_long']:.3f}"]
        row.append(pct(scores.loc[task, "tfm_lora"] / scores.loc[task, "tfm_default"]) + (mark if task in kept["tfm_lora_long"] else ""))
        for model in ("tfm_lora_long", "tfm_lora_long_s1", "tfm_lora_long_s2"):
            row.append(pct(scores.loc[task, model] / scores.loc[task, "tfm_long"]) + (mark if task in kept[model] else ""))
        row.append(f"{scores.loc[task, 'chr_default']:.3f}")
        for model, reference in (("chr_lora_c2048", "chr_ctx2048"), ("chr_lora_long", "chr_default")):
            row.append(pct(scores.loc[task, model] / scores.loc[task, reference]) + (mark if task in kept["chr"] else ""))
        rows.append(row)
    agg = {m: geomean(scores[m]) for m in scores.columns}
    rows.append(
        [
            "Aggregate",
            f"{agg['tfm_long']:.3f}",
            pct(agg["tfm_lora"] / agg["tfm_default"]),
            *[pct(agg[m] / agg["tfm_long"]) for m in ("tfm_lora_long", "tfm_lora_long_s1", "tfm_lora_long_s2")],
            f"{agg['chr_default']:.3f}",
            pct(agg["chr_lora_c2048"] / agg["chr_ctx2048"]),
            pct(agg["chr_lora_long"] / agg["chr_default"]),
        ]
    )
    footnotesize_table(
        out / "finetune.tex",
        "LoRA fine-tuning per task: change in relative MASE (\\%, $\\downarrow$) vs the zero-shot model at the "
        "same inference context. TimesFM: LoRA trained at ctx 1024 (r=4, all linear layers), evaluated at ctx 1024 "
        "(vs zero-shot 1024) and at ctx 16256 (vs zero-shot 16256) for seeds 0--2. Chronos-2: library-default LoRA "
        "trained at ctx 2048, evaluated at 2048 and 8192. Score columns: zero-shot relative MASE. "
        "$^{\\ast}$: validation selection kept the zero-shot model. " + COMMON_NOTE,
        "tab:transfer-finetune",
        ["Task", "TFM 16256", "LoRA@1024", "LoRA@16256 s0", "s1", "s2", "Chr 8192", "LoRA@2048", "LoRA@8192"],
        rows,
        spec="lrrrrrrrr",
    )


def _boot(resample: str = "both") -> pd.DataFrame:
    boot = pd.read_csv(STRATEGY / "bootstrap.csv")
    return boot[boot["resample"] == resample]


def table_final(out: Path) -> None:
    accuracy = pd.read_csv(STRATEGY / "accuracy_cost.csv").set_index("strategy")
    boot = _boot()
    def ci(model, reference):
        row = boot[(boot["model"] == model) & (boot["comparison"] == f"ratio_to_{reference}")]
        return pct_ci(*row[["estimate", "ci_low", "ci_high"]].iloc[0]) if len(row) else "--"
    order = [
        "tfm_default", "tfm_ctx8192", "tfm_long", "ctx_select:select_task", "tfm_lora", "tfm_lora_long",
        "tfm_lora_long_s1", "tfm_lora_long_s2", "chr_default", "chr_lora_long", "tabpfn",
        "cheap_pair:nnls_task", "default_pair:nnls_task", "default_trio:nnls_task", "long_pair:nnls_task",
        "lora_long_pair:nnls_task", "both_ft_pair:nnls_task",
    ]
    aggregate = pd.read_csv(STRATEGY / "aggregate.csv").set_index("model")["geomean_relative_mase"]
    for seed in ("tfm_lora_long_s1", "tfm_lora_long_s2"):  # same cost as seed 0
        if seed in aggregate.index and seed not in accuracy.index:
            accuracy.loc[seed] = accuracy.loc["tfm_lora_long"]
            accuracy.loc[seed, "geomean_relative_mase"] = aggregate[seed]
    rows = []
    for model in order:
        if model not in accuracy.index:
            continue
        a = accuracy.loc[model]
        rows.append(
            [
                STRATEGY_NAMES[model],
                f"{a['geomean_relative_mase']:.3f}",
                ci(model, "tfm_default") if model != "tfm_default" else "--",
                ci(model, "tfm_long") if model != "tfm_long" else "--",
                f"{a['recurring_gpu_seconds_per_origin']:.1f}",
                f"{a['one_time_gpu_minutes']:.0f}" if a["one_time_gpu_minutes"] > 0 else "--",
            ]
        )
    footnotesize_table(
        out / "final.tex",
        "Joint comparison of transfer strategies: aggregate relative MASE ($\\downarrow$), change vs default TimesFM "
        "and vs TimesFM at ctx 16256 (\\% [95\\% CI]), GPU seconds per forecast origin summed over the 11 tasks, and "
        "one-time training GPU minutes (RTX A5000; TimesFM context capped at the longest history; adapters merged; "
        "TabPFN-TS from run logs). Combinations: task-level NNLS weights fitted on earlier windows; their cost is the "
        "sum of the members'. " + COMMON_NOTE + " " + CI_NOTE,
        "tab:transfer-final",
        ["Strategy", "Score", "\\% vs default [CI]", "\\% vs ctx 16256 [CI]", "GPU s/origin", "Train min"],
        rows,
        spec="lrrrrr",
    )


def table_contrasts(out: Path) -> None:
    contrasts = pd.read_csv(STRATEGY / "gain_contrasts.csv")
    names = {
        "gain_default": "Combination gain, TimesFM default + Chronos-2",
        "gain_long": "Combination gain, TimesFM 16256 + Chronos-2",
        "gain_lora_long": "Combination gain, TimesFM LoRA@16256 + Chronos-2",
        "gain_both_ft": "Combination gain, TimesFM LoRA@16256 + Chronos-2 LoRA",
        "gain_long_vs_default": "Difference: 16256 vs default",
        "gain_lora_long_vs_long": "Difference: LoRA@16256 vs 16256",
        "gain_lora_long_vs_default": "Difference: LoRA@16256 vs default",
        "gain_both_ft_vs_lora_long": "Difference: Chronos-2 LoRA added",
    }
    rows = []
    for key, label in names.items():
        both = contrasts[(contrasts["contrast"] == key) & (contrasts["resample"] == "both")]
        blocks = contrasts[(contrasts["contrast"] == key) & (contrasts["resample"] == "series_blocks")]
        if both.empty:
            continue
        rows.append(
            [
                label,
                pct_ci(*both[["estimate", "ci_low", "ci_high"]].iloc[0], ),
                pct_ci(*blocks[["estimate", "ci_low", "ci_high"]].iloc[0]),
            ]
        )
    footnotesize_table(
        out / "contrasts.tex",
        "Combination gain (combination vs its TimesFM member, \\%) and differences between gains (ratio of ratios, "
        "\\%; positive: the gain is smaller than in the reference). Intervals: series and windows resampled, or "
        "series and circular blocks of three consecutive windows. " + COMMON_NOTE,
        "tab:transfer-contrasts",
        ["Quantity", "\\% [CI], series $\\times$ windows", "\\% [CI], series $\\times$ window blocks"],
        rows,
        spec="lrr",
    )


def table_final_tasks(out: Path) -> None:
    scores = pd.read_csv(STRATEGY / "task_scores.csv").pivot(index="dataset", columns="model", values="relative_mase")
    columns = [c for c in SHORT if c != "tfm_ctx8192"]
    rows = []
    for task in TASKS:
        values = scores.loc[task, columns]
        rows.append([NAMES[task], *[bold(f"{v:.3f}", v == values.min()) for v in values]])
    rows.append(["Aggregate", *[f"{geomean(scores[c]):.3f}" for c in columns]])
    legend = "; ".join(f"{SHORT[c]}: {STRATEGY_NAMES[c]}" for c in columns)
    footnotesize_table(
        out / "final_tasks.tex",
        "Relative MASE ($\\downarrow$) per task and strategy. Bold: best per task. " + legend + ". " + COMMON_NOTE,
        "tab:transfer-final-tasks",
        ["Task", *columns_header(columns)],
        rows,
        spec="l" + "r" * len(columns),
        sep="2pt",
    )


def columns_header(columns: list[str]) -> list[str]:
    return [r"\rotatebox{75}{" + SHORT[c] + "}" for c in columns]


def table_tabpfn(out: Path) -> None:
    aggregate = pd.read_csv(STRATEGY / "aggregate.csv").set_index("model")["geomean_relative_mase"]
    contrasts = pd.read_csv(STRATEGY / "gain_contrasts.csv")
    labels = [("default", "TimesFM ctx 1024 + Chronos-2"), ("long", "TimesFM ctx 16256 + Chronos-2"),
              ("lora_long", "TimesFM LoRA@16256 + Chronos-2")]
    rows = []
    for key, label in labels:
        trio, pair = f"{key}_trio:nnls_task", f"{key}_pair:nnls_task"
        cells = []
        for resample in ("both", "series_blocks"):
            row = contrasts[(contrasts["contrast"] == f"tabpfn_added_{key}") & (contrasts["resample"] == resample)]
            cells.append(pct_ci(*row[["estimate", "ci_low", "ci_high"]].iloc[0]))
        rows.append([label, f"{aggregate[pair]:.3f}", f"{aggregate[trio]:.3f}", *cells])
    footnotesize_table(
        out / "tabpfn_added.tex",
        "Adding TabPFN-TS as a third member: aggregate relative MASE ($\\downarrow$) of the TimesFM + Chronos-2 "
        "pair and of the pair plus TabPFN-TS (task-level NNLS weights), and the change from adding TabPFN-TS "
        "(\\% [95\\% CI]). " + COMMON_NOTE,
        "tab:transfer-tabpfn",
        ["Combination", "Pair", "+ TabPFN-TS", "Change, series $\\times$ windows", "Change, series $\\times$ blocks"],
        rows,
        spec="lrrrr",
    )


def table_dm(out: Path) -> None:
    summary = pd.read_csv(STRATEGY / "dm_summary.csv")
    models = [m for m in SHORT if m not in ("tfm_default", "tfm_long", "tfm_ctx8192")]
    rows = []
    for model in models:
        cells = [STRATEGY_NAMES[model]]
        for reference in ("tfm_default", "tfm_long"):
            row = summary[(summary["model"] == model) & (summary["reference"] == reference)]
            cells.append(
                "--" if row.empty else f"{int(row['better'].iloc[0])} / {int(row['sig_better'].iloc[0])} / {int(row['sig_worse'].iloc[0])}"
            )
        rows.append(cells)
    footnotesize_table(
        out / "dm_summary.tex",
        "Per-task Diebold--Mariano tests on window-level MASE (Newey--West variance, HLN correction, Holm across "
        "the 11 tasks): tasks with lower mean loss / significantly lower / significantly higher at 5\\%. "
        "TimesFM ctx 16256 is a reference and is not tested against the default here. "
        + COMMON_NOTE,
        "tab:transfer-dm",
        ["Strategy", "vs TimesFM default", "vs TimesFM 16256"],
        rows,
        spec="lrr",
    )


# --- figures


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def figure_correlation_plane(out: Path) -> None:
    pairs = pd.read_csv(COMBINATION / "pairwise_theory.csv")
    pairs = pairs[pairs["pool"] == "fm"]
    styles = {
        frozenset({"timesfm_2p5", "chronos2"}): ("#0072B2", "TimesFM--Chronos-2"),
        frozenset({"timesfm_2p5", "tabpfn_ts"}): ("#D55E00", "TimesFM--TabPFN-TS"),
        frozenset({"chronos2", "tabpfn_ts"}): ("#009E73", "Chronos-2--TabPFN-TS"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, (klim, rlim, levels) in zip(
        axes, [((0.3, 1.0), (-0.1, 1.0), [0.80, 0.90, 0.95]), ((0.78, 1.0), (0.70, 1.0), [0.95, 0.97, 0.98, 0.99, 0.995])]
    ):
        k, rho = np.meshgrid(np.linspace(*klim, 300), np.linspace(*rlim, 300))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(rho < k, (1 - rho**2) / (1 + k**2 - 2 * rho * k), 1.0)
        contour = ax.contour(k, rho, ratio, levels=levels, colors="#999999", linewidths=0.7)
        ax.clabel(contour, fmt=lambda v: f"{100 * (v - 1):+.1f}%", fontsize=6.5)
        ax.plot(klim, klim, color="black", lw=0.8, ls="--", label="$\\rho = k$")
        for key, (color, label) in styles.items():
            part = pairs[pairs[["better_member", "worse_member"]].apply(frozenset, axis=1) == key]
            ax.scatter(part["accuracy_ratio_k"], part["rho"], s=20, color=color, label=label, zorder=3)
        ax.set_xlim(*klim)
        ax.set_ylim(*rlim)
        ax.set_xlabel("Accuracy ratio $k = \\sigma_{better} / \\sigma_{worse}$")
    ili = pairs[pairs["dataset"] == "ili_h48"]
    for _, row in ili[ili["accuracy_ratio_k"] < 0.6].iterrows():
        axes[0].annotate("ILI", (row["accuracy_ratio_k"], row["rho"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[0].add_patch(plt.Rectangle((0.78, 0.70), 0.22, 0.30, fill=False, lw=0.6, ls=":", color="black"))
    axes[0].set_ylabel("Error correlation $\\rho$ (uncentred)")
    axes[0].set_title("All task pairs", fontsize=9)
    axes[1].set_title("Zoom (dotted box); contours: optimal MSE change", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, out, "correlation_plane")


def figure_residual_scatter(out: Path) -> None:
    sample = pd.read_csv(COMBINATION / "residual_sample.csv")
    corr = pd.read_csv(COMBINATION / "residual_correlations.csv")
    corr = corr[(corr["member_a"] == "timesfm_2p5") & (corr["member_b"] == "chronos2")].set_index("dataset")["pearson"]
    tasks = ["ecl_h48", "traffic_h48", "weather_h48", "ett_m1_h48", "m4_hourly", "ili_h48"]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.4))
    for ax, task in zip(axes.flat, tasks):
        part = sample[sample["dataset"] == task]
        lim = np.quantile(np.abs(part[["timesfm_2p5", "chronos2"]].to_numpy()), 0.99)
        ax.scatter(part["timesfm_2p5"], part["chronos2"], s=2, alpha=0.25, color="#0072B2", rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.7, ls="--")
        ax.axhline(0, color="#cccccc", lw=0.5)
        ax.axvline(0, color="#cccccc", lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{NAMES[task]}  (r = {corr[task]:.2f})", fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.supxlabel("TimesFM residual (MASE-scaled)", fontsize=8.5)
    fig.supylabel("Chronos-2 residual (MASE-scaled)", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "residual_scatter")


def figure_accuracy_cost(out: Path) -> None:
    accuracy = pd.read_csv(STRATEGY / "accuracy_cost.csv").set_index("strategy")
    offsets = {
        "tfm_default": (5, 2), "tfm_lora": (5, -9), "cheap_pair:nnls_task": (5, 3), "default_pair:nnls_task": (5, 3),
        "tfm_ctx8192": (5, 3), "tfm_long": (5, 3), "tfm_lora_long": (5, -9), "long_pair:nnls_task": (-62, 3),
        "lora_long_pair:nnls_task": (5, -9), "ctx_select:select_task": (-30, 5),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model, offset in offsets.items():
        a = accuracy.loc[model]
        trained = a["one_time_gpu_minutes"] > 0
        combo = ":" in model and not model.startswith("ctx_select")
        color = "#D55E00" if combo else "#0072B2"
        ax.scatter(a["recurring_gpu_seconds_per_origin"], a["geomean_relative_mase"], s=36,
                   facecolors="none" if trained else color, edgecolors=color, marker="s" if combo else "o", zorder=3)
        ax.annotate(SHORT[model], (a["recurring_gpu_seconds_per_origin"], a["geomean_relative_mase"]),
                    fontsize=7, xytext=offset, textcoords="offset points")
    frontier = accuracy.loc[list(offsets)].sort_values("recurring_gpu_seconds_per_origin")
    best = np.minimum.accumulate(frontier["geomean_relative_mase"].to_numpy())
    ax.step(frontier["recurring_gpu_seconds_per_origin"], best, where="post", color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("GPU seconds per forecast origin, 11 tasks (log)")
    ax.set_ylabel("Aggregate relative MASE")
    ax.set_title("Accuracy vs recurring cost (circles: TimesFM, squares: combinations; hollow: needs LoRA training)", fontsize=8)
    save(fig, out, "accuracy_cost")


def figure_context_curves(out: Path) -> None:
    profile = pd.read_csv(CONFIG / "context_profile.csv").set_index("dataset")
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 5.6), sharex=True)
    for ax, task in zip(axes.flat, TASKS):
        on = [100 * (profile.loc[task, f"timesfm_2p5_ctx{c}"] - 1) for c in CONTEXTS]
        off = [100 * (profile.loc[task, f"timesfm_2p5_ctx{c}_noflip"] - 1) for c in CONTEXTS]
        ax.plot(CONTEXTS, on, marker="o", ms=3, color="#0072B2", label="flip on")
        ax.plot(CONTEXTS, off, marker="o", ms=3, color="#0072B2", ls="--", alpha=0.6, label="flip off")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_title(NAMES[task], fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xticks(CONTEXTS, [str(c) for c in CONTEXTS], rotation=60)
    fig.supylabel("% change in relative MASE vs ctx 1024", fontsize=8)
    fig.supxlabel("TimesFM context length", fontsize=8)
    fig.tight_layout()
    save(fig, out, "context_curves")


def figure_learning_curves(out: Path) -> None:
    curves = pd.read_csv(COMBINATION / "learning_curves.csv")
    curves = curves[curves["variant"] == "all_tasks"].copy()
    order = ["1", "2", "3", "5", "all"]
    curves = curves[curves["past_windows"].astype(str).isin(order)]
    curves["x"] = curves["past_windows"].astype(str).map({v: i for i, v in enumerate(order)})
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    fm = curves[curves["pool"] == "fm"]
    for method, color in [("nnls_task", "#D55E00"), ("select_task", "#0072B2"), ("inv_mase_p4", "#009E73"),
                          ("nnls_series", "#CC79A7"), ("nnls_task_h3", "#E69F00")]:
        part = fm[fm["method"] == method].sort_values("x")
        left.plot(part["x"], part["geomean_relative_mase"], marker="o", ms=3, color=color, label=method)
    member = pd.read_csv(COMBINATION / "aggregate.csv").set_index("model")["geomean_relative_mase"]
    left.axhline(member["timesfm_2p5"], color="#555555", lw=0.8, ls=":", label="TimesFM alone")
    left.set_title("Foundation-model pool", fontsize=9)
    for pool, color, label in [("fm", "#D55E00", "FMs"), ("fm_stats", "#0072B2", "FMs + ETS/Theta"),
                               ("fm_sn_stats", "#009E73", "FMs + SNaive + ETS/Theta")]:
        part = curves[(curves["pool"] == pool) & (curves["method"] == "nnls_task")].sort_values("x")
        right.plot(part["x"], part["geomean_relative_mase"], marker="o", ms=3, color=color, label=label)
    right.set_title("NNLS per task, by pool", fontsize=9)
    for ax in (left, right):
        ax.set_xticks(range(len(order)), order)
        ax.set_xlabel("Past windows used for fitting")
        ax.legend(fontsize=6.5)
    left.set_ylabel("Aggregate relative MASE")
    fig.tight_layout()
    save(fig, out, "learning_curves")


def figure_gain_shrinkage(out: Path) -> None:
    contrasts = pd.read_csv(STRATEGY / "gain_contrasts.csv")
    contrasts = contrasts[contrasts["resample"] == "both"].set_index("contrast")
    keys = ["gain_default", "gain_long", "gain_lora_long", "gain_both_ft"]
    labels = ["default", "ctx 16256", "LoRA@16256", "LoRA@16256 +\nChronos-2 LoRA"]
    keys, labels = zip(*[(k, l) for k, l in zip(keys, labels) if k in contrasts.index])
    est = 100 * (contrasts.loc[list(keys), "estimate"] - 1)
    low = 100 * (contrasts.loc[list(keys), "ci_low"] - 1)
    high = 100 * (contrasts.loc[list(keys), "ci_high"] - 1)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(keys))
    ax.bar(x, est, color="#D55E00", alpha=0.8, width=0.6)
    ax.errorbar(x, est, yerr=[est - low, high - est], fmt="none", ecolor="black", capsize=3, lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_xlabel("TimesFM member of the TimesFM + Chronos-2 combination")
    ax.set_ylabel("% change vs the TimesFM member")
    ax.set_title("Combination gain shrinks as the member improves", fontsize=9)
    save(fig, out, "gain_shrinkage")


def figure_task_heatmap(out: Path) -> None:
    scores = pd.read_csv(STRATEGY / "task_scores.csv").pivot(index="dataset", columns="model", values="relative_mase")
    columns = [c for c in SHORT if c not in ("tfm_default", "tfm_ctx8192")]
    change = 100 * (scores.loc[TASKS, columns].div(scores.loc[TASKS, "tfm_default"], axis=0) - 1)
    change.loc["aggregate"] = [100 * (geomean(scores[c]) / geomean(scores["tfm_default"]) - 1) for c in columns]
    limit = 15
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    image = ax.imshow(change.clip(-limit, limit).to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(change.shape[0]):
        for j in range(change.shape[1]):
            value = change.iat[i, j]
            label = "0.0" if round(value, 1) == 0 else f"{value:+.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(value) > 0.7 * limit else "black")
    ax.set_xticks(range(len(columns)), [SHORT[c] for c in columns], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(change.shape[0]), [NAMES.get(t, "Aggregate") for t in change.index], fontsize=7.5)
    fig.colorbar(image, ax=ax, label="% change vs TimesFM default (clipped at ±15)")
    ax.set_title("Per-task change in relative MASE vs TimesFM ctx 1024", fontsize=9)
    save(fig, out, "task_heatmap")


def figure_lora_seeds(out: Path) -> None:
    scores = pd.read_csv(STRATEGY / "task_scores.csv").pivot(index="dataset", columns="model", values="relative_mase")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    y = np.arange(len(TASKS))
    for model, color, offset, label in [("tfm_lora_long", "#0072B2", -0.15, "TimesFM seed 0"),
                                        ("tfm_lora_long_s1", "#56B4E9", 0.0, "TimesFM seed 1"),
                                        ("tfm_lora_long_s2", "#003f63", 0.15, "TimesFM seed 2")]:
        ax.scatter(100 * (scores.loc[TASKS, model] / scores.loc[TASKS, "tfm_long"] - 1), y + offset, s=14, color=color, label=label)
    ax.scatter(100 * (scores.loc[TASKS, "chr_lora_long"] / scores.loc[TASKS, "chr_default"] - 1), y + 0.3, s=14,
               color="#009E73", marker="D", label="Chronos-2 (ctx 8192)")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, [NAMES[t] for t in TASKS], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("% change vs zero-shot at the same context (TimesFM: ctx 16256)")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("LoRA effect per task", fontsize=9)
    save(fig, out, "lora_seeds")


# --- preview


def preview(tables: Path, figures: Path, target: Path) -> None:
    build = ROOT / "outputs/thesis_preview"
    build.mkdir(parents=True, exist_ok=True)
    body = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=2cm]{geometry}",
        r"\usepackage{graphicx,xcolor,float}",
        r"\newcommand{\revisedstyle}{}\newcommand{\revised}[1]{#1}",
        r"\title{Transfer-strategy results: thesis tables and figures}\date{\today}",
        r"\begin{document}\maketitle",
    ]
    for name in ["combiners", "residual_correlations", "redundancy", "context_grid", "flip_grid", "chronos_context", "finetune",
                 "final", "contrasts", "tabpfn_added", "final_tasks", "dm_summary"]:
        body.append(r"\input{" + str(tables / name) + "}")
        body.append(r"\clearpage" if name in ("redundancy", "finetune", "final_tasks") else "")
    for name, caption in [
        ("residual_scatter", "TimesFM vs Chronos-2 residuals at individual forecast points (random sample of 4,000 per task)."),
        ("correlation_plane", "Task pairs on the correlation--accuracy plane with Bates--Granger iso-gain curves."),
        ("learning_curves", "Combiner learning curves over the number of past windows used for fitting."),
        ("context_curves", "TimesFM context-length curves per task (\\% change vs ctx 1024), flip invariance on and off."),
        ("lora_seeds", "LoRA effect per task: three TimesFM seeds and Chronos-2."),
        ("gain_shrinkage", "Combination gain with 95\\% intervals for successively improved TimesFM members."),
        ("task_heatmap", "Per-task change vs default TimesFM for the main strategies (common evaluation set)."),
        ("accuracy_cost", "Accuracy vs recurring inference cost; the grey step is the cost--accuracy frontier. "
         "Chronos-2 alone (0.702, 17 s) and TabPFN-TS (0.758, $\\approx$4,100 s) lie outside the plotted range."),
    ]:
        body += [r"\begin{figure}[H]\centering", r"\includegraphics[width=0.95\linewidth]{" + str(figures / name) + ".pdf}",
                 r"\caption{" + caption + "}\\end{figure}"]
    body.append(r"\end{document}")
    tex = build / "transfer_preview.tex"
    tex.write_text("\n".join(body))
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-quiet", f"-outdir={build}", str(tex)],
                   check=True, capture_output=True)
    shutil.copy(build / "transfer_preview.pdf", target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thesis", default="/home/ubuntu/thesis")
    parser.add_argument("--preview", action="store_true", help="Compile all artifacts into one PDF.")
    parser.add_argument("--preview-out", default=str(ROOT / "docs/transfer-results-preview.pdf"))
    args = parser.parse_args()
    tables = Path(args.thesis) / "tables/transfer"
    figures = Path(args.thesis) / "img/transfer"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for build in (table_combiners, table_residuals, table_redundancy, table_context, table_flip, table_chronos_context,
                  table_finetune, table_final, table_contrasts, table_tabpfn, table_final_tasks, table_dm):
        build(tables)
    for build in (figure_residual_scatter, figure_correlation_plane, figure_accuracy_cost, figure_context_curves, figure_learning_curves,
                  figure_gain_shrinkage, figure_task_heatmap, figure_lora_seeds):
        build(figures)
    print(f"tables: {tables}\nfigures: {figures}")
    if args.preview:
        preview(tables, figures, Path(args.preview_out))
        print(f"preview: {args.preview_out}")


if __name__ == "__main__":
    main()
