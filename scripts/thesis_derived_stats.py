"""Derived statistics quoted in the thesis Results chapter; reads saved outputs only.

Run after build_thesis_artifacts.py:  .venv/bin/python scripts/thesis_derived_stats.py
Writes <evidence>/derived_stats.json (flat key -> value) and linear_scaling.csv. Every
value is recomputed from outputs/; no forecasting model is fitted.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from timestransfer.data import load_dataset, rolling_origin_splits
from build_thesis_artifacts import MODELS, EXCLUDED, TASKS, SKIPPED, FOUNDATION


def geomean(values):
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "thesis-package/evidence_rolling_origin"
    )
    args = parser.parse_args()
    ev = args.evidence.resolve()
    ev.mkdir(parents=True, exist_ok=True)
    out: dict[str, object] = {}
    metrics = pd.read_csv(ROOT / "outputs/metrics/metrics.csv", dtype={"unique_id": str})
    metrics = metrics[~metrics.model.isin(EXCLUDED)]
    overall = metrics[metrics.scope.eq("overall")]
    series = metrics[metrics.scope.eq("series")]
    window = metrics[metrics.scope.eq("window")].copy()
    window["window"] = window.window.astype(float).astype(int)
    rel = pd.read_csv(ROOT / "outputs/metrics/relative_to_seasonal_naive.csv")
    rel = (
        rel[~rel.model.isin(EXCLUDED)]
        .pivot(index="dataset", columns="model", values="relative_mase")
        .reindex(index=TASKS, columns=MODELS)
    )
    agg = pd.read_csv(ROOT / "outputs/metrics/aggregate_relative_scores.csv").set_index("model")

    # Coverage.
    f = pd.read_parquet(
        ROOT / "outputs/forecasts/all_datasets_all_models.parquet", columns=["model"]
    )
    tf = pd.read_parquet(
        ROOT / "outputs/timesfm_tuning/forecasts/all_datasets_all_models.parquet", columns=["model"]
    )
    out["coverage.main_forecast_rows"] = int(len(f))
    out["coverage.tuning_forecast_rows"] = int(len(tf))
    out["coverage.scored_combinations"] = int(len(overall))
    out["coverage.skipped"] = len(SKIPPED)

    # Headline aggregate and the per-task picture.
    for m in MODELS:
        out[f"agg.relmase.{m}"] = float(agg.loc[m, "geomean_relative_mase"])
        out[f"agg.relmae.{m}"] = float(agg.loc[m, "geomean_relative_mae"])
        out[f"agg.ntasks.{m}"] = int(agg.loc[m, "n_tasks"])
        out[f"beats_snaive_tasks.{m}"] = int((rel[m] < 1).sum())
    arima_tasks = [t for t in TASKS if (t, "auto_arima") not in SKIPPED]
    for m in MODELS:
        out[f"agg8.relmase.{m}"] = geomean(rel.loc[arima_tasks, m])
    mase = overall.pivot(index="dataset", columns="model", values="mase").reindex(
        index=TASKS, columns=MODELS
    )
    smape = overall.pivot(index="dataset", columns="model", values="smape").reindex(
        index=TASKS, columns=MODELS
    )
    for t in TASKS:
        out[f"winner_mase.{t}"] = mase.loc[t].idxmin()
        out[f"winner_smape.{t}"] = smape.loc[t].idxmin()
        out[f"best_nonsnaive_relmase.{t}"] = float(rel.loc[t].drop("seasonal_naive").min())
        out[f"fm_ratio_max_min.{t}"] = float(
            mase.loc[t, FOUNDATION].max() / mase.loc[t, FOUNDATION].min()
        )
        out[f"winner_fm.{t}"] = mase.loc[t, FOUNDATION].idxmin()
    out["metric_disagreement_tasks"] = [
        t for t in TASKS if out[f"winner_mase.{t}"] != out[f"winner_smape.{t}"]
    ]
    out["n_metric_disagreement_tasks"] = len(out["metric_disagreement_tasks"])
    out["fm_mase_wins"] = int(sum(out[f"winner_mase.{t}"] in FOUNDATION for t in TASKS))
    # Leave-one-task-out geometric means of the three pretrained models.
    loto = {}
    for t in TASKS:
        rest = [x for x in TASKS if x != t]
        loto[t] = {m: geomean(rel.loc[rest, m]) for m in FOUNDATION}
    for t, d in loto.items():
        out[f"loto_order.{t}"] = sorted(d, key=d.get)
        for m in FOUNDATION:
            out[f"loto.{t}.{m}"] = d[m]
    out["loto_n_timesfm_first"] = int(
        sum(out[f"loto_order.{t}"][0] == "timesfm_2p5" for t in TASKS)
    )

    # Window-to-window variability.
    spread = pd.read_csv(ROOT / "outputs/metrics/window_spread.csv")
    spread = spread[~spread.model.isin(EXCLUDED)]
    cv = spread.mase_std / spread.mase_mean
    out["cv.median"] = float(cv.median())
    out["cv.min"] = float(cv.min())
    out["cv.max"] = float(cv.max())
    out["cv.n_pairs"] = int(len(cv))
    for t, g in spread.assign(cv=cv).groupby("dataset"):
        out[f"cv.median_task.{t}"] = float(g.cv.median())
    for m, g in spread.assign(cv=cv).groupby("model"):
        out[f"cv.median_model.{m}"] = float(g.cv.median())
    wm = window.pivot_table(index=["dataset", "window"], columns="model", values="mase")
    wwin = wm.idxmin(axis=1)
    fwin = wm[FOUNDATION].idxmin(axis=1)
    pooled_win = mase.idxmin(axis=1)
    pooled_fm = mase[FOUNDATION].idxmin(axis=1)
    out["windows.n_task_windows"] = int(len(wm))
    out["windows.pooled_winner_wins"] = int(
        sum(wwin.loc[(t, k)] == pooled_win[t] for t, k in wm.index)
    )
    out["windows.tasks_pooled_winner_all5"] = [
        t for t in TASKS if all(wwin.loc[(t, k)] == pooled_win[t] for k in range(5))
    ]
    out["windows.tasks_pooled_fm_all5"] = [
        t for t in TASKS if all(fwin.loc[(t, k)] == pooled_fm[t] for k in range(5))
    ]
    out["windows.snaive_lowest"] = int((wwin == "seasonal_naive").sum())
    out["windows.window0_winner_differs"] = [t for t in TASKS if wwin.loc[(t, 0)] != pooled_win[t]]
    out["windows.n_window0_winner_differs"] = len(out["windows.window0_winner_differs"])
    out["windows.n_distinct_winners_per_task"] = {t: int(wwin.loc[t].nunique()) for t in TASKS}
    relw = wm.div(wm["seasonal_naive"], axis=0)
    for m in FOUNDATION + ["auto_arima"]:
        out[f"windows.below_snaive.{m}"] = int((relw[m] < 1).sum())
        out[f"windows.scored.{m}"] = int(relw[m].notna().sum())
    for t in TASKS:
        for m in FOUNDATION:
            out[f"windows.relmase_range.{t}.{m}"] = [
                float(relw.loc[t, m].min()),
                float(relw.loc[t, m].max()),
            ]

    # Metric disagreement: Weather per-series sMAPE of SeasonalNaive and Chronos-2.
    def per_series(task, metric):
        return series[series.dataset.eq(task)].pivot(
            index="unique_id", columns="model", values=metric
        )

    ws, wq = per_series("weather_h48", "smape"), per_series("weather_h48", "mase")
    zero_inflated = [
        "rain (mm)",
        "raining (s)",
        "SWDR (W/m�)",
        "PAR (�mol/m�/s)",
        "max. PAR (�mol/m�/s)",
    ]
    assert set(zero_inflated) <= set(ws.index), set(ws.index)
    out["weather.zero_inflated_snaive_smape_range"] = [
        float(ws.loc[zero_inflated, "seasonal_naive"].min()),
        float(ws.loc[zero_inflated, "seasonal_naive"].max()),
    ]
    out["weather.zero_inflated_chronos2_smape_range"] = [
        float(ws.loc[zero_inflated, "chronos2"].min()),
        float(ws.loc[zero_inflated, "chronos2"].max()),
    ]
    out["weather.chronos2_lower_smape_series"] = int((ws.chronos2 < ws.seasonal_naive).sum())
    out["weather.chronos2_lower_mase_series"] = int((wq.chronos2 < wq.seasonal_naive).sum())
    out["weather.n_series"] = int(len(ws))
    rest = [u for u in ws.index if u not in zero_inflated]
    out["weather.other16_mean_smape.seasonal_naive"] = float(ws.loc[rest, "seasonal_naive"].mean())
    out["weather.other16_mean_smape.chronos2"] = float(ws.loc[rest, "chronos2"].mean())
    data_cfg = {
        d["name"]: d
        for d in yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())["datasets"]
    }
    raw = load_dataset(ROOT / "data", data_cfg["weather_h48"])
    test = pd.concat([s.test for s in rolling_origin_splits(raw, 48, 5)])
    zero_share = (
        test[test.unique_id.isin(zero_inflated)]
        .groupby("unique_id")
        .y.apply(lambda v: float((v == 0).mean()))
    )
    out["weather.zero_inflated_test_zero_share_range"] = [
        float(zero_share.min()),
        float(zero_share.max()),
    ]
    hs = per_series("ett_h2_h48", "smape")
    out["etth2.hull_smape"] = {
        m: float(hs.loc["HULL", m]) for m in ["seasonal_naive", "chronos2", "timesfm_2p5"]
    }

    # Linear regression: per-series MASE = scaled absolute error * a_i / q_i.
    scales = pd.read_csv(ROOT / "outputs/metrics/mase_scales.csv", dtype={"unique_id": str})
    rows = []
    for t in TASKS:
        d = load_dataset(ROOT / "data", data_cfg[t])
        split = rolling_origin_splits(d, data_cfg[t]["horizon"], 5)[0]
        a = split.train.groupby("unique_id").y.apply(lambda v: float(np.mean(np.abs(v))))
        q = scales[(scales.dataset == t) & (scales.window == 0)].set_index("unique_id").mase_scale
        sm = per_series(t, "mase")
        for uid in sm.index:
            rows.append(
                {
                    "dataset": t,
                    "unique_id": uid,
                    "a_window0": a[uid],
                    "q_window0": q[uid],
                    "q_over_a": q[uid] / a[uid],
                    "linear_mase": sm.loc[uid, "linear_regression"],
                    "snaive_mase": sm.loc[uid, "seasonal_naive"],
                }
            )
    lin = pd.DataFrame(rows)
    lin["ratio"] = lin.linear_mase / lin.snaive_mase
    lin.to_csv(ev / "linear_scaling.csv", index=False)
    for t, g in lin.groupby("dataset"):
        out[f"linear.spearman_qa_ratio.{t}"] = float(spearmanr(g.q_over_a, g.ratio).statistic)
        out[f"linear.worse_than_snaive.{t}"] = int((g.ratio > 1).sum())
        out[f"linear.n_series.{t}"] = int(len(g))
        out[f"linear.median_series_mase.{t}"] = float(g.linear_mase.median())
        out[f"linear.median_q_over_a.{t}"] = float(g.q_over_a.median())
    w = lin[lin.dataset.eq("weather_h48")].set_index("unique_id")
    top3 = w.linear_mase.sort_values(ascending=False).head(3)
    out["linear.weather_top3_ids"] = top3.index.tolist()
    out["linear.weather_top3_share"] = float(top3.sum() / w.linear_mase.sum())
    out["linear.weather_top3_q_over_a_max"] = float(w.loc[top3.index, "q_over_a"].max())
    for uid in top3.index:
        out[f"linear.weather.{uid}.mase"] = float(w.loc[uid, "linear_mase"])
        out[f"linear.weather.{uid}.a"] = float(w.loc[uid, "a_window0"])
        out[f"linear.weather.{uid}.q_over_a"] = float(w.loc[uid, "q_over_a"])
    m4 = lin[lin.dataset.eq("m4_hourly")].set_index("unique_id")
    out["linear.m4_max_series"] = m4.linear_mase.idxmax()
    out["linear.m4_max_series_mase"] = float(m4.linear_mase.max())
    out["linear.m4_max_series_q_over_a"] = float(m4.loc[m4.linear_mase.idxmax(), "q_over_a"])

    # Experiment 3: per-window best context and flip-invariance deltas.
    tuning = pd.read_csv(
        ROOT / "outputs/timesfm_tuning/metrics/metrics.csv", dtype={"unique_id": str}
    )
    to = tuning[tuning.scope.eq("overall")].set_index(["dataset", "model"])
    tw = tuning[tuning.scope.eq("window")].copy()
    tw["window"] = tw.window.astype(float).astype(int)
    contexts = [512, 1024, 2048, 4096, 8192, 16256]
    for t in ["m4_hourly", "ett_h1_h48", "ett_m1_h48", "weather_h48"]:
        for metric in ["smape", "mase"]:
            vals = {c: float(to.loc[(t, f"timesfm_2p5_ctx{c}"), metric]) for c in contexts}
            out[f"tuning.best_ctx.{metric}.{t}"] = min(vals, key=vals.get)
        g = tw[tw.dataset.eq(t)].pivot(index="window", columns="model", values="mase")
        flip = g["timesfm_2p5_no_flip_invariance"] - g["timesfm_2p5_ctx1024"]
        out[f"tuning.flip_mase_worse_windows.{t}"] = int((flip > 0).sum())
        ctxbest = g[[f"timesfm_2p5_ctx{c}" for c in contexts]].idxmin(axis=1)
        out[f"tuning.window_best_ctx_distinct.{t}"] = int(ctxbest.nunique())
    (ev / "derived_stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
