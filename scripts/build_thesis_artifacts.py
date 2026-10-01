"""Reproduce thesis tables/figures from saved rolling-origin forecasts; never fit forecasting models.

Run from the code checkout: .venv/bin/python scripts/build_thesis_artifacts.py
Tables and figures are written into the thesis checkout (--thesis); derived evidence
(inventories, recomputation checks, manifest) goes to --evidence.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import yaml

logging.getLogger("fontTools").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from timestransfer.data import load_dataset, rolling_origin_splits
from timestransfer.analysis import spearman_correlations
from timestransfer.metrics import seasonal_naive_scales

# Experiment 1 comparators. tabpfn_ts_v3p5 is an optional configuration that was not
# evaluated (license pending); it is excluded from every artifact.
MODELS = [
    "linear_regression",
    "tabpfn_ts",
    "timesfm_2p5",
    "chronos2",
    "prophet",
    "auto_arima",
    "auto_ets",
    "auto_theta",
    "seasonal_naive",
]
EXCLUDED = ["tabpfn_ts_v3p5"]
LABELS = dict(
    zip(
        MODELS,
        [
            "Linear",
            "TabPFN-TS",
            "TimesFM",
            "Chronos-2",
            "Prophet",
            "ARIMA",
            "ETS",
            "Theta",
            "SNaive",
        ],
    )
)
FULL_NAMES = dict(
    zip(
        MODELS,
        [
            "Linear regression",
            "TabPFN-TS",
            "TimesFM",
            "Chronos-2",
            "Prophet",
            "AutoARIMA",
            "AutoETS",
            "AutoTheta",
            "SeasonalNaive",
        ],
    )
)
TASKS = [
    "m4_hourly",
    "ett_h1_h48",
    "ett_h1_h96",
    "ett_h2_h48",
    "ett_m1_h48",
    "ett_m2_h48",
    "ecl_h48",
    "traffic_h48",
    "weather_h48",
    "exchange_h48",
    "ili_h48",
]
NAMES = dict(
    zip(
        TASKS,
        [
            "M4 Hourly",
            "ETTh1/48",
            "ETTh1/96",
            "ETTh2/48",
            "ETTm1/48",
            "ETTm2/48",
            "ECL",
            "Traffic",
            "Weather",
            "Exchange",
            "ILI",
        ],
    )
)
# Seasonal AutoARIMA is not run for season lengths above 52 (recorded as skipped).
SKIPPED = {
    ("ett_m1_h48", "auto_arima"),
    ("ett_m2_h48", "auto_arima"),
    ("weather_h48", "auto_arima"),
}
N_WINDOWS = 5
MISSING = "---"
COLORS = dict(
    zip(
        MODELS,
        [
            "#777777",
            "#9467bd",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#8c564b",
            "#009E73",
            "#bcbd22",
            "#56B4E9",
        ],
    )
)
SELECT = [
    ("m4_hourly", "best"),
    ("m4_hourly", "worst"),
    ("ett_h1_h48", "median"),
    ("traffic_h48", "median"),
    ("weather_h48", "worst"),
    ("ili_h48", "median"),
]
SHOWN = ["timesfm_2p5", "chronos2", "tabpfn_ts", "seasonal_naive", "prophet"]
FOUNDATION = ["timesfm_2p5", "chronos2", "tabpfn_ts"]
CORR_MODELS = ["timesfm_2p5", "chronos2", "tabpfn_ts", "seasonal_naive"]
SCORE_CAPTION = (
    r"Bold: minimum before rounding; exact ties share bold. ARIMA, ETS and Theta denote AutoARIMA, AutoETS and AutoTheta; "
    r"SNaive denotes SeasonalNaive; " + MISSING + r": not run because the season length exceeds 52."
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 160,
    }
)


def esc(s):
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("�", "?")
    )


def revised(text):
    return r"\revised{" + text + "}"


def authored(text):
    return r"\authored{" + text + "}"


def table(
    path, caption, label, header, rows, spec=None, small=True, blue=True, size=None, sep="3.5pt"
):
    """Write one float. Regenerated tables carry the blue revision style inside the float."""
    spec = spec or ("l" + "r" * (len(header) - 1))
    size = size or (r"\small" if small else "")
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[!htbp]",
                r"\centering",
                *([r"\revisedstyle"] if blue else []),
                r"\caption{" + caption + "}",
                r"\label{" + label + "}",
                size,
                r"\setlength{\tabcolsep}{" + sep + "}",
                r"\begin{tabular}{" + spec + "}",
                r"\hline",
                " & ".join(header) + r" \\",
                r"\hline",
                *[" & ".join(map(str, row)) + r" \\" for row in rows],
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
    )


def fmt_cells(values, places):
    """Format a row of scores; bold the unrounded minimum among available values."""
    available = values.dropna()
    best = available.min() if len(available) else np.nan
    return [
        MISSING
        if pd.isna(v)
        else (r"\textbf{" + f"{v:.{places}f}" + "}" if v == best else f"{v:.{places}f}")
        for v in values
    ]


def score(frame, scale_column="mase_scale"):
    err = frame.y_pred - frame.y_true
    return frame.assign(
        mae=err.abs(),
        rmse=err**2,
        smape=200 * err.abs() / (frame.y_true.abs() + frame.y_pred.abs() + 1e-8),
        mase=err.abs() / frame[scale_column],
    )


def recompute(scored, saved, scope, group):
    calc = scored.groupby(group)[["mae", "rmse", "smape", "mase"]].mean()
    calc.rmse = np.sqrt(calc.rmse)
    reference = saved[saved.scope.eq(scope)].copy()
    if "window" in group:
        reference["window"] = reference.window.astype(float).astype(int)
    if "horizon" in group:
        reference["horizon"] = reference.horizon.astype(float).astype(int)
    reference = reference.set_index(group).sort_index()
    pd.testing.assert_index_equal(calc.index, reference.index, exact=False)
    np.testing.assert_allclose(calc, reference[calc.columns], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(scored.groupby(group).size(), reference.n_obs)
    return len(calc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis", type=Path, default=Path("/home/ubuntu/thesis"))
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "thesis-package/evidence_rolling_origin"
    )
    args = parser.parse_args()
    thesis = args.thesis.resolve()
    ev = args.evidence.resolve()
    tabs = thesis / "tables/experiments"
    figs = thesis / "img/experiments"
    for p in [ev, tabs, figs]:
        p.mkdir(parents=True, exist_ok=True)
    inputs = set()
    artifacts = []
    checks = {}

    def record(path, sources, transformation):
        inputs.update(s for s in sources if not s.startswith("evidence/"))
        artifacts.append({"output": str(path), "inputs": sources, "transformation": transformation})

    def savefig(fig, name, sources, method):
        for suffix in ["pdf", "png"]:
            p = figs / f"{name}.{suffix}"
            fig.savefig(
                p, bbox_inches="tight", metadata={"CreationDate": None} if suffix == "pdf" else None
            )
            record(p, sources, method)
        plt.close(fig)

    def readcsv(p):
        inputs.add(p)
        return pd.read_csv(ROOT / p, dtype={"unique_id": str})

    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    metrics = readcsv("outputs/metrics/metrics.csv")
    tuning = readcsv("outputs/timesfm_tuning/metrics/metrics.csv")
    metrics = metrics[~metrics.model.isin(EXCLUDED)].copy()
    forecasts = {}
    scales = {}
    envs = {}
    for key, base, expected in [
        ("main", "outputs", (96, 480)),
        ("tuning", "outputs/timesfm_tuning", (44, 220)),
    ]:
        mp = f"{base}/metadata/environment.json"
        inputs.add(mp)
        envs[key] = json.loads((ROOT / mp).read_text())
        cfg = "configs/experiment.yaml" if key == "main" else "configs/timesfm_tuning.yaml"
        inputs.add(cfg)
        assert envs[key]["config"] == yaml.safe_load((ROOT / cfg).read_text())
        fp = f"{base}/forecasts/all_datasets_all_models.parquet"
        inputs.add(fp)
        f = pd.read_parquet(ROOT / fp)
        f.unique_id = f.unique_id.astype(str)
        assert not f.model.isin(EXCLUDED).any()
        assert not f.duplicated(["dataset", "model", "window", "unique_id", "horizon"]).any()
        assert sorted(f.window.unique()) == list(range(N_WINDOWS))
        assert np.isfinite(f[["y_true", "y_pred"]]).all().all()
        sc = readcsv(f"{base}/metrics/mase_scales.csv")
        scales[key] = sc
        assert not sc.duplicated(["dataset", "window", "unique_id"]).any()
        assert (np.isfinite(sc.mase_scale) & sc.mase_scale.gt(0)).all()
        f = f.merge(sc, on=["dataset", "window", "unique_id"], validate="many_to_one")
        scored = score(f)
        saved = metrics if key == "main" else tuning
        counts = {}
        # overall/series/horizon pool all five windows; the window scope scores each origin.
        for scope, group in [
            ("overall", ["dataset", "model"]),
            ("window", ["dataset", "model", "window"]),
            ("series", ["dataset", "model", "unique_id"]),
            ("horizon", ["dataset", "model", "horizon"]),
        ]:
            counts[scope] = recompute(scored, saved, scope, group)
        assert (counts["overall"], counts["window"]) == expected
        statuses = envs[key]["model_statuses"]
        for task, d in statuses.items():
            for model, v in d.items():
                if model in EXCLUDED:
                    assert v["status"] != "ok"
                    continue
                assert v["status"] == ("skipped" if (task, model) in SKIPPED else "ok"), (
                    task,
                    model,
                )
                assert v["n_windows"] == N_WINDOWS and v["n_windows_ok"] == (
                    0 if (task, model) in SKIPPED else N_WINDOWS
                )
        checks[key] = {
            "forecast_rows": len(f),
            "metric_rows_recomputed": counts,
            "all_metrics_match": True,
        }
        forecasts[key] = scored
    status_rows = metrics[metrics.scope.eq("model_status")]
    assert (
        set(zip(status_rows.dataset, status_rows.model)) == SKIPPED
        and status_rows.status.eq("skipped").all()
    )
    f = forecasts["main"]
    overall = metrics[metrics.scope.eq("overall")]
    # Relative and aggregate scores: recompute from overall rows and compare with saved CSVs.
    relative = readcsv("outputs/metrics/relative_to_seasonal_naive.csv")
    relative = relative[~relative.model.isin(EXCLUDED)]
    aggregate = readcsv("outputs/metrics/aggregate_relative_scores.csv")
    aggregate = aggregate[~aggregate.model.isin(EXCLUDED)].set_index("model")
    spread = readcsv("outputs/metrics/window_spread.csv")
    spread = spread[~spread.model.isin(EXCLUDED)]
    base = overall[overall.model.eq("seasonal_naive")].set_index("dataset")
    rel = overall.assign(
        relative_mase=overall.mase.to_numpy() / base.loc[overall.dataset, "mase"].to_numpy(),
        relative_mae=overall.mae.to_numpy() / base.loc[overall.dataset, "mae"].to_numpy(),
    )
    merged = rel.merge(
        relative, on=["dataset", "model"], suffixes=("", "_saved"), validate="one_to_one"
    )
    assert len(merged) == 96
    np.testing.assert_allclose(
        merged[["relative_mase", "relative_mae"]],
        merged[["relative_mase_saved", "relative_mae_saved"]],
        rtol=1e-12,
    )
    agg = rel.groupby("model").agg(
        n_tasks=("dataset", "size"),
        geomean_relative_mase=("relative_mase", lambda x: np.exp(np.log(x).mean())),
        geomean_relative_mae=("relative_mae", lambda x: np.exp(np.log(x).mean())),
    )
    np.testing.assert_array_equal(agg.n_tasks, aggregate.loc[agg.index].n_tasks)
    np.testing.assert_allclose(
        agg[["geomean_relative_mase", "geomean_relative_mae"]],
        aggregate.loc[agg.index, ["geomean_relative_mase", "geomean_relative_mae"]],
        rtol=1e-12,
    )
    wscope = metrics[metrics.scope.eq("window")]
    wstat = wscope.groupby(["dataset", "model"]).agg(
        mase_mean=("mase", "mean"),
        mase_std=("mase", "std"),
        smape_mean=("smape", "mean"),
        smape_std=("smape", "std"),
    )
    sp = spread.set_index(["dataset", "model"]).loc[wstat.index]
    np.testing.assert_allclose(wstat, sp[wstat.columns], rtol=1e-10)
    checks["relative_aggregate_and_spread_recomputed"] = {
        "relative_rows": len(merged),
        "aggregate_models": len(agg),
        "spread_rows": len(wstat),
    }

    for metric, places in [("smape", 2), ("mase", 3), ("mae", 3), ("rmse", 3)]:
        pivot = overall.pivot(index="dataset", columns="model", values=metric).reindex(
            index=TASKS, columns=MODELS
        )
        assert pivot.isna().sum().sum() == len(SKIPPED)
        # Split tasks, not models: all nine comparators remain visible in every row.
        for part, ds in enumerate([TASKS[:6], TASKS[6:]], 1):
            rows = [[NAMES[task]] + fmt_cells(pivot.loc[task], places) for task in ds]
            headers = ["Task"] + [LABELS[m] for m in MODELS]
            if metric in ["mae", "rmse"]:
                rows = [
                    [LABELS[model]] + [rows[j][i + 1] for j in range(len(ds))]
                    for i, model in enumerate(MODELS)
                ]
                headers = ["Model"] + [NAMES[task] for task in ds]
            p = tabs / f"{metric}_{part}.tex"
            name = metric.upper() if metric != "smape" else "sMAPE"
            table(
                p,
                revised(
                    f"{name} by task, part {part} ($\\downarrow$), pooled over the five test windows. "
                    + SCORE_CAPTION
                ),
                f"tab:{metric}{part}",
                headers,
                rows,
                **({} if metric in ["mae", "rmse"] else {"size": r"\footnotesize", "sep": "2.5pt"}),
            )
            record(
                p,
                ["outputs/metrics/metrics.csv"],
                f"overall (all-window) {metric}; task order fixed; {places} decimals; exact unrounded minima; skipped tasks shown as {MISSING}",
            )
    # Headline aggregate: geometric mean over tasks of MASE/MAE relative to SeasonalNaive.
    order = agg.sort_values("geomean_relative_mase").index.tolist()
    rows = [
        [
            FULL_NAMES[m],
            f"{agg.loc[m, 'geomean_relative_mase']:.3f}",
            f"{agg.loc[m, 'geomean_relative_mae']:.3f}",
            int(agg.loc[m, "n_tasks"]),
        ]
        for m in order
    ]
    p = tabs / "aggregate.tex"
    table(
        p,
        revised(
            r"Aggregate relative scores ($\downarrow$): geometric mean over tasks of each model's pooled MASE and MAE divided by SeasonalNaive's on the same task. "
            r"Values below one indicate a lower error than SeasonalNaive on a typical task. AutoARIMA is averaged over its eight completed tasks only."
        ),
        "tab:aggregate",
        ["Model", "Relative MASE", "Relative MAE", "Tasks"],
        rows,
        spec="lrrr",
    )
    record(
        p,
        [
            "outputs/metrics/aggregate_relative_scores.csv",
            "outputs/metrics/relative_to_seasonal_naive.csv",
        ],
        "geometric means recomputed from overall rows; sorted by relative MASE; 3 decimals",
    )
    relpivot = rel.pivot(index="dataset", columns="model", values="relative_mase").reindex(
        index=TASKS, columns=MODELS
    )
    rows = [[NAMES[t]] + fmt_cells(relpivot.loc[t].drop("seasonal_naive"), 3) for t in TASKS]
    rows.append(
        [r"\hline Geo.\ mean"]
        + [f"{agg.loc[m, 'geomean_relative_mase']:.3f}" for m in MODELS if m != "seasonal_naive"]
    )
    p = tabs / "relmase.tex"
    table(
        p,
        revised(
            r"Per-task MASE relative to SeasonalNaive ($\downarrow$; pooled over the five windows). Values below one beat SeasonalNaive on that task. "
            r"Bold: minimum before rounding. ARIMA, ETS and Theta denote AutoARIMA, AutoETS and AutoTheta. The final row repeats the geometric means of Table~\ref{tab:aggregate}; "
            + MISSING
            + r": not run."
        ),
        "tab:relmase",
        ["Task"] + [LABELS[m] for m in MODELS if m != "seasonal_naive"],
        rows,
    )
    record(
        p,
        ["outputs/metrics/relative_to_seasonal_naive.csv"],
        "relative MASE per task; SeasonalNaive column omitted (identically one)",
    )
    # Window-to-window spread of window-level MASE: mean +/- sample standard deviation.
    for part, ds in enumerate([TASKS[:6], TASKS[6:]], 1):
        rows = []
        for m in MODELS:
            cells = []
            for t in ds:
                if (t, m) in SKIPPED:
                    cells.append(MISSING)
                    continue
                s = wstat.loc[(t, m)]
                cells.append(f"${s.mase_mean:.3f}\\pm{s.mase_std:.3f}$")
            rows.append([LABELS[m]] + cells)
        p = tabs / f"spread_{part}.tex"
        table(
            p,
            revised(
                f"Window-level MASE, part {part}: mean $\\pm$ sample standard deviation over the five test windows. "
                r"The mean equals the pooled MASE of Tables~\ref{tab:mase1} and~\ref{tab:mase2} because every window contains the same number of scored observations. "
                + MISSING
                + ": not run."
            ),
            f"tab:spread{part}",
            ["Model"] + [NAMES[t] for t in ds],
            rows,
            size=r"\footnotesize",
            sep="2.5pt",
        )
        record(
            p,
            ["outputs/metrics/window_spread.csv"],
            "window scope MASE mean and sample SD (ddof=1); 3 decimals",
        )
    spread_cv = wstat.assign(cv=wstat.mase_std / wstat.mase_mean).reset_index()
    spread_cv.to_csv(ev / "window_spread_cv.csv", index=False)
    # Window-level winners and relative scores for the variability analysis.
    wpiv = wscope.assign(window=wscope.window.astype(float).astype(int)).pivot_table(
        index=["dataset", "window"], columns="model", values=["mase", "smape"]
    )
    winners = []
    for (task, window), row in wpiv.iterrows():
        mase = row["mase"].dropna()
        smape = row["smape"].dropna()
        fm = mase[FOUNDATION]
        winners.append(
            {
                "dataset": task,
                "window": window,
                "best_mase": mase.idxmin(),
                "best_smape": smape.idxmin(),
                "best_foundation_mase": fm.idxmin(),
                "snaive_rank_mase": int(mase.rank().loc["seasonal_naive"]),
                **{f"relative_mase_{m}": mase[m] / mase["seasonal_naive"] for m in mase.index},
            }
        )
    pd.DataFrame(winners).to_csv(ev / "window_winners.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 3.0), layout="constrained")
    wrel = pd.DataFrame(winners)
    for j, m in enumerate(["timesfm_2p5", "chronos2", "tabpfn_ts", "auto_arima"]):
        for i, t in enumerate(TASKS):
            vals = wrel.loc[wrel.dataset.eq(t), f"relative_mase_{m}"].dropna()
            if vals.empty:
                continue
            x = i + (j - 1.5) * 0.18
            ax.scatter(
                np.full(len(vals), x),
                vals,
                s=9,
                color=COLORS[m],
                alpha=0.75,
                label=("AutoARIMA" if m == "auto_arima" else LABELS[m]) if i == 0 else None,
                zorder=3,
            )
            ax.plot(
                [x - 0.07, x + 0.07], [relpivot.loc[t, m]] * 2, color=COLORS[m], lw=1.6, zorder=4
            )
    ax.axhline(1, color="0.3", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(TASKS)), [NAMES[t] for t in TASKS], rotation=35, ha="right")
    ax.set(ylabel="MASE relative to SeasonalNaive\n(same window, log scale)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    savefig(
        fig,
        "window_relative_mase",
        ["outputs/metrics/metrics.csv"],
        "window scope MASE divided by same-window SeasonalNaive MASE; points=windows, bars=pooled relative MASE",
    )

    inventory = []
    series_inventory = []
    selections = []
    missing_intervals = []
    w0rows = []
    n_scale_checks = 0
    for dc in config["datasets"]:
        task = dc["name"]
        H = dc["horizon"]
        print("Checking cached dataset:", task, flush=True)
        if task == "m4_hourly":
            rawpaths = [
                "data/m4/datasets/Hourly.p",
                "data/m4/datasets/Hourly-train.csv",
                "data/m4/datasets/Hourly-test.csv",
            ]
            rawlen = None
        else:
            rawpaths = [f"data/longhorizon2/all_six_datasets/{dc['group']}/Y_df.csv"]
            rawlen = len(pd.read_csv(ROOT / rawpaths[0], usecols=["date"]))
        inputs.update(rawpaths)
        data = load_dataset(ROOT / "data", dc)
        splits = rolling_origin_splits(data, H, N_WINDOWS)
        assert not data.duplicated(["unique_id", "ds"]).any()
        assert np.isfinite(data.y).all()
        lag = dc.get("mase_seasonality", dc["seasonality"])
        tf = f[f.dataset.eq(task)]
        for window, split in enumerate(splits):
            # Window k's training part ends k*H + H steps before the series end; later data are dropped.
            assert (
                split.train.groupby("unique_id")
                .size()
                .add(H + window * H)
                .eq(data.groupby("unique_id").size())
                .all()
            )
            loaded_test = split.test.sort_values(["unique_id", "ds"])
            for model, g in tf[tf.window.eq(window)].groupby("model"):
                g = g.sort_values(["unique_id", "horizon"])
                assert g.unique_id.tolist() == loaded_test.unique_id.tolist()
                np.testing.assert_array_equal(g.y_true, loaded_test.y)
                if task == "m4_hourly":
                    np.testing.assert_array_equal(pd.to_numeric(g.ds), loaded_test.ds)
                else:
                    np.testing.assert_array_equal(pd.to_datetime(g.ds), loaded_test.ds)
            assert set(tf[tf.window.eq(window)].model) == {
                m for m in MODELS if (task, m) not in SKIPPED
            }
            recalculated = seasonal_naive_scales(split.train, lag).set_index("unique_id").mase_scale
            saved_scale = (
                scales["main"]
                .query("dataset == @task and window == @window")
                .set_index("unique_id")
                .mase_scale
            )
            np.testing.assert_allclose(recalculated.loc[saved_scale.index], saved_scale, rtol=1e-12)
            n_scale_checks += len(saved_scale)
        first, last = splits[0], splits[-1]
        lengths = []
        lengths_last = []
        negative = []
        gaps = 0
        groups = [
            dict(tuple(part.groupby("unique_id", sort=True)))
            for part in (first.train, first.test, last.train, last.test)
        ]
        for uid, g in data.groupby("unique_id", sort=True):
            tr, te, tr4, te4 = (part[uid] for part in groups)
            lengths.append(len(tr))
            lengths_last.append(len(tr4))
            dif = g.ds.diff().dropna()
            step = 1 if task == "m4_hourly" else pd.tseries.frequencies.to_offset(dc["time_freq"])
            # W-TUE is a calendar offset; compare its fixed seven-day duration.
            if task == "ili_h48":
                step = pd.Timedelta(days=7)
            elif task != "m4_hourly":
                step = pd.Timedelta(step)
            nbad = int((dif != step).sum())
            gaps = max(gaps, nbad)
            if nbad and uid == data.unique_id.iloc[0]:
                for idx in dif[dif != step].index:
                    missing_intervals.append(
                        {
                            "dataset": task,
                            "previous": str(g.loc[:idx].ds.iloc[-2]),
                            "next": str(g.loc[idx, "ds"]),
                            "delta": str(dif.loc[idx]),
                        }
                    )
            negative.append(bool((g.y < 0).any()))
            series_inventory.append(
                {
                    "dataset": task,
                    "unique_id": uid,
                    "evaluated_length": len(g),
                    "train_length_window0": len(tr),
                    "train_length_window4": len(tr4),
                    "first": str(g.ds.iloc[0]),
                    "window4_train_end": str(tr4.ds.iloc[-1]),
                    "window4_test_start": str(te4.ds.iloc[0]),
                    "window4_test_end": str(te4.ds.iloc[-1]),
                    "window0_train_end": str(tr.ds.iloc[-1]),
                    "window0_test_start": str(te.ds.iloc[0]),
                    "test_end": str(te.ds.iloc[-1]),
                    "nonfinite": int((~np.isfinite(g.y)).sum()),
                    "nonstandard_intervals": nbad,
                    "negative_values": int((g.y < 0).sum()),
                    "zero_values": int(g.y.eq(0).sum()),
                    "minimum": g.y.min(),
                    "maximum": g.y.max(),
                }
            )
        inventory.append(
            {
                "dataset": task,
                "source_rows_per_variable": rawlen,
                "n_series": len(lengths),
                "train_min": min(lengths),
                "train_max": max(lengths),
                "train_window4_min": min(lengths_last),
                "train_window4_max": max(lengths_last),
                "evaluated_min": min(lengths) + H,
                "evaluated_max": max(lengths) + H,
                "frequency": dc["time_freq"],
                "horizon": H,
                "seasonality": dc["seasonality"],
                "mase_lag": lag,
                "first": str(data.ds.min()),
                "last": str(data.ds.max()),
                "irregular_intervals_per_series": gaps,
                "series_with_negative_values": sum(negative),
                "nonfinite_values": 0,
            }
        )
        # Diagnostic examples use window 0 only: its training data form the displayed history.
        w0 = (
            tf[tf.window.eq(0)]
            .groupby(["model", "unique_id"])[["mae", "smape", "mase"]]
            .mean()
            .reset_index()
        )
        w0.insert(0, "dataset", task)
        w0rows.append(w0)
        for selected_task, rule in SELECT:
            if task != selected_task:
                continue
            ref = w0[w0.model.eq("timesfm_2p5")].sort_values(["smape", "unique_id"], kind="stable")
            ix = {"best": 0, "median": len(ref) // 2, "worst": len(ref) - 1}[rule]
            uid = ref.iloc[ix].unique_id
            history = first.train[first.train.unique_id.eq(uid)].tail(
                max(3 * dc["seasonality"], 2 * H)
            )
            sf = tf[tf.window.eq(0) & tf.unique_id.eq(uid)]
            actual = sf[sf.model.eq("timesfm_2p5")].sort_values("horizon")
            if task == "weather_h48":
                fig, (ax, zoom) = plt.subplots(
                    1,
                    2,
                    figsize=(7.1, 2.8),
                    layout="constrained",
                    gridspec_kw={"width_ratios": [1.6, 1]},
                )
                zoom.plot(actual.horizon, actual.y_true, color="black", lw=1.6, zorder=5)
                for model in SHOWN:
                    zg = sf[sf.model.eq(model)].sort_values("horizon")
                    zoom.plot(zg.horizon, zg.y_pred, color=COLORS[model], ls="--", lw=1.15)
                zoom.set(title="Continuation detail", xlabel="Forecast step", ylabel="Stored units")
            else:
                fig, ax = plt.subplots(figsize=(7.1, 2.8), layout="constrained")
            ax.plot(
                np.arange(1 - len(history), 1),
                history.y,
                color="0.55",
                label="Training history (window 0)",
                lw=1,
            )
            ax.plot(
                np.arange(0, H + 1),
                np.r_[history.y.iloc[-1], actual.y_true],
                color="black",
                label="Actual continuation",
                lw=1.6,
                zorder=5,
            )
            ax.axvline(0, color="0.3", ls=":", lw=1)
            ax.axvspan(0, H, color="#f4f4f4", zorder=-1)
            for model in SHOWN:
                g = sf[sf.model.eq(model)].sort_values("horizon")
                ax.plot(
                    g.horizon, g.y_pred, color=COLORS[model], ls="--", lw=1.15, label=LABELS[model]
                )
            display_id = "max. PAR" if task == "weather_h48" else uid
            ax.set(
                xlabel=f"Observations from forecast origin ({dc['time_freq']} per step)",
                ylabel="Value (stored units)",
                title=f"{NAMES[task]} / {display_id} — {rule} TimesFM window-0 sMAPE",
            )
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="outside lower center", ncol=4, fontsize=8, frameon=False
            )
            name = f"forecast_{len(selections) + 1}"
            source = [
                "outputs/forecasts/all_datasets_all_models.parquet",
                "outputs/metrics/mase_scales.csv",
                *rawpaths,
            ]
            savefig(
                fig,
                name,
                source,
                f"{task}/{uid}; window 0; {rule} ranked TimesFM window-0 sMAPE; ties by string ID; history={len(history)}; models={SHOWN}",
            )
            scores = {
                m: float(w0.query("model == @m and unique_id == @uid").smape.iloc[0]) for m in SHOWN
            }
            pooled = {
                m: float(
                    metrics.query(
                        'dataset == @task and unique_id == @uid and model == @m and scope == "series"'
                    ).smape.iloc[0]
                )
                for m in SHOWN
            }
            selections.append(
                {
                    "figure": name,
                    "dataset": task,
                    "unique_id": uid,
                    "rule": rule,
                    "reference_model": "timesfm_2p5",
                    "window": 0,
                    "rank_1based": ix + 1,
                    "n_series": len(ref),
                    "history_points": len(history),
                    "origin": str(history.ds.iloc[-1]),
                    "window0_smape": scores,
                    "pooled_smape": pooled,
                }
            )
    checks["loader_alignment"] = {
        "tasks": 11,
        "windows": N_WINDOWS,
        "targets_and_timestamps_match": True,
        "per_window_mase_scales_match": n_scale_checks,
    }
    pd.concat(w0rows).to_csv(ev / "window0_series_metrics.csv", index=False)
    inv = pd.DataFrame(inventory)
    inv.to_csv(ev / "dataset_inventory.csv", index=False)
    pd.DataFrame(series_inventory).to_csv(ev / "series_inventory.csv", index=False)
    (ev / "timestamp_gaps.json").write_text(json.dumps(missing_intervals, indent=2) + "\n")
    (ev / "forecast_selections.json").write_text(json.dumps(selections, indent=2) + "\n")
    rawrows = []
    for r in inventory:
        if r["dataset"] == "ett_h1_h96":
            continue
        available = (
            f"{r['evaluated_min']}--{r['evaluated_max']}"
            if r["source_rows_per_variable"] is None
            else str(r["source_rows_per_variable"])
        )
        evaluated = (
            str(r["evaluated_min"])
            if r["evaluated_min"] == r["evaluated_max"]
            else f"{r['evaluated_min']}--{r['evaluated_max']}"
        )
        rawrows.append([NAMES[r["dataset"]], available, evaluated])
    # Unchanged by the revision: keep the original (red) authorship marking.
    table(
        tabs / "raw_lengths.tex",
        authored(
            "Cached source-file and evaluated lengths per variable. M4 source-file padding is excluded from observation counts."
        ),
        "tab:rawlengths",
        ["Panel", "Available observations", "Evaluated observations"],
        rawrows,
        blue=False,
    )
    record(
        tabs / "raw_lengths.tex",
        ["evidence/dataset_inventory.csv"],
        "available versus loader-retained counts; repeated ETTh1/96 omitted",
    )
    rows = []
    durations = {
        "h": "2 days",
        "15min": "12 hours",
        "10min": "8 hours",
        "D": "48 days*",
        "W-TUE": "48 weeks",
    }
    for r in inventory:
        length = (
            str(r["train_min"])
            if r["train_min"] == r["train_max"]
            else f"{r['train_min']}--{r['train_max']}"
        )
        duration = "4 days" if r["horizon"] == 96 else durations[r["frequency"]]
        rows.append(
            [
                NAMES[r["dataset"]],
                "M4" if r["dataset"] == "m4_hourly" else "LH2",
                esc(r["frequency"]),
                r["n_series"],
                length,
                f"{r['horizon']} / {duration}",
                f"{r['seasonality']} / {r['mase_lag']}",
            ]
        )
    table(
        tabs / "datasets.tex",
        authored(
            r"Evaluated data (datasetsforecast 1.0.1). LH2: cached LongHorizon2 archive, unnormalized. $s/m$: configured seasonal lag / MASE lag. *Exchange elapsed time refers to the archive labels."
        )
        + " "
        + revised(
            r"Train length refers to window 0, the final holdout; the training part of window $k$ is $kH$ observations shorter."
        ),
        "tab:datasets",
        ["Task", "Version", "Step", "Series", "Train length", "$H$ / duration", "$s/m$"],
        rows,
        spec="lllrlll",
        blue=False,
    )
    (tabs / "datasets.tex").write_text(
        (tabs / "datasets.tex")
        .read_text()
        .replace(r"\begin{table}[!htbp]", r"\begin{table}[H]" + "\n% [!htbp]", 1)
    )
    record(
        tabs / "datasets.tex",
        sorted(p for p in inputs if p.startswith("data/")) + ["configs/experiment.yaml"],
        "exact loader and window-0 split; per-series inventory in evidence",
    )
    bounds = []
    for task in TASKS[1:]:
        r = next(r for r in series_inventory if r["dataset"] == task)
        bounds.append(
            [
                NAMES[task],
                r["first"][:16],
                r["window4_test_start"][:16],
                r["window0_test_start"][:16],
                r["test_end"][:16],
            ]
        )
    # Use dates and times in separate lines to keep all boundaries legible.
    bounds = [
        [row[0]] + [r"\shortstack{" + v[:10] + r"\\" + v[11:] + "}" for v in row[1:]]
        for row in bounds
    ]
    table(
        tabs / "boundaries.tex",
        revised(
            "Exact loaded boundaries shared by all variables within each task; these are archive labels, not independently verified observation dates. "
            "The five test windows are contiguous: window 4 starts the earliest test span and window 0 ends at the last observation. "
            "The training part of window $k$ ends immediately before its first test observation."
        ),
        "tab:boundaries",
        ["Task", "First training", "Window 4 first test", "Window 0 first test", "Last test"],
        bounds,
    )
    record(
        tabs / "boundaries.tex",
        ["evidence/series_inventory.csv"],
        "first timestamp, first test timestamps of windows 4 and 0, last timestamp",
    )
    for metric in ["smape", "mase"]:
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), layout="constrained")
        for ax, task in zip(axes.flat, ["m4_hourly", "ett_h1_h96", "traffic_h48", "weather_h48"]):
            for model in SHOWN:
                g = metrics.query(
                    'dataset == @task and model == @model and scope == "horizon"'
                ).copy()
                g["horizon"] = g.horizon.astype(float).astype(int)
                g = g.sort_values("horizon")
                ax.plot(g.horizon, g[metric], label=LABELS[model], color=COLORS[model], lw=1)
            ax.set(
                title=NAMES[task],
                xlabel="Forecast step",
                ylabel="sMAPE (%)" if metric == "smape" else "MASE",
            )
        axes.flat[0].legend(fontsize=7, ncol=2)
        savefig(
            fig,
            f"horizon_{metric}",
            ["outputs/metrics/metrics.csv"],
            f"horizon scope pooled over five windows; fixed tasks and five models; {metric}",
        )
    joined = readcsv("outputs/analysis/series_features_with_metrics.csv")
    features = readcsv("outputs/analysis/series_features.csv")
    joined = joined[~joined.model.isin(EXCLUDED)]
    assert len(features) == 1668 and len(joined) == 14977
    assert not features.duplicated(["dataset", "unique_id"]).any()
    # Features come from window 0's training part; per-series metrics pool all five windows.
    jcheck = metrics.query('scope == "series"').merge(
        features, on=["dataset", "unique_id"], validate="many_to_one"
    )
    cols = [c for c in joined.columns if c not in ("horizon", "window")]
    pd.testing.assert_frame_equal(
        joined.sort_values(["dataset", "model", "unique_id"]).reset_index(drop=True)[cols],
        jcheck[cols].sort_values(["dataset", "model", "unique_id"]).reset_index(drop=True),
        check_dtype=False,
    )
    for metric in ["smape", "mase"]:
        cor = spearman_correlations(joined, metric=metric)
        if metric == "smape":
            old = readcsv("outputs/analysis/feature_metric_correlations.csv")
            pd.testing.assert_frame_equal(cor, old, check_dtype=False, rtol=1e-10, atol=1e-10)
        cor.to_csv(ev / f"correlations_{metric}.csv", index=False)
        rows = []
        for task in ["m4_hourly", "ecl_h48", "traffic_h48"]:
            for feature in ["acf_seasonal", "seasonal_strength"]:
                c = cor[(cor.dataset == task) & (cor.feature == feature)].set_index("model")
                assert c.loc[CORR_MODELS, "n_series"].nunique() == 1
                rows.append(
                    [
                        NAMES[task],
                        "ACF($s$)" if feature == "acf_seasonal" else "STL seasonal",
                        int(c.loc["timesfm_2p5", "n_series"]),
                        *[f"{c.loc[m, 'spearman_rho']:.3f}" for m in CORR_MODELS],
                    ]
                )
        p = tabs / f"correlations_{metric}.tex"
        table(
            p,
            revised(
                f"Within-task Spearman correlations between window-0 training features and per-series {'sMAPE' if metric == 'smape' else 'MASE'} pooled over the five windows. "
                "$n$ is the valid-pair count, identical for the displayed models."
                + (
                    " The MASE view is an additional descriptive analysis of saved results."
                    if metric == "mase"
                    else ""
                )
            ),
            f"tab:corr{metric}",
            ["Task", "Feature", "$n$"] + [LABELS[m] for m in CORR_MODELS],
            rows,
        )
        record(
            p,
            ["outputs/analysis/series_features_with_metrics.csv"],
            f"Spearman {metric}; within-task complete pairs; fixed three large panels and two features",
        )
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.8), layout="constrained")
    for row, task in enumerate(["m4_hourly", "ecl_h48", "traffic_h48"]):
        g = joined[(joined.dataset == task) & (joined.model == "timesfm_2p5")]
        for col, feat in enumerate(["acf_seasonal", "seasonal_strength"]):
            valid = g[[feat, "smape"]].dropna()
            ax = axes[row, col]
            ax.scatter(valid[feat], valid.smape, s=7, alpha=0.4, color=COLORS["timesfm_2p5"])
            ax.set(
                xlabel="Seasonal ACF" if col == 0 else "STL seasonal strength",
                ylabel="TimesFM sMAPE (%)",
                title=f"{NAMES[task]} (n={len(valid)})",
            )
    savefig(
        fig,
        "feature_scatter",
        ["outputs/analysis/series_features_with_metrics.csv"],
        "TimesFM; three large panels; window-0 ACF and STL vs all-window sMAPE; no significance filtering",
    )
    tune = tuning[tuning.scope.eq("overall")]
    contexts = [512, 1024, 2048, 4096, 8192, 16256]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), layout="constrained")
    flags = {
        "no_normalize": "Normalization",
        "no_quantile_head": "Quantile head",
        "no_flip_invariance": "Flip invariance",
        "no_infer_positive": "Positive inference",
        "no_fix_crossing": "Crossing repair",
    }
    flagkeys = [
        "normalize_inputs",
        "use_continuous_quantile_head",
        "force_flip_invariance",
        "infer_is_positive",
        "fix_quantile_crossing",
    ]
    deltas = []
    ctrows = []
    for ax, task in zip(axes.flat, ["m4_hourly", "ett_h1_h48", "ett_m1_h48", "weather_h48"]):
        g = tune[tune.dataset.eq(task)].set_index("model")
        sweep = g.loc[[f"timesfm_2p5_ctx{x}" for x in contexts]]
        ax.plot(range(6), sweep.smape, "o-", color=COLORS["timesfm_2p5"], ms=3)
        ax.set(ylabel="sMAPE (%)", title=NAMES[task], xlabel="Maximum context (observations)")
        ax.set_xticks(range(6), ["512", "1k", "2k", "4k", "8k", "16256"], rotation=35)
        right = ax.twinx()
        right.plot(range(6), sweep.mase, "s--", color=COLORS["chronos2"], ms=3)
        right.set_ylabel("MASE", color=COLORS["chronos2"])
        right.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.tick_params(axis="y", labelcolor=COLORS["timesfm_2p5"])
        baseline = g.loc["timesfm_2p5_ctx1024"]
        for ctx in contexts:
            ctrows.append(
                [
                    NAMES[task],
                    f"{ctx:,}",
                    f"{g.loc[f'timesfm_2p5_ctx{ctx}', 'smape']:.3f}",
                    f"{g.loc[f'timesfm_2p5_ctx{ctx}', 'mase']:.4f}",
                ]
            )
            status = envs["tuning"]["model_statuses"][task][f"timesfm_2p5_ctx{ctx}"]
            assert status["max_context"] == ctx and all(
                status["forecast_config"][k] for k in flagkeys
            )
            assert status["revision"] == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
        for flag, label in flags.items():
            v = g.loc[f"timesfm_2p5_{flag}"]
            deltas.append(
                {
                    "dataset": task,
                    "flag": flag,
                    "smape_delta": v.smape - baseline.smape,
                    "mase_delta": v.mase - baseline.mase,
                }
            )
            status = envs["tuning"]["model_statuses"][task][f"timesfm_2p5_{flag}"]
            assert (
                status["max_context"] == 1024
                and sum(not status["forecast_config"][k] for k in flagkeys) == 1
            )
            assert status["revision"] == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    savefig(
        fig,
        "timesfm_context",
        ["outputs/timesfm_tuning/metrics/metrics.csv"],
        "six contexts; per-task sMAPE left blue / MASE right orange; independent axes; five windows pooled",
    )
    delta = pd.DataFrame(deltas)
    delta.to_csv(ev / "flag_deltas.csv", index=False)
    for metric, places in [("smape", 3), ("mase", 4)]:
        rows = []
        for flag, label in flags.items():
            g = delta[delta.flag.eq(flag)].set_index("dataset")
            rows.append(
                [label]
                + [
                    f"{g.loc[t, metric + '_delta']:+.{places}f}"
                    for t in ["m4_hourly", "ett_h1_h48", "ett_m1_h48", "weather_h48"]
                ]
            )
        p = tabs / f"flags_{metric}.tex"
        table(
            p,
            revised(
                f"Change in {'sMAPE (percentage points)' if metric == 'smape' else 'MASE'} when the named flag is disabled, relative to the same-run 1,024-context baseline; scores pool the five test windows. Negative is better; rounded zeros can conceal small differences."
            ),
            f"tab:flags{metric}",
            ["Disabled flag", "M4", "ETTh1", "ETTm1", "Weather"],
            rows,
        )
        record(
            p,
            ["outputs/timesfm_tuning/metrics/metrics.csv"],
            f"{metric}(ablation) - {metric}(ctx1024), within tuning run",
        )
    table(
        tabs / "contexts.tex",
        revised(
            r"All saved context-study scores ($\downarrow$), pooled over the five test windows; no validation selection."
        ),
        "tab:contexts",
        ["Task", "Context", "sMAPE", "MASE"],
        ctrows,
    )
    record(
        tabs / "contexts.tex",
        ["outputs/timesfm_tuning/metrics/metrics.csv"],
        "six contexts, four tasks, overall scores",
    )
    # The tuning baseline must reproduce the main-run TimesFM forecasts (same checkpoint, flags and windows).
    key = ["dataset", "window", "unique_id", "horizon"]
    tb = forecasts["tuning"].query('model == "timesfm_2p5_ctx1024"').set_index(key).sort_index()
    mb = (
        f[f.model.eq("timesfm_2p5") & f.dataset.isin(tb.index.get_level_values(0).unique())]
        .set_index(key)
        .sort_index()
    )
    pd.testing.assert_index_equal(tb.index, mb.index)
    np.testing.assert_array_equal(tb.y_true, mb.y_true)
    checks["tuning_baseline_max_abs_diff_vs_main_timesfm"] = float(
        np.max(np.abs(tb.y_pred.to_numpy() - mb.y_pred.to_numpy()))
    )
    selrows = [
        [
            NAMES[s["dataset"]],
            ("max. PAR" if s["dataset"] == "weather_h48" else esc(s["unique_id"])),
            s["rule"],
        ]
        + [f"{s['window0_smape'][m]:.2f}" for m in SHOWN]
        for s in selections
    ]
    table(
        tabs / "selections.tex",
        revised(
            "Diagnostic examples ranked by TimesFM window-0 sMAPE; all entries are per-series window-0 sMAPE ($\\downarrow$). Median means the middle sorted series (upper middle for even counts)."
        ),
        "tab:selections",
        ["Task", "ID", "Rule"] + [LABELS[m] for m in SHOWN],
        selrows,
    )
    record(
        tabs / "selections.tex",
        ["outputs/forecasts/all_datasets_all_models.parquet"],
        "window-0 per-series sMAPE recomputed from forecasts; ranked by TimesFM then string ID",
    )
    checks["feature_join_and_original_correlations_match"] = True
    (ev / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    source_hashes = {}
    for source in sorted(inputs):
        p = ROOT / source
        if p.exists():
            h = hashlib.sha256()
            with p.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            source_hashes[source] = h.hexdigest()
    codepaths = [
        "scripts/build_thesis_artifacts.py",
        *[
            f"src/timestransfer/{x}.py"
            for x in [
                "data",
                "features",
                "models",
                "runner",
                "metrics",
                "reporting",
                "series_features",
                "analysis",
                "metadata",
            ]
        ],
    ]
    for source in codepaths:
        source_hashes[source] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    (ev / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "command": ".venv/bin/python scripts/build_thesis_artifacts.py",
                "input_sha256": source_hashes,
                "artifacts": artifacts,
                "note": "Outputs are written into the thesis checkout; hashes identify present inputs.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(checks, indent=2))
    print(inv.to_string(index=False))
    print(json.dumps(selections, indent=2))


if __name__ == "__main__":
    main()
