"""Registry of numbers and verbal quantitative claims in the revised thesis prose.

Every entry is recomputed from saved outputs, configs or recorded metadata:
  numeric entries: file, text (as written), expected (recomputed, formatted), anchors
  (substrings that must occur in the same sentence), source;
  claims: (file, description, bool) for statements made in words ("eight of the eleven").
Used by check_thesis_numbers.py; no forecasting model is fitted.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP, RES, APP = "tex/experiments.tex", "tex/results.tex", "tex/experiment-appendix.tex"
FM = ["timesfm_2p5", "chronos2", "tabpfn_ts"]


def f(v, places):
    return f"{v:.{places}f}"


def th(v):
    return f"{int(v):,}"


def constants(root: Path) -> dict[str, str]:
    """Structural, definitional or literature constants, each with its provenance."""
    cfg = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    tune = yaml.safe_load((root / "configs/timesfm_tuning.yaml").read_text())
    c = {}
    for k in range(cfg["project"]["n_windows"] + 1):
        c[str(k)] = "window index or count (config n_windows)"
    for d in cfg["datasets"]:
        c[str(d["horizon"])] = "horizon (config)"
        c[str(d["seasonality"])] = "seasonal lag (config)"
    c[th(cfg["models"]["auto_arima"]["max_season_length"])] = "AutoARIMA max_season_length (config)"
    for m in tune["models"].values():
        c[th(m["max_context"])] = "TimesFM context grid (tuning config)"
    c["200"] = "upper bound of the 0--200 sMAPE convention (definition)"
    c["10"] = "base of 10^-8 / 10^-5 (definition)"
    c["-8"] = "sMAPE epsilon exponent (definition)"
    c["2025"] = "publication year (bibliography)"
    c["2026"] = "publication year (bibliography)"
    c["12"] = (
        "tabpfn-time-series 1.3.0 AutoSeasonalFeature.Config.max_top_k; FPP3 Section 10.5 monthly period"
    )
    c["35"] = "AutoARIMA s=96 timing (configs/experiment.yaml comment, README)"
    c["6"] = "upper end of the author-provided 2--6 minute AutoARIMA timing (flagged with a todo)"
    c["130"] = (
        "about 130,000,000 synthetic datasets per TabPFN model (Hollmann et al., Nature 2025)"
    )
    c["370"] = "370 series of the Electricity entry in Table 6 of the Chronos-2 paper"
    return c


def registry(root: Path, ev: Path):
    cfg = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    env = json.loads((root / "outputs/metadata/environment.json").read_text())
    inv = pd.read_csv(ev / "dataset_inventory.csv").set_index("dataset")
    d = json.loads((ev / "derived_stats.json").read_text())
    models = cfg["models"]
    R, C = [], []

    def add(file, text, expected, anchors=(), source=""):
        R.append(
            {
                "file": file,
                "text": text,
                "expected": expected,
                "anchors": list(anchors),
                "source": source,
            }
        )

    def claim(file, description, ok):
        C.append({"file": file, "claim": description, "ok": bool(ok)})

    metrics = pd.read_csv(root / "outputs/metrics/metrics.csv", dtype={"unique_id": str})
    metrics = metrics[metrics.model != "tabpfn_ts_v3p5"]
    ov = metrics[metrics.scope.eq("overall")]
    mase = ov.pivot(index="dataset", columns="model", values="mase")
    smape = ov.pivot(index="dataset", columns="model", values="smape")
    rel = pd.read_csv(root / "outputs/metrics/relative_to_seasonal_naive.csv").pivot(
        index="dataset", columns="model", values="relative_mase"
    )
    agg = pd.read_csv(root / "outputs/metrics/aggregate_relative_scores.csv").set_index("model")
    rowcap = {x["features"]["train_row_cap"] for x in cfg["datasets"]}
    assert len(rowcap) == 1
    rowcap = rowcap.pop()
    assert all(r == rowcap for s in env["dataset"] for r in s["lag_feature_train_rows_per_window"])
    tab = models["tabpfn_ts"]["max_context_length"]
    chr_ = models["chronos2"]["context_length"]
    tfm = models["timesfm_2p5"]["max_context"]

    # ---------------- Section 3 ----------------
    m4 = inv.loc["m4_hourly"]
    add(
        EXP, "700", th(m4.train_min), ["window~0, the training part"], "dataset_inventory.train_min"
    )
    add(
        EXP, "960", th(m4.train_max), ["window~0, the training part"], "dataset_inventory.train_max"
    )
    add(
        EXP,
        "508",
        th(m4.train_window4_min),
        ["window~4, from"],
        "dataset_inventory.train_window4_min",
    )
    add(
        EXP,
        "768",
        th(m4.train_window4_max),
        ["window~4, from"],
        "dataset_inventory.train_window4_max",
    )
    add(
        EXP,
        "480",
        th(cfg["project"]["n_windows"] * 96),
        ["ETTh1/96 windows span the final"],
        "n_windows * H",
    )
    for anchors in (
        ["at most 200,000 rows per task and window"],
        ["200,000-row budget"],
        ["retains at most 200,000"],
    ):
        add(
            EXP,
            "200,000",
            th(rowcap),
            anchors,
            "config train_row_cap; metadata lag_feature_train_rows_per_window",
        )
    claim(
        EXP,
        "all tasks reach the row cap in all five windows",
        all(r == rowcap for s in env["dataset"] for r in s["lag_feature_train_rows_per_window"]),
    )
    add(EXP, "4,096", th(tab), ["last 4,096 observations"], "config tabpfn_ts.max_context_length")
    add(EXP, "8,192", th(chr_), ["context 8,192"], "config chronos2.context_length")
    add(EXP, "1,024", th(tfm), ["TimesFM receives"], "config timesfm_2p5.max_context")
    add(EXP, "8,192", th(chr_), ["Chronos-2 at most 8,192"], "config")
    add(EXP, "4,096", th(tab), ["TabPFN-TS at most 4,096"], "config")
    assert {v["chronos2"]["model_context_length"] for v in env["model_statuses"].values()} == {chr_}
    add(EXP, "8,192", th(chr_), ["maximum of 8,192"], "metadata chronos2.model_context_length")
    hf = json.loads(
        (
            Path.home()
            / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots"
            / models["chronos2"]["revision"]
            / "config.json"
        ).read_text()
    )
    cc = hf.get("chronos_config", hf)
    add(
        EXP,
        "1,024",
        th(cc["output_patch_size"] * cc["max_output_patches"]),
        ["maximum prediction length of 1,024"],
        "Chronos-2 config.json; model card",
    )
    add(EXP, "4,096", th(tab), ["most recent 4,096 training observations"], "config")
    add(
        EXP,
        "4,096",
        th(tab),
        ["4,096 most recent observations"],
        "TabPFN-TS v4 Appendix A.3; config",
    )
    add(
        EXP,
        "5,000",
        th(models["auto_arima"]["max_train_length"]),
        ["last 5,000 observations; no orders"],
        "config auto_arima.max_train_length",
    )
    add(
        EXP,
        "5,000",
        th(models["auto_arima"]["max_train_length"]),
        ["5,000-observation series"],
        "config",
    )
    add(EXP, "16", str(models["prophet"]["n_jobs"]), ["spawned workers"], "config prophet.n_jobs")
    add(EXP, "1,024", th(tfm), ["1,024 observations for TimesFM"], "config")
    add(EXP, "8,192", th(chr_), ["8,192 for Chronos-2"], "config")
    add(EXP, "4,096", th(tab), ["4,096 for TabPFN-TS"], "config")
    scales = pd.read_csv(root / "outputs/metrics/mase_scales.csv")
    add(EXP, "8,340", th(len(scales)), ["window-specific"], "mase_scales.csv rows")
    claim(
        EXP,
        "all scales positive and finite",
        (np.isfinite(scales.mase_scale) & (scales.mase_scale > 0)).all(),
    )
    add(
        EXP,
        "10,000",
        th(10000),
        ["its last 10,000 observations"],
        "analyze_series.py --max-points default",
    )
    add(EXP, "2", "2", ["2--6 minutes"], "author-provided timing (todo)")
    add(EXP, "24", "24", ["$s=24$"], "config seasonality of hourly tasks")
    add(EXP, "24", "24", ["24-hour cycle"], "definition of hourly sampling")
    sk = {
        (t, m)
        for t, dd in env["model_statuses"].items()
        for m, v in dd.items()
        if v["status"] == "skipped"
    }
    claim(
        EXP,
        "AutoARIMA skipped exactly on ETTm1, ETTm2, Weather",
        sk
        == {
            ("ett_m1_h48", "auto_arima"),
            ("ett_m2_h48", "auto_arima"),
            ("weather_h48", "auto_arima"),
        },
    )
    claim(
        EXP,
        "AutoARIMA uses only season_length kwargs",
        all(
            v["auto_arima"].get("model_kwargs", {"season_length": 0}).keys() <= {"season_length"}
            for v in env["model_statuses"].values()
            if v["auto_arima"]["status"] == "ok"
        ),
    )
    # ---------------- Appendix ----------------
    add(APP, "508", th(m4.train_window4_min), ["window~4, the earliest"], "dataset_inventory")
    add(APP, "768", th(m4.train_window4_max), ["window~4, the earliest"], "dataset_inventory")
    si = pd.read_csv(ev / "series_inventory.csv", dtype={"unique_id": str})
    s4 = si[si.dataset.eq("m4_hourly")]
    ends = sorted(set(s4.window4_test_end.astype(int)))
    add(APP, "556", th(ends[0]), ["test positions end at 556"], "series_inventory.window4_test_end")
    add(
        APP,
        "816",
        th(ends[-1]),
        ["test positions end at 556 and 816"],
        "series_inventory.window4_test_end",
    )
    add(APP, "200,000", th(rowcap), ["its row cap is 200,000"], "config")
    add(
        APP,
        "100",
        str(models["chronos2"]["batch_size"]),
        ["batches of 100"],
        "config; metadata chronos2.batch_size",
    )
    add(APP, "16", str(models["prophet"]["n_jobs"]), ["MAP estimation"], "config prophet.n_jobs")
    add(APP, "16", str(models["auto_ets"]["n_jobs"]), ["StatsForecast 2.0.3"], "config n_jobs")

    # ---------------- Results: coverage and aggregate ----------------
    add(
        RES,
        "96",
        str(d["coverage.scored_combinations"]),
        ["scored 96 of the 99"],
        "metrics overall rows",
    )
    add(RES, "99", str(11 * 9), ["96 of the 99"], "11 tasks x 9 models")
    add(RES, "321", th(inv.loc["ecl_h48"].n_series), ["All 321 ECL"], "inventory")
    add(RES, "862", th(inv.loc["traffic_h48"].n_series), ["862 Traffic series"], "inventory")
    add(
        RES,
        "3,609,600",
        th(d["coverage.main_forecast_rows"]),
        ["saved forecast rows"],
        "forecast parquet rows",
    )
    add(
        RES,
        "1,185,360",
        th(d["coverage.tuning_forecast_rows"]),
        ["Experiment~3 contains"],
        "tuning parquet rows",
    )
    tun = pd.read_csv(root / "outputs/timesfm_tuning/metrics/metrics.csv")
    tov = tun[tun.scope.eq("overall")].set_index(["dataset", "model"])
    add(RES, "44", str(len(tov)), ["44 successful combinations"], "tuning overall rows")
    A = lambda m: f(agg.loc[m, "geomean_relative_mase"], 3)
    add(RES, A("timesfm_2p5"), A("timesfm_2p5"), ["relative to SeasonalNaive (0.748)"], "aggregate")
    add(RES, A("chronos2"), A("chronos2"), ["Chronos-2 (0.785)"], "aggregate")
    add(RES, A("tabpfn_ts"), A("tabpfn_ts"), ["TabPFN-TS (0.840)"], "aggregate")
    add(RES, A("auto_arima"), A("auto_arima"), ["AutoARIMA (0.926)"], "aggregate")
    for m, name in [
        ("auto_theta", "AutoTheta"),
        ("auto_ets", "AutoETS"),
        ("prophet", "Prophet"),
        ("linear_regression", "linear regression"),
    ]:
        add(RES, A(m), A(m), [f"{name} ({A(m)})"], "aggregate")
    for m in FM:
        add(
            RES,
            f(d[f"agg8.relmase.{m}"], 3),
            f(d[f"agg8.relmase.{m}"], 3),
            ["same eight tasks"],
            "derived agg8",
        )
    claim(RES, "AutoARIMA n_tasks = 8", agg.loc["auto_arima", "n_tasks"] == 8)
    claim(
        RES,
        "AutoARIMA trails all three FMs on its eight tasks",
        all(d[f"agg8.relmase.{m}"] < d["agg8.relmase.auto_arima"] for m in FM),
    )
    claim(
        RES,
        "MAE-based aggregate gives identical order",
        agg.sort_values("geomean_relative_mase").index.tolist()
        == agg.sort_values("geomean_relative_mae").index.tolist(),
    )
    claim(
        RES,
        "only FMs and AutoARIMA below one",
        set(agg.index[agg.geomean_relative_mase < 1]) == set(FM) | {"auto_arima"},
    )
    # ---------------- Results: per-task picture ----------------
    M = lambda t, m: f(mase.loc[t, m], 3)
    RL = lambda t, m: f(rel.loc[t, m], 3)
    claim(RES, "pretrained model lowest MASE on 8 tasks", d["fm_mase_wins"] == 8)
    win = {t: d[f"winner_mase.{t}"] for t in mase.index}
    claim(
        RES,
        "MASE winners as stated",
        win
        == {
            "m4_hourly": "timesfm_2p5",
            "ett_h2_h48": "timesfm_2p5",
            "ili_h48": "timesfm_2p5",
            "ett_m2_h48": "chronos2",
            "ecl_h48": "chronos2",
            "traffic_h48": "chronos2",
            "weather_h48": "chronos2",
            "ett_m1_h48": "tabpfn_ts",
            "ett_h1_h48": "seasonal_naive",
            "ett_h1_h96": "auto_arima",
            "exchange_h48": "linear_regression",
        },
    )
    add(
        RES,
        M("ett_h1_h48", "seasonal_naive"),
        M("ett_h1_h48", "seasonal_naive"),
        ["no model beats SeasonalNaive (0.854)"],
        "mase",
    )
    add(
        RES,
        M("ett_h1_h48", "auto_arima"),
        M("ett_h1_h48", "auto_arima"),
        ["AutoARIMA (0.886)"],
        "mase",
    )
    add(
        RES,
        M("ett_h1_h48", "tabpfn_ts"),
        M("ett_h1_h48", "tabpfn_ts"),
        ["TabPFN-TS (0.888)"],
        "mase",
    )
    claim(
        RES,
        "closest to SNaive on ETTh1/48 are AutoARIMA then TabPFN-TS",
        mase.loc["ett_h1_h48"].drop("seasonal_naive").sort_values().index[:2].tolist()
        == ["auto_arima", "tabpfn_ts"],
    )
    add(
        RES,
        M("ett_h1_h96", "auto_arima"),
        M("ett_h1_h96", "auto_arima"),
        ["AutoARIMA leads with 0.865"],
        "mase",
    )
    add(
        RES,
        M("ett_h1_h96", "timesfm_2p5"),
        M("ett_h1_h96", "timesfm_2p5"),
        ["TimesFM (1.033)"],
        "mase",
    )
    add(
        RES,
        M("ett_h1_h96", "seasonal_naive"),
        M("ett_h1_h96", "seasonal_naive"),
        ["SeasonalNaive (0.986)"],
        "mase",
    )
    claim(RES, "every Exchange MASE exceeds five", mase.loc["exchange_h48"].dropna().min() > 5)
    add(
        RES,
        M("exchange_h48", "linear_regression"),
        M("exchange_h48", "linear_regression"),
        ["linear regression (5.349)"],
        "mase",
    )
    add(
        RES,
        M("exchange_h48", "chronos2"),
        M("exchange_h48", "chronos2"),
        ["Chronos-2 (5.374)"],
        "mase",
    )
    add(
        RES,
        M("exchange_h48", "prophet"),
        M("exchange_h48", "prophet"),
        ["Prophet (19.498)"],
        "mase",
    )
    claim(
        RES,
        "beats SNaive counts 10/9/9/6/4/3/2/1",
        [
            d[f"beats_snaive_tasks.{m}"]
            for m in [
                "chronos2",
                "timesfm_2p5",
                "tabpfn_ts",
                "auto_arima",
                "auto_theta",
                "auto_ets",
                "linear_regression",
                "prophet",
            ]
        ]
        == [10, 9, 9, 6, 4, 3, 2, 1],
    )
    # ---------------- Close pretrained models ----------------
    close = ["m4_hourly", "ett_h1_h48", "ett_h1_h96", "ett_h2_h48", "traffic_h48"]
    ratios = {t: d[f"fm_ratio_max_min.{t}"] for t in mase.index}
    add(
        RES,
        "14",
        "14" if max(ratios[t] for t in close) < 1.14 else "FAIL",
        ["less than 14"],
        "max/min FM MASE ratio on five tasks",
    )
    claim(
        RES,
        "ILI, Weather, ECL, ETTm1 have the four largest FM ratios",
        sorted(ratios, key=ratios.get)[-4:][::-1]
        == ["ili_h48", "weather_h48", "ecl_h48", "ett_m1_h48"],
    )
    add(
        RES,
        RL("ili_h48", "timesfm_2p5"),
        RL("ili_h48", "timesfm_2p5"),
        ["relative MASE of 0.275"],
        "relative",
    )
    add(
        RES,
        RL("ili_h48", "tabpfn_ts"),
        RL("ili_h48", "tabpfn_ts"),
        ["TabPFN-TS's 0.802"],
        "relative",
    )
    add(
        RES, RL("ili_h48", "chronos2"), RL("ili_h48", "chronos2"), ["Chronos-2's 0.842"], "relative"
    )
    claim(
        RES,
        "ILI ratio about a third",
        0.3 < rel.loc["ili_h48", "timesfm_2p5"] / rel.loc["ili_h48", "tabpfn_ts"] < 0.36
        and 0.3 < rel.loc["ili_h48", "timesfm_2p5"] / rel.loc["ili_h48", "chronos2"] < 0.36,
    )
    ww = metrics[metrics.scope.eq("window")].pivot_table(
        index=["dataset", "window"], columns="model", values="mase"
    )
    claim(
        RES,
        "TimesFM below both other FMs in every ILI window",
        (
            ww.loc["ili_h48", "timesfm_2p5"]
            < ww.loc["ili_h48", ["chronos2", "tabpfn_ts"]].min(axis=1)
        ).all(),
    )
    for m, txt in [
        ("chronos2", "0.780 for Chronos-2"),
        ("timesfm_2p5", "0.827 for TimesFM"),
        ("tabpfn_ts", "0.844 for TabPFN-TS"),
    ]:
        add(
            RES,
            f(d[f"loto.ili_h48.{m}"], 3),
            f(d[f"loto.ili_h48.{m}"], 3),
            [txt],
            "leave-ILI-out geometric mean",
        )
    claim(
        RES,
        "TimesFM first when omitting any other task",
        d["loto_n_timesfm_first"] == 10 and d["loto_order.ili_h48"][0] == "chronos2",
    )
    for t, order, vals_anchor in [
        (
            "weather_h48",
            ["chronos2", "tabpfn_ts", "timesfm_2p5"],
            "Chronos-2, TabPFN-TS, TimesFM (relative MASE",
        ),
        (
            "ecl_h48",
            ["chronos2", "timesfm_2p5", "tabpfn_ts"],
            "Chronos-2, TimesFM, TabPFN-TS (0.632",
        ),
    ]:
        claim(RES, f"FM order on {t}", rel.loc[t, FM].sort_values().index.tolist() == order)
        for m in order:
            add(RES, RL(t, m), RL(t, m), [vals_anchor], "relative")
    add(
        RES,
        RL("ett_m1_h48", "tabpfn_ts"),
        RL("ett_m1_h48", "tabpfn_ts"),
        ["TabPFN-TS (0.790)"],
        "relative",
    )
    add(
        RES,
        RL("ett_m1_h48", "chronos2"),
        RL("ett_m1_h48", "chronos2"),
        ["Chronos-2 (0.858)"],
        "relative",
    )
    add(
        RES,
        RL("ett_m1_h48", "timesfm_2p5"),
        RL("ett_m1_h48", "timesfm_2p5"),
        ["TimesFM (0.992)"],
        "relative",
    )
    claim(
        RES, "TimesFM barely below SNaive on ETTm1", 0.95 < rel.loc["ett_m1_h48", "timesfm_2p5"] < 1
    )
    add(RES, "1,024", th(tfm), ["1,024 observations span"], "config")
    add(RES, "8,192", th(chr_), ["8,192 observations available to Chronos-2"], "config")
    claim(
        RES,
        "durations: 1,024 obs ~ one week (10-min) and eleven days (15-min); 8,192 ~ eight and twelve weeks",
        round(1024 * 10 / 1440) == 7
        and round(1024 * 15 / 1440) == 11
        and round(8192 * 10 / 1440 / 7) == 8
        and round(8192 * 15 / 1440 / 7) == 12,
    )
    tw = tun[tun.scope.eq("overall")].set_index(["dataset", "model"])
    claim(
        RES,
        "longer TimesFM context lowers Weather MASE; shortest best on ETTm1",
        tw.loc[("weather_h48", "timesfm_2p5_ctx16256"), "mase"]
        < tw.loc[("weather_h48", "timesfm_2p5_ctx1024"), "mase"]
        and d["tuning.best_ctx.mase.ett_m1_h48"] == 512,
    )
    # ---------------- Metric disagreements ----------------
    claim(
        RES,
        "six disagreement tasks as listed",
        d["metric_disagreement_tasks"]
        == ["m4_hourly", "ett_h2_h48", "ett_m2_h48", "traffic_h48", "weather_h48", "exchange_h48"],
    )
    S = lambda t, m: f(smape.loc[t, m], 2)
    add(
        RES,
        S("weather_h48", "seasonal_naive"),
        S("weather_h48", "seasonal_naive"),
        ["lowest sMAPE (21.66"],
        "smape",
    )
    add(
        RES,
        S("weather_h48", "chronos2"),
        S("weather_h48", "chronos2"),
        ["45.27 for Chronos-2"],
        "smape",
    )
    claim(RES, "SNaive lowest Weather sMAPE", d["winner_smape.weather_h48"] == "seasonal_naive")
    add(
        RES,
        M("weather_h48", "seasonal_naive"),
        M("weather_h48", "seasonal_naive"),
        ["MASE of 0.386"],
        "mase",
    )
    add(
        RES,
        M("weather_h48", "chronos2"),
        M("weather_h48", "chronos2"),
        ["Chronos-2's 0.198"],
        "mase",
    )
    claim(
        RES,
        "almost twice",
        1.8 < mase.loc["weather_h48", "seasonal_naive"] / mase.loc["weather_h48", "chronos2"] < 2.0,
    )
    zs = d["weather.zero_inflated_test_zero_share_range"]
    add(RES, "56", f(100 * zs[0], 0), ["zero in 56--100"], "derived zero share")
    add(RES, "100", f(100 * zs[1], 0), ["zero in 56--100"], "derived zero share")
    a, b = d["weather.zero_inflated_snaive_smape_range"]
    add(RES, "0.00", f(a, 2), ["between 0.00 and 19.80"], "derived")
    add(RES, "19.80", f(b, 2), ["between 0.00 and 19.80"], "derived")
    a, b = d["weather.zero_inflated_chronos2_smape_range"]
    add(RES, "90.70", f(a, 2), ["90.70 to 192.50"], "derived")
    add(RES, "192.50", f(b, 2), ["90.70 to 192.50"], "derived")
    add(RES, "16", str(d["weather.n_series"] - 5), ["remaining 16 variables"], "derived")
    add(
        RES,
        "17.41",
        f(d["weather.other16_mean_smape.chronos2"], 2),
        ["17.41 against 24.98"],
        "derived",
    )
    add(
        RES,
        "24.98",
        f(d["weather.other16_mean_smape.seasonal_naive"], 2),
        ["17.41 against 24.98"],
        "derived",
    )
    add(RES, "20", str(d["weather.chronos2_lower_mase_series"]), ["20 of the 21 series"], "derived")
    add(RES, "21", str(d["weather.n_series"]), ["20 of the 21 series"], "derived")
    h = d["etth2.hull_smape"]
    add(RES, "83.85", f(h["seasonal_naive"], 2), ["83.85 for SeasonalNaive"], "series smape")
    add(RES, "163.47", f(h["chronos2"], 2), ["163.47 for Chronos-2"], "series smape")
    add(RES, "162.37", f(h["timesfm_2p5"], 2), ["162.37 for TimesFM"], "series smape")
    # ---------------- AutoARIMA, linear regression, Prophet ----------------
    claim(
        RES,
        "AutoARIMA second on ETTh1/48 behind SNaive",
        mase.loc["ett_h1_h48"].sort_values().index[:2].tolist() == ["seasonal_naive", "auto_arima"],
    )
    claim(
        RES,
        "AutoARIMA worse than SNaive exactly on ETTh1/48 and Traffic",
        set(rel.index[rel.auto_arima > 1]) == {"ett_h1_h48", "traffic_h48"},
    )
    add(
        RES,
        RL("m4_hourly", "auto_arima"),
        RL("m4_hourly", "auto_arima"),
        ["M4 (0.764)"],
        "relative",
    )
    add(RES, RL("ecl_h48", "auto_arima"), RL("ecl_h48", "auto_arima"), ["ECL (0.865)"], "relative")
    add(
        RES,
        RL("traffic_h48", "auto_arima"),
        RL("traffic_h48", "auto_arima"),
        ["Traffic (1.076)"],
        "relative",
    )
    add(
        RES,
        A("linear_regression"),
        A("linear_regression"),
        ["highest aggregate relative MASE (1.757)"],
        "aggregate",
    )
    claim(
        RES, "linear highest aggregate", agg.geomean_relative_mase.idxmax() == "linear_regression"
    )
    add(
        RES,
        RL("weather_h48", "linear_regression"),
        RL("weather_h48", "linear_regression"),
        ["Weather (10.958)"],
        "relative",
    )
    add(
        RES,
        RL("m4_hourly", "linear_regression"),
        RL("m4_hourly", "linear_regression"),
        ["M4 (7.612)"],
        "relative",
    )
    claim(
        RES,
        "Weather and M4 are linear regression's largest relative scores",
        rel.linear_regression.sort_values().index[-2:].tolist() == ["m4_hourly", "weather_h48"],
    )
    add(RES, "990", f(d["linear.weather.p (mbar).a"], 0), [r"a_i\approx990"], "linear_scaling.csv")
    top = d["linear.weather_top3_ids"]
    claim(
        RES, "Weather top three are p, Tpot, rho", top == ["p (mbar)", "Tpot (K)", "rho (g/m**3)"]
    )
    qa = [d[f"linear.weather.{u}.q_over_a"] for u in top]
    add(RES, "0.005", f(min(qa), 3), ["between 0.005 and 0.012"], "linear_scaling.csv")
    add(RES, "0.012", f(max(qa), 3), ["between 0.005 and 0.012"], "linear_scaling.csv")
    add(
        RES,
        "75",
        f(100 * d["linear.weather_top3_share"], 0),
        ["75\\% of the summed"],
        "linear_scaling.csv",
    )
    add(
        RES,
        "35.931",
        f(d["linear.weather.p (mbar).mase"], 3),
        ["pressure alone has a MASE of 35.931"],
        "series mase",
    )
    add(
        RES,
        "-0.919",
        f(d["linear.spearman_qa_ratio.m4_hourly"], 3),
        ["Spearman correlation of"],
        "linear_scaling.csv",
    )
    add(RES, "414", str(d["linear.n_series.m4_hourly"]), ["over the 414 series"], "inventory")
    add(RES, "255.945", f(d["linear.m4_max_series_mase"], 3), ["255.945 for H187"], "series mase")
    claim(
        RES,
        "H187 is the max and q/a < 0.001",
        d["linear.m4_max_series"] == "H187" and d["linear.m4_max_series_q_over_a"] < 0.001,
    )
    add(RES, "0.001", "0.001", ["below 0.001"], "bound (claim)")
    add(
        RES,
        "402",
        str(d["linear.worse_than_snaive.m4_hourly"]),
        ["402 of the 414 M4 series"],
        "linear_scaling.csv",
    )
    add(RES, "414", str(d["linear.n_series.m4_hourly"]), ["402 of the 414 M4 series"], "inventory")
    add(
        RES,
        "19",
        str(d["linear.worse_than_snaive.weather_h48"]),
        ["19 of the 21 Weather series"],
        "linear_scaling.csv",
    )
    add(
        RES,
        "21",
        str(d["linear.n_series.weather_h48"]),
        ["19 of the 21 Weather series"],
        "inventory",
    )
    medqa = {k.split(".")[-1]: v for k, v in d.items() if k.startswith("linear.median_q_over_a.")}
    claim(RES, "Exchange has the smallest q/a ratios", min(medqa, key=medqa.get) == "exchange_h48")
    claim(RES, "Prophet below SNaive only on ECL", set(rel.index[rel.prophet < 1]) == {"ecl_h48"})
    add(
        RES, RL("ecl_h48", "prophet"), RL("ecl_h48", "prophet"), ["relative MASE 0.780"], "relative"
    )
    add(
        RES,
        RL("exchange_h48", "prophet"),
        RL("exchange_h48", "prophet"),
        ["Exchange (3.084)"],
        "relative",
    )
    add(
        RES,
        RL("ett_m1_h48", "prophet"),
        RL("ett_m1_h48", "prophet"),
        ["ETTm1/48 (2.215)"],
        "relative",
    )
    claim(
        RES,
        "Exchange and ETTm1 are Prophet's largest relative scores",
        rel.prophet.sort_values().index[-2:].tolist() == ["ett_m1_h48", "exchange_h48"],
    )
    # ---------------- Window variability ----------------
    add(RES, "96", str(d["cv.n_pairs"]), ["96 task--model pairs"], "window_spread rows")
    add(RES, "0.31", f(d["cv.median"], 2), ["is 0.31"], "window_spread")
    add(RES, "0.015", f(d["cv.min"], 3), ["from 0.015 to 1.066"], "window_spread")
    add(RES, "1.066", f(d["cv.max"], 3), ["from 0.015 to 1.066"], "window_spread")
    cvt = {k.split(".")[-1]: v for k, v in d.items() if k.startswith("cv.median_task.")}
    add(RES, "0.041", f(cvt["m4_hourly"], 3), ["median over models 0.041"], "window_spread")
    add(RES, "0.645", f(cvt["ecl_h48"], 3), ["ECL (0.645)"], "window_spread")
    add(RES, "0.448", f(cvt["ili_h48"], 3), ["ILI (0.448)"], "window_spread")
    claim(
        RES,
        "M4 smallest, ECL and ILI largest per-task CV",
        sorted(cvt, key=cvt.get)[0] == "m4_hourly"
        and sorted(cvt, key=cvt.get)[-2:] == ["ili_h48", "ecl_h48"],
    )
    add(RES, "414", str(int(inv.loc["m4_hourly"].n_series)), ["414 series average"], "inventory")
    add(RES, "25", str(d["windows.pooled_winner_wins"]), ["only 25 of the 55"], "window winners")
    add(RES, "55", str(d["windows.n_task_windows"]), ["only 25 of the 55"], "window winners")
    claim(
        RES,
        "ILI only task with same leader in all windows",
        d["windows.tasks_pooled_winner_all5"] == ["ili_h48"],
    )
    claim(
        RES,
        "four tasks have four different leaders",
        sum(v == 4 for v in d["windows.n_distinct_winners_per_task"].values()) == 4,
    )
    claim(
        RES,
        "pooled FM leader best in all five windows only on Exchange and ILI",
        d["windows.tasks_pooled_fm_all5"] == ["exchange_h48", "ili_h48"],
    )
    add(RES, "6", str(d["windows.snaive_lowest"]), ["6 of the 55"], "window winners")
    add(RES, "55", str(d["windows.n_task_windows"]), ["6 of the 55"], "window winners")
    claim(
        RES,
        "window-0-only winner differs on M4, ETTh2, ETTm1, Weather",
        d["windows.window0_winner_differs"]
        == ["m4_hourly", "ett_h2_h48", "ett_m1_h48", "weather_h48"],
    )
    # ---------------- Horizon ----------------
    hz = metrics[metrics.scope.eq("horizon")].copy()
    hz["horizon"] = hz.horizon.astype(float).astype(int)
    P = {
        t: hz[hz.dataset.eq(t)].pivot(index="horizon", columns="model", values="smape")
        for t in ["m4_hourly", "ett_h1_h96", "traffic_h48", "weather_h48"]
    }
    p = P["m4_hourly"]
    fm = p[FM]
    add(
        RES, "2.82", f((fm.max(axis=1) - fm.min(axis=1)).max(), 2), ["within 2.82"], "horizon scope"
    )
    add(RES, "11.02", f(fm.iloc[-1].min(), 2), ["between 11.02 and 13.27"], "horizon")
    add(RES, "13.27", f(fm.iloc[-1].max(), 2), ["between 11.02 and 13.27"], "horizon")
    add(RES, "9.91", f(p.seasonal_naive.min(), 2), ["SeasonalNaive (9.91 to 20.11)"], "horizon")
    add(RES, "20.11", f(p.seasonal_naive.max(), 2), ["SeasonalNaive (9.91 to 20.11)"], "horizon")
    add(RES, "10.00", f(p.prophet.min(), 2), ["Prophet (10.00 to 29.32)"], "horizon")
    add(RES, "29.32", f(p.prophet.max(), 2), ["Prophet (10.00 to 29.32)"], "horizon")
    claim(
        RES, "M4 FM curves rise to their maximum at step 48", all(fm[m].idxmax() == 48 for m in FM)
    )
    p = P["ett_h1_h96"]
    add(
        RES,
        "48",
        str(int((p.seasonal_naive < p[FM].min(axis=1)).sum())),
        ["at 48 of the 96 steps"],
        "horizon",
    )
    p = P["traffic_h48"]
    add(RES, "69.41", f(p.prophet.iloc[0], 2), ["first step (69.41)"], "horizon")
    add(RES, "99.13", f(p.prophet.max(), 2), ["peaks at 99.13"], "horizon")
    add(RES, "82.48", f(p.seasonal_naive.max(), 2), ["peak of 82.48"], "horizon")
    add(RES, "30", str(int(p.seasonal_naive.idxmax())), ["at step 30"], "horizon")
    add(RES, "22.40", f(p[FM].min().min(), 2), ["between 22.40 and 49.80"], "horizon")
    add(RES, "49.80", f(p[FM].max().max(), 2), ["between 22.40 and 49.80"], "horizon")
    p = P["weather_h48"]
    add(RES, "15.56", f(p.seasonal_naive.min(), 2), ["between 15.56 and 27.61"], "horizon")
    add(RES, "27.61", f(p.seasonal_naive.max(), 2), ["between 15.56 and 27.61"], "horizon")
    add(
        RES,
        "47",
        str(int((p.seasonal_naive < p[FM].min(axis=1)).sum())),
        ["47 of the 48 steps"],
        "horizon",
    )
    add(RES, "50.13", f(p[FM].iloc[-1].min(), 2), ["between 50.13 and 57.16"], "horizon")
    add(RES, "57.16", f(p[FM].iloc[-1].max(), 2), ["between 50.13 and 57.16"], "horizon")
    claim(RES, "Weather windows start eight hours apart", 48 * 10 == 8 * 60)
    # ---------------- Qualitative examples ----------------
    sel = json.loads((ev / "forecast_selections.json").read_text())
    W = lambda i, m: f(sel[i]["window0_smape"][m], 2)
    add(RES, "960", sel[0]["origin"], ["integer position 960"], "selection origin")
    add(RES, "414", str(sel[0]["n_series"]), ["rank 1/414"], "selection")
    add(RES, "414", str(sel[1]["rank_1based"]), ["rank 414/414"], "selection")
    add(RES, "96", str(sel[0]["history_points"]), ["96 historical observations shown"], "selection")
    add(RES, "96", str(sel[2]["history_points"]), ["Both show 96 historical"], "selection")
    add(
        RES,
        W(0, "seasonal_naive"),
        W(0, "seasonal_naive"),
        ["SeasonalNaive obtains 0.09"],
        "selection",
    )
    add(RES, W(0, "tabpfn_ts"), W(0, "tabpfn_ts"), ["TabPFN-TS 0.10"], "selection")
    add(RES, W(0, "timesfm_2p5"), W(0, "timesfm_2p5"), ["0.15 for TimesFM"], "selection")
    add(RES, W(1, "timesfm_2p5"), W(1, "timesfm_2p5"), ["TimesFM's sMAPE is 84.71"], "selection")
    add(RES, W(1, "prophet"), W(1, "prophet"), ["Prophet's 102.42"], "selection")
    pooled = sel[1]["pooled_smape"]
    add(RES, "30.66", f(min(pooled.values()), 2), ["between 30.66 and 44.12"], "series scope")
    add(RES, "44.12", f(max(pooled.values()), 2), ["between 30.66 and 44.12"], "series scope")
    claim(
        RES,
        "H349 pooled lower than window 0 for every displayed model",
        all(sel[1]["pooled_smape"][m] < sel[1]["window0_smape"][m] for m in pooled),
    )
    add(RES, "4", str(sel[2]["rank_1based"]), ["HUFL (rank 4/7)"], "selection")
    add(RES, "7", str(sel[2]["n_series"]), ["HUFL (rank 4/7)"], "selection")
    add(RES, "432", str(sel[3]["rank_1based"]), ["rank 432/862"], "selection")
    add(RES, "862", str(sel[3]["n_series"]), ["rank 432/862"], "selection")
    for m, name in [
        ("seasonal_naive", "SeasonalNaive"),
        ("timesfm_2p5", "TimesFM"),
        ("tabpfn_ts", "TabPFN-TS"),
        ("chronos2", "Chronos-2"),
        ("prophet", "Prophet"),
    ]:
        add(RES, W(2, m), W(2, m), ["follows the repeated daily continuation"], "selection")
        add(RES, W(3, m), W(3, m), ["Traffic ID 193"], "selection")
    claim(
        RES,
        "Traffic 193 order TimesFM, SNaive, TabPFN-TS, Chronos-2, Prophet",
        sorted(sel[3]["window0_smape"], key=sel[3]["window0_smape"].get)
        == ["timesfm_2p5", "seasonal_naive", "tabpfn_ts", "chronos2", "prophet"],
    )
    add(RES, "21", str(sel[4]["rank_1based"]), ["rank 21/21"], "selection")
    add(RES, "21", str(sel[4]["n_series"]), ["rank 21/21"], "selection")
    add(RES, "432", str(sel[4]["history_points"]), ["432 training observations shown"], "selection")
    add(RES, "4", str(sel[5]["rank_1based"]), ["24 (rank 4/7)"], "selection")
    add(RES, "7", str(sel[5]["n_series"]), ["24 (rank 4/7)"], "selection")
    add(RES, "156", str(sel[5]["history_points"]), ["156 training observations shown"], "selection")
    for m in ["seasonal_naive", "timesfm_2p5", "tabpfn_ts", "chronos2"]:
        add(
            RES,
            W(4, m),
            W(4, m),
            ["obtain 158.46"] if m != "seasonal_naive" else ["SeasonalNaive obtains 3.79"],
            "selection",
        )
    for m in ["timesfm_2p5", "chronos2", "tabpfn_ts", "seasonal_naive", "prophet"]:
        add(
            RES,
            W(5, m),
            W(5, m),
            ["9.95 sMAPE"] if m == "timesfm_2p5" else ["obtain 75.17"],
            "selection",
        )
    claim(RES, "Chronos-2 leads Weather MASE", win["weather_h48"] == "chronos2")
    # ---------------- Experiment 2 ----------------
    feats = pd.read_csv(root / "outputs/analysis/series_features.csv")
    joined = pd.read_csv(root / "outputs/analysis/series_features_with_metrics.csv")
    add(RES, "14,977", th(len(joined)), ["14,977 feature--model rows"], "outputs/analysis")
    claim(RES, "1,668 entries", len(feats) == 1668)
    claim(
        RES,
        "joined grid incomplete only by skipped AutoARIMA",
        len(joined)
        == len(feats) * 9
        - sum(inv.loc[t].n_series for t in ["ett_m1_h48", "ett_m2_h48", "weather_h48"]),
    )
    cs = pd.read_csv(ev / "correlations_smape.csv")
    cm = pd.read_csv(ev / "correlations_mase.csv")

    def rho(c, t, feat, m):
        return float(
            c[(c.dataset == t) & (c.feature == feat) & (c.model == m)].spearman_rho.iloc[0]
        )

    add(
        RES,
        "-0.851",
        f(rho(cs, "m4_hourly", "acf_seasonal", "timesfm_2p5"), 3),
        ["TimesFM has $\\rho=-0.851"],
        "correlations",
    )
    add(
        RES,
        "-0.887",
        f(rho(cs, "m4_hourly", "seasonal_strength", "timesfm_2p5"), 3),
        ["$-0.887$ for STL"],
        "correlations",
    )
    others = [
        rho(cs, "m4_hourly", ft, m)
        for ft in ["acf_seasonal", "seasonal_strength"]
        for m in ["chronos2", "tabpfn_ts", "seasonal_naive"]
    ]
    add(RES, "-0.836", f(max(others), 3), ["lie between"], "correlations")
    add(RES, "-0.884", f(min(others), 3), ["lie between"], "correlations")
    add(
        RES,
        "-0.098",
        f(rho(cs, "ecl_h48", "acf_seasonal", "timesfm_2p5"), 3),
        ["seasonal-ACF correlations of TimesFM"],
        "correlations",
    )
    add(
        RES,
        "-0.154",
        f(rho(cs, "traffic_h48", "acf_seasonal", "timesfm_2p5"), 3),
        ["seasonal-ACF correlations of TimesFM"],
        "correlations",
    )
    add(
        RES,
        "0.097",
        f(rho(cs, "ecl_h48", "acf_seasonal", "tabpfn_ts"), 3),
        ["0.097 for seasonal ACF"],
        "correlations",
    )
    add(
        RES,
        "0.134",
        f(rho(cs, "ecl_h48", "seasonal_strength", "tabpfn_ts"), 3),
        ["0.134 for STL"],
        "correlations",
    )
    add(
        RES,
        "0.017",
        f(rho(cm, "m4_hourly", "seasonal_strength", "timesfm_2p5"), 3),
        ["0.017 for TimesFM"],
        "correlations",
    )
    add(
        RES,
        "-0.021",
        f(rho(cm, "m4_hourly", "seasonal_strength", "tabpfn_ts"), 3),
        ["for TabPFN-TS"],
        "correlations",
    )
    add(
        RES,
        "0.493",
        f(rho(cm, "m4_hourly", "seasonal_strength", "chronos2"), 3),
        ["0.493 for Chronos-2"],
        "correlations",
    )
    ecl = [
        rho(cm, "ecl_h48", ft, m)
        for ft in ["acf_seasonal", "seasonal_strength"]
        for m in ["timesfm_2p5", "chronos2", "tabpfn_ts", "seasonal_naive"]
    ]
    ecl_stl = [
        rho(cm, "ecl_h48", "seasonal_strength", m)
        for m in ["timesfm_2p5", "chronos2", "tabpfn_ts", "seasonal_naive"]
    ]
    claim(RES, "all ECL MASE correlations positive", min(ecl) > 0)
    add(RES, "0.577", f(min(ecl_stl), 3), ["between 0.577 and 0.710"], "correlations")
    add(RES, "0.710", f(max(ecl_stl), 3), ["between 0.577 and 0.710"], "correlations")
    # ---------------- Experiment 3 ----------------
    T = lambda t, c, metric, pl: f(tw.loc[(t, f"timesfm_2p5_ctx{c}"), metric], pl)
    add(
        RES,
        T("m4_hourly", 512, "smape", 3),
        T("m4_hourly", 512, "smape", 3),
        ["lowers sMAPE from 8.442"],
        "tuning",
    )
    add(
        RES,
        T("m4_hourly", 1024, "smape", 3),
        T("m4_hourly", 1024, "smape", 3),
        ["to 8.327"],
        "tuning",
    )
    add(
        RES,
        T("m4_hourly", 512, "mase", 4),
        T("m4_hourly", 512, "mase", 4),
        ["MASE from 0.7201"],
        "tuning",
    )
    add(
        RES,
        T("m4_hourly", 1024, "mase", 4),
        T("m4_hourly", 1024, "mase", 4),
        ["to 0.7121"],
        "tuning",
    )
    claim(
        RES,
        "M4 contexts >= 1,024 identical at displayed precision",
        all(
            T("m4_hourly", c, "smape", 3) == T("m4_hourly", 1024, "smape", 3)
            and T("m4_hourly", c, "mase", 4) == T("m4_hourly", 1024, "mase", 4)
            for c in [2048, 4096, 8192, 16256]
        ),
    )
    claim(RES, "no M4 training part exceeds 960", inv.loc["m4_hourly"].train_max == 960)
    add(
        RES,
        T("ett_h1_h48", 1024, "smape", 3),
        T("ett_h1_h48", 1024, "smape", 3),
        ["(42.156 sMAPE"],
        "tuning",
    )
    add(
        RES,
        T("ett_h1_h48", 1024, "mase", 4),
        T("ett_h1_h48", 1024, "mase", 4),
        ["0.8983 MASE"],
        "tuning",
    )
    claim(
        RES,
        "ETTh1 best at 1,024 for both and all longer contexts worse",
        d["tuning.best_ctx.smape.ett_h1_h48"] == 1024
        and d["tuning.best_ctx.mase.ett_h1_h48"] == 1024,
    )
    add(
        RES,
        T("ett_h1_h48", 16256, "smape", 3),
        T("ett_h1_h48", 16256, "smape", 3),
        ["to 44.162 and 0.9552"],
        "tuning",
    )
    add(
        RES,
        T("ett_h1_h48", 16256, "mase", 4),
        T("ett_h1_h48", 16256, "mase", 4),
        ["to 44.162 and 0.9552"],
        "tuning",
    )
    add(
        RES,
        T("ett_m1_h48", 512, "smape", 3),
        T("ett_m1_h48", 512, "smape", 3),
        ["(24.348 and 0.4030)"],
        "tuning",
    )
    add(
        RES,
        T("ett_m1_h48", 512, "mase", 4),
        T("ett_m1_h48", 512, "mase", 4),
        ["(24.348 and 0.4030)"],
        "tuning",
    )
    claim(
        RES,
        "ETTm1 best at 512 for both",
        d["tuning.best_ctx.smape.ett_m1_h48"] == 512
        and d["tuning.best_ctx.mase.ett_m1_h48"] == 512,
    )
    add(
        RES,
        T("ett_m1_h48", 1024, "smape", 3),
        T("ett_m1_h48", 1024, "smape", 3),
        ["28.473 and 0.5541"],
        "tuning",
    )
    add(
        RES,
        T("ett_m1_h48", 1024, "mase", 4),
        T("ett_m1_h48", 1024, "mase", 4),
        ["28.473 and 0.5541"],
        "tuning",
    )
    add(
        RES,
        T("weather_h48", 1024, "mase", 4),
        T("weather_h48", 1024, "mase", 4),
        ["MASE falls from 0.2661"],
        "tuning",
    )
    add(
        RES,
        T("weather_h48", 16256, "mase", 4),
        T("weather_h48", 16256, "mase", 4),
        ["0.1844 at 16,256"],
        "tuning",
    )
    add(
        RES,
        T("weather_h48", 8192, "smape", 3),
        T("weather_h48", 8192, "smape", 3),
        ["minimum of 33.609"],
        "tuning",
    )
    claim(
        RES,
        "Weather MASE best at 16,256 and sMAPE at 8,192",
        d["tuning.best_ctx.mase.weather_h48"] == 16256
        and d["tuning.best_ctx.smape.weather_h48"] == 8192,
    )
    claim(
        RES,
        "three distinct window-best contexts on ETTh1, ETTm1, Weather",
        all(
            d[f"tuning.window_best_ctx_distinct.{t}"] == 3
            for t in ["ett_h1_h48", "ett_m1_h48", "weather_h48"]
        ),
    )
    fl = pd.read_csv(ev / "flag_deltas.csv").set_index(["dataset", "flag"])
    D = lambda t, fg, metric, pl: f(abs(fl.loc[(t, fg), metric + "_delta"]), pl)
    add(
        RES,
        D("m4_hourly", "no_flip_invariance", "mase", 4),
        D("m4_hourly", "no_flip_invariance", "mase", 4),
        ["by 0.0218 on M4"],
        "flag_deltas",
    )
    add(
        RES,
        D("ett_h1_h48", "no_flip_invariance", "mase", 4),
        D("ett_h1_h48", "no_flip_invariance", "mase", 4),
        ["0.0035 on ETTh1/48"],
        "flag_deltas",
    )
    add(
        RES,
        D("ett_m1_h48", "no_flip_invariance", "mase", 4),
        D("ett_m1_h48", "no_flip_invariance", "mase", 4),
        ["lowers it by 0.0515"],
        "flag_deltas",
    )
    add(
        RES,
        D("weather_h48", "no_flip_invariance", "smape", 3),
        D("weather_h48", "no_flip_invariance", "smape", 3),
        ["lowers sMAPE by 2.620"],
        "flag_deltas",
    )
    add(
        RES,
        D("weather_h48", "no_flip_invariance", "mase", 4),
        D("weather_h48", "no_flip_invariance", "mase", 4),
        ["raises MASE by 0.0060"],
        "flag_deltas",
    )
    sg = {(t, fg): np.sign(fl.loc[(t, fg), "mase_delta"]) for t, fg in fl.index}
    claim(
        RES,
        "flip signs: +M4, +ETTh1, -ETTm1, Weather sMAPE - / MASE +",
        sg[("m4_hourly", "no_flip_invariance")] > 0
        and sg[("ett_h1_h48", "no_flip_invariance")] > 0
        and sg[("ett_m1_h48", "no_flip_invariance")] < 0
        and fl.loc[("weather_h48", "no_flip_invariance"), "smape_delta"] < 0
        and sg[("weather_h48", "no_flip_invariance")] > 0,
    )
    claim(
        RES,
        "flip window agreement 5/5/4/4",
        d["tuning.flip_mase_worse_windows.m4_hourly"] == 5
        and d["tuning.flip_mase_worse_windows.ett_m1_h48"] == 0
        and d["tuning.flip_mase_worse_windows.ett_h1_h48"] == 4
        and d["tuning.flip_mase_worse_windows.weather_h48"] == 4,
    )
    pos = fl.xs("no_infer_positive", level="flag")
    claim(
        RES,
        "positive inference changes only Weather",
        (pos.drop("weather_h48")[["smape_delta", "mase_delta"]] == 0).all().all(),
    )
    add(
        RES,
        D("weather_h48", "no_infer_positive", "smape", 3),
        D("weather_h48", "no_infer_positive", "smape", 3),
        ["raises sMAPE by 2.024"],
        "flag_deltas",
    )
    mpos = fl.loc[("weather_h48", "no_infer_positive"), "mase_delta"]
    add(RES, "8.2", f(mpos / 1e-5, 1), ["8.2\\times10^{-5}"], "flag_deltas")
    add(RES, "-5", "-5", ["8.2\\times10^{-5}"], "exponent")
    add(RES, "0.0001", f(mpos, 4), ["rounds to"], "flag_deltas")
    norm = fl.xs("no_normalize", level="flag")
    claim(RES, "normalization sMAPE differences below 1e-5", norm.smape_delta.abs().max() < 1e-5)
    add(RES, "1", "1", ["below $1\\times10^{-5}$"], "bound")
    add(RES, "-5", "-5", ["below $1\\times10^{-5}$"], "exponent")
    for fg in ["no_quantile_head", "no_fix_crossing"]:
        claim(
            RES,
            f"{fg} exactly equal",
            (fl.xs(fg, level="flag")[["smape_delta", "mase_delta"]] == 0).all().all(),
        )
    claim(
        RES,
        "flip invariance largest absolute MASE effect overall",
        fl.mase_delta.abs().groupby(level="flag").sum().idxmax() == "no_flip_invariance",
    )
    return R, C
