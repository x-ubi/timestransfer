"""Verify generated thesis tables, provenance, and LaTeX links without fitting models.
Run after build_thesis_artifacts.py; optional --build-dir also checks the compiled thesis log.
"""

from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
from build_thesis_artifacts import (
    ROOT,
    TASKS,
    MODELS,
    NAMES,
    LABELS,
    FULL_NAMES,
    EXCLUDED,
    SKIPPED,
    MISSING,
    SHOWN,
    CORR_MODELS,
    N_WINDOWS,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--thesis", type=Path, default=Path("/home/ubuntu/thesis"))
p.add_argument("--evidence", type=Path, default=ROOT / "thesis-package/evidence_rolling_origin")
p.add_argument("--build-dir", type=Path)
a = p.parse_args()
thesis = a.thesis.resolve()
ev = a.evidence.resolve()
tabs = thesis / "tables/experiments"


def rows(path):
    lines = path.read_text().splitlines()
    return [
        [cell.strip() for cell in line[:-2].split("&")]
        for line in lines
        if line.endswith(r"\\") and "&" in line
    ]


def checkcell(cell, value, places, bold=None):
    if pd.isna(value):
        assert cell == MISSING, (cell, value)
        return
    if bold is not None:
        assert ("\\textbf{" in cell) == bold, (cell, value, bold)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    assert text == f"{value:.{places}f}", (text, value)


metrics = pd.read_csv(ROOT / "outputs/metrics/metrics.csv", dtype={"unique_id": str})
metrics = metrics[~metrics.model.isin(EXCLUDED)]
overall = metrics[metrics.scope.eq("overall")]
checked = 0
for metric, places in [("smape", 2), ("mase", 3), ("mae", 3), ("rmse", 3)]:
    pivot = overall.pivot(index="dataset", columns="model", values=metric).reindex(
        index=TASKS, columns=MODELS
    )
    for part, tasks in enumerate([TASKS[:6], TASKS[6:]], 1):
        rr = rows(tabs / f"{metric}_{part}.tex")
        if metric in ["mae", "rmse"]:
            assert rr[0] == ["Model"] + [NAMES[t] for t in tasks]
            assert len(rr) == 10
            for model, row in zip(MODELS, rr[1:], strict=True):
                assert row[0] == LABELS[model]
                for task, cell in zip(tasks, row[1:], strict=True):
                    value = pivot.loc[task, model]
                    checkcell(cell, value, places, value == pivot.loc[task].min())
                    checked += 1
        else:
            assert rr[0] == ["Task"] + [LABELS[m] for m in MODELS]
            assert len(rr) == len(tasks) + 1
            for task, row in zip(tasks, rr[1:], strict=True):
                assert row[0] == NAMES[task]
                for model, cell in zip(MODELS, row[1:], strict=True):
                    value = pivot.loc[task, model]
                    checkcell(cell, value, places, value == pivot.loc[task].min())
                    checked += 1
assert checked == 396
missing_cells = sum(
    cell == MISSING
    for m in ["smape", "mase", "mae", "rmse"]
    for part in (1, 2)
    for row in rows(tabs / f"{m}_{part}.tex")
    for cell in row
)
assert missing_cells == 4 * len(SKIPPED), missing_cells

# Aggregate, per-task relative MASE and window spread.
agg = pd.read_csv(ROOT / "outputs/metrics/aggregate_relative_scores.csv")
agg = agg[~agg.model.isin(EXCLUDED)].set_index("model")
rr = rows(tabs / "aggregate.tex")[1:]
assert len(rr) == 9
order = agg.sort_values("geomean_relative_mase").index.tolist()
for row, m in zip(rr, order, strict=True):
    assert row[0] == FULL_NAMES[m] and int(row[3]) == agg.loc[m, "n_tasks"]
    checkcell(row[1], agg.loc[m, "geomean_relative_mase"], 3)
    checkcell(row[2], agg.loc[m, "geomean_relative_mae"], 3)
assert agg.loc["auto_arima", "n_tasks"] == 8 and (agg.drop("auto_arima").n_tasks == 11).all()
rel = pd.read_csv(ROOT / "outputs/metrics/relative_to_seasonal_naive.csv")
rel = rel[~rel.model.isin(EXCLUDED)]
relp = rel.pivot(index="dataset", columns="model", values="relative_mase").reindex(
    index=TASKS, columns=[m for m in MODELS if m != "seasonal_naive"]
)
rr = rows(tabs / "relmase.tex")[1:]
assert len(rr) == len(TASKS) + 1
for task, row in zip(TASKS, rr[:-1], strict=True):
    assert row[0] == NAMES[task]
    for m, cell in zip(relp.columns, row[1:], strict=True):
        checkcell(cell, relp.loc[task, m], 3, relp.loc[task, m] == relp.loc[task].min())
for m, cell in zip(relp.columns, rr[-1][1:], strict=True):
    checkcell(cell, agg.loc[m, "geomean_relative_mase"], 3)
spread = pd.read_csv(ROOT / "outputs/metrics/window_spread.csv").set_index(["dataset", "model"])
nspread = 0
for part, tasks in enumerate([TASKS[:6], TASKS[6:]], 1):
    rr = rows(tabs / f"spread_{part}.tex")[1:]
    assert len(rr) == 9
    for m, row in zip(MODELS, rr, strict=True):
        assert row[0] == LABELS[m]
        for t, cell in zip(tasks, row[1:], strict=True):
            if (t, m) in SKIPPED:
                assert cell == MISSING
                continue
            s = spread.loc[(t, m)]
            assert s.n_windows == N_WINDOWS
            assert cell == f"${s.mase_mean:.3f}\\pm{s.mase_std:.3f}$", (cell, t, m)
            nspread += 1
assert nspread == 96

for metric in ["smape", "mase"]:
    cor = pd.read_csv(ev / f"correlations_{metric}.csv")
    rr = rows(tabs / f"correlations_{metric}.tex")[1:]
    assert len(rr) == 6
    for row, (task, feature) in zip(
        rr,
        [
            (t, f)
            for t in ["m4_hourly", "ecl_h48", "traffic_h48"]
            for f in ["acf_seasonal", "seasonal_strength"]
        ],
        strict=True,
    ):
        for cell, model in zip(row[3:], CORR_MODELS, strict=True):
            v = cor[(cor.dataset == task) & (cor.feature == feature) & (cor.model == model)].iloc[0]
            assert int(row[2]) == v.n_series
            checkcell(cell, v.spearman_rho, 3)

tuning = pd.read_csv(ROOT / "outputs/timesfm_tuning/metrics/metrics.csv")
tune = tuning[tuning.scope.eq("overall")].set_index(["dataset", "model"])
assert len(tune) == 44 and tuning[tuning.scope.eq("window")].shape[0] == 220
flags = [
    "no_normalize",
    "no_quantile_head",
    "no_flip_invariance",
    "no_infer_positive",
    "no_fix_crossing",
]
flagkeys = [
    "normalize_inputs",
    "use_continuous_quantile_head",
    "force_flip_invariance",
    "infer_is_positive",
    "fix_quantile_crossing",
]
tasks = ["m4_hourly", "ett_h1_h48", "ett_m1_h48", "weather_h48"]
contexts = [512, 1024, 2048, 4096, 8192, 16256]
env = json.loads((ROOT / "outputs/timesfm_tuning/metadata/environment.json").read_text())
for task in tasks:
    for ctx in contexts:
        v = env["model_statuses"][task][f"timesfm_2p5_ctx{ctx}"]
        assert v["max_context"] == ctx and v["status"] == "ok" and v["n_windows_ok"] == N_WINDOWS
        assert all(v["forecast_config"][k] for k in flagkeys)
        assert (
            v["forecast_config"]["max_horizon"] == 48
            and v["model_id"] == "google/timesfm-2.5-200m-pytorch"
        )
        assert v["revision"] == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    for flag, key in zip(flags, flagkeys, strict=True):
        v = env["model_statuses"][task][f"timesfm_2p5_{flag}"]["forecast_config"]
        assert v["max_context"] == 1024
        assert all(v[k] == (k != key) for k in flagkeys)
for metric, places in [("smape", 3), ("mase", 4)]:
    rr = rows(tabs / f"flags_{metric}.tex")[1:]
    assert len(rr) == 5
    for flag, row in zip(flags, rr, strict=True):
        for task, cell in zip(tasks, row[1:], strict=True):
            delta = (
                tune.loc[(task, "timesfm_2p5_" + flag), metric]
                - tune.loc[(task, "timesfm_2p5_ctx1024"), metric]
            )
            assert cell == f"{delta:+.{places}f}"
rr = rows(tabs / "contexts.tex")[1:]
assert len(rr) == 24
for row, (task, ctx) in zip(rr, [(t, c) for t in tasks for c in contexts], strict=True):
    assert row[:2] == [NAMES[task], f"{ctx:,}"]
    for cell, metric, places in zip(row[2:], ["smape", "mase"], [3, 4], strict=True):
        checkcell(cell, tune.loc[(task, f"timesfm_2p5_ctx{ctx}"), metric], places)

# Diagnostic examples: window-0 per-series sMAPE recomputed independently from the forecasts.
main_f = pd.read_parquet(ROOT / "outputs/forecasts/all_datasets_all_models.parquet")
w0 = main_f[main_f.window.eq(0)].copy()
err = (w0.y_pred - w0.y_true).abs()
w0["smape"] = 200 * err / (w0.y_true.abs() + w0.y_pred.abs() + 1e-8)
w0s = w0.groupby(["dataset", "model", "unique_id"]).smape.mean()
selections = json.loads((ev / "forecast_selections.json").read_text())
assert len(selections) == 6
for selection in selections:
    task = selection["dataset"]
    ref = (
        w0s.loc[(task, "timesfm_2p5")]
        .reset_index()
        .sort_values(["smape", "unique_id"], kind="stable")
    )
    assert ref.iloc[selection["rank_1based"] - 1].unique_id == selection["unique_id"]
    assert len(ref) == selection["n_series"]
    for model, value in selection["window0_smape"].items():
        np.testing.assert_allclose(
            w0s.loc[(task, model, selection["unique_id"])], value, rtol=1e-12
        )
    for model, value in selection["pooled_smape"].items():
        actual = metrics[
            (metrics.dataset == task)
            & (metrics.model == model)
            & (metrics.unique_id == selection["unique_id"])
            & (metrics.scope == "series")
        ].smape.iloc[0]
        assert actual == value
rr = rows(tabs / "selections.tex")[1:]
assert len(rr) == 6
for row, selection in zip(rr, selections, strict=True):
    for cell, model in zip(row[3:], SHOWN, strict=True):
        checkcell(cell, selection["window0_smape"][model], 2)

tune_f = pd.read_parquet(ROOT / "outputs/timesfm_tuning/forecasts/all_datasets_all_models.parquet")
key = ["dataset", "window", "unique_id", "horizon"]
reference = main_f[main_f.model.eq("timesfm_2p5")].set_index(key)
for model, g in tune_f.groupby("model"):
    indexed = g.set_index(key).sort_index()
    matched = reference.loc[indexed.index]
    np.testing.assert_array_equal(indexed.y_true, matched.y_true)
    assert indexed.ds.tolist() == matched.ds.tolist()
main_scales = pd.read_csv(
    ROOT / "outputs/metrics/mase_scales.csv", dtype={"unique_id": str}
).set_index(["dataset", "window", "unique_id"])
tune_scales = pd.read_csv(
    ROOT / "outputs/timesfm_tuning/metrics/mase_scales.csv", dtype={"unique_id": str}
).set_index(["dataset", "window", "unique_id"])
np.testing.assert_allclose(
    tune_scales.mase_scale, main_scales.loc[tune_scales.index].mase_scale, rtol=1e-12
)

# Check only rendered inputs; unrelated template samples are not included.
seen = set()
citations = set()
labels = []
references = set()
todos = {}


def visit(path):
    path = path.resolve()
    if path in seen:
        return
    assert path.exists(), path
    seen.add(path)
    text = "\n".join(re.split(r"(?<!\\)%", line)[0] for line in path.read_text().splitlines())
    n = len(re.findall(r"\\todo\b", text))
    if n:
        todos[str(path.relative_to(thesis))] = n
    assert "/home/ubuntu" not in text, path
    assert "tabpfn_ts_v3p5" not in text and "TabPFN-TS-3.5" not in text, path
    labels.extend(re.findall(r"\\label\{([^}]+)\}", text))
    for keys in re.findall(r"\\cite\w*(?:\[[^]]*\])?\{([^}]+)\}", text):
        citations.update(k.strip() for k in keys.split(","))
    for keys in re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", text):
        references.add(keys)
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        visit(thesis / (name if Path(name).suffix else name + ".tex"))
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        assert (thesis / name).exists() or (thesis / "img" / name).exists(), name


visit(thesis / "main.tex")
assert len(labels) == len(set(labels)), "duplicate labels"
assert references <= set(labels), references - set(labels)
bibkeys = set(re.findall(r"@\w+\{([^,]+),", (thesis / "bibliografia.bib").read_text()))
assert citations <= bibkeys, citations - bibkeys
manifest = json.loads((ev / "artifact_manifest.json").read_text())
for source, expected in manifest["input_sha256"].items():
    assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, source
for record in manifest["artifacts"]:
    assert Path(record["output"]).exists(), record["output"]

report = {
    "main_score_cells_and_winners_checked": checked,
    "missing_cells_for_skipped_autoarima": missing_cells,
    "aggregate_rows_checked": 9,
    "relative_mase_cells_checked": int(relp.notna().sum().sum()),
    "window_spread_cells_checked": nspread,
    "correlation_tables_checked": 2,
    "context_scores_checked": 48,
    "flag_deltas_checked": 40,
    "selection_ids_and_scores_checked": 6,
    "requested_contexts_and_flags_verified": 44,
    "rendered_tex_files": len(seen),
    "citations_resolved": len(citations),
    "todos_by_file": todos,
    "unique_labels": len(labels),
    "source_hashes_verified": len(manifest["input_sha256"]),
}
if a.build_dir:
    log = (a.build_dir / "main.log").read_text(errors="replace")
    assert (a.build_dir / "main.pdf").stat().st_size > 0
    for pattern in [
        "Undefined control sequence",
        "LaTeX Error",
        "There were undefined references",
        "Citation .* undefined",
        "Reference .* undefined",
        "Missing character",
    ]:
        assert not re.search(pattern, log), pattern
    report["build_pdf_exists_and_no_unresolved_references_or_missing_glyphs"] = True
    report["overfull_boxes"] = len(re.findall("Overfull", log))
(ev / "package_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
