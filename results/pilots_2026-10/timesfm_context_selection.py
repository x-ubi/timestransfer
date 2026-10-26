"""Pilot: choose TimesFM's context length per task from past windows (window k uses the
mean MASE of windows >k) on the Experiment 3 sweep (4 tasks, 5 windows); evaluated on
windows 0-3 against the fixed 1024 context and the ex-post best context."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import TUNING

f = pd.read_parquet(TUNING/"forecasts"/"all_datasets_all_models.parquet"); f["unique_id"] = f.unique_id.astype(str)
sc = pd.read_csv(TUNING/"metrics"/"mase_scales.csv"); sc["unique_id"] = sc.unique_id.astype(str)
ctx = sorted([m for m in f.model.unique() if "ctx" in m], key=lambda s: int(s.split("ctx")[1]))
f = f[f.model.isin(ctx)]
key = ["dataset","unique_id","window","horizon"]
w = f.pivot_table(index=key, columns="model", values="y_pred")
w["y"] = f.drop_duplicates(key).set_index(key).y_true
w = w.reset_index().merge(sc, on=["dataset","window","unique_id"])
A = pd.DataFrame({m: (w[m]-w.y).abs()/w.mase_scale for m in ctx}); A[["dataset","window"]] = w[["dataset","window"]]
tw = A.groupby(["dataset","window"]).mean()
rows = []
for d, g in tw.groupby(level=0):
    g = g.droplevel(0)
    for k in range(4):
        best = g.loc[k+1:4].mean().idxmin()
        rows.append((d, k, best, g.loc[k, best], g.loc[k, "timesfm_2p5_ctx1024"], g.loc[k].min()))
r = pd.DataFrame(rows, columns=["dataset","window","chosen","selected","ctx1024","oracle"])
print(r.to_string(index=False))
t = r.groupby("dataset")[["selected","ctx1024","oracle"]].mean(); t["selected/ctx1024"] = t.selected/t.ctx1024
print(t.round(3))
E = pd.DataFrame({m: (w[m]-w.y)/w.mase_scale for m in ctx}); E["dataset"] = w.dataset
for d, g in E.groupby("dataset"):
    c = g[ctx].corr().values; print(d, "mean error corr between context variants: %.2f" % c[np.triu_indices(len(ctx),1)].mean())
