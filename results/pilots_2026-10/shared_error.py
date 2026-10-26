"""Pilot: share of FM error variance in a common component, and whether a per-series bias
correction fitted on past windows (>k, min 3) reduces MASE of the equal-weight FM mean."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import FORECASTS, SCALES, FM

f = pd.read_parquet(FORECASTS, columns=["dataset","model","unique_id","window","horizon","y_true","y_pred"])
f = f[f.model.isin(FM)]; f["unique_id"] = f.unique_id.astype(str)
sc = pd.read_csv(SCALES); sc["unique_id"] = sc.unique_id.astype(str)
key = ["dataset","unique_id","window","horizon"]
w = f.pivot_table(index=key, columns="model", values="y_pred")
w["y"] = f[f.model == "chronos2"].set_index(key).y_true
w = w.reset_index().merge(sc, on=["dataset","window","unique_id"])
E = pd.DataFrame({m: (w[m]-w.y)/w.mase_scale for m in FM})
print("Share of member-error variance in the first principal component:")
for d, idx in w.groupby("dataset").groups.items():
    ev = np.linalg.eigvalsh(np.cov(E.loc[idx].values.T)); print(f"  {d:13s} {ev[-1]/ev.sum():.3f}")
w["c"] = w[FM].mean(1); w["e"] = (w.c-w.y)/w.mase_scale
b = w.groupby(["dataset","unique_id","window"]).e.mean().unstack()
print("Correlation of a series' mean scaled error between consecutive windows:")
for d, bb in b.groupby(level=0):
    cols = [k for k in bb.columns if bb[k].notna().all()]
    r = np.nanmean([np.corrcoef(bb[k], bb[k+1])[0,1] for k in cols if k+1 in cols])
    print(f"  {d:13s} {r:.2f}")
res = []
for d, g in w.groupby("dataset"):
    for k in sorted(g.window.unique()):
        past = g[g.window > k]
        if past.window.nunique() < 3: continue
        cur = g[g.window == k]; sb = past.groupby("unique_id").e.mean(); base = np.abs(cur.e).mean()
        for lam in (0.5, 1.0):
            corr = cur.c - lam*cur.unique_id.map(sb).values*cur.mase_scale
            res.append((d, f"bias_shift_{lam}", np.mean(np.abs(corr-cur.y)/cur.mase_scale), base))
res = pd.DataFrame(res, columns=["dataset","method","mase","base"])
t = res.groupby(["dataset","method"])[["mase","base"]].mean()
print("MASE ratio (corrected / uncorrected equal mean):"); print((t.mase/t.base).unstack().round(3))
