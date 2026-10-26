"""Summarise corruption.py: task MASE per condition, degradation vs clean, combination
gain (equal mean / median vs best single member) and error correlation per condition."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import OUT

pd.set_option("display.width", 220)
r = pd.read_parquet(OUT/"corruption.parquet")
FM = ["timesfm","chronos2","tabpfn"]; key = ["dataset","window","kind","unique_id","horizon"]
w = r.pivot_table(index=key, columns="model", values="yhat").join(r.drop_duplicates(key).set_index(key)[["y","scale"]]).dropna()
w["mean3"] = w[FM].mean(1); w["median3"] = w[FM].median(1); M = FM+["mean3","median3"]
A = pd.DataFrame({m: (w[m]-w.y).abs()/w.scale for m in M}, index=w.index)
t = A.groupby(level=["dataset","kind"]).mean(); print("Task MASE by condition"); print(t.round(3))
clean = t.xs("clean", level="kind"); print("Degradation vs clean"); print((t/clean.reindex(t.index, level="dataset")).round(3))
best = t[FM].min(1)
print("Equal mean / best single member"); print((t.mean3/best).unstack().round(3))
print("Median / best single member"); print((t.median3/best).unstack().round(3))
E = pd.DataFrame({m: (w[m]-w.y)/w.scale for m in FM}, index=w.index); out = {}
for k, g in E.groupby(level=["dataset","kind"]):
    c = g.corr(); ev = np.linalg.eigvalsh(np.cov(g.values.T))
    out[k] = dict(mean_corr=c.values[np.triu_indices(3,1)].mean(), common_share=ev[-1]/ev.sum())
print("Error correlation and common-component share"); print(pd.DataFrame(out).T.round(3))
