"""Pilot: 80% interval coverage of each FM, and equal-weight quantile averaging
(Vincentization) vs linear (mixture) pooling, scored by mean scaled pinball loss.
The mixture is approximated by pooling the 3x9 member quantiles as equally weighted
samples and re-reading quantiles at plotting positions."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import FORECASTS, SCALES, FM

pd.set_option("display.width", 220)
Q = [f"q{i}0" for i in range(1, 10)]; L = np.arange(1, 10)/10
key = ["dataset","unique_id","window","horizon"]
f = pd.read_parquet(FORECASTS, columns=["dataset","model","unique_id","window","horizon","y_true"]+Q)
f = f[f.model.isin(FM)]; f["unique_id"] = f.unique_id.astype(str)
sc = pd.read_csv(SCALES); sc["unique_id"] = sc.unique_id.astype(str)
f = f.merge(sc, on=["dataset","window","unique_id"])
piv = {m: f[f.model == m].set_index(key)[Q+["y_true","mase_scale"]] for m in FM}
idx = piv[FM[0]].index
piv = {m: p.loc[idx] for m, p in piv.items()}
y = piv[FM[0]].y_true.values; s = piv[FM[0]].mase_scale.values
def ql(qv): e = y[:,None]-qv; return np.maximum(L*e, (L-1)*e).mean(1)*2/s
res = pd.DataFrame(index=idx)
stack = np.stack([piv[m][Q].values for m in FM])
pooled = np.sort(np.concatenate(list(stack), axis=1), axis=1)
fi = L*27-0.5; lo = np.floor(fi).astype(int); fr = fi-lo
cands = {m: piv[m][Q].values for m in FM}
cands["vincentization"] = stack.mean(0); cands["mixture"] = pooled[:,lo]*(1-fr)+pooled[:,lo+1]*fr
for n, qv in cands.items():
    res["ql_"+n] = ql(qv); res["cov_"+n] = (y >= qv[:,0]) & (y <= qv[:,-1])
t = res.groupby(level=0).mean(); names = list(cands)
print("80% interval coverage (nominal 0.80)"); print(t[["cov_"+n for n in names]].round(3))
qlt = t[["ql_"+n for n in names]]; best = qlt[["ql_"+m for m in FM]].min(1)
print("Scaled quantile loss relative to the best single FM per task"); print(qlt.div(best, axis=0).round(3))
for n in names: print(f"{n:15s} geomean ratio to best single FM: {np.exp(np.log(qlt['ql_'+n]/best).mean()):.4f}")
