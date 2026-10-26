"""Pilot: Chronos-2 univariate (as in the benchmark) vs multivariate inference, 20 windows.
ETT/Weather: all variables of the dataset forecast jointly as one multivariate input.
ECL: cross-learning over batches of 100 series. Same windows, scales and nonnegative
clipping as the combination run; the univariate arm reproduces the stored forecasts."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import ROOT, OUT, SCALES
sys.path.insert(0, str(ROOT/"src"))
from timestransfer.data import load_long_horizon2
from chronos import Chronos2Pipeline

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", revision="29ec3766d36d6f73f0696f85560a422f50e8498c", device_map="cuda")
sc = pd.read_csv(SCALES); sc["unique_id"] = sc.unique_id.astype(str)
H, NW = 48, 20
tasks = [("ett_h1_h48","ETTh1"),("ett_h2_h48","ETTh2"),("ett_m1_h48","ETTm1"),("ett_m2_h48","ETTm2"),("weather_h48","Weather"),("ecl_h48","ECL")]
rows = []
for task, grp in tasks:
    df = load_long_horizon2(str(ROOT/"data"), grp); df["unique_id"] = df.unique_id.astype(str)
    Y = df.pivot(index="ds", columns="unique_id", values="y").sort_index()
    ids = list(Y.columns); X = Y.values.T.astype(np.float32); T = X.shape[1]; t0 = time.time()
    for k in range(NW):
        cut = T-k*H-H; ctx = X[:, max(0, cut-8192):cut]; yt = X[:, cut:cut+H]
        qu, _ = pipe.predict_quantiles([c for c in ctx], prediction_length=H, quantile_levels=[0.5], batch_size=100, cross_learning=False)
        pu = np.stack([q[0,:,0].numpy() for q in qu])
        if task == "ecl_h48":
            qm, _ = pipe.predict_quantiles([c for c in ctx], prediction_length=H, quantile_levels=[0.5], batch_size=100, cross_learning=True)
            pm = np.stack([q[0,:,0].numpy() for q in qm])
        else:
            qm, _ = pipe.predict_quantiles([ctx], prediction_length=H, quantile_levels=[0.5]); pm = qm[0][:,:,0].numpy()
        nonneg = (X[:, :cut].min(1) >= 0)[:, None]
        pu = np.where(nonneg, np.maximum(pu, 0), pu); pm = np.where(nonneg, np.maximum(pm, 0), pm)
        s = sc[(sc.dataset == task) & (sc.window == k)].set_index("unique_id").loc[ids].mase_scale.values[:, None]
        for i, u in enumerate(ids):
            rows.append((task, u, k, np.mean(np.abs(pu[i]-yt[i])/s[i]), np.mean(np.abs(pm[i]-yt[i])/s[i])))
    print(task, f"{time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows, columns=["dataset","unique_id","window","uni","multi"])
r.to_parquet(OUT/"chronos2_multivariate.parquet")
t = r.groupby("dataset")[["uni","multi"]].mean(); t["ratio"] = t.multi/t.uni
tw = r.groupby(["dataset","window"])[["uni","multi"]].mean(); t["windows_improved"] = (tw.multi < tw.uni).groupby(level=0).mean()
ser = r.groupby(["dataset","unique_id"])[["uni","multi"]].mean(); t["series_improved"] = (ser.multi < ser.uni).groupby(level=0).mean()
print(t.round(3))
