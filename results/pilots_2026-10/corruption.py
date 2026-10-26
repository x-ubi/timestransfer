"""Pilot: do damaged contexts make TimesFM, Chronos-2 and TabPFN-TS disagree more?
ETTh1, ETTm1, Weather (35 series), windows 0-3. Damage is applied to the last 1,024
points of each training history: 20% MCAR NaNs, one 96-point NaN block near the end,
or 1% +-5 sd spikes. Scoring uses clean actuals and the clean MASE scales."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import ROOT, OUT, SCALES
sys.path.insert(0, str(ROOT/"src"))
from timestransfer.data import load_long_horizon2
from timestransfer import models as M

H, WINS = 48, [0, 1, 2, 3]
sc = pd.read_csv(SCALES); sc["unique_id"] = sc.unique_id.astype(str)
tasks = [("ett_h1_h48","ETTh1","h"),("ett_m1_h48","ETTm1","15min"),("weather_h48","Weather","10min")]
tkw = dict(model_id="google/timesfm-2.5-200m-pytorch", max_context=1024, max_horizon=128, horizon=H,
           revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3", forecast_config_overrides={"per_core_batch_size": 64})
ckw = dict(model_id="amazon/chronos-2", revision="29ec3766d36d6f73f0696f85560a422f50e8498c", horizon=H)
def corrupt(tr, kind, rng):
    tr = tr.copy()
    for u, idx in tr.groupby("unique_id").groups.items():
        idx = np.array(idx); tail = idx[-1024:]; y = tr.loc[idx, "y"].values.astype(float)
        if kind == "mcar20":
            m = rng.random(len(tail)) < 0.2; m[-1] = False; tr.loc[tail[m], "y"] = np.nan
        elif kind == "block":
            L = 96; st = rng.integers(len(tail)-2*L, len(tail)-L); tr.loc[tail[st:st+L], "y"] = np.nan
        elif kind == "outliers":
            sd = np.nanstd(y[-1024:]); m = rng.random(len(tail)) < 0.01
            tr.loc[tail[m], "y"] = tr.loc[tail[m], "y"] + rng.choice([-1, 1], m.sum())*5*sd
    return tr
rows = []
for task, grp, freq in tasks:
    df = load_long_horizon2(str(ROOT/"data"), grp).sort_values(["unique_id","ds"]); df["unique_id"] = df.unique_id.astype(str)
    n = df.groupby("unique_id").cumcount(ascending=False)
    for k in WINS:
        te = df[(n >= k*H) & (n < k*H+H)]; tr = df[n >= k*H+H]
        s = sc[(sc.dataset == task) & (sc.window == k)].set_index("unique_id").mase_scale
        for kind in ["clean","mcar20","block","outliers"]:
            # Deterministic per (task, window, condition); the first run (2026-10-15) used
            # Python's salted hash() instead, so its exact damage pattern is not reproducible.
            rng = np.random.default_rng([[t[0] for t in tasks].index(task), k, ["clean","mcar20","block","outliers"].index(kind)])
            trc = tr if kind == "clean" else corrupt(tr, kind, rng)
            runs = [("timesfm", lambda a, b: M.run_timesfm_2p5(a, b, task, **tkw)),
                    ("chronos2", lambda a, b: M.run_chronos2(a, b, task, **ckw)),
                    ("tabpfn", lambda a, b: M.run_tabpfn_ts(a, b, task, horizon=H, time_freq=freq, max_context_length=4096, model_version="v2"))]
            for name, fn in runs:
                r = fn(trc, te)
                if r.status != "ok": print("FAIL", task, k, kind, name, r.details, flush=True); continue
                fr = r.forecasts; fr["unique_id"] = fr.unique_id.astype(str)
                rows += [(task, k, kind, name, u, h, yt, yp, s[u]) for u, h, yt, yp in fr[["unique_id","horizon","y_true","y_pred"]].itertuples(index=False)]
            print(task, k, kind, time.strftime("%H:%M:%S"), flush=True)
pd.DataFrame(rows, columns=["dataset","window","kind","model","unique_id","horizon","y","yhat","scale"]).to_parquet(OUT/"corruption.parquet")
print("DONE")
