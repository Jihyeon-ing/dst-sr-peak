"""Formal test of the null a=1 (linear injection) for Dst = c - k*X^a, X = SR_result.

(1) Event-level bootstrap of the exponent on training storms (LSQ and MAE losses).
(2) Predictive check: free-a vs a=1 (linear) on the 2025+ holdout, paired bootstrap.
Output -> output_exponent_test.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error
from dst_pysr_cv import SEED
from dst_pysr_temporal_test_claude import build_split, load_dst
from dst_coupling_benchmark import load_couplings, integrate_pre_peak, powerlaw

root = Path("."); out = root/"output_exponent_test"; out.mkdir(exist_ok=True)
manifest = build_split(root); dst = load_dst(root); omni = load_couplings(root)
tr = pd.DatetimeIndex(manifest.loc[manifest.subset=="train","peak_time"])
te = pd.DatetimeIndex(manifest.loc[manifest.subset=="test","peak_time"])
Xtr_df, ytr, _ = integrate_pre_peak(tr, dst, omni)
Xte_df, yte, _ = integrate_pre_peak(te, dst, omni)
X, Xt = Xtr_df["Ey_rect"].to_numpy(), Xte_df["Ey_rect"].to_numpy()
n = len(ytr); print(f"train={n} test={len(yte)}")

def fit_lsq(x, y):
    p,_ = curve_fit(powerlaw, x, y, p0=[1.5,1.2,-30], bounds=([0,0.1,-500],[np.inf,5,100]), maxfev=20000)
    return p

def fit_mae(x, y, p0):
    xc = np.clip(x,1e-9,None)
    f = lambda p: np.abs(y-(p[2]-p[0]*xc**p[1])).mean()
    r = minimize(f, p0, method="Nelder-Mead", options={"maxiter":2000,"xatol":1e-4,"fatol":1e-4})
    return r.x

p_lsq = fit_lsq(X, ytr); p_mae = fit_mae(X, ytr, p_lsq)
print(f"point est.  LSQ a={p_lsq[1]:.4f}   MAE a={p_mae[1]:.4f}")

rng = np.random.default_rng(SEED)
NB_L, NB_M = 10000, 3000
bl = []
for i in range(NB_L):
    s = rng.integers(0,n,n)
    try: bl.append(fit_lsq(X[s], ytr[s])[1])
    except Exception: pass
bl = np.array(bl)
bm = []
rng2 = np.random.default_rng(SEED+1)
for i in range(NB_M):
    s = rng2.integers(0,n,n)
    try:
        p0 = fit_lsq(X[s], ytr[s])
        bm.append(fit_mae(X[s], ytr[s], p0)[1])
    except Exception: pass
bm = np.array(bm)

def summ(b):
    return dict(median=float(np.median(b)), ci=[float(np.percentile(b,2.5)),float(np.percentile(b,97.5))],
                p_gt1=float((b>1).mean()), n=len(b))
res = {"lsq": {"point": float(p_lsq[1]), **summ(bl)},
       "mae": {"point": float(p_mae[1]), **summ(bm)}}

# (2) predictive: free-a vs a=1 on holdout
def lin(x,a,c): return c - a*x
pl,_ = curve_fit(lin, X, ytr, p0=[3,-30])
pred_free = powerlaw(Xt, *p_lsq); pred_lin = lin(Xt, *pl)
r_free = float(np.sqrt(mean_squared_error(yte, pred_free)))
r_lin  = float(np.sqrt(mean_squared_error(yte, pred_lin)))
idx = rng.integers(0,len(yte),size=(10000,len(yte))); d=np.empty(10000)
for i in range(10000):
    s=idx[i]
    d[i]=np.sqrt(np.mean((yte[s]-pred_lin[s])**2))-np.sqrt(np.mean((yte[s]-pred_free[s])**2))
res["holdout"] = {"rmse_free": r_free, "rmse_linear": r_lin,
                  "dRMSE_lin_minus_free": {"median": float(np.median(d)),
                  "ci":[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]}}
json.dump(res, open(out/"result.json","w"), indent=2)
print(json.dumps(res, indent=2))
