"""How long is the ring current's memory really? Extend the integration window beyond 5 h.

The published feature set capped lags at t-5, so 'the data chose a 5 h window' is only
'the data chose the longest window offered'. Here we offer up to 24 h:

(a) box windows  X_N = sum_{k=1..N} Ey_rect(t_p - k),  N = 1..24
(b) exponential kernels  X_tau = sum_{k=1..48} exp(-(k-1)/tau) * Ey_rect(t_p - k)

Each driver is fit with the same power law Dst = c - a X^b on training storms
(5-fold CV for selection-free skill curves), evaluated on the 2025+ holdout.
Output -> output_memory_kernel.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from dst_pysr_cv import SEED
from dst_pysr_temporal_test_claude import build_split, load_dst
from dst_offset_validity import load_eyrect, powerlaw

MAXLAG = 48
root = Path(".").resolve()
out = root / "output_memory_kernel"; out.mkdir(exist_ok=True)

manifest = build_split(root); dst = load_dst(root); ey = load_eyrect(root)

def lag_matrix(peaks):
    rows, ys = [], []
    for p in peaks:
        if p not in dst.index: continue
        hours = pd.date_range(p - pd.Timedelta(hours=MAXLAG), p - pd.Timedelta(hours=1), freq="1h")
        v = ey.reindex(hours).to_numpy(dtype=float)
        if np.isnan(v).sum() >= len(v) * 0.5: continue
        v = pd.Series(v).interpolate("linear").ffill().bfill().to_numpy()
        if not np.isfinite(v).all(): continue
        rows.append(v[::-1])          # index k-1 = lag k (t-1 first)
        ys.append(float(dst.loc[p]))
    return np.asarray(rows), np.asarray(ys)

tr_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset=="train","peak_time"])
te_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset=="test","peak_time"])
Ltr, ytr = lag_matrix(tr_peaks); Lte, yte = lag_matrix(te_peaks)
print(f"train={len(ytr)} test={len(yte)} (48h lookback filter)")

def fit_eval(Xtr, Xte):
    kf = KFold(5, shuffle=True, random_state=SEED); errs=[]
    for fi, vi in kf.split(Xtr):
        try: p,_ = curve_fit(powerlaw, Xtr[fi], ytr[fi], p0=[1,1,-20],
                             bounds=([0,0.1,-500],[np.inf,5,100]), maxfev=20000)
        except Exception: return np.nan, np.nan, np.nan
        errs.append(np.sqrt(mean_squared_error(ytr[vi], powerlaw(Xtr[vi],*p))))
    p,_ = curve_fit(powerlaw, Xtr, ytr, p0=[1,1,-20],
                    bounds=([0,0.1,-500],[np.inf,5,100]), maxfev=20000)
    return float(np.mean(errs)), float(r2_score(yte, powerlaw(Xte,*p))), float(p[1])

# (a) box windows
box = []
for N in range(1, 25):
    cv, tr2, b = fit_eval(Ltr[:, :N].sum(1), Lte[:, :N].sum(1))
    box.append({"N": N, "cv_rmse": cv, "test_r2": tr2, "b": b})
box = pd.DataFrame(box); box.to_csv(out/"box_windows.csv", index=False)

# (b) exponential kernels
ks = np.arange(MAXLAG)
expd = []
for tau in list(np.arange(0.5, 12.1, 0.5)) + [14,16,20,24]:
    w = np.exp(-ks / tau)
    cv, tr2, b = fit_eval(Ltr @ w, Lte @ w)
    expd.append({"tau": tau, "cv_rmse": cv, "test_r2": tr2, "b": b})
expd = pd.DataFrame(expd); expd.to_csv(out/"exp_kernels.csv", index=False)

best_box = box.loc[box.cv_rmse.idxmin()]; best_exp = expd.loc[expd.cv_rmse.idxmin()]
print("best box N =", int(best_box.N), "cv", round(best_box.cv_rmse,2), "testR2", round(best_box.test_r2,3))
print("best tau  =", best_exp.tau, "cv", round(best_exp.cv_rmse,2), "testR2", round(best_exp.test_r2,3))
print("N=5 row:", box.loc[box.N==5].round(3).to_dict("records"))

fig, (a1,a2) = plt.subplots(1,2, figsize=(13,5))
a1.plot(box.N, box.cv_rmse, "o-", color="black")
a1.axvline(5, color="red", ls="--", lw=1.2)
a1.set_xlabel("Window length N (hours)", fontsize=13); a1.set_ylabel("Train CV RMSE (nT)", fontsize=13)
a1.grid(alpha=.4); a1.text(0.02,1.03,"(a)", transform=a1.transAxes, fontsize=14, fontweight="bold")
a2.plot(expd.tau, expd.cv_rmse, "o-", color="black")
a2.set_xlabel("Kernel e-folding τ (hours)", fontsize=13); a2.set_ylabel("Train CV RMSE (nT)", fontsize=13)
a2.grid(alpha=.4); a2.text(0.02,1.03,"(b)", transform=a2.transAxes, fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(out/"memory_kernel_scan.png", dpi=200, facecolor="white")
plt.savefig("paper_figures/memory_kernel_scan.png", dpi=200, facecolor="white")
json.dump({"train":len(ytr),"test":len(yte),"best_box_N":int(best_box.N),
           "best_tau":float(best_exp.tau),"N5_cv":float(box.loc[box.N==5,"cv_rmse"].iloc[0])},
          open(out/"summary.json","w"), indent=2)
print("saved figures")
