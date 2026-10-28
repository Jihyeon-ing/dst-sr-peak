"""Verification for the t0-inclusive pipeline: extend lags to t-10 AND keep t0 on offer.
4 skilful variables, lags T0..T10, ALL contiguous windows 0<=s<e<=10 (55) x 5 stats.
Single training fit, select_k=9, gain-efficiency pick, holdout eval. -> output_extlag10_t0
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dst_pysr_cv import SEED, canonicalize
from dst_pysr_temporal_test_claude import build_split, load_dst, make_model, select_by_gain_efficiency
from dst_pysr_temporal_inp2 import omni_preprocess_ext

N_LAGS = 11
VARS = ["Ey_rect", "Bz", "Ey", "Byz"]
WINDOWS = [(s, e) for s in range(0, N_LAGS - 1) for e in range(s + 1, N_LAGS)]  # 55 incl. t0
STATS = ["sum", "mean", "std", "min", "max"]
root = Path("."); out = root / "output_extlag10_t0"; out.mkdir(exist_ok=True)
manifest = build_split(root); dst = load_dst(root)
files = sorted((root / "OMNI_5m").glob("*.txt"))
omni = pd.concat([omni_preprocess_ext(str(p)) for p in files]).sort_index()[VARS]

def stat(sub, s):
    return {"sum": sub.sum(1), "mean": sub.mean(1), "std": sub.std(1),
            "min": sub.min(1), "max": sub.max(1)}[s]

def make(peaks):
    W, y = [], []
    for p in peaks:
        hrs = pd.date_range(p - pd.Timedelta(hours=N_LAGS - 1), p, freq="1h")
        v = omni.reindex(hrs).to_numpy()
        if np.isnan(v).sum() >= v.size * 0.5: continue
        v = pd.DataFrame(v, columns=VARS).interpolate("linear").ffill().bfill().to_numpy()
        if not np.isfinite(v).all() or p not in dst.index: continue
        W.append(v); y.append(float(dst.loc[p]))
    W = np.asarray(W)
    blocks = [W.reshape(len(W), -1)]
    names = [f"{v}_T{l}" for l in range(N_LAGS - 1, -1, -1) for v in VARS]
    for s_, e_ in WINDOWS:
        pos = [(N_LAGS - 1) - l for l in range(s_, e_ + 1)]
        sub = W[:, pos, :]
        for st in STATS:
            blocks.append(stat(sub, st))
            names.extend(f"{v}_{st}_L{s_}_{e_}" for v in VARS)
    return np.concatenate(blocks, axis=1), np.asarray(y), names

tr = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
te = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])
Xtr, ytr, names = make(tr); Xte, yte, _ = make(te)
print(f"[data] train={len(ytr)} test={len(yte)} features={len(names)} (windows: {len(WINDOWS)}, incl. t0 & up to 10h)", flush=True)

m = make_model(out, "extlag10_t0", SEED + 100, 2000, 9)
m.fit(Xtr, ytr, variable_names=names)
print("[RF]", list(m.feature_names_in_), flush=True)
i, _ = select_by_gain_efficiency(m.equations_)
ex = canonicalize(m.equations_.loc[i, "sympy_format"])
pr = np.asarray(m.predict(Xte, index=i), dtype=float)
res = {"equation": str(ex), "complexity": float(m.equations_.loc[i, "complexity"]),
       "rf": list(m.feature_names_in_),
       "test_r2": float(r2_score(yte, pr)), "test_rmse": float(np.sqrt(mean_squared_error(yte, pr))),
       "test_mae": float(mean_absolute_error(yte, pr))}
json.dump(res, open(out / "result.json", "w"), indent=2)
print("[pareto]"); print(m.equations_[["complexity", "loss", "equation"]].head(8).to_string(index=False), flush=True)
print(f"[selected] {res['equation']} | test R2={res['test_r2']:.3f}", flush=True)
