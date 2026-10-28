"""Full pipeline re-run with the t0-INCLUSIVE feature grammar.

Windows: every contiguous (s, e) with 0 <= s < e <= 5 (15 windows, including L0_*) x 5 stats
+ 6 raw lags, for all 11 variables. Stages mirror the published pipeline exactly:

  A. per-variable screening (5-fold CV, select_k=5, niter=1000)
  B. combine: keep CV R^2 >= 0.5, top-3 channels each -> combined 5-fold CV + final fit + holdout
  C. complexity diagnostic: 10-fold, every Pareto equation on validation, 1-SE rule

Output -> output_t0_full/{screening,combine,complexity}
"""
import json, platform
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, pysr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from dst_pysr_cv import N_LAGS, SEED, canonicalize, mean_std
from dst_pysr_temporal_test_claude import (build_split, load_dst, make_model,
                                           select_by_gain_efficiency, fold_variable_union)
from dst_pysr_temporal_inp2 import EXT_VARIABLES, STATS, load_omni

WINDOWS0 = [(s, e) for s in range(0, N_LAGS - 1) for e in range(s + 1, N_LAGS)]  # 15 incl. t0

root = Path(".").resolve()
OUT = root / "output_t0_full"
for d in ["screening", "combine", "complexity"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

def stat(sub, s):
    return {"sum": sub.sum(1), "mean": sub.mean(1), "std": sub.std(1),
            "min": sub.min(1), "max": sub.max(1)}[s]

def make_features_t0(peaks, dst, omni):
    W, y, times = [], [], []
    for p in peaks:
        hrs = pd.date_range(p - pd.Timedelta(hours=N_LAGS - 1), p, freq="1h")
        f = omni.reindex(hrs); v = f.to_numpy()
        if np.isnan(v).sum() >= v.size * 0.5: continue
        v = pd.DataFrame(v, columns=EXT_VARIABLES).interpolate("linear").ffill().bfill().to_numpy()
        if not np.isfinite(v).all() or p not in dst.index: continue
        W.append(v); y.append(float(dst.loc[p])); times.append(p)
    W = np.asarray(W)
    blocks = [W.reshape(len(W), -1)]
    names = [f"{v}_T{l}" for l in range(N_LAGS - 1, -1, -1) for v in EXT_VARIABLES]
    for s_, e_ in WINDOWS0:
        pos = [(N_LAGS - 1) - l for l in range(s_, e_ + 1)]
        sub = W[:, pos, :]
        for st in STATS:
            blocks.append(stat(sub, st))
            names.extend(f"{v}_{st}_L{s_}_{e_}" for v in EXT_VARIABLES)
    return np.concatenate(blocks, axis=1), np.asarray(y), names, pd.DatetimeIndex(times)

def var_channels(v):
    names = [f"{v}_T{l}" for l in range(N_LAGS - 1, -1, -1)]
    for s_, e_ in WINDOWS0:
        names.extend(f"{v}_{st}_L{s_}_{e_}" for st in STATS)
    return names

manifest = build_split(root); dst = load_dst(root); omni = load_omni(root)
trp = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
tep = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])
X_train, y_train, feat_all, _ = make_features_t0(trp, dst, omni)
X_test, y_test, feat_all2, test_times = make_features_t0(tep, dst, omni)
assert feat_all == feat_all2
print(f"[data] train={len(y_train)} test={len(y_test)} features={len(feat_all)} "
      f"(windows incl. t0: {len(WINDOWS0)})", flush=True)

# ---------------- Stage A: per-variable screening -----------------------------------
rank_rows = []
for v in EXT_VARIABLES:
    vdir = OUT / "screening" / v; vdir.mkdir(exist_ok=True)
    ch = var_channels(v)
    cols = [feat_all.index(c) for c in ch]
    Xv = X_train[:, cols]
    counts = Counter(); frows = []
    kf = KFold(5, shuffle=True, random_state=SEED)
    for fold, (fi, vi) in enumerate(kf.split(Xv), 1):
        m = make_model(vdir, f"cv_fold_{fold}", SEED + fold, 1000, 5)
        m.fit(Xv[fi], y_train[fi], variable_names=ch)
        i, _ = select_by_gain_efficiency(m.equations_)
        counts.update(fold_variable_union(m, ch))
        pr = np.asarray(m.predict(Xv[vi], index=i), dtype=float)
        frows.append({"fold": fold,
                      "validation_rmse": float(np.sqrt(mean_squared_error(y_train[vi], pr))),
                      "validation_r2": float(r2_score(y_train[vi], pr))})
    folds = pd.DataFrame(frows); folds.to_csv(vdir / "cv_fold_summary.csv", index=False)
    stab = pd.DataFrame([{"channel": c, "fold_count": counts[c]} for c in ch]) \
             .sort_values(["fold_count", "channel"], ascending=[False, True]).reset_index(drop=True)
    stab.to_csv(vdir / "channel_stability.csv", index=False)
    row = {"variable": v, "cv_rmse": float(folds.validation_rmse.mean()),
           "cv_r2": float(folds.validation_r2.mean()),
           "top_channels": ";".join(stab.loc[stab.fold_count > 0].head(3)["channel"])}
    rank_rows.append(row)
    print(f"[screen] {v:10s} cv_r2={row['cv_r2']:.3f} top={row['top_channels']}", flush=True)

ranking = pd.DataFrame(rank_rows).sort_values("cv_rmse").reset_index(drop=True)
ranking.to_csv(OUT / "screening" / "variable_ranking.csv", index=False)

# ---------------- Stage B: combine ---------------------------------------------------
kept = ranking.loc[ranking.cv_r2 >= 0.5]
top_features = []
for r in kept.itertuples():
    for c in r.top_channels.split(";"):
        if c and c not in top_features:
            top_features.append(c)
print(f"[combine] kept vars: {kept.variable.tolist()} -> {len(top_features)} features", flush=True)
json.dump(top_features, open(OUT / "combine" / "selected_features.json", "w"), indent=2)
idx = [feat_all.index(c) for c in top_features]
Xtr, Xte = X_train[:, idx], X_test[:, idx]

cdir = OUT / "combine"
frows = []
kf = KFold(5, shuffle=True, random_state=SEED)
for fold, (fi, vi) in enumerate(kf.split(Xtr), 1):
    m = make_model(cdir, f"combined_cv_fold_{fold}", SEED + fold, 2000, None)
    m.fit(Xtr[fi], y_train[fi], variable_names=top_features)
    i, _ = select_by_gain_efficiency(m.equations_)
    ex = canonicalize(m.equations_.loc[i, "sympy_format"])
    pr = np.asarray(m.predict(Xtr[vi], index=i), dtype=float)
    frows.append({"fold": fold, "equation": str(ex),
                  "complexity": float(m.equations_.loc[i, "complexity"]),
                  "validation_rmse": float(np.sqrt(mean_squared_error(y_train[vi], pr))),
                  "validation_mae": float(mean_absolute_error(y_train[vi], pr)),
                  "validation_r2": float(r2_score(y_train[vi], pr))})
    print(f"[combine fold {fold}] {ex}", flush=True)
cfolds = pd.DataFrame(frows); cfolds.to_csv(cdir / "combined_cv_fold_summary.csv", index=False)

fm = make_model(cdir, "final_train_topfeat", SEED + 100, 2000, None)
fm.fit(Xtr, y_train, variable_names=top_features)
fi_, curve = select_by_gain_efficiency(fm.equations_)
curve.to_csv(cdir / "final_model_gain_efficiency.csv", index=False)
fex = canonicalize(fm.equations_.loc[fi_, "sympy_format"])
pte = np.asarray(fm.predict(Xte, index=fi_), dtype=float)
res = {"equation": str(fex), "complexity": float(fm.equations_.loc[fi_, "complexity"]),
       "cv_rmse": mean_std(cfolds.validation_rmse.tolist()),
       "cv_mae": mean_std(cfolds.validation_mae.tolist()),
       "cv_r2": mean_std(cfolds.validation_r2.tolist()),
       "test_r2": float(r2_score(y_test, pte)),
       "test_rmse": float(np.sqrt(mean_squared_error(y_test, pte))),
       "test_mae": float(mean_absolute_error(y_test, pte))}
pd.DataFrame({"peak_time": test_times, "observed_dst": y_test, "predicted_dst": pte,
              "residual": pte - y_test}).to_csv(cdir / "test_predictions.csv", index=False)
json.dump(res, open(cdir / "result.json", "w"), indent=2)
print(f"[final] {res['equation']} | test R2={res['test_r2']:.3f} RMSE={res['test_rmse']:.2f}", flush=True)

# ---------------- Stage C: complexity diagnostic (10-fold, 1-SE) ---------------------
from dst_intensity_multivar import distinct_bases  # base-variable mapping uses name prefixes
recs = []
kf10 = KFold(10, shuffle=True, random_state=SEED)
for fold, (fi2, vi2) in enumerate(kf10.split(Xtr), 1):
    m = make_model(OUT / "complexity", f"cv_fold_{fold}", SEED + fold, 2000, None)
    m.fit(Xtr[fi2], y_train[fi2], variable_names=top_features)
    for i, row in m.equations_.iterrows():
        try:
            pr = np.asarray(m.predict(Xtr[vi2], index=int(i)), dtype=float)
            vr = float(np.sqrt(mean_squared_error(y_train[vi2], pr)))
        except Exception:
            continue
        recs.append({"fold": fold, "complexity": int(row["complexity"]),
                     "val_rmse": vr, "train_loss": float(row["loss"])})
    print(f"[cx fold {fold}] done", flush=True)
rec = pd.DataFrame(recs); rec.to_csv(OUT / "complexity" / "per_fold_per_equation.csv", index=False)
agg = rec.groupby("complexity").agg(mean_val_rmse=("val_rmse", "mean"),
                                    std_val_rmse=("val_rmse", "std"),
                                    n_folds=("fold", "nunique"),
                                    mean_train_loss=("train_loss", "mean")).reset_index()
agg.to_csv(OUT / "complexity" / "complexity_cv_curve.csv", index=False)
rel = agg[agg.n_folds >= 5].sort_values("complexity")
mn = rel.loc[rel.mean_val_rmse.idxmin()]
se = float(mn.std_val_rmse) / np.sqrt(float(mn.n_folds))
thr = float(mn.mean_val_rmse) + se
chosen = int(rel[rel.mean_val_rmse <= thr].iloc[0]["complexity"])
json.dump({"argmin": int(mn.complexity), "threshold": thr, "chosen_1se": chosen},
          open(OUT / "complexity" / "summary.json", "w"), indent=2)
print(f"[1-SE] argmin c={int(mn.complexity)} thr={thr:.2f} -> chosen c={chosen}", flush=True)
print("PIPELINE DONE", flush=True)
