"""Regenerate all paper figures with consistent, reader-friendly labels.

- Ey_rect is labelled "SR_result" everywhere.
- Feature codes like "sum_L1_5" become human-readable ("5 h integral", "2 h minimum", ...).
- The model-selection figure marks only the chosen (1-SE) complexity.

Figures -> ``paper_figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

from dst_pysr_temporal_test_claude import build_split, load_dst
from dst_pysr_temporal_inp2 import load_omni, make_features
from dst_coupling_benchmark import COUPLINGS, load_couplings, integrate_pre_peak, powerlaw

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "paper_figures"
FIG.mkdir(exist_ok=True)

STAT_WORD = {"sum": "integral", "mean": "mean", "min": "minimum", "max": "maximum"}
VAR_LABEL = {"Ey_rect": "SR_result", "Bz": "Bz", "Byz": "Byz", "Ey": "Ey", "By": "By",
             "Bx": "Bx", "V": "V", "Nden": "N", "P": "P", "beta": "β",
             "sin_theta": "sinθ", "T": "T"}
COUP_LABEL = {"Ey_rect": "Our result", "Boyle": "Boyle", "KanLee": "Kan-Lee",
              "Newell": "Newell", "epsilon": "ε", "VBs": "VBz", "absBz": "|Bz|"}


def readable(feature: str) -> str:
    for base in sorted(VAR_LABEL, key=len, reverse=True):
        if feature == base:
            return VAR_LABEL[base]
        if feature.startswith(base + "_"):
            rest = feature[len(base) + 1:]
            lbl = VAR_LABEL[base]
            if rest.startswith("T") and rest[1:].isdigit():
                return f"{lbl} (t−{rest[1:]} h)"
            parts = rest.split("_")            # e.g. ['sum', 'L1', '5']
            stat = parts[0]
            hours = int(parts[2]) - int(parts[1][1:]) + 1
            return f"{lbl} ({hours} h {STAT_WORD.get(stat, stat)})"
    return feature


STAR = dict(marker="*", s=150, edgecolors="black", linewidths=0.5, zorder=10)


def gain_eff_pick(hof: pd.DataFrame) -> str:
    curve = (hof.rename(columns={"Complexity": "complexity", "Loss": "loss"})[["complexity", "loss"]]
             .dropna().sort_values("complexity").drop_duplicates("complexity").reset_index())
    curve["ge"] = (-curve["loss"].diff()) / curve["complexity"].diff()
    valid = curve.dropna(subset=["ge"]).sort_values("complexity")
    cand = valid.iloc[1:] if len(valid) > 1 else valid
    return hof.loc[int(cand.loc[cand["ge"].idxmax(), "index"]), "Equation"]


# ----------------------------------------------------------------------------------------
def fig_model_scatter():
    top = json.loads((ROOT / "output_cv_perinput_r05/selected_features.json").read_text())
    manifest = build_split(ROOT); dst = load_dst(ROOT); omni = load_omni(ROOT)
    trp = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
    tep = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])
    Xtr, ytr, feats, _ = make_features(trp, dst, omni)
    Xte, yte, _, _ = make_features(tep, dst, omni)
    hof = pd.read_csv(ROOT / "output_cv_perinput_r05/final/final_train_topfeat/hall_of_fame.csv")
    eq = gain_eff_pick(hof)
    expr = sp.sympify(eq.replace("^", "**"), locals={"abs": sp.Abs})
    syms = sorted(expr.free_symbols, key=str)
    f = sp.lambdify(syms, expr, "numpy")

    def pred(X):
        return np.asarray(f(*[X[:, feats.index(str(s))] for s in syms]), dtype=float)
    ptr, pte = pred(Xtr), pred(Xte)
    lo = min(ytr.min(), yte.min(), ptr.min(), pte.min()) - 20

    plt.figure(figsize=(6, 6))
    plt.scatter(ytr, ptr, color="blue", alpha=0.45, label="train")
    plt.scatter(yte, pte, color="orange", label="test (2025–2026)", **STAR)
    plt.plot([lo, 0], [lo, 0], "r--")
    plt.xlim(lo, 0); plt.ylim(lo, 0); plt.grid(alpha=0.4); plt.legend(fontsize=11)
    plt.xlabel("Observed Dst peak (nT)", fontsize=13, fontweight="bold")
    plt.ylabel("Predicted Dst peak (nT)", fontsize=13, fontweight="bold")
    a = float(sp.Poly(expr, syms[0]).all_coeffs()[0]) if expr.is_polynomial(syms[0]) else None
    plt.title("Discovered model:  Dst = −(SR_result)$^{1.25}$ − 41.6\n"
              "SR_result = 5 h integral of rectified VBs",
              fontsize=11)
    r2 = r2_score(yte, pte)
    plt.text(0.05, 0.92, f"test R² = {r2:.3f}", transform=plt.gca().transAxes,
             fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG / "fig2_model_scatter.png", dpi=180); plt.close()
    print("fig2 done")


def fig_screening():
    # t0-inclusive screening run; black-and-white, no title (paper style).
    rk = pd.read_csv(ROOT / "output_t0_full/screening/variable_ranking.csv")
    label_map = dict(VAR_LABEL, Ey_rect="Ey,rect")
    rk["label"] = rk["variable"].map(lambda v: label_map.get(v, v))
    rk = rk.sort_values("cv_r2", ascending=False)
    plt.figure(figsize=(9, 4.6))
    plt.bar(rk["label"], rk["cv_r2"], color="black", width=0.6)
    plt.axhline(0, color="k", lw=0.8)
    plt.axhline(0.5, color="gray", ls="--", lw=1.2)
    plt.ylabel("CV R²", fontsize=13)
    plt.ylim(-0.75, 0.9)  # Bx (-45.3) is far below; bar is clipped at the axis floor
    plt.xticks(rotation=25, ha="right", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "fig3_variable_screening.png", dpi=180); plt.close()
    print("fig3 done")


def fig_coupling_2x2():
    manifest = build_split(ROOT); dst = load_dst(ROOT); omni = load_couplings(ROOT)
    trp = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
    tep = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])
    Xtr_df, ytr, _ = integrate_pre_peak(trp, dst, omni)
    Xte_df, yte, _ = integrate_pre_peak(tep, dst, omni)
    # All six benchmark models (epsilon excluded), identical styling, no metrics in titles.
    models = [("Our result", "Ey_rect"),
              ("Boyle (1997)", "Boyle"),
              ("$-VB_z$", "VBs"),
              ("|Bz|", "absBz"),
              ("Kan-Lee (1979)", "KanLee"),
              ("Newell (2007)", "Newell")]
    fits = {}
    for label, col in models:
        if col == "Ey_rect":
            # The discovered equation (t0-inclusive pipeline, linear) applied verbatim - no refit.
            eqf = lambda X: -3.1542134 * X - 23.5927860
            ptr = eqf(Xtr_df[col].to_numpy())
            pte = eqf(Xte_df[col].to_numpy())
        else:
            popt, _ = curve_fit(powerlaw, Xtr_df[col].to_numpy(), ytr, p0=[1, 1, -20],
                                bounds=([0, 0.1, -500], [np.inf, 5, 100]), maxfev=20000)
            ptr = powerlaw(Xtr_df[col].to_numpy(), *popt)
            pte = powerlaw(Xte_df[col].to_numpy(), *popt)
        fits[col] = dict(ptr=ptr, pte=pte, r2=r2_score(yte, pte),
                         rmse=np.sqrt(mean_squared_error(yte, pte)))
    allv = np.concatenate([ytr, yte] + [fits[c]["ptr"] for _, c in models]
                          + [fits[c]["pte"] for _, c in models])
    lo = allv.min() - 20
    panel = "abcdef"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (ax, (label, col)) in enumerate(zip(axes.ravel(), models)):
        d = fits[col]
        ax.scatter(ytr, d["ptr"], color="blue", alpha=0.4, label="train")
        ax.scatter(yte, d["pte"], color="orange", label="test", **STAR)
        ax.plot([lo, 0], [lo, 0], "r--")
        ax.set_xlim(lo, 0); ax.set_ylim(lo, 0); ax.grid(alpha=0.4)
        ax.set_xlabel("Observed Dst (nT)", fontsize=12)
        ax.set_ylabel("Estimated Dst (nT)", fontsize=12)
        ax.set_title(f"({panel[i]}) {label}", fontsize=16)
        ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_coupling_2x3.png", dpi=160); plt.close()
    print("fig4 done")


def fig_forest():
    # Akasofu epsilon excluded from the benchmark (decision 2026-10-08); black, no title.
    bs = pd.read_csv(ROOT / "output_coupling_benchmark/coupling_bootstrap.csv")
    comp = bs[~bs["coupling"].isin(["Ey_rect", "epsilon"])].copy()
    comp["label"] = comp["coupling"].map(lambda c: COUP_LABEL.get(c, c))
    comp = comp.sort_values("dRMSE_vs_ref_median")
    ys = np.arange(len(comp))
    med = comp["dRMSE_vs_ref_median"].to_numpy()
    xlo = med - comp["dRMSE_ci_lo"].to_numpy()
    xhi = comp["dRMSE_ci_hi"].to_numpy() - med
    plt.figure(figsize=(8, 4.5))
    plt.errorbar(med, ys, xerr=[xlo, xhi], fmt="o", color="black", capsize=4, lw=1.2)
    plt.axvline(0, color="r", ls="--", lw=1)
    plt.yticks(ys, comp["label"], fontsize=12)
    plt.xlabel("ΔRMSE (nT)", fontsize=13)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "fig5_bootstrap_forest.png", dpi=180); plt.close()
    print("fig5 done")


def fig_overfit():
    agg = pd.read_csv(ROOT / "output_complexity_overfit/complexity_cv_curve.csv")
    reliable = agg[agg["n_folds"] >= 5].sort_values("complexity")
    minr = reliable.loc[reliable["mean_val_rmse"].idxmin()]
    se = float(minr["std_val_rmse"]) / np.sqrt(float(minr["n_folds"]))
    thr = float(minr["mean_val_rmse"]) + se
    chosen = int(reliable[reliable["mean_val_rmse"] <= thr].iloc[0]["complexity"])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    c = reliable["complexity"].to_numpy()
    m = reliable["mean_val_rmse"].to_numpy()
    s = reliable["std_val_rmse"].fillna(0).to_numpy()
    ax1.plot(c, m, "o-", color="#1f77b4", label="CV validation RMSE")
    ax1.fill_between(c, m - s, m + s, color="#1f77b4", alpha=0.15)
    ax1.axhline(thr, color="purple", lw=1, alpha=0.6, label=f"min + 1 SE ({thr:.1f} nT)")
    ax1.axvline(chosen, color="green", ls="--", lw=2,
                label=f"selected complexity = {chosen}")
    ax1.set_xlabel("Model complexity", fontsize=12)
    ax1.set_ylabel("CV validation RMSE (nT)", fontsize=12, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(reliable["complexity"], reliable["mean_train_loss"], "s--", color="gray", alpha=0.7,
             label="training loss")
    ax2.set_ylabel("Training loss (MAE)", fontsize=12, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=9, loc="upper right")
    plt.title("Model-selection rigour: CV validation RMSE vs. complexity (1-SE rule)", fontsize=12)
    plt.tight_layout(); plt.savefig(FIG / "fig6_overfitting.png", dpi=180); plt.close()
    print("fig6 done")


def fig_corr():
    manifest = build_split(ROOT); dst = load_dst(ROOT); omni = load_couplings(ROOT)
    trp = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
    Xtr, ytr, _ = integrate_pre_peak(trp, dst, omni)
    cols = ["Ey_rect", "Boyle", "KanLee", "Newell", "VBs", "absBz"]
    labels = [COUP_LABEL[c] for c in cols]
    corr = Xtr[cols].corr().to_numpy()
    plt.figure(figsize=(6.4, 5.6))
    im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson r")
    plt.xticks(range(len(cols)), labels, rotation=40, ha="right")
    plt.yticks(range(len(cols)), labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr[i, j]) > 0.6 else "black", fontsize=8)
    plt.title("Correlation among coupling-function integrals\n(all share ~V·B physics)", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "figS1_correlation.png", dpi=180); plt.close()
    print("figS1 done")


def fig_optwindow():
    s = pd.read_csv(ROOT / "output_coupling_optwindow/optimal_window_summary.csv")
    s["label"] = s["coupling"].map(lambda c: COUP_LABEL.get(c, c))
    s = s.sort_values("test_r2_optimal", ascending=False)
    x = np.arange(len(s))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - 0.2, s["test_r2_fixed5h"], width=0.4, label="fixed 5 h integral", color="#aec7e8")
    plt.bar(x + 0.2, s["test_r2_optimal"], width=0.4, label="best window (train-CV)", color="#1f77b4")
    for xi, h in enumerate(s["window_hours"]):
        plt.text(xi + 0.2, s["test_r2_optimal"].iloc[xi] + 0.01, f"{int(h)} h",
                 ha="center", fontsize=8)
    plt.xticks(x, s["label"], rotation=25, ha="right")
    plt.ylabel("Test R² (2025–2026)", fontsize=12)
    plt.title("Robustness: each function's own best integration window", fontsize=12)
    plt.legend(fontsize=10); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "figS2_optwindow.png", dpi=180); plt.close()
    print("figS2 done")


if __name__ == "__main__":
    fig_screening()
    fig_forest()
    fig_overfit()
    fig_optwindow()
    fig_corr()
    fig_coupling_2x2()
    fig_model_scatter()
    print("ALL FIGURES DONE ->", FIG)
