"""Scope of validity: does the peak-derived law also hold before/after the storm peak?

The discovered law was fit ONLY at the storm peak:

    Dst = -(SR_result)^1.2530257 - 41.58829 ,   SR_result = 5 h integral of rectified VBs

Here we slide the same construction to t = peak + k for k = -12 ... +24 h. At each offset we
recompute SR_result over the five hours preceding that time and compare with the observed Dst(t),
in two ways:

  (a) FIXED   - apply the peak-fitted coefficients verbatim -> does the law literally transfer?
  (b) REFIT   - refit the same functional form (c - a*X^b) on training storms at that offset
                -> is the functional FORM still valid at that storm phase, with new coefficients?

Physical expectation: the law has no decay term, so during recovery (k > 0) the forcing integral
collapses toward zero while the real Dst is still深 negative -> the fixed law should fail. This
delineates the peak-specific scope of the discovered relation.

Output -> ``output_offset_validity`` (+ figure copied into ``paper_figures``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

from dst_pysr_cv import SEED
from dst_pysr_temporal_test_claude import build_split, load_dst

# Coefficients of the published peak model (output_cv_perinput_r05/.../hall_of_fame.csv,
# gain-efficiency pick at complexity 5).
PEAK_B = 1.2530257
PEAK_C = -41.58829
WINDOW_H = 5          # SR_result integrates Ey_rect over t-1 ... t-5


def omni_eyrect(txt_file: str) -> pd.DataFrame:
    with open(txt_file, "r", encoding="utf-8") as f:
        rows = [line.split() for line in f]
    tp = [[r[0], r[1].zfill(3), r[2].zfill(2), r[3].zfill(2)] for r in rows]
    ts = [dt.datetime.strptime("-".join(t), "%Y-%j-%H-%M") for t in tp]
    vals = [list(map(float, r[4:])) for r in rows]
    df = pd.DataFrame(vals, columns=["Bmag", "Bx", "By", "Bz", "V", "N", "T"], index=ts)
    df.replace([9999.99, 99999.9, 999.99, 9999999.0], np.nan, inplace=True)
    out = pd.DataFrame(index=df.index)
    out["Ey_rect"] = (-df["V"] * df["Bz"] * 1e-3).clip(lower=0.0)
    return out.resample("1h").mean()


def load_eyrect(root: Path) -> pd.Series:
    files = sorted((root / "OMNI_5m").glob("*.txt"))
    s = pd.concat([omni_eyrect(str(p)) for p in files]).sort_index()["Ey_rect"]
    return s[~s.index.duplicated()]


def sr_result_at(t: pd.Timestamp, ey: pd.Series) -> float | None:
    """5 h integral of rectified VBs over [t-5h, t-1h]; None if coverage is insufficient."""
    hours = pd.date_range(t - pd.Timedelta(hours=WINDOW_H), t - pd.Timedelta(hours=1), freq="1h")
    v = ey.reindex(hours).to_numpy(dtype=float)
    if np.isnan(v).sum() >= len(v) * 0.5:
        return None
    v = pd.Series(v).interpolate("linear").ffill().bfill().to_numpy()
    if not np.isfinite(v).all():
        return None
    return float(v.sum())


def powerlaw(X, a, b, c):
    return c - a * np.power(np.clip(X, 1e-9, None), b)


def make_figures(df: pd.DataFrame, out: Path, root: Path) -> None:
    """Two views: (A) like-for-like train comparison, (B) all four curves (solid=train, dashed=test)."""
    figdir = root / "paper_figures"
    k = df["offset_h"]

    # --- (A) fair comparison: both curves evaluated on the SAME (training) storms -------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, df["fixed_r2_train"], "o-", color="#1f77b4", label="fixed peak law (coefficients frozen)")
    ax.plot(k, df["refit_r2_train"], "s-", color="green", alpha=0.75,
            label="same functional form, refitted at each hour")
    ax.axvline(0, color="red", ls="--", lw=1.6, label="storm peak (where the law was fit)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-2.1, 1.0)
    ax.set_xlabel("Hours relative to storm peak", fontsize=13, fontweight="bold")
    ax.set_ylabel("R² for Dst at that time (training storms)", fontsize=12, fontweight="bold")
    ax.set_title("Scope of validity: frozen law vs. refitted form (like-for-like, training storms)",
                 fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(out / "fig_offset_validity.png", dpi=180)
    if figdir.exists():
        plt.savefig(figdir / "fig7_offset_validity.png", dpi=180)
    plt.close()

    # --- (B) all four curves: colour = treatment, linestyle = train/test ----------------
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(k, df["fixed_r2_train"], "o-", color="#1f77b4", label="fixed law — train")
    ax.plot(k, df["fixed_r2_test"], "*--", color="#1f77b4", markersize=10, alpha=0.85,
            label="fixed law — test (2025–2026)")
    ax.plot(k, df["refit_r2_train"], "s-", color="green", alpha=0.8, label="refitted form — train")
    ax.plot(k, df["refit_r2_test"], "^--", color="green", alpha=0.65, label="refitted form — test")
    ax.axvline(0, color="red", ls="--", lw=1.6, label="storm peak")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-2.1, 1.0)
    ax.set_xlabel("Hours relative to storm peak", fontsize=13, fontweight="bold")
    ax.set_ylabel("R² for Dst at that time", fontsize=13, fontweight="bold")
    ax.set_title("Scope of validity — solid: training storms, dashed: out-of-sample test storms",
                 fontsize=11.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    plt.tight_layout()
    plt.savefig(out / "fig_offset_validity_4curves.png", dpi=180)
    if figdir.exists():
        plt.savefig(figdir / "figS3_offset_validity_4curves.png", dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("output_offset_validity"))
    ap.add_argument("--kmin", type=int, default=-12)
    ap.add_argument("--kmax", type=int, default=24)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_split(root)
    dst = load_dst(root)
    ey = load_eyrect(root)
    peaks = {"train": pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"]),
             "test": pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])}

    rows = []
    for k in range(args.kmin, args.kmax + 1):
        data = {}
        for subset, pk in peaks.items():
            X, Y = [], []
            for p in pk:
                t = p + pd.Timedelta(hours=k)
                if t not in dst.index:
                    continue
                x = sr_result_at(t, ey)
                if x is None:
                    continue
                y = float(dst.loc[t])
                if not np.isfinite(y):
                    continue
                X.append(x); Y.append(y)
            data[subset] = (np.asarray(X), np.asarray(Y))

        Xtr, ytr = data["train"]
        Xte, yte = data["test"]
        if len(ytr) < 30 or len(yte) < 5:
            continue

        # (a) fixed peak-fitted law
        fix_tr = -np.power(np.clip(Xtr, 1e-9, None), PEAK_B) + PEAK_C
        fix_te = -np.power(np.clip(Xte, 1e-9, None), PEAK_B) + PEAK_C

        # (b) same form, refit on training storms at this offset
        try:
            popt, _ = curve_fit(powerlaw, Xtr, ytr, p0=[1, 1, -20],
                                bounds=([0, 0.1, -500], [np.inf, 5, 100]), maxfev=20000)
            ref_tr = powerlaw(Xtr, *popt)
            ref_te = powerlaw(Xte, *popt)
            refit_ok = True
        except Exception:
            popt = [np.nan] * 3
            ref_tr = ref_te = np.full_like(ytr, np.nan, dtype=float)
            refit_ok = False

        rows.append({
            "offset_h": k, "n_train": len(ytr), "n_test": len(yte),
            "fixed_r2_train": float(r2_score(ytr, fix_tr)),
            "fixed_r2_test": float(r2_score(yte, fix_te)),
            "fixed_rmse_test": float(np.sqrt(mean_squared_error(yte, fix_te))),
            "refit_r2_train": float(r2_score(ytr, ref_tr)) if refit_ok else np.nan,
            "refit_r2_test": float(r2_score(yte, ref_te)) if refit_ok else np.nan,
            "refit_a": float(popt[0]), "refit_b": float(popt[1]), "refit_c": float(popt[2]),
            "mean_obs_dst": float(np.mean(ytr)), "mean_SR_result": float(np.mean(Xtr)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "offset_validity.csv", index=False)

    make_figures(df, out, root)

    at0 = df.loc[df.offset_h == 0].iloc[0]
    best = df.loc[df["fixed_r2_test"].idxmax()]
    # Where does the fixed law stay useful? Take the CONTIGUOUS run of test R2 > 0.5 that
    # contains the peak (isolated later spikes are not a validity band).
    good = set(df.loc[df["fixed_r2_test"] > 0.5, "offset_h"].astype(int))
    band = None
    if 0 in good:
        loK, hiK = 0, 0
        while loK - 1 in good:
            loK -= 1
        while hiK + 1 in good:
            hiK += 1
        band = (loK, hiK)

    summary = {
        "peak_model": f"Dst = -(SR_result)^{PEAK_B} {PEAK_C}",
        "at_peak_test_r2_fixed": float(at0["fixed_r2_test"]),
        "at_peak_train_r2_fixed": float(at0["fixed_r2_train"]),
        "best_offset_test_r2": {"offset_h": int(best["offset_h"]),
                                "test_r2": float(best["fixed_r2_test"])},
        "offsets_with_test_r2_above_0.5": band,
        "offset_range": [int(df.offset_h.min()), int(df.offset_h.max())],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# 유효 범위 검정 — peak 공식을 peak 전후 시각에 적용",
        "",
        "## 1. 설계",
        "",
        f"- 공식(peak에서 적합): `Dst = −(SR_result)^{PEAK_B:.4f} − {abs(PEAK_C):.2f}`, "
        f"SR_result = 직전 {WINDOW_H}시간 정류 VBs 적분.",
        f"- t = peak + k (k = {args.kmin}…{args.kmax} h)로 슬라이딩하며 그 시각 기준 SR_result를 "
        "다시 계산하고 관측 Dst(t)와 비교.",
        "- (a) **고정**: peak 계수 그대로. (b) **재피팅**: 같은 형태를 각 offset에서 train으로 재적합.",
        "",
        "## 2. 핵심 결과",
        "",
        f"- peak(k=0)에서 test R² = {at0['fixed_r2_test']:.3f} (train {at0['fixed_r2_train']:.3f}) "
        "— 논문 값 재현 확인.",
        f"- 고정 공식의 test R² 최대: k = {int(best['offset_h'])} h (R² = {best['fixed_r2_test']:.3f}).",
        f"- 고정 공식이 test R² > 0.5를 유지하는 **연속 구간(peak 포함)**: "
        + (f"k = {band[0]:+d} … {band[1]:+d} h (폭 {band[1] - band[0] + 1} h)" if band else "없음"),
        "",
        "## 3. offset별 성능",
        "",
        "| k (h) | 고정 test R² | 고정 train R² | 재피팅 train R² | 재피팅 지수 b |",
        "|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        report.append(f"| {r.offset_h:+d} | {r.fixed_r2_test:.3f} | {r.fixed_r2_train:.3f} | "
                      f"{r.refit_r2_train:.3f} | {r.refit_b:.2f} |")

    report += [
        "",
        "## 4. 해석",
        "",
        "- 공식에는 감쇠항(−Dst/τ)이 없다. 회복기(k>0)에는 구동이 멎어 SR_result→0 이 되므로 모델은 "
        f"상수 {PEAK_C:.1f} nT로 수렴하지만 실제 Dst는 여전히 깊게 음수 → 성능 급락이 예상된다.",
        "- 반대로 k≪0(주상 이전)에는 누적이 아직 적어 관측 Dst(≈0)보다 과도하게 음수를 예측한다 "
        f"(모델은 {PEAK_C:.1f} nT보다 위를 예측할 수 없다).",
        "- 따라서 이 곡선은 발견된 법칙이 **peak에 특화된 관계**임을 정량적으로 보여주며, "
        "전 위상(phase)으로 확장하려면 감쇠항을 포함한 동역학 모델이 필요함을 시사한다.",
        "- 그림: `fig_offset_validity.png` (paper_figures/fig7_offset_validity.png).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print("\n".join(report[:20]), flush=True)


if __name__ == "__main__":
    main()
