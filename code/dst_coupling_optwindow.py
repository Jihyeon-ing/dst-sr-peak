"""Per-coupling optimal-window robustness test.

Fairness concern: the main benchmark fixed a 5h pre-peak integral (t-1..t-5), which is the window
Ey_rect prefers. Kan-Lee / Newell / etc. might couple to Dst best over a different window. Here
EACH coupling function chooses its own best window by cross-validation on the TRAINING set only
(the test set is never used for window selection), then is evaluated on the 2025+ holdout.

Candidate windows: every contiguous lag range (s, e) with 0 <= s <= e <= N_LAGS-1. This spans
single instantaneous values (s == e, e.g. the peak-hour value) up to the full 6-hour integral,
so "papers often use a single value" is included as a candidate. Sum vs. mean is irrelevant here
(the power-law scale absorbs the constant), so only the window itself matters.

Each (coupling, window) is fit with Dst = c - a * X^b. Output -> ``output_coupling_optwindow``.
"""

from __future__ import annotations

import argparse
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

from dst_pysr_cv import N_LAGS, SEED
from dst_pysr_temporal_test_claude import build_split, load_dst
from dst_coupling_benchmark import COUPLINGS, load_couplings


def build_tensor(peak_times, dst, omni):
    """Return (windows tensor [storm, lag(t-5..t0), coupling], y, times) with the study's filter."""
    mats, y, times = [], [], []
    cols = list(omni.columns)
    for peak in peak_times:
        idx = pd.date_range(peak - pd.Timedelta(hours=N_LAGS - 1), peak, freq="1h")
        frame = omni.reindex(idx)
        v = frame.to_numpy()
        if np.isnan(v).sum() >= v.size * 0.5:
            continue
        filled = pd.DataFrame(v, columns=cols).interpolate("linear").ffill().bfill()
        if not np.isfinite(filled.to_numpy()).all():
            continue
        if peak not in dst.index:
            raise KeyError(peak)
        mats.append(filled.to_numpy())
        y.append(float(dst.loc[peak]))
        times.append(peak)
    return np.asarray(mats), np.asarray(y), pd.DatetimeIndex(times)


def window_positions(s, e):
    return [(N_LAGS - 1) - lag for lag in range(s, e + 1)]


def window_name(s, e):
    return f"T{s}" if s == e else f"L{s}_{e}"


def powerlaw(X, a, b, c):
    return c - a * np.power(np.clip(X, 1e-9, None), b)


def fit_pl(X, y):
    popt, _ = curve_fit(powerlaw, X, y, p0=[1, 1, -20],
                        bounds=([0, 0.1, -500], [np.inf, 5, 100]), maxfev=20000)
    return popt


def cv_rmse(X, y, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    errs = []
    for fit_idx, val_idx in kf.split(X):
        try:
            popt = fit_pl(X[fit_idx], y[fit_idx])
        except Exception:
            return np.inf
        pred = powerlaw(X[val_idx], *popt)
        errs.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
    return float(np.mean(errs))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output_coupling_optwindow"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_split(root)
    dst = load_dst(root)
    omni = load_couplings(root)
    train_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
    test_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])
    Ttr, ytr, _ = build_tensor(train_peaks, dst, omni)
    Tte, yte, test_times = build_tensor(test_peaks, dst, omni)
    n_test = len(yte)
    print(f"train={len(ytr)}  test={n_test}", flush=True)

    candidates = [(s, e) for s in range(N_LAGS) for e in range(s, N_LAGS)]  # 0<=s<=e<=5
    print(f"{len(candidates)} candidate windows: "
          + ", ".join(window_name(s, e) for s, e in candidates), flush=True)

    col_idx = {c: omni.columns.get_loc(c) for c in COUPLINGS}
    all_rows = []       # every (coupling, window) diagnostic
    chosen = {}         # coupling -> selection dict
    test_pred = {}      # coupling -> test predictions of the chosen window

    for c in COUPLINGS:
        ci = col_idx[c]
        best = None
        for s, e in candidates:
            pos = window_positions(s, e)
            Xtr = Ttr[:, pos, ci].sum(axis=1)
            Xte = Tte[:, pos, ci].sum(axis=1)
            cv = cv_rmse(Xtr, ytr, args.cv_folds)
            # Full-train refit for diagnostic test metrics (NOT used for selection).
            try:
                popt = fit_pl(Xtr, ytr)
                pte = powerlaw(Xte, *popt)
                tr2 = float(r2_score(ytr, powerlaw(Xtr, *popt)))
                te2 = float(r2_score(yte, pte))
                termse = float(np.sqrt(mean_squared_error(yte, pte)))
            except Exception:
                popt, pte, tr2, te2, termse = None, None, np.nan, np.nan, np.nan
            all_rows.append({"coupling": c, "window": window_name(s, e),
                             "window_hours": e - s + 1, "cv_rmse": cv,
                             "train_r2": tr2, "test_r2": te2, "test_rmse": termse})
            if np.isfinite(cv) and (best is None or cv < best["cv_rmse"]):
                best = {"window": window_name(s, e), "s": s, "e": e, "cv_rmse": cv,
                        "train_r2": tr2, "test_r2": te2, "test_rmse": termse, "pte": pte}
        chosen[c] = best
        test_pred[c] = best["pte"]
        print(f"  {c:9s} -> best window {best['window']:6s} "
              f"(cv_rmse={best['cv_rmse']:.2f})  test R2={best['test_r2']:.3f}  RMSE={best['test_rmse']:.2f}",
              flush=True)

    pd.DataFrame(all_rows).to_csv(output_dir / "all_windows_diagnostics.csv", index=False)

    # Fixed 5h (L1_5) reference for each coupling, for the "changed vs fixed" comparison.
    fixed = {}
    for c in COUPLINGS:
        ci = col_idx[c]
        pos = window_positions(1, 5)
        Xtr = Ttr[:, pos, ci].sum(axis=1)
        Xte = Tte[:, pos, ci].sum(axis=1)
        popt = fit_pl(Xtr, ytr)
        fixed[c] = float(r2_score(yte, powerlaw(Xte, *popt)))

    summary = pd.DataFrame(
        [{"coupling": c, "best_window": chosen[c]["window"],
          "window_hours": chosen[c]["e"] - chosen[c]["s"] + 1,
          "cv_rmse": chosen[c]["cv_rmse"], "test_r2_optimal": chosen[c]["test_r2"],
          "test_rmse_optimal": chosen[c]["test_rmse"], "test_r2_fixed5h": fixed[c]}
         for c in COUPLINGS]
    ).sort_values("test_r2_optimal", ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "optimal_window_summary.csv", index=False)

    # Paired test-set bootstrap: dRMSE(competitor - Ey_rect) with optimal windows.
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, n_test, size=(args.n_boot, n_test))
    ref = test_pred["Ey_rect"]
    boot = {}
    for c in COUPLINGS:
        if c == "Ey_rect":
            continue
        d = np.empty(args.n_boot)
        for b in range(args.n_boot):
            sel = bidx[b]
            d[b] = rmse(yte[sel], test_pred[c][sel]) - rmse(yte[sel], ref[sel])
        boot[c] = {"median": float(np.median(d)),
                   "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                   "prob_Ey_rect_better": float(np.mean(d > 0))}

    (output_dir / "bootstrap_vs_Ey_rect.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    # Plot: fixed-5h vs optimal-window test R2.
    order = summary["coupling"].tolist()
    x = np.arange(len(order))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - 0.2, summary["test_r2_fixed5h"], width=0.4, label="fixed 5h (L1_5)", color="#aec7e8")
    plt.bar(x + 0.2, summary["test_r2_optimal"], width=0.4, label="optimal window (train-CV)", color="#1f77b4")
    for xi, (c, w) in enumerate(zip(order, summary["best_window"])):
        plt.text(xi + 0.2, summary["test_r2_optimal"].iloc[xi] + 0.01, w, ha="center", fontsize=8)
    plt.xticks(x, order, rotation=30, ha="right")
    plt.ylabel("Test $R^2$ (2025–2026)", fontsize=12)
    plt.title("Per-coupling optimal window vs. fixed 5h — window chosen by train CV", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "optwindow_vs_fixed_R2.png", dpi=150)
    plt.close()

    def verdict(lo, hi):
        return "유의 (CI 0 미포함)" if (lo > 0 or hi < 0) else "구분 불가 (CI 0 포함)"

    report = [
        "# 결합함수별 최적 창 robustness 실험",
        "",
        "## 1. 동기",
        "",
        "기존 벤치마크는 5시간 적분(L1_5)을 모든 함수에 고정했는데, 이는 Ey_rect가 선호한 창이다. "
        "각 결합함수가 **자기 최적 창을 train CV로 스스로 선택**하게 하여 창 고정의 편향을 제거한다.",
        "",
        "## 2. 설계",
        "",
        f"- 후보 창: 0≤s≤e≤{N_LAGS - 1} 의 모든 연속 구간 {len(candidates)}개 "
        "(단일 lag `T{s}` = 순간값 포함 ~ 전 구간 적분).",
        f"- 창 선택: **train {args.cv_folds}-fold CV RMSE 최소** (test 미사용).",
        "- 각 (함수, 창)은 `Dst = c − a·X^b` 피팅. 합/평균 무관(스케일이 흡수).",
        f"- 표본: train {len(ytr)}, test {n_test}. 홀드아웃 2025+.",
        "",
        "## 3. 결과 — 각 함수의 최적 창과 성능",
        "",
        "| 결합함수 | 선택 창 (train-CV) | 창 길이(h) | test R² (최적) | test R² (고정5h) | test RMSE (최적) |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        report.append(
            f"| {r.coupling} | {r.best_window} | {r.window_hours} | {r.test_r2_optimal:.3f} | "
            f"{r.test_r2_fixed5h:.3f} | {r.test_rmse_optimal:.2f} |"
        )

    report += [
        "",
        "## 4. Ey_rect(최적 창) 대비 짝지은 ΔRMSE 부트스트랩 (test, 10,000회)",
        "",
        "| 결합함수 | ΔRMSE median [95% CI] (nT) | P(Ey_rect 우세) | 판정 |",
        "|---|---|---|---|",
    ]
    for c in order:
        if c == "Ey_rect":
            continue
        bd = boot[c]
        report.append(
            f"| {c} | {bd['median']:+.2f} [{bd['ci'][0]:+.2f}, {bd['ci'][1]:+.2f}] | "
            f"{bd['prob_Ey_rect_better']:.0%} | {verdict(bd['ci'][0], bd['ci'][1])} |"
        )

    report += [
        "",
        "## 5. 해석 포인트",
        "",
        "- 각 함수가 자기 창을 골랐는데도 순위가 유지되면, '창 고정이 불공정했다'는 반박을 차단한다.",
        "- 어떤 함수가 **단일값(T*)** 을 골랐다면 그 함수는 순간값이 최적이라는 뜻이고, "
        "적분형(L*)을 골랐다면 누적이 유리하다는 뜻 → 물리적으로도 흥미로운 정보.",
        "- 그림: `optwindow_vs_fixed_R2.png` (고정5h vs 최적, 막대 위 라벨 = 선택 창).",
        "",
        "## 6. 산출물",
        "",
        "- `optimal_window_summary.csv`, `all_windows_diagnostics.csv`, `bootstrap_vs_Ey_rect.json`",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
