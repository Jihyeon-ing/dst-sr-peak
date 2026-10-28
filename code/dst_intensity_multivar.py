"""Richer peak-Dst law: mine the Pareto front for a physically explicit multi-variable model.

The gain-efficiency pick collapses to a single term (Ey_rect), because Ey_rect = V*Bs already
folds the velocity x magnetic-field interaction into one channel. Here we KEEP the coupling
channels but deliberately look further up the Pareto front for the simplest equation that uses
>= 2 distinct physical variables (so an explicit magnetic-field term appears), then test whether
that extra term is statistically justified out-of-sample.

Features: the 12 skilful channels from the screening study (Ey_rect, Bz, Ey, Byz families).
Method mirrors the peak study (temporal holdout 2025+, PySR, gain-efficiency baseline).
Output -> ``output_cv_intensity_multivar``.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pysr
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from dst_pysr_cv import SEED, canonicalize, expression_features, mean_std
from dst_pysr_temporal_test_claude import (
    TEST_START, build_split, load_dst, make_model, select_by_gain_efficiency,
)
from dst_pysr_temporal_inp2 import load_omni, make_features
from dst_pysr_perinput import variable_feature_names


BASE_VARS = ["Ey_rect", "Bz", "Ey", "Byz"]
_BASE_LOOKUP = {name: v for v in BASE_VARS for name in variable_feature_names(v)}


def base_of(feature: str) -> str | None:
    return _BASE_LOOKUP.get(feature)


def distinct_bases(expr, feature_names):
    present, _ = expression_features(expr, feature_names)
    return sorted({base_of(f) for f in present if base_of(f)})


def simplest_multivar_index(model, feature_names, min_vars=2):
    """Lowest-complexity Pareto equation whose canonical form uses >= min_vars base variables."""
    eqs = model.equations_.sort_values("complexity")
    for idx, row in eqs.iterrows():
        expr = canonicalize(row["sympy_format"])
        if len(distinct_bases(expr, feature_names)) >= min_vars:
            return int(idx), expr
    return None, None


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("output_cv_intensity_multivar"))
    parser.add_argument("--features-from", type=Path,
                        default=Path("output_cv_perinput_r05/selected_features.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    top_features = json.loads((root / args.features_from).read_text())
    print(f"[features] {len(top_features)}: {top_features}", flush=True)

    manifest = build_split(root)
    dst = load_dst(root)
    omni = load_omni(root)
    train_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
    test_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])
    X_train_full, y_train, feat_all, _ = make_features(train_peaks, dst, omni)
    X_test_full, y_test, _, test_times = make_features(test_peaks, dst, omni)
    idx = [feat_all.index(f) for f in top_features]
    Xtr, Xte = X_train_full[:, idx], X_test_full[:, idx]

    # --- CV: compare the gain-efficiency (single) model vs the simplest >=2-variable model ---
    rows_ge, rows_mv = [], []
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
    for fold, (fit_idx, val_idx) in enumerate(kf.split(Xtr), start=1):
        print(f"[fold {fold}/{args.cv_folds}]", flush=True)
        model = make_model(output_dir, f"cv_fold_{fold}", SEED + fold, args.niterations, None)
        model.fit(Xtr[fit_idx], y_train[fit_idx], variable_names=top_features)

        ge_idx, _ = select_by_gain_efficiency(model.equations_)
        ge_expr = canonicalize(model.equations_.loc[ge_idx, "sympy_format"])
        ge_pred = np.asarray(model.predict(Xtr[val_idx], index=ge_idx), dtype=float)
        rows_ge.append({"fold": fold, "n_vars": len(distinct_bases(ge_expr, top_features)),
                        "equation": str(ge_expr),
                        "val_rmse": rmse(y_train[val_idx], ge_pred),
                        "val_r2": float(r2_score(y_train[val_idx], ge_pred)),
                        "complexity": float(model.equations_.loc[ge_idx, "complexity"])})

        mv_idx, mv_expr = simplest_multivar_index(model, top_features, 2)
        if mv_idx is None:
            rows_mv.append({"fold": fold, "n_vars": np.nan, "equation": "(none)",
                            "val_rmse": np.nan, "val_r2": np.nan, "complexity": np.nan})
        else:
            mv_pred = np.asarray(model.predict(Xtr[val_idx], index=mv_idx), dtype=float)
            rows_mv.append({"fold": fold, "n_vars": len(distinct_bases(mv_expr, top_features)),
                            "equation": str(mv_expr),
                            "val_rmse": rmse(y_train[val_idx], mv_pred),
                            "val_r2": float(r2_score(y_train[val_idx], mv_pred)),
                            "complexity": float(model.equations_.loc[mv_idx, "complexity"])})

    cv_ge = pd.DataFrame(rows_ge)
    cv_mv = pd.DataFrame(rows_mv)
    cv_ge.to_csv(output_dir / "cv_gain_efficiency.csv", index=False)
    cv_mv.to_csv(output_dir / "cv_multivar.csv", index=False)

    # --- Final model on all training storms: full Pareto analysis -------------------------
    final = make_model(output_dir, "final_multivar", SEED + 100, args.niterations, None)
    final.fit(Xtr, y_train, variable_names=top_features)

    pareto = []
    for i, row in final.equations_.iterrows():
        expr = canonicalize(row["sympy_format"])
        bases = distinct_bases(expr, top_features)
        pred_te = np.asarray(final.predict(Xte, index=int(i)), dtype=float)
        pareto.append({
            "equation_index": int(i), "complexity": float(row["complexity"]),
            "training_loss": float(row["loss"]), "n_base_vars": len(bases),
            "base_vars": ";".join(bases), "canonical_equation": str(expr),
            "test_r2": float(r2_score(y_test, pred_te)),
            "test_rmse": rmse(y_test, pred_te),
        })
    pareto_df = pd.DataFrame(pareto)
    pareto_df.to_csv(output_dir / "final_pareto_analysis.csv", index=False)

    ge_idx, ge_expr = select_by_gain_efficiency(final.equations_)
    ge_expr = canonicalize(final.equations_.loc[ge_idx, "sympy_format"])
    mv_idx, mv_expr = simplest_multivar_index(final, top_features, 2)
    ge_pred_te = np.asarray(final.predict(Xte, index=ge_idx), dtype=float)
    ge_pred_tr = np.asarray(final.predict(Xtr, index=ge_idx), dtype=float)

    result = {"single_var_model": {"equation": str(ge_expr),
                                   "complexity": float(final.equations_.loc[ge_idx, "complexity"]),
                                   "test_r2": float(r2_score(y_test, ge_pred_te)),
                                   "test_rmse": rmse(y_test, ge_pred_te)}}

    if mv_idx is not None:
        mv_pred_te = np.asarray(final.predict(Xte, index=mv_idx), dtype=float)
        mv_pred_tr = np.asarray(final.predict(Xtr, index=mv_idx), dtype=float)
        result["multivar_model"] = {
            "equation": str(mv_expr),
            "base_vars": distinct_bases(mv_expr, top_features),
            "complexity": float(final.equations_.loc[mv_idx, "complexity"]),
            "test_r2": float(r2_score(y_test, mv_pred_te)),
            "test_rmse": rmse(y_test, mv_pred_te),
        }
        # Paired bootstrap on test: does the multivar model beat the single-var one?
        rng = np.random.default_rng(SEED)
        n = len(y_test)
        bidx = rng.integers(0, n, size=(args.n_boot, n))
        d = np.empty(args.n_boot)
        for b in range(args.n_boot):
            sel = bidx[b]
            d[b] = rmse(y_test[sel], mv_pred_te[sel]) - rmse(y_test[sel], ge_pred_te[sel])
        result["dRMSE_multivar_minus_single"] = {
            "median": float(np.median(d)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "prob_multivar_better": float(np.mean(d < 0)),
        }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # --- Scatter of the multivariable model ---------------------------------------------
    if mv_idx is not None:
        lo = min(y_train.min(), y_test.min(), mv_pred_tr.min(), mv_pred_te.min()) - 20
        plt.figure(figsize=(6, 6))
        plt.scatter(y_train, mv_pred_tr, color="blue", alpha=0.45, label="train")
        plt.scatter(y_test, mv_pred_te, color="orange", marker="*", s=150, edgecolors="black",
                    linewidths=0.5, zorder=10, label="test (2025+)")
        plt.plot([lo, 0], [lo, 0], "r--")
        plt.xlim(lo, 0); plt.ylim(lo, 0); plt.grid(alpha=0.4); plt.legend(fontsize=11)
        plt.xlabel("Dst observed (nT)", fontsize=13, fontweight="bold")
        plt.ylabel("Prediction (nT)", fontsize=13, fontweight="bold")
        plt.title("Multi-variable peak-Dst model (explicit B term)", fontsize=12)
        plt.tight_layout(); plt.savefig(output_dir / "scatter_multivar.png", dpi=150); plt.close()

    # --- Report -------------------------------------------------------------------------
    def cvstat(df, col):
        v = df[col].dropna().tolist()
        return mean_std(v) if v else "—"

    report = [
        "# 다변수 peak-Dst 식 — Pareto front에서 자기장 항 포함 모델 발굴",
        "",
        f"- 생성 스크립트: `{Path(__file__).name}`",
        f"- 실행 환경: Python {platform.python_version()}, PySR {pysr.__version__}",
        "",
        "## 1. 목적",
        "",
        "gain-efficiency는 단일항(Ey_rect)로 수렴한다 — Ey_rect=V·Bs가 이미 속도×자기장 상호작용을 "
        "품고 있기 때문. 여기서는 결합채널을 유지하되 Pareto front에서 **≥2개 물리변수를 쓰는 가장 단순한 식**"
        "(자기장 항이 명시적으로 등장)을 뽑아, 그 추가 항이 **홀드아웃에서 통계적으로 정당한지** 검정한다.",
        "",
        f"- 입력 feature({len(top_features)}): " + ", ".join(top_features),
        f"- 물리 기저변수: {', '.join(BASE_VARS)}. 홀드아웃 2025+.",
        "",
        "## 2. 교차검증 — 단일항 vs 최단 다변수식",
        "",
        f"- **단일항(gain-eff)**: val RMSE {cvstat(cv_ge,'val_rmse')} nT, val R² {cvstat(cv_ge,'val_r2')}",
        f"- **다변수(≥2)**: val RMSE {cvstat(cv_mv,'val_rmse')} nT, val R² {cvstat(cv_mv,'val_r2')}",
        "",
        "## 3. 최종 Pareto front (복잡도별)",
        "",
        "| complexity | #변수 | 변수 | test R² | canonical equation |",
        "|---|---|---|---|---|",
    ]
    for r in pareto_df.itertuples():
        report.append(f"| {r.complexity:.0f} | {r.n_base_vars} | {r.base_vars} | {r.test_r2:.3f} | "
                      f"`{r.canonical_equation}` |")

    report += [
        "",
        "## 4. 대표 모델",
        "",
        f"**단일항 (gain-efficiency)** — test R²={result['single_var_model']['test_r2']:.3f}, "
        f"RMSE={result['single_var_model']['test_rmse']:.2f}:",
        "",
        f"`Dst = {result['single_var_model']['equation']}`",
        "",
    ]
    if mv_idx is not None:
        mv = result["multivar_model"]
        dd = result["dRMSE_multivar_minus_single"]
        crosses = dd["ci95"][0] < 0 < dd["ci95"][1]
        verd = "구분 불가 (CI 0 포함) → 추가 항 통계적 미정당" if crosses else \
               ("다변수 유의 우세" if dd["median"] < 0 else "단일항 유의 우세")
        report += [
            f"**최단 다변수 (변수: {', '.join(mv['base_vars'])})** — test R²={mv['test_r2']:.3f}, "
            f"RMSE={mv['test_rmse']:.2f}, complexity={mv['complexity']:.0f}:",
            "",
            f"`Dst = {mv['equation']}`",
            "",
            "## 5. 추가 항(자기장)의 통계적 정당성 — 부트스트랩",
            "",
            f"- ΔRMSE(다변수−단일항) median = {dd['median']:+.2f} nT, "
            f"95% CI [{dd['ci95'][0]:+.2f}, {dd['ci95'][1]:+.2f}], "
            f"P(다변수 우세)={dd['prob_multivar_better']:.0%}",
            f"- **판정: {verd}**",
            "",
            "## 6. 해석",
            "",
            "- 다변수식은 자기장 항을 **명시적으로** 드러내 물리적 가독성이 높다(원하던 형태).",
            "- 다만 test 개선이 유의하지 않다면, 이는 변수들이 강하게 공선(≈0.9)이라 자기장 항이 "
            "Ey_rect의 정보를 크게 넘어서지 못함을 뜻한다 → '더 해석적이나 예측력 향상은 미미'로 정직하게 서술.",
            "- 그림: `scatter_multivar.png`.",
        ]
    else:
        report += ["Pareto front에 ≥2 변수 식이 없었음."]
    report.append("")
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
