"""Temporal holdout experiment, expanded-input variant (`output_cv_inp2`).

Builds on ``dst_pysr_temporal_test_claude.py`` (gain-efficiency model selection, variable
stability over the whole Pareto front, research report) but enlarges the input space:

1. Every *contiguous* pre-peak lag window is summarised, not only windows anchored at t-1.
   For N_LAGS=6 the windows are (start_lag, end_lag) with 1 <= s < e <= 5 -> 10 windows,
   e.g. t-1..t-2, t-1..t-5, t-2..t-3, ... , t-4..t-5. This exposes *which* interval matters.
2. A rectified dawn-dusk electric field ``Ey_rect = max(Ey, 0)`` is added as an 11th physical
   variable (southward-IMF ring-current injection only), computed at 5-min resolution.
3. For every window each variable gets the full set of statistics: sum, mean, std, min, max.

Original datasets and scripts are read-only; artifacts are written below ``output_cv_inp2``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pysr
import sklearn
import sympy as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from dst_pysr_cv import (
    N_LAGS,
    SEED,
    VARIABLES,
    candidate_metrics,
    canonicalize,
    expression_features,
    mean_std,
)
from dst_pysr_temporal_test_claude import (
    TEST_START,
    build_split,
    fold_variable_union,
    load_dst,
    make_model,
    select_by_gain_efficiency,
)


# 10 physical variables + rectified southward electric field.
EXT_VARIABLES = list(VARIABLES) + ["Ey_rect"]

# All contiguous pre-peak lag windows (start_lag, end_lag), 1 <= s < e <= N_LAGS-1.
WINDOWS = [(s, e) for s in range(1, N_LAGS - 1) for e in range(s + 1, N_LAGS)]

# Per-window statistics.
STATS = ["sum", "mean", "std", "min", "max"]


def omni_preprocess_ext(txt_file: str) -> pd.DataFrame:
    """Same physics as ``dst_pysr_cv.omni_preprocess`` plus a rectified Ey channel."""
    with open(txt_file, "r", encoding="utf-8") as f:
        rows = [line.split() for line in f]

    time_parts = [[r[0], r[1].zfill(3), r[2].zfill(2), r[3].zfill(2)] for r in rows]
    timestamps = [dt.datetime.strptime("-".join(t), "%Y-%j-%H-%M") for t in time_parts]
    values = [list(map(float, r[4:])) for r in rows]
    df = pd.DataFrame(values, columns=["Bmag", "Bx", "By", "Bz", "V", "N", "T"], index=timestamps)
    df.replace([9999.99, 99999.9, 999.99, 9999999.0], np.nan, inplace=True)
    df["P"] = 2e-6 * df["N"] * df["V"] ** 2
    df["Ey"] = -df["V"] * df["Bz"] * 1e-3
    df["beta"] = ((df["T"] * 4.16 / 1e5) + 5.34) * df["N"] / df["Bmag"] ** 2
    df["Byz"] = np.sqrt(df["By"] ** 2 + df["Bz"] ** 2)
    theta = np.arctan2(np.abs(df["By"]), df["Bz"])
    df["sin_theta"] = np.sin(theta / 2)
    # Ring-current injection is driven only by southward IMF (Ey > 0 in this sign convention).
    df["Ey_rect"] = df["Ey"].clip(lower=0.0)
    return df[EXT_VARIABLES].resample("1h").mean()


def load_omni(root: Path) -> pd.DataFrame:
    files = sorted((root / "OMNI_5m").glob("*.txt"))
    if not files:
        raise RuntimeError("No OMNI_5m text files found")
    omni = pd.concat([omni_preprocess_ext(str(path)) for path in files]).sort_index()
    if omni.index.has_duplicates:
        raise ValueError(f"Duplicate OMNI timestamps: {omni.index.duplicated().sum()}")
    return omni


def _window_positions(start_lag: int, end_lag: int) -> list[int]:
    """Time-axis indices in the (t-5 .. t0) window for lags start_lag..end_lag."""
    return [(N_LAGS - 1) - lag for lag in range(start_lag, end_lag + 1)]


def _window_stat(sub: np.ndarray, stat: str) -> np.ndarray:
    if stat == "sum":
        return sub.sum(axis=1)
    if stat == "mean":
        return sub.mean(axis=1)
    if stat == "std":
        return sub.std(axis=1)
    if stat == "min":
        return sub.min(axis=1)
    if stat == "max":
        return sub.max(axis=1)
    raise ValueError(f"Unknown stat: {stat}")


def make_features(
    peak_times: pd.DatetimeIndex, dst: pd.Series, omni: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    windows = []
    targets = []
    retained_times = []
    for peak in peak_times:
        times = pd.date_range(peak - pd.Timedelta(hours=N_LAGS - 1), peak, freq="1h")
        frame = omni.reindex(times)
        values = frame.to_numpy()
        if np.isnan(values).sum() >= values.size * 0.5:
            continue
        values = pd.DataFrame(values, columns=EXT_VARIABLES).interpolate("linear").ffill().bfill().to_numpy()
        if not np.isfinite(values).all():
            continue
        if peak not in dst.index:
            raise KeyError(f"Dst target missing at {peak}")
        windows.append(values)
        targets.append(float(dst.loc[peak]))
        retained_times.append(peak)

    windows = np.asarray(windows)  # (storm, time t-5..t0, variable)

    # Raw per-lag values.
    blocks = [windows.reshape(len(windows), -1)]
    names = [f"{v}_T{lag}" for lag in range(N_LAGS - 1, -1, -1) for v in EXT_VARIABLES]

    # Every contiguous window x every statistic.
    for start_lag, end_lag in WINDOWS:
        sub = windows[:, _window_positions(start_lag, end_lag), :]
        for stat in STATS:
            blocks.append(_window_stat(sub, stat))
            names.extend(f"{v}_{stat}_L{start_lag}_{end_lag}" for v in EXT_VARIABLES)

    X = np.concatenate(blocks, axis=1)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values remain after windowing")
    return X, np.asarray(targets), names, pd.DatetimeIndex(retained_times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("output_cv_inp2"))
    parser.add_argument("--cv-folds", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_folds = args.cv_folds

    manifest = build_split(root)
    dst = load_dst(root)
    omni = load_omni(root)
    train_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
    test_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])
    X_train, y_train, feature_names, train_times = make_features(train_peaks, dst, omni)
    X_test, y_test, test_feature_names, test_times = make_features(test_peaks, dst, omni)
    if feature_names != test_feature_names:
        raise AssertionError("Train/test feature definitions differ")

    print(f"[features] variables={len(EXT_VARIABLES)} windows={len(WINDOWS)} "
          f"stats={len(STATS)} total_features={len(feature_names)}", flush=True)

    manifest["usable_model_input"] = manifest["peak_time"].isin(train_times.union(test_times))
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    manifest.loc[~manifest.usable_model_input].to_csv(output_dir / "excluded_missing_omni.csv", index=False)

    metadata = {
        "temporal_split": "train peak < 2025-01-01; test peak >= 2025-01-01",
        "model_selection": "pareto gain-efficiency, first valid step excluded",
        "variable_stability_rule": "union of variables over all equations on each fold's Pareto front",
        "variables": EXT_VARIABLES,
        "added_variable": "Ey_rect = max(Ey, 0)  (southward-IMF injection)",
        "window_definition": "all contiguous lag windows (start_lag, end_lag), 1<=s<e<=5",
        "windows": [f"L{s}_{e}" for s, e in WINDOWS],
        "window_statistics": STATS,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "excluded_missing_omni_train": int(len(train_peaks) - len(train_times)),
        "excluded_missing_omni_test": int(len(test_peaks) - len(test_times)),
        "input_features": len(feature_names),
        "cv_folds": n_folds,
        "seed": SEED,
        "niterations": args.niterations,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pysr": pysr.__version__,
        "sklearn": sklearn.__version__,
        "feature_names": feature_names,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fold_rows = []
    candidates_all = []
    counts: Counter[str] = Counter()
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for fold, (fit_idx, val_idx) in enumerate(kfold.split(X_train), start=1):
        print(f"[fold {fold}/{n_folds}] train={len(fit_idx)} validation={len(val_idx)}", flush=True)
        model = make_model(output_dir, f"cv_fold_{fold}", SEED + fold, args.niterations, 9)
        model.fit(X_train[fit_idx], y_train[fit_idx], variable_names=feature_names)

        eq_index, curve = select_by_gain_efficiency(model.equations_)
        curve.to_csv(output_dir / f"cv_fold_{fold}_gain_efficiency.csv", index=False)
        gain_at_pick = curve.loc[curve["equation_index"] == eq_index, "gain_efficiency"]
        gain_value = float(gain_at_pick.iloc[0]) if len(gain_at_pick) else float("nan")
        expr = canonicalize(model.equations_.loc[eq_index, "sympy_format"])

        counts.update(fold_variable_union(model, feature_names))

        candidates = candidate_metrics(model, X_train[val_idx], y_train[val_idx], fold)
        candidates.to_csv(output_dir / f"cv_fold_{fold}_candidates.csv", index=False)
        candidates_all.append(candidates)

        pred = np.asarray(model.predict(X_train[val_idx], index=eq_index), dtype=float)
        fold_rows.append(
            {
                "fold": fold,
                "equation_index": eq_index,
                "canonical_equation": str(expr),
                "gain_efficiency": gain_value,
                "validation_rmse": float(np.sqrt(mean_squared_error(y_train[val_idx], pred))),
                "validation_mae": float(mean_absolute_error(y_train[val_idx], pred)),
                "validation_r2": float(r2_score(y_train[val_idx], pred)),
                "complexity": float(model.equations_.loc[eq_index, "complexity"]),
            }
        )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / "cv_fold_summary.csv", index=False)
    pd.concat(candidates_all, ignore_index=True).to_csv(output_dir / "cv_all_candidates.csv", index=False)

    stability = pd.DataFrame(
        [{"variable": name, "fold_count": counts[name], "selection_frequency": counts[name] / n_folds}
         for name in feature_names]
    ).sort_values(["fold_count", "variable"], ascending=[False, True]).reset_index(drop=True)
    stability.to_csv(output_dir / "variable_stability.csv", index=False)

    stable_nonzero = stability.loc[stability["fold_count"] > 0]
    top_features = stable_nonzero.head(10)["variable"].tolist()
    if not top_features:
        raise RuntimeError("No features selected")

    top_indices = [feature_names.index(name) for name in top_features]
    (output_dir / "selected_top10.json").write_text(json.dumps(top_features, indent=2), encoding="utf-8")

    print(f"[final] selected variables: {top_features}", flush=True)
    final_model = make_model(output_dir, "final_train_top10", SEED + 100, args.niterations, None)
    final_model.fit(X_train[:, top_indices], y_train, variable_names=top_features)

    final_eq_index, final_curve = select_by_gain_efficiency(final_model.equations_)
    final_curve.to_csv(output_dir / "final_model_gain_efficiency.csv", index=False)
    final_expr = canonicalize(final_model.equations_.loc[final_eq_index, "sympy_format"])
    final_complexity = float(final_model.equations_.loc[final_eq_index, "complexity"])

    test_pred = np.asarray(final_model.predict(X_test[:, top_indices], index=final_eq_index), dtype=float)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_r2 = float(r2_score(y_test, test_pred))
    pd.DataFrame(
        {"peak_time": test_times, "observed_dst": y_test, "predicted_dst": test_pred,
         "residual": test_pred - y_test}
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"[final] selected model: {final_expr}", flush=True)
    candidate_metrics(final_model, X_test[:, top_indices], y_test, 0).to_csv(
        output_dir / "final_model_candidates_test_diagnostics.csv", index=False
    )

    report = [
        "# Dst 최저점 예측: 확장 입력(구간 통계 + Ey_rect) 실험 보고서",
        "",
        f"- 생성 스크립트: `{Path(__file__).name}`",
        f"- 실행 환경: Python {platform.python_version()}, PySR {pysr.__version__}, "
        f"NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}",
        "",
        "## 1. 연구 목적",
        "",
        "태양풍(OMNI)으로부터 Dst 최저점을 예측하는 해석 가능한 기호회귀 관계를 data-driven으로 찾는다. "
        "본 변형은 (a) 어느 **사전 시간 구간**의 태양풍이 중요한지, (b) 물리적으로 정류된 남향 전기장 "
        "`Ey_rect`가 유효한지를 확인하기 위해 입력 특징을 확장한다. 시간적 홀드아웃(2025년 이후 테스트)은 유지.",
        "",
        "## 2. 입력 특징 확장",
        "",
        f"- 물리 변수 {len(EXT_VARIABLES)}종: {', '.join(EXT_VARIABLES)}.",
        "  - **추가**: `Ey_rect = max(Ey, 0)` — 남향 IMF(Ey>0)에 의한 환전류 주입만 반영 (5분 해상도에서 정류 후 시간평균).",
        f"- 사전 lag 윈도우: t-{N_LAGS - 1} … t0 원자료.",
        f"- **모든 연속 구간** (start_lag, end_lag), 1≤s<e≤{N_LAGS - 1} → {len(WINDOWS)}개 윈도우: "
        + ", ".join(f"L{s}_{e}(t-{s}~t-{e})" for s, e in WINDOWS) + ".",
        f"- 각 윈도우 통계량 {len(STATS)}종: {', '.join(STATS)}.",
        f"- 특징 이름 규칙: `{{var}}_{{stat}}_L{{s}}_{{e}}` (예: `Ey_sum_L1_5` = t-1~t-5 합).",
        f"- **총 입력 특징 수: {len(feature_names)}** "
        f"(원자료 {len(EXT_VARIABLES) * N_LAGS} + 구간통계 {len(EXT_VARIABLES) * len(WINDOWS) * len(STATS)}).",
        "",
        "## 3. 데이터/분할",
        "",
        f"- 기준: peak 시각 < {TEST_START.date()} → train, 이후 → test.",
        f"- Train 사용 표본: {len(y_train)} (OMNI 결측 제외 {len(train_peaks) - len(train_times)}건).",
        f"- Test 사용 표본: {len(y_test)} (OMNI 결측 제외 {len(test_peaks) - len(test_times)}건).",
        "",
        "## 4. 방법론",
        "",
        f"- {n_folds}-fold KFold(shuffle, seed={SEED}) 교차검증, fold마다 PySR(`select_k_features=9`)로 Pareto front 생성.",
        "- 모델 선택: gain-efficiency = (직전 대비 training_loss 감소)/(complexity 증가). "
        "상수모델 직후 첫 스텝은 제외하고 그 이후 최댓값을 선택.",
        "- fold validation 지표는 위에서 선택한 모델로 산출.",
        "- 변수 안정성: fold별 Pareto front의 **모든 식**에 등장한 변수의 합집합을 집계.",
        "",
        "## 5. 교차검증 결과",
        "",
        f"- {n_folds}-fold validation RMSE: {mean_std(folds.validation_rmse.tolist())} nT",
        f"- {n_folds}-fold validation MAE: {mean_std(folds.validation_mae.tolist())} nT",
        f"- {n_folds}-fold validation R²: {mean_std(folds.validation_r2.tolist())}",
        "",
        "| fold | eq_index | complexity | gain_efficiency | val RMSE | val MAE | val R² |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in folds.itertuples():
        report.append(
            f"| {r.fold} | {r.equation_index} | {r.complexity:.0f} | {r.gain_efficiency:.4f} | "
            f"{r.validation_rmse:.4f} | {r.validation_mae:.4f} | {r.validation_r2:.4f} |"
        )

    report += [
        "",
        "## 6. 변수 안정성 (상위 15) — 어느 구간/변수가 중요한가",
        "",
        "| variable | fold_count | selection_frequency |",
        "|---|---|---|",
    ]
    for row in stability.head(15).itertuples():
        report.append(f"| {row.variable} | {row.fold_count}/{n_folds} | {row.selection_frequency:.0%} |")

    report += [
        "",
        f"최종 모델 입력으로 선택된 상위 특징: {', '.join(top_features)}",
        "",
        "## 7. 최종 모델 (전체 학습셋, 상위 특징)",
        "",
        f"`Dst_peak = {final_expr}`",
        "",
        f"- 선택 complexity: {final_complexity:.0f} (gain-efficiency 기준)",
        "",
        "## 8. 완전 외삽 테스트 성능 (2025–2026)",
        "",
        f"- RMSE: {test_rmse:.4f} nT",
        f"- MAE: {test_mae:.4f} nT",
        f"- R²: {test_r2:.4f}",
        "",
        "## 9. 산출물 목록",
        "",
        "- `metadata.json`, `split_manifest.csv`, `excluded_missing_omni.csv`",
        "- `cv_fold_summary.csv`, `cv_all_candidates.csv`, `cv_fold_{k}_gain_efficiency.csv`",
        "- `variable_stability.csv`, `selected_top10.json`",
        "- `final_model_gain_efficiency.csv`, `test_predictions.csv`, "
        "`final_model_candidates_test_diagnostics.csv`",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
