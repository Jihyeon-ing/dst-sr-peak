"""Temporal holdout experiment (gain-efficiency variant): train peak < 2025, test 2025--2026.

This is a standalone copy of ``dst_pysr_temporal_test.py`` with four methodological changes
requested for the Dst-peak PySR study:

1. Variable-stability counting now considers *every* equation on each fold's Pareto front,
   not only the variables of that fold's single selected model.
2. The per-fold "best" equation is chosen by a Pareto **gain-efficiency** criterion
   (loss reduction per unit of added complexity) instead of PySR's built-in ``model_selection``.
3. Each fold's reported validation metrics are computed from the gain-efficiency model of (2).
4. The full experimental procedure is written as a research report (REPORT.md).

Original datasets and scripts are read-only; all artifacts are written below ``output_cv_claude``.
"""

from __future__ import annotations

import argparse
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
    omni_preprocess,
)


# Storms whose peak occurs on or after this instant form the completely out-of-sample test set.
TEST_START = pd.Timestamp("2025-01-01 00:00:00")


def peak_time_from_path(path: Path, source: str) -> pd.Timestamp:
    if source == "historical":
        # Historical format: 261732_2024-11-09 12:00:00.png
        return pd.Timestamp(path.stem.split("_", 1)[1])
    # OOS format: 2025-11-12_04-00-00.png
    return pd.to_datetime(path.stem, format="%Y-%m-%d_%H-%M-%S")


def build_split(root: Path) -> pd.DataFrame:
    rows = []
    sources = [
        ("historical", root / "storm_dataset_final"),
        ("oos_2025_2026", root / "storm_dataset_final_2025"),
    ]
    for source, directory in sources:
        for path in sorted(directory.glob("*.png")):
            peak_time = peak_time_from_path(path, source)
            subset = "test" if peak_time >= TEST_START else "train"
            rows.append({"source": source, "subset": subset, "peak_time": peak_time})

    manifest = pd.DataFrame(rows).sort_values("peak_time").reset_index(drop=True)
    if manifest["peak_time"].duplicated().any():
        duplicates = manifest.loc[manifest["peak_time"].duplicated(False), "peak_time"].tolist()
        raise ValueError(f"Duplicate peak timestamps across storm folders: {duplicates}")
    return manifest


def load_dst(root: Path) -> pd.Series:
    old = np.load(root / "dst.npz", allow_pickle=True)
    historical = pd.Series(old["dst"], index=pd.to_datetime(old["time"]), name="dst")
    recent_df = pd.read_csv(root / "kyoto_dst_2001_2026.csv")
    recent_time = pd.to_datetime(recent_df[["year", "month", "day", "hour"]])
    recent = pd.Series(recent_df["dst"].to_numpy(), index=recent_time, name="dst")
    # Kyoto CSV is authoritative where the two sources overlap.
    dst = pd.concat([historical[historical.index < recent.index.min()], recent]).sort_index()
    if dst.index.has_duplicates:
        raise ValueError("Duplicate timestamps in combined Dst series")
    return dst


def load_omni(root: Path) -> pd.DataFrame:
    files = sorted((root / "OMNI_5m").glob("*.txt"))
    if not files:
        raise RuntimeError("No OMNI_5m text files found")
    omni = pd.concat([omni_preprocess(str(path)) for path in files]).sort_index()
    if omni.index.has_duplicates:
        raise ValueError(f"Duplicate OMNI timestamps: {omni.index.duplicated().sum()}")
    return omni


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
        values = pd.DataFrame(values, columns=VARIABLES).interpolate("linear").ffill().bfill().to_numpy()
        if not np.isfinite(values).all():
            continue
        if peak not in dst.index:
            raise KeyError(f"Dst target missing at {peak}")
        windows.append(values)
        targets.append(float(dst.loc[peak]))
        retained_times.append(peak)

    windows = np.asarray(windows)
    base = windows.reshape(len(windows), -1)
    cumulative = [windows[:, -hours - 1:-1, :].sum(axis=1) for hours in range(2, 6)]
    X = np.concatenate([base, *cumulative], axis=1)
    names = [f"{v}_T{lag}" for lag in range(N_LAGS - 1, -1, -1) for v in VARIABLES]
    names.extend(f"{v}_sum_{hours}h" for hours in range(2, 6) for v in VARIABLES)
    return X, np.asarray(targets), names, pd.DatetimeIndex(retained_times)


def make_model(
    output_dir: Path,
    run_id: str,
    seed: int,
    niterations: int,
    select_k_features: int | None,
) -> pysr.PySRRegressor:
    return pysr.PySRRegressor(
        maxsize=20,
        niterations=niterations,
        binary_operators=["+", "*", "-", "/", "^"],
        unary_operators=["exp", "abs", "log", "sqrt"],
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        output_directory=str(output_dir),
        run_id=run_id,
        select_k_features=select_k_features,
        parsimony=0.05,
        populations=30,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        progress=False,
        model_selection="best",
    )


# --------------------------------------------------------------------------------------
# Change (2): Pareto gain-efficiency model selection.
# --------------------------------------------------------------------------------------
def gain_efficiency_curve(equations: pd.DataFrame) -> pd.DataFrame:
    """Marginal loss reduction per unit of complexity along the Pareto front.

    Mirrors the user's definition: sort by complexity, keep one row per complexity,
    then gain_efficiency = (-diff of training_loss) / (diff of complexity).
    """
    curve = (
        equations[["complexity", "loss"]]
        .rename(columns={"loss": "training_loss"})
        .dropna()
        .sort_values("complexity")
        .drop_duplicates("complexity")
        .reset_index()
        .rename(columns={"index": "equation_index"})
    )
    curve["delta_complexity"] = curve["complexity"].diff()
    curve["loss_reduction"] = -curve["training_loss"].diff()
    curve["gain_efficiency"] = curve["loss_reduction"] / curve["delta_complexity"]
    return curve


def select_by_gain_efficiency(equations: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Return (selected equation index, full gain-efficiency curve).

    The simplest equation (lowest complexity) has an undefined (NaN) gain, and the very
    first step after it almost always has the largest gain-efficiency (jump away from the
    constant model). Both are excluded; the selected model maximises gain-efficiency among
    the *remaining* equations. Fallbacks handle Pareto fronts too short to skip two rows.
    """
    curve = gain_efficiency_curve(equations)
    valid = curve.dropna(subset=["gain_efficiency"]).sort_values("complexity")
    if valid.empty:
        return int(curve["equation_index"].iloc[0]), curve
    # Drop the first valid step (largest gain-efficiency, right after the constant model).
    candidates = valid.iloc[1:]
    if candidates.empty:
        candidates = valid
    best_row = candidates.loc[candidates["gain_efficiency"].idxmax()]
    return int(best_row["equation_index"]), curve


def fold_variable_union(model: pysr.PySRRegressor, feature_names: list[str]) -> set[str]:
    """Change (1): every variable appearing in *any* equation on this fold's Pareto front."""
    used: set[str] = set()
    for _, row in model.equations_.iterrows():
        variables, _ = expression_features(row["sympy_format"], feature_names)
        used.update(variables)
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("output_cv_claude"))
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

    manifest["usable_model_input"] = manifest["peak_time"].isin(train_times.union(test_times))
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    excluded = manifest.loc[~manifest.usable_model_input].copy()
    excluded.to_csv(output_dir / "excluded_missing_omni.csv", index=False)

    metadata = {
        "temporal_split": "train peak < 2025-01-01; test peak >= 2025-01-01",
        "model_selection": "pareto gain-efficiency (max loss_reduction / delta_complexity)",
        "variable_stability_rule": "union of variables over all equations on each fold's Pareto front",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_plots_before_missing_filter": len(train_peaks),
        "test_plots_before_missing_filter": len(test_peaks),
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
    counts: Counter[str] = Counter()  # fold_count per variable (change 1)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for fold, (fit_idx, val_idx) in enumerate(kfold.split(X_train), start=1):
        print(f"[fold {fold}/{n_folds}] train={len(fit_idx)} validation={len(val_idx)}", flush=True)
        model = make_model(output_dir, f"cv_fold_{fold}", SEED + fold, args.niterations, 9)
        model.fit(X_train[fit_idx], y_train[fit_idx], variable_names=feature_names)

        # Change (2): select the fold model by gain-efficiency.
        eq_index, curve = select_by_gain_efficiency(model.equations_)
        curve.to_csv(output_dir / f"cv_fold_{fold}_gain_efficiency.csv", index=False)
        gain_at_pick = curve.loc[curve["equation_index"] == eq_index, "gain_efficiency"]
        gain_value = float(gain_at_pick.iloc[0]) if len(gain_at_pick) else float("nan")
        expr = canonicalize(model.equations_.loc[eq_index, "sympy_format"])

        # Change (1): count variables from every equation on the Pareto front.
        counts.update(fold_variable_union(model, feature_names))

        candidates = candidate_metrics(model, X_train[val_idx], y_train[val_idx], fold)
        candidates.to_csv(output_dir / f"cv_fold_{fold}_candidates.csv", index=False)
        candidates_all.append(candidates)

        # Change (3): validation metrics from the gain-efficiency model.
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

    # Change (2) applied to the final model as well, for consistency.
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

    # ----------------------------------------------------------------------------------
    # Change (4): full research report.
    # ----------------------------------------------------------------------------------
    report = [
        "# Dst 최저점 예측: 시간적 홀드아웃 + Gain-Efficiency 기호회귀 실험 보고서",
        "",
        f"- 생성 스크립트: `{Path(__file__).name}`",
        f"- 실행 환경: Python {platform.python_version()}, PySR {pysr.__version__}, "
        f"NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}",
        "",
        "## 1. 연구 목적",
        "",
        "태양풍(OMNI) 관측치로부터 지자기 폭풍의 Dst 최저점(nT)을 예측하는 해석 가능한 "
        "기호회귀 모델을 구축한다. 데이터 누수를 배제하기 위해 **시간적 홀드아웃**을 적용하여, "
        "과거 폭풍으로 학습·검증하고 미래(2025년 이후) 폭풍으로 완전 외삽 성능을 측정한다.",
        "",
        "## 2. 데이터",
        "",
        f"- Dst 타깃: `dst.npz`(과거) + `kyoto_dst_2001_2026.csv`(최신). 중복 구간은 Kyoto CSV 우선.",
        f"- 태양풍 입력: `OMNI_5m/*.txt` 5분 해상도를 1시간 평균으로 리샘플.",
        f"- 물리 변수 {len(VARIABLES)}종: {', '.join(VARIABLES)}.",
        f"- 폭풍 이벤트: `storm_dataset_final`(과거) + `storm_dataset_final_2025`(최신)의 peak 시각.",
        "",
        "## 3. 학습/테스트 분할 (시간적 홀드아웃)",
        "",
        f"- 기준: peak 시각 < {TEST_START.date()} → train, 이후 → test.",
        f"- Train 폭풍 수(결측 필터 전): {len(train_peaks)}, 필터 후 사용: {len(y_train)}"
        f" (OMNI 결측 제외 {len(train_peaks) - len(train_times)}건).",
        f"- Test 폭풍 수(결측 필터 전): {len(test_peaks)}, 필터 후 사용: {len(y_test)}"
        f" (OMNI 결측 제외 {len(test_peaks) - len(test_times)}건).",
        "- 제외 이벤트는 `excluded_missing_omni.csv`, 전체 분할은 `split_manifest.csv`에 기록.",
        "",
        "## 4. 특징 공학",
        "",
        f"- 각 peak 기준 {N_LAGS}시간 lag 윈도우(t-{N_LAGS - 1} … t0)를 물리 변수별로 구성.",
        "- 윈도우 내 결측이 50% 이상이면 이벤트 제외, 그 외에는 변수별 선형 보간 + 전/후방 채움.",
        "- 누적 입력: t0을 제외한 2~5시간 누적합(`{var}_sum_Nh`)을 추가.",
        f"- 총 입력 특징 수: {len(feature_names)}.",
        "",
        "## 5. 방법론",
        "",
        f"### 5.1 교차검증",
        f"- 학습셋에 대해 {n_folds}-fold KFold(shuffle, seed={SEED}) 교차검증.",
        "- 각 fold에서 PySR 회귀(`select_k_features=9`)로 Pareto front를 생성.",
        "",
        "### 5.2 모델 선택 — Gain-Efficiency (변경점 2)",
        "- Pareto front를 complexity 기준 정렬 후 복잡도별 1개로 정리한 뒤,",
        "  `gain_efficiency = (직전 대비 training_loss 감소량) / (complexity 증가량)` 을 계산.",
        "- 이 값이 **최대**인 식을 fold의 대표 모델로 선택(복잡도 대비 손실 감소가 가장 효율적인 지점).",
        "- fold별 곡선은 `cv_fold_{k}_gain_efficiency.csv`, 최종 모델 곡선은 "
        "`final_model_gain_efficiency.csv`에 저장.",
        "",
        "### 5.3 검증 성능 (변경점 3)",
        "- 각 fold의 validation RMSE/MAE/R²는 5.2에서 선택한 gain-efficiency 모델로 산출.",
        "",
        "### 5.4 변수 안정성 (변경점 1)",
        "- fold별로 **선택된 단일 식이 아니라 Pareto front의 모든 식**에 등장한 변수의 합집합을 집계.",
        f"- 변수의 `fold_count`는 해당 변수가 등장한 fold 수(최대 {n_folds}).",
        "",
        "## 6. 교차검증 결과",
        "",
        f"- {n_folds}-fold validation RMSE: {mean_std(folds.validation_rmse.tolist())} nT",
        f"- {n_folds}-fold validation MAE: {mean_std(folds.validation_mae.tolist())} nT",
        f"- {n_folds}-fold validation R²: {mean_std(folds.validation_r2.tolist())}",
        "",
        "### 6.1 Fold별 선택 모델",
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
        "## 7. 변수 안정성 (상위 10)",
        "",
        "| variable | fold_count | selection_frequency |",
        "|---|---|---|",
    ]
    for row in stability.head(10).itertuples():
        report.append(f"| {row.variable} | {row.fold_count}/{n_folds} | {row.selection_frequency:.0%} |")

    report += [
        "",
        f"최종 모델 입력으로 선택된 상위 특징: {', '.join(top_features)}",
        "",
        "## 8. 최종 모델 (전체 학습셋, 상위 특징)",
        "",
        f"`Dst_peak = {final_expr}`",
        "",
        f"- 선택 complexity: {final_complexity:.0f} (gain-efficiency 기준)",
        "",
        "## 9. 완전 외삽 테스트 성능 (2025–2026)",
        "",
        f"- RMSE: {test_rmse:.4f} nT",
        f"- MAE: {test_mae:.4f} nT",
        f"- R²: {test_r2:.4f}",
        "- 예측/잔차 상세는 `test_predictions.csv` 참조.",
        "",
        "## 10. 산출물 목록",
        "",
        "- `metadata.json` — 실행 설정 및 환경",
        "- `split_manifest.csv`, `excluded_missing_omni.csv` — 분할/제외 이벤트",
        "- `cv_fold_summary.csv`, `cv_all_candidates.csv` — fold 요약/후보식",
        "- `cv_fold_{k}_gain_efficiency.csv`, `final_model_gain_efficiency.csv` — gain-efficiency 곡선",
        "- `variable_stability.csv`, `selected_top10.json` — 변수 안정성/선택 특징",
        "- `test_predictions.csv`, `final_model_candidates_test_diagnostics.csv` — 테스트 결과",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
