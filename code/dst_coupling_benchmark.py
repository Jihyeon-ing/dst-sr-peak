"""Fair benchmark: solar-wind-only coupling functions vs. the discovered Ey_rect model.

Burton/O'Brien-McPherron-type models are ODEs driven by the *previous Dst*, so comparing them
to a pure solar-wind -> peak-Dst mapping is apples-to-oranges. The fair competitors are
coupling functions built from solar-wind parameters alone:

* Ey_rect (this study)  : max(-V*Bz, 0)                          (rectified VBs)
* Kan-Lee (1979)        : V * B_T * sin^2(theta/2)
* Boyle  (1997)         : 1e-4 * V^2 + 11.7 * B * sin^3(theta/2)  (polar-cap potential)
* Newell (2007)         : V^(4/3) * B_T^(2/3) * sin^(8/3)(theta/2)
* Akasofu epsilon       : V * B^2 * sin^4(theta/2)
* VBs (non-rectified Ey) and |Bz| as simple references

Every coupling function is integrated over the SAME pre-peak window (t-1..t-5) and fit with the
SAME 3-parameter power law  Dst = c - a * X^b  on the training storms, then evaluated on the
2025+ temporal holdout. Output -> ``output_coupling_benchmark``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dst_pysr_cv import N_LAGS, SEED
from dst_pysr_temporal_test_claude import TEST_START, build_split, load_dst


COUPLINGS = ["Ey_rect", "KanLee", "Boyle", "Newell", "epsilon", "VBs", "absBz"]


def omni_couplings(txt_file: str) -> pd.DataFrame:
    with open(txt_file, "r", encoding="utf-8") as f:
        rows = [line.split() for line in f]
    time_parts = [[r[0], r[1].zfill(3), r[2].zfill(2), r[3].zfill(2)] for r in rows]
    timestamps = [dt.datetime.strptime("-".join(t), "%Y-%j-%H-%M") for t in time_parts]
    values = [list(map(float, r[4:])) for r in rows]
    df = pd.DataFrame(values, columns=["Bmag", "Bx", "By", "Bz", "V", "N", "T"], index=timestamps)
    df.replace([9999.99, 99999.9, 999.99, 9999999.0], np.nan, inplace=True)

    V, Bmag, Bz, By = df["V"], df["Bmag"], df["Bz"], df["By"]
    Byz = np.sqrt(By ** 2 + Bz ** 2)
    theta = np.arctan2(np.abs(By), Bz)            # clock angle (By-sign symmetric)
    s = np.sin(theta / 2.0)

    out = pd.DataFrame(index=df.index)
    out["Ey_rect"] = (-V * Bz * 1e-3).clip(lower=0.0)             # rectified VBs (mV/m)
    out["KanLee"] = V * Byz * s ** 2 * 1e-3                       # Kan-Lee merging E-field (mV/m)
    out["Boyle"] = 1e-4 * V ** 2 + 11.7 * Bmag * s ** 3           # Boyle polar-cap potential (kV)
    out["Newell"] = V ** (4.0 / 3.0) * Byz ** (2.0 / 3.0) * s ** (8.0 / 3.0)
    out["epsilon"] = V * Bmag ** 2 * s ** 4                       # Akasofu epsilon (arb. units)
    out["VBs"] = (-V * Bz * 1e-3)                                 # non-rectified Ey
    out["absBz"] = np.abs(Bz)
    return out.resample("1h").mean()


def load_couplings(root: Path) -> pd.DataFrame:
    files = sorted((root / "OMNI_5m").glob("*.txt"))
    if not files:
        raise RuntimeError("No OMNI_5m text files found")
    omni = pd.concat([omni_couplings(str(path)) for path in files]).sort_index()
    if omni.index.has_duplicates:
        raise ValueError(f"Duplicate OMNI timestamps: {omni.index.duplicated().sum()}")
    return omni


def integrate_pre_peak(
    peak_times: pd.DatetimeIndex, dst: pd.Series, omni: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """5h pre-peak integral (t-1..t-5) of each coupling function; same NaN filter as the study."""
    rows = []
    targets = []
    kept_times = []
    for peak in peak_times:
        times = pd.date_range(peak - pd.Timedelta(hours=N_LAGS - 1), peak, freq="1h")
        frame = omni.reindex(times)
        values = frame.to_numpy()
        if np.isnan(values).sum() >= values.size * 0.5:
            continue
        filled = pd.DataFrame(values, columns=omni.columns).interpolate("linear").ffill().bfill()
        if not np.isfinite(filled.to_numpy()).all():
            continue
        if peak not in dst.index:
            raise KeyError(f"Dst target missing at {peak}")
        integral = filled.iloc[0:N_LAGS - 1].sum(axis=0)   # rows t-5..t-1, excluding t0
        rows.append(integral)
        targets.append(float(dst.loc[peak]))
        kept_times.append(peak)
    return pd.DataFrame(rows).reset_index(drop=True), np.asarray(targets), pd.DatetimeIndex(kept_times)


def powerlaw(X, a, b, c):
    return c - a * np.power(np.clip(X, 1e-9, None), b)


def fit_and_eval(name, Xtr, ytr, Xte, yte):
    result = {"coupling": name}
    try:
        popt, _ = curve_fit(
            powerlaw, Xtr, ytr, p0=[1.0, 1.0, -20.0],
            bounds=([0.0, 0.1, -500.0], [np.inf, 5.0, 100.0]), maxfev=20000,
        )
        pred_tr = powerlaw(Xtr, *popt)
        pred_te = powerlaw(Xte, *popt)
        result.update(
            {
                "a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2]),
                "train_r2": float(r2_score(ytr, pred_tr)),
                "test_r2": float(r2_score(yte, pred_te)),
                "test_rmse": float(np.sqrt(mean_squared_error(yte, pred_te))),
                "test_mae": float(mean_absolute_error(yte, pred_te)),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = repr(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output_coupling_benchmark"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_split(root)
    dst = load_dst(root)
    omni = load_couplings(root)
    train_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "train", "peak_time"])
    test_peaks = pd.DatetimeIndex(manifest.loc[manifest.subset == "test", "peak_time"])

    Xtr_df, ytr, _ = integrate_pre_peak(train_peaks, dst, omni)
    Xte_df, yte, _ = integrate_pre_peak(test_peaks, dst, omni)
    print(f"train storms={len(ytr)}  test storms={len(yte)}", flush=True)

    results = [fit_and_eval(name, Xtr_df[name].to_numpy(), ytr, Xte_df[name].to_numpy(), yte)
               for name in COUPLINGS]
    table = pd.DataFrame(results).sort_values("test_r2", ascending=False).reset_index(drop=True)
    table.to_csv(output_dir / "coupling_benchmark.csv", index=False)

    metadata = {
        "comparison": "solar-wind-only coupling functions, 5h pre-peak integral, power-law fit Dst=c-a*X^b",
        "note": "Burton/OBM excluded: they are ODEs driven by previous Dst (autoregressive), not solar-wind-only",
        "window": "t-1..t-5 (excludes t0)",
        "train_storms": len(ytr),
        "test_storms": len(yte),
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Bar chart of test R2.
    ok = table[table["test_r2"].notna()]
    plt.figure(figsize=(8, 4.5))
    colors = ["#d62728" if c == "Ey_rect" else "#1f77b4" for c in ok["coupling"]]
    plt.bar(ok["coupling"], ok["test_r2"], color=colors)
    plt.axhline(0, color="k", linewidth=0.8)
    plt.ylabel("Test $R^2$ (2025–2026)", fontsize=12)
    plt.title("Solar-wind coupling functions vs. peak Dst (5h integral, power-law fit)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "coupling_benchmark_testR2.png", dpi=150)
    plt.close()

    report = [
        "# 태양풍 결합함수 벤치마크 — peak Dst 예측 (Burton류 제외)",
        "",
        f"- 생성 스크립트: `{Path(__file__).name}`",
        f"- 실행 환경: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}",
        "",
        "## 1. 왜 Burton류를 제외했나",
        "",
        "Burton et al.(1975)·O'Brien & McPherron(2000)은 `dDst*/dt = Q(VBs) - Dst*/τ` 형태의 ODE로, "
        "**이전 Dst가 입력**으로 들어가는 자기회귀 모델이다. 태양풍만으로 peak Dst를 예측하는 본 연구와 "
        "직접 비교하면 불공정하므로, **태양풍 파라미터만 사용하는 결합함수**들과 비교한다.",
        "",
        "## 2. 공정 비교 설계",
        "",
        "- 모든 결합함수를 **동일한 peak 직전 5시간 적분(t-1…t-5)** 으로 요약.",
        "- 모두 **동일한 3-parameter 멱함수** `Dst = c - a·X^b` 로 train 피팅 (비선형 최소제곱).",
        "- 2025+ 시간적 홀드아웃에서 평가. train {}개 / test {}개 폭풍.".format(len(ytr), len(yte)),
        "",
        "## 3. 결과 (test R² 내림차순)",
        "",
        "| 결합함수 | train R² | **test R²** | test RMSE | test MAE | 지수 b |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        if pd.isna(getattr(r, "test_r2", np.nan)):
            report.append(f"| {r.coupling} | — | fit 실패 | — | — | — |")
        else:
            report.append(
                f"| {r.coupling} | {r.train_r2:.3f} | **{r.test_r2:.3f}** | {r.test_rmse:.2f} | "
                f"{r.test_mae:.2f} | {r.b:.2f} |"
            )

    report += [
        "",
        "## 4. 해석",
        "",
        "- `Ey_rect`(본 연구, 정류 VBs)와 다른 결합함수의 상대 성능을 통해 SR이 고른 변수가 "
        "고전 결합함수 대비 경쟁력이 있는지 확인한다.",
        "- 각 함수의 피팅 지수 b는 `coupling_benchmark.csv` 참조. 막대그래프: `coupling_benchmark_testR2.png`.",
        "",
        "## 5. 논문 활용",
        "",
        "이 표는 리뷰어의 '고전 결합함수와 비교했는가' 요구에 대한 직접 응답이다. Kan-Lee·Boyle·Newell·ε 는 "
        "모두 태양풍 전용이라 본 연구와 동일 조건이며, Burton류는 정당한 사유(자기회귀)로 제외했음을 명시한다.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
