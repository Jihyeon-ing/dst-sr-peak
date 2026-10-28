"""Five-fold cross-validation and stability analysis for Dst peak PySR.

This is intentionally standalone: the existing research scripts are not imported
or modified.  All generated artifacts are written below ``outputs_cv``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import platform
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pysr
import sklearn
import sympy as sp
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


VARIABLES = ["Bx", "By", "Bz", "V", "P", "T", "Ey", "beta", "Byz", "sin_theta"]
N_LAGS = 6
SEED = 42


def omni_preprocess(txt_file: str) -> pd.DataFrame:
    with open(txt_file, "r", encoding="utf-8") as f:
        rows = [line.split() for line in f]

    time_parts = [[r[0], r[1].zfill(3), r[2].zfill(2), r[3].zfill(2)] for r in rows]
    timestamps = [dt.datetime.strptime("-".join(t), "%Y-%j-%H-%M") for t in time_parts]
    values = [list(map(float, r[4:])) for r in rows]
    df = pd.DataFrame(
        values,
        columns=["Bmag", "Bx", "By", "Bz", "V", "N", "T"],
        index=timestamps,
    )
    df.replace([9999.99, 99999.9, 999.99, 9999999.0], np.nan, inplace=True)
    df["P"] = 2e-6 * df["N"] * df["V"] ** 2
    df["Ey"] = -df["V"] * df["Bz"] * 1e-3
    df["beta"] = ((df["T"] * 4.16 / 1e5) + 5.34) * df["N"] / df["Bmag"] ** 2
    df["Byz"] = np.sqrt(df["By"] ** 2 + df["Bz"] ** 2)
    theta = np.arctan2(np.abs(df["By"]), df["Bz"])
    df["sin_theta"] = np.sin(theta / 2)
    return df[VARIABLES].resample("1h").mean()


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    npz = np.load(root / "dst.npz", allow_pickle=True)
    dst = np.asarray(npz["dst"])
    dst_time = np.asarray(npz["time"])

    peak_files = sorted((root / "storm_dataset_final").glob("*.png"))
    if not peak_files:
        raise RuntimeError("No PNG files found in storm_dataset_final")
    peaks = np.asarray([int(p.name[:6]) for p in peak_files], dtype=int)
    if np.any(peaks < N_LAGS - 1) or np.any(peaks >= len(dst)):
        raise IndexError("At least one storm peak cannot support the requested lag window")

    omni_parts = [omni_preprocess(p) for p in sorted(glob.glob(str(root / "OMNI_5m/*.txt")))]
    omni = pd.concat(omni_parts)
    expected_time = pd.to_datetime(dst_time)
    if omni.index.has_duplicates:
        duplicate_count = int(omni.index.duplicated().sum())
        raise ValueError(f"OMNI contains {duplicate_count} duplicate hourly timestamps")
    missing_times = expected_time.difference(omni.index)
    if len(missing_times):
        raise ValueError(f"OMNI is missing {len(missing_times)} Dst timestamps")
    # OMNI may extend beyond the Dst target period. Align by timestamp instead of
    # relying on equal array lengths or positional truncation.
    omni = omni.reindex(expected_time)

    indices = np.asarray([np.arange(p - (N_LAGS - 1), p + 1) for p in peaks])
    windows = omni.to_numpy()[indices]  # (storm, time t-5..t0, variable)
    base_flat = windows.reshape(len(windows), -1)
    keep = np.isnan(base_flat).sum(axis=1) < base_flat.shape[1] * 0.5
    # Interpolate along time independently for each physical variable.  Doing this
    # after flattening would incorrectly interpolate between different variables.
    windows = np.asarray(
        [pd.DataFrame(window, columns=VARIABLES).interpolate("linear").ffill().bfill().to_numpy()
         for window in windows[keep]]
    )
    base_flat = windows.reshape(len(windows), -1)

    # Cumulative pre-peak inputs exclude t0.  For example, sum_2h is t-2+t-1
    # and sum_5h is t-5+...+t-1 for every physical variable.
    cumulative_blocks = [windows[:, -hours - 1:-1, :].sum(axis=1) for hours in range(2, 6)]
    X = np.concatenate([base_flat, *cumulative_blocks], axis=1)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values remain after interpolation")

    names = [f"{v}_T{lag}" for lag in range(N_LAGS - 1, -1, -1) for v in VARIABLES]
    names.extend(f"{v}_sum_{hours}h" for hours in range(2, 6) for v in VARIABLES)
    return X, dst[peaks][keep], names, peaks[keep]


def make_model(output_dir: Path, run_id: str, seed: int, niterations: int) -> pysr.PySRRegressor:
    return pysr.PySRRegressor(
        maxsize=30,
        niterations=niterations,
        binary_operators=["+", "*", "-", "/", "pow"],
        unary_operators=["sqrt", "exp", "abs", "sin", "cos"],
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        output_directory=str(output_dir),
        run_id=run_id,
        select_k_features=9,
        parsimony=0.45,
        populations=70,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        progress=False,
    )


def canonicalize(expr: sp.Expr, zero_tol: float = 1e-8) -> sp.Expr:
    expr = expr.xreplace({f: sp.Integer(0) for f in expr.atoms(sp.Float) if abs(float(f)) < zero_tol})
    # Float rounding prevents insignificant serialization noise from defining structure.
    expr = expr.xreplace({f: sp.Float(round(float(f), 8)) for f in expr.atoms(sp.Float)})
    return sp.simplify(sp.expand(expr))


def structure_signature(expr: sp.Expr) -> str:
    """Coefficient-insensitive canonical tree used for structural grouping."""
    replacements = {f: sp.Symbol("C") for f in expr.atoms(sp.Float)}
    replacements.update({i: sp.Symbol("C") for i in expr.atoms(sp.Integer) if i not in (0, 1, -1)})
    return sp.srepr(sp.simplify(expr.xreplace(replacements)))


def expression_features(expr: sp.Expr, all_names: list[str]) -> tuple[list[str], list[str]]:
    present = sorted({str(s) for s in expr.free_symbols if str(s) in all_names})
    operators: list[str] = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Add):
            operators.append("add")
        elif isinstance(node, sp.Mul):
            operators.append("mul")
        elif isinstance(node, sp.Pow):
            operators.append("pow")
        elif getattr(node, "is_Function", False):
            operators.append(node.func.__name__)
    return present, operators


def candidate_metrics(
    model: pysr.PySRRegressor, X_val: np.ndarray, y_val: np.ndarray, fold: int
) -> pd.DataFrame:
    records = []
    for index, row in model.equations_.iterrows():
        try:
            pred = np.asarray(model.predict(X_val, index=index), dtype=float)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
            r2 = float(r2_score(y_val, pred))
            expr = canonicalize(row["sympy_format"])
            variables, operators = expression_features(expr, list(model.feature_names_in_))
            records.append(
                {
                    "fold": fold,
                    "equation_index": int(index),
                    "complexity": float(row["complexity"]),
                    "training_loss": float(row["loss"]),
                    "equation": str(row["equation"]),
                    "canonical_equation": str(expr),
                    "structure_signature": structure_signature(expr),
                    "variables": ";".join(variables),
                    "operators": ";".join(operators),
                    "validation_rmse": rmse,
                    "validation_r2": r2,
                }
            )
        except Exception as exc:
            records.append({"fold": fold, "equation_index": int(index), "error": repr(exc)})
    return pd.DataFrame(records)


def mean_std(values: list[float]) -> str:
    a = np.asarray(values, dtype=float)
    return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_cv"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    X, y, feature_names, peak_indices = load_dataset(root)
    pd.DataFrame({"peak_index": peak_indices, "target_dst": y}).to_csv(out / "samples.csv", index=False)

    metadata = {
        "seed": SEED,
        "folds": 5,
        "niterations": args.niterations,
        "samples": len(y),
        "features": len(feature_names),
        "cumulative_feature_definition": "sum_Nh = t-1 + ... + t-N, excluding t0",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pysr": pysr.__version__,
        "sklearn": sklearn.__version__,
        "feature_names": feature_names,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fold_rows: list[dict] = []
    all_candidates: list[pd.DataFrame] = []
    selected_exprs: list[sp.Expr] = []
    selected_predictions: list[np.ndarray] = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        print(f"[fold {fold}/5] train={len(train_idx)} validation={len(val_idx)}", flush=True)
        model = make_model(out, f"fold_{fold}", SEED + fold, args.niterations)
        model.fit(X[train_idx], y[train_idx], variable_names=feature_names)
        candidates = candidate_metrics(model, X[val_idx], y[val_idx], fold)
        candidates.to_csv(out / f"fold_{fold}_candidates.csv", index=False)
        all_candidates.append(candidates)
        valid = candidates.dropna(subset=["validation_rmse"])
        best = valid.loc[valid["validation_rmse"].idxmin()]
        best_index = int(best["equation_index"])
        expr = canonicalize(model.equations_.loc[best_index, "sympy_format"])
        pred_val = np.asarray(model.predict(X[val_idx], index=best_index), dtype=float)
        pred_all = np.asarray(model.predict(X, index=best_index), dtype=float)
        selected_exprs.append(expr)
        selected_predictions.append(pred_all)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "selected_equation_index": best_index,
                "equation": best["equation"],
                "canonical_equation": str(expr),
                "structure_signature": structure_signature(expr),
                "validation_rmse": float(np.sqrt(mean_squared_error(y[val_idx], pred_val))),
                "validation_r2": float(r2_score(y[val_idx], pred_val)),
                "complexity": float(best["complexity"]),
            }
        )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "fold_summary.csv", index=False)
    pd.concat(all_candidates, ignore_index=True).to_csv(out / "all_candidates.csv", index=False)

    variable_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    structure_counts = Counter(structure_signature(e) for e in selected_exprs)
    for expr in selected_exprs:
        variables, operators = expression_features(expr, feature_names)
        variable_counts.update(set(variables))
        operator_counts.update(set(operators))
    variable_table = pd.DataFrame(
        [{"variable": n, "fold_count": variable_counts[n], "selection_frequency": variable_counts[n] / 5} for n in feature_names]
    ).sort_values(["fold_count", "variable"], ascending=[False, True])
    variable_table.to_csv(out / "variable_stability.csv", index=False)
    pd.DataFrame(
        [{"operator": n, "fold_count": c, "selection_frequency": c / 5} for n, c in operator_counts.most_common()]
    ).to_csv(out / "operator_stability.csv", index=False)
    pd.DataFrame(
        [{"structure_signature": s, "fold_count": c, "frequency": c / 5} for s, c in structure_counts.most_common()]
    ).to_csv(out / "structure_stability.csv", index=False)

    pred_matrix = np.vstack(selected_predictions)
    corr = np.corrcoef(pred_matrix)
    pd.DataFrame(corr, index=range(1, 6), columns=range(1, 6)).to_csv(out / "prediction_correlation.csv")
    pair_rows = []
    scale = float(np.std(y))
    for i in range(5):
        for j in range(i + 1, 5):
            pair_rows.append(
                {
                    "fold_a": i + 1,
                    "fold_b": j + 1,
                    "prediction_correlation": corr[i, j],
                    "prediction_nrmse": float(np.sqrt(np.mean((pred_matrix[i] - pred_matrix[j]) ** 2)) / scale),
                }
            )
    pd.DataFrame(pair_rows).to_csv(out / "prediction_similarity.csv", index=False)

    print("[final] fitting all samples", flush=True)
    final_model = make_model(out, "final_all_data", SEED + 100, args.niterations)
    final_model.fit(X, y, variable_names=feature_names)
    final_best = final_model.get_best()
    final_expr = canonicalize(final_best["sympy_format"])
    final_candidates = candidate_metrics(final_model, X, y, 0)
    final_candidates.to_csv(out / "final_all_data_candidates.csv", index=False)

    stable_variables = variable_table[variable_table["selection_frequency"] >= 0.8]
    report = [
        "# Dst peak PySR 5-fold CV report",
        "",
        f"- Samples: {len(y)}",
        f"- CV RMSE: {mean_std(folds['validation_rmse'].tolist())}",
        f"- CV R²: {mean_std(folds['validation_r2'].tolist())}",
        f"- Equation complexity: {mean_std(folds['complexity'].tolist())}",
        f"- Mean pairwise prediction correlation: {np.mean(corr[np.triu_indices(5, 1)]):.4f}",
        "",
        "## Selected equation by fold (lowest validation RMSE)",
        "",
    ]
    for row in fold_rows:
        report.append(
            f"- Fold {row['fold']}: `{row['canonical_equation']}` "
            f"(RMSE={row['validation_rmse']:.4f}, R²={row['validation_r2']:.4f}, complexity={row['complexity']:.0f})"
        )
    report += ["", "## Stable variables (≥80% of folds)", ""]
    report += [f"- {r.variable}: {r.selection_frequency:.0%}" for r in stable_variables.itertuples()]
    if stable_variables.empty:
        report.append("- None")
    report += [
        "",
        "## Final equation fitted on all data",
        "",
        f"`Dst_peak = {final_expr}`",
        "",
        f"PySR selected complexity: {float(final_best['complexity']):.0f}",
        f"Training loss reported by PySR: {float(final_best['loss']):.6g}",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
