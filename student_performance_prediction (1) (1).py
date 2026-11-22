"""
Student Performance Prediction
------------------------------
Predicts final scores (regression) and flags at-risk students (classification).

Usage:
    python student_performance_prediction.py                    # runs on synthetic demo data
    python student_performance_prediction.py --csv students.csv --target final_score

Requirements: pandas, numpy, scikit-learn, matplotlib
    pip install pandas numpy scikit-learn matplotlib
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)

SEED = 42


def make_synthetic_data(n=1000, seed=SEED):
    """Synthetic demo data. Replace with your real, de-identified records."""
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(12, 5, n), 0, 30)
    attend = np.clip(rng.normal(85, 10, n), 40, 100)
    prev = np.clip(rng.normal(65, 14, n), 15, 100)
    assign = np.clip(0.6 * prev + rng.normal(28, 12, n), 0, 100)
    sleep = np.clip(rng.normal(7, 1.2, n), 3.5, 10)
    parent = rng.integers(0, 5, n)
    tutor = rng.binomial(1, 0.3, n)
    absences = np.clip(np.round((100 - attend) / 3 + rng.normal(0, 1.5, n)), 0, None)
    score = (0.35 * prev + 0.22 * assign + 0.9 * study + 0.18 * attend + 1.2 * sleep
             + 1.5 * parent + 3 * tutor - 0.4 * absences + rng.normal(0, 5, n))
    score = np.clip(score - score.mean() + 68, 0, 100)
    return pd.DataFrame({
        "study_hours": study, "attendance_pct": attend, "previous_score": prev,
        "assignment_score": assign, "sleep_hours": sleep, "parental_education": parent,
        "tutoring": tutor, "absences": absences, "final_score": score,
    }).round(2)


def prepare(df, target):
    """Basic cleaning: drop duplicate rows, fill missing values, one-hot encode text columns."""
    df = df.drop_duplicates().dropna(subset=[target]).copy()
    X = df.drop(columns=target)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna("unknown")
        else:
            X[col] = X[col].fillna(X[col].median())
    X = pd.get_dummies(X, drop_first=True)
    return X, df[target]


def run_regression(Xtr, Xte, ytr, yte):
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        cv_r2 = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2").mean()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": mean_squared_error(yte, pred) ** 0.5,
            "R2_test": r2_score(yte, pred),
            "R2_cv": cv_r2,
        })
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("R2_test", ascending=False).reset_index(drop=True)
    return results, fitted


def run_classification(X, y_score, risk_threshold):
    """Flag students whose final score is below the risk threshold."""
    y = (y_score < risk_threshold).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, class_weight="balanced"),
    }
    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1": f1_score(yte, pred, zero_division=0),
        })
        matrices[name] = confusion_matrix(yte, pred)
    return pd.DataFrame(rows), matrices


def save_plots(best_name, best_model, Xte, yte, prefix="plot"):
    pred = best_model.predict(Xte)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(yte, pred, s=14, alpha=0.6)
    lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    ax.plot(lim, lim, "--", color="orange")
    ax.set_xlabel("Actual score")
    ax.set_ylabel("Predicted score")
    ax.set_title(f"{best_name}: predicted vs actual")
    fig.tight_layout()
    fig.savefig(f"{prefix}_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    imp = permutation_importance(best_model, Xte, yte, n_repeats=15, random_state=SEED)
    imp = pd.Series(imp.importances_mean, index=Xte.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Permutation importance (drop in R²)")
    fig.tight_layout()
    fig.savefig(f"{prefix}_feature_importance.png", dpi=150)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser(description="Student performance prediction")
    ap.add_argument("--csv", help="Path to your dataset (CSV). Omit to use synthetic demo data.")
    ap.add_argument("--target", default="final_score", help="Name of the score column to predict")
    ap.add_argument("--risk-threshold", type=float, default=60, help="Scores below this are 'at risk'")
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else make_synthetic_data()
    if not args.csv:
        print("No --csv given: using SYNTHETIC demo data (results are illustrative only).\n")
        df.to_csv("synthetic_students.csv", index=False)

    X, y = prepare(df, args.target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Rows: {len(X)} | Features: {X.shape[1]} | Train/Test: {len(Xtr)}/{len(Xte)}\n")

    # 1. Predict the final score
    reg, fitted = run_regression(Xtr, Xte, ytr, yte)
    print("=== Regression: predicting the final score ===")
    print(reg.round(3).to_string(index=False), "\n")

    best = reg.loc[0, "Model"]
    importance = save_plots(best, fitted[best], Xte, yte)
    print(f"Best model: {best}")
    print("Top features by permutation importance:")
    print(importance.round(3).to_string(), "\n")

    # 2. Flag at-risk students
    cls, matrices = run_classification(X, y, args.risk_threshold)
    print(f"=== Classification: at risk = score < {args.risk_threshold:g} ===")
    print(cls.round(3).to_string(index=False))
    for name, cm in matrices.items():
        print(f"\nConfusion matrix, {name} (rows = actual, cols = predicted; 0 = on track, 1 = at risk)")
        print(cm)

    print("\nSaved: plot_pred_vs_actual.png, plot_feature_importance.png")


if __name__ == "__main__":
    main()
