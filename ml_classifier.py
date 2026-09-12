import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from severity_rules import compute_visual_severity

def train_ml_models(csv_path="logs/eda/dataset_features.csv", output_dir="logs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs/eda", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset features: {len(df)} samples")

    # Generate pseudo-labels using rule-based severity logic
    pseudo_labels = []
    for _, row in df.iterrows():
        feat_dict = {
            "has_crack": bool(row["has_crack"]),
            "crack_pixel_count": int(row["crack_pixel_count"]),
            "max_width_px": float(row["max_width_px"]),
            "length_px": float(row["length_px"]),
            "affected_area_pct": float(row["affected_area_pct"])
        }
        sev_res = compute_visual_severity(feat_dict)
        pseudo_labels.append(sev_res["severity"])

    df["severity"] = pseudo_labels
    print("Pseudo-label distribution across dataset:")
    print(df["severity"].value_counts())

    feature_cols = ["length_px", "max_width_px", "affected_area_pct", "branch_points", "component_count"]
    X = df[feature_cols]
    y = df["severity"]

    # Train / Test split (80 / 20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 1. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_rep = classification_report(y_test, rf_pred, output_dict=True)
    classes = sorted(y.unique().tolist())
    rf_cm = confusion_matrix(y_test, rf_pred, labels=classes)

    # 2. Support Vector Machine (Comparison Model)
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, random_state=42))
    ])
    svm_pipe.fit(X_train, y_train)
    svm_pred = svm_pipe.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_rep = classification_report(y_test, svm_pred, output_dict=True)
    svm_cm = confusion_matrix(y_test, svm_pred, labels=classes)

    report = {
        "metadata": {
            "description": "Classical ML models trained on rule-based pseudo-labels as validation layer",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features_used": feature_cols,
            "classes": classes
        },
        "random_forest": {
            "test_accuracy": round(float(rf_acc), 4),
            "macro_f1": round(float(rf_rep["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(rf_rep["weighted avg"]["f1-score"]), 4),
            "confusion_matrix": rf_cm.tolist(),
            "per_class": {c: rf_rep[c] for c in classes if c in rf_rep},
            "feature_importances": {col: round(float(imp), 4) for col, imp in zip(feature_cols, rf.feature_importances_)}
        },
        "svm_comparison": {
            "test_accuracy": round(float(svm_acc), 4),
            "macro_f1": round(float(svm_rep["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(svm_rep["weighted avg"]["f1-score"]), 4),
            "confusion_matrix": svm_cm.tolist(),
            "per_class": {c: svm_rep[c] for c in classes if c in svm_rep}
        }
    }

    # Save models
    joblib.dump(rf, "models/rf_classifier.joblib")
    joblib.dump(svm_pipe, "models/svm_classifier.joblib")

    # Save metrics report
    rep_path = os.path.join(output_dir, "ml_classifier_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {rep_path}")

    # Plot & Save Feature Importance
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    plt.figure(figsize=(7, 4))
    colors = ["#38bdf8", "#818cf8", "#a78bfa", "#f472b6", "#fb7185"]
    importances.plot(kind="barh", color=colors[:len(importances)])
    plt.title("Random Forest Feature Importance for Severity", fontsize=12, fontweight="bold")
    plt.xlabel("Importance (Gini impurity decrease)", fontsize=10)
    plt.tight_layout()
    fi_plot = "logs/eda/rf_feature_importance.png"
    plt.savefig(fi_plot, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {fi_plot}")

    # Boxplot of features per severity class
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(["length_px", "max_width_px", "affected_area_pct", "branch_points"]):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x="severity", y=col, data=df, order=["Low", "Medium", "High"], palette={"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"})
        plt.title(f"{col} by Severity Class", fontweight="bold")
    plt.tight_layout()
    box_plot = "logs/eda/severity_feature_boxplots.png"
    plt.savefig(box_plot, dpi=150)
    plt.close()
    print(f"Severity boxplots saved to {box_plot}")

    return report

if __name__ == "__main__":
    train_ml_models()
