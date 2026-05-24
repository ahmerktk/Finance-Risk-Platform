import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Where trained models are saved
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ["age", "income", "debt_ratio", "monthly_expenses",
            "late_payments", "num_loans", "dependents"]


# Synthetic Dataset
def generate_dataset(n=1000, seed=42):
    """
    Generates a synthetic customer dataset.
    Replace this with:  pd.read_csv("data/customers.csv")
    """
    rng = np.random.default_rng(seed)

    age               = rng.integers(22, 70, n)
    income            = rng.uniform(20000, 120000, n)
    debt_ratio        = rng.uniform(0.05, 0.85, n)
    monthly_expenses  = income / 12 * rng.uniform(0.3, 0.9, n)
    late_payments     = rng.integers(0, 10, n)
    num_loans         = rng.integers(0, 6, n)
    dependents        = rng.integers(0, 5, n)

    # Default risk: higher debt + late payments = more likely to default
    risk_score = (debt_ratio * 0.4 + late_payments / 10 * 0.4 +
                  num_loans / 6 * 0.2)
    default = (risk_score + rng.normal(0, 0.05, n) > 0.45).astype(int)

    # Credit score (for ANN regression): inverse of risk
    credit_score = np.clip(850 - risk_score * 400 +
                           rng.normal(0, 20, n), 300, 850)

    # CLV (for linear regression)
    clv = income * 3.5 - debt_ratio * 50000 + rng.normal(0, 5000, n)

    df = pd.DataFrame({
        "age": age, "income": income, "debt_ratio": debt_ratio,
        "monthly_expenses": monthly_expenses, "late_payments": late_payments,
        "num_loans": num_loans, "dependents": dependents,
        "default": default, "credit_score": credit_score, "clv": clv
    })
    return df


# Train Models
def train_all():
    print("📦 Generating dataset...")
    df = generate_dataset()

    X = df[FEATURES]
    y_class  = df["default"]          # classification target
    y_score  = df["credit_score"]     # ANN regression target
    y_clv    = df["clv"]              # linear regression target

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    X_train, X_test, yc_train, yc_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42
    )

    metrics = {}

    # ── 1. Logistic Regression ──────────────────
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, yc_train)
    joblib.dump(lr, f"{MODELS_DIR}/logistic_regression.pkl")
    metrics["Logistic Regression"] = _clf_metrics(lr, X_test, yc_test)

    # ── 2. Random Forest ───────────────────────
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, yc_train)
    joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
    metrics["Random Forest"] = _clf_metrics(rf, X_test, yc_test)

    # ── 3. XGBoost ─────────────────────────────
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False,
                        eval_metric="logloss", random_state=42)
    xgb.fit(X_train, yc_train)
    joblib.dump(xgb, f"{MODELS_DIR}/xgboost.pkl")
    metrics["XGBoost"] = _clf_metrics(xgb, X_test, yc_test)

    # ── 4. ANN — MLPClassifier for credit score bucket ──
    print("Training ANN (MLPClassifier)...")
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                        random_state=42)
    ann.fit(X_train, yc_train)
    joblib.dump(ann, f"{MODELS_DIR}/ann.pkl")
    metrics["ANN"] = _clf_metrics(ann, X_test, yc_test)

    # ── 5. Linear Regression (CLV) ─────────────
    print("Training Linear Regression (CLV)...")
    _, X_test_r, _, y_clv_test = train_test_split(
        X_scaled, y_clv, test_size=0.2, random_state=42
    )
    lin = LinearRegression()
    lin.fit(X_scaled, y_clv)      # train on all for CLV
    joblib.dump(lin, f"{MODELS_DIR}/linear_regression.pkl")

    # ── 6. K-Means (Customer Segmentation) ─────
    print("Training K-Means (k=4)...")
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_scaled)
    joblib.dump(km, f"{MODELS_DIR}/kmeans.pkl")

    print("\n✅ All models trained and saved to /models\n")
    _print_metrics(metrics)
    return metrics

# -------------------------------------------------------------

def _clf_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_prob), 4),
    }

# -------------------------------------------------------------

def _print_metrics(metrics):
    print(f"{'Model':<22} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10}")
    print("-" * 54)
    for name, m in metrics.items():
        print(f"{name:<22} {m['accuracy']:>10} {m['f1_score']:>8} {m['roc_auc']:>10}")

# -------------------------------------------------------------

if __name__ == "__main__":
    train_all()
