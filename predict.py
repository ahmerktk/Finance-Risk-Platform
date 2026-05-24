import numpy as np
import joblib

MODELS_DIR = "models"

FEATURES = ["age", "income", "debt_ratio", "monthly_expenses",
            "late_payments", "num_loans", "dependents"]

CLUSTER_NAMES = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
    3: "Very High Risk"
}


def _load(filename):
    return joblib.load(f"{MODELS_DIR}/{filename}")

# -------------------------------------------------------------

def _scale(input_dict):
    scaler = _load("scaler.pkl")
    values = [[input_dict[f] for f in FEATURES]]
    return scaler.transform(values)

# -------------------------------------------------------------

# Default Predictions
#  Returns predictions from all 4 classifiers.
def predict_default(input_dict):
    X = _scale(input_dict)
    results = []

    for model_name, filename in [
        ("Logistic Regression", "logistic_regression.pkl"),
        ("Random Forest",       "random_forest.pkl"),
        ("XGBoost",             "xgboost.pkl"),
        ("ANN",                 "ann.pkl"),
    ]:
        model = _load(filename)
        pred  = model.predict(X)[0]
        prob  = model.predict_proba(X)[0][1]
        results.append({
            "model":       model_name,
            "prediction":  "Default" if pred == 1 else "No Default",
            "probability": round(float(prob), 4),
        })

    return results


# CLV Prediction
# Returns predicted Customer Lifetime Value
def predict_clv(input_dict):
    X = _scale(input_dict)
    model = _load("linear_regression.pkl")
    clv = model.predict(X)[0]
    return round(float(clv), 2)

# -------------------------------------------------------------

# Cluster Assignment
# Returns cluster label and name for a single customer
def predict_cluster(input_dict):
    X = _scale(input_dict)
    model = _load("kmeans.pkl")
    label = int(model.predict(X)[0])
    return label, CLUSTER_NAMES.get(label, f"Cluster {label}")

# -------------------------------------------------------------

# Assigns cluster labels to a whole DataFrame.
# df must contain the 7 feature columns.
# Returns: numpy array of cluster labels.
def predict_clusters_bulk(df):
    scaler = _load("scaler.pkl")
    model  = _load("kmeans.pkl")
    X = scaler.transform(df[FEATURES])
    return model.predict(X)
