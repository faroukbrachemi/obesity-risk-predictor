
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from preprocessing import CATEGORICAL_FEATURES, SORTED_LABELS, build_features

# XGBoost and LightGBM need the OpenMP runtime (libomp) on macOS. If it isn't
# installed we skip them gracefully so the app still trains on the other models.
# CatBoost bundles its own runtime; LogisticRegression / RandomForest are pure sklearn.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:  # ImportError or XGBoostError (missing libomp)
    HAS_XGB = False
    print(f"[skip] XGBoost unavailable: {type(e).__name__}")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception as e:
    HAS_LGBM = False
    print(f"[skip] LightGBM unavailable: {type(e).__name__}")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception as e:
    HAS_CATBOOST = False
    print(f"[skip] CatBoost unavailable: {type(e).__name__}")

TRAIN_CSV = "train.csv"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 0


def main():
    # ---- load ----
    df = pd.read_csv(TRAIN_CSV)
    if "id" in df.columns:
        df = df.set_index("id")

    y = df["NObeyesdad"].map({l: i for i, l in enumerate(SORTED_LABELS)})
    X_raw = df.drop(columns=["NObeyesdad"])

    # ---- fit encoder + build features ----
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_raw[CATEGORICAL_FEATURES])
    X = build_features(X_raw, encoder)
    feature_columns = X.columns.tolist()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    n_classes = int(y.nunique())

    # ---- candidate models (CPU-friendly, fixed defaults) ----
    # Always-available models (no libomp needed):
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    # Optional models, added only if their library loaded successfully:
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            loss_function="MultiClass", random_seed=RANDOM_STATE, verbose=0,
        )
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, tree_method="hist",
            objective="multi:softmax", num_class=n_classes,
            verbosity=0, random_state=RANDOM_STATE,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=9,
            objective="multiclass", num_class=n_classes,
            random_state=RANDOM_STATE, verbosity=-1,
        )

    print(f"\nTraining {len(models)} models: {', '.join(models)}\n")

    # ---- train + evaluate ----
    best_name, best_model, best_acc = None, None, -1.0
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, np.asarray(model.predict(X_val)).ravel())
        print(f"{name:20s} val acc = {acc:.4f}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} (val acc = {best_acc:.4f})")
    print("Refitting the best model on all data...")
    best_model.fit(X, y)

    # ---- save everything the app needs ----
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "encoder": encoder,
            "feature_columns": feature_columns,
            "labels": SORTED_LABELS,
        },
        MODEL_PATH,
    )
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
