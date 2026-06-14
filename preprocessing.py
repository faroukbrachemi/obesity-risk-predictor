"""
Shared preprocessing for the Obesity Risk Predictor.

Both the training script and the Streamlit app import from here, so the feature
engineering and encoding are guaranteed to be identical at train and inference time.
"""

import pandas as pd

# The 7 obesity classes in their natural order (insufficient -> highest obesity).
SORTED_LABELS = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]

# Raw column groups (before any encoding).
CATEGORICAL_FEATURES = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]
NUMERIC_FEATURES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Order mapping for the naturally ordinal habit columns.
LEVELS = {"Always": 3, "Frequently": 2, "Sometimes": 1, "no": 0}


def _engineer_features(df):
    """Add the domain features. Expects FAVC_no to already exist (from one-hot)."""
    df = df.copy()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Physical_Activity_Level"] = df["FAF"] - df["TUE"]
    df["Meal_Habits"] = df["FCVC"] * df["NCP"]
    df["Healthy_Nutrition_Habits"] = df["FCVC"] / (2 * df["FAVC_no"] - 1)
    df["Tech_Usage_Score"] = df["TUE"] / df["Age"]
    return df


def build_features(raw_df, encoder):
    """
    Turn a raw input dataframe (same schema as the Kaggle data, minus the target)
    into the final model-ready feature matrix, using an already-fitted OneHotEncoder.
    """
    raw_df = raw_df.copy()

    # 1) One-hot encode the categorical columns with the fitted encoder.
    encoded = pd.DataFrame(
        encoder.transform(raw_df[CATEGORICAL_FEATURES]),
        columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        index=raw_df.index,
    )

    # 2) Merge the rare 'CALC_Always' level into 'CALC_Frequently' for stability.
    combine = [c for c in ["CALC_Always", "CALC_Frequently"] if c in encoded.columns]
    if combine:
        encoded["CALC_Always|Frequently"] = encoded[combine].sum(axis=1)
        encoded = encoded.drop(columns=combine)

    # 3) Ordinal encodings that preserve the natural order of the habit columns.
    raw_df["CALC_ord"] = raw_df["CALC"].map(LEVELS)
    raw_df["CAEC_ord"] = raw_df["CAEC"].map(LEVELS)

    # 4) Assemble numeric + ordinal + one-hot, then add engineered features.
    base = raw_df.drop(columns=CATEGORICAL_FEATURES)
    out = pd.concat([base, encoded], axis=1)
    out = _engineer_features(out)
    return out
