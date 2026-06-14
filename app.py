"""
Obesity Risk Predictor - Streamlit app.

Run:
    streamlit run app.py

Loads the model saved by train_and_save.py, collects a lifestyle profile through a
form, and predicts the obesity-risk category with per-class probabilities.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from preprocessing import build_features

st.set_page_config(page_title="Obesity Risk Predictor", page_icon="⚖️", layout="centered")


@st.cache_resource
def load_artifact(path="model.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


art = load_artifact()
if art is None:
    st.error(
        "**model.pkl not found.** Train the model first:\n\n"
        "1. Put the Kaggle S4E2 `train.csv` in this folder\n"
        "2. Run `python train_and_save.py`\n"
        "3. Reload this app"
    )
    st.stop()

model = art["model"]
encoder = art["encoder"]
feature_columns = art["feature_columns"]
labels = art["labels"]

st.title("⚖️ Obesity Risk Predictor")
st.caption(f"Multi-class lifestyle-based classification · best model: {art['model_name']}")
st.write(
    "Fill in the profile below. The model predicts the obesity-risk category "
    "and shows the probability it assigns to every level."
)

# -------------------- input form --------------------
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 10, 100, 25)
    height = st.slider("Height (m)", 1.30, 2.10, 1.70, 0.01)
    weight = st.slider("Weight (kg)", 30.0, 180.0, 70.0, 0.5)
    family = st.selectbox("Family history of overweight", ["yes", "no"])
    favc = st.selectbox("Frequently eats high-calorie food (FAVC)", ["yes", "no"])
    fcvc = st.slider("Vegetable consumption frequency (FCVC)", 1.0, 3.0, 2.0, 0.1)
    ncp = st.slider("Main meals per day (NCP)", 1.0, 4.0, 3.0, 0.1)
with col2:
    caec = st.selectbox("Snacking between meals (CAEC)",
                        ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Smokes", ["yes", "no"])
    ch2o = st.slider("Daily water in liters (CH2O)", 1.0, 3.0, 2.0, 0.1)
    scc = st.selectbox("Monitors calorie intake (SCC)", ["yes", "no"])
    faf = st.slider("Physical activity per week (FAF)", 0.0, 3.0, 1.0, 0.1)
    tue = st.slider("Time on tech devices (TUE)", 0.0, 2.0, 1.0, 0.1)
    calc = st.selectbox("Alcohol consumption (CALC)",
                        ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportation (MTRANS)",
                          ["Public_Transportation", "Automobile", "Walking",
                           "Motorbike", "Bike"])

# -------------------- prediction --------------------
if st.button("Predict", type="primary"):
    raw = pd.DataFrame([{
        "Gender": gender, "Age": age, "Height": height, "Weight": weight,
        "family_history_with_overweight": family, "FAVC": favc, "FCVC": fcvc,
        "NCP": ncp, "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o, "SCC": scc,
        "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
    }])

    # encode + engineer exactly as in training, then align columns
    X = build_features(raw, encoder)
    X = X.reindex(columns=feature_columns, fill_value=0)

    pred = int(np.asarray(model.predict(X)).ravel()[0])
    bmi = weight / (height ** 2)

    st.success(f"Predicted category: **{labels[pred].replace('_', ' ')}**")
    st.metric("BMI (for reference)", f"{bmi:.1f}")

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X)).ravel()
        proba_df = pd.DataFrame(
            {"Probability": proba},
            index=[l.replace("_", " ") for l in labels],
        )
        st.subheader("Probability per risk level")
        st.bar_chart(proba_df)

    st.caption(
        "Note: BMI is the dominant predictor because the risk categories are partly "
        "defined from it. The model's harder job is separating adjacent levels."
    )
