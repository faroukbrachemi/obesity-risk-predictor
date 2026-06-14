# Obesity Risk Predictor — Streamlit App

Lifestyle-based multi-class Obesity Risk Predictor.
Trains five models (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM),
keeps the best, and serves it through an interactive form.

## Files

| File | Purpose |
|------|---------|
| `preprocessing.py` | Shared encoding + feature engineering (used by training **and** the app) |
| `train_and_save.py` | Trains all 5 models, picks the best on a holdout, saves `model.pkl` |
| `app.py` | Streamlit UI: input form → predicted category + probability chart |
| `requirements.txt` | Dependencies |

## Setup

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train and save the best model (one-time, CPU only)
python train_and_save.py

# 3. launch the app
streamlit run app.py
```

The app reads `model.pkl`. If it is missing, the app tells you to run step 2 first.
