# Obesity Risk Predictor — Streamlit App

Lifestyle-based multi-class classifier for the **Kaggle Playground Series S4E2** task.
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

# 2. download the dataset from Kaggle (Playground Series S4E2)
#    and place train.csv in this folder

# 3. train and save the best model (one-time, CPU only)
python train_and_save.py

# 4. launch the app
streamlit run app.py
```

The app reads `model.pkl`. If it is missing, the app tells you to run step 3 first.

## Notes

- Everything runs on **CPU** — no GPU needed.
- The same `build_features()` function is used at training and inference time, so a
  single user input is encoded exactly the way the training data was. Columns are
  re-aligned to the training feature set before prediction.
- BMI is intentionally kept as a feature; it is the dominant signal because the target
  categories are partly BMI-derived. The interesting modelling challenge is separating
  *adjacent* risk levels.
