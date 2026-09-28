"""
train_model.py
---------------
Trains a Random Forest classifier on the "Rain in Australia" dataset
and saves the fitted model + preprocessing objects to the model/ folder.

Run this ONCE before starting the Streamlit app:
    python train_model.py

Requires: data/weatherAUS.csv (download instructions in README.md)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

RANDOM_STATE = 42
SAMPLE_FRACTION = 1.0        # use the full dataset by default; lower this (e.g. 0.3) for faster training
DATA_PATH = os.path.join("data", "weatherAUS.csv")
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Download weatherAUS.csv from Kaggle "
        "(see README.md) and place it inside the data/ folder."
    )

print("Loading data...")
raw_df = pd.read_csv(DATA_PATH)

if SAMPLE_FRACTION < 1.0:
    raw_df = raw_df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE).reset_index(drop=True)

raw_df.dropna(subset=["RainTomorrow"], inplace=True)
print("Data shape after dropping missing targets:", raw_df.shape)

# ---------------------------------------------------------------
# 2. Train / validation split (chronological)
# ---------------------------------------------------------------
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
val_df = raw_df[year >= 2015]

input_cols = list(raw_df.columns)[1:-1]  # exclude Date and RainTomorrow
target_col = "RainTomorrow"

train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes(include="object").columns.tolist()

# ---------------------------------------------------------------
# 3. Preprocessing: impute, scale, encode
# ---------------------------------------------------------------
print("Fitting preprocessors...")
imputer = SimpleImputer(strategy="mean").fit(raw_df[numeric_cols])
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

def preprocess(df_inputs):
    df_inputs = df_inputs.copy()
    df_inputs[numeric_cols] = imputer.transform(df_inputs[numeric_cols])
    df_inputs[numeric_cols] = scaler.transform(df_inputs[numeric_cols])
    df_inputs[encoded_cols] = encoder.transform(df_inputs[categorical_cols])
    return df_inputs[numeric_cols + encoded_cols]

X_train = preprocess(train_inputs)
X_val = preprocess(val_inputs)

# ---------------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------------
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, train_targets)

# ---------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------
def evaluate(X, y, pos_label="Yes"):
    preds = model.predict(X)
    pos_idx = list(model.classes_).index(pos_label)
    probs = model.predict_proba(X)[:, pos_idx]
    y_binary = (y == pos_label).astype(int)
    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds, pos_label=pos_label, zero_division=0),
        "Recall": recall_score(y, preds, pos_label=pos_label, zero_division=0),
        "F1 Score": f1_score(y, preds, pos_label=pos_label, zero_division=0),
        "ROC AUC": roc_auc_score(y_binary, probs),
    }

print("\nValidation metrics:")
for k, v in evaluate(X_val, val_targets).items():
    print(f"  {k}: {v:.4f}")

metrics = evaluate(X_val, val_targets)
os.makedirs("output", exist_ok=True)
results_path = os.path.join("output", "model_results.csv")
row = pd.DataFrame([{
    "experiment": f"n_estimators={model.n_estimators}, max_depth={model.max_depth}",
    **metrics,
}])
row.to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)    

# ---------------------------------------------------------------
# 6. Save model + preprocessors
# ---------------------------------------------------------------
print("\nSaving model and preprocessors to model/ ...")
joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
joblib.dump(
    {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoded_cols": encoded_cols,
        "input_cols": input_cols,
    },
    os.path.join(MODEL_DIR, "columns.pkl"),
)

print("Done! You can now run: streamlit run app.py")
