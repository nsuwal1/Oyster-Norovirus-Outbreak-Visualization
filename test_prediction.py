import pandas as pd
import joblib
import json
from pathlib import Path

# Paths
MODEL_PATH = Path("models/lightgbm_model_20260302_150003.pkl")
PARAMS_PATH = Path("models/lightgbm_model_20260302_150003_params.json")
TEST_DATA_PATH = Path("Testing.csv")

# Load model and params
loaded = joblib.load(MODEL_PATH)
model = loaded['model']
with open(PARAMS_PATH, "r") as f:
    params = json.load(f)

# Load test data
df = pd.read_csv(TEST_DATA_PATH)
print("Columns in test data:", df.columns.tolist())

# Prepare features
features_df = df.copy()
drop_cols = []
if "Date" in features_df.columns:
    features_df["Date"] = pd.to_datetime(features_df["Date"], errors="coerce")
    drop_cols.append("Date")
if "ID" in features_df.columns:
    drop_cols.append("ID")
if drop_cols:
    features_df = features_df.drop(columns=drop_cols)

print("Columns after dropping Date/ID:", features_df.columns.tolist())

# Select only training features
feature_names = params.get("feature_names", [])
if feature_names:
    features_df = features_df[feature_names]

print("Selected features:", features_df.columns.tolist())
print("Dtypes:", features_df.dtypes)

# Convert categorical
categorical_features = params.get("categorical_features", [])
for col in categorical_features:
    if col in features_df.columns:
        features_df[col] = features_df[col].astype('category')

# Test prediction
try:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)
        print("Prediction successful, shape:", proba.shape)
    else:
        print("Model has no predict_proba")
except Exception as e:
    print(f"Error: {e}")