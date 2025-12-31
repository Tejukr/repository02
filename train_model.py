# train_model.py
"""
Train an improved Gradient Boosting model for house price prediction with feature engineering.

Outputs:
  models/model.pkl            -> dict { model, imputer, scaler, log_target }
  models/model_features.pkl   -> list of feature names
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost for even better performance
try:
    import xgboost as xgb
    USE_XGBOOST = True
    print("[OK] Using XGBoost for better performance")
except ImportError:
    USE_XGBOOST = False
    print("[WARNING] XGBoost not available, using GradientBoostingRegressor")
    print("   Install with: pip install xgboost")

# -----------------------------
# Paths
# -----------------------------
DATA_CSV = os.path.join("data", "house_data.csv")
MODELS_DIR = "models"
MODEL_OUT = os.path.join(MODELS_DIR, "model.pkl")
FEATURES_OUT = os.path.join(MODELS_DIR, "model_features.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_CSV)
print(f"Loaded rows: {len(df):,}")
print("Columns:", df.columns.tolist())

if "Price" not in df.columns:
    raise RuntimeError("[ERROR] CSV must contain a 'Price' column.")

# -----------------------------
# Clean Target & Remove Outliers
# -----------------------------
print("\n[STEP] Cleaning data...")
df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0]

# More aggressive outlier removal
initial_rows = len(df)
df = df[df["Price"] <= df["Price"].quantile(0.995)]
df = df[df["Price"] >= df["Price"].quantile(0.005)]
df = df[df["living area"] <= df["living area"].quantile(0.995)]
df = df[df["living area"] >= df["living area"].quantile(0.005)]
print(f"   Removed {initial_rows - len(df)} outliers ({initial_rows - len(df):,} rows)")

# -----------------------------
# Feature Engineering
# -----------------------------
print("\n[STEP] Engineering features...")

# Calculate house age (assuming current year is 2024, or use max year in data)
current_year = datetime.now().year
if "Built Year" in df.columns:
    df["house_age"] = current_year - df["Built Year"].fillna(current_year)
    df["house_age"] = df["house_age"].clip(lower=0, upper=200)

# Renovation status
if "Renovation Year" in df.columns:
    df["is_renovated"] = (df["Renovation Year"] > 0).astype(int)
    df["years_since_renovation"] = current_year - df["Renovation Year"].fillna(0)
    df["years_since_renovation"] = df["years_since_renovation"].clip(lower=0, upper=200)
else:
    df["is_renovated"] = 0
    df["years_since_renovation"] = 0

# Area ratios and derived features
if "living area" in df.columns and "lot area" in df.columns:
    df["living_to_lot_ratio"] = df["living area"] / (df["lot area"] + 1)  # +1 to avoid division by zero
    df["total_area"] = df["living area"] + df.get("Area of the basement", 0).fillna(0)

if "living area" in df.columns and "number of bedrooms" in df.columns:
    df["area_per_bedroom"] = df["living area"] / (df["number of bedrooms"] + 1)

if "living area" in df.columns and "number of bathrooms" in df.columns:
    df["area_per_bathroom"] = df["living area"] / (df["number of bathrooms"] + 1)

# Basement ratio
if "Area of the basement" in df.columns and "living area" in df.columns:
    df["basement_ratio"] = df["Area of the basement"] / (df["living area"] + df["Area of the basement"] + 1)

# -----------------------------
# Select Features (including new engineered ones)
# -----------------------------
base_features = [
    "living area",
    "number of bedrooms",
    "number of bathrooms",
    "number of floors",
    "condition of the house",
    "grade of the house",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year"
]

# Additional features if available
additional_features = []
if "lot area" in df.columns:
    additional_features.append("lot area")
if "waterfront present" in df.columns:
    additional_features.append("waterfront present")
if "number of views" in df.columns:
    additional_features.append("number of views")
if "Number of schools nearby" in df.columns:
    additional_features.append("Number of schools nearby")
if "Distance from the airport" in df.columns:
    additional_features.append("Distance from the airport")

# Engineered features
engineered_features = []
if "house_age" in df.columns:
    engineered_features.append("house_age")
if "is_renovated" in df.columns:
    engineered_features.append("is_renovated")
if "years_since_renovation" in df.columns:
    engineered_features.append("years_since_renovation")
if "living_to_lot_ratio" in df.columns:
    engineered_features.append("living_to_lot_ratio")
if "total_area" in df.columns:
    engineered_features.append("total_area")
if "area_per_bedroom" in df.columns:
    engineered_features.append("area_per_bedroom")
if "area_per_bathroom" in df.columns:
    engineered_features.append("area_per_bathroom")
if "basement_ratio" in df.columns:
    engineered_features.append("basement_ratio")

feature_cols = base_features + additional_features + engineered_features

# Validate feature presence
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f"[WARNING] Missing features (will be skipped): {missing}")
    feature_cols = [c for c in feature_cols if c in df.columns]

print(f"   Using {len(feature_cols)} features:")
print(f"   - Base: {len(base_features)}")
print(f"   - Additional: {len(additional_features)}")
print(f"   - Engineered: {len(engineered_features)}")

# -----------------------------
# Prepare Features & Target
# -----------------------------
X = df[feature_cols].copy()
y = df["Price"].copy()

# Use log1p transform for smoother training
y_log = np.log1p(y)

# -----------------------------
# Impute Missing Numeric Data
# -----------------------------
print("\n[STEP] Imputing missing values...")
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# -----------------------------
# Feature Scaling (helps with gradient boosting)
# -----------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_scaled, y_log, test_size=0.20, random_state=42, shuffle=True
)

print(f"\n[INFO] Training set: {len(X_train):,} samples")
print(f"[INFO] Test set: {len(X_test):,} samples")

# -----------------------------
# Train Model (Gradient Boosting or XGBoost)
# -----------------------------
print("\n[STEP] Training model...")
if USE_XGBOOST:
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
else:
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        max_features='sqrt',
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        verbose=0
    )

model.fit(X_train, y_train_log)

# -----------------------------
# Evaluate Performance
# -----------------------------
print("\n[STEP] Evaluating model...")
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test_log)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# Calculate percentage errors
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n" + "="*60)
print("[RESULT] MODEL PERFORMANCE")
print("="*60)
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")
print("="*60)

# Compare with baseline (mean prediction)
baseline_mae = mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))
baseline_rmse = np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_true.mean())))
print(f"\n[INFO] Improvement over baseline (mean):")
print(f"   MAE improvement:  {((baseline_mae - mae) / baseline_mae * 100):.1f}%")
print(f"   RMSE improvement: {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")

# -----------------------------
# Feature Importance
# -----------------------------
importances = (
    pd.Series(model.feature_importances_, index=feature_cols)
    .sort_values(ascending=False)
)

print("\n[INFO] TOP 15 FEATURE IMPORTANCES")
print("="*60)
for i, (feature, importance) in enumerate(importances.head(15).items(), 1):
    print(f"{i:2d}. {feature:35s} : {importance:.4f}")
print("="*60)

# -----------------------------
# Save Model + Imputer + Scaler
# -----------------------------
print("\n[STEP] Saving model...")
with open(MODEL_OUT, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "log_target": True
        },
        f
    )

with open(FEATURES_OUT, "wb") as f:
    pickle.dump(feature_cols, f)

print(f"   [OK] Saved model -> {MODEL_OUT}")
print(f"   [OK] Saved feature list -> {FEATURES_OUT}")

# -----------------------------
# Sample Predictions (Safe)
# -----------------------------
print("\n[SAMPLE] Sample predictions (first 10 rows)")
print("="*60)

sample_indices = min(10, len(X_test))
sample_actual = np.expm1(y_test_log.values[:sample_indices])
sample_pred = np.expm1(model.predict(X_test.values[:sample_indices]))
sample_error = np.abs(sample_actual - sample_pred)
sample_error_pct = (sample_error / sample_actual) * 100

sample_df = pd.DataFrame({
    "Actual Price": sample_actual,
    "Predicted Price": sample_pred,
    "Error": sample_error,
    "Error %": sample_error_pct
})

print(sample_df.to_string(index=False))
print("="*60)
