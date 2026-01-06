"""
Real Estate Price Predictor - Model Training Script (V3 - Fixed Compatibility)

Usage:
    python train_model.py

This script:
- Loads AmesHousing CSV/XLSX from a few possible paths
- Selects numeric and categorical features, cleans data
- Encodes categorical features, scales numeric features
- Trains an XGBoost regressor
- Evaluates performance (R2, RMSE, MAE)
- Saves artifacts in formats suitable for deployment:
    - XGBoost native JSON booster (model.json)
    - sklearn wrapper via joblib (model.pkl)
    - scaler parameters (scaler_params.json)
    - feature names, importance, model_info, app_config
    - cleaned CSV for visualizations (AmesHousing_clean.csv)
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import xgboost and joblib
try:
    import xgboost as xgb
    USE_XGBOOST = True
except ImportError:
    USE_XGBOOST = False
    print("❌ XGBoost not found! Please install it: pip install xgboost")
    raise SystemExit(1)

try:
    import joblib
except ImportError:
    print("❌ joblib not found! Please install it: pip install joblib")
    raise SystemExit(1)

print("=" * 60)
print("🏠 Real Estate Price Predictor - Training (V3 Fixed)")
print("=" * 60)

# ============================================
# STEP 1: Load Data
# ============================================
print("\n📂 Loading data...")

possible_paths = [
    'AmesHousing.csv',
    'AmesHousing.xlsx',
    'data/AmesHousing.csv',
    'data/AmesHousing.xlsx',
]

df_ames = None
loaded_path = None
for path in possible_paths:
    try:
        if path.endswith('.xlsx'):
            df_ames = pd.read_excel(path)
        else:
            df_ames = pd.read_csv(path)
        loaded_path = path
        print(f"✅ Loaded data from: {path}")
        break
    except FileNotFoundError:
        continue

if df_ames is None:
    print("❌ Could not find AmesHousing data file!")
    print("   Please place AmesHousing.csv or AmesHousing.xlsx in the same folder as this script or in a data/ subfolder.")
    raise SystemExit(1)

print(f"   Shape: {df_ames.shape}")

# ============================================
# STEP 2: Select Features
# ============================================
print("\n🔧 Selecting features...")

NUMERIC_FEATURES = [
    'Overall Qual',
    'Gr Liv Area',
    'Garage Cars',
    'Garage Area',
    'Total Bsmt SF',
    '1st Flr SF',
    'Year Built',
    'Full Bath',
    'Year Remod/Add',
    'TotRms AbvGrd',
    'Lot Area',
]

CATEGORICAL_FEATURES = [
    'Neighborhood',
    'House Style',
    'Exterior 1st',
]

TARGET = 'SalePrice'

# Validate columns exist
missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if c not in df_ames.columns]
if missing_cols:
    print(f"❌ Missing expected columns in dataset: {missing_cols}")
    raise SystemExit(1)

# ============================================
# STEP 3: Clean & Save Visualization Data
# ============================================
print("\n🧹 Cleaning data...")

VIZ_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
df_clean = df_ames[VIZ_COLUMNS].copy()

# Fill missing values
for col in NUMERIC_FEATURES:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in CATEGORICAL_FEATURES:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Save for visualizations
df_clean.to_csv('AmesHousing_clean.csv', index=False)
print(f"✅ Saved clean data: AmesHousing_clean.csv")

# ============================================
# STEP 4: Prepare Features
# ============================================
print("\n🔄 Encoding features...")

# Save dropdown options
dropdown_options = {
    'neighborhoods': sorted(df_clean['Neighborhood'].unique().tolist()),
    'house_styles': sorted(df_clean['House Style'].unique().tolist()),
    'exterior': sorted(df_clean['Exterior 1st'].unique().tolist()),
}

# Save feature ranges
feature_ranges = {}
for col in NUMERIC_FEATURES:
    feature_ranges[col] = {
        'min': float(df_clean[col].min()),
        'max': float(df_clean[col].max()),
        'median': float(df_clean[col].median())
    }

# One-hot encode categorical features (drop_first to avoid multicollinearity)
df_encoded = pd.get_dummies(df_clean, columns=CATEGORICAL_FEATURES, drop_first=True)

X = df_encoded.drop(TARGET, axis=1)
y = df_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# ============================================
# STEP 5: Scale Features
# ============================================
print("\n📏 Scaling features...")

scaler = StandardScaler()
numeric_cols_in_X = [col for col in NUMERIC_FEATURES if col in X_train.columns]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if numeric_cols_in_X:
    X_train_scaled[numeric_cols_in_X] = scaler.fit_transform(X_train[numeric_cols_in_X])
    X_test_scaled[numeric_cols_in_X] = scaler.transform(X_test[numeric_cols_in_X])
else:
    print("⚠️ No numeric columns found in training set to scale.")

# Save scaler parameters as JSON (more compatible than joblib for some deployments)
scaler_params = {
    'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [],
    'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [],
    'var': scaler.var_.tolist() if hasattr(scaler, 'var_') else [],
    'feature_names': numeric_cols_in_X
}

with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)
print("✅ Saved: scaler_params.json")

# ============================================
# STEP 6: Train Model
# ============================================
print("\n🚀 Training XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)

model.fit(X_train_scaled, y_train)
print("✅ Model trained!")

# ============================================
# STEP 7: Evaluate
# ============================================
print("\n📊 Evaluating model...")

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)")
print(f"   RMSE: ${rmse:,.0f}")
print(f"   MAE: ${mae:,.0f}")

# ============================================
# STEP 8: Save Everything (Compatible Format)
# ============================================
print("\n💾 Saving files (compatible format)...")

# 1) Save native XGBoost booster JSON (portable)
try:
    booster = model.get_booster()
    booster.save_model('model.json')
    print("✅ Saved: model.json (XGBoost native format via booster)")
except Exception as e:
    print(f"❌ Failed to save booster JSON: {e}")

# 2) Save sklearn wrapper with joblib (for reloading in Python)
try:
    joblib.dump(model, 'model.pkl')
    print("✅ Saved: model.pkl (sklearn wrapper via joblib)")
except Exception as e:
    print(f"❌ Failed to save sklearn wrapper with joblib: {e}")

# 3) Save feature names as JSON
feature_names = list(X_train_scaled.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print("✅ Saved: feature_names.json")

# 4) Save feature importance (if available)
try:
    if hasattr(model, 'feature_importances_'):
        importance = {name: float(imp) for name, imp in zip(feature_names, model.feature_importances_)}
        with open('feature_importance.json', 'w') as f:
            json.dump(importance, f, indent=2)
        print("✅ Saved: feature_importance.json")
except Exception as e:
    print(f"❌ Failed to save feature importance: {e}")

# 5) Save model info
model_info = {
    'model_name': 'XGBoost',
    'r2_score': float(r2),
    'rmse': float(rmse),
    'mae': float(mae),
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'feature_columns': feature_names,
    'n_features': len(feature_names)
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("✅ Saved: model_info.json")

# 6) Save app config (dropdowns, ranges)
app_config = {
    'dropdown_options': dropdown_options,
    'feature_ranges': feature_ranges,
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES
}

with open('app_config.json', 'w') as f:
    json.dump(app_config, f, indent=2)
print("✅ Saved: app_config.json")

# ============================================
# DONE!
# ============================================
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"""
Files created (expected):
  ✅ model.json              (XGBoost booster - native format)
  ✅ model.pkl               (sklearn wrapper via joblib)
  ✅ scaler_params.json      (Scaler parameters)
  ✅ feature_names.json      (Feature column names)
  ✅ feature_importance.json (Feature importance scores)
  ✅ model_info.json         (Model metadata)
  ✅ app_config.json         (App configuration)
  ✅ AmesHousing_clean.csv   (Data for visualizations)

Model Performance:
  📊 R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)
  💰 Average Error (MAE): ${mae:,.0f}

Next steps:
  - Upload these files to your hosting environment (Hugging Face, S3, etc.)
  - Ensure inference code applies the same scaling and feature ordering before prediction.
""")
