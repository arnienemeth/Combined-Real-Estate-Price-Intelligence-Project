# 🏠 Real Estate Price Predictor (V3 - Fixed Compatibility)

This version uses JSON format for all saved files to avoid version compatibility issues with Hugging Face Spaces.

## 🔧 What's Fixed

- ✅ No more `numpy._core` errors
- ✅ No more XGBoost version warnings  
- ✅ All files saved as JSON (universal format)
- ✅ Compatible with Hugging Face Spaces

## 🚀 Quick Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements_local.txt
```

### Step 2: Add Your Data
Place `AmesHousing.csv` in this folder.

### Step 3: Train Model
```bash
python train_model.py
```

### Step 4: Upload to Hugging Face (9 files)
Upload these files to your Space:

1. `app.py`
2. `requirements.txt`
3. `model.json`
4. `scaler_params.json`
5. `feature_names.json`
6. `feature_importance.json`
7. `model_info.json`
8. `app_config.json`
9. `AmesHousing_clean.csv`

## 📁 Files Explained

| File | Format | Purpose |
|------|--------|---------|
| `model.json` | XGBoost native | Trained model |
| `scaler_params.json` | JSON | Scaler mean/std values |
| `feature_names.json` | JSON | Column names |
| `feature_importance.json` | JSON | Feature importance scores |
| `model_info.json` | JSON | Model metadata |
| `app_config.json` | JSON | Dropdown options |
| `AmesHousing_clean.csv` | CSV | Data for visualizations |

## 📊 Visualizations

- Price Gauge (speedometer)
- Scatter Plot (your property vs market)
- Price Distribution (histogram)
- Neighborhood Comparison
- Quality Box Plots
- Year vs Price Trend
- Feature Importance

---
Built with ❤️ using XGBoost + Gradio + Plotly
