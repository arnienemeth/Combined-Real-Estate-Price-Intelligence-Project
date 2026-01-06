Real Estate Price Intelligence — Combined Project
Production‑oriented demo combining a house price predictor (Ames Housing) with market trends and forecasting (Zillow ZHVI). The project demonstrates end‑to‑end machine learning and deep learning practices, reproducible artifact management on AWS S3, and a shareable web demo hosted on Hugging Face Spaces.

Live demo / examples: https://huggingface.co/spaces/Arnie1980/real-estate-predictor

Project overview
Purpose  
Deliver a compact, portfolio‑ready application that showcases practical ML/DL skills across the full lifecycle: data acquisition, exploration, feature engineering, model training (regression + time series), evaluation, visualization, and cloud‑backed deployment.

What the app does

Price Predictor: Input house features → predicted sale price (regression).

Market Trends: Visualize historical home‑value indices (ZHVI) by region.

Forecast: Short‑term price forecasts using time‑series models.

Compare: Compare a property to neighborhood averages and similar listings.

Interactive UI: Gradio/Streamlit interface with charts and gauges for quick interpretation.

Key features & highlights
Models

XGBoost regression for price prediction (Ames Housing).

Time‑series model(s) for market trend forecasting (Zillow ZHVI).

Visualizations

Plotly gauges, histograms, scatter plots, and feature‑importance charts.

Static assets included for README compatibility.

UI & Hosting

Gradio (or Streamlit) app for single and batch predictions.

Hosted on Hugging Face Spaces for instant sharing.

Cloud & Reproducibility

AWS S3 used to store datasets, model artifacts, and outputs.

Saved artifacts: model.json (XGBoost booster), model.pkl (sklearn wrapper), scaler_params.json, feature_names.json, model_info.json, app_config.json, AmesHousing_clean.csv.

Performance (example)

Example regression metrics reported in training logs: R² ≈ 0.908, RMSE ≈ $27k, MAE ≈ $16k (your results may vary depending on preprocessing and model settings).

Architecture & process workflow
High‑level flow

Kód
Data (Ames CSV, Zillow ZHVI) → S3 storage
      ↓
Data exploration & cleaning (notebooks)
      ↓
Feature engineering (numeric scaling, one‑hot encoding)
      ↓
Model training
  • XGBoost regressor (price)
  • Time‑series model (ZHVI forecasting)
      ↓
Evaluation & tuning (R², RMSE, MAE; forecast accuracy)
      ↓
Save artifacts → S3 (model.json, model.pkl, scaler_params.json, metadata)
      ↓
Serve via Gradio/Streamlit app (loads artifacts, applies same preprocessing)
      ↓
Hosted on Hugging Face Spaces (public demo)
Components

Data sources: Ames Housing (Kaggle) and Zillow ZHVI (Kaggle / Zillow).

Training environment: Local / Colab / SageMaker (optional for larger runs).

Storage: AWS S3 for datasets and model artifacts.

Serving: Gradio/Streamlit app that loads saved artifacts and applies identical preprocessing.

Hosting: Hugging Face Spaces for public demo and sharing.

Quick start (local)
Clone

bash
git clone https://github.com/arnienemeth/Combined-Real-Estate-Price-Intelligence-Project
cd real-estate-intelligence
Create environment & install

bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
Place data

Put AmesHousing.csv and Zillow ZHVI files in data/ or upload to S3 and update paths in notebooks/scripts.

Run training

bash
python train_model.py
This script produces model.json, model.pkl, scaler_params.json, feature_names.json, model_info.json, app_config.json, and AmesHousing_clean.csv.

Run the app

bash
python app.py
The app loads saved artifacts, applies the same preprocessing, and serves predictions and visualizations.

Files & artifacts
train_model.py — training pipeline for XGBoost regression (Ames).

app.py — Gradio/Streamlit app that loads artifacts and serves predictions + visualizations.

scripts/export_visuals.py — generate static Plotly images for README.

requirements.txt — Python dependencies.

Saved artifacts (output):

model.json — XGBoost booster (native JSON).

model.pkl — sklearn wrapper (joblib).

scaler_params.json — scaler parameters for reproducible preprocessing.

feature_names.json — ordered feature list.

feature_importance.json — importance scores.

model_info.json — metadata and performance metrics.

app_config.json — UI dropdowns and ranges.

AmesHousing_clean.csv — cleaned subset for visualizations.

AWS & deployment notes
S3: Use S3 to store raw datasets, cleaned CSVs, and model artifacts. Organize with:

Kód
s3://your-bucket/
  ├─ data/ames/
  ├─ data/zillow/
  ├─ models/price-predictor/
  └─ output/
IAM: create a least‑privilege IAM user for automation (S3 read/write).

Training scale: use Colab for quick GPU runs; use SageMaker for larger experiments and managed endpoints.

Hosting: Hugging Face Spaces hosts the Gradio app; S3 provides persistent storage for artifacts. Ensure the app loads artifacts from local files or S3 (via boto3) depending on deployment.

Best practices & next steps
Reproducibility: commit requirements.txt, save random seeds, and store artifacts in S3.

Explainability: add SHAP or LIME to explain predictions.

CI/CD: add a CI step to regenerate static visuals and validate model artifact integrity.

Extensions: multi‑region forecasting, multi‑model ensembling, or a marketplace UI for uploading custom CSVs.

Contact & contribution
Project by Arnold Nemeth.
Feedback, issues, and pull requests are welcome. Include the live demo link and a short note when opening PRs.
