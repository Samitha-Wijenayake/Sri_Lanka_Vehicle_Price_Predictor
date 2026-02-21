## Live Demo

**Try the app here:** 
[https://srilankavehiclepricepredictor.streamlit.app/](https://srilankavehiclepricepredictor.streamlit.app/)

# Sri Lanka Vehicle Price Predictor (Machine Learning Assignment)

This repository contains a complete end-to-end Machine Learning pipeline to predict used vehicle prices in Sri Lanka. The project conforms to the assignment guidelines, utilizing scraped data from `ikman.lk` and employing an **XGBoost Regressor** along with a **Streamlit** front-end. Optuna was used for hyperparameter tuning, and early stopping was implemented on the validation set.

It also extensively uses **SHAP** (`TreeExplainer`) for global and local model explainability.

## Project Structure

```
.
├── app/
│   └── streamlit_app.py    # The Streamlit Front-End application
├── data/                   # Directory containing CSV files (raw, processed, train, val, test)
├── models/                 # Output directory for the saved XGBoost model, encoders, feature names, and SHAP explainer
├── outputs/
│   ├── metrics.json        # Evaluation metrics in JSON format
│   ├── metrics_table.csv   # Evaluation metrics as a CSV table
│   └── plots/              # Directory containing saved plots (RMSE, residuals, Feature Importance, SHAP summaries)
├── src/
│   ├── utils.py            # Shared helper functions (JSON/Pickle loading & saving)
│   ├── preprocess.py       # Data cleaning, feature engineering, Label Encoding, Train/Val/Test splitting
│   ├── train.py            # XGBoost training script with Optuna HPO and early stopping
│   ├── evaluate.py         # Script to calculate regression metrics and plot charts
│   └── explain.py          # SHAP TreeExplainer rendering and plot generation
├── scrape.py               # Web scraping script for ikman.lk vehicle ads
├── report_outline.md       # Markdown structure outlining the final report to be converted to PDF
├── requirements.txt        # Python package dependencies
└── README.md               # You're reading it
```

## Quick Start Guide

Follow these sequential steps to run the pipeline end-to-end.

### 1. Install Requirements

First, ensure you have Python 3.8+ installed. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Data Scraping (Optional)

If you need to fetch fresh data (respecting the server's load with ethical limits):

```bash
python scrape.py --pages 5 --output data/raw_vehicles.csv
```
*(By default, this script provides dummy structure execution to not abuse live sites. Use your provided `data.csv` if ready).*

### 3. Preprocess the Data

The preprocessing script handles Sri Lankan currency (`Rs`, `Lakh`, `Mn`), detects the target `price` column, cleans numerical columns, applies `LabelEncoder` to categorical features, and performs a **70% Train, 15% Validation, 15% Test** split.

```bash
python src/preprocess.py --input data/data.csv --output data
```
*(Replace `data/data.csv` with your actual raw CSV file path).*

### 4. Train the Model

This script trains an **XGBoost** regressor, utilizing **Optuna** for Bayesian Hyperparameter Optimization and the validation set generated in the previous step for **Early Stopping**.

```bash
python src/train.py --data data --output models/xgboost_model.json
```

### 5. Evaluate the Model

After training, evaluate the model on the isolated Test set. This generates `RMSE`, `MAE`, `R2` metrics and saves visualizations (Actual vs Predicted, Residuals, Feature Importance).

```bash
python src/evaluate.py --data data --model models/xgboost_model.json
```

### 6. Generate Explanations (SHAP)

Execute global and feature explanations using the SHAP `TreeExplainer`. This creates the SHAP Summary plot and SHAP Dependence plot. 

```bash
python src/explain.py --data data --model models/xgboost_model.json
```

### 7. Launch the Streamlit Front-End

Start the Streamlit application to display the interactive UI. The app takes user inputs for prediction, shows model metrics, and renders global/local SHAP visual explanations.

```bash
streamlit run app/streamlit_app.py
```

