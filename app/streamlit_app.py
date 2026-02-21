import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import json
import matplotlib.pyplot as plt
import os
import pickle

st.set_page_config(page_title="Sri Lanka Vehicle Price Predictor", layout="wide", page_icon="🚗")

# --- 1. Load Resources (no caching to always pick up latest files) ---
def load_resources():
    # Load Model
    model = xgb.XGBRegressor()
    model_path = "models/xgboost_model.json"
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        model = None
        
    # Load Mappings (classes from LabelEncoders)
    mappings_path = "models/mappings.json"
    if os.path.exists(mappings_path):
        with open(mappings_path, "r") as f:
            mappings = json.load(f)
    else:
        mappings = {}
        
    # Load Feature Names
    features_path = "models/feature_names.json"
    if os.path.exists(features_path):
         with open(features_path, "r") as f:
             feature_names = json.load(f)["features"]
    else:
        feature_names = []

    # Load Metrics
    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f).get("Test", {})
    else:
        metrics = {}

    return model, mappings, feature_names, metrics

model, mappings, feature_names, metrics = load_resources()

# Helper: check if a column has real options (more than just "Unknown")
def has_real_options(col_name):
    opts = mappings.get(col_name, [])
    return len(opts) > 0 and opts != ["Unknown"]

def get_options(col_name):
    return mappings.get(col_name, ["Unknown"])

# --- 2. Build UI ---
st.title("Sri Lanka Vehicle Price Predictor")
st.markdown("Predict the used car price in Sri Lanka using Machine Learning, powered by an **XGBoost Regressor** trained on dataset scraped from Ikman.lk.")

# Sidebar - Model Metrics Summary
st.sidebar.header("Model Metrics(Test Set)")
if metrics:
    st.sidebar.metric("RMSE (Error)", f"Rs. {metrics.get('RMSE', 0):,.2f}")
    st.sidebar.metric("MAE (Avg Error)", f"Rs. {metrics.get('MAE', 0):,.2f}")
    st.sidebar.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
else:
    st.sidebar.warning("Model metrics not found. Train the model first.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Explainability**\n\nThe SHAP Global Summary shows the general trend of which features impact the price over the entire training data.")
if os.path.exists("outputs/plots/shap_summary.png"):
    st.sidebar.image("outputs/plots/shap_summary.png", use_container_width=True, caption="SHAP Global Summary")
elif os.path.exists("outputs/plots/feature_importance.png"):
     st.sidebar.image("outputs/plots/feature_importance.png", use_container_width=True, caption="XGB Feature Importance")


# Central Input Area
st.subheader("Enter Vehicle Details")
col1, col2, col3 = st.columns(3)

input_data = {}
location = None  # Initialize for Loc_ one-hot encoding later

with col1:
    if has_real_options("Brand"):
        brand = st.selectbox("Brand", options=get_options("Brand"))
        input_data["Brand"] = get_options("Brand").index(brand)
    if has_real_options("Model"):
        model_name = st.selectbox("Model", options=get_options("Model"))
        input_data["Model"] = get_options("Model").index(model_name)
    if "Year" in feature_names:
        year = st.number_input("Year of Manufacture", min_value=1950, max_value=2025, value=2015)
        input_data["Year"] = int(year)

with col2:
    if "Mileage" in feature_names:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=50000)
        input_data["Mileage"] = float(mileage)
    if has_real_options("Transmission"):
        transmission = st.selectbox("Transmission", options=get_options("Transmission"))
        input_data["Transmission"] = get_options("Transmission").index(transmission)
    if has_real_options("FuelType"):
        fuel_type = st.selectbox("Fuel Type", options=get_options("FuelType"))
        input_data["FuelType"] = get_options("FuelType").index(fuel_type)

with col3:
    if has_real_options("Condition"):
        condition = st.selectbox("Condition", options=get_options("Condition"))
        input_data["Condition"] = get_options("Condition").index(condition)
    if has_real_options("Location"):
        location = st.selectbox("Location", options=get_options("Location"))
        input_data["Location"] = get_options("Location").index(location)
    if "EngineCapacity" in feature_names:
        engine_cap = st.number_input("Engine Capacity (cc)", min_value=0, max_value=10000, value=1500)
        input_data["EngineCapacity"] = float(engine_cap)


# Prediction Logic
if st.button("🔍 Predict Price", type="primary"):
    if model is None or not feature_names:
        st.error("Model or Feature mapping not found. Please train the model and save assets.")
    else:
        # Align features perfectly to model's training order
        ordered_input = []
        for feat in feature_names:
             # Calculate dynamic features based on user input
             if feat == "Brand_Encoded":
                 ordered_input.append(input_data.get("Brand", 0))
             elif feat == "Model_Encoded":
                 ordered_input.append(input_data.get("Model", 0))
             elif feat.startswith("Loc_"):
                 # One-hot: set to 1.0 if the location name matches
                 loc_name = feat.replace("Loc_", "")
                 ordered_input.append(1.0 if location and loc_name == location else 0.0)
             elif feat == "Price_Normalized":
                 ordered_input.append(0.0)  # Cannot know price ahead of time
             elif feat == "Mileage_Normalized":
                 ordered_input.append(input_data.get("Mileage", 0) / 100000.0)
             else:
                 ordered_input.append(input_data.get(feat, 0))

        df_input = pd.DataFrame([ordered_input], columns=feature_names)

        # Predict
        predicted_price = model.predict(df_input)[0]

        # Display Result
        st.success(f"### Predicted Price: **Rs. {predicted_price:,.2f}**")
        
        # SHAP Local Explanation
        st.subheader("Explaining this Prediction (Local SHAP)")
        st.markdown(
            "The following waterfall plot illustrates how the specific features of this vehicle pushed the price up or down from the general baseline (expected value). "
            "Features in **red** increased the vehicle's price, while features in **blue** decreased it."
        )
        
        # We need the TreeExplainer for local explanations
        explainer_path = "models/shap_explainer.pkl"
        try:
            with open(explainer_path, "rb") as f:
                explainer = pickle.load(f)
                
            plt.close("all")  # Close any previous figures
            # Get SHAP values for the single row
            shap_values = explainer(df_input)
            
            # Matplotlib shap local waterfall plot
            shap.plots.waterfall(shap_values[0], show=False, max_display=20)
            
            # Grab the current figure and resize it
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            fig.tight_layout()
            
            st.pyplot(fig)
            plt.close("all")
        except Exception as e:
            st.error(f"Could not load SHAP explainer for local explanation. Have you run `python src/explain.py`?\n Error details: {e}")
