import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os

# Set page config
st.set_page_config(
    page_title="Sri Lanka Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    /* Gradient animated glowing background for prediction */
    .prediction-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(42, 82, 152, 0.3);
        margin-top: 1rem;
        animation: slideUpFade 0.7s ease-out forwards;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .price-value {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin: 15px 0 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        letter-spacing: -1px;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 201, 255, 0.4);
        border: none;
    }
    
    /* Clean up the Streamlit UI slightly */
    MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Define paths correctly based on project structure (app is in src/app)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_PATH = os.path.join(OUTPUTS_DIR, 'model.json')
MAPPINGS_PATH = os.path.join(OUTPUTS_DIR, 'mappings.json')


@st.cache_resource(show_spinner="Loading XGBoost Model...")
def load_model():
    model_path = os.path.join(OUTPUTS_DIR, 'model.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Invalidate cache trick
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

@st.cache_data
def load_mappings():
    with open(MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)
    return mappings

@st.cache_data
def load_brand_models():
    # Load mapping of which brand owns which models
    path = os.path.join(OUTPUTS_DIR, 'brand_models.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

try:
    model = load_model()
    mappings = load_mappings()
    brand_models = load_brand_models()
except Exception as e:
    st.error(f"⚠️ Error loading model or mappings: {e}")
    st.info("Please make sure the model has been trained and outputs/model.json exists.")
    st.stop()
    
brand_mapping = mappings.get('Brand', {})
model_mapping = mappings.get('Model', {})
location_cols = mappings.get('Location_Columns', [])

# Extract valid locations for the UI by removing 'Loc_' prefix
valid_locations = [loc.replace('Loc_', '') for loc in location_cols if loc != 'Loc_Other']
if 'Loc_Other' in location_cols:
    valid_locations.append('Other')
valid_locations.sort()

# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🚗 Sri Lanka Vehicle Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 3rem;'>AI and ML powered market valuation for your vehicle</p>", unsafe_allow_html=True)


# Create main layout with 3 columns to center the content beautifully
left_spacer, main_col, right_spacer = st.columns([1, 2, 1])

with main_col:
    st.markdown("### 📋 Enter Vehicle Details")
    
    # Input container
    with st.container():
        c1, c2 = st.columns(2)
        
        with c1:
            brand_options = sorted(list(brand_mapping.keys()))
            default_brand_idx = brand_options.index("Toyota") if "Toyota" in brand_options else 0
            selected_brand = st.selectbox("Vehicle Brand", options=brand_options, index=default_brand_idx)
            
            # Extract models dynamically
            available_models = brand_models.get(selected_brand, [])
            if not available_models:
                # Fallback if no specific models found for brand
                available_models = ["Unknown"]
                
            model_options = sorted(available_models)
            selected_model = st.selectbox("Vehicle Model", options=model_options)
            
            selected_location = st.selectbox("Registered Location", options=valid_locations)
            
        with c2:
            selected_year = st.slider("Manufacture Year", min_value=1990, max_value=2024, value=2018, step=1)
            
            selected_mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=45000, step=1000)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Generate Price Estimate 🎯", use_container_width=True)

# Inference logic
if predict_btn:
    with st.spinner("Analyzing current market trends..."):
        # Features mapping
        brand_encoded = brand_mapping.get(selected_brand, 0)
        model_encoded = model_mapping.get(selected_model, 0)
        
        loc_features = {}
        for col in location_cols:
            loc_features[col] = 0
            
        loc_key = f"Loc_{selected_location}"
        if loc_key in location_cols:
            loc_features[loc_key] = 1
        elif "Loc_Other" in location_cols:
            loc_features["Loc_Other"] = 1
                
        # Consolidate features
        input_data = {
            'Mileage': [selected_mileage],
            'Year': [selected_year],
            'Brand_Encoded': [brand_encoded],
            'Model_Encoded': [model_encoded]
        }
        for col in location_cols:
            input_data[col] = [loc_features[col]]
            
        input_df = pd.DataFrame(input_data)
        
        try:
            prediction = model.predict(input_df)[0]
            
            # Prevent negative predictions out of model boundaries
            if prediction < 0:
                prediction = 150000 
                
            formatted_price = f"Rs {prediction:,.0f}"
            
            with main_col:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.8; margin-bottom: 5px;">Estimated Market Value</div>
                    <div class="price-value">{formatted_price}</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">For a {selected_year} {selected_brand} {selected_model} with {selected_mileage:,} km</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple explainability
                st.markdown("---")
                st.markdown(f"**💡 Insight:** Values are primarily driven by **Year ({selected_year})** and **Mileage**, alongside Sri Lanka's dynamic market conditions for **{selected_brand} {selected_model}** vehicles.")

        except Exception as e:
            st.error(f"Prediction calculation failed: {e}")
