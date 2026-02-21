import pandas as pd
import numpy as np
import argparse
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import utils

def clean_price(price_str):
    """
    Cleans price strings handling "Rs", commas, "Lakh", "Mn", etc.
    """
    if pd.isna(price_str):
        return np.nan
    
    price_str = str(price_str).lower().replace(',', '').strip()
    
    # Remove currency symbols/words
    price_str = re.sub(r'rs\.?|lkr|rupees?', '', price_str).strip()
    
    multiplier = 1
    if 'lakh' in price_str:
        multiplier = 100000
        price_str = price_str.replace('lakh', '').strip()
    elif 'mn' in price_str or 'million' in price_str:
        multiplier = 1000000
        price_str = re.sub(r'mn|million', '', price_str).strip()
        
    # Extract only valid numeric parts (allowing decimals)
    numeric_match = re.search(r'\d+(\.\d+)?', price_str)
    
    if numeric_match:
        val = float(numeric_match.group())
        return val * multiplier
    return np.nan

def clean_mileage(mileage_str):
    """
    Cleans mileage strings.
    """
    if pd.isna(mileage_str):
        return np.nan
    
    m_str = str(mileage_str).lower().replace(',', '').strip()
    
    # Extract numeric part
    numeric_match = re.search(r'\d+(\.\d+)?', m_str)
    if numeric_match:
        return float(numeric_match.group())
    return np.nan

def preprocess_data(input_path, output_dir="data"):
    """
    Loads, cleans, engineers features, encodes, and splits the data.
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find the input file at {input_path}")
        return

    # 1. Target Column Check
    if 'price' not in df.columns:
        print("\nWARNING: Target column 'price' NOT found in dataset.")
        print("Please ensure your dataset has a 'price' column or update the script.")
        # Attempt minimal fallback detection
        possible_price = [c for c in df.columns if 'price' in c.lower()]
        if possible_price:
            print(f"Auto-detecting price column as: '{possible_price[0]}'")
            df.rename(columns={possible_price[0]: 'price'}, inplace=True)
            print("Renamed to 'price'.")
        else:
            raise KeyError("Cannot proceed without a target column 'price'.")
    else:
        print("Target column 'price' validated.")

    print(f"Original shape: {df.shape}")

    # 2. Cleaning Features
    df['price'] = df['price'].apply(clean_price)
    
    # Drop rows where price couldn't be parsed
    df = df.dropna(subset=['price'])
    
    # Assume we might have 'mileage', 'year'
    if 'Mileage' in df.columns:
        df['Mileage'] = df['Mileage'].apply(clean_mileage)
        # Impute missing mileage with median
        df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())

    if 'Year' in df.columns:
        # Simple year conversion
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Year'] = df['Year'].fillna(df['Year'].mode()[0]) # Replace with mode

    # 3. Categorical Encoding
    cat_columns = ['Brand', 'Model', 'Transmission', 'FuelType', 'Condition', 'Location']
    
    label_encoders = {}
    mappings_dict = {}

    for col in cat_columns:
        if col in df.columns:
            # Fill NA for categorical
            df[col] = df[col].fillna('Unknown')
            
            le = LabelEncoder()
            # Fit and transform
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
            # Store mapping for frontend
            mappings_dict[col] = list(le.classes_)
            
            print(f"Encoded '{col}' with {len(le.classes_)} classes.")

    # Save encoders and mappings for inference
    utils.save_json(mappings_dict, "models/mappings.json")
    
    # Save the label encoder objects directly using pickle if needed
    for col, le in label_encoders.items():
         utils.save_pickle(le, f"models/le_{col}.pkl")

    # Final cleanup (drop unneeded text columns like title/description if they exist)
    drop_cols = ['Title', 'Description', 'url', 'id', 'Link', 'ImageURL', 'PublishedDate', 'Location_Clean']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Processed shape: {df.shape}")

    # 4. Train / Validation / Test Split (70% / 15% / 15%)
    print("\nSplitting data: 70% Train, 15% Validation, 15% Test...")
    
    X = df.drop('price', axis=1)
    y = df['price']

    # Split: Train=70%, Temp=30%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Split Temp into Val=15% (half of 30%) and Test=15% (half of 30%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    
    # Combine back to save full sets
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Also save a full processed dataset just in case
    df.to_csv(os.path.join(output_dir, "processed.csv"), index=False)

    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split vehicle dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input raw CSV.")
    parser.add_argument("--output", type=str, default="data", help="Output directory for processed files.")
    
    args = parser.parse_args()
    preprocess_data(args.input, args.output)
