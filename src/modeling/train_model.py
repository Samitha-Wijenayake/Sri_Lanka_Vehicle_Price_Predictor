import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def train_and_evaluate(input_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # --- SAVE MAPPINGS FOR WEB APP ---
    import json
    mappings = {}
    
    # 1. Brand and Model Mapping
    if 'Brand' in df.columns and 'Brand_Encoded' in df.columns:
        brand_map = df[['Brand', 'Brand_Encoded']].drop_duplicates().set_index('Brand')['Brand_Encoded'].to_dict()
        mappings['Brand'] = {str(k): int(v) for k, v in brand_map.items()}
        
    if 'Model' in df.columns and 'Model_Encoded' in df.columns:
        model_map = df[['Model', 'Model_Encoded']].drop_duplicates().set_index('Model')['Model_Encoded'].to_dict()
        mappings['Model'] = {str(k): int(v) for k, v in model_map.items()}
    
    # 2. Location Columns
    # Identify Loc_* columns
    loc_cols = [c for c in df.columns if c.startswith('Loc_')]
    mappings['Location_Columns'] = loc_cols
    
    # Save mappings
    with open(os.path.join('outputs', 'mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=4)
    print("Mappings saved to outputs/mappings.json")
    # ---------------------------------
    
    # Feature Selection
    # Drop target and non-feature columns
    # We use Brand_Encoded and Location Encoded columns
    # Dropping: Title, Price, Brand, Location, Location_Clean, Price_Normalized, Mileage_Normalized (using raw scaled)
    
    # Identify feature columns
    # We need to drop: Title, Price, Brand, Location, Location_Clean, Price_Normalized, Mileage_Normalized
    # We KEEP: Mileage, Year, Brand_Encoded, and Loc_* columns
    
    # Drop known non-feature columns
    drop_cols = ['Title', 'Price', 'Brand', 'Model', 'Location', 'Location_Clean', 'Price_Normalized', 'Mileage_Normalized', 'Description', 'PublishedDate', 'Link', 'ImageURL']
    if 'Unnamed: 0' in df.columns:
        drop_cols.append('Unnamed: 0')
        
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Ensure all remaining columns are numeric
    # Identify non-numeric columns
    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        print(f"Warning: Dropping remaining non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)
        
    y = df['Price']
    
    print("Feature Data Types:")
    print(X.dtypes)
    print(f"Features: {X.columns.tolist()}")
    print(f"Target: Price")
    
    # Train/Test Split (80/20)
    print("Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    print("Starting RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    print(f"Best Parameters: {random_search.best_params_}")

    # Save the model
    model_path = os.path.join('outputs', 'model.json')
    best_model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = f"""
    Model Performance Results:
    --------------------------
    Best Parameters: {random_search.best_params_}
    
    RMSE: {rmse:,.2f}
    MAE: {mae:,.2f}
    R2 Score: {r2:.4f}
    """
    
    print(results)
    
    with open('model_results.txt', 'w') as f:
        f.write(results)
        
    # Visualizations
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price (XGBoost)")
    plt.ticklabel_format(style='plain', axis='both') 
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png')
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(best_model, max_num_features=15, height=0.5)
    plt.title("Top 15 Feature Importance")
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    # 3. Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.title("Residuals Distribution")
    plt.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
    plt.savefig('plots/residuals.png')
    plt.close()
    
    print("Plots saved to plots/ directory.")

if __name__ == "__main__":
    train_and_evaluate('data/processed/vehicle_data_withprocessed.csv')
