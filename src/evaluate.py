import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import utils
import json

def fetch_metrics(y_true, y_pred):
    """Calculates evaluation metrics for regression."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def evaluate_model(data_dir, model_path, outputs_dir="outputs"):
    """
    Evaluates the XGBoost model, producing metrics and plots.
    """
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    print("Loading test dataset...")
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_test = test_df.drop('price', axis=1)
    y_test = test_df['price']

    print("Predicting...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse, mae, r2 = fetch_metrics(y_test, y_pred)
    
    # Also evaluate train/val for reference
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    y_train = train_df['price']
    y_train_pred = model.predict(train_df.drop('price', axis=1))
    
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    y_val = val_df['price']
    y_val_pred = model.predict(val_df.drop('price', axis=1))

    # Compile Results
    metrics_dict = {
        "Train": {"RMSE": rmse, "MAE": mae, "R2": r2}, # Placeholder, real below
        "Validation": {},
        "Test": {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
    }
    
    train_rmse, train_mae, train_r2 = fetch_metrics(y_train, y_train_pred)
    val_rmse, val_mae, val_r2 = fetch_metrics(y_val, y_val_pred)
    
    metrics_dict["Train"] = {"RMSE": float(train_rmse), "MAE": float(train_mae), "R2": float(train_r2)}
    metrics_dict["Validation"] = {"RMSE": float(val_rmse), "MAE": float(val_mae), "R2": float(val_r2)}

    # Save to metrics.json
    utils.save_json(metrics_dict, os.path.join(outputs_dir, "metrics.json"))

    # Save to metrics_table.csv
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.to_csv(os.path.join(outputs_dir, "metrics_table.csv"))
    
    print("\n--- Model Evaluation Metrics ---")
    print(metrics_df)

    # Plots
    print("\nGenerating Evaluation Plots...")

    # 1. Predicted vs Actual scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual Price (Test Set)')
    plt.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"))
    plt.close()

    # 2. Residual histogram
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=40)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.savefig(os.path.join(plots_dir, "residual_histogram.png"))
    plt.close()

    # 3. Feature importance bar chart
    importance = model.feature_importances_
    features = X_test.columns
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
    plt.close()

    print(f"Metrics and plots successfully saved in {outputs_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data", help="Data directory.")
    parser.add_argument("--model", type=str, default="models/xgboost_model.json", help="Path to trained model.")
    
    args = parser.parse_args()
    evaluate_model(args.data, args.model)
