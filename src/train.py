import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
import argparse
import os
import utils

def train_xgboost(data_dir, model_out="models/xgboost_model.json"):
    """
    Trains an XGBoost Regressor using Optuna for hyperparameter tuning.
    Validation set is used for early stopping.
    """
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    
    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    
    X_val = val_df.drop('price', axis=1)
    y_val = val_df['price']

    # We use Optuna here because it provides a highly efficient, Bayesian optimization approach
    # compared to RandomizedSearchCV. It handles early stopping naturally within the objective function.
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'eval_metric': 'rmse'
        }
        
        # XGBoost handles early stopping natively with `early_stopping_rounds` in .fit()
        model = xgb.XGBRegressor(**params)
        
        # Fit with evaluation set and early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # We need the best score from the early stopped model
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        return rmse

    print("Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20) # 20 trials for demonstration

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    print("\nTraining final model with best parameters and early stopping...")
    best_params = study.best_params
    best_params['random_state'] = 42
    
    final_model = xgb.XGBRegressor(**best_params)
    
    # Train final model on train set, stopping on validation
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Save the model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    final_model.save_model(model_out)
    print(f"\nFinal model saved to {model_out}")

    # Also save feature names to help Streamlit and SHAP
    utils.save_json({"features": list(X_train.columns)}, "models/feature_names.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model.")
    parser.add_argument("--data", type=str, default="data", help="Directory containing train/val/test CSVs.")
    parser.add_argument("--output", type=str, default="models/xgboost_model.json", help="Path to save the trained model.")
    
    args = parser.parse_args()
    train_xgboost(args.data, args.output)
