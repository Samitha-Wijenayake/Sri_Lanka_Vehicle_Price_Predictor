import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import argparse

def run_shap_analysis(data_dir, model_path, plots_dir="outputs/plots"):
    """
    Runs SHAP TreeExplainer on the XGBoost model to generate global explainability plots.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    print("Loading test dataset for SHAP explanation...")
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_test = test_df.drop('price', axis=1)

    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    # Summary plot: violin/dot plot of impacts
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary.png"))
    plt.close()

    # Find the top feature based on mean absolute SHAP value
    vals = pd.DataFrame(shap_values.values, columns=X_test.columns).abs().mean().sort_values(ascending=False)
    top_feature = vals.index[0]
    print(f"Top feature identified: {top_feature}")

    print(f"Generating SHAP dependence plot for {top_feature}...")
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(top_feature, shap_values.values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_dependence.png"))
    plt.close()

    # Save explainer metadata/values for Streamlit
    import pickle
    with open("models/shap_explainer.pkl", "wb") as f:
         pickle.dump(explainer, f)
    
    print(f"SHAP plots successfully saved in {plots_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data", help="Data directory.")
    parser.add_argument("--model", type=str, default="models/xgboost_model.json", help="Path to trained model.")
    
    args = parser.parse_args()
    run_shap_analysis(args.data, args.model)
