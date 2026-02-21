# Machine Learning Assignment Report
**Project:** Sri Lanka Vehicle Price Prediction  

## 1. Introduction
- **Problem Statement:** Predicting the price of used vehicles in Sri Lanka using scraped data.
- **Task Type:** Regression.
- **Data Source:** Scraping vehicle advertisements from **Ikman.lk**.

## 2. Selection of a New Machine Learning Algorithm
- **Chosen Algorithm:** XGBoost (Extreme Gradient Boosting).
- **Justification:** Why XGBoost was selected over basic algorithms (e.g., Linear Regression, Decision Trees).
  - Ability to handle non-linear relationships.
  - Built-in regularization (L1/L2) preventing overfitting.
  - Native handling of missing values.
  - Ensemble method (boosting) providing superior predictive performance over single or bagging models (like Random Forest).
- **Difference from Basic Models:** Brief theoretical comparison.

## 3. Data Preprocessing and Feature Engineering
- **Handling Currency:** How the prices with "Rs", "Lakh", "Mn", and commas were cleaned into numeric format.
- **Handling Missing Values:** Strategy for missing mileage and year.
- **Categorical Encoding:** Usage of `LabelEncoder` for features like `brand`, `model`, `transmission`, `fuel_type`, `condition`, and `location`.
- **Target Variable Checking:** Ensuring `price` is precisely targeted.

## 4. Model Training and Evaluation
- **Data Split Strategy:** 70% Training, 15% Validation, 15% Test Split using `random_state=42`.
- **Hyperparameter Tuning:** Usage of **Optuna** (Bayesian Optimization) over RandomizedSearchCV for more efficient and intelligent search space exploration.
- **Early Stopping:** Utilizing the validation set to stop training when performance no longer improves, preventing overfitting.
- **Evaluation Metrics (Regression):**
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R²** (R-Squared)
- **Results Compilation:** Displaying the final `metrics_table.csv` inside the report.
- **Visualizations:**
  - *Predicted vs Actual Scatter Plot:* Showing model fit.
  - *Residual Histogram:* Showing the distribution of errors.

## 5. Explainability (SHAP & Feature Importance)
- **Global Explainability:**
  - *XGBoost Native Feature Importance Bar Chart:* What the model internally prioritized.
  - *SHAP Summary Plot:* Analyzing the global impact of features (magnitude and direction) using `TreeExplainer`.
  - *SHAP Dependence Plot:* Interaction and impact of the top feature on the predicted price.
- **Local Explainability:**
  - Brief explanation of the Streamlit Waterfall plot used to explain individual user predictions.

## 6. Streamlit Front-End Interface
- Brief description of the user interface.
- How users interact with the app.
- How the SHAP local explanations are embedded into the UI to provide transparency to end users.

## 7. Ethical Scraping Note
- Details on the `scrape.py` script.
- Explanation of request delays and respecting `.lk` platform limits.

## 8. Conclusion
- Summary of the project achievements.
- Potential future improvements (e.g., more complex NLP on ad descriptions, deep learning if allowed in the future).
