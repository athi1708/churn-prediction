"""
SHAP Explainability Engine
Returns feature impact values for a single prediction.

Output format:
[
    ("feature_name", shap_value),
    ...
]
"""

import shap
import joblib
import pandas as pd
import os

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model/churn_model.pkl"))

# Create SHAP explainer
explainer = shap.Explainer(model)


def explain_prediction(input_df, top_n=10):
    try:
        shap_values = explainer(input_df)

        values = shap_values.values[0]
        features = input_df.columns

        results = list(zip(features, values))

        # sort by importance
        results = sorted(results, key=lambda x: abs(x[1]), reverse=True)

        return results[:top_n]

    except Exception as e:
        print("SHAP Error:", e)
        return []