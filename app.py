import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Bank Marketing ML App")

st.title("🏦 Bank Marketing Classification App")
st.write("Upload Bank Marketing CSV file to predict term deposit subscription.")

# ==============================
# Load saved objects
# ==============================

scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

# ==============================
# File Upload & Model Selection
# ==============================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    list(models.keys())
)

# ==============================
# Prediction Logic
# ==============================

if uploaded_file is not None:

    try:
        # Read CSV (Bank dataset uses ;)
        data = pd.read_csv(uploaded_file, sep=";")

        # Drop target if present
        if "y" in data.columns:
            data = data.drop("y", axis=1)

        # One-hot encode
        data = pd.get_dummies(data, drop_first=True)

        # ------------------------------
        # SAFE COLUMN ALIGNMENT
        # ------------------------------

        # Add missing columns
        for col in feature_columns:
            if col not in data.columns:
                data[col] = 0

        # Keep only training columns in correct order
        data = data[feature_columns]

        # Ensure numeric
        data = data.astype(float)

        # Scale
        data_scaled = scaler.transform(data)

        # Predict
        model = models[model_choice]
        predictions = model.predict(data_scaled)

        # Convert 0/1 to labels
        label_map = {0: "No Subscription", 1: "Subscribed"}
        predictions_label = [label_map[p] for p in predictions]

        st.subheader("📊 Predictions")
        st.write(predictions_label)

        # Show probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(data_scaled)[:, 1]
            st.subheader("📈 Subscription Probability")
            st.write(probabilities)

    except Exception as e:
        st.error("Error processing file. Please ensure you uploaded correct Bank Marketing dataset format.")
        st.write(e)
