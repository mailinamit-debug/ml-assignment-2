import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Bank Marketing ML App")

st.title("🏦 Bank Marketing Classification App")
st.write("Upload Bank Marketing CSV file to predict term deposit subscription.")

# Load saved objects
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

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    list(models.keys())
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file, sep=";")

    if "y" in data.columns:
        data = data.drop("y", axis=1)

    # One-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    # Align columns with training data
    data = data.reindex(columns=feature_columns, fill_value=0)

    # Scale
    data_scaled = scaler.transform(data)

    model = models[model_choice]

    predictions = model.predict(data_scaled)

    st.subheader("Predictions")
    st.write(predictions)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data_scaled)[:, 1]
        st.subheader("Prediction Probabilities")
        st.write(probabilities)
