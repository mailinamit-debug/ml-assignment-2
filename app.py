import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="ML Classification App")

st.title("Machine Learning Classification App")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview:")
    st.write(df.head())

    # Define correct feature columns used during training
    feature_columns = [
        "Cl.thickness",
        "Cell.size",
        "Cell.shape",
        "Marg.adhesion",
        "Epith.c.size",
        "Bare.nuclei",
        "Bl.cromatin",
        "Normal.nucleoli",
        "Mitoses"
    ]

    # Extract only correct columns
    X = df[feature_columns]
    y = df["Class"]

    # Convert to numpy
    X = X.values

    # Load preprocessing objects
    imputer = joblib.load("model/imputer.pkl")
    scaler = joblib.load("model/scaler.pkl")

    # Apply preprocessing
    X = imputer.transform(X)
    X = scaler.transform(X)

    # Load selected model
    model_paths = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl"
    }

    model = joblib.load(model_paths[model_option])

    y_pred = model.predict(X)

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        st.write("AUC:", roc_auc_score(y, y_prob))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
