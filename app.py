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
    "S
