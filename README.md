# ML Assignment 2 – Classification Models & Streamlit Deployment

## Problem Statement
Implement multiple machine learning classification models, evaluate their performance,
and deploy the models using a Streamlit web application.

## Dataset Description
The dataset used is the UCI Bank Marketing Dataset from the UCI Machine Learning Repository.
It contains 45,211 instances and 16 original predictive features related to client demographic
and marketing campaign information.

After one-hot encoding of categorical variables, the total number of features increased to 42.

The target variable is binary:
1 → Client subscribed to term deposit
0 → Client did not subscribe

## Models Used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)

## Observations
Ensemble models such as Random Forest and XGBoost achieved the highest Accuracy and AUC,
indicating better generalization performance. Logistic Regression and Naive Bayes also
performed strongly, while Decision Tree showed comparatively lower generalization.
Overall, ensemble methods provided better robustness compared to individual classifiers.

## Deployment
The application is deployed using Streamlit Community Cloud.
