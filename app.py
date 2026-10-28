# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

st.set_page_config(page_title="Bike Demand Classifier", layout="wide")
st.title("Bike Sharing: High-Demand Hour Classifier")
st.caption("Predicts whether an hour has above-median bike rentals. Upload the test CSV to evaluate.")

TARGET_COL = "high_demand"
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))

if uploaded is None:
    st.info("Upload the test_data.csv file to evaluate the models.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET_COL not in df.columns:
    st.error(f"The CSV must contain the target column '{TARGET_COL}'.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Load the saved pipeline. No retraining happens here.
model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

auc = roc_auc_score(y, y_proba[:, 1]) if y_proba is not None else None

metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "AUC": auc,
    "Precision": precision_score(y, y_pred, zero_division=0),
    "Recall": recall_score(y, y_pred, zero_division=0),
    "F1": f1_score(y, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y, y_pred),
}

st.subheader(f"Metrics: {model_name}")
cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, "N/A" if value is None else f"{value:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low", "High"], yticklabels=["Low", "High"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(y, y_pred, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose())
