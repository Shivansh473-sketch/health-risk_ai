"""
HealthRisk AI - Disease Risk Prediction & Health Awareness System
====================================================================
Main Streamlit application.

Run with:  streamlit run app.py

This app is an EDUCATIONAL project. It does not diagnose disease and is
not a substitute for professional medical advice.
"""

import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import (  # noqa: E402
    load_diabetes_data, clean_diabetes_data, DIABETES_FEATURES,
    load_heart_data, HEART_FEATURES,
)
from prediction import predict_diabetes, predict_heart, load_model  # noqa: E402
from recommendations import diabetes_recommendations, heart_recommendations, DISCLAIMER  # noqa: E402
from explainability import get_feature_importance, EXPLAINABILITY_NOTE  # noqa: E402
from database import init_db, save_prediction, get_all_predictions, clear_history  # noqa: E402

# ---------------------------------------------------------------------------
# Page config & shared setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthRisk AI",
    page_icon="favicon: heart with pulse",
    layout="wide",
)

MODELS_DIR = "models"
init_db()

DISEASE_DISCLAIMER = (
    "⚠️ **Disclaimer:** HealthRisk AI is an educational student project. "
    "It does **not** diagnose disease, does **not** replace a doctor, and "
    "should never be used for real medical decisions. Always consult a "
    "qualified healthcare professional."
)


@st.cache_data
def get_diabetes_df():
    return clean_diabetes_data(load_diabetes_data())


@st.cache_data
def get_heart_df():
    return load_heart_data()


@st.cache_data
def get_metrics():
    with open(f"{MODELS_DIR}/model_metrics.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 HealthRisk AI")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Diabetes Risk", "Heart Disease Risk", "Data Analysis",
     "Model Performance", "Prediction History"],
)
st.sidebar.markdown("---")
st.sidebar.caption(DISEASE_DISCLAIMER)

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("🩺 HealthRisk AI")
    st.subheader("Disease Risk Prediction & Health Awareness System")

    st.warning(DISEASE_DISCLAIMER)

    st.markdown("""
### About this project
HealthRisk AI is a 45-day student Machine Learning project that estimates
**educational risk categories** for two common health conditions using
publicly available datasets, and offers general, non-diagnostic health
awareness recommendations based on the values a user enters.

### Available risk modules
- **Diabetes Risk** — based on the Pima Indians Diabetes Dataset
- **Heart Disease Risk** — based on the UCI Heart Disease Dataset

### What this project demonstrates
Python · NumPy · Pandas · Data Cleaning · Statistics · EDA ·
Data Visualization · Feature Engineering · Machine Learning ·
Classification · Model Evaluation · Explainable ML ·
AI-based recommendation logic · Streamlit · SQLite

### How to use
Use the sidebar to open a risk module, enter the requested values, and
view the model's educational risk estimate, the factors it considered,
and general health-awareness suggestions.
""")

# ---------------------------------------------------------------------------
# DIABETES RISK
# ---------------------------------------------------------------------------
elif page == "Diabetes Risk":
    st.title("Diabetes Risk Estimate")
    st.warning(DISEASE_DISCLAIMER)

    st.write("Enter the values below. These correspond to the features used "
             "in the Pima Indians Diabetes Dataset.")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01)
        age = st.number_input("Age", 1, 120, 33)

    if st.button("Estimate Diabetes Risk", type="primary"):
        input_dict = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }
        result = predict_diabetes(input_dict)

        st.markdown("### Result")
        if result["prediction"] == 1:
            st.error(result["message"])
        else:
            st.success(result["message"])

        st.markdown("#### Input Factors")
        st.dataframe(pd.DataFrame([input_dict]), use_container_width=True)

        st.markdown("#### Important Factors")
        bundle = load_model("diabetes")
        importance = get_feature_importance(bundle["pipeline"], DIABETES_FEATURES)
        if importance:
            imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"])
            st.bar_chart(imp_df.set_index("Feature"))
            st.caption(EXPLAINABILITY_NOTE)
        else:
            st.info(f"Feature importance is not available for the selected "
                    f"model type ({bundle['model_name']}).")

        st.markdown("#### Health Awareness Recommendations")
        for tip in diabetes_recommendations(input_dict):
            st.info(tip)
        st.caption(DISCLAIMER)

        save_prediction(
            "diabetes", input_dict,
            "Higher risk" if result["prediction"] == 1 else "Lower risk",
            result["probability"],
        )
        st.toast("Prediction saved to history.")

# ---------------------------------------------------------------------------
# HEART DISEASE RISK
# ---------------------------------------------------------------------------
elif page == "Heart Disease Risk":
    st.title("Heart Disease Risk Estimate")
    st.warning(DISEASE_DISCLAIMER)

    st.write("Enter the values below. These correspond to the features used "
              "in the UCI Heart Disease Dataset.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_h = st.number_input("Age", 1, 120, 50, key="h_age")
        sex_h = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)],
                              format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest Pain Type", options=[
                ("Typical angina", 0), ("Atypical angina", 1),
                ("Non-anginal pain", 2), ("Asymptomatic", 3)],
            format_func=lambda x: x[0])[1]
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 220)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                            options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resting ECG", options=[
                ("Normal", 0), ("ST-T wave abnormality", 1),
                ("Left ventricular hypertrophy", 2)],
            format_func=lambda x: x[0])[1]
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina",
                              options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col3:
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment", options=[
                ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
            format_func=lambda x: x[0])[1]
        ca = st.selectbox("Number of Major Vessels Colored (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia", options=[
                ("Value 0", 0), ("Normal", 1),
                ("Fixed Defect", 2), ("Reversible Defect", 3)],
            format_func=lambda x: x[0])[1]

    if st.button("Estimate Heart Disease Risk", type="primary"):
        input_dict = {
            "age": age_h, "sex": sex_h, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
            "thal": thal,
        }
        result = predict_heart(input_dict)

        st.markdown("### Result")
        if result["prediction"] == 1:
            st.error(result["message"])
        else:
            st.success(result["message"])

        st.markdown("#### Input Factors")
        st.dataframe(pd.DataFrame([input_dict]), use_container_width=True)

        st.markdown("#### Important Factors")
        bundle = load_model("heart")
        importance = get_feature_importance(bundle["pipeline"], HEART_FEATURES)
        if importance:
            imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"])
            st.bar_chart(imp_df.set_index("Feature"))
            st.caption(EXPLAINABILITY_NOTE)
        else:
            st.info(f"Feature importance is not available for the selected "
                    f"model type ({bundle['model_name']}).")

        st.markdown("#### Health Awareness Recommendations")
        for tip in heart_recommendations(input_dict):
            st.info(tip)
        st.caption(DISCLAIMER)

        save_prediction(
            "heart", input_dict,
            "Higher risk" if result["prediction"] == 1 else "Lower risk",
            result["probability"],
        )
        st.toast("Prediction saved to history.")

# ---------------------------------------------------------------------------
# DATA ANALYSIS
# ---------------------------------------------------------------------------
elif page == "Data Analysis":
    st.title("Data Analysis & Exploratory Data Analysis")
    dataset_choice = st.selectbox("Choose dataset", ["Diabetes", "Heart Disease"])

    if dataset_choice == "Diabetes":
        df = get_diabetes_df()
        raw_df = load_diabetes_data()
    else:
        df = get_heart_df()
        raw_df = df

    st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("#### Missing Values (raw data, before cleaning)")
    st.dataframe(raw_df.isna().sum().rename("Missing Count"))

    target_col = "Outcome" if dataset_choice == "Diabetes" else "target"
    st.markdown("#### Target Class Distribution")
    fig, ax = plt.subplots(figsize=(4, 3))
    df[target_col].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown("#### Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.markdown("#### Feature Distribution")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature = st.selectbox("Choose a feature to visualize", numeric_cols)
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.histplot(df[feature], kde=True, ax=ax3)
    st.pyplot(fig3)

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    metrics = get_metrics()
    module_choice = st.selectbox("Choose module", ["diabetes", "heart"])
    module_metrics = metrics[module_choice]

    st.markdown(f"**Best model selected:** {module_metrics['best_model']}  \n"
                f"(Train size: {module_metrics['n_train']}, "
                f"Test size: {module_metrics['n_test']})")
    st.caption(
        "Selection criterion: highest F1-score (balances precision & "
        "recall), using ROC-AUC as a tie-breaker — not simply the highest "
        "accuracy. See README for rationale."
    )

    rows = []
    for model_name, m in module_metrics["results"].items():
        rows.append({
            "Model": model_name, "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1-score": m["f1_score"], "ROC-AUC": m.get("roc_auc"),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("#### Confusion Matrix (best model)")
    best_cm = module_metrics["results"][module_metrics["best_model"]]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
    st.pyplot(fig)

    roc = module_metrics["results"][module_metrics["best_model"]].get("roc_curve")
    if roc:
        st.markdown("#### ROC Curve (best model)")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.plot(roc["fpr"], roc["tpr"], label="ROC curve")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)

    st.info(
        "**Limitations:** These datasets are small, imbalanced, and were "
        "collected under specific study conditions decades ago. Real-world "
        "performance would need much larger, more diverse, and more recent "
        "data, plus clinical validation."
    )

# ---------------------------------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------------------------------
elif page == "Prediction History":
    st.title("Prediction History")
    st.caption("Stored locally in SQLite (healthrisk.db). No personally "
               "identifiable information is stored.")

    module_filter = st.selectbox("Filter by module", ["All", "diabetes", "heart"])
    records = get_all_predictions(module=None if module_filter == "All" else module_filter)

    if not records:
        st.info("No predictions saved yet. Try the Diabetes or Heart Disease Risk pages.")
    else:
        table_rows = []
        for r in records:
            table_rows.append({
                "ID": r["id"], "Module": r["module"], "Result": r["prediction_result"],
                "Probability": r["prediction_probability"], "Date/Time": r["created_at"],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        with st.expander("View raw input values for a record"):
            record_id = st.number_input("Record ID", min_value=1, step=1)
            match = next((r for r in records if r["id"] == record_id), None)
            if match:
                st.json(match["input_values"])

        if st.button("Clear all history", type="secondary"):
            clear_history()
            st.rerun()
