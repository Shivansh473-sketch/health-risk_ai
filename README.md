# HealthRisk AI
### Disease Risk Prediction & Health Awareness System

A 45-day student-level AI/ML project built with **Python, scikit-learn, and
Streamlit**. It estimates **educational risk categories** for two common
health conditions (diabetes and heart disease) from user-provided health and
lifestyle values, and offers general, non-diagnostic health-awareness
recommendations.

> ⚠️ **This project does not diagnose disease and is not a substitute for
> professional medical advice.** All outputs are educational estimates based
> on public datasets and simple ML models.

---

## 1. Project Objective

Analyze user-provided health/lifestyle parameters and estimate a **risk
category** (Lower / Moderate / Higher) for:
- Diabetes
- Heart disease

The system never claims to diagnose a disease. Output is always phrased as
an educational estimate (e.g. *"Model-estimated risk: Higher"*), never as a
diagnosis (e.g. never *"You have diabetes"*).

## 2. Technology Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit ·
SQLite · Joblib

## 3. Datasets

| Module | Dataset | Rows | Source |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes Database | 768 | National Institute of Diabetes and Digestive and Kidney Diseases (via UCI/Kaggle mirrors) |
| Heart Disease | UCI Heart Disease Dataset (Cleveland, processed) | 298 | UCI Machine Learning Repository, collected by Robert Detrano, M.D., Ph.D. |

Both files are included under `data/`. See `notebooks/` for full feature
explanations, missing-value analysis, and EDA for each dataset.

**Diabetes feature notes:** Several columns (`Glucose`, `BloodPressure`,
`SkinThickness`, `Insulin`, `BMI`) use `0` in the original data as a
placeholder for missing readings (nobody has a blood pressure of 0). These
are treated as missing values and median-imputed — see
`src/preprocessing.py`.

**Heart disease feature notes:** The original UCI encoding used `1`/`2` for
binary fields; this project re-codes them to the more conventional `0`/`1`
(e.g. `sex`: 0=female, 1=male) and re-bases categorical codes to start at 0.
The original multi-class `num` target (0-4, severity) is binarized to
`target` (0 = no disease, 1 = disease present), which is the standard
approach for this dataset in published research.

## 4. Project Structure

```
HealthRisk_AI/
│
├── app.py                     # Streamlit application (entry point)
├── requirements.txt
├── README.md
├── healthrisk.db               # created at runtime (prediction history)
│
├── data/
│   ├── diabetes.csv
│   └── heart.csv
│
├── models/
│   ├── diabetes_model.pkl      # created by train_models.py
│   ├── heart_model.pkl         # created by train_models.py
│   └── model_metrics.json      # created by train_models.py
│
├── notebooks/
│   ├── diabetes_analysis.ipynb
│   └── heart_analysis.ipynb
│
├── src/
│   ├── preprocessing.py        # loading, cleaning, pipelines, train/test split
│   ├── train_models.py         # trains & compares 5 algorithms, saves best
│   ├── prediction.py           # loads saved pipeline, formats risk output
│   ├── recommendations.py      # rule-based health-awareness messages
│   ├── explainability.py       # feature importance / coefficients
│   └── database.py             # SQLite prediction history
│
└── assets/
```

## 5. Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (creates models/*.pkl and model_metrics.json)
cd src
python train_models.py
cd ..

# 3. Run the app
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## 6. Machine Learning Approach

Five classical algorithms are trained and compared **per module**:
Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors,
and Support Vector Machine.

**Model selection is not based on accuracy alone.** For a health-risk
screening tool, missing an at-risk person (a false negative) is generally
more costly than a false alarm, so the best model is selected using the
**F1-score** (which balances precision and recall), with **ROC-AUC** as a
tie-breaker. All five models' metrics (Accuracy, Precision, Recall,
F1-score, Confusion Matrix, ROC-AUC) are visible on the app's
**Model Performance** page for transparency.

**Preprocessing** uses scikit-learn `Pipeline`s (imputation + scaling)
fit only on the training split, avoiding data leakage into the test set.

**Explainability:** For tree-based/linear models, feature importance or
coefficients are shown for each prediction, with an explicit note that this
reflects a statistical pattern learned by the model — not a proven medical
cause of disease.

## 7. Limitations

- Both datasets are small (298-768 rows), moderately old, and collected
  under specific study conditions (e.g. the diabetes dataset only includes
  female patients of Pima Indian heritage aged 21+).
- No hyperparameter tuning beyond basic defaults is performed, in keeping
  with the educational scope of a 45-day project.
- Models are not clinically validated and must never be used for real
  health decisions.

## 8. Prediction History & Privacy

Predictions are stored locally in a SQLite database (`healthrisk.db`)
with only the input values and result — no names or identifying
information are collected or stored.

---

## 45-Day Development Plan

| Days | Focus |
|---|---|
| 1–5 | Python revision, project introduction, healthcare ML concepts, dataset understanding |
| 6–10 | NumPy, Pandas, dataset loading, data cleaning |
| 11–15 | Statistics, EDA, data visualization, correlation analysis |
| 16–20 | Feature engineering, encoding, scaling, train-test split, preprocessing pipeline |
| 21–27 | Classification algorithms, training multiple models, evaluation, confusion matrix, ROC-AUC |
| 28–32 | Model comparison, hyperparameter tuning, final model selection, model saving |
| 33–36 | Prediction module, health-awareness recommendation engine, explainability module |
| 37–40 | Streamlit UI, dashboard, prediction forms, charts |
| 41–43 | SQLite, prediction history, testing, bug fixing |
| 44–45 | Final documentation, PPT, project report, viva preparation, final demonstration |

## Student Learning Outcomes

Python · NumPy · Pandas · Data Cleaning · Statistics · EDA ·
Data Visualization · Feature Engineering · Machine Learning ·
Classification · Model Evaluation · Explainable ML ·
AI-based recommendation logic · Streamlit · SQLite ·
Model deployment · Project documentation
