# 📡 Telco Customer Churn — Business Intelligence & Prediction Hub

A complete, single-file Business Intelligence project for analysing, predicting, and acting on customer churn in a telecommunications dataset.

---

## 📌 Project Overview

This project delivers a **full BI pipeline** from raw CSV to actionable retention recommendations:

| Layer | What it does |
|-------|-------------|
| **Data Processing** | Cleans blank `TotalCharges`, converts types, engineers tenure buckets & charge tiers |
| **EDA** | Surfaces churn patterns across contract type, internet service, payment method, tenure, and add-on services |
| **Prediction Model** | Trains Logistic Regression *and* Random Forest; selects the best by AUC-ROC; scores every customer |
| **BI Dashboard** | 4-page interactive Streamlit app with KPI cards, charts, risk tables, and retention action cards |

---

## 📂 Dataset

**Source:** Kaggle — Telco Customer Churn by *blastchar*  
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**File name expected by the app:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
Place this file in the **same directory** as `app.py` before running.

**Columns (21):** `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

---

## 🚀 Setup & Run Instructions

### 1 — Prerequisites

- Python **3.9 – 3.12** (tested on 3.11)
- `pip` package manager

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add the dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle link above and copy it into the project folder (next to `app.py`).

```
project/
├── app.py
├── requirements.txt
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← place here
```

### 4 — Launch the dashboard

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## 📊 Dashboard Pages

| Page | Content |
|------|---------|
| **🏠 Executive Overview** | Churn rate KPI, total customers, monthly revenue at risk, average tenure; churn distribution donut, charge distribution histogram, contract-type bar chart |
| **📊 Trend & Driver Analysis** | Churn by internet service, tenure bucket (line), payment method, charge tier, contract × service heatmap, add-on service comparison |
| **🚨 Risk & Action Center** | High/Medium/Low risk KPIs; top-20 high-value customers predicted to churn; 6 prioritised retention action cards |
| **🤖 Model Performance** | Side-by-side Accuracy / Precision / Recall / F1 / AUC-ROC for both models; confusion matrices; ROC curves; Random Forest feature importance |

### Sidebar Filters (apply globally)
- Contract type (multi-select)
- Internet service (multi-select)
- Tenure range (slider)

---

## 🧠 Modelling Notes

- **Train/Test split:** 80 / 20, stratified on `Churn`
- **Class imbalance:** handled via `class_weight="balanced"` on both models
- **Winner selection:** highest AUC-ROC on the held-out test set (typically Random Forest)
- **Scoring:** winner model scores *all* 7,043 customers for the Risk page

---

## 📦 Dependencies

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
```

---

## 📄 Files

| File | Description |
|------|-------------|
| `app.py` | Single-file Python app (data processing + models + Streamlit UI) |
| `requirements.txt` | Pinned library versions |
| `README.md` | This file |
| `Churn_BI_Report.docx` | Full project report (Word document) |
