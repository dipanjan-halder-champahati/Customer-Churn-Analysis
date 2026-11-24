# Customer Churn Prediction
### Team: DataMinds

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-GLM%20Binomial-green.svg)](https://www.statsmodels.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

---

## Executive Summary

Customer churn is one of the most critical metrics for subscription-based telecommunication providers. Acquiring a new customer costs **5 to 7 times more** than retaining an existing one. 

This project delivers a complete, statistically rigorous, and presentation-ready **Customer Churn Prediction Workflow** developed for **Team DataMinds**. Following the instructor's analytical methodology, we implement an end-to-end pipeline using **Recursive Feature Elimination (RFE)** for feature selection, an iterative **Statsmodels Generalized Linear Model (GLM Binomial)** for logistic regression modeling, and **optimal probability threshold tuning (0.30)** to maximize the detection of at-risk customers.

> **Academic Presentation Scope:** This is a 100% Jupyter Notebook-based machine learning project. No external web application, API, or deployment interface is created. All analytical code, statistical summaries, and presentation-ready visualizations reside within the two structured notebooks.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Architecture & Folder Structure](#project-architecture--folder-structure)
- [Dataset Overview](#dataset-overview)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Performance & Evaluation](#model-performance--evaluation)
- [Key Business Insights & Recommendations](#key-business-insights--recommendations)
- [Saved Model Artifact](#saved-model-artifact)
- [How to Run the Project](#how-to-run-the-project)
  - [Environment Setup](#1-environment-setup)
  - [Option A: Running Locally in Jupyter / VS Code](#option-a-running-locally-jupyter--vs-code)
  - [Option B: Running on Google Colab (Smart Auto-Detection)](#option-b-running-on-google-colab)
- [Utility Scripts: validate_pipeline.py](#utility-scripts-validate_pipelinepy)
- [Contributors & Academic Reference](#contributors--academic-reference)

---

## Problem Statement

A major telecom operator is experiencing revenue loss due to customer turnover. The core business objectives are:
1. **Identify the primary drivers** that lead customers to terminate their contracts.
2. **Build an interpretable predictive model** capable of flagging churning customers with high sensitivity before departure.
3. **Establish an optimal operational decision threshold** balancing customer retention outreach against marketing costs.
4. **Formulate actionable, data-backed retention strategies** for customer success and marketing teams.

---

## Project Architecture & Folder Structure

```
Customer-Churn-Analysis/
│
├── data/
│   ├── raw/
│   │   ├── churn_data.csv                   # Billing, contract, churn status
│   │   ├── customer_data.csv                # Demographics: gender, dependents, seniors
│   │   └── internet_data.csv                # Internet service & add-on subscriptions
│   │
│   └── preprocessed/
│       └── telecom_cleaned.csv              # Cleaned & merged dataset (from Notebook 1)
│
├── models/
│   └── churn_model.joblib                   # Serialized model artifact & preprocessing state
│
├── notebooks/
│   ├── 01_EDA_ML.ipynb                      # Data merging, profiling, comprehensive EDA
│   └── 02_Feature_Engineering_Modeling.ipynb# Encoding, scaling, RFE, GLM, tuning, evaluation
│
├── reports/
│   ├── important_graphs/                    # 9 High-resolution EDA graphs (300 DPI)
│   └── model_evaluation/                   # 5 Evaluation charts + model_metrics.csv
│
├── DataMindsNotebook.ipynb                  # Original instructor reference notebook
├── validate_pipeline.py                     # Headless verification & automated test script
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
└── .gitignore                               # Git ignore configuration
```

---

## Dataset Overview

The project analyzes telecom customer records distributed across three source files linked by `customerID`:

| Source File | Records | Features | Description |
|---|---|---|---|
| `churn_data.csv` | 7,043 | 9 | Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, **Churn (Target)** |
| `customer_data.csv` | 7,043 | 5 | Gender, SeniorCitizen, Partner, Dependents |
| `internet_data.csv` | 7,043 | 9 | InternetService, OnlineSecurity, TechSupport, StreamingTV, StreamingMovies, etc. |

### Data Cleaning Highlights:
- **Inner Join:** Datasets merged on `customerID` resulting in 7,043 rows × 21 columns.
- **Handling Whitespace Values:** `TotalCharges` contained 11 whitespace entries corresponding to 0-month tenure accounts. These were coerced to `NaN` and safely dropped, yielding **7,032 clean rows**.
- **Target Distribution:** Baseline churn rate is **26.58%** (imbalanced classification).

---

## Machine Learning Workflow

```
Raw CSVs (3) ──► Inner Merge on customerID ──► TotalCharges Coercion (7,032 rows)
                                                      │
                                                      ▼
                                       Binary Encoding (Yes/No ➔ 1/0)
                                                      │
                                                      ▼
                                      70:30 Train-Test Split (seed=100)
                                                      │
                                                      ▼
                                     One-Hot Encoding + Dropped Redundancies
                                                      │
                                                      ▼
                                      StandardScaler (fit on train only)
                                                      │
                                                      ▼
                                      RFE: Top 15 Predictors Selected
                                                      │
                                                      ▼
                       Iterative Statsmodels GLM Binomial (p-value & VIF Pruning)
                       ├── Model 1 (15 features): MonthlyCharges dropped (VIF = 26.77)
                       ├── Model 2 (14 features): TotalCharges dropped   (VIF = 8.77)
                       └── Model 3 (13 features): All p < 0.05, All VIF < 2.59 (FINAL)
                                                      │
                                                      ▼
                             Optimal Probability Cutoff Selection: Threshold = 0.30
                                                      │
                                                      ▼
                              Evaluation on 30% Unseen Test Set (2,110 customers)
                                                      │
                                                      ▼
                             Model Artifact Serialization (churn_model.joblib)
```

### The 13 Final Selected Features:
1. `tenure` (Continuous, Standardized)
2. `PaperlessBilling`
3. `SeniorCitizen`
4. `Contract_One year`
5. `Contract_Two year`
6. `PaymentMethod_Credit card (automatic)`
7. `PaymentMethod_Mailed check`
8. `MultipleLines_Yes`
9. `InternetService_Fiber optic`
10. `InternetService_No`
11. `TechSupport_Yes`
12. `StreamingTV_Yes`
13. `StreamingMovies_Yes`

---

## Model Performance & Evaluation

Because detecting churning customers (True Positives) is paramount for retention, evaluating solely on default 0.50 threshold is suboptimal for imbalanced data. By analyzing the intersection of **Accuracy, Sensitivity (Recall), and Specificity**, the optimal decision threshold was identified as **0.30**.

### Verified Metrics Summary:

| Metric | Training Set (Threshold = 0.30) | Test Set (Threshold = 0.30) | Evaluation Meaning |
|---|:---:|:---:|---|
| **Accuracy** | **77.00%** | **74.08%** | Overall classification correctness |
| **Sensitivity (Recall)** | **77.62%** | **71.99%** | **Catches ~72% of all churning customers** |
| **Specificity** | **76.78%** | **74.87%** | Correctly identifies ~75% of non-churning customers |
| **Precision (PPV)** | — | **52.18%** | Over half of flagged customers are true churners |
| **Negative Predictive Value**| — | **87.53%** | Very high reliability when predicting customer retention |
| **AUC Score** | **0.8478** | **0.8277** | Outstanding discriminative ability (minimal train-test gap) |

### Test Set Confusion Matrix (N = 2,110):

```
                     Predicted: No Churn    Predicted: Churn
Actual: No Churn             1,144                  384
Actual: Churn                 163                   419
```

- **True Positives (419):** Churners correctly identified for retention outreach.
- **False Negatives (163):** Churners missed (minimized by using threshold 0.30 instead of 0.50).
- **False Positives (384):** Loyal customers offered retention perks (acceptable marketing cost).

---

## Key Business Insights & Recommendations

1. **Two-Year Contracts are the Strongest Retention Anchor:**
   - Two-year contracts reduce churn likelihood by **~45.8%** relative to month-to-month contracts.
   - *Action:* Offer sign-up discounts or device upgrades to migrate month-to-month users to annual commitments.

2. **Fiber Optic Quality & Value Perception:**
   - Fiber Optic Internet is the **single largest positive predictor of churn (+28.4% impact)** despite higher prices.
   - *Action:* Investigate network latency, outages, and customer support responsiveness specifically for fiber subscribers.

3. **Tenure Builds Loyalty:**
   - Churn probability drops significantly as tenure increases. First-year customers exhibit the highest vulnerability (~47% churn rate).
   - *Action:* Implement a structured 90-day onboarding check-in sequence to stabilize early customer tenure.

4. **Electronic Check Payment Method Friction:**
   - Customers paying via electronic check churn at **~45%** (double that of automated bank/card payments).
   - *Action:* Incentivize automated autopay (credit card / bank transfer) with a \$2–\$5 monthly billing credit.

5. **Senior Citizen Customer Care:**
   - Senior citizens have a **41.7% churn rate** compared to 23.6% for non-seniors.
   - *Action:* Provide dedicated, non-automated customer helpline routing and simplified billing formats.

---

## Saved Model Artifact

The final trained pipeline is saved at `models/churn_model.joblib`. It contains the complete inference bundle:

```python
import joblib
import statsmodels.api as sm

# 1. Load the artifact
artifact = joblib.load('models/churn_model.joblib')

model           = artifact['model']           # Fitted Statsmodels GLM Binomial model
scaler          = artifact['scaler']          # Fitted StandardScaler
final_features  = artifact['final_features']  # Exact 13 features used in training
threshold       = artifact['threshold']       # Operational threshold (0.30)

# 2. Predict on new incoming customer data (preprocessed)
X_new_scaled    = scaler.transform(X_new[['tenure', 'MonthlyCharges', 'TotalCharges']])
X_new_prepared  = sm.add_constant(X_new[final_features], has_constant='add')
churn_prob      = model.predict(X_new_prepared)
churn_decision  = (churn_prob > threshold).astype(int)
```

---

## How to Run the Project

### 1. Environment Setup

Clone this repository and install the verified dependencies:

```bash
git clone https://github.com/<your-username>/customer-churn-analysis.git
cd customer-churn-analysis

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

### Option A: Running Locally (Jupyter / VS Code)

1. Launch Jupyter Notebook from the project root:
   ```bash
   jupyter notebook
   ```
2. Run the notebooks in sequential order:
   - **Step 1:** Open and run `notebooks/01_EDA_ML.ipynb`
     - Merges raw data, performs visual analysis, exports EDA charts to `reports/important_graphs/`, and saves `data/preprocessed/telecom_cleaned.csv`.
   - **Step 2:** Open and run `notebooks/02_Feature_Engineering_Modeling.ipynb`
     - Executes feature scaling, RFE, GLM modeling, test evaluation, saves evaluation charts to `reports/model_evaluation/`, and outputs `models/churn_model.joblib`.

---

### Option B: Running on Google Colab

Both notebooks are equipped with **Smart Path Auto-Detection**. They automatically detect whether they are running inside a local directory hierarchy or in Google Colab's flat runtime environment.

#### Simple Colab Upload (No Code Modifications Required):
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `01_EDA_ML.ipynb`.
3. In Colab's left sidebar, click the **Folder icon (Files)** and upload the 3 raw dataset files:
   - `churn_data.csv`
   - `customer_data.csv`
   - `internet_data.csv`
4. Click **Runtime ➔ Run All**. The notebook will automatically detect the flat directory structure, process the data, and generate all outputs inline!
5. Next, upload `02_Feature_Engineering_Modeling.ipynb` into Colab and click **Runtime ➔ Run All** to generate the model evaluation, graphs, and saved model artifact.

*(Optional)* If you prefer the exact folder structure in Colab, you can create it with this 1-line command in a top cell:
```python
!mkdir -p ../data/raw ../data/preprocessed ../reports/important_graphs ../reports/model_evaluation ../models
!cp -f *.csv ../data/raw/ 2>/dev/null || true
```

---

## Utility Scripts: `validate_pipeline.py`

`validate_pipeline.py` is a standalone, headless Python validation script included for testing and continuous integration:
- **What it does:** Executes the complete pipeline from terminal (data loading, preprocessing, model fitting, metric calculation, and chart saving) without opening a Jupyter interface.
- **When to use it:** Use it if you want to quickly test the entire analytical pipeline from the command line:
  ```bash
  python validate_pipeline.py
  ```
- **Note:** It is strictly a developer validation tool. Your primary deliverables and presentation outputs remain the two `.ipynb` notebooks.

---

## Contributors & Academic Reference

- **Team:** DataMinds
- **Project Domain:** Customer Churn Analytics & Predictive Modeling
- **Methodological Reference:** `DataMindsNotebook.ipynb` (Instructor Reference)
- **Primary Algorithm:** Statsmodels Generalized Linear Model (Binomial Logistic Regression) with Scikit-Learn RFE Feature Selection
