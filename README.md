# 📊 Telecom Customer Churn Prediction

> An internship project implementing a **Logistic Regression** churn prediction system for a telecom company, with full EDA, feature engineering, model evaluation, persistence, and business recommendations.

---

## 🗂️ Project Structure

```
customer-churn-analysis/
│
├── 📁 data/
│   ├── raw/                          # Original source CSVs (never modified)
│   │   ├── churn_data.csv            # Tenure, contract, billing, target
│   │   ├── customer_data.csv         # Demographics (gender, partner, senior citizen)
│   │   ├── internet_data.csv         # Internet service, streaming, security add-ons
│   │   └── Telecom_Churn_Data_Dictionary.csv
│   └── processed/
│       └── telecom_cleaned.csv       # Merged + cleaned dataset (output of Notebook 1)
│
├── 📁 notebooks/
│   ├── 01_EDA_ML.ipynb               # Problem statement → Data loading → EDA
│   └── 02_Feature_Engineering_Modeling.ipynb  # FE → Split → Model → Eval → Insights
│
├── 📁 models/
│   ├── churn_model.joblib            # Full model package (preferred)
│   └── churn_model.pkl               # Pickle backup
│
├── 📁 reports/                       # Presentation slides, screenshots (optional)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📋 Notebook Breakdown

| Notebook | Covers | Key Outputs |
|----------|--------|-------------|
| **01_EDA_ML.ipynb** | Problem Statement, Data Loading, 3-way Merge, Quality Inspection, Full EDA | `telecom_cleaned.csv` |
| **02_Feature_Engineering_Modeling.ipynb** | Feature Engineering, Train/Test Split, OHE, Scaling, RFE, Iterative GLM, ROC, Threshold Tuning, Test Evaluation, Model Saving, Inferencing, Business Insights | `churn_model.joblib` |

---

## 🏗️ Methodology

### Data Pipeline
```
3 CSV Files → Inner Join → 7,043 rows × 21 columns
           → TotalCharges fix (11 rows dropped)
           → 7,032 clean records
```

### Feature Engineering Pipeline
```
Binary encoding (Yes/No → 0/1)
→ Stratified 70/30 Train-Test Split
→ One-Hot Encoding (drop_first, leak-proof)
→ Redundant dummy removal (7 columns)
→ StandardScaler (fit on train only)
→ RFE (15 features selected)
→ Iterative GLM: VIF + p-value pruning
→ Final 13-feature model
```

### Model Selection
| Step | Decision | Rationale |
|------|----------|-----------|
| Algorithm | Logistic Regression (GLM) | Interpretable coefficients, instructor requirement |
| Split | 70/30 stratified | Dataset size (7K), stable class proportion |
| Threshold | ~0.30 (Sens=Spec intersection) | Business priority: maximize churn capture |
| Feature selection | RFE + VIF < 5 + p < 0.05 | Statistical rigor, no multicollinearity |

---

## 📈 Model Performance

| Metric | Train | Test |
|--------|-------|------|
| **AUC** | ~0.85 | ~0.84 |
| **Accuracy** | ~77% | ~74% |
| **Churn Recall** | ~78% | ~72% |
| **Specificity** | ~77% | ~75% |

**Conclusion**: Minimal train-test gap — no overfitting detected.

---

## 🔑 Top Business Insights

| Factor | Direction | Priority Action |
|--------|-----------|----------------|
| Two-year contract | 🟢 Protects (-1.30 coef) | Incentivize contract upgrades |
| Fiber Optic | 🔴 Drives churn (+0.81) | Audit service quality |
| Tenure | 🟢 Protects (-0.94 coef) | Early-warning < 12-month alerts |
| Electronic Check | 🔴 Correlated with churn | Autopay migration campaign |
| Senior Citizen | 🔴 Higher risk (+0.46) | Dedicated senior plans |
| Tech Support | 🟢 Protects (-0.40) | Bundle in onboarding |

---

## 🚀 How to Run the Project

You can run this project either **Locally** (VS Code / Jupyter Notebook) or on the cloud via **Google Colab**. Follow the steps below based on your preference:

### 💻 Method 1: Running Locally (VS Code / Jupyter)

#### 1. Setup & Installation
Open PowerShell or Terminal in the project root folder (`customer-churn-analysis/`):
```bash
# Optional: Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

#### 2. Running in VS Code (Recommended)
1. Open the project folder `customer-churn-analysis` in VS Code (`File > Open Folder...`).
2. Make sure the official **Python** and **Jupyter** extensions are installed in VS Code.
3. Open `notebooks/01_EDA_ML.ipynb`.
4. In the top-right corner of the notebook editor, click **Select Kernel** and choose your Python environment (Python 3.10+ / 3.12).
5. Click **Run All** at the top bar, or run cells one-by-one with `Shift + Enter`.
6. Once finished, open and run `notebooks/02_Feature_Engineering_Modeling.ipynb`.

#### 3. Running with Classic Jupyter Notebook / JupyterLab
```bash
# Start the Jupyter server
jupyter notebook
# Or: jupyter lab
```
In your browser, navigate to the `notebooks/` folder:
1. Open `01_EDA_ML.ipynb` → click **Kernel > Restart & Run All**.
2. Open `02_Feature_Engineering_Modeling.ipynb` → click **Kernel > Restart & Run All**.

---

### ☁️ Method 2: Running on Google Colab

Both Jupyter and Google Colab use the identical `.ipynb` file format. **You do not need to rewrite your code.** Furthermore, all core data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `joblib`) are **pre-installed by default** on Google Colab!

Choose whichever option is most convenient:

#### 🔹 Option A: Direct File Upload to Session Storage (Fastest & Simplest)
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File > Upload Notebook** and choose `01_EDA_ML.ipynb`.
3. In Colab's left sidebar, click the **Folder (📁 Files)** icon.
4. Drag-and-drop the 3 CSV files (`churn_data.csv`, `customer_data.csv`, `internet_data.csv` from your `data/raw/` folder) into the file pane.
5. **Only 1 minor edit needed in Cell 2 (Data Loading)**:
   Change the relative folder paths:
   ```python
   # From local paths:
   churn_data    = pd.read_csv('../data/raw/churn_data.csv')
   customer_data = pd.read_csv('../data/raw/customer_data.csv')
   internet_data = pd.read_csv('../data/raw/internet_data.csv')
   ```
   To direct filenames:
   ```python
   # To Colab session paths:
   churn_data    = pd.read_csv('churn_data.csv')
   customer_data = pd.read_csv('customer_data.csv')
   internet_data = pd.read_csv('internet_data.csv')
   ```
   *Similarly, in Notebook 2, point `telecom_cleaned.csv` to `'telecom_cleaned.csv'`.*
6. Click **Runtime > Run all** (`Ctrl + F9`).

#### 🔹 Option B: Google Drive Mounting (Zero Code Changes Needed)
If you upload your entire `customer-churn-analysis/` folder to your Google Drive:
1. Open [Google Colab](https://colab.research.google.com/) and open the notebook from your Drive.
2. Add a new code cell at the very top and run:
   ```python
   from google.colab import drive
   import os

   # Mount your Google Drive
   drive.mount('/content/drive')

   # Change working directory to your project's notebooks directory
   os.chdir('/content/drive/MyDrive/customer-churn-analysis/notebooks')
   ```
3. **No file path edits needed!** All relative paths (`../data/raw/...`, `../data/processed/...`, `../models/...`) will work seamlessly just like they do locally.
4. Click **Runtime > Run all** (`Ctrl + F9`).

---

### Load the saved model for inference
```python
import joblib
import statsmodels.api as sm

pkg = joblib.load('models/churn_model.joblib')

# pkg contains: model, scaler, rfe_cols, optimal_threshold, ...
print(f"Model AUC (test): {pkg['test_auc']}")
```

---

## 📦 Dependencies

See `requirements.txt`. Core packages:
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `statsmodels`
- `joblib`

---

## 👤 Author

**Summer Internship Project** | Customer Churn Analysis  
*Supervised by Sir Arijit Midya — based on DataMinds baseline notebook*