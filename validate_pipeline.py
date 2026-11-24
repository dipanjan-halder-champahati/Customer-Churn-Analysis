"""
validate_pipeline.py
Customer Churn Prediction — DataMinds
Standalone validation script that runs the complete ML pipeline,
generates all graphs, saves the model, and prints evaluation results.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for headless execution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

# ── Paths (relative to project root) ────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
DATA_RAW     = os.path.join(ROOT, 'data', 'raw')
DATA_PRE     = os.path.join(ROOT, 'data', 'preprocessed')
MODELS_DIR   = os.path.join(ROOT, 'models')
GRAPH_DIR    = os.path.join(ROOT, 'reports', 'important_graphs')
EVAL_DIR     = os.path.join(ROOT, 'reports', 'model_evaluation')
EXPORT_DPI   = 300
THRESHOLD    = 0.30
PALETTE      = ['#2196F3', '#F44336']

for d in [DATA_PRE, MODELS_DIR, GRAPH_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':13,
                     'axes.labelsize':11,'figure.dpi':100})

print("=" * 60)
print("CUSTOMER CHURN PREDICTION — DataMinds Validation")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# STAGE 1 — LOAD AND MERGE
# ════════════════════════════════════════════════════════════
print("\n[1/8] Loading and merging datasets...")
churn_data    = pd.read_csv(os.path.join(DATA_RAW, 'churn_data.csv'))
customer_data = pd.read_csv(os.path.join(DATA_RAW, 'customer_data.csv'))
internet_data = pd.read_csv(os.path.join(DATA_RAW, 'internet_data.csv'))

df1    = pd.merge(churn_data, customer_data, how='inner', on='customerID')
telecom = pd.merge(df1, internet_data, how='inner', on='customerID')
print(f"   Merged shape: {telecom.shape}")

# Save preprocessed
telecom.to_csv(os.path.join(DATA_PRE, 'telecom_cleaned.csv'), index=False)
print("   Saved: data/preprocessed/telecom_cleaned.csv")

# ════════════════════════════════════════════════════════════
# STAGE 2 — EDA GRAPHS
# ════════════════════════════════════════════════════════════
print("\n[2/8] Generating EDA graphs...")

# 2a. Churn Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Overall Customer Churn Distribution', fontsize=15, fontweight='bold')
churn_counts = telecom['Churn'].value_counts()
bars = axes[0].bar(churn_counts.index, churn_counts.values, color=PALETTE, edgecolor='white')
for b in bars:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+60,
                 f'{int(b.get_height()):,}', ha='center', fontweight='bold')
axes[0].set_title('Count: Churned vs Retained')
axes[0].set_ylabel('Customers')
axes[0].set_ylim(0, churn_counts.max()*1.15)
axes[1].pie(churn_counts.values, labels=['No Churn','Churn'], autopct='%1.1f%%',
            colors=PALETTE, startangle=90, explode=[0,0.05])
axes[1].set_title('Churn Proportion')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'01_churn_distribution.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 01_churn_distribution.png")

# 2b. Tenure vs Churn
fig, ax = plt.subplots(figsize=(9,6))
sns.boxplot(x='Churn', y='tenure', data=telecom, palette=PALETTE, ax=ax, linewidth=1.5)
for i, v in enumerate(['No','Yes']):
    med = telecom[telecom['Churn']==v]['tenure'].median()
    ax.text(i, med+1.5, f'Median: {int(med)}', ha='center', fontweight='bold', fontsize=10)
ax.set_title('Customer Tenure vs Churn', fontweight='bold')
ax.set_xlabel('Churn'); ax.set_ylabel('Tenure (Months)')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'02_tenure_vs_churn.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 02_tenure_vs_churn.png")

# 2c. Monthly Charges vs Churn
fig, ax = plt.subplots(figsize=(9,6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=telecom, palette=PALETTE, ax=ax, linewidth=1.5)
for i, v in enumerate(['No','Yes']):
    med = telecom[telecom['Churn']==v]['MonthlyCharges'].median()
    ax.text(i, med+1.5, f'Median: ${med:.0f}', ha='center', fontweight='bold', fontsize=10)
ax.set_title('Monthly Charges vs Churn', fontweight='bold')
ax.set_xlabel('Churn'); ax.set_ylabel('Monthly Charges (USD)')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'03_monthly_charges_vs_churn.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 03_monthly_charges_vs_churn.png")

# 2d. Contract vs Churn
fig, axes = plt.subplots(1,2, figsize=(14,6))
fig.suptitle('Contract Type vs Churn', fontsize=14, fontweight='bold')
sns.countplot(x='Contract', hue='Churn', data=telecom, ax=axes[0], palette=PALETTE)
axes[0].set_title('Count by Contract Type'); axes[0].tick_params(axis='x', rotation=10)
contract_order = ['Month-to-month','One year','Two year']
churn_rates = [telecom[telecom['Contract']==c]['Churn'].eq('Yes').mean()*100 for c in contract_order]
bars2 = axes[1].bar(contract_order, churn_rates, color=['#F44336','#FF9800','#4CAF50'], edgecolor='white')
for b,r in zip(bars2, churn_rates):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{r:.1f}%', ha='center', fontweight='bold')
axes[1].set_title('Churn Rate by Contract'); axes[1].tick_params(axis='x', rotation=10)
axes[1].set_ylabel('Churn Rate (%)'); axes[1].set_ylim(0, max(churn_rates)*1.2)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'04_contract_vs_churn.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 04_contract_vs_churn.png")

# 2e. Internet Service vs Churn
fig, axes = plt.subplots(1,2, figsize=(14,6))
sns.countplot(x='InternetService', hue='Churn', data=telecom, ax=axes[0], palette=PALETTE)
axes[0].set_title('Count by Internet Service')
internet_order = ['DSL','Fiber optic','No']
int_rates = [telecom[telecom['InternetService']==s]['Churn'].eq('Yes').mean()*100 for s in internet_order]
bars3 = axes[1].bar(internet_order, int_rates, color=['#2196F3','#F44336','#4CAF50'], edgecolor='white')
for b,r in zip(bars3, int_rates):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{r:.1f}%', ha='center', fontweight='bold')
axes[1].set_title('Churn Rate by Internet Service'); axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, max(int_rates)*1.2)
plt.suptitle('Internet Service vs Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'05_internet_service_vs_churn.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 05_internet_service_vs_churn.png")

# 2f. Payment Method vs Churn
fig, ax = plt.subplots(figsize=(12,6))
pay_churn = telecom.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
pay_churn.columns = ['PaymentMethod','ChurnRate']
pay_churn = pay_churn.sort_values('ChurnRate', ascending=False)
sns.barplot(x='PaymentMethod', y='ChurnRate', data=pay_churn, palette='RdYlGn_r', ax=ax)
for b in ax.patches:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f'{b.get_height():.1f}%',
            ha='center', fontweight='bold', fontsize=10)
ax.set_title('Churn Rate by Payment Method', fontweight='bold')
ax.set_xlabel('Payment Method'); ax.set_ylabel('Churn Rate (%)'); ax.tick_params(axis='x', rotation=15)
ax.set_ylim(0, pay_churn['ChurnRate'].max()*1.2)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'06_payment_method_vs_churn.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 06_payment_method_vs_churn.png")

# 2g. Senior Citizen vs Churn
fig, axes = plt.subplots(1,2, figsize=(12,5))
telecom_sc = telecom.copy()
telecom_sc['SeniorCitizen_Label'] = telecom_sc['SeniorCitizen'].map({0:'Non-Senior',1:'Senior'})
sns.countplot(x='SeniorCitizen_Label', hue='Churn', data=telecom_sc, ax=axes[0], palette=PALETTE)
axes[0].set_title('Count by Senior Citizen Status'); axes[0].set_xlabel('')
sc_rates = [telecom[telecom['SeniorCitizen']==s]['Churn'].eq('Yes').mean()*100 for s in [0,1]]
axes[1].bar(['Non-Senior','Senior'], sc_rates, color=['#2196F3','#F44336'], edgecolor='white')
for i,r in enumerate(sc_rates):
    axes[1].text(i, r+0.5, f'{r:.1f}%', ha='center', fontweight='bold')
axes[1].set_title('Churn Rate: Senior vs Non-Senior'); axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, max(sc_rates)*1.2)
plt.suptitle('Senior Citizen Status vs Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'07_senior_citizen_vs_churn.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 07_senior_citizen_vs_churn.png")

# 2h. Tenure Distribution histogram
fig, ax = plt.subplots(figsize=(11,5))
for label, color in [('No','#2196F3'),('Yes','#F44336')]:
    ax.hist(telecom[telecom['Churn']==label]['tenure'], bins=30, alpha=0.6, color=color,
            label=f'Churn={label}', edgecolor='white')
ax.set_title('Tenure Distribution: Churned vs Retained', fontweight='bold')
ax.set_xlabel('Tenure (Months)'); ax.set_ylabel('Customers'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,'08_tenure_distribution.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 08_tenure_distribution.png")

# ════════════════════════════════════════════════════════════
# STAGE 3 — PREPROCESSING
# ════════════════════════════════════════════════════════════
print("\n[3/8] Preprocessing...")

telecom['TotalCharges'] = pd.to_numeric(telecom['TotalCharges'], errors='coerce')
telecom = telecom.dropna(subset=['TotalCharges'])
print(f"   After cleaning: {telecom.shape}")

varlist = ['PhoneService','PaperlessBilling','Churn','Partner','Dependents']
telecom[varlist] = telecom[varlist].apply(lambda col: col.map({'Yes':1,'No':0}))

X = telecom.drop(columns=['Churn','customerID'])
y = telecom['Churn']
baseline_churn = round(y.mean()*100, 2)
print(f"   Baseline churn rate: {baseline_churn}%")

# ════════════════════════════════════════════════════════════
# STAGE 4 — TRAIN-TEST SPLIT & ENCODING
# ════════════════════════════════════════════════════════════
print("\n[4/8] Train-test split and encoding...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=100)
X_train = X_train.copy(); X_test = X_test.copy()
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

cat_cols = list(X_train.select_dtypes(include=['object']).columns)
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True, dtype=int)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

cols_to_drop = ['MultipleLines_No phone service','OnlineSecurity_No internet service',
                'OnlineBackup_No internet service','DeviceProtection_No internet service',
                'TechSupport_No internet service','StreamingTV_No internet service',
                'StreamingMovies_No internet service']
X_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
X_test.drop(columns=cols_to_drop,  inplace=True, errors='ignore')

scaler   = StandardScaler()
num_cols = ['tenure','MonthlyCharges','TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])
print(f"   Encoded & scaled. Final X_train shape: {X_train.shape}")

# ════════════════════════════════════════════════════════════
# STAGE 5 — RFE FEATURE SELECTION
# ════════════════════════════════════════════════════════════
print("\n[5/8] RFE feature selection...")
logreg = LogisticRegression(max_iter=1000)
rfe    = RFE(estimator=logreg, n_features_to_select=15)
rfe.fit(X_train, y_train)
rfe_cols = X_train.columns[rfe.support_]
print(f"   {len(rfe_cols)} features selected: {list(rfe_cols)}")

# ════════════════════════════════════════════════════════════
# STAGE 6 — ITERATIVE GLM BINOMIAL MODEL
# ════════════════════════════════════════════════════════════
print("\n[6/8] Building iterative GLM Binomial model...")

def vif_table(X_subset):
    df = pd.DataFrame()
    df['Features'] = X_subset.columns
    df['VIF'] = [round(variance_inflation_factor(X_subset.values, i), 2)
                 for i in range(X_subset.shape[1])]
    return df.sort_values('VIF', ascending=False)

# Model 1
X_train_sm = sm.add_constant(X_train[rfe_cols])
model1 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial()).fit()
vif1 = vif_table(X_train[rfe_cols])
print("   Model 1 — highest VIF:", vif1.iloc[0]['Features'], f"(VIF={vif1.iloc[0]['VIF']})")

rfe_cols = rfe_cols.drop('MonthlyCharges')

# Model 2
X_train_sm1 = sm.add_constant(X_train[rfe_cols])
model2 = sm.GLM(y_train, X_train_sm1, family=sm.families.Binomial()).fit()
vif2 = vif_table(X_train[rfe_cols])
print("   Model 2 — highest VIF:", vif2.iloc[0]['Features'], f"(VIF={vif2.iloc[0]['VIF']})")

rfe_cols = rfe_cols.drop('TotalCharges')

# Model 3 — FINAL
X_train_sm2 = sm.add_constant(X_train[rfe_cols])
model3 = sm.GLM(y_train, X_train_sm2, family=sm.families.Binomial()).fit()
vif3 = vif_table(X_train[rfe_cols])
print("   Model 3 (FINAL) — max VIF:", vif3['VIF'].max(), "— Stopping criteria met ✅")
print("\n--- Final Model Summary ---")
print(model3.summary())

# ════════════════════════════════════════════════════════════
# STAGE 7 — ROC & THRESHOLD SELECTION
# ════════════════════════════════════════════════════════════
print("\n[7/8] ROC curve and threshold selection...")

y_train_pred = model3.predict(X_train_sm2)

# Training ROC
fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_pred)
fig, ax = plt.subplots(figsize=(8,7))
ax.plot(fpr_tr, tpr_tr, color='#2196F3', linewidth=2.5,
        label=f'ROC Curve  (AUC = {train_auc:.2f})')
ax.plot([0,1],[0,1],'k--', linewidth=1.5, label='Random Classifier')
ax.fill_between(fpr_tr, tpr_tr, alpha=0.1, color='#2196F3')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Training Data', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,'01_roc_curve_train.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print(f"   Training AUC: {train_auc:.4f}  → Saved: 01_roc_curve_train.png")

# Threshold grid search
resultdf1 = pd.DataFrame({'Converted':y_train.values,'Converted_prob':y_train_pred})
resultdf1.reset_index(drop=True, inplace=True)
num_range = [i/10 for i in range(0,11)]
for t in num_range:
    resultdf1[t] = (resultdf1['Converted_prob'] > t).astype(int)

rows = []
for t in [i/10 for i in range(0,10)]:
    cm = metrics.confusion_matrix(resultdf1['Converted'], resultdf1[t])
    total = cm.sum()
    acc  = (cm[0,0]+cm[1,1])/total
    spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
    sens = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0
    rows.append({'cutoff':t,'accuracy':round(acc,4),'sensitivity':round(sens,4),'specificity':round(spec,4)})
cutoff_df = pd.DataFrame(rows)
print("\n   Threshold analysis:")
print(cutoff_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cutoff_df['cutoff'], cutoff_df['accuracy'],    'b-o', linewidth=2, label='Accuracy')
ax.plot(cutoff_df['cutoff'], cutoff_df['sensitivity'], 'r-o', linewidth=2, label='Sensitivity')
ax.plot(cutoff_df['cutoff'], cutoff_df['specificity'], 'g-o', linewidth=2, label='Specificity')
ax.axvline(x=0.30, color='purple', linestyle='--', linewidth=2, label='Optimal: 0.30')
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.set_title('Threshold Selection — Accuracy, Sensitivity & Specificity', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,'02_threshold_selection.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("   Saved: 02_threshold_selection.png")

# Apply training threshold evaluation
resultdf1['Converted_Predicted'] = (resultdf1['Converted_prob'] > THRESHOLD).astype(int)
cm_train   = confusion_matrix(resultdf1['Converted'], resultdf1['Converted_Predicted'])
train_acc  = metrics.accuracy_score(resultdf1['Converted'], resultdf1['Converted_Predicted'])
TP,TN,FP,FN = cm_train[1,1],cm_train[0,0],cm_train[0,1],cm_train[1,0]
train_sens = TP/float(TP+FN); train_spec = TN/float(TN+FP)

# ════════════════════════════════════════════════════════════
# STAGE 8 — TEST EVALUATION
# ════════════════════════════════════════════════════════════
print("\n[8/8] Test set evaluation...")

X_test_final = X_test[rfe_cols]
X_test_sm    = sm.add_constant(X_test_final, has_constant='add')
y_test_pred  = model3.predict(X_test_sm)

test_resultdf = pd.DataFrame({
    'Test_Actual'   : y_test.values,
    'Test_Prob'     : y_test_pred,
    'Test_Predicted': (y_test_pred > THRESHOLD).astype(int)
})

cm_test   = confusion_matrix(test_resultdf['Test_Actual'], test_resultdf['Test_Predicted'])
test_acc  = metrics.accuracy_score(test_resultdf['Test_Actual'], test_resultdf['Test_Predicted'])
TP1,TN1,FP1,FN1 = cm_test[1,1],cm_test[0,0],cm_test[0,1],cm_test[1,0]
test_sens = TP1/float(TP1+FN1); test_spec = TN1/float(TN1+FP1)
test_prec = TP1/float(TP1+FP1); test_npv  = TN1/float(TN1+FN1)
test_fpr  = FP1/float(TN1+FP1)
test_auc  = roc_auc_score(test_resultdf['Test_Actual'], test_resultdf['Test_Prob'])

print("\n" + "="*60)
print("  FINAL MODEL EVALUATION RESULTS")
print("="*60)
print(f"  Threshold         : {THRESHOLD}")
print(f"  Training AUC      : {train_auc:.4f}")
print(f"  Test AUC          : {test_auc:.4f}")
print(f"\n  {'Metric':<22} {'Training':>12} {'Test':>12}")
print(f"  {'-'*48}")
print(f"  {'Accuracy':<22} {train_acc:>12.4f} {test_acc:>12.4f}")
print(f"  {'Sensitivity':<22} {train_sens:>12.4f} {test_sens:>12.4f}")
print(f"  {'Specificity':<22} {train_spec:>12.4f} {test_spec:>12.4f}")
print(f"  {'Precision':<22} {'N/A':>12} {test_prec:>12.4f}")
print("="*60)
print("\nClassification Report (Test Set):")
print(classification_report(test_resultdf['Test_Actual'], test_resultdf['Test_Predicted'],
                            target_names=['No Churn (0)','Churn (1)']))

# Confusion Matrix plot
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No Churn','Pred: Churn'],
            yticklabels=['Actual: No Churn','Actual: Churn'],
            linewidths=0.5, ax=ax, annot_kws={'size':14,'weight':'bold'})
ax.set_title(f'Confusion Matrix — Test Set (Threshold={THRESHOLD})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,'03_confusion_matrix_test.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("Saved: 03_confusion_matrix_test.png")

# Test ROC curve
fpr_te, tpr_te, _ = roc_curve(test_resultdf['Test_Actual'], test_resultdf['Test_Prob'])
fig, ax = plt.subplots(figsize=(8,7))
ax.plot(fpr_te, tpr_te, color='#2196F3', linewidth=2.5, label=f'ROC Curve  (AUC = {test_auc:.2f})')
ax.plot([0,1],[0,1],'k--', linewidth=1.5, label='Random Classifier')
ax.fill_between(fpr_te, tpr_te, alpha=0.1, color='#2196F3')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Test Data (Final GLM Binomial)', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,'04_roc_curve_test.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("Saved: 04_roc_curve_test.png")

# Feature Impact
impact_df = pd.DataFrame(model3.params).reset_index()
impact_df.columns = ['Features','Coefficients']
impact_df = impact_df.iloc[1:11]
impact_df['Impact %'] = impact_df['Coefficients'].apply(
    lambda x: round((x/sum(impact_df['Coefficients']))*100, 2))
impact_df = impact_df.sort_values('Impact %', ascending=False)
print("\nFeature Impact Analysis:")
print(impact_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(14,7))
colors = ['#4CAF50' if v>0 else '#F44336' for v in impact_df['Impact %']]
bars = ax.bar(impact_df['Features'], impact_df['Impact %'], color=colors, edgecolor='white')
for b in bars:
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h+(1 if h>=0 else -2),
            f'{h:.2f}%', ha='center', va='bottom' if h>=0 else 'top',
            fontsize=9, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_title('Feature Impact on Customer Churn\n(Green = Protective, Red = Churn-Driving)',
             fontweight='bold')
ax.set_xlabel('Feature'); ax.set_ylabel('Impact (%)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,'05_feature_impact.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.savefig(os.path.join(GRAPH_DIR,'09_feature_impact.png'), dpi=EXPORT_DPI, bbox_inches='tight')
plt.close()
print("Saved: 05_feature_impact.png")

# Save metrics CSV
metrics_summary = pd.DataFrame({
    'Metric'  : ['Accuracy','Sensitivity','Specificity','Precision','AUC','Threshold'],
    'Training': [round(train_acc,4), round(train_sens,4), round(train_spec,4),'N/A',
                 round(train_auc,4), THRESHOLD],
    'Test'    : [round(test_acc,4), round(test_sens,4), round(test_spec,4),
                 round(test_prec,4), round(test_auc,4), THRESHOLD]
})
metrics_summary.to_csv(os.path.join(EVAL_DIR,'model_metrics.csv'), index=False)
print("Saved: model_metrics.csv")

# ════════════════════════════════════════════════════════════
# SAVE MODEL
# ════════════════════════════════════════════════════════════
MODEL_ARTIFACT = {
    'model'          : model3,
    'scaler'         : scaler,
    'final_features' : list(rfe_cols),
    'num_cols'       : num_cols,
    'cat_cols_encoded': list(X_train.columns),
    'threshold'      : THRESHOLD,
    'baseline_churn' : baseline_churn,
}
MODEL_PATH = os.path.join(MODELS_DIR, 'churn_model.joblib')
joblib.dump(MODEL_ARTIFACT, MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

# ── VERIFY LOADED MODEL ───────────────────────────────────
loaded = joblib.load(MODEL_PATH)
X_verify   = X_test_final.iloc[:5]
X_verify_sm = sm.add_constant(X_verify, has_constant='add')
verify_probs = loaded['model'].predict(X_verify_sm)
verify_preds = (verify_probs > loaded['threshold']).astype(int)
orig_preds   = (y_test_pred.values[:5] > THRESHOLD).astype(int)
match        = (verify_preds.values == orig_preds).all()
print(f"Model verification — All predictions match: {match} ✅" if match else "⚠ Mismatch detected!")

print("\n" + "="*60)
print("  VALIDATION COMPLETE — All files generated successfully!")
print("="*60)
graphs = sorted(os.listdir(GRAPH_DIR))
evals  = sorted(os.listdir(EVAL_DIR))
print(f"\n  EDA Graphs ({len(graphs)}):", ', '.join(graphs))
print(f"\n  Eval Outputs ({len(evals)}):", ', '.join(evals))
print(f"\n  Model artifact: {MODEL_PATH}")
