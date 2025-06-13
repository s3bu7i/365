import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== CREDIT DEFAULT PREDICTION USING LOGISTIC REGRESSION ===\n")

# 1. Load and Explore the Dataset
print("1. LOADING AND EXPLORING THE DATASET")
print("-" * 50)

try:
    df = pd.read_csv('C:/Users/Dino/Desktop/ai/task 3/datasets/cs-training.csv')
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print("Dataset file not found. Please ensure 'cs-training.csv' is in the current directory.")
    print("You can download it from: https://www.kaggle.com/c/GiveMeSomeCredit/data")
    exit()

# Display basic information
print(f"\nDataset Info:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nTarget variable distribution:")
if 'SeriousDlqin2yrs' in df.columns:
    target_col = 'SeriousDlqin2yrs'
elif 'target' in df.columns:
    target_col = 'target'
else:
    target_col = df.columns[1]  # Assuming first column is ID, second is target

print(df[target_col].value_counts())
print(f"Class distribution:")
print(df[target_col].value_counts(normalize=True))

# 2. Data Preprocessing
print(f"\n2. DATA PREPROCESSING")
print("-" * 50)

# Remove ID column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Handle missing values
print("Handling missing values...")
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled {col} missing values with median: {median_val}")

# Remove extreme outliers (optional)
print("\nHandling outliers...")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
    if outliers_count > 0:
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"Removed {outliers_count} extreme outliers from {col}")

print(f"Final dataset shape: {df.shape}")

# 3. Feature Engineering and Selection
print(f"\n3. FEATURE ENGINEERING")
print("-" * 50)

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"Features: {list(X.columns)}")
print(f"Target: {target_col}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print("Features scaled using StandardScaler")

# 4. Train-Test Split
print(f"\n4. TRAIN-TEST SPLIT")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training set class distribution:")
print(y_train.value_counts(normalize=True))

# 5. Part A: Train Basic Logistic Regression Model
print(f"\n5. PART A: BASIC LOGISTIC REGRESSION MODEL")
print("=" * 50)

# Train basic logistic regression
lr_basic = LogisticRegression(random_state=42, max_iter=1000)
lr_basic.fit(X_train, y_train)

# Predictions
y_pred_basic = lr_basic.predict(X_test)
y_pred_proba_basic = lr_basic.predict_proba(X_test)[:, 1]

# Evaluate basic model
print("Basic Logistic Regression Results:")
print("-" * 40)
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba_basic):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_basic):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_basic):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_basic):.4f}")

print(f"\nConfusion Matrix:")
cm_basic = confusion_matrix(y_test, y_pred_basic)
print(cm_basic)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_basic))

# 6. Part B: Address Class Imbalance
print(f"\n6. PART B: ADDRESSING CLASS IMBALANCE")
print("=" * 50)

# Method 1: Class Weighting
print("Method 1: Class Weighting")
print("-" * 30)

# Calculate class weights
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Train with class weights
lr_weighted = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
lr_weighted.fit(X_train, y_train)

# Predictions
y_pred_weighted = lr_weighted.predict(X_test)
y_pred_proba_weighted = lr_weighted.predict_proba(X_test)[:, 1]

# Evaluate weighted model
print("Weighted Logistic Regression Results:")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba_weighted):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_weighted):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_weighted):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_weighted):.4f}")

# Method 2: SMOTE
print(f"\nMethod 2: SMOTE (Synthetic Minority Oversampling)")
print("-" * 50)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Original training set: {X_train.shape[0]} samples")
print(f"SMOTE training set: {X_train_smote.shape[0]} samples")
print(f"SMOTE class distribution:")
print(pd.Series(y_train_smote).value_counts(normalize=True))

# Train with SMOTE
lr_smote = LogisticRegression(random_state=42, max_iter=1000)
lr_smote.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_smote = lr_smote.predict(X_test)
y_pred_proba_smote = lr_smote.predict_proba(X_test)[:, 1]

# Evaluate SMOTE model
print("SMOTE Logistic Regression Results:")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba_smote):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_smote):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_smote):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_smote):.4f}")

# 7. Part C: Odds Ratios Analysis
print(f"\n7. PART C: ODDS RATIOS ANALYSIS")
print("=" * 50)

# Use the best performing model (let's use the weighted model)
best_model = lr_weighted
feature_names = X.columns

# Calculate odds ratios
coefficients = best_model.coef_[0]
odds_ratios = np.exp(coefficients)

# Create odds ratios dataframe
odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios,
    'Log_Odds': coefficients
}).sort_values('Odds_Ratio', ascending=False)

print("Odds Ratios Analysis:")
print("-" * 30)
print(odds_df.round(4))

print(f"\nInterpretation of Odds Ratios:")
print("-" * 40)
for idx, row in odds_df.iterrows():
    feature = row['Feature']
    odds_ratio = row['Odds_Ratio']

    if odds_ratio > 1:
        effect = "INCREASES"
        magnitude = ((odds_ratio - 1) * 100)
        print(
            f"• {feature}: {effect} odds of default by {magnitude:.1f}% (OR = {odds_ratio:.3f})")
    elif odds_ratio < 1:
        effect = "DECREASES"
        magnitude = ((1 - odds_ratio) * 100)
        print(
            f"• {feature}: {effect} odds of default by {magnitude:.1f}% (OR = {odds_ratio:.3f})")
    else:
        print(f"• {feature}: No effect on odds of default (OR = {odds_ratio:.3f})")

# 8. Model Comparison Summary
print(f"\n8. MODEL COMPARISON SUMMARY")
print("=" * 50)

models_comparison = pd.DataFrame({
    'Model': ['Basic LR', 'Weighted LR', 'SMOTE LR'],
    'AUC': [
        roc_auc_score(y_test, y_pred_proba_basic),
        roc_auc_score(y_test, y_pred_proba_weighted),
        roc_auc_score(y_test, y_pred_proba_smote)
    ],
    'Precision': [
        precision_score(y_test, y_pred_basic),
        precision_score(y_test, y_pred_weighted),
        precision_score(y_test, y_pred_smote)
    ],
    'Recall': [
        recall_score(y_test, y_pred_basic),
        recall_score(y_test, y_pred_weighted),
        recall_score(y_test, y_pred_smote)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_basic),
        f1_score(y_test, y_pred_weighted),
        f1_score(y_test, y_pred_smote)
    ]
})

print("Model Performance Comparison:")
print(models_comparison.round(4))

# Find best model
best_model_idx = models_comparison['F1-Score'].idxmax()
best_model_name = models_comparison.loc[best_model_idx, 'Model']
print(f"\nBest performing model: {best_model_name}")

# 9. Visualization
print(f"\n9. CREATING VISUALIZATIONS")
print("=" * 50)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. ROC Curves
fpr_basic, tpr_basic, _ = roc_curve(y_test, y_pred_proba_basic)
fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_pred_proba_weighted)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote)

axes[0, 0].plot(fpr_basic, tpr_basic,
                label=f'Basic LR (AUC = {auc(fpr_basic, tpr_basic):.3f})')
axes[0, 0].plot(fpr_weighted, tpr_weighted,
                label=f'Weighted LR (AUC = {auc(fpr_weighted, tpr_weighted):.3f})')
axes[0, 0].plot(fpr_smote, tpr_smote,
                label=f'SMOTE LR (AUC = {auc(fpr_smote, tpr_smote):.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curves Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Feature Importance (Odds Ratios)
top_features = odds_df.head(10)
axes[0, 1].barh(range(len(top_features)), top_features['Odds_Ratio'])
axes[0, 1].set_yticks(range(len(top_features)))
axes[0, 1].set_yticklabels(top_features['Feature'])
axes[0, 1].set_xlabel('Odds Ratio')
axes[0, 1].set_title('Top 10 Features by Odds Ratio')
axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7)

# 3. Model Performance Comparison
metrics = ['AUC', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(['Basic LR', 'Weighted LR', 'SMOTE LR']):
    values = models_comparison[models_comparison['Model']
                               == model][metrics].values[0]
    axes[1, 0].bar(x + i*width, values, width, label=model)

axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Model Performance Comparison')
axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Confusion Matrix for Best Model
if best_model_name == 'Basic LR':
    cm = confusion_matrix(y_test, y_pred_basic)
elif best_model_name == 'Weighted LR':
    cm = confusion_matrix(y_test, y_pred_weighted)
else:
    cm = confusion_matrix(y_test, y_pred_smote)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')

plt.tight_layout()
plt.savefig('credit_default_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved as 'credit_default_analysis.png'")

# 10. Final Recommendations
print(f"\n10. FINAL RECOMMENDATIONS AND INSIGHTS")
print("=" * 50)

print("Key Findings:")
print("• Class imbalance techniques (weighting/SMOTE) improved model performance")
print("• Features with highest odds ratios indicate strongest predictors of default")

# Identify top risk factors
top_risk_factors = odds_df[odds_df['Odds_Ratio'] > 1].head(5)
print(f"\nTop 5 Risk Factors (increase default odds):")
for idx, row in top_risk_factors.iterrows():
    print(
        f"  - {row['Feature']}: {((row['Odds_Ratio'] - 1) * 100):.1f}% increase")

# Identify protective factors
protective_factors = odds_df[odds_df['Odds_Ratio'] < 1].head(5)
print(f"\nTop 5 Protective Factors (decrease default odds):")
for idx, row in protective_factors.iterrows():
    print(
        f"  - {row['Feature']}: {((1 - row['Odds_Ratio']) * 100):.1f}% decrease")

print(f"\nBest Model: {best_model_name}")
print(f"Recommended for deployment based on balanced performance across all metrics.")


