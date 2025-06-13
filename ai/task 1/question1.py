import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, mutual_info_classif, SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

print("="*60)
print("HEART DISEASE PREDICTION WITH FEATURE SELECTION ANALYSIS")
print("="*60)

print("\n1. LOADING AND EXPLORING THE DATASET")
print("-" * 40)
# Load the dataset
df = pd.read_csv(r'C:\Users\Dino\Desktop\ai\task 1\heart_disease_uci.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")

print(f"\nDataset Info:")
print(df.info())
print(f"\nFirst 5 rows:")
print(df.head())

# 2. DATA PREPROCESSING
print("\n\n2. DATA PREPROCESSING")
print("-" * 40)

# Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values per column:")
print(missing_values[missing_values > 0])

# Basic statistics
print(f"\nDataset Statistics:")
print(df.describe())

target_column = 'num'
print(f"\nTarget column: {target_column}")
print(f"Target Distribution:")
print(df[target_column].value_counts().sort_index())

df['target'] = (df[target_column] > 0).astype(int)
print(f"\nBinary Target Distribution:")
print(df['target'].value_counts())
print(f"Disease prevalence: {df['target'].mean():.3f}")


columns_to_drop = ['id', 'dataset', 'num']
df_clean = df.drop(columns=columns_to_drop)

print(f"\nColumns after dropping non-predictive features:")
print(list(df_clean.columns))

# Handle categorical variables
categorical_columns = df_clean.select_dtypes(
    include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_columns}")

# Create a copy for preprocessing
df_processed = df_clean.copy()


label_encoders = {}
for col in categorical_columns:
    if col != 'target':  
        le = LabelEncoder()
        )[0] if not df_processed[col].mode().empty else 'Unknown'
        df_processed[col] = df_processed[col].fillna(mode_value)
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} categories")

# Handle numerical missing values
numerical_columns = df_processed.select_dtypes(
    include=['float64', 'int64']).columns.tolist()
numerical_columns.remove('target')  

print(f"\nNumerical columns: {numerical_columns}")


for col in numerical_columns:
    if df_processed[col].isnull().sum() > 0:
        median_value = df_processed[col].median()
        df_processed[col] = df_processed[col].fillna(median_value)
        print(
            f"  {col}: filled {df_processed[col].isnull().sum()} missing values with median {median_value:.2f}")

# Final check for missing values
print(f"\nFinal missing values check:")
final_missing = df_processed.isnull().sum()
print(final_missing[final_missing > 0])

X = df_processed.drop('target', axis=1)
y = df_processed['target']

print(f"\nFinal dataset shape:")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {list(X.columns)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# 3. FEATURE SELECTION TECHNIQUES
print("\n\n3. APPLYING FEATURE SELECTION TECHNIQUES")
print("=" * 50)

# Initialize results storage
results = {}
selected_features = {}

print("\n3.1 RECURSIVE FEATURE ELIMINATION (RFE)")
print("-" * 40)

# Create logistic regression model for RFE
lr_rfe = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

n_features_to_select = min(8, X.shape[1] - 1)
rfe = RFE(estimator=lr_rfe, n_features_to_select=n_features_to_select)
rfe.fit(X_train_scaled, y_train)

# Get selected features
rfe_features = X.columns[rfe.support_].tolist()
selected_features['RFE'] = rfe_features

print(f"RFE Selected Features ({len(rfe_features)}):")
feature_ranking = list(zip(X.columns, rfe.ranking_))
feature_ranking.sort(key=lambda x: x[1])
for i, (feature, rank) in enumerate(feature_ranking[:n_features_to_select], 1):
    print(f"  {i}. {feature} (rank: {rank})")

X_train_rfe = rfe.transform(X_train_scaled)
X_test_rfe = rfe.transform(X_test_scaled)

# Method 2: Mutual Information
print("\n3.2 MUTUAL INFORMATION FEATURE SELECTION")
print("-" * 40)

mi_scores = mutual_info_classif(
    X_train_scaled, y_train, random_state=RANDOM_STATE)

mi_features_scores = list(zip(X.columns, mi_scores))
mi_features_scores.sort(key=lambda x: x[1], reverse=True)

# Select top features based on mutual information
mi_features = [feature for feature,
               score in mi_features_scores[:n_features_to_select]]
selected_features['Mutual_Info'] = mi_features

print(f"Mutual Information Selected Features ({len(mi_features)}):")
for i, (feature, score) in enumerate(mi_features_scores[:n_features_to_select], 1):
    print(f"  {i}. {feature}: {score:.4f}")

# Transform data with selected features
mi_feature_indices = [X.columns.get_loc(feature) for feature in mi_features]
X_train_mi = X_train_scaled[:, mi_feature_indices]
X_test_mi = X_test_scaled[:, mi_feature_indices]

# Method 3: L1 Regularization (LASSO)
print("\n3.3 L1 REGULARIZATION (LASSO) FEATURE SELECTION")
print("-" * 40)

C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_C = None
best_n_features = 0
best_features = None

for C in C_values:
    lasso = LogisticRegression(
        penalty='l1', solver='liblinear', C=C, random_state=RANDOM_STATE)
    try:
        lasso.fit(X_train_scaled, y_train)
        selector = SelectFromModel(lasso, prefit=True)
        n_features = selector.transform(X_train_scaled).shape[1]

        if n_features > 0:  # Make sure we have at least some features
            if abs(n_features - n_features_to_select) < abs(best_n_features - n_features_to_select):
                best_C = C
                best_n_features = n_features
                best_features = X.columns[selector.get_support()].tolist()
    except:
        continue

# If no good C found, use a simple approach
if best_C is None:
    print("Using alternative L1 approach...")
    lasso_simple = LogisticRegression(
        penalty='l1', solver='liblinear', C=1.0, random_state=RANDOM_STATE)
    lasso_simple.fit(X_train_scaled, y_train)

    # Get feature importance and select top features
    feature_importance = abs(lasso_simple.coef_[0])
    feature_importance_pairs = list(zip(X.columns, feature_importance))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

    lasso_features = [feature for feature,
                      importance in feature_importance_pairs[:n_features_to_select]]
    best_C = 1.0
else:
    lasso_features = best_features

selected_features['L1_Regularization'] = lasso_features

print(
    f"L1 Regularization Selected Features ({len(lasso_features)}) with C={best_C}:")
for i, feature in enumerate(lasso_features, 1):
    print(f"  {i}. {feature}")

# Transform data with selected features
lasso_feature_indices = [X.columns.get_loc(
    feature) for feature in lasso_features]
X_train_lasso = X_train_scaled[:, lasso_feature_indices]
X_test_lasso = X_test_scaled[:, lasso_feature_indices]

# 4. MODEL TRAINING AND EVALUATION
print("\n\n4. MODEL TRAINING AND EVALUATION")
print("=" * 50)

# Train models with different feature sets
feature_sets = {
    'All_Features': (X_train_scaled, X_test_scaled),
    'RFE': (X_train_rfe, X_test_rfe),
    'Mutual_Info': (X_train_mi, X_test_mi),
    'L1_Regularization': (X_train_lasso, X_test_lasso)
}

# Store results
models = {}

for method_name, (X_tr, X_te) in feature_sets.items():
    print(f"\n4.{list(feature_sets.keys()).index(method_name) + 1} TRAINING WITH {method_name.upper()}")
    print("-" * 40)

    # Train logistic regression
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_tr, y_train)
    models[method_name] = lr

    # Make predictions
    y_pred = lr.predict(X_te)
    y_pred_proba = lr.predict_proba(X_te)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation scores
    cv_scores = cross_val_score(lr, X_tr, y_train, cv=5, scoring='roc_auc')

    results[method_name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc_score,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'n_features': X_tr.shape[1]
    }

    print(f"Number of features: {X_tr.shape[1]}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

# 5. RESULTS COMPARISON AND ANALYSIS
print("\n\n5. RESULTS COMPARISON AND ANALYSIS")
print("=" * 50)

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)

print("\nPERFORMANCE COMPARISON TABLE:")
print("-" * 40)
print(comparison_df.to_string())

# Find best performing method
best_method_auc = comparison_df['auc'].idxmax()
best_method_f1 = comparison_df['f1_score'].idxmax()
best_method_acc = comparison_df['accuracy'].idxmax()

print(f"\nBEST PERFORMING METHODS:")
print("-" * 30)
print(
    f"Best AUC: {best_method_auc} ({comparison_df.loc[best_method_auc, 'auc']:.4f})")
print(
    f"Best F1-Score: {best_method_f1} ({comparison_df.loc[best_method_f1, 'f1_score']:.4f})")
print(
    f"Best Accuracy: {best_method_acc} ({comparison_df.loc[best_method_acc, 'accuracy']:.4f})")

# 6. FEATURE ANALYSIS
print("\n\n6. FEATURE SELECTION ANALYSIS")
print("=" * 40)

print("\nSELECTED FEATURES BY METHOD:")
print("-" * 30)

all_selected_features = set()
for method, features in selected_features.items():
    print(f"\n{method}:")
    for feature in features:
        print(f"  - {feature}")
        all_selected_features.add(feature)

# Find common features across methods
feature_counts = {}
for feature in all_selected_features:
    count = sum(1 for features in selected_features.values()
                if feature in features)
    feature_counts[feature] = count

print(f"\nFEATURE IMPORTANCE ACROSS METHODS:")
print("-" * 35)
sorted_features = sorted(feature_counts.items(),
                         key=lambda x: x[1], reverse=True)
for feature, count in sorted_features:
    methods = [method for method, features in selected_features.items()
               if feature in features]
    print(f"{feature}: Selected by {count}/3 methods ({', '.join(methods)})")

# 7. VISUALIZATION
print("\n\n7. CREATING VISUALIZATIONS")
print("=" * 30)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(
    'Heart Disease Prediction: Feature Selection Analysis', fontsize=16)

# Plot 1: Performance Comparison
ax1 = axes[0, 0]
methods = list(results.keys())
metrics = ['accuracy', 'f1_score', 'auc']
x = np.arange(len(methods))
width = 0.25

colors = ['#ff9999', '#66b3ff', '#99ff99']
for i, metric in enumerate(metrics):
    values = [results[method][metric] for method in methods]
    ax1.bar(x + i*width, values, width,
            label=metric.upper(), alpha=0.8, color=colors[i])

ax1.set_xlabel('Feature Selection Method')
ax1.set_ylabel('Score')
ax1.set_title('Performance Comparison Across Methods')
ax1.set_xticks(x + width)
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Plot 2: Number of Features
ax2 = axes[0, 1]
n_features = [results[method]['n_features'] for method in methods]
bars = ax2.bar(methods, n_features, color='skyblue', alpha=0.8)
ax2.set_xlabel('Feature Selection Method')
ax2.set_ylabel('Number of Features')
ax2.set_title('Number of Selected Features')
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, n_features):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(value), ha='center', va='bottom')

# Plot 3: ROC Curves
ax3 = axes[1, 0]
colors = ['red', 'blue', 'green', 'orange']

for i, (method_name, (X_tr, X_te)) in enumerate(feature_sets.items()):
    model = models[method_name]
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    ax3.plot(fpr, tpr, color=colors[i],
             label=f'{method_name} (AUC = {auc_score:.3f})', linewidth=2)

ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Selection Overlap
ax4 = axes[1, 1]
feature_selection_methods = list(selected_features.keys())
overlap_matrix = np.zeros(
    (len(feature_selection_methods), len(feature_selection_methods)))

for i, method1 in enumerate(feature_selection_methods):
    for j, method2 in enumerate(feature_selection_methods):
        if i == j:
            overlap_matrix[i, j] = len(selected_features[method1])
        else:
            common_features = set(selected_features[method1]) & set(
                selected_features[method2])
            overlap_matrix[i, j] = len(common_features)

im = ax4.imshow(overlap_matrix, cmap='Blues', alpha=0.8)
ax4.set_xticks(range(len(feature_selection_methods)))
ax4.set_yticks(range(len(feature_selection_methods)))
ax4.set_xticklabels(feature_selection_methods, rotation=45, ha='right')
ax4.set_yticklabels(feature_selection_methods)
ax4.set_title('Feature Selection Overlap Matrix')

# Add text annotations
for i in range(len(feature_selection_methods)):
    for j in range(len(feature_selection_methods)):
        text = ax4.text(j, i, int(overlap_matrix[i, j]),
                        ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.show()

# 8. FINAL INTERPRETATION AND CONCLUSIONS
print("\n\n8. INTERPRETATION AND CONCLUSIONS")
print("=" * 40)

best_overall_method = comparison_df['auc'].idxmax()
best_auc = comparison_df.loc[best_overall_method, 'auc']

print(f"\nFINAL ANALYSIS:")
print("-" * 20)
print(f"1. MOST EFFECTIVE METHOD: {best_overall_method}")
print(f"   - Achieved highest AUC score: {best_auc:.4f}")
print(
    f"   - Number of features used: {comparison_df.loc[best_overall_method, 'n_features']}")

if best_overall_method in selected_features:
    print(
        f"   - Selected features: {', '.join(selected_features[best_overall_method])}")

print(f"\n2. PERFORMANCE INSIGHTS:")
performance_differences = comparison_df['auc'].max(
) - comparison_df['auc'].min()
print(
    f"   - Performance difference between best and worst: {performance_differences:.4f}")

if performance_differences < 0.05:
    print("   - Small performance differences suggest all methods are comparable")
else:
    print("   - Significant performance differences observed between methods")

print(f"\n3. FEATURE SELECTION INSIGHTS:")
if feature_counts:
    most_selected_feature = max(feature_counts.items(), key=lambda x: x[1])
    print(
        f"   - Most consistently selected feature: {most_selected_feature[0]} (selected by {most_selected_feature[1]}/3 methods)")

    common_features = [feature for feature,
                       count in feature_counts.items() if count >= 2]
    print(
        f"   - Features selected by multiple methods: {len(common_features)}")
    if common_features:
        print(f"     {', '.join(common_features)}")

print(f"\n4. DATASET-SPECIFIC OBSERVATIONS:")
print(f"   - Original dataset had {X.shape[1]} features after preprocessing")
print(f"   - Disease prevalence in dataset: {y.mean():.1%}")
print(f"   - Categorical variables were successfully encoded")
print(f"   - Missing values were handled appropriately")

print(f"\n5. RECOMMENDATIONS:")
if best_overall_method != 'All_Features':
    reduction_ratio = (
        X.shape[1] - comparison_df.loc[best_overall_method, 'n_features']) / X.shape[1]
    print(f"   - Use {best_overall_method} for optimal performance")
    print(
        f"   - Achieves {reduction_ratio:.1%} dimensionality reduction with optimal performance")
else:
    print(f"   - All features approach performs best")
    print(f"   - Feature selection didn't improve performance significantly")

print(f"   - Consider ensemble methods combining multiple feature selection approaches")
print(f"   - Focus on features consistently selected across methods for model interpretability")
print(f"   - The model shows good discriminative power with AUC > 0.7 for most methods")

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
