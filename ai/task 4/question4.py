import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load and Explore the Dataset
print("=== STEP 1: LOADING AND EXPLORING DATASET ===")

try:
    # The Spambase dataset doesn't have headers, so we need to add them
    column_names = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
        'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
        'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
        'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
        'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
        'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
        'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
        'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
        'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
        'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_semicolon', 'char_freq_parenthesis',
        'char_freq_bracket', 'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash',
        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam'
    ]

    # Try to load the dataset
    df = pd.read_csv('C://Users/Dino/Desktop/ai/task 4/spambase/spambase.data', header=None, names=column_names)
    print("Dataset loaded successfully!")

except FileNotFoundError:
    print("Dataset file not found. Please ensure 'spambase.data' is in your working directory.")
    print("You can download it from: https://archive.ics.uci.edu/dataset/94/spambase")
    # Create dummy data for demonstration
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(4601, 58), columns=column_names[:-1])
    df['spam'] = np.random.randint(0, 2, 4601)
    print("Using dummy data for demonstration purposes.")

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1]-1}")
print(f"Samples: {df.shape[0]}")

# Basic statistics
print("\n=== DATASET OVERVIEW ===")
print(f"Spam emails: {df['spam'].sum()} ({df['spam'].mean()*100:.1f}%)")
print(
    f"Ham emails: {(df['spam']==0).sum()} ({(1-df['spam'].mean())*100:.1f}%)")
print(f"Missing values: {df.isnull().sum().sum()}")

# Step 2: Data Preprocessing
print("\n=== STEP 2: DATA PREPROCESSING ===")

# Separate features and target
X = df.drop('spam', axis=1)
y = df['spam']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# Step 3a: Train SVM with Linear Kernel
print("\n=== STEP 3A: SVM WITH LINEAR KERNEL ===")

# Linear SVM
svm_linear = SVC(kernel='linear', random_state=42, probability=True)
svm_linear.fit(X_train_scaled, y_train)

# Predictions
y_pred_linear = svm_linear.predict(X_test_scaled)
y_prob_linear = svm_linear.predict_proba(X_test_scaled)[:, 1]

# Performance metrics
accuracy_linear = accuracy_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)
auc_linear = roc_auc_score(y_test, y_prob_linear)

print(f"Linear SVM Results:")
print(f"Accuracy: {accuracy_linear:.4f}")
print(f"F1-Score: {f1_linear:.4f}")
print(f"AUC: {auc_linear:.4f}")

# Step 3b: Train SVM with RBF Kernel (before optimization)
print("\n=== STEP 3B: SVM WITH RBF KERNEL (DEFAULT) ===")

# RBF SVM with default parameters
svm_rbf = SVC(kernel='rbf', random_state=42, probability=True)
svm_rbf.fit(X_train_scaled, y_train)

# Predictions
y_pred_rbf = svm_rbf.predict(X_test_scaled)
y_prob_rbf = svm_rbf.predict_proba(X_test_scaled)[:, 1]

# Performance metrics
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf)
auc_rbf = roc_auc_score(y_test, y_prob_rbf)

print(f"RBF SVM Results (default parameters):")
print(f"Accuracy: {accuracy_rbf:.4f}")
print(f"F1-Score: {f1_rbf:.4f}")
print(f"AUC: {auc_rbf:.4f}")

# Step 3c: Grid Search for RBF Kernel Optimization
print("\n=== STEP 3C: GRID SEARCH FOR RBF KERNEL OPTIMIZATION ===")

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

print("Performing Grid Search...")
print("Parameter grid:")
print(f"C: {param_grid['C']}")
print(f"gamma: {param_grid['gamma']}")

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=42, probability=True),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

# Train optimized RBF SVM
svm_rbf_optimized = grid_search.best_estimator_
y_pred_rbf_opt = svm_rbf_optimized.predict(X_test_scaled)
y_prob_rbf_opt = svm_rbf_optimized.predict_proba(X_test_scaled)[:, 1]

# Performance metrics for optimized RBF
accuracy_rbf_opt = accuracy_score(y_test, y_pred_rbf_opt)
f1_rbf_opt = f1_score(y_test, y_pred_rbf_opt)
auc_rbf_opt = roc_auc_score(y_test, y_prob_rbf_opt)

print(f"\nOptimized RBF SVM Results:")
print(f"Accuracy: {accuracy_rbf_opt:.4f}")
print(f"F1-Score: {f1_rbf_opt:.4f}")
print(f"AUC: {auc_rbf_opt:.4f}")

# Step 4: Comprehensive Performance Comparison
print("\n=== STEP 4: PERFORMANCE COMPARISON ===")

# Create comparison table
results_df = pd.DataFrame({
    'Model': ['Linear SVM', 'RBF SVM (default)', 'RBF SVM (optimized)'],
    'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_rbf_opt],
    'F1-Score': [f1_linear, f1_rbf, f1_rbf_opt],
    'AUC': [auc_linear, auc_rbf, auc_rbf_opt]
})

print(results_df.round(4))

# Detailed classification reports
print("\n=== DETAILED CLASSIFICATION REPORTS ===")

print("\nLinear SVM:")
print(classification_report(y_test, y_pred_linear))

print("\nOptimized RBF SVM:")
print(classification_report(y_test, y_pred_rbf_opt))

# Step 5: Visualization
print("\n=== STEP 5: VISUALIZATIONS ===")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Performance Comparison Bar Plot
ax1 = axes[0, 0]
metrics = ['Accuracy', 'F1-Score', 'AUC']
linear_scores = [accuracy_linear, f1_linear, auc_linear]
rbf_opt_scores = [accuracy_rbf_opt, f1_rbf_opt, auc_rbf_opt]

x = np.arange(len(metrics))
width = 0.35

ax1.bar(x - width/2, linear_scores, width, label='Linear SVM', alpha=0.8)
ax1.bar(x + width/2, rbf_opt_scores, width,
        label='RBF SVM (optimized)', alpha=0.8)

ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.set_title('Performance Comparison: Linear vs Optimized RBF SVM')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. ROC Curves
ax2 = axes[0, 1]
fpr_linear, tpr_linear, _ = roc_curve(y_test, y_prob_linear)
fpr_rbf_opt, tpr_rbf_opt, _ = roc_curve(y_test, y_prob_rbf_opt)

ax2.plot(fpr_linear, tpr_linear, label=f'Linear SVM (AUC = {auc_linear:.3f})')
ax2.plot(fpr_rbf_opt, tpr_rbf_opt,
         label=f'RBF SVM Optimized (AUC = {auc_rbf_opt:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix - Linear SVM
ax3 = axes[1, 0]
cm_linear = confusion_matrix(y_test, y_pred_linear)
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title('Confusion Matrix - Linear SVM')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# 4. Confusion Matrix - Optimized RBF SVM
ax4 = axes[1, 1]
cm_rbf_opt = confusion_matrix(y_test, y_pred_rbf_opt)
sns.heatmap(cm_rbf_opt, annot=True, fmt='d', cmap='Greens', ax=ax4)
ax4.set_title('Confusion Matrix - Optimized RBF SVM')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Grid Search Results Visualization
print("\n=== GRID SEARCH RESULTS ANALYSIS ===")

# Convert grid search results to DataFrame for better analysis
results = pd.DataFrame(grid_search.cv_results_)
pivot_table = results.pivot_table(
    values='mean_test_score', index='param_C', columns='param_gamma')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Grid Search Results: F1-Score Heatmap')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.show()

# Step 6: Analysis and Conclusions
print("\n=== STEP 6: ANALYSIS AND CONCLUSIONS ===")

print("PERFORMANCE ANALYSIS:")
print("====================")

best_model = "Linear SVM" if f1_linear > f1_rbf_opt else "Optimized RBF SVM"
print(f"Best performing model: {best_model}")

print(f"\nPerformance Improvements:")
print(f"RBF optimization improved F1-score by: {(f1_rbf_opt - f1_rbf):.4f}")
print(
    f"RBF optimization improved accuracy by: {(accuracy_rbf_opt - accuracy_rbf):.4f}")

print(f"\nModel Comparison:")
if f1_linear > f1_rbf_opt:
    print("• Linear SVM outperforms RBF SVM")
    print("• This suggests the data is linearly separable")
    print("• Linear SVM is also faster and more interpretable")
else:
    print("• RBF SVM outperforms Linear SVM")
    print("• This suggests non-linear patterns in the data")
    print("• RBF kernel captures complex relationships better")

print(f"\nHyperparameter Analysis:")
print(f"• Best C value: {grid_search.best_params_['C']}")
print(f"• Best gamma value: {grid_search.best_params_['gamma']}")

if grid_search.best_params_['C'] == 0.1:
    print("• Low C suggests preference for simpler model (less overfitting)")
elif grid_search.best_params_['C'] == 100:
    print("• High C suggests need for complex model (low bias)")

if grid_search.best_params_['gamma'] == 'scale':
    print("• Gamma = 'scale' uses 1/(n_features * X.var()) as value")
elif isinstance(grid_search.best_params_['gamma'], float):
    if grid_search.best_params_['gamma'] < 0.01:
        print("• Low gamma creates smooth decision boundary")
    else:
        print("• Higher gamma creates more complex decision boundary")

print(f"\nRECOMMENDATION:")
print(f"================")
if abs(f1_linear - f1_rbf_opt) < 0.01:
    print("Performance difference is minimal (<1%).")
    print("Recommendation: Use Linear SVM for:")
    print("• Better interpretability")
    print("• Faster training and prediction")
    print("• Lower computational complexity")
else:
    if f1_linear > f1_rbf_opt:
        print("Linear SVM is recommended because:")
        print("• Superior performance")
        print("• Simpler model (Occam's razor)")
        print("• Faster execution")
        print("• Better generalization potential")
    else:
        print("Optimized RBF SVM is recommended because:")
        print("• Superior performance on test set")
        print("• Better handling of non-linear patterns")
        print("• Optimized hyperparameters")

print(f"\nFINAL METRICS SUMMARY:")
print(f"======================")
print(f"Best Model: {best_model}")
print(f"Best Accuracy: {max(accuracy_linear, accuracy_rbf_opt):.4f}")
print(f"Best F1-Score: {max(f1_linear, f1_rbf_opt):.4f}")
print(f"Best AUC: {max(auc_linear, auc_rbf_opt):.4f}")

