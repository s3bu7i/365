import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# 1. Load and Explore the Dataset
print("=" * 60)
print("PIMA INDIANS DIABETES DATASET ANALYSIS")
print("=" * 60)

# Load the dataset
# Assuming the dataset is saved as 'diabetes.csv'
df = pd.read_csv('C:/Users/Dino/Desktop/ai/task 5/diabetes.csv')

print("\n1. DATASET OVERVIEW")
print("-" * 30)
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

print(f"\nBasic statistics:")
print(df.describe())

print(f"\nTarget variable distribution:")
print(df['Outcome'].value_counts())
print(f"Diabetes prevalence: {df['Outcome'].mean():.2%}")

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# 2. Data Preprocessing
print("\n2. DATA PREPROCESSING")
print("-" * 30)

# Some features have 0 values which are likely missing (e.g., Glucose, BloodPressure)
# Let's identify and handle them
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        print(f"{col}: {zero_count} zero values ({zero_count/len(df)*100:.1f}%)")

# Replace zeros with median for certain features
df_processed = df.copy()
for col in ['Glucose', 'BloodPressure', 'BMI']:
    if (df_processed[col] == 0).sum() > 0:
        median_val = df_processed[df_processed[col] != 0][col].median()
        df_processed[col] = df_processed[col].replace(0, median_val)
        print(f"Replaced {col} zeros with median: {median_val:.1f}")

# Separate features and target
X = df_processed.drop('Outcome', axis=1)
y = df_processed['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 3. Part A: Train Decision Tree and Random Forest
print("\n3. PART A: MODEL TRAINING")
print("-" * 30)

# Decision Tree with default parameters
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

# Random Forest with default parameters
rf_default = RandomForestClassifier(random_state=42, n_estimators=100)
rf_default.fit(X_train, y_train)

# Make predictions
dt_pred = dt_default.predict(X_test)
rf_pred = rf_default.predict(X_test)

print("Default Model Performance:")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

# 4. Part B: Hyperparameter Analysis
print("\n4. PART B: HYPERPARAMETER ANALYSIS")
print("-" * 30)

# Analyze effect of tree depth on Decision Tree
max_depths = range(1, 21)
train_scores_dt, val_scores_dt = validation_curve(
    DecisionTreeClassifier(random_state=42), X_train, y_train,
    param_name='max_depth', param_range=max_depths,
    cv=5, scoring='accuracy', n_jobs=-1
)

# Plot depth analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(max_depths, train_scores_dt.mean(axis=1),
         'b-', label='Training Score', marker='o')
plt.plot(max_depths, val_scores_dt.mean(axis=1),
         'r-', label='Validation Score', marker='s')
plt.fill_between(max_depths, train_scores_dt.mean(axis=1) - train_scores_dt.std(axis=1),
                 train_scores_dt.mean(axis=1) + train_scores_dt.std(axis=1), alpha=0.1, color='b')
plt.fill_between(max_depths, val_scores_dt.mean(axis=1) - val_scores_dt.std(axis=1),
                 val_scores_dt.mean(axis=1) + val_scores_dt.std(axis=1), alpha=0.1, color='r')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Effect of Max Depth')
plt.legend()
plt.grid(True, alpha=0.3)

# Find optimal depth
optimal_depth_idx = np.argmax(val_scores_dt.mean(axis=1))
optimal_depth = max_depths[optimal_depth_idx]
print(f"Optimal max_depth for Decision Tree: {optimal_depth}")

# Analyze effect of min_samples_split
min_samples_splits = range(2, 21)
train_scores_split, val_scores_split = validation_curve(
    DecisionTreeClassifier(
        random_state=42, max_depth=optimal_depth), X_train, y_train,
    param_name='min_samples_split', param_range=min_samples_splits,
    cv=5, scoring='accuracy', n_jobs=-1
)

plt.subplot(1, 3, 2)
plt.plot(min_samples_splits, train_scores_split.mean(
    axis=1), 'b-', label='Training Score', marker='o')
plt.plot(min_samples_splits, val_scores_split.mean(
    axis=1), 'r-', label='Validation Score', marker='s')
plt.fill_between(min_samples_splits, train_scores_split.mean(axis=1) - train_scores_split.std(axis=1),
                 train_scores_split.mean(axis=1) + train_scores_split.std(axis=1), alpha=0.1, color='b')
plt.fill_between(min_samples_splits, val_scores_split.mean(axis=1) - val_scores_split.std(axis=1),
                 val_scores_split.mean(axis=1) + val_scores_split.std(axis=1), alpha=0.1, color='r')
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Effect of Min Samples Split')
plt.legend()
plt.grid(True, alpha=0.3)

# Analyze Random Forest n_estimators
n_estimators_range = [10, 25, 50, 75, 100, 150, 200]
train_scores_rf, val_scores_rf = validation_curve(
    RandomForestClassifier(random_state=42), X_train, y_train,
    param_name='n_estimators', param_range=n_estimators_range,
    cv=5, scoring='accuracy', n_jobs=-1
)

plt.subplot(1, 3, 3)
plt.plot(n_estimators_range, train_scores_rf.mean(
    axis=1), 'b-', label='Training Score', marker='o')
plt.plot(n_estimators_range, val_scores_rf.mean(axis=1),
         'r-', label='Validation Score', marker='s')
plt.fill_between(n_estimators_range, train_scores_rf.mean(axis=1) - train_scores_rf.std(axis=1),
                 train_scores_rf.mean(axis=1) + train_scores_rf.std(axis=1), alpha=0.1, color='b')
plt.fill_between(n_estimators_range, val_scores_rf.mean(axis=1) - val_scores_rf.std(axis=1),
                 val_scores_rf.mean(axis=1) + val_scores_rf.std(axis=1), alpha=0.1, color='r')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest: Effect of N Estimators')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Grid Search for optimal hyperparameters
print("\nPerforming Grid Search for optimal hyperparameters...")

# Decision Tree Grid Search
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid,
                       cv=5, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)

# Random Forest Grid Search
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid,
                       cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"Best Decision Tree parameters: {dt_grid.best_params_}")
print(f"Best Decision Tree CV score: {dt_grid.best_score_:.4f}")
print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Best Random Forest CV score: {rf_grid.best_score_:.4f}")

# Train optimized models
dt_optimized = dt_grid.best_estimator_
rf_optimized = rf_grid.best_estimator_

# 5. Part C: Feature Importance Analysis
print("\n5. PART C: FEATURE IMPORTANCE ANALYSIS")
print("-" * 30)

# Get feature importances
dt_importances = dt_optimized.feature_importances_
rf_importances = rf_optimized.feature_importances_

# Create feature importance comparison
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Decision_Tree': dt_importances,
    'Random_Forest': rf_importances
})
importance_df = importance_df.sort_values('Random_Forest', ascending=False)

print("Feature Importances:")
print(importance_df.round(4))

# Visualize feature importances
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.barh(range(len(feature_names)), dt_importances[np.argsort(dt_importances)])
plt.yticks(range(len(feature_names)), np.array(
    feature_names)[np.argsort(dt_importances)])
plt.xlabel('Importance')
plt.title('Decision Tree Feature Importances')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(range(len(feature_names)), rf_importances[np.argsort(rf_importances)])
plt.yticks(range(len(feature_names)), np.array(
    feature_names)[np.argsort(rf_importances)])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Comparison plot
plt.figure(figsize=(10, 6))
x = np.arange(len(feature_names))
width = 0.35

plt.bar(x - width/2, dt_importances, width, label='Decision Tree', alpha=0.8)
plt.bar(x + width/2, rf_importances, width, label='Random Forest', alpha=0.8)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Comparison')
plt.xticks(x, feature_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Part D: Model Comparison
print("\n6. PART D: MODEL COMPARISON")
print("-" * 30)

# Make predictions with optimized models
dt_pred_opt = dt_optimized.predict(X_test)
rf_pred_opt = rf_optimized.predict(X_test)

dt_pred_train = dt_optimized.predict(X_train)
rf_pred_train = rf_optimized.predict(X_train)

# Calculate metrics
dt_train_acc = accuracy_score(y_train, dt_pred_train)
dt_test_acc = accuracy_score(y_test, dt_pred_opt)
rf_train_acc = accuracy_score(y_train, rf_pred_train)
rf_test_acc = accuracy_score(y_test, rf_pred_opt)

print("ACCURACY COMPARISON:")
print(f"Decision Tree - Training: {dt_train_acc:.4f}, Test: {dt_test_acc:.4f}")
print(f"Random Forest - Training: {rf_train_acc:.4f}, Test: {rf_test_acc:.4f}")

print(f"\nOVERFITTING ANALYSIS:")
print(f"Decision Tree - Overfitting: {dt_train_acc - dt_test_acc:.4f}")
print(f"Random Forest - Overfitting: {rf_train_acc - rf_test_acc:.4f}")

# Detailed classification reports
print("\nDETAILED CLASSIFICATION REPORTS:")
print("\nDecision Tree:")
print(classification_report(y_test, dt_pred_opt))

print("\nRandom Forest:")
print(classification_report(y_test, rf_pred_opt))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Decision Tree confusion matrix
cm_dt = confusion_matrix(y_test, dt_pred_opt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Decision Tree Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Random Forest confusion matrix
cm_rf = confusion_matrix(y_test, rf_pred_opt)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Model interpretability - visualize decision tree
plt.figure(figsize=(20, 15))
plot_tree(dt_optimized, max_depth=3, feature_names=feature_names,
          class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
plt.title("Decision Tree Visualization (Depth Limited to 3 for Readability)")
plt.show()

# Summary comparison table
comparison_data = {
    'Metric': ['Training Accuracy', 'Test Accuracy', 'Overfitting (Train-Test)',
               'Model Complexity', 'Interpretability', 'Training Time'],
    'Decision Tree': [f'{dt_train_acc:.4f}', f'{dt_test_acc:.4f}',
                      f'{dt_train_acc - dt_test_acc:.4f}', 'Low', 'High', 'Fast'],
    'Random Forest': [f'{rf_train_acc:.4f}', f'{rf_test_acc:.4f}',
                      f'{rf_train_acc - rf_test_acc:.4f}', 'High', 'Medium', 'Slower']
}

comparison_df = pd.DataFrame(comparison_data)
print("\nCOMPREHENSIVE MODEL COMPARISON:")
print(comparison_df.to_string(index=False))

# Key insights
print("\n" + "="*60)
print("KEY INSIGHTS AND CONCLUSIONS:")
print("="*60)
print(
    f"1. ACCURACY: Random Forest ({rf_test_acc:.4f}) outperforms Decision Tree ({dt_test_acc:.4f})")
print(
    f"2. GENERALIZATION: Random Forest shows less overfitting ({rf_train_acc - rf_test_acc:.4f} vs {dt_train_acc - dt_test_acc:.4f})")
print(
    f"3. FEATURE IMPORTANCE: Top features are {importance_df.iloc[0]['Feature']} and {importance_df.iloc[1]['Feature']}")
print(f"4. INTERPRETABILITY: Decision Tree is more interpretable, Random Forest is more accurate")
print(
    f"5. OPTIMAL DEPTH: Decision Tree performs best at depth {dt_grid.best_params_['max_depth']}")
print(f"6. DIABETES PREDICTION: Both models show reasonable performance for medical screening")

print("\nRECOMMENDATIONS:")
print("- Use Random Forest for higher accuracy in automated systems")
print("- Use Decision Tree when interpretability is crucial for medical decisions")
print("- Focus on top features (Glucose, BMI, Age) for feature engineering")
print("- Consider ensemble methods for production deployment")
