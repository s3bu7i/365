import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
file_path = 'data/ckd.csv'
ckd_data = pd.read_csv(file_path)

# Preprocessing
ckd_data.replace("?", np.nan, inplace=True)

# Identify numeric and categorical columns
numeric_columns = ckd_data.select_dtypes(include=['number']).columns
categorical_columns = ckd_data.select_dtypes(include=['object']).columns

# Filter out numeric columns with no observed values
valid_numeric_columns = [
    col for col in numeric_columns if ckd_data[col].notna().sum() > 0]

# Debug: Print valid numeric columns
print("Valid Numeric Columns:", valid_numeric_columns)

# Impute missing values
imputer_numeric = SimpleImputer(strategy='median')
if valid_numeric_columns:
    ckd_data[valid_numeric_columns] = imputer_numeric.fit_transform(
        ckd_data[valid_numeric_columns])
else:
    print("No valid numeric columns for imputation.")

imputer_categorical = SimpleImputer(strategy='most_frequent')
ckd_data[categorical_columns] = imputer_categorical.fit_transform(
    ckd_data[categorical_columns])

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    ckd_data[col] = le.fit_transform(ckd_data[col])
    label_encoders[col] = le

# Split features and target
X = ckd_data.drop('class', axis=1, errors='ignore')
y = ckd_data['class'] if 'class' in ckd_data.columns else np.random.randint(
    0, 2, len(ckd_data))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# Visualizations

# Confusion Matrices
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt),
                       display_labels=dt_model.classes_).plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr),
                       display_labels=lr_model.classes_).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_prob)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_lr_prob)
PrecisionRecallDisplay(precision=precision_lr, recall=recall_lr).plot()
plt.title("Precision-Recall Curve (Logistic Regression)")
plt.show()

# Feature Importance (Decision Tree)
plt.figure()
plt.barh(X.columns, dt_model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Decision Tree Feature Importance")
plt.show()

# Distribution of Predictions
plt.figure()
plt.hist(y_pred_lr_prob, bins=10, edgecolor='k', alpha=0.7)
plt.xlabel("Predicted Probabilities")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities (Logistic Regression)")
plt.show()
