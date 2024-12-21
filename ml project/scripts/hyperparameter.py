from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'data/ckd.csv'  # Update with the correct file path
ckd_data = pd.read_csv(file_path)

# Replace "?" with NaN and preprocess the data
ckd_data.replace("?", pd.NA, inplace=True)
categorical_columns = ckd_data.select_dtypes(include=['object']).columns

# Handle categorical data

ckd_data[categorical_columns] = ckd_data[categorical_columns].astype(str)
imputer_categorical = SimpleImputer(strategy='most_frequent')
ckd_data[categorical_columns] = imputer_categorical.fit_transform(
    ckd_data[categorical_columns])

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    ckd_data[col] = le.fit_transform(ckd_data[col])
    label_encoders[col] = le

# Split features and target
X = ckd_data.drop('class', axis=1)  # Features
y = ckd_data['class']              # Target

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain Logistic Regression with scaled data
lr_model_scaled = LogisticRegression(max_iter=1000, random_state=42)
lr_model_scaled.fit(X_train_scaled, y_train)
y_pred_lr_scaled = lr_model_scaled.predict(X_test_scaled)

# Re-evaluate Logistic Regression
lr_scaled_accuracy = accuracy_score(y_test, y_pred_lr_scaled)
lr_scaled_report = classification_report(y_test, y_pred_lr_scaled)

# Output the new evaluation metrics for Logistic Regression
print(f"Logistic Regression (Scaled) Accuracy: {lr_scaled_accuracy}")
print("\nLogistic Regression (Scaled) Classification Report:")
print(lr_scaled_report)
