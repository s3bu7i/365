from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate both models
dt_accuracy = accuracy_score(y_test, y_pred_dt)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# Generate evaluation reports
dt_report = classification_report(y_test, y_pred_dt)
lr_report = classification_report(y_test, y_pred_lr)

# Output evaluation metrics
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Logistic Regression Accuracy: {lr_accuracy}\n")
print("Decision Tree Classification Report:")
print(dt_report)
print("\nLogistic Regression Classification Report:")
print(lr_report)
