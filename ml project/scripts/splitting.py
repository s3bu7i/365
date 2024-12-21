from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the uploaded CKD dataset
file_path = 'data/ckd.csv'  # Update the path to your file
ckd_data = pd.read_csv(file_path)

# Replace "?" with NaN for handling missing values
ckd_data.replace("?", pd.NA, inplace=True)

# Identify categorical columns explicitly
categorical_columns = ckd_data.select_dtypes(include=['object']).columns

# Convert categorical columns to string type explicitly
ckd_data[categorical_columns] = ckd_data[categorical_columns].astype(str)

# Impute categorical columns with the most frequent value (mode)
imputer_categorical = SimpleImputer(strategy='most_frequent')
ckd_data[categorical_columns] = imputer_categorical.fit_transform(
    ckd_data[categorical_columns])

# Encode categorical variables using label encoding
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    ckd_data[col] = le.fit_transform(ckd_data[col])
    label_encoders[col] = le

# Verify the processed dataset
print("\nProcessed Dataset Info:")
ckd_data.info()
print("\nSample Data:")
print(ckd_data.head())


# Split the dataset into features (X) and target (y)
X = ckd_data.drop('class', axis=1)  # Features
y = ckd_data['class']              # Target

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape
