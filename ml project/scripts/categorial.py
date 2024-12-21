import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'data/ckd.csv'  # Replace with the correct file path
ckd_data = pd.read_csv(file_path)

# Replace missing values represented as "?" with NaN
ckd_data.replace("?", pd.NA, inplace=True)

# Convert all columns to numeric where possible
for column in ckd_data.columns:
    ckd_data[column] = pd.to_numeric(ckd_data[column], errors='coerce')

# Identify numeric columns
numeric_columns = ckd_data.select_dtypes(include=['number']).columns

# Filter out numeric columns with no observed values
valid_numeric_columns = [
    col for col in numeric_columns if ckd_data[col].notna().sum() > 0]

# Debug: Check valid numeric columns
print("Valid Numeric Columns for Imputation:")
print(valid_numeric_columns)

# Impute missing values in valid numeric columns with the median
imputer_numeric = SimpleImputer(strategy='median')
ckd_data[valid_numeric_columns] = imputer_numeric.fit_transform(
    ckd_data[valid_numeric_columns])

# Debug: Confirm missing values have been handled
print("\nProcessed Numeric Columns:")
print(ckd_data[valid_numeric_columns].head())

# Output the processed data