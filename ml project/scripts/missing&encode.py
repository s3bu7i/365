import pandas as pd

# Load the uploaded CKD dataset
file_path = 'data/ckd.csv'
ckd_data = pd.read_csv(file_path)

# Replace missing values represented as "?" with NaN
ckd_data.replace("?", pd.NA, inplace=True)

# Convert columns to appropriate data types where possible
for column in ckd_data.columns:
    # Attempt to convert to numeric where applicable
    try:
        ckd_data[column] = pd.to_numeric(ckd_data[column])
    except ValueError:
        pass  # Skip non-numeric columns

# Check the dataset again for updated info and missing values
ckd_data.info(), ckd_data.isnull().sum()
