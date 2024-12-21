import pandas as pd

# Load the uploaded CKD dataset
file_path = 'data/ckd.csv'
ckd_data = pd.read_csv(file_path)

# Display the first few rows and basic info about the dataset
ckd_data.head(), 
ckd_data.info()





