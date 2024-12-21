
# ML Project

## Overview
This project provides a machine learning pipeline to analyze and predict chronic kidney disease (CKD) using a dataset. The project includes data preprocessing, model training, evaluation, and hyperparameter tuning.

---

## Prerequisites
Before using this program, ensure you have the following installed:

1. **Python 3.7 or later**
2. **Pip** (Python package installer)
3. Virtual environment setup (optional but recommended)

Install the required Python libraries listed in `requirements.txt` by running:
```bash
pip install -r requirements.txt
```

---

### 2. Activate Virtual Environment (Optional)
It is recommended to use a virtual environment to manage dependencies.

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Prepare the Dataset
Ensure the dataset file `ckd.csv` is placed in the `data/` directory. If the directory does not exist, create it and add the dataset.

### 4. Run Preprocessing
Execute the preprocessing script to clean and prepare the data:
```bash
python scripts/preprocess.py
```

### 5. Train and Evaluate Models
Use the model script to train the machine learning models and evaluate their performance:
```bash
python scripts/model.py
```

### 6. Tune Hyperparameters
For better performance, run the hyperparameter tuning script:
```bash
python scripts/hyperparameter.py
```

---

## Results
After executing the above steps, you will:
1. Generate cleaned and preprocessed datasets.
2. Train Logistic Regression and Decision Tree models.
3. Evaluate model performance with accuracy scores and classification reports.

---

## Project Structure
```plaintext
ml-project/
|— data/                 # Directory for input dataset
|— scripts/              # Directory for Python scripts
|   |— preprocess.py       # Data cleaning and preprocessing
|   |— categorial.py        # Handling categorical data
|   |— model.py            # Model training and evaluation
|   |— hyperparameter.py   # Hyperparameter tuning
|— requirements.txt     # Required Python libraries
|— README.md            # Instructions for using the program
```

---

## Dependencies
The program relies on the following Python libraries:
- pandas
- numpy
- scikit-learn
- joblib

---