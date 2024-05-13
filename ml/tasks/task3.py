import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK resources
nltk.download('stopwords')

# Update with the path to your dataset
df = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='latin1')

# Display the first few rows of the dataset
print(df.head())


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower() if isinstance(text, str) else ""

    # Remove links to external pages and media
    text = re.sub(r'http\S+', '', text)  # Removes http links
    text = re.sub(r'www\S+', '', text)   # Removes www links

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove hashtags
    text = re.sub(r'#\S+', '', text)

    # Remove emojis (optional)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text


# Check if the column 'text' exists in the DataFrame
if 'text' in df.columns:
    df['clean_text'] = df['text'].apply(preprocess_text)
else:
    print("Error: 'text' column not found in the DataFrame.")

# Assuming 'emotion_dict' is a dictionary mapping words to emotions
emotion_dict = {'happy': 'positive', 'sad': 'negative',
                'excited': 'positive', 'angry': 'negative'}

# Map each word in clean_text to its associated emotion


def map_emotions(text):
    words = text.split()
    emotions = [emotion_dict[word]
                if word in emotion_dict else 'neutral' for word in words]
    return ' '.join(emotions)


if 'clean_text' in df.columns:
    df['emotions'] = df['clean_text'].apply(map_emotions)
else:
    print("Error: 'clean_text' column not found in the DataFrame.")

# Feature Engineering
if 'emotions' in df.columns:
    # Limiting to top 1000 features
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['emotions'])
    y = df['sentiment']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Model Training - Support Vector Machine (SVM)
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
else:
    print("Error: 'emotions' column not found in the DataFrame.")
