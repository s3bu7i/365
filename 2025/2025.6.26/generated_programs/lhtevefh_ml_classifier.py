
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLClassificationPipeline:
    def __init__(self, n_samples=1000, n_features=20, n_classes=3):
        self.X, self.y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_classes=n_classes, 
            n_informative=15, 
            random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train_model(self):
        """Train Random Forest Classifier."""
        self.classifier.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """Evaluate model performance."""
        y_pred = self.classifier.predict(self.X_test)
        return {
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'feature_importances': dict(zip(range(self.X.shape[1]), self.classifier.feature_importances_))
        }
    
    def plot_confusion_matrix(self, output_path='confusion_matrix.png'):
        """Plot confusion matrix visualization."""
        y_pred = self.classifier.predict(self.X_test)
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    ml_pipeline = MLClassificationPipeline()
    ml_pipeline.train_model()
    results = ml_pipeline.evaluate_model()
    print(json.dumps(results, indent=2))
    ml_pipeline.plot_confusion_matrix()

if __name__ == '__main__':
    main()
