
import re
from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextAnalyzer:
    def __init__(self, text: str):
        self.text = text
        self.tokens = self._tokenize()
    
    def _tokenize(self) -> List[str]:
        """Tokenize and preprocess text."""
        # Convert to lowercase
        text = self.text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
    
    def get_word_frequency(self, top_n: int = 10) -> Dict[str, int]:
        """Calculate word frequencies."""
        word_counts = Counter(self.tokens)
        return dict(word_counts.most_common(top_n))
    
    def calculate_text_metrics(self) -> Dict[str, float]:
        """Calculate various text metrics."""
        total_words = len(self.tokens)
        unique_words = len(set(self.tokens))
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'lexical_diversity': unique_words / total_words if total_words > 0 else 0
        }
    
    def generate_summary(self, num_sentences: int = 3) -> str:
        """Generate basic extractive summary."""
        # Very simplified summary extraction
        # In a real scenario, you'd use more advanced NLP techniques
        sentences = re.split(r'[.!?]', self.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Sort sentences by word frequency
        sentence_scores = {
            sentence: sum(self.tokens.count(word) for word in word_tokenize(sentence.lower()))
            for sentence in sentences
        }
        
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        return '. '.join(top_sentences) + '.'

def main():
    sample_text = """
    Natural language processing is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language. The goal is to enable computers 
    to understand, interpret, and manipulate human language in valuable ways. The field of NLP draws from 
    many disciplines, including computer science and computational linguistics.
    """
    
    analyzer = TextAnalyzer(sample_text)
    
    print("Word Frequencies:")
    print(json.dumps(analyzer.get_word_frequency(), indent=2))
    
    print("
Text Metrics:")
    print(json.dumps(analyzer.calculate_text_metrics(), indent=2))
    
    print("
Summary:")
    print(analyzer.generate_summary())

if __name__ == '__main__':
    main()
