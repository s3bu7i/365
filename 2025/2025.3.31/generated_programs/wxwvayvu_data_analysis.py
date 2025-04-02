
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class DataAnalyzer:
    def __init__(self, data: List[Dict[str, float]]):
        self.df = pd.DataFrame(data)
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics."""
        return {
            'mean': self.df.mean(),
            'median': self.df.median(),
            'std_dev': self.df.std(),
            'correlation_matrix': self.df.corr()
        }
    
    def generate_visualization(self, output_path='analysis_plot.png'):
        """Create multi-dimensional visualization."""
        plt.figure(figsize=(12, 8))
        for column in self.df.columns:
            plt.plot(self.df.index, self.df[column], label=column)
        plt.title('Multi-Dimensional Data Visualization')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    rng = np.random.default_rng(seed=42)
    sample_data = [
        {'x': rng.normal(0, 1), 'y': rng.exponential(2), 'z': rng.uniform(0, 10)} 
        for _ in range(100)
    ]
    
    analyzer = DataAnalyzer(sample_data)
    stats = analyzer.calculate_statistics()
    print(json.dumps(stats, indent=2))
    analyzer.generate_visualization()

if __name__ == '__main__':
    main()
