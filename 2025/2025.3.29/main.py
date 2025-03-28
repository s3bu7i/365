import os
import random
import string
import json
import math
import datetime

class ProgramGenerator:
    def __init__(self, output_dir='generated_programs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _generate_random_name(self, length=8):
        """Generate a random program name."""
        return ''.join(random.choices(string.ascii_lowercase, k=length))

    def generate_data_analysis_program(self):
        """Generate a data analysis program with pandas and matplotlib."""
        program_name = f"{self._generate_random_name()}_data_analysis.py"
        content = f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class DataAnalyzer:
    def __init__(self, data: List[Dict[str, float]]):
        self.df = pd.DataFrame(data)
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics."""
        return {{
            'mean': self.df.mean(),
            'median': self.df.median(),
            'std_dev': self.df.std(),
            'correlation_matrix': self.df.corr()
        }}
    
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
    sample_data = [
        {{'x': np.random.normal(0, 1), 'y': np.random.exponential(2), 'z': np.random.uniform(0, 10)}} 
        for _ in range(100)
    ]
    
    analyzer = DataAnalyzer(sample_data)
    stats = analyzer.calculate_statistics()
    print(json.dumps(stats, indent=2))
    analyzer.generate_visualization()

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def generate_network_simulation(self):
        """Generate a network simulation program."""
        program_name = f"{self._generate_random_name()}_network_sim.py"
        content = f'''
import random
import networkx as nx
import matplotlib.pyplot as plt

class NetworkSimulator:
    def __init__(self, num_nodes=50, connection_probability=0.1):
        self.graph = nx.erdos_renyi_graph(num_nodes, connection_probability)
    
    def calculate_network_metrics(self):
        """Compute advanced network metrics."""
        return {{
            'density': nx.density(self.graph),
            'avg_clustering_coefficient': nx.average_clustering(self.graph),
            'connected_components': list(nx.connected_components(self.graph)),
            'centrality_measures': {{
                'degree_centrality': dict(nx.degree_centrality(self.graph)),
                'betweenness_centrality': dict(nx.betweenness_centrality(self.graph))
            }}
        }}
    
    def visualize_network(self, output_path='network_graph.png'):
        """Visualize network structure."""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=50)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
        plt.title('Network Simulation Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    simulator = NetworkSimulator()
    metrics = simulator.calculate_network_metrics()
    print(json.dumps(metrics, indent=2))
    simulator.visualize_network()

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def generate_machine_learning_script(self):
        """Generate a machine learning classification script."""
        program_name = f"{self._generate_random_name()}_ml_classifier.py"
        content = f'''
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
        return {{
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'feature_importances': dict(zip(range(self.X.shape[1]), self.classifier.feature_importances_))
        }}
    
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
'''
        self._write_program(program_name, content)

    def generate_cryptography_tool(self):
        """Generate a basic cryptography tool."""
        program_name = f"{self._generate_random_name()}_crypto_tool.py"
        content = f'''
import hashlib
import secrets
import base64
from typing import Union

class CryptoTool:
    @staticmethod
    def generate_salt(length: int = 16) -> bytes:
        """Generate cryptographically secure random salt."""
        return secrets.token_bytes(length)
    
    @staticmethod
    def hash_password(password: str, salt: bytes = None) -> dict:
        """Hash password with SHA-256 and optional salt."""
        if salt is None:
            salt = CryptoTool.generate_salt()
        
        # Combine password and salt
        salted_password = password.encode() + salt
        
        # Hash using SHA-256
        hashed_password = hashlib.sha256(salted_password).hexdigest()
        
        return {{
            'salt': base64.b64encode(salt).decode(),
            'hashed_password': hashed_password
        }}
    
    @staticmethod
    def verify_password(input_password: str, stored_salt: Union[str, bytes], 
                        stored_hash: str) -> bool:
        """Verify password against stored hash."""
        if isinstance(stored_salt, str):
            stored_salt = base64.b64decode(stored_salt)
        
        verification_result = CryptoTool.hash_password(input_password, stored_salt)
        return verification_result['hashed_password'] == stored_hash

def main():
    # Example usage
    password = "MySecurePassword123!"
    crypto_result = CryptoTool.hash_password(password)
    print("Password Hashing Result:")
    print(json.dumps(crypto_result, indent=2))
    
    # Verification test
    verification = CryptoTool.verify_password(
        password, 
        crypto_result['salt'], 
        crypto_result['hashed_password']
    )
    print(f"\nPassword Verification: {{verification}}")

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def generate_financial_simulator(self):
        """Generate a financial portfolio simulation script."""
        program_name = f"{self._generate_random_name()}_finance_sim.py"
        content = f'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

class PortfolioSimulator:
    def __init__(self, initial_capital: float = 100000, 
                 risk_free_rate: float = 0.02, 
                 simulation_years: int = 10):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.simulation_years = simulation_years
        self.assets = [
            {{'name': 'Stock', 'expected_return': 0.08, 'volatility': 0.15}},
            {{'name': 'Bond', 'expected_return': 0.03, 'volatility': 0.05}},
            {{'name': 'Real Estate', 'expected_return': 0.06, 'volatility': 0.10}}
        ]
    
    def simulate_portfolio_performance(self, num_simulations: int = 1000):
        """Simulate portfolio performance using Monte Carlo method."""
        results = []
        for _ in range(num_simulations):
            portfolio_value = self.initial_capital
            yearly_returns = []
            
            for _ in range(self.simulation_years):
                portfolio_return = sum(
                    asset['expected_return'] * np.random.normal(1, asset['volatility'])
                    for asset in self.assets
                )
                portfolio_value *= (1 + portfolio_return)
                yearly_returns.append(portfolio_return)
            
            results.append({{
                'final_value': portfolio_value,
                'total_return': (portfolio_value / self.initial_capital - 1) * 100,
                'yearly_returns': yearly_returns
            }})
        
        return results
    
    def analyze_simulation(self, simulation_results):
        """Analyze simulation results."""
        df = pd.DataFrame(simulation_results)
        return {{
            'mean_final_value': df['final_value'].mean(),
            'median_final_value': df['final_value'].median(),
            'value_at_risk_95': np.percentile(df['final_value'], 5),
            'success_probability': (df['final_value'] > self.initial_capital).mean() * 100,
            'return_distribution': {{
                'mean_return': df['total_return'].mean(),
                'median_return': df['total_return'].median(),
                'std_dev_return': df['total_return'].std()
            }}
        }}
    
    def plot_monte_carlo_simulation(self, simulation_results, output_path='monte_carlo_sim.png'):
        """Visualize Monte Carlo simulation results."""
        plt.figure(figsize=(12, 8))
        df = pd.DataFrame(simulation_results)
        
        plt.subplot(2, 1, 1)
        plt.title('Final Portfolio Value Distribution')
        df['final_value'].hist(bins=50)
        plt.xlabel('Final Portfolio Value')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 1, 2)
        plt.title('Total Return Distribution')
        df['total_return'].hist(bins=50)
        plt.xlabel('Total Return (%)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    simulator = PortfolioSimulator()
    simulation_results = simulator.simulate_portfolio_performance()
    analysis = simulator.analyze_simulation(simulation_results)
    
    print("Portfolio Simulation Analysis:")
    print(json.dumps(analysis, indent=2))
    
    simulator.plot_monte_carlo_simulation(simulation_results)

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def generate_game_of_life_simulation(self):
        """Generate Conway's Game of Life simulation."""
        program_name = f"{self._generate_random_name()}_game_of_life.py"
        content = f'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GameOfLife:
    def __init__(self, grid_size=(100, 100), initial_live_prob=0.25):
        """Initialize game grid with random or custom initial state."""
        self.grid_size = grid_size
        self.grid = np.random.choice([0, 1], size=grid_size, 
                                     p=[1-initial_live_prob, initial_live_prob])
    
    def count_neighbors(self, x, y):
        """Count live neighbors for a given cell."""
        neighborhood = self.grid[
            max(0, x-1):min(x+2, self.grid_size[0]), 
            max(0, y-1):min(y+2, self.grid_size[1])
        ]
        return np.sum(neighborhood) - self.grid[x, y]
    
    def update_grid(self):
        """Apply Game of Life rules."""
        new_grid = self.grid.copy()
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                live_neighbors = self.count_neighbors(x, y)
                
                # Classic Game of Life rules
                if self.grid[x, y] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_grid[x, y] = 0
                else:
                    if live_neighbors == 3:
                        new_grid[x, y] = 1
        
        self.grid = new_grid
    
    def simulate(self, num_generations=200, interval=50, output_path='game_of_life.gif'):
        """Simulate Game of Life and generate animated visualization."""
        fig, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(self.grid, interpolation='nearest', cmap='binary')
        plt.title('Conway\'s Game of Life Simulation')
        plt.axis('off')
        
        def update_frame(frame):
            nonlocal img
            self.update_grid()
            img.set_data(self.grid)
            return [img]
        
        anim = animation.FuncAnimation(
            fig, update_frame, frames=num_generations, 
            interval=interval, blit=True
        )
        
        anim.save(output_path, writer='pillow')
        plt.close()

def main():
    np.random.seed(42)  # For reproducibility
    game = GameOfLife(grid_size=(200, 200), initial_live_prob=0.3)
    game.simulate()

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def generate_text_processor(self):
        """Generate an advanced text processing script."""
        program_name = f"{self._generate_random_name()}_text_processor.py"
        content = f'''
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
        text = re.sub(r'[^\\w\\s]', '', text)
        
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
        
        return {{
            'total_words': total_words,
            'unique_words': unique_words,
            'lexical_diversity': unique_words / total_words if total_words > 0 else 0
        }}
    
    def generate_summary(self, num_sentences: int = 3) -> str:
        """Generate basic extractive summary."""
        # Very simplified summary extraction
        # In a real scenario, you'd use more advanced NLP techniques
        sentences = re.split(r'[.!?]', self.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Sort sentences by word frequency
        sentence_scores = {{
            sentence: sum(self.tokens.count(word) for word in word_tokenize(sentence.lower()))
            for sentence in sentences
        }}
        
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
    
    print("\nText Metrics:")
    print(json.dumps(analyzer.calculate_text_metrics(), indent=2))
    
    print("\nSummary:")
    print(analyzer.generate_summary())

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def generate_image_processor(self):
        """Generate an image processing script."""
        program_name = f"{self._generate_random_name()}_image_processor.py"
        content = f'''
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.original_image = Image.open(image_path)
    
    def apply_filters(self, output_dir='image_outputs'):
        """Apply multiple image transformations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Original Image
        self.original_image.save(os.path.join(output_dir, 'original.png'))
        
        # Grayscale
        grayscale = self.original_image.convert('L')
        grayscale.save(os.path.join(output_dir, 'grayscale.png'))
        
        # Blur
        blurred = self.original_image.filter(ImageFilter.GaussianBlur(radius=5))
        blurred.save(os.path.join(output_dir, 'blurred.png'))
        
        # Enhance Contrast
        enhancer = ImageEnhance.Contrast(self.original_image)
        high_contrast = enhancer.enhance(2.0)
        high_contrast.save(os.path.join(output_dir, 'high_contrast.png'))
        
        # Edge Detection
        edges = self.original_image.filter(ImageFilter.FIND_EDGES)
        edges.save(os.path.join(output_dir, 'edges.png'))
        
        # Color Histogram
        plt.figure(figsize=(15, 5))
        colors = ('r', 'g', 'b')
        channel_ids = (0, 1, 2)
        
        for channel_id, c in zip(channel_ids, colors):
            histogram = self.original_image.split()[channel_id].histogram()
            plt.subplot(1, 3, channel_id + 1)
            plt.title(f'{{c.upper()}} Channel Histogram')
            plt.bar(range(256), histogram, color=c, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'color_histogram.png'))
        plt.close()
    
    def image_metadata(self):
        """Extract and return image metadata."""
        return {{
            'format': self.original_image.format,
            'mode': self.original_image.mode,
            'size': self.original_image.size,
            'color_palette': str(self.original_image.getpalette()[:15]) if self.original_image.palette else 'No palette'
        }}

def main():
    # For demonstration, create a sample image if no image exists
    if not os.path.exists('sample_image.png'):
        test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        plt.imsave('sample_image.png', test_image)
    
    processor = ImageProcessor('sample_image.png')
    
    # Apply image filters
    processor.apply_filters()
    
    # Print metadata
    print(json.dumps(processor.image_metadata(), indent=2))

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def generate_astronomical_simulator(self):
        """Generate an astronomical simulation script."""
        program_name = f"{self._generate_random_name()}_astronomy_sim.py"
        content = f'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

class NBodySimulation:
    def __init__(self, num_bodies=5, time_steps=1000):
        """Initialize N-body simulation with random initial conditions."""
        self.num_bodies = num_bodies
        self.time_steps = time_steps
        
        # Gravitational constant
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        
        # Randomize initial positions, velocities, and masses
        self.positions = np.random.uniform(-1e9, 1e9, (num_bodies, 3))
        self.velocities = np.random.uniform(-1000, 1000, (num_bodies, 3))
        self.masses = np.random.uniform(1e20, 1e30, num_bodies)
    
    def gravitational_acceleration(self, state, t):
        """Compute gravitational accelerations for all bodies."""
        positions = state[:self.num_bodies*3].reshape((self.num_bodies, 3))
        velocities = state[self.num_bodies*3:].reshape((self.num_bodies, 3))
        
        accelerations = np.zeros_like(positions)
        
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j:
                    r = positions[j] - positions[i]
                    distance = np.linalg.norm(r)
                    
                    # Gravitational force calculation
                    acceleration = self.G * self.masses[j] * r / (distance**3 + 1e-10)
                    accelerations[i] += acceleration
        
        return np.concatenate([velocities.flatten(), accelerations.flatten()])
    
    def simulate(self, simulation_time=1e10):
        """Run N-body simulation."""
        initial_state = np.concatenate([
            self.positions.flatten(), 
            self.velocities.flatten()
        ])
        
        time_span = np.linspace(0, simulation_time, self.time_steps)
        solution = odeint(self.gravitational_acceleration, initial_state, time_span)
        
        # Reshape solution back to positions and velocities
        positions = solution[:, :self.num_bodies*3].reshape((self.time_steps, self.num_bodies, 3))
        velocities = solution[:, self.num_bodies*3:].reshape((self.time_steps, self.num_bodies, 3))
        
        return positions, velocities, time_span
    
    def visualize_simulation(self, positions, output_path='n_body_simulation.gif'):
        """Create animated visualization of N-body simulation."""
        fig = plt.figure(figsize=(10, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Plot settings
        ax.set_facecolor('black')
        ax.grid(False)
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_zlabel('Z (m)', color='white')
        
        # Change tick colors to white
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        scatter = ax.scatter([], [], [], c='white', s=50)
        ax.set_title('N-Body Astronomical Simulation', color='white')
        
        def update(frame):
            scatter._offsets3d = (positions[frame, :, 0], 
                                   positions[frame, :, 1], 
                                   positions[frame, :, 2])
            return scatter,
        
        anim = FuncAnimation(fig, update, frames=positions.shape[0], 
                             interval=50, blit=True)
        
        anim.save(output_path, writer='pillow')
        plt.close()

def main():
    np.random.seed(42)  # For reproducibility
    sim = NBodySimulation(num_bodies=7, time_steps=500)
    positions, velocities, time_span = sim.simulate()
    sim.visualize_simulation(positions)
    
    print(f"Simulation Details:")
    print(f"Number of Bodies: {{sim.num_bodies}}")
    print(f"Simulation Time Steps: {{sim.time_steps}}")
    print(f"Final Positions:\n{{positions[-1]}}")

if __name__ == '__main__':
    main()
'''
        self._write_program(program_name, content)

    def _write_program(self, filename, content):
        """Write program to file."""
        full_path = os.path.join(self.output_dir, filename)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"Generated: {filename}")

    def generate_all_programs(self):
        """Generate all 10 programs."""
        generation_methods = [
            self.generate_data_analysis_program,
            self.generate_network_simulation,
            self.generate_machine_learning_script,
            self.generate_cryptography_tool,
            self.generate_financial_simulator,
            self.generate_game_of_life_simulation,
            self.generate_text_processor,
            self.generate_image_processor,
            self.generate_astronomical_simulator,
            self.generate_network_simulation  # Additional network simulation
        ]

        for method in generation_methods:
            method()

def main():
    generator = ProgramGenerator()
    generator.generate_all_programs()
    print("\nAll programs have been generated in the 'generated_programs' directory.")

if __name__ == '__main__':
    main()