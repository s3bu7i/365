
import json
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
            {'name': 'Stock', 'expected_return': 0.08, 'volatility': 0.15},
            {'name': 'Bond', 'expected_return': 0.03, 'volatility': 0.05},
            {'name': 'Real Estate', 'expected_return': 0.06, 'volatility': 0.10}
        ]
    
    def simulate_portfolio_performance(self, num_simulations: int = 1000):
        """Simulate portfolio performance using Monte Carlo method."""
        results = []
        rng = np.random.default_rng(seed=42)
        for _ in range(num_simulations):
            portfolio_value = self.initial_capital
            yearly_returns = []
            
            for _ in range(self.simulation_years):
                portfolio_return = sum(
                    asset['expected_return'] * rng.normal(1, asset['volatility'])
                    for asset in self.assets
                )
                portfolio_value *= (1 + portfolio_return)
                yearly_returns.append(portfolio_return)
            
            results.append({
                'final_value': portfolio_value,
                'total_return': (portfolio_value / self.initial_capital - 1) * 100,
                'yearly_returns': yearly_returns
            })
        
        return results
    
    def analyze_simulation(self, simulation_results):
        """Analyze simulation results."""
        df = pd.DataFrame(simulation_results)
        return {
            'mean_final_value': df['final_value'].mean(),
            'median_final_value': df['final_value'].median(),
            'value_at_risk_95': np.percentile(df['final_value'], 5),
            'success_probability': (df['final_value'] > self.initial_capital).mean() * 100,
            'return_distribution': {
                'mean_return': df['total_return'].mean(),
                'median_return': df['total_return'].median(),
                'std_dev_return': df['total_return'].std()
            }
        }
    
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
