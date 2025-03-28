
import json
import random
import networkx as nx
import matplotlib.pyplot as plt

class NetworkSimulator:
    def __init__(self, num_nodes=50, connection_probability=0.1):
        self.graph = nx.erdos_renyi_graph(num_nodes, connection_probability)
    
    def calculate_network_metrics(self):
        """Compute advanced network metrics."""
        return {
            'density': nx.density(self.graph),
            'avg_clustering_coefficient': nx.average_clustering(self.graph),
            'connected_components': list(nx.connected_components(self.graph)),
            'centrality_measures': {
                'degree_centrality': dict(nx.degree_centrality(self.graph)),
                'betweenness_centrality': dict(nx.betweenness_centrality(self.graph))
            }
        }
    
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
