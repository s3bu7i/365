
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GameOfLife:
    def __init__(self, grid_size=(100, 100), initial_live_prob=0.25):
        """Initialize game grid with random or custom initial state."""
        self.grid_size = grid_size
        rng = np.random.default_rng(seed=42)
        self.grid = rng.choice([0, 1], size=grid_size, 
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
                new_grid[x, y] = self._apply_rules(x, y)
        self.grid = new_grid

    def _apply_rules(self, x, y):
        """Apply Game of Life rules to a single cell."""
        live_neighbors = self.count_neighbors(x, y)
        if self.grid[x, y] == 1:
            return 1 if 2 <= live_neighbors <= 3 else 0
        else:
            return 1 if live_neighbors == 3 else 0
    
    def simulate(self, num_generations=200, interval=50, output_path='game_of_life.gif'):
        """Simulate Game of Life and generate animated visualization."""
        fig, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(self.grid, interpolation='nearest', cmap='binary')
        plt.title("Conway's Game of Life Simulation")
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