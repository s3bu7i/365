
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
        rng = np.random.default_rng(seed=42)
        self.positions = rng.uniform(-1e9, 1e9, (num_bodies, 3))
        self.velocities = rng.uniform(-1000, 1000, (num_bodies, 3))
        self.masses = rng.uniform(1e20, 1e30, num_bodies)
    
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
    positions, _, _ = sim.simulate()
    sim.visualize_simulation(positions)
    return sim, positions
    
if __name__ == '__main__':
    sim, positions = main()
    print("Simulation Details:")
    print(f"Number of Bodies: {sim.num_bodies}")
    print(f"Simulation Time Steps: {sim.time_steps}")
    print(f"Final Positions:\n{positions[-1]}")
