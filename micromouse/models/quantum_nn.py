"""Quantum-inspired neural network for micromouse maze solving."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, List, Optional

class QuantumMicromouseNN:
    """Quantum-inspired neural network for micromouse maze solving."""
    
    def __init__(self, maze_size: Tuple[int, int], hidden_units: int = 64):
        """Initialize the quantum neural network.
        
        Args:
            maze_size: Size of the maze as (height, width).
            hidden_units: Number of hidden units in the network.
        """
        self.maze_size = maze_size
        self.hidden_units = hidden_units
        
        # Initialize quantum-inspired weights
        key = jax.random.PRNGKey(0)
        self.weights = {
            'h': jax.random.normal(key, (np.prod(maze_size), hidden_units)) / np.sqrt(np.prod(maze_size)),
            'o': jax.random.normal(key, (hidden_units, 4)) / np.sqrt(hidden_units)
        }
        
    def quantum_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Quantum-inspired activation function.
        
        Args:
            x: Input array.
            
        Returns:
            jnp.ndarray: Activated values.
        """
        return jnp.tanh(x) * jnp.cos(x)
        
    def get_move(self, state: np.ndarray) -> int:
        """Get the next move based on the current state.
        
        Args:
            state: Current state representation.
            
        Returns:
            int: Index of the chosen move (0: right, 1: down, 2: left, 3: up).
        """
        # Flatten state
        state_flat = state.reshape(-1)
        
        # Forward pass
        hidden = self.quantum_activation(jnp.dot(state_flat, self.weights['h']))
        output = jnp.dot(hidden, self.weights['o'])
        
        # Return move with highest activation
        return int(jnp.argmax(output))
        
    def flood_fill_heuristic(self, maze: np.ndarray) -> np.ndarray:
        """Calculate flood-fill heuristic values for the maze.
        
        Args:
            maze: The maze array where 0 represents paths and 1 represents walls.
            
        Returns:
            np.ndarray: Array of same size as maze with flood-fill distance values.
        """
        # Initialize distances
        distances = np.full(maze.shape, np.inf)
        distances[-1, -1] = 0  # Goal position
        
        # Flood fill
        changed = True
        while changed:
            changed = False
            for i in range(maze.shape[0]):
                for j in range(maze.shape[1]):
                    if maze[i, j] == 1:  # Wall
                        continue
                        
                    # Check neighbors
                    min_dist = distances[i, j]
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1]:
                            min_dist = min(min_dist, distances[ni, nj] + 1)
                            
                    if min_dist < distances[i, j]:
                        distances[i, j] = min_dist
                        changed = True
                        
        return distances 