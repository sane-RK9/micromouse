import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from functools import partial

class QuantumMicromouseNN:
    """
    Quantum-inspired neural network to guide a micromouse through a maze.

    Attributes:
        maze_size (tuple): Dimensions of the maze grid.
        hidden_units (int): Number of units in the hidden layer.
        weights1 (jnp.array): Complex-valued weights for the first layer.
        weights2 (jnp.array): Complex-valued weights for the output layer.
    """

    def __init__(self, maze_size=(16, 16), hidden_units=64):
        self.maze_size = maze_size
        self.hidden_units = hidden_units

        # Initialize weights with quantum-inspired complex numbers
        key = jax.random.PRNGKey(0)
        self.weights1 = self._initialize_weights(key, (maze_size[0] * maze_size[1], hidden_units))
        self.weights2 = self._initialize_weights(key, (hidden_units, 4))  # 4 possible directions

    def _initialize_weights(self, key, shape):
        """
        Initialize weights with complex values.

        Args:
            key (jax.random.PRNGKey): JAX random key for reproducibility.
            shape (tuple): Shape of the weight matrix.

        Returns:
            jnp.array: Complex-valued weights.
        """
        key1, key2 = jax.random.split(key)
        real = jax.random.normal(key1, shape) * 0.1
        imag = jax.random.normal(key2, shape) * 0.1
        return real + 1j * imag

    @partial(jit, static_argnums=(0,))
    def quantum_activation(self, x):
        """
        Quantum-inspired activation function using complex numbers.

        Args:
            x (jnp.array): Input array.

        Returns:
            jnp.array: Activated values.
        """
        return jnp.tanh(x.real) + 1j * jnp.tanh(x.imag)

    def flood_fill_heuristic(self, maze):
        """
        Calculate flooding distance heuristic to guide path planning.

        Args:
            maze (np.array): Maze layout with walls.

        Returns:
            np.array: Flood-fill distances.
        """
        distances = np.full(self.maze_size, np.inf)
        distances[self.maze_size[0] - 1, self.maze_size[1] - 1] = 0
        queue = [(self.maze_size[0] - 1, self.maze_size[1] - 1)]

        while queue:
            x, y = queue.pop(0)
            current_dist = distances[x, y]
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.maze_size[0] and 
                    0 <= new_y < self.maze_size[1] and 
                    maze[new_x, new_y] == 0):
                    move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1
                    new_dist = current_dist + move_cost
                    if new_dist < distances[new_x, new_y]:
                        distances[new_x, new_y] = new_dist
                        queue.append((new_x, new_y))

        return distances

    @partial(jit, static_argnums=(0,))
    def forward(self, state):
        """
        Forward pass through the neural network.

        Args:
            state (jnp.array): Flattened maze state.

        Returns:
            jnp.array: Probabilities of each movement direction.
        """
        hidden = self.quantum_activation(jnp.dot(state, self.weights1))
        output = jnp.dot(hidden, self.weights2)
        return jax.nn.softmax(output.real)

    def get_move(self, state):
        """
        Determine the next move based on current state.

        Args:
            state (np.array): Current maze state.

        Returns:
            int: Direction with highest probability.
        """
        state = jnp.array(state.flatten())
        probs = self.forward(state)
        return jnp.argmax(probs) 