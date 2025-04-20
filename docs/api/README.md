# Quantum Micromouse API Documentation

This directory contains the API documentation for the Quantum Micromouse project.

## Structure

- `models/`: Documentation for neural network models
  - `quantum_nn.md`: Quantum neural network implementation
  - `maze_router.md`: Pathfinding algorithms
- `utils/`: Utility functions and tools
  - `maze_generator.md`: Maze generation utilities
  - `simulator.md`: Java simulator interface
- `config.md`: Configuration options and parameters
- `main.md`: Main execution script documentation

## Quick Reference

### Core Classes
- `QuantumMicromouseNN`: Main neural network implementation
- `MazeRouter`: Pathfinding and navigation
- `SimulatorCommunicator`: Java simulator interface

### Key Functions
- `create_test_maze()`: Generate test mazes
- `flood_fill_heuristic()`: Path planning algorithm
- `quantum_activation()`: Quantum-inspired activation function

## Usage Examples

### Basic Usage
```python
from models.quantum_nn import QuantumMicromouseNN
from utils.maze_generator import create_test_maze

# Initialize network
nn = QuantumMicromouseNN(maze_size=(16, 16))

# Generate maze
maze = create_test_maze((16, 16))

# Get next move
move = nn.get_move(maze)
```

### Advanced Configuration
```python
from config import MAZE_CONFIG

# Custom configuration
MAZE_CONFIG.update({
    "size": (32, 32),
    "quantum_layers": [128j, 256j, 128j]
})
```

