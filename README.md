# Quantum Micromouse

A quantum-inspired neural network implementation for micromouse maze solving, with future plans for Rust implementation.

## Project Overview
This project implements a quantum-inspired neural network to solve micromouse mazes. It combines:
- Quantum-inspired neural networks with complex-valued weights
- Classical pathfinding algorithms
- Real-time visualization through a Java simulator
- Future Rust implementation for production use

## Features
- [x] Quantum-inspired neural network architecture
- [x] Complex-valued weights and activations
- [x] Flood-fill heuristic for path planning
- [x] Real-time maze visualization
- [x] Interactive simulator controls
- [ ] Rust implementation (planned)
- [ ] Advanced quantum interface features
- [ ] Progress tracking with tqdm
- [ ] HTML logging interface

## Performance Metrics
- Path finding time
- Number of routes discovered/tested
- Success rate in different maze configurations
- Memory usage and computational efficiency

## Installation

### Prerequisites
- Python 3.11
- Java Runtime Environment (JRE)
- Required Python packages:
  - numpy
  - jax
  - tqdm
  - argparse

### Setup
1. Clone the repository:
```bash
git clone https://github.com/sane-RK9/micromouse.git
cd micromouse
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Compile and run the Java simulator:
```bash
cd simulator
javac MicromouseSimulator.java
java MicromouseSimulator
```

## Usage
[Add detailed usage instructions with examples]

## Project Structure
```
micromouse/
├── models/              # Neural network implementations
│   ├── __init__.py
│   └── quantum_nn.py    # Quantum-inspired neural network
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── maze_generator.py
│   └── simulator.py
├── simulator/          # Java simulator
│   ├── README.md
│   └── MicromouseSimulator.java
├── config.py          # Configuration parameters
└── main.py           # Main execution script
```

## Configuration
- Maze Size: 16x16 (development/testing)
- Target Size: 128x128 (production)
- Default Start Position: (0, 0)
- Default Goal Position: (15, 15) for 16x16, (127, 127) for 128x128

## Error Handling
- Connection error handling for simulator communication
- HTML-based logging interface
- Progress tracking with tqdm
- Command-line argument parsing with argparse

## Documentation
- Usage documentation
- Theoretical background of quantum-inspired neural networks
- API documentation
- Example implementations

## Future Plans
- Rust implementation for production use
- Enhanced quantum interface features
- Support for larger maze sizes (128x128)
- Performance optimization for real-world deployment



