import numpy as np
from micromouse.models.quantum_nn import QuantumMicromouseNN
from micromouse.utils.maze_generator import create_test_maze, is_valid_position
from micromouse.utils.simulator import SimulatorCommunicator
from micromouse.config import *

def main():
    print("Starting Quantum Micromouse simulation...")
    
    # Create a random test maze
    print("Generating test maze...")
    maze = create_test_maze(MAZE_SIZE)
    print(f"Maze size: {MAZE_SIZE}")
    
    # Initialize the Quantum Micromouse NN
    print("Initializing Quantum Neural Network...")
    micromouse = QuantumMicromouseNN(maze_size=MAZE_SIZE, hidden_units=HIDDEN_UNITS)
    
    # Initialize simulator communicator
    print(f"Connecting to simulator at {SIMULATOR_HOST}:{SIMULATOR_PORT}...")
    simulator = SimulatorCommunicator(host=SIMULATOR_HOST, port=SIMULATOR_PORT)
    
    try:
        simulator.connect()
        simulator_connected = True
        print("Successfully connected to simulator.")
    except Exception as e:
        print(f"Warning: Could not connect to simulator ({e}). Running in simulation-only mode.")
        simulator_connected = False
    
    # Plan the path through the maze
    print("\nStarting maze navigation...")
    current_pos = START_POSITION
    path = [current_pos]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    while current_pos != GOAL_POSITION:
        # Create state representation
        state = np.zeros(MAZE_SIZE)
        state[current_pos] = 1
        state += 0.5 * (1 / (micromouse.flood_fill_heuristic(maze) + 1))

        # Get next move
        move = micromouse.get_move(state)
        dx, dy = moves[move]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)

        if is_valid_position(maze, new_pos):
            current_pos = new_pos
            path.append(current_pos)
            
            # Send move to simulator if connected
            if simulator_connected:
                try:
                    simulator.send_move(dx, dy)
                except Exception as e:
                    print(f"Warning: Failed to send move to simulator ({e})")
                    simulator_connected = False

    print("\nNavigation complete!")
    print(f"Path length: {len(path)}")
    print("Path:", path)
    
    # Disconnect from simulator
    if simulator_connected:
        simulator.disconnect()

if __name__ == "__main__":
    main() 