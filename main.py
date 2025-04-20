from models.quantum_nn import QuantumMicromouseNN
from utils.maze_generator import create_test_maze, is_valid_position
from utils.simulator import SimulatorCommunicator
from config import *

def main():
    # Create a random test maze
    maze = create_test_maze(MAZE_SIZE)
    
    # Initialize the Quantum Micromouse NN
    micromouse = QuantumMicromouseNN(maze_size=MAZE_SIZE, hidden_units=HIDDEN_UNITS)
    
    # Initialize simulator communicator
    simulator = SimulatorCommunicator(host=SIMULATOR_HOST, port=SIMULATOR_PORT)
    
    # Plan the path through the maze
    current_pos = START_POSITION
    path = [current_pos]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

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
            
            # Send move to simulator
            simulator.send_move(dx, dy)

    print("Planned Path:", path)

if __name__ == "__main__":
    main() 