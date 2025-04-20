import numpy as np

def create_test_maze(size=(16, 16)):
    """
    Generate a test maze with random walls.

    Args:
        size (tuple): Maze dimensions.

    Returns:
        np.array: Generated maze with walls.
    """
    maze = np.zeros(size)
    maze[np.random.rand(*size) < 0.2] = 1
    maze[0, 0] = maze[size[0] - 1, size[1] - 1] = 0
    return maze

def is_valid_position(maze, pos):
    """
    Check if a position is valid in the maze.

    Args:
        maze (np.array): Maze layout.
        pos (tuple): Position to check.

    Returns:
        bool: True if position is valid, False otherwise.
    """
    return (0 <= pos[0] < maze.shape[0] and 
            0 <= pos[1] < maze.shape[1] and 
            maze[pos] == 0) 