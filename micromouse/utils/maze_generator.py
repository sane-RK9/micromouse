"""Maze generation utilities for the Quantum Micromouse project."""

import numpy as np
from typing import Tuple, List

def create_test_maze(size: Tuple[int, int]) -> np.ndarray:
    """Create a random test maze of the given size.
    
    Args:
        size: Tuple of (height, width) for the maze.
        
    Returns:
        np.ndarray: A binary maze where 0 represents paths and 1 represents walls.
    """
    maze = np.zeros(size, dtype=np.int8)
    
    # Add some random walls (30% density)
    maze[np.random.random(size) < 0.3] = 1
    
    # Ensure start and goal are accessible
    maze[0, 0] = 0  # Start
    maze[-1, -1] = 0  # Goal
    
    return maze

def is_valid_position(maze: np.ndarray, pos: Tuple[int, int]) -> bool:
    """Check if a position is valid in the maze.
    
    Args:
        maze: The maze array.
        pos: The position to check as (row, col).
        
    Returns:
        bool: True if the position is valid and not a wall.
    """
    row, col = pos
    if 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]:
        return maze[row, col] == 0
    return False 