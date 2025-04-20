"""Simulator communication utilities for the Quantum Micromouse project."""

import socket
from typing import Tuple

class SimulatorCommunicator:
    """Class to handle communication with the micromouse simulator."""
    
    def __init__(self, host: str = 'localhost', port: int = 12345):
        """Initialize the simulator communicator.
        
        Args:
            host: The hostname of the simulator.
            port: The port number of the simulator.
        """
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self):
        """Connect to the simulator."""
        if not self.socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            
    def disconnect(self):
        """Disconnect from the simulator."""
        if self.socket:
            self.socket.close()
            self.socket = None
            
    def send_move(self, dx: int, dy: int):
        """Send a move command to the simulator.
        
        Args:
            dx: Change in x position.
            dy: Change in y position.
        """
        try:
            if not self.socket:
                self.connect()
                
            # Format: "MOVE x,y"
            command = f"MOVE {dx},{dy}\n"
            self.socket.send(command.encode())
            
            # Wait for acknowledgment
            response = self.socket.recv(1024).decode().strip()
            if response != "OK":
                print(f"Warning: Unexpected response from simulator: {response}")
                
        except Exception as e:
            print(f"Error communicating with simulator: {e}")
            self.disconnect() 