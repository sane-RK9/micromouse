import socket
import time
from typing import Tuple, Optional

class SimulatorCommunicator:
    """
    Handles communication with the Java simulator.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 12345, retry_interval: float = 1.0):
        self.host = host
        self.port = port
        self.retry_interval = retry_interval
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        """
        Establish connection with the simulator.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        while not self.connected:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.connected = True
                print("Connected to simulator")
                return True
            except ConnectionError as e:
                print(f"Failed to connect to simulator: {e}")
                print(f"Retrying in {self.retry_interval} seconds...")
                time.sleep(self.retry_interval)
        return False

    def send_move(self, dx: int, dy: int) -> bool:
        """
        Send movement command to the Java simulator.

        Args:
            dx (int): Change in x-direction.
            dy (int): Change in y-direction.

        Returns:
            bool: True if move was sent successfully, False otherwise
        """
        if not self.connected and not self.connect():
            return False

        try:
            message = f"{dx},{dy}\n"
            self.socket.sendall(message.encode('utf-8'))
            return True
        except ConnectionError as e:
            print(f"Connection lost: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"Error sending move: {e}")
            return False

    def close(self) -> None:
        """Close the connection to the simulator."""
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                print(f"Error closing connection: {e}")
            finally:
                self.socket = None
                self.connected = False 