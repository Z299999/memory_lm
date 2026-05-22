"""
Console and file logging module.
"""

from pathlib import Path
from typing import Optional


class ConsoleLogger:
    """
    Logger that writes to both terminal and file.
    """

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize the console logger.

        Args:
            log_path: Path to log file (None for terminal-only)
        """
        self.log_path = log_path
        self._file = None

        if log_path:
            # Ensure parent directory exists
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(log_path, "w")

    def log(self, message: str):
        """
        Log a message to terminal and file.

        Args:
            message: Message to log
        """
        print(message)
        if self._file:
            self._file.write(message + "\n")
            self._file.flush()

    def close(self):
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
