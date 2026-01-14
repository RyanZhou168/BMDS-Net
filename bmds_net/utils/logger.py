"""
Logger Utility Module for BMDS-Net.
Handles console output, file logging, and experiment tracking (TensorBoard/WandB).
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console logs."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logger(name: str, 
                 log_dir: Optional[str] = None, 
                 level: int = logging.INFO,
                 console_output: bool = True,
                 file_output: bool = True) -> logging.Logger:
    """
    Setup a system-wide logger.
    
    Args:
        name (str): Name of the logger.
        log_dir (str, optional): Directory to save log files.
        level (int): Logging level (default: logging.INFO).
        console_output (bool): Whether to print to stdout.
        file_output (bool): Whether to save to file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(console_handler)
    
    if file_output and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class TensorBoardLogger:
    """Simple wrapper for TensorBoard SummaryWriter."""
    def __init__(self, log_dir: str):
        self.enabled = True
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            self.enabled = False
            print("[Warning] TensorBoard not installed. Logging disabled.")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.enabled:
            self.writer.close()
