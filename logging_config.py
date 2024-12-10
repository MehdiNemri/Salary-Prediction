import logging
import os
from datetime import datetime

def setup_logger(log_dir='logs', logger_name='SalaryPredictionLogger', log_level=logging.DEBUG):
    """
    Configure a comprehensive logging system for the project.
    
    Args:
        log_dir (str): Directory to store log files.
        logger_name (str): Name of the logger.
        log_level (int): Logging level for the file handler (default: DEBUG).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique log filename with timestamp
    log_filename = os.path.join(log_dir, f'{logger_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create or retrieve the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if the logger is already configured
    if not logger.handlers:
        # Console handler: Shows INFO level logs and higher
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler: Logs everything starting at DEBUG level
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create logger at the module level
logger = setup_logger()
