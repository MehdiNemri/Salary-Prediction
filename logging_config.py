import logging
import os
from datetime import datetime

def setup_logger(log_dir='logs'):
    """
    Configure a comprehensive logging system for the Salary Prediction project.
    
    Args:
        log_dir (str): Directory to store log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique log filename with timestamp
    log_filename = os.path.join(log_dir, f'salary_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create logger
    logger = logging.getLogger('SalaryPredictionLogger')
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate log messages
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Create logger at the module level
logger = setup_logger()

