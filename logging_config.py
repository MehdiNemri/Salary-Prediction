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

# Example usage in different project scenarios
logger = setup_logger()

def data_preprocessing_example():
    """Example of logging during data preprocessing."""
    try:
        logger.info("Starting data preprocessing")
        logger.debug("Loading raw salary dataset")
        # Simulated data loading
        dataset_size = 1000
        logger.info(f"Loaded dataset with {dataset_size} records")
        
        logger.debug("Cleaning missing values")
        missing_values_count = 50
        logger.warning(f"Removed {missing_values_count} rows with missing data")
        
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}", exc_info=True)

def model_training_example():
    """Example of logging during model training."""
    try:
        logger.info("Initiating model training")
        logger.debug("Splitting data into train and test sets")
        train_size, test_size = 800, 200
        logger.info(f"Train set: {train_size} samples, Test set: {test_size} samples")
        
        logger.debug("Starting model training")
        model_type = "Random Forest Regressor"
        logger.info(f"Training {model_type}")
        
        # Simulated hyperparameters
        hyperparameters = {
            'n_estimators': 100,
            'max_depth': 10
        }
        logger.debug(f"Model hyperparameters: {hyperparameters}")
        
        # Simulated model performance
        r2_score = 0.85
        mae = 5000
        logger.info(f"Model performance - R² Score: {r2_score}, MAE: {mae}")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)

# Demonstrate logging levels
def main():
    data_preprocessing_example()
    model_training_example()

if __name__ == "__main__":
    main()