# Stock Prediction Utilities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from typing import List, Dict, Union, Tuple, Optional, Any

# Configure logging
def setup_logging(target_feature: str) -> logging.Logger:
    """
    Set up logging configuration with log files in the execution subfolder.
    
    Args:
        target_feature: The target feature being predicted (used for subfolder naming)
    
    Returns:
        Logger object configured for the current session with output_dirs attribute
    """
    # Get output directories (which creates the directory structure)
    output_dirs = create_output_directories(target_feature)
    
    # Configure logger to use the logs directory in the execution folder
    log_dir = output_dirs['logs']
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    log_file = os.path.join(log_dir, "execution.log")
    
    # Create logger
    logger = logging.getLogger(f"stock_prediction_{target_feature}")
    logger.setLevel(logging.INFO)
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for {target_feature} prediction")
    
    # Store output directories in the logger object for easy access
    logger.output_dirs = output_dirs
    
    return logger

def clear_zero_values(data: pd.DataFrame, features: List[str], logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Remove rows with zero values in specified features.
    
    Args:
        data: Input DataFrame
        features: List of feature columns to check for zeros
        logger: Optional logger for logging information
    
    Returns:
        DataFrame with zero values removed
    """
    if logger:
        logger.info(f"Original data shape: {data.shape}")
        logger.info(f"Zero values per column: {(data==0).sum().to_dict()}")
    
    # Drop rows with zero values
    for feat in data.columns:
        data.drop(data[data[feat] == 0].index, inplace=True)
    
    if logger:
        logger.info(f"Data shape after removing zeros: {data.shape}")
        logger.info(f"Remaining zero values: {(data==0).sum().to_dict()}")
    
    # Filter to only include specified features
    data = data[features]
    
    return data

def normalize_data(data: Union[pd.DataFrame, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Normalize data to range [0, 1].
    
    Args:
        data: Input data (DataFrame or numpy array)
    
    Returns:
        Tuple of (normalized_data, min_values, max_values)
    """
    if isinstance(data, pd.DataFrame):
        data_min = data.min()
        data_max = data.max()
        normalized_data = (data - data_min) / (data_max - data_min)
    else:  # For numpy arrays
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        normalized_data = (data - data_min) / (data_max - data_min)
    
    return normalized_data, data_min, data_max

def prepare_sequence_data(input_data: pd.DataFrame, target_data: pd.DataFrame, 
                         shift: int, logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequence data for RNN training/prediction.
    
    Args:
        input_data: Input features DataFrame
        target_data: Target features DataFrame
        shift: Number of time steps to use for sequence
        logger: Optional logger for logging information
    
    Returns:
        Tuple of (X_sequence, y_sequence) as numpy arrays
    """
    X, y = [], []
    
    if shift > 0:
        for i in range(len(input_data) - shift*2):
            X.append(input_data.iloc[i:i+shift].values)  # Past N days
            y.append(target_data.iloc[i+shift:i+shift*2].values)  # Next N days
        
        X_sequence = np.array(X)
        y_sequence = np.array(y)
        y_sequence = y_sequence.reshape((y_sequence.shape[0], y_sequence.shape[1]))  # (samples, time_steps)
    else:
        X_sequence = input_data.values
        y_sequence = target_data.values
        X_sequence = X_sequence.reshape((X_sequence.shape[0], 1, X_sequence.shape[1]))
    
    if logger:
        logger.info(f"Sequence data shapes - X: {X_sequence.shape}, y: {y_sequence.shape}")
    
    return X_sequence, y_sequence

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float, shuffle: bool = True, 
                    random_state: int = 42, logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Args:
        X: Input features array
        y: Target array
        test_size: Proportion of data to use for testing
        shuffle: Whether to shuffle the data before splitting
        random_state: Random seed for reproducibility
        logger: Optional logger for logging information
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    if test_size > 0.0:
        # Extract test set from end of sequence for time series data
        X_test = X[int(-len(X)*test_size)+1:]
        y_test = y[int(-len(y)*test_size)+1:]
        
        X_train = X[:int(len(X)*(1-test_size))]
        y_train = y[:int(len(y)*(1-test_size))]
        
        if shuffle:
            X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=test_size, 
                                                     shuffle=True, random_state=random_state)
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y
    
    if logger:
        logger.info(f"Train-test split - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and validation loss).
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, feature_name: str, 
                    save_path: Optional[str] = None) -> None:
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        feature_name: Name of the feature being predicted
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Ensure we're plotting the first column if multi-dimensional
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_plot = y_true[:, 0]
    else:
        y_true_plot = y_true.flatten() if len(y_true.shape) > 1 else y_true
        
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_plot = y_pred[:, 0]
    else:
        y_pred_plot = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
    
    plt.plot(range(len(y_true_plot)), y_true_plot, color='blue', label='Actual')
    plt.plot(range(len(y_pred_plot)), y_pred_plot, color='red', label='Predicted')
    plt.xlabel("Sample Number")
    plt.ylabel(feature_name)
    plt.title(f"{feature_name} - Actual vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def create_output_directories(target_feature: str) -> Dict[str, str]:
    """
    Create necessary output directories for models, logs, and visualizations.
    All files from a single execution are stored in the same timestamped subfolder.
    
    Args:
        target_feature: The target feature being predicted
    
    Returns:
        Dictionary with paths to created directories
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = target_feature.lower()
    
    # Create a single timestamped execution folder for all outputs
    execution_dir = os.path.join('results', target_dir, timestamp)
    
    # Create directory structure with all outputs in the same execution folder
    dirs = {
        'models': os.path.join(execution_dir, 'models'),
        'logs': os.path.join(execution_dir, 'logs'),
        'plots': os.path.join(execution_dir, 'plots'),
        'results': execution_dir
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs