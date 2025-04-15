# Recursive Neural Network for Stock Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from utils import (
    clear_zero_values,
    normalize_data,
    plot_predictions,
    load_normalization_params
)

def prepare_sequence_input_data(input_data: pd.DataFrame, target_data: pd.DataFrame, 
                         shift: int, logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequence data for RNN training/prediction.
    
    Args:
        input_data: Input features DataFrame
        target_data: Target features DataFrame
        shift: Number of time steps to use for sequence
        logger: Optional logger for logging information
    
    Returns:
        X_sequence as numpy arrays
    """
    X = []
    
    if shift > 0:
        for i in range(len(input_data) - shift*2):
            X.append(input_data.iloc[i:i+shift].values)  # Past N days
        
        X_sequence = np.array(X)
    else:
        X_sequence = input_data.values
        X_sequence = X_sequence.reshape((X_sequence.shape[0], 1, X_sequence.shape[1]))
    
    if logger:
        logger.info(f"Sequence data shapes - X: {X_sequence.shape}")
    
    return X_sequence

def load_and_prepare_data(config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for prediction.
    
    Args:
        config: Configuration dictionary
        logger: Logger for logging information
        
    Returns:
        Tuple of (input_data, target_data, original_data)
    """
    # Extract configuration parameters
    features = config['data_preparation']['features']
    target_feature = config['data_preparation']['target_feature']
    include_target = config['data_preparation']['include_target']
    
    # Load dataset
    logger.info("Loading dataset...")
    data = pd.read_csv('.\\csv\\S&P500.csv')#[-100:]
    data = data[[feat for feat in data.columns if feat != "Price" and feat != "Adj Close"]]
    logger.info(f"Dataset loaded with shape: {data.shape}")
    logger.info(f"Zero values per column: {(data==0).sum().to_dict()}")
    
    # Clear zero values
    data = clear_zero_values(data, features, logger)
    
    # Log dataset statistics
    logger.info("Dataset statistics:")
    logger.info(data.describe().to_string())
    
    # Prepare input and target feature sets
    if include_target:
        input_features = features
    else:
        input_features = [feat for feat in features if feat != target_feature]
        
    target_features = [feat for feat in features if feat == target_feature]
    logger.info(f"Input features: {input_features}")
    logger.info(f"Target feature: {target_features}")
    
    # Create input and target datasets
    input_data = data[input_features]
    target_data = data[target_features]
    
    logger.info(f"Input shape: {input_data.shape}, Target shape: {target_data.shape}")
    
    return input_data, target_data, data

def create_prediction_directories(model_path: str, target_feature: str) -> Dict[str, str]:
    """
    Create prediction output directories in the same parent folder as the model.
    
    Args:
        model_path: Path to the model file
        target_feature: The target feature being predicted
        
    Returns:
        Dictionary with paths to created directories
    """
    # Get the model's parent directory
    model_dir = os.path.dirname(os.path.abspath(model_path))
    parent_dir = os.path.dirname(model_dir)
    
    # Create a predictions folder in the parent directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    predictions_dir = os.path.join(parent_dir, "predictions", timestamp)
    
    # Create directory structure with all outputs in the predictions folder
    dirs = {
        'logs': os.path.join(predictions_dir, 'logs'),
        'plots': os.path.join(predictions_dir, 'plots'),
        'results': predictions_dir
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def predict_future(config: Dict[str, Any]) -> np.ndarray:
    """
    Load model and make predictions.
    
    Args:
        config: Configuration dictionary with prediction parameters
        
    Returns:
        Array of predictions
    """
    # Extract configuration parameters
    target_feature = config['data_preparation']['target_feature']
    days = config['data_preparation']['days']
    model_path = config['prediction']['model_path']
    
    # Create prediction directories in the model's parent folder
    output_dirs = create_prediction_directories(model_path, target_feature)
    
    # Setup logging
    logger = logging.getLogger(f"stock_prediction_{target_feature}")
    logger.setLevel(logging.INFO)
    
    # Create file handler with UTF-8 encoding
    log_file = os.path.join(output_dirs['logs'], "execution.log")
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
    
    logger.info("Starting prediction process with configuration:")
    logger.info(f"Target feature: {target_feature}, Days: {days}")
    logger.info(f"Model path: {model_path}")
    
    try:
        # Load the model
        logger.info(f"Loading model from {model_path}...")
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load and prepare data
        input_data, target_data, original_data = load_and_prepare_data(config, logger)
        
        # Load normalization parameters from the model directory
        model_dir = os.path.dirname(os.path.abspath(model_path))
        data_min, data_max = load_normalization_params(model_dir, logger)
        
        # Normalize the input data using the saved parameters
        if data_min is not None and data_max is not None:
            logger.info("Using saved normalization parameters")
            input_normalized, _, _ = normalize_data(input_data, data_min, data_max)
        else:
            logger.warning("Saved normalization parameters not found, calculating from current data")
            input_normalized, input_min, input_max = normalize_data(input_data)
        logger.info("Data normalized")
        
        # Prepare sequence data for prediction
        shift = days
        X_sequence = prepare_sequence_input_data(input_normalized, target_data, shift, logger)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_sequence)
        logger.info(f"Predictions shape: {y_pred.shape}")
        
        # Plot and save predictions
        predictions_plot_path = os.path.join(output_dirs['plots'], 'predictions_real.png')
        plot_predictions(y_pred, y_pred, target_feature, save_path=predictions_plot_path)
        logger.info(f"Predictions plot saved to {predictions_plot_path}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'predicted': y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        })
        predictions_csv_path = os.path.join(output_dirs['results'], 'predictions_real.csv')
        predictions_df.to_csv(predictions_csv_path, index=False)
        logger.info(f"Predictions saved to {predictions_csv_path}")
        
        return y_pred
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import json
    
    # Load configuration from JSON file
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Make predictions using the configuration
        predictions = predict_future(config['target_prediction'])
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")