# Recursive Neural Network for Stock Price Prediction

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging
from datetime import datetime
from typing import Dict, Any

from utils import (
    normalize_data,
    prepare_sequence_data,
    plot_predictions,
    load_normalization_params,
    load_and_prepare_data
)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, logger: logging.Logger) -> Dict[str, float]:
    """
    Evaluate predictions using multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        logger: Logger for logging information
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure we're using the first column if multi-dimensional
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_eval = y_true[:, 0]
    else:
        y_true_eval = y_true.flatten() if len(y_true.shape) > 1 else y_true
        
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_eval = y_pred[:, 0]
    else:
        y_pred_eval = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_eval, y_pred_eval)
    mse = mean_squared_error(y_true_eval, y_pred_eval)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_eval - y_pred_eval) / y_true_eval)) * 100
    
    # Log results
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"  Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

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
    evaluate = config.get('prediction', {}).get('evaluate', False)
    
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
        input_data, target_data, _ = load_and_prepare_data(config, logger, days*2)
        
        # Load normalization parameters from the model directory
        model_dir = os.path.dirname(os.path.abspath(model_path))
        data_min, data_max = load_normalization_params(model_dir, logger)
        
        # Normalize the input data using the saved parameters
        if data_min is not None and data_max is not None:
            logger.info("Using saved normalization parameters")
            input_normalized, _, _ = normalize_data(input_data, data_min, data_max)
        else:
            logger.warning("Saved normalization parameters not found, calculating from current data")
            input_normalized, _, _ = normalize_data(input_data)
        logger.info("Data normalized")
        
        # Prepare sequence data for prediction
        shift = days
        X_sequence, y_sequence = prepare_sequence_data(input_normalized, target_data, shift, logger)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_sequence)
        logger.info(f"Predictions shape: {y_pred.shape}")
        
        # Evaluate predictions if requested
        if evaluate:
            metrics = evaluate_predictions(y_sequence, y_pred, logger)
            
            # Save metrics to file
            metrics_path = os.path.join(output_dirs['results'], 'metrics.txt')
            with open(metrics_path, 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
            logger.info(f"Metrics saved to {metrics_path}")
        
        # Plot and save predictions
        predictions_plot_path = os.path.join(output_dirs['plots'], 'predictions.png')
        plot_predictions(y_sequence, y_pred, target_feature, save_path=predictions_plot_path)
        logger.info(f"Predictions plot saved to {predictions_plot_path}")

        # Plot and save predictions with offset 
        predictions_plot_path = os.path.join(output_dirs['plots'], 'predictions_offset.png')
        y_pred_offset = y_pred + (y_pred[0] - y_sequence[0]) if y_pred[0] > y_sequence[0] else y_pred + (y_sequence[0] - y_pred[0])
        plot_predictions(y_sequence, y_pred_offset, target_feature, save_path=predictions_plot_path)
        logger.info(f"Predictions plot saved to {predictions_plot_path}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'actual': y_sequence[:, 0] if len(y_sequence.shape) > 1 else y_sequence,
            'predicted': y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        })
        predictions_csv_path = os.path.join(output_dirs['results'], 'predictions.csv')
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