"""Test script for Stock Price Prediction RNN model."""

import os

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from typing import Dict, Any

from src.utils import (
    Logger,
    Plotter,
    DataPreprocessor,
    DataLoader,
    ModelEvaluator
)

# Disable oneDNN optimizations for TensorFlow
# This is a workaround for a known issue with TensorFlow and oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def test_model(config: Dict[str, Any]) -> np.ndarray:
    """Load model and make predictions."""
    # Extract configuration parameters
    target_feature = config['general']['target_feature']
    model_path = config['prediction']['model_path']
    evaluate = config['prediction']['evaluate']
    
    # Extract days from model filename
    import re
    days_match = re.search(r'_lstm_\w+_(\d+)\.keras$', model_path)
    days = int(days_match.group(1)) if days_match else config['general']['days']  # Default to days if not found
    
    # Setup logging with 'test' subfolder
    logger = Logger.setup(target_feature, is_training=False, model_path=model_path, subfolder_name='test')
    
    logger.info("Starting prediction process with configuration:")
    logger.info(f"Target feature: {target_feature}, Days: {days}")
    logger.info(f"Model path: {model_path}")
    
    try:
        # Load the model
        logger.info(f"Loading model from {model_path}...")
        model = load_model(model_path)
        logger.info("Model loaded successfully")

        # Load raw data
        data, dates = DataLoader.load_raw_data(config, logger, is_training=False)
        
        # Clear zero values from raw data
        data = DataPreprocessor.clear_zero_values(data, logger)
        
        # Split into input and target features
        input_data, target_data = DataLoader.prepare_features(data, config, logger)
        
        # Load normalization parameters from the model directory
        model_dir = os.path.dirname(os.path.abspath(model_path))
        scaler_x, scaler_y = DataLoader.load_normalization_params(model_dir, logger)
        
        # Normalize input data using saved parameters
        if scaler_x is not None and scaler_y is not None:
            logger.info("Using saved normalization scalers")
            input_normalized, target_normalized, _, _ = DataPreprocessor.normalize_data(input_data, target_data, scaler_x=scaler_x, scaler_y=scaler_y)
        else:
            logger.warning("Saved normalization scalers not found, generating from current data")
            input_normalized, target_normalized, scaler_x, scaler_y = DataPreprocessor.normalize_data(input_data)
        logger.info(f"Data normalized:\n {input_normalized.describe()}")
        logger.info(f"Target data normalized:\n {target_normalized.describe()}")
        
        # Prepare sequence data for prediction
        X_sequence, y_sequence = DataPreprocessor.prepare_sequence_data(days, input_normalized, target_normalized, logger)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_sequence)
        logger.info(f"Predictions shape: {y_pred.shape}")

        # Denormalize predictions
        y_pred = scaler_y.inverse_transform(y_pred)
        #logger.info(f"Predictions denormalized:\n {y_pred.describe()}")
        y_sequence = scaler_y.inverse_transform(y_sequence)
        #logger.info(f"Target data denormalized:\n {y_sequence.describe()}")
        
        # Align the output with the target feature
        y_pred = y_pred[days:]
        y_sequence = y_sequence[:-days]
        dates = dates[-len(y_sequence):]
        
        # Evaluate predictions if requested
        if evaluate:
            metrics = ModelEvaluator.evaluate_predictions(y_sequence, y_pred, logger)
            
            # Save metrics to file
            metrics_path = os.path.join(logger.output_dirs['results'], 'metrics.txt')
            with open(metrics_path, 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
            logger.info(f"Metrics saved to {metrics_path}")
        
        # Plot and save predictions
        test_plot_path = os.path.join(logger.output_dirs['plots'], 'test.png')
        Plotter.plot_predictions(target_feature, y_pred, y_sequence, dates=dates, save_path=test_plot_path, days_shift=days)
        logger.info(f"Test plot saved to {test_plot_path}")

        # Plot and save predictions with offset 
        #test_plot_path = os.path.join(logger.output_dirs['plots'], 'test_offset.png')
        #y_pred_offset = y_pred + (y_pred[0] - y_sequence[0]) if y_pred[0] > y_sequence[0] else y_pred + (y_sequence[0] - y_pred[0])
        #Plotter.plot_predictions(y_sequence, y_pred_offset, target_feature, save_path=test_plot_path)
        #logger.info(f"Test plot saved to {test_plot_path}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'actual': y_sequence[:, 0] if len(y_sequence.shape) > 1 else y_sequence,
            'predicted': y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        })
        predictions_csv_path = os.path.join(logger.output_dirs['results'], 'test.csv')
        predictions_df.to_csv(predictions_csv_path, index=False)
        logger.info(f"Test predictions saved to {predictions_csv_path}")
        
        return y_pred
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import json
    
    # Load configuration from JSON file
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Make predictions using the configuration
        predictions = test_model(config)
        print("Testing completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")