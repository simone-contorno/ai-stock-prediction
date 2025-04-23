"""Prediction script for Stock Price Prediction RNN model."""

import os

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from typing import Dict, Any

from src.utils import (
    Logger,
    Plotter,
    DataPreprocessor,
    DataLoader
)

def predict_future(config: Dict[str, Any]) -> np.ndarray:
    """Load model and make predictions."""
    # Extract configuration parameters
    target_feature = config['general']['target_feature']
    model_path = config['prediction']['model_path']
    
    # Extract days from model filename
    import re
    days_match = re.search(r'_lstm_\w+_(\d+)\.keras$', model_path)
    days = int(days_match.group(1)) if days_match else config['general']['days']  # Default to days if not found
    
    # Setup logging with 'predict' subfolder
    logger = Logger.setup(target_feature, is_training=False, model_path=model_path, subfolder_name='predict')
    
    logger.info("Starting prediction process with configuration:")
    logger.info(f"Target feature: {target_feature}, Days: {days}")
    logger.info(f"Model path: {model_path}")
    
    try:
        # Load the model
        logger.info(f"Loading model from {model_path}...")
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load raw data
        data, dates = DataLoader.load_raw_data(config, logger)

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
            input_normalized, _, _, _ = DataPreprocessor.normalize_data(input_data, scaler_x=scaler_x, scaler_y=scaler_y)
        else:
            logger.warning("Saved normalization scalers not found, generating from current data")
            input_normalized, _, scaler_x, scaler_y = DataPreprocessor.normalize_data(input_data)
        logger.info(f"Data normalized:\n {input_normalized.describe()}")
        
        # Prepare sequence data for prediction
        X_sequence, _ = DataPreprocessor.prepare_sequence_data(days, input_normalized, logger=logger)
        
        # Take the needed data subset
        X_sequence = X_sequence[-days*2:]
        target_data = target_data[-days*2:]

        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_sequence)
        
        # Denormalize predictions
        y_pred = scaler_y.inverse_transform(y_pred)
        logger.info(f"Predictions shape: {y_pred.shape}")

        # Plot and save predictions
        predictions_plot_path = os.path.join(logger.output_dirs['plots'], 'predictions_real.png')
        
        # Get the target data for historical values
        target_data_array = target_data.values if isinstance(target_data, pd.DataFrame) else target_data
        
        # Use the new plot_future_predictions function
        Plotter.plot_future_predictions(
            feature_name=target_feature, 
            y_pred=y_pred[-days:], 
            historical_data=target_data_array, 
            dates=dates[-len(target_data_array):], 
            save_path=predictions_plot_path
        )
        logger.info(f"Future predictions plot saved to {predictions_plot_path}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'predicted': y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        })
        predictions_csv_path = os.path.join(logger.output_dirs['results'], 'predictions_real.csv')
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
        predictions = predict_future(config)
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")