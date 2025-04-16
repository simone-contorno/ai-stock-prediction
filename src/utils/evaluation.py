"""Evaluation utilities for stock prediction models."""

import numpy as np
import logging
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelEvaluator:
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                           logger: logging.Logger) -> Dict[str, float]:
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