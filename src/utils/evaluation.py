"""Evaluation utilities for stock price prediction models.

This module provides functions for evaluating the performance of stock price prediction models
using various statistical metrics. It calculates and reports standard regression metrics that
quantify the accuracy of the predictions compared to actual values.

The main metrics calculated are:
- Mean Absolute Error (MAE): Average absolute difference between predicted and actual values
- Mean Squared Error (MSE): Average of squared differences, penalizes larger errors more heavily
- Root Mean Squared Error (RMSE): Square root of MSE, in the same unit as the target variable
- Mean Absolute Percentage Error (MAPE): Average percentage difference, scale-independent

These metrics help in understanding model performance and comparing different models.
"""

import numpy as np
import logging
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelEvaluator:
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                           logger: logging.Logger) -> Dict[str, float]:
        """
        Evaluate stock price predictions using multiple statistical metrics.
        
        This function calculates several regression metrics to assess how well
        the predicted stock prices match the actual values. Each metric provides
        a different perspective on model performance:
        
        - MAE (Mean Absolute Error): Average magnitude of errors without considering direction
          Lower is better. Unit is the same as the stock price.
        
        - MSE (Mean Squared Error): Average of squared errors, giving more weight to larger errors
          Lower is better. Unit is squared stock price.
        
        - RMSE (Root Mean Squared Error): Square root of MSE, bringing the unit back to stock price
          Lower is better. More interpretable than MSE.
        
        - MAPE (Mean Absolute Percentage Error): Average percentage error, scale-independent
          Lower is better. Expressed as a percentage.
        
        Args:
            y_true: Array of actual stock prices
            y_pred: Array of predicted stock prices
            logger: Logger for logging the calculated metrics
            
        Returns:
            Dictionary containing the calculated evaluation metrics
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
        logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f} %")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }