"""Plotting utilities for stock prediction visualization."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os

class Plotter:
    @staticmethod
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

    @staticmethod
    def plot_predictions(feature_name: str, y_pred: np.ndarray, y_true: np.ndarray = None, 
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
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_plot = y_pred[:, 0]
        else:
            y_pred_plot = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
        
        if y_true is not None:
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true_plot = y_true[:, 0]
            else:
                y_true_plot = y_true.flatten() if len(y_true.shape) > 1 else y_true

            plt.plot(range(len(y_true_plot)), y_true_plot, color='blue', label='Actual')

        plt.plot(range(len(y_pred_plot)), y_pred_plot, color='red', label='Predicted')
        plt.xlabel("Sample Number")
        plt.ylabel(feature_name)
        plt.title(f"{feature_name} - Stock Prediction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()