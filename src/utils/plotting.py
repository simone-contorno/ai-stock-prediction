"""Plotting utilities for stock prediction visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
                        dates: Optional[pd.Series] = None, save_path: Optional[str] = None,
                        days_shift: int = 0) -> None:
        """
        Plot actual vs predicted values with dates on x-axis if available.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            feature_name: Name of the feature being predicted
            dates: Optional pandas Series containing dates for x-axis
            save_path: Optional path to save the plot
            days_shift: Number of days shifted in the prediction (for date alignment)
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
        else:
            y_true_plot = None
        
        # Create numeric x-axis positions for plotting
        x_positions = np.arange(len(y_pred_plot))
        
        # Prepare x-axis values (dates or sample numbers)
        if dates is not None and len(dates) > 0:
            x_label = "Date"
            
            # Create figure and axis objects
            fig = plt.gcf()
            ax = plt.gca()
            
            # Align dates with the data points if days_shift is provided
            aligned_dates = None
            if days_shift > 0 and len(dates) > days_shift:
                # For predictions, we need to use dates[days_shift:] to align with y_pred
                # which starts at position days_shift in the original sequence
                if hasattr(dates, 'iloc'):
                    aligned_dates = dates.iloc[days_shift:days_shift+len(y_pred_plot)]
                else:
                    aligned_dates = dates[days_shift:days_shift+len(y_pred_plot)]
            else:
                # If no shift or not enough dates, use the original dates
                aligned_dates = dates[:len(y_pred_plot)] if len(dates) > len(y_pred_plot) else dates
            
            # Determine a reasonable number of ticks based on figure width
            fig_width_inches = fig.get_figwidth()
            max_ticks = max(5, int(fig_width_inches * 1.5))  # Scale with figure width
            
            # Get the total number of dates
            n_dates = len(aligned_dates)
            
            # Select indices for tick positions (first, last, and evenly spaced points)
            if n_dates > max_ticks:
                indices = [0]  # Always include the first date
                
                # Add evenly spaced indices in the middle
                if max_ticks > 2:  # If we can show more than just first and last
                    step = n_dates / (max_ticks - 1)  # -1 because we already include first and last
                    indices.extend([int(i * step) for i in range(1, max_ticks - 1)])
                
                indices.append(n_dates - 1)  # Always include the last date
            else:
                # If we have few dates, show all of them
                indices = list(range(n_dates))
            
            # Filter indices to ensure they're within bounds of x_positions
            valid_indices = [i for i in indices if i < len(x_positions)]
            
            # Get tick positions (numeric indices for plotting)
            tick_positions = [min(i, len(x_positions)-1) for i in valid_indices]
            
            # Get tick labels (formatted dates) - ensure same length as positions
            if hasattr(aligned_dates, 'iloc'):
                # For pandas Series
                tick_labels = [aligned_dates.iloc[min(i, len(aligned_dates)-1)] for i in valid_indices]
            else:
                # For regular lists/arrays
                tick_labels = [aligned_dates[min(i, len(aligned_dates)-1)] for i in valid_indices]
            
            # Format dates if they're strings
            if tick_labels and isinstance(tick_labels[0], str):
                try:
                    # Try to convert to datetime for better display
                    formatted_labels = [pd.to_datetime(label).strftime('%Y-%m-%d') for label in tick_labels]
                    tick_labels = formatted_labels
                except Exception:
                    # Keep original strings if conversion fails
                    pass
            
            # Set the tick positions and labels
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
        else:
            # Otherwise use sample numbers
            x_label = "Sample Number"
        
        # Plot the data using numeric positions for x-axis
        if y_true_plot is not None:
            # Make sure we don't exceed array bounds
            plot_len = min(len(x_positions), len(y_true_plot))
            plt.plot(x_positions[:plot_len], y_true_plot[:plot_len], color='blue', label='Actual')

        # Plot predictions
        plot_len = min(len(x_positions), len(y_pred_plot))
        plt.plot(x_positions[:plot_len], y_pred_plot[:plot_len], color='red', label='Predicted')
        
        # Set labels and title
        plt.xlabel(x_label)
        plt.ylabel(feature_name)
        plt.title(f"{feature_name} - Stock Prediction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()