"""Plotting utilities for stock prediction visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime, timedelta

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
    def initialize_figure(aligned_dates: pd.Series, x_positions: np.ndarray) -> None:
        """
        Initialize a figure for plotting with date ticks.

        Args:
            aligned_dates: Pandas Series containing aligned dates
            x_positions: Numpy array of x-axis positions
        """
        # Create figure object
        fig = plt.gcf()

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
        
        # Create axis objects
        ax = plt.gca()

        # Set the tick positions and labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

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
        y_pred_plot = y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        if y_true is not None:
            y_true_plot = y_true[:, 0] if len(y_true.shape) > 1 else y_true
        else:
            y_true_plot = None
        
        # Create numeric x-axis positions for plotting
        x_positions = np.arange(len(y_pred_plot))
        
        # Prepare x-axis values (dates or sample numbers)
        if dates is not None and len(dates) > 0:
            x_label = "Date"            
            
            # Align dates with the data points
            if hasattr(dates, 'iloc'):
                # For pandas Series
                aligned_dates = dates.iloc[-len(y_pred_plot):] if len(dates) >= len(y_pred_plot) else dates
            else:
                # For regular lists/arrays
                aligned_dates = dates[-len(y_pred_plot):] if len(dates) >= len(y_pred_plot) else dates
            
            # Initialize figure with aligned dates
            Plotter.initialize_figure(aligned_dates, x_positions)
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
        
    @staticmethod
    def generate_future_dates(last_date: str, num_days: int) -> List[str]:
        """
        Generate future dates starting from the last available date.
        
        Args:
            last_date: The last date in the historical data (format: 'YYYY-MM-DD')
            num_days: Number of future dates to generate
            
        Returns:
            List of future dates in 'YYYY-MM-DD' format
        """
        # Convert the last date to datetime object
        try:
            last_datetime = pd.to_datetime(last_date)
        except Exception as e:
            # If conversion fails, use current date
            print(f"Warning: Could not parse date '{last_date}'. Using current date instead.")
            last_datetime = datetime.now()
            
        # Generate future dates
        future_dates = []
        for i in range(1, num_days + 1):
            future_date = last_datetime + timedelta(days=i)
            future_dates.append(future_date.strftime('%Y-%m-%d'))
            
        return future_dates
    
    @staticmethod
    def plot_future_predictions(feature_name: str, y_pred: np.ndarray, 
                              historical_data: Optional[np.ndarray] = None,
                              dates: Optional[pd.Series] = None, 
                              save_path: Optional[str] = None) -> None:
        """
        Plot historical data and future predictions on the same graph with a connecting line.
        
        Args:
            feature_name: Name of the feature being predicted
            y_pred: Predicted future values
            historical_data: Historical values to display before predictions
            dates: Optional pandas Series containing dates for historical data
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Ensure we're plotting the first column if multi-dimensional
        y_pred_plot = y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        
        # Process historical data if provided
        if historical_data is not None:
            hist_plot = historical_data[:, 0] if len(historical_data.shape) > 1 else historical_data
            
            # Create x-axis positions for plotting
            hist_x = np.arange(len(hist_plot))
            pred_x = np.arange(len(hist_plot), len(hist_plot) + len(y_pred_plot))
            
            # Prepare dates for x-axis if available
            if dates is not None and len(dates) > 0:
                x_label = "Date"
                
                # Format historical dates
                if hasattr(dates, 'iloc'):
                    # For pandas Series
                    hist_dates = dates.iloc[:len(hist_plot)]
                    last_date = dates.iloc[-1] if len(dates) > 0 else None
                else:
                    # For regular lists/arrays
                    hist_dates = dates[:len(hist_plot)]
                    last_date = dates[-1] if len(dates) > 0 else None
                
                # Generate future dates for predictions
                if last_date is not None:
                    future_dates = Plotter.generate_future_dates(last_date, len(y_pred_plot))
                    all_dates = list(hist_dates) + future_dates
                else:
                    all_dates = hist_dates
                    
                # Create combined x positions
                all_x = np.concatenate([hist_x, pred_x])
                
                # Initialize figure with all dates
                Plotter.initialize_figure(all_dates, all_x)
            else:
                x_label = "Sample Number"
            
            # Plot historical data
            plt.plot(hist_x, hist_plot, color='blue', label='Historical')
            
            # Create a connecting line between historical and predicted data
            if len(hist_plot) > 0 and len(y_pred_plot) > 0:
                # Create a small array with just the last historical point and first prediction point
                connect_x = [hist_x[-1], pred_x[0]]
                connect_y = [hist_plot[-1], y_pred_plot[0]]
                # Plot the connecting line with the same color as predictions but no label
                plt.plot(connect_x, connect_y, color='red', linestyle='-', linewidth=1.5)
            
            # Plot predictions
            plt.plot(pred_x, y_pred_plot, color='red', label='Predicted')
        else:
            # If no historical data, just plot predictions
            x_positions = np.arange(len(y_pred_plot))
            plt.plot(x_positions, y_pred_plot, color='red', label='Predicted')
            x_label = "Sample Number"
        
        # Set labels and title
        plt.xlabel(x_label)
        plt.ylabel(feature_name)
        plt.title(f"{feature_name} - Stock Prediction with Future Forecast")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()