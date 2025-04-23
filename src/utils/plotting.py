"""Plotting utilities for stock price prediction visualization.

This module provides a comprehensive set of functions for creating visualizations of stock price data,
model predictions, and training history. It encapsulates complex matplotlib operations into a simple,
consistent API through the Plotter class, making it easy to generate professional-quality visualizations
for financial time series analysis and forecasting.

The module includes utilities for:

1. Plotting training history (loss curves) - Visualize model convergence and identify potential
   overfitting or underfitting during the training process.
   
2. Visualizing predictions vs actual values - Compare model predictions against ground truth
   to evaluate model performance and accuracy on historical data.
   
3. Plotting future predictions alongside historical data - Create forecasting visualizations
   that show both historical trends and future projections with clear visual distinction.
   
4. Configuring plot aesthetics and date formatting - Handle the complexities of date-based
   x-axes with appropriate tick spacing and formatting for time series data.

These visualizations are essential for:
- Understanding model performance and limitations
- Identifying patterns and trends in financial data
- Communicating results effectively to stakeholders
- Making data-driven investment decisions based on visual analysis

All plotting functions support saving the generated visualizations to files for inclusion
in reports or presentations. The module handles various edge cases such as multi-dimensional
input arrays, date alignment, and appropriate scaling of visual elements.
"""

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
        Plot training history (loss and validation loss). This method visualizes the training
        progress of a machine learning model by plotting the loss metrics over epochs.
        
        The visualization shows both training loss and validation loss on the same graph,
        allowing for easy comparison and identification of potential overfitting or underfitting.
        This is crucial for understanding model convergence and training effectiveness.
        
        Args:
            history: Training history dictionary containing at minimum 'loss' and 'val_loss' keys,
                    each mapping to a list of float values representing the loss at each epoch.
                    This is typically the history object returned by model.fit() in frameworks
                    like TensorFlow/Keras.
            save_path: Optional path to save the plot as an image file. If provided, the method
                      will ensure the directory exists and save the figure.
                      
        Note:
            The plot uses a consistent color scheme with grid lines for better readability.
            Training loss is typically expected to decrease over epochs, while validation loss
            should follow a similar trend without significant divergence from training loss.
            Divergence between the two curves can indicate overfitting.
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
        Initialize a figure for plotting with date ticks. This method handles the complex task of
        creating appropriate date-based tick marks on the x-axis of time series plots.

        The method intelligently determines the optimal number of date ticks to display based on
        the figure width, ensuring readability while avoiding overcrowding. It selects evenly
        spaced dates across the time range, always including the first and last dates for context.

        Args:
            aligned_dates: Pandas Series containing aligned dates for the x-axis. Can be either
                          a pandas Series or a regular list/array of date strings or datetime objects.
            x_positions: Numpy array of x-axis positions that correspond to each date. These are
                        the numeric positions used for actual plotting coordinates.

        Note:
            This method automatically handles different date formats and ensures that tick labels
            are properly formatted as 'YYYY-MM-DD' when possible. It also rotates date labels by
            45 degrees to prevent overlap with long date strings.
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
        Plot actual vs predicted values with dates on x-axis if available. This method creates a
        visualization comparing model predictions against actual values, which is essential for
        evaluating model performance in stock price prediction tasks.
        
        The method handles both single-dimensional and multi-dimensional input arrays by
        automatically extracting the first column for plotting. It also manages date alignment
        and formatting when date information is provided.
        
        Args:
            feature_name: Name of the feature being predicted (e.g., 'Close Price', 'Volume').
                         This will be used for the y-axis label and plot title.
            y_pred: Predicted values as a numpy array. Can be either 1D or 2D (in which case
                   the first column will be used).
            y_true: True/actual values as a numpy array, optional. When provided, these will be
                   plotted alongside predictions for comparison.
            dates: Optional pandas Series containing dates for x-axis. When provided, the plot
                  will use dates instead of sample numbers, making the visualization more
                  interpretable for time series data.
            save_path: Optional path to save the plot as an image file. If provided, the method
                      will ensure the directory exists and save the figure.
            days_shift: Number of days shifted in the prediction (for date alignment). This parameter
                       helps align predictions with actual values when there's a time offset.
        
        Note:
            The method automatically handles array length mismatches by plotting only the
            overlapping portion of the data. It also applies consistent styling with blue for
            actual values and red for predictions.
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
        Generate future dates starting from the last available date. This method is essential for
        creating forecasting visualizations that extend beyond the available historical data.
        
        The method takes the last known date from historical data and generates a sequence of
        future dates at daily intervals. These dates can then be used to properly label the x-axis
        for future predictions, maintaining temporal continuity in visualizations.
        
        Args:
            last_date: The last date in the historical data (format: 'YYYY-MM-DD' or any format
                      that pandas.to_datetime can parse). This serves as the starting point for
                      generating future dates.
            num_days: Number of future dates to generate. This should match the number of future
                     predictions being visualized.
            
        Returns:
            List of future dates in 'YYYY-MM-DD' format, starting from the day after last_date
            and continuing for num_days.
            
        Note:
            The method includes robust error handling for date parsing. If the provided last_date
            cannot be parsed, it falls back to using the current date as a starting point and
            issues a warning message. This ensures the method doesn't fail even with invalid input.
            
            The generated dates follow calendar days and do not account for trading days or
            holidays. For financial applications requiring only business days, additional
            filtering may be needed.
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
        This method creates a comprehensive visualization that shows both historical stock data
        and future predictions in a continuous timeline, making it ideal for forecasting analysis.
        
        The method creates a visual distinction between historical (known) data and predicted
        (future) values while maintaining visual continuity through a connecting line. This helps
        users understand the relationship between past performance and future projections.
        
        Args:
            feature_name: Name of the feature being predicted (e.g., 'Close Price', 'Volume').
                         This will be used for the y-axis label and plot title.
            y_pred: Predicted future values as a numpy array. Can be either 1D or 2D (in which case
                   the first column will be used).
            historical_data: Historical values to display before predictions. When provided, these
                           values are plotted in blue to distinguish them from predictions (red).
                           The method automatically handles the transition between historical and
                           predicted data with a connecting line.
            dates: Optional pandas Series containing dates for historical data. When provided, the
                  method will generate appropriate future dates for predictions and display a
                  date-based x-axis instead of sample numbers.
            save_path: Optional path to save the plot as an image file. If provided, the method
                      will ensure the directory exists and save the figure.
        
        Note:
            The connecting line between historical and predicted data is created by drawing a line
            segment between the last historical data point and the first prediction point. This
            creates a smooth visual transition while still maintaining the distinction between
            actual historical data and model predictions.
            
            When dates are provided, the method automatically generates future dates for the
            prediction period using the generate_future_dates method, ensuring proper temporal
            alignment of the visualization.
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