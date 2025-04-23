"""Data preprocessing utilities for stock price prediction.

This module provides utilities for preprocessing time series data for stock price prediction:

1. Data cleaning - removing rows with zero values that could skew the model
2. Normalization - scaling features to a [0,1] range for better model convergence
3. Sequence preparation - transforming time series data into supervised learning format
4. Train/test splitting - dividing data into training and testing sets

These preprocessing steps are essential for preparing raw stock data for LSTM models,
which require normalized inputs and sequential data formatting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union
import logging
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    @staticmethod
    def clear_zero_values(data: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
        """
        Remove rows with zero values in specified features.
        
        Args:
            data: Input DataFrame
            logger: Optional logger for logging information
        
        Returns:
            DataFrame with zero values removed
        """
        if logger:
            logger.info(f"Original data shape: {data.shape}")
            logger.info(f"Zero values per column: {(data==0).sum().to_dict()}")
        
        # Drop rows with zero values
        for feat in data.columns:
            data.drop(data[data[feat] == 0].index, inplace=True)
        
        if logger:
            logger.info(f"Data shape after removing zeros: {data.shape}")
            logger.info(f"Remaining zero values: {(data==0).sum().to_dict()}")
        
        return data

    @staticmethod
    def normalize_data(input_data: Union[pd.DataFrame, np.ndarray], target_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                       scaler_x: Optional[MinMaxScaler] = None, scaler_y: Optional[MinMaxScaler] = None
                      ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray, Dict], Union[pd.Series, np.ndarray, Dict]]:
        """
        Normalize data to range [0, 1] using MinMaxScaler.
        
        Normalization is crucial for neural networks to ensure all features contribute equally
        to the learning process and to help the model converge faster. This function scales
        all input features and target values to the range [0, 1].
        
        The function can either:
        1. Create new scalers and fit them to the data (for training)
        2. Use existing scalers to transform the data (for testing/prediction)
        
        Args:
            input_data: Input features data (DataFrame or numpy array)
            target_data: Optional target data to predict (DataFrame or numpy array)
            scaler_x: Optional pre-fitted MinMaxScaler for input data (used for test/prediction)
            scaler_y: Optional pre-fitted MinMaxScaler for target data (used for test/prediction)
        
        Returns:
            Tuple of (normalized input data, normalized target data, scaler_x, scaler_y)
            where normalized data maintains the same format as input (DataFrame or numpy array)
        """

        # Normalize using the provided or calculated min/max values
        if scaler_x is None:
            scaler_x = MinMaxScaler()
            normalized_input_array = scaler_x.fit_transform(input_data)
        else:
            normalized_input_array = scaler_x.transform(input_data)

        if scaler_y is None:
            scaler_y = MinMaxScaler()
            normalized_target_array = scaler_y.fit_transform(target_data) if target_data is not None else None
        else:
            normalized_target_array = scaler_y.transform(target_data) if target_data is not None else None

        # Convert back to DataFrame 
        normalized_input_data = pd.DataFrame(normalized_input_array, columns=input_data.columns, index=input_data.index)
        normalized_target_data = pd.DataFrame(normalized_target_array, columns=target_data.columns, index=target_data.index) if target_data is not None else None

        return normalized_input_data, normalized_target_data, scaler_x, scaler_y

    @staticmethod
    def prepare_sequence_data(shift: int, input_data: pd.DataFrame, target_data: Optional[pd.DataFrame] = None, 
                            logger: Optional[logging.Logger] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM training/prediction by creating sliding windows.
        
        This function transforms time series data into a supervised learning format
        suitable for LSTM models. It creates sequences of 'shift' length from the input data,
        where each sequence is paired with a target value that is 'shift' steps in the future.
        
        For example, with shift=3:
        - Input sequence: [day1, day2, day3] → Target: day6
        - Input sequence: [day2, day3, day4] → Target: day7
        - And so on...
        
        This sliding window approach allows the LSTM to learn patterns over time
        and make predictions for future time steps.
        
        Args:
            shift: Number of time steps to use for each input sequence and
                   how many steps ahead to predict (prediction horizon)
            input_data: Input features DataFrame (normalized)
            target_data: Target features DataFrame (normalized) or None for prediction mode
            logger: Optional logger for logging information
        
        Returns:
            Tuple of (X_sequence, y_sequence) as numpy arrays, where:
            - X_sequence has shape (samples, time_steps, features)
            - y_sequence has shape (samples, target_features) or None if target_data is None
        """
        X, y = [], []
        
        total_shift = shift*2 if target_data is not None else shift

        if shift > 0:
            logger.info(f"Preparing sequence data with shift {shift}")
            for i in range(len(input_data) - total_shift):
                X.append(input_data.iloc[i:i+shift].values)  # Past N days
                if target_data is not None:
                    y.append(target_data.iloc[i+total_shift])  # Next N days
            
            X_sequence = np.array(X)
            y_sequence = np.array(y) if target_data is not None else None
        else:
            X_sequence = input_data.values
            y_sequence = target_data if target_data is not None else None
            X_sequence = X_sequence.reshape((X_sequence.shape[0], 1, X_sequence.shape[1]))
        
        if logger:
            logger.info(f"Sequence data shapes --> X: {X_sequence.shape}, y: {y_sequence.shape if y_sequence is not None else 'N/A'}")
        
        return X_sequence, y_sequence

    @staticmethod
    def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float, 
                        shuffle: bool = True, random_state: int = 42, 
                        logger: Optional[logging.Logger] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.
        
        Args:
            X: Input features array
            y: Target array 
            test_size: Proportion of data to use for testing
            shuffle: Whether to shuffle the data before splitting
            random_state: Random seed for reproducibility
            logger: Optional logger for logging information
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        #from sklearn.model_selection import train_test_split
        
        # Extract test set from end of sequence for time series data
        X_test = X[int(-len(X)*test_size)+1:]
        y_test = y[int(-len(y)*test_size)+1:]
        
        X_train = X[:int(len(X)*(1-test_size))]
        y_train = y[:int(len(y)*(1-test_size))]
        
        # Shuffle training data
        if shuffle:
            np.random.seed(random_state)
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

        if logger:
            logger.info(f"Train-test split - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test