"""Data preprocessing utilities for stock prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import json
import logging

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
    def normalize_data(data: Union[pd.DataFrame, np.ndarray], 
                      data_min: Optional[Union[pd.Series, np.ndarray, Dict]] = None, 
                      data_max: Optional[Union[pd.Series, np.ndarray, Dict]] = None
                      ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray, Dict], Union[pd.Series, np.ndarray, Dict]]:
        """
        Normalize data to range [0, 1].
        
        Args:
            data: Input data (DataFrame or numpy array)
            data_min: Optional minimum values for normalization (if None, calculated from data)
            data_max: Optional maximum values for normalization (if None, calculated from data)
        
        Returns:
            Tuple of (normalized_data, min_values, max_values)
        """

        if data_min is None or data_max is None:
            # Calculate min and max from the data
            if isinstance(data, pd.DataFrame):
                data_min = data.min() if data_min is None else data_min
                data_max = data.max() if data_max is None else data_max
            else:  # For numpy arrays
                data_min = np.min(data, axis=0) if data_min is None else data_min
                data_max = np.max(data, axis=0) if data_max is None else data_max
        
        # Convert dictionary min/max values to Series if needed
        if isinstance(data_min, dict) and isinstance(data, pd.DataFrame):
            data_min = pd.Series(data_min)
        if isinstance(data_max, dict) and isinstance(data, pd.DataFrame):
            data_max = pd.Series(data_max)
        
        # Normalize using the provided or calculated min/max values
        normalized_data = (data - data_min) / (data_max - data_min)
        
        return normalized_data, data_min, data_max

    @staticmethod
    def prepare_sequence_data(input_data: pd.DataFrame, target_data: pd.DataFrame, 
                            shift: int, logger: Optional[logging.Logger] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM training/prediction.
        
        Args:
            input_data: Input features DataFrame
            target_data: Target features DataFrame
            shift: Number of time steps to use for sequence
            logger: Optional logger for logging information
        
        Returns:
            Tuple of (X_sequence, y_sequence) as numpy arrays
        """
        X, y = [], []
        
        if shift > 0:
            for i in range(len(input_data) - shift*2):
                X.append(input_data.iloc[i:i+shift].values)  # Past N days
                y.append(target_data.iloc[i+shift*2])  # Next N days
            
            X_sequence = np.array(X)
            y_sequence = np.array(y)
        else:
            X_sequence = input_data.values
            y_sequence = target_data
            X_sequence = X_sequence.reshape((X_sequence.shape[0], 1, X_sequence.shape[1]))
        
        if logger:
            logger.info(f"Sequence data shapes --> X: {X_sequence.shape}, y: {y_sequence.shape}")
        
        return X_sequence, y_sequence

    @staticmethod
    def prepare_sequence_input_data(input_data: pd.DataFrame, shift: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
        """
        Prepare sequence data for LSTM prediction (input only).
        
        Args:
            input_data: Input features DataFrame
            target_data: Target features DataFrame
            shift: Number of time steps to use for sequence
            logger: Optional logger for logging information
        
        Returns:
            X_sequence as numpy array
        """

        X = []
        
        if shift > 0:
            for i in range(shift):
                X.append(input_data.iloc[i-shift*2:i-shift].values)  # Past N days
            
            X_sequence = np.array(X)
        else:
            X_sequence = input_data.values
            X_sequence = X_sequence.reshape((X_sequence.shape[0], 1, X_sequence.shape[1]))
        
        if logger:
            logger.info(f"Sequence data shapes --> X: {X_sequence.shape}")
        
        return X_sequence

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

        #X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=test_size, 
        #                                            shuffle=shuffle, random_state=random_state)
        
        if logger:
            logger.info(f"Train-test split - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test