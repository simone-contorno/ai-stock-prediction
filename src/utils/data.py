"""Data loading and saving utilities for stock prediction."""

import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, Tuple, Union, Optional, Any, List

class DataLoader:
    @staticmethod
    def load_raw_data(config: Dict[str, Any], logger: logging.Logger, days: int, 
                     is_training: bool = False) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            config: Configuration dictionary
            logger: Logger for logging information
            days: Number of days to predict
            is_training: Whether this is for training (load all data) or prediction (load last N days)
            
        Returns:
            Raw DataFrame with all columns
        """
        csv_path = config['data_preparation'].get('csv_path', '.\\csv\\S&P500.csv')
        
        # Load dataset
        logger.info("Loading dataset...")
        if is_training:
            # For training, load all data
            data = pd.read_csv(csv_path)
        else:
            # For prediction/testing, load only last N days
            data = pd.read_csv(csv_path)[-days*2:]
            
        data = data[[feat for feat in data.columns if feat != "Price" and feat != "Adj Close"]]
        logger.info(f"Dataset loaded with shape: {data.shape}")
        logger.info(f"Zero values per column: {(data==0).sum().to_dict()}")
        
        return data

    @staticmethod
    def prepare_features(data: pd.DataFrame, config: Dict[str, Any], 
                        logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into input and target features.
        
        Args:
            data: Raw DataFrame
            config: Configuration dictionary
            logger: Logger for logging information
            
        Returns:
            Tuple of (input_data, target_data)
        """
        features = config['data_preparation']['features']
        target_feature = config['data_preparation']['target_feature']
        include_target = config['data_preparation']['include_target']
        
        # Prepare input and target feature sets
        if include_target:
            input_features = features
        else:
            input_features = [feat for feat in features if feat != target_feature]
            
        target_features = [feat for feat in features if feat == target_feature]
        logger.info(f"Input features: {input_features}")
        logger.info(f"Target feature: {target_features}")
        
        # Create input and target datasets
        input_data = data[input_features]
        target_data = data[target_features]
        
        logger.info(f"Input shape: {input_data.shape}, Target shape: {target_data.shape}")
        
        return input_data, target_data

    @staticmethod
    def load_and_prepare_data(config: Dict[str, Any], logger: logging.Logger, days: int, 
                            is_training: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for prediction.
        
        Args:
            config: Configuration dictionary
            logger: Logger for logging information
            days: Number of days to predict
            is_training: Whether this is for training (load all data) or prediction (load last N days)
            
        Returns:
            Tuple of (input_data, target_data, original_data)
        """
        # 1. Load raw data
        data = DataLoader.load_raw_data(config, logger, days, is_training)
        
        # 2. Split into features
        input_data, target_data = DataLoader.prepare_features(data, config, logger)
        
        return input_data, target_data, data

    @staticmethod
    def save_normalization_params(data_min: Union[pd.Series, np.ndarray], 
                                data_max: Union[pd.Series, np.ndarray],
                                model_dir: str,
                                logger: Optional[logging.Logger] = None) -> str:
        """
        Save normalization parameters (min and max values) to a JSON file.
        
        Args:
            data_min: Minimum values used for normalization
            data_max: Maximum values used for normalization
            model_dir: Directory where the model is saved
            logger: Optional logger for logging information
            
        Returns:
            Path to the saved normalization parameters file
        """
        # Convert numpy arrays or pandas Series to lists for JSON serialization
        if isinstance(data_min, (np.ndarray, pd.Series)):
            data_min_list = data_min.tolist() if isinstance(data_min, np.ndarray) else data_min.to_dict()
        else:
            data_min_list = data_min
            
        if isinstance(data_max, (np.ndarray, pd.Series)):
            data_max_list = data_max.tolist() if isinstance(data_max, np.ndarray) else data_max.to_dict()
        else:
            data_max_list = data_max
        
        # Create normalization parameters dictionary
        norm_params = {
            'data_min': data_min_list,
            'data_max': data_max_list
        }
        
        # Save to JSON file in the same directory as the model
        norm_params_path = os.path.join(model_dir, 'normalization_params.json')
        with open(norm_params_path, 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        if logger:
            logger.info(f"Normalization parameters saved to {norm_params_path}")
        
        return norm_params_path

    @staticmethod
    def load_normalization_params(model_dir: str, 
                                logger: Optional[logging.Logger] = None
                                ) -> Tuple[Union[pd.Series, dict], Union[pd.Series, dict]]:
        """
        Load normalization parameters (min and max values) from a JSON file.
        
        Args:
            model_dir: Directory where the model is saved
            logger: Optional logger for logging information
            
        Returns:
            Tuple of (data_min, data_max) loaded from the file
        """
        # Construct the path to the normalization parameters file
        norm_params_path = os.path.join(model_dir, 'normalization_params.json')
        
        if not os.path.exists(norm_params_path):
            if logger:
                logger.warning(f"Normalization parameters file not found at {norm_params_path}")
            return None, None
        
        # Load from JSON file
        with open(norm_params_path, 'r') as f:
            norm_params = json.load(f)
        
        data_min = norm_params['data_min']
        data_max = norm_params['data_max']
        
        if logger:
            logger.info(f"Normalization parameters loaded from {norm_params_path}")
        
        return data_min, data_max