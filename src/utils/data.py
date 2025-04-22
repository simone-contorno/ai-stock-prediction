""" Data loading and saving utilities for stock prediction. """

import pandas as pd
import os
import logging
import re
from typing import Dict, Tuple, Union, Optional, Any
from joblib import load

class DataLoader:
    @staticmethod
    def load_raw_data(config: Dict[str, Any], logger: logging.Logger, is_training: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load raw data from CSV file.
        
        Args:
            config: Configuration dictionary
            logger: Logger for logging information
            days: Number of days to predict
            is_training: Whether this is for training (load all data) or prediction (load last N days)
            
        Returns:
            Tuple of (DataFrame with features, Series with dates)
        """

        csv_path = config['general']['csv_path']
        test_size = config['training']['test_size']

        input_features = [feat for feat in config['general']['input_features']]
        target_feature = [config['general']['target_feature']]
        
        # Load dataset
        logger.info("Loading dataset...")
        full_data = pd.read_csv(csv_path)
        
        # Extract dates by detecting column with datetime format pattern (e.g., 1927-12-30 00:00:00+00:00)
        dates = None
        # Look for a column with datetime format pattern
        for col in full_data.columns:
            # Check first row to see if it matches a datetime pattern
            if full_data[col].iloc[0] and isinstance(full_data[col].iloc[0], str) and re.search(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}', str(full_data[col].iloc[0])):
                # Found a column with datetime format
                dates = full_data[col]
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(dates):
                    dates = pd.to_datetime(dates)
                # Extract only the date part (YYYY-MM-DD)
                dates = dates.dt.strftime('%Y-%m-%d')
                break
        
        # For prediction/testing, load only the test size
        if is_training == False:
            slice_idx = -int(len(full_data)*test_size)
            full_data = full_data[slice_idx:]
            if dates is not None:
                dates = dates[slice_idx:]
            
        data = full_data[[feat for feat in full_data.columns if feat in input_features or feat in target_feature]]
        logger.info(f"Dataset loaded with shape: {data.shape}")
        logger.info(f"Zero values per column: {(data==0).sum().to_dict()}")
        
        return data, dates

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

        input_features = [feat for feat in config['general']['input_features']]
        target_feature = [config['general']['target_feature']]

        logger.info(f"Input features: {input_features}")
        logger.info(f"Target feature: {target_feature}")
        
        # Create input and target datasets
        input_data = data[input_features]
        target_data = data[target_feature]
        
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

        # Load the scalers for normalization
        if os.path.exists(os.path.join(model_dir, 'scaler_x.joblib')):
            scaler_x = load(os.path.join(model_dir, 'scaler_x.joblib'))
        else:
            if logger:
                logger.warning(f"Normalization parameters files not found at {os.path.join(model_dir, 'scaler_x.joblib')}")
            return None, None

        if os.path.exists(os.path.join(model_dir, 'scaler_y.joblib')):
            scaler_y = load(os.path.join(model_dir, 'scaler_y.joblib'))
        else:
            if logger:
                logger.warning(f"Normalization parameters files not found at {os.path.join(model_dir, 'scaler_y.joblib')}")
            return None, None
        
        if logger:
            logger.info(f"Normalization parameters loaded from {model_dir}")
        
        return scaler_x, scaler_y