"""Logging utility for stock prediction."""

import logging
import os
from typing import Dict, Optional
from datetime import datetime

class Logger:
    @staticmethod
    def setup(target_feature: str, is_training: bool = True, model_path: str = None, 
             subfolder_name: Optional[str] = None) -> logging.Logger:
        """
        Set up logging configuration with log files in appropriate subfolder.
        
        Args:
            target_feature: The target feature being predicted (used for subfolder naming)
            is_training: Whether this is for training (True) or prediction/testing (False)
            model_path: Path to the model file (required for prediction/testing)
            subfolder_name: Name of the subfolder to create for outputs (e.g. 'predict', 'test')
            
        Returns:
            Logger object configured for the current session with output_dirs attribute
        """
        # Create output directories based on operation type
        output_dirs = Logger._create_output_directories(
            target_feature, 
            is_training=is_training, 
            model_path=model_path,
            subfolder_name=subfolder_name
        )
        
        # Configure logger to use the logs directory
        log_dir = output_dirs['logs']
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logger
        log_file = os.path.join(log_dir, "execution.log")
        
        # Create logger
        logger = logging.getLogger(f"stock_prediction_{target_feature}")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized for {target_feature} {'training' if is_training else 'prediction'}")
        
        # Store output directories in the logger object for easy access
        logger.output_dirs = output_dirs
        
        return logger

    @staticmethod
    def _create_output_directories(target_feature: str, is_training: bool = True, 
                                 model_path: str = None, subfolder_name: Optional[str] = None) -> Dict[str, str]:
        """
        Create necessary output directories for models, logs, and visualizations.
        For training: creates in results/feature/timestamp/
        For prediction/testing: creates next to model directory in specified subfolder
        
        Args:
            target_feature: The target feature being predicted
            is_training: Whether this is for training (True) or prediction/testing (False)
            model_path: Path to the model file (required for prediction/testing)
            subfolder_name: Name of the subfolder to create for outputs (e.g. 'predict', 'test')
            
        Returns:
            Dictionary with paths to created directories
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if is_training:
            # For training: create in results/feature/timestamp/
            target_dir = target_feature.lower()
            execution_dir = os.path.join('results', target_dir, timestamp)
            
            dirs = {
                'models': os.path.join(execution_dir, 'models'),
                'logs': os.path.join(execution_dir, 'logs'),
                'plots': os.path.join(execution_dir, 'plots'),
                'results': execution_dir
            }
        else:
            # For prediction/testing: create next to model directory
            if not model_path:
                raise ValueError("model_path is required for prediction/testing")
                
            model_dir = os.path.dirname(os.path.abspath(model_path))
            # Use provided subfolder name or default to 'predict'
            output_subfolder = subfolder_name if subfolder_name else 'predict'
            predictions_dir = os.path.join(os.path.dirname(model_dir), output_subfolder, timestamp)
            
            dirs = {
                'logs': os.path.join(predictions_dir, 'logs'),
                'plots': os.path.join(predictions_dir, 'plots'),
                'results': predictions_dir
            }
        
        # Create directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return dirs