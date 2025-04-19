""" Training script for Stock Price Prediction model. """

import os

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from datetime import datetime
from typing import Dict, Any
from joblib import dump

from src.utils import (
    Logger,
    Plotter,
    DataPreprocessor,
    DataLoader,
    ModelEvaluator
)
from src.models.lstm_model import LSTModelBuilder

def train_model(config: Dict[str, Any]) -> Sequential:
    """
    Train a model for stock price prediction.
    
    Args:
        config (dict): Configuration parameters for training.
        
    Returns:
        Sequential: Trained model.
    """

    # Extract configuration parameters
    input_features = config['general']['input_features']
    target_feature = config['general']['target_feature']
    days = config['general']['days']
    test_size = config['training']['test_size']
    validation_split = config['training']['validation_split']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    patience = config['training']['patience']
    verbose = config['training']['verbose']
    save_model = config['training']['save_model']
    random_seed = config['general']['random_seed']
    shuffle = config['training']['shuffle']
    restore_best_weights = config['training']['restore_best_weights']
    learning_rate_reduction = config['training']['learning_rate_reduction']
    learning_rate_min = config['training']['learning_rate_min']
    
    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Setup logging - explicitly set is_training=True
    logger = Logger.setup(target_feature, is_training=True)
    output_dirs = logger.output_dirs
    
    logger.info("Starting model training with configuration:")
    logger.info(f"Features: {input_features}, Target: {target_feature}, Days: {days}")
    
    try:
        # Load raw data
        data = DataLoader.load_raw_data(config, logger, is_training=True)
        print("Loaded data head:\n", data.head())
        print("Loaded data tail:\n", data.tail())
        
        # Clear zero values from raw data
        data = DataPreprocessor.clear_zero_values(data, logger)
        print("Cleared data head:\n", data.head())
        print("Cleared data tail:\n", data.tail())

        # Split into input and target features
        input_data, target_data = DataLoader.prepare_features(data, config, logger)
        print("Input head:\n", input_data.head(), target_data.head())
        print("Input tail:\n", input_data.tail(), target_data.tail())
        
        # Normalize input data
        input_normalized, target_normalized, scaler_x, scaler_y = DataPreprocessor.normalize_data(input_data, target_data)
        #logger.info(f"Input data normalized:\n {input_normalized.describe()}")
        #logger.info(f"Target data normalized:\n {target_normalized.describe()}")
        #print("Normalized head:\n", input_normalized.head())
        #print("Normalized tail:\n", input_normalized.tail())
        
        # Save scalers
        dump(scaler_x, os.path.join(output_dirs['models'], 'scaler_x.joblib'))
        dump(scaler_y, os.path.join(output_dirs['models'], 'scaler_y.joblib'))
        #DataLoader.save_normalization_params(scaler_x.data_min_, scaler_x.data_max_, scaler_y.data_min_, scaler_y.data_max_, output_dirs['models'], logger)
        
        # Prepare sequence data for LSTM
        X_sequence, y_sequence = DataPreprocessor.prepare_sequence_data(
            days, input_normalized, target_normalized, logger
        )
        print("Sequence head:\n", X_sequence[:5], y_sequence[:5])
        print("Sequence tail:\n", X_sequence[-5:], y_sequence[-5:])

        # Split into training and test sets
        X_train, X_test, y_train, y_test = DataPreprocessor.split_train_test(
            X_sequence, y_sequence, test_size, shuffle=shuffle, random_state=random_seed, logger=logger
        )
        print("Train head:\n", X_train[:5], y_train[:5])
        print("Train tail:\n", X_train[-5:], y_train[-5:])
        print("Test head:\n", X_test[:5], y_test[:5])
        print("Test tail:\n", X_test[-5:], y_test[-5:])

        logger.info(f"Train-test split --> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Build the model using LSTModelBuilder
        model = LSTModelBuilder.build(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=y_train.shape[1],
            config=config,
            logger=logger
        )
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience['early_stopping'],
            verbose=verbose['early_stopping'],
            restore_best_weights=restore_best_weights
        )
        
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='loss',
            patience=patience['learning_rate_reduction'],
            verbose=verbose['learning_rate_reduction'],
            factor=learning_rate_reduction,
            min_lr=learning_rate_min
        )
        
        # Train the model
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}...")
        start_time = datetime.now()
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, learning_rate_reduction],
            verbose=verbose['fit'],
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Plot and save training history
        history_plot_path = os.path.join(output_dirs['plots'], 'training_history.png')
        Plotter.plot_training_history(history.history, save_path=history_plot_path)
        logger.info(f"Training history plot saved to {history_plot_path}")
        
        # Evaluate the model on test data
        if test_size > 0:
            logger.info("Evaluating model on test data...")
            y_pred = model.predict(X_test)

            # Rescale the output
            y_pred = scaler_y.inverse_transform(y_pred)
            y_test = scaler_y.inverse_transform(y_test)

            # Align the output with the target feature
            y_pred = y_pred[days:]
            y_test = y_test[:-days]

            logger.info(f"Predictions shape: {y_pred.shape}")
            logger.info(f"Test shape: {y_test.shape}")

            # Plot and save predictions
            predictions_plot_path = os.path.join(output_dirs['plots'], 'training_predictions.png')
            Plotter.plot_predictions(target_feature, y_pred, y_test, save_path=predictions_plot_path)
            logger.info(f"Predictions plot saved to {predictions_plot_path}")
        
            # Evaluate predictions
            ModelEvaluator.evaluate_predictions(y_test, y_pred, logger)
        
        # Save the model
        if save_model:
            parent_folder_name = os.path.basename(output_dirs['results'])
            model_filename = f"{parent_folder_name}_lstm_{target_feature.lower()}_{days}.keras"
            model_path = os.path.join(output_dirs['models'], model_filename)
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":    
    # Load configuration from JSON file
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        # Train the model using the configuration
        model = train_model(config)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")