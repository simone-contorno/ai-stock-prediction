# Recursive Neural Network for Feature Prediction in Market Stocks

import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from datetime import datetime
import os
import logging
from typing import Dict, Any, Optional

from utils import (
    setup_logging,
    clear_zero_values,
    normalize_data,
    prepare_sequence_data,
    split_train_test,
    plot_training_history,
    plot_predictions,
    save_normalization_params
)

def build_model(input_shape: tuple, output_shape: int, config: Dict[str, Any], 
               logger: Optional[logging.Logger] = None) -> Sequential:
    """
    Build and compile the RNN model.
    
    Args:
        input_shape: Shape of input data (time_steps, features)
        output_shape: Shape of output data
        config: Model configuration dictionary
        logger: Logger for logging information
        
    Returns:
        Compiled Keras Sequential model
    """
    # Extract model parameters from config
    lstm_units = config['model'].get('lstm_units', 256)
    dropout_rate = config['model'].get('dropout_rate', 0.0)
    dense_units = config['model'].get('dense_units', 128)
    learning_rate = config['training'].get('learning_rate', 0.001)
    
    if logger:
        logger.info(f"Building model with LSTM units: {lstm_units}, Dense units: {dense_units}")
    
    # Create the RNN model
    model = Sequential()
    model.add(Input(input_shape))
    
    # LSTM layer
    l2 = 0.001
    model.add(LSTM(lstm_units, 
                   activation='tanh', 
                   kernel_initializer=GlorotUniform(), 
                   kernel_regularizer=regularizers.l2(l2),
                   recurrent_regularizer=regularizers.l2(l2),
                   #bias_regularizer=regularizers.l2(l2),
                   return_sequences=False))
    
    # Dropout for generalization
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate))
    
    # Normalization and dense layers
    model.add(BatchNormalization())
    model.add(Dense(dense_units, 
                    activation='relu', 
                    kernel_regularizer=regularizers.l2(l2)))
    
    # Output layer
    model.add(Dense(output_shape, 
                    activation='linear'))
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mae')
    
    if logger:
        # Create a custom print function that handles Unicode characters
        def safe_info(msg):
            try:
                logger.info(msg)
            except UnicodeEncodeError:
                # If Unicode error occurs, replace problematic characters
                logger.info(msg.encode('ascii', 'replace').decode('ascii'))
        
        model.summary(print_fn=safe_info)
    
    return model

def train_model(config: Dict[str, Any]) -> Sequential:
    """
    Train a RNN model for stock price prediction.
    
    Args:
        config: Configuration dictionary with training parameters
        
    Returns:
        Trained Keras model
    """
    # Extract configuration parameters
    features = config['data_preparation']['features']
    target_feature = config['data_preparation']['target_feature']
    include_target = config['data_preparation']['include_target']
    days = config['data_preparation']['days']
    shift = config['data_preparation']['shift']
    test_size = config['training']['test_size']
    validation_split = config['training']['validation_split']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    patience = config['training']['patience']
    
    # Setup logging and output directories
    # Note: setup_logging now calls create_output_directories internally
    logger = setup_logging(target_feature)
    output_dirs = logger.output_dirs  # Access the output directories created during logging setup
    
    logger.info("Starting model training with configuration:")
    logger.info(f"Features: {features}, Target: {target_feature}, Shift: {shift} days")
    
    # Load and prepare data
    logger.info("Loading dataset...")
    try:
        if days > 0:
            data = pd.read_csv('.\\csv\\S&P500.csv')[:-days]
        else:
            data = pd.read_csv('.\\csv\\S&P500.csv')
        
        # Filter out Price and Adj Close columns
        data = data[[feat for feat in data.columns if feat != "Price" and feat != "Adj Close"]]
        logger.info(f"Dataset loaded with shape: {data.shape}")
        
        # Clear zero values
        data = clear_zero_values(data, features, logger)
        
        # Log dataset statistics
        logger.info("Dataset statistics:")
        logger.info(data.describe().to_string())
        
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
        
        # Normalize the input data
        input_normalized, input_min, input_max = normalize_data(input_data)
        logger.info("Data normalized")
        
        # Save normalization parameters
        save_normalization_params(input_min, input_max, output_dirs['models'], logger)
        
        # Prepare sequence data for RNN
        X_sequence, y_sequence = prepare_sequence_data(input_normalized, target_data, shift, logger)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = split_train_test(
            X_sequence, y_sequence, test_size, shuffle=True, random_state=3, logger=logger
        )
        
        # Build the model
        model = build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=y_train.shape[1],
            config=config,
            logger=logger
        )
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='loss',
            patience=5,
            verbose=1,
            factor=0.5,
            min_lr=0.0001
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
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Analyze training results
        last_losses = history.history['loss'][-5:]
        last_val_losses = history.history['val_loss'][-5:]
        logger.info(f"Final training loss: {last_losses[-1]:.6f}")
        logger.info(f"Final validation loss: {last_val_losses[-1]:.6f}")
        
        # Check if early stopping occurred
        if len(history.history['loss']) < epochs:
            logger.info("Training stopped early!")
            
            if last_val_losses[-1] >= last_val_losses[-2]:
                logger.info("Reason: Early stopping triggered - Validation loss stopped improving")
            elif last_losses[-1] >= last_losses[-2]:
                logger.info("Reason: Learning rate reduction - Training loss stopped improving")
        else:
            logger.info("Training completed for all epochs")
        
        # Plot and save training history
        history_plot_path = os.path.join(output_dirs['plots'], 'training_history.png')
        plot_training_history(history.history, save_path=history_plot_path)
        logger.info(f"Training history plot saved to {history_plot_path}")
        
        # Evaluate the model on test data
        logger.info("Evaluating model on test data...")
        y_pred = model.predict(X_test)
        
        # Plot and save predictions
        predictions_plot_path = os.path.join(output_dirs['plots'], 'predictions.png')
        plot_predictions(y_test, y_pred, target_features[0], save_path=predictions_plot_path)
        logger.info(f"Predictions plot saved to {predictions_plot_path}")
        
        # Save the model
        # Use a consistent naming without timestamp since all files are already in a timestamped folder
        # get the parent output_dirs['models']
        parent_output_dir = os.path.dirname(output_dirs['models'])
        parent_folder_name = os.path.basename(parent_output_dir)
        model_filename = f"{parent_folder_name}_rnn_{target_features[0].lower()}_{shift}.keras"
        model_path = os.path.join(output_dirs['models'], model_filename)
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import json
    
    # Load configuration from JSON file
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Train the model using the configuration
        model = train_model(config['feature_prediction'])
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")