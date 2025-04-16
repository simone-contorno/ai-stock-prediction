"""Training script for Stock Price Prediction RNN model."""

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from datetime import datetime
import os
from typing import Dict, Any

from src.utils import (
    Logger,
    Plotter,
    DataPreprocessor,
    DataLoader,
    ModelEvaluator
)
from src.models.lstm_model import RNNModelBuilder

def train_model(config: Dict[str, Any]) -> Sequential:
    """Train a RNN model for stock price prediction."""
    # Extract configuration parameters
    features = config['data_preparation']['features']
    target_feature = config['data_preparation']['target_feature']
    days = config['data_preparation']['days']
    test_size = config['training']['test_size']
    validation_split = config['training']['validation_split']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    patience = config['training']['patience']
    
    # Setup logging - explicitly set is_training=True
    logger = Logger.setup(target_feature, is_training=True)
    output_dirs = logger.output_dirs
    
    logger.info("Starting model training with configuration:")
    logger.info(f"Features: {features}, Target: {target_feature}, Days: {days}")
    
    try:
        # 1. Load raw data
        data = DataLoader.load_raw_data(config, logger, days, is_training=True)
        
        # 2. Clear zero values from raw data
        data = DataPreprocessor.clear_zero_values(data, features, logger)
        
        # 3. Split into input and target features
        input_data, target_data = DataLoader.prepare_features(data, config, logger)
        
        # 4. Normalize input data
        input_normalized, input_min, input_max = DataPreprocessor.normalize_data(input_data)
        logger.info("Data normalized")
        
        # Save normalization parameters
        DataLoader.save_normalization_params(input_min, input_max, output_dirs['models'], logger)
        
        # 5. Prepare sequence data for RNN
        X_sequence, y_sequence = DataPreprocessor.prepare_sequence_data(
            input_normalized, target_data, days, logger
        )
        
        # 6. Split into training and test sets
        X_train, X_test, y_train, y_test = DataPreprocessor.split_train_test(
            X_sequence, y_sequence, test_size, shuffle=True, random_state=3, logger=logger
        )
        
        # Build the model using RNNModelBuilder
        model = RNNModelBuilder.build(
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
        Plotter.plot_training_history(history.history, save_path=history_plot_path)
        logger.info(f"Training history plot saved to {history_plot_path}")
        
        # Evaluate the model on test data
        logger.info("Evaluating model on test data...")
        y_pred = model.predict(X_test)
        
        # Plot and save predictions
        predictions_plot_path = os.path.join(output_dirs['plots'], 'predictions.png')
        Plotter.plot_predictions(y_test, y_pred, features[0], save_path=predictions_plot_path)
        logger.info(f"Predictions plot saved to {predictions_plot_path}")
        
        # Evaluate predictions
        ModelEvaluator.evaluate_predictions(y_test, y_pred, logger)
        
        # Save the model
        parent_folder_name = os.path.basename(output_dirs['results'])
        model_filename = f"{parent_folder_name}_rnn_{target_feature.lower()}_{days}.keras"
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