"""LSTM model builder for stock price prediction.

This module provides a builder class for creating and configuring LSTM neural network models
specifically designed for time series stock price prediction. The LSTModelBuilder class
implements a factory method pattern to construct Keras Sequential models with configurable:

- LSTM layers and units
- Dropout for regularization
- Dense layers
- Activation functions
- Optimizers (Adam or SGD)
- Loss functions
- Regularization parameters

The architecture is optimized for time series forecasting with options to tune
hyperparameters through the configuration dictionary.
"""

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.initializers import GlorotUniform # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # type: ignore
from tensorflow.keras import regularizers # type: ignore
from typing import Dict, Any, Optional
import logging

class LSTModelBuilder:
    @staticmethod
    def build(input_shape: tuple, output_shape: int, config: Dict[str, Any], 
             logger: Optional[logging.Logger] = None) -> Sequential:
        """
        Build and compile an LSTM model for stock price prediction.
        
        This method constructs a Sequential model with one or more LSTM layers
        followed by Dense layers. The architecture includes regularization techniques
        such as dropout and L2 regularization to prevent overfitting.
        
        Args:
            input_shape: Shape of input data as tuple (time_steps, features)
                         where time_steps is the sequence length and features is the number of input features
            output_shape: Number of output units (typically 1 for single value prediction)
            config: Model configuration dictionary containing hyperparameters under config['training']['model']
                   including lstm_layers, lstm_units, dropout_rate, etc.
            logger: Optional logger for logging model architecture and compilation details
            
        Returns:
            Compiled Keras Sequential model ready for training
        """
        # Extract model parameters from config
        lstm_layers = config['training']['model']['lstm_layers']
        lstm_units = config['training']['model']['lstm_units']
        dropout_rate = config['training']['model']['dropout_rate']
        dense_units = config['training']['model']['dense_units']
        learning_rate = config['training']['learning_rate']
        l2_reg = config['training']['model']['l2_reg']
        lstm_activation = config['training']['model']['lstm_activation']
        dense_activation = config['training']['model']['dense_activation']
        optimizer = config['training']['model']['optimizer']
        clipnorm = config['training']['model']['clipnorm']
        momentum = config['training']['model']['momentum']
        loss = config['training']['model']['loss']
        
        if logger:
            logger.info(f"Building model with LSTM units: {lstm_units}, Dense units: {dense_units}")
        
        # Initialize the Sequential model
        # Sequential is a linear stack of layers where each layer has exactly one input and one output
        model = Sequential()
        
        # Add input layer with specified shape (time_steps, features)
        # This explicitly defines the input shape expected by the model
        model.add(Input(input_shape))
        
        for i in range(lstm_layers):
            # Add LSTM layers with decreasing units for each subsequent layer
            # This creates a funnel architecture that gradually reduces dimensionality
            units = int(lstm_units/(i+1))  # Decrease units for deeper layers
            
            # LSTM layer configuration:
            model.add(LSTM(
                units=units,  # Number of LSTM cells/units in this layer
                activation=lstm_activation,  # Activation function within LSTM cells (typically tanh)
                kernel_initializer=GlorotUniform(),  # Weight initialization method (aka Xavier initialization)
                kernel_regularizer=regularizers.l2(l2_reg),  # L2 regularization on weights to prevent overfitting
                recurrent_regularizer=regularizers.l2(l2_reg),  # L2 regularization on recurrent weights
                # return_sequences=True means output the full sequence for stacked LSTM layers
                # return_sequences=False means output only the final state for the last layer
                return_sequences=True if i == 0 and lstm_layers > 1 else False
            ))
            
            # Dropout for generalization
            # Randomly sets a fraction of input units to 0 during training to prevent overfitting
            if dropout_rate > 0.0:
                model.add(Dropout(dropout_rate))
        
        # Normalization and dense layers
        # BatchNormalization is commented out but could be used to normalize layer outputs
        #model.add(BatchNormalization())
        
        # Hidden dense layer with regularization
        model.add(Dense(dense_units, 
                       activation=dense_activation, 
                       kernel_regularizer=regularizers.l2(l2_reg)))
        
        # Output layer with linear activation for regression task
        model.add(Dense(output_shape, 
                       activation='linear'))
        
        # Compile the model with specified optimizer and loss function
        if optimizer.lower() == 'adam':
            # Adam optimizer: Adaptive learning rate optimization algorithm
            optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
            logger.info(f"Using Adam optimizer with learning rate: {learning_rate}")
        elif optimizer.lower() == 'sgd':
            # SGD optimizer: Stochastic gradient descent with momentum
            optimizer = SGD(learning_rate=learning_rate, clipnorm=clipnorm, momentum=momentum)
            logger.info(f"Using SGD optimizer with learning rate: {learning_rate}, momentum: {momentum}")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Use 'adam' or 'sgd'.")
        
        # Compile the model with the selected optimizer and loss function
        model.compile(optimizer=optimizer, loss=loss)
        
        # Log model summary if logger is provided
        if logger:
            model.summary(print_fn=logger.info)
        
        return model