"""RNN model builder for stock prediction."""

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.initializers import GlorotUniform # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import regularizers # type: ignore
from typing import Dict, Any, Optional
import logging

class LSTModelBuilder:
    @staticmethod
    def build(input_shape: tuple, output_shape: int, config: Dict[str, Any], 
             logger: Optional[logging.Logger] = None) -> Sequential:
        """
        Build and compile the LSTM model.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            output_shape: Shape of output data
            config: Model configuration dictionary
            logger: Optional logger for logging information
            
        Returns:
            Compiled Keras Sequential model
        """
        # Extract model parameters from config
        lstm_units = config['training']['model']['lstm_units']
        dropout_rate = config['training']['model']['dropout_rate']
        dense_units = config['training']['model']['dense_units']
        learning_rate = config['training']['learning_rate']
        
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
            model.summary(print_fn=logger.info)
        
        return model