# Recursive Neural Network for Feature Prediction in Market Stocks

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import os
from utils import *

def train_model(config):
    # Configuration
    features = config['data_preparation']['features']
    target_feature = config['data_preparation']['target_feature']
    include_target = config['data_preparation']['include_target']
    days = config['data_preparation']['days']
    shift = config['data_preparation']['shift']
    test_size_ = config['training']['test_size']
    validation_split_ = config['training']['validation_split']
    epochs_ = config['training']['epochs']
    
    # Preparing data
    # Import the dataset
    if days > 0:
        data = pd.read_csv('.\\csv\\S&P500.csv')[:-days]
    else:
        data = pd.read_csv('.\\csv\\S&P500.csv')
    data = data[[feat for feat in data.columns if feat != "Price" and feat != "Adj Close"]]
    print("Data shape:", data.shape)
    
    # Clear zero values
    data = clear_zero_values(data, features)
    
    # Look to the dataset description
    print(data.describe())
    
    # Prepare training and test set
    # Create input and target feature sets
    if include_target:
        input_feature = [feat for feat in features]
    else:
        input_feature = [feat for feat in features if feat != target_feature] 
    target_feature = [feat for feat in features if feat == target_feature]
    print("Input features:", input_feature)
    print("Target feature:", target_feature)
    
    # Create input and target data sets
    input = data[input_feature] 
    target = data[target_feature] 
    
    print("Input shape:", input.shape)
    print(input.head())
    
    print("Target shape:", target.shape)
    print(target.head())
    
    # Normalize the input data
    input_min = input.min()
    input_max = input.max()
    input = (input - input_min) / (input_max - input_min)
    print(input.head())
    
    # Create the Recursive Neural Network (RNN)
    # Prepare the data for the RNN
    X, y = [], []
    if shift > 0:
        for i in range(len(data) - shift*2):
            X.append(input[i:i+shift]) # Past N days
            y.append(target[i+shift:i+shift*2]) # Next N days
        X_train, y_train = np.array(X), np.array(y)
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))  # (samples, 30)
    else:
        X_train, y_train = input.values, target
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    
    if test_size_ > 0.0:
        X_test = X_train[int(-len(X_train)*test_size_)+1:]
        y_test = y_train[int(-len(y_train)*test_size_)+1:]
    
        X_train = X_train[:int(len(X_train)*(1-test_size_))]
        y_train = y_train[:int(len(y_train)*(1-test_size_))]
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=test_size_, shuffle=True, random_state=3)
        
    else:
        X_test = X_train
        y_test = y_train
        
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_train example:", X_train[0][0][0])
    print("y_train example:", y_train[0][0])
    
    # Create the RNN model
    model = Sequential()
    model.add(Input((X_train.shape[1], X_train.shape[2])))
    
    factor=1
    model.add(LSTM(256+factor, activation='tanh', kernel_initializer=GlorotUniform(), return_sequences=False))
    #model.add(Dropout(0.1))
    
    #model.add(LSTM(128, activation='tanh', return_sequences=False))
    #model.add(Dropout(0.2))
    
    #model.add(LSTM(64, activation='tanh', return_sequences=False))
    #model.add(Dropout(0.2))
    
    model.add(BatchNormalization())
    model.add(Dense(128*factor, activation='relu'))
    
    model.add(Dense(y_train.shape[1], activation='linear'))
    
    # Compile the model
    lr = config['training']['learning_rate']
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mae')
    
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=config['training']['patience'], 
                                   restore_best_weights=True)

    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                                patience=5, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.0001)

    start = datetime.now()
    # Train the model and get history
    history = model.fit(
        X_train, y_train,
        epochs=epochs_,
        batch_size=config['training']['batch_size'],
        validation_split=validation_split_,
        callbacks=[early_stopping, learning_rate_reduction],
        verbose=1
    )
    
    # Get the last 5 values of loss and validation loss
    last_losses = history.history['loss'][-5:]
    last_val_losses = history.history['val_loss'][-5:]
    print("Last 5 losses:", last_losses)
    print("Last 5 validation losses:", last_val_losses)
    
    # Check if early stopping occurred
    if len(history.history['loss']) < epochs_:
        print("\nTraining stopped early!")
        
        # Check if it was due to early stopping (validation loss)
        if last_val_losses[-1] >= last_val_losses[-2]:
            print("Reason: Early stopping triggered - Validation loss stopped improving")
        # Check if it was due to learning rate reduction
        elif last_losses[-1] >= last_losses[-2]:
            print("Reason: Learning rate reduction - Training loss stopped improving")
    else:
        print("\nTraining completed for all epochs")
    end = datetime.now()
    print("Training time:", end - start, "seconds")
    
    # Plot the training history
    plt.figure("Training History")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Plot the results
    plt.figure("Prediction")
    plt.plot(range(len(y_test)), y_test[:, 0], color='blue', label='Actual')
    plt.plot(range(len(y_pred)), y_pred[:, 0], color='red', label='Predicted')
    plt.xlabel("Sample Number")
    plt.ylabel(target_feature[0])
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Save the model
    sub_folder = target_feature[0].lower()
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create directory if it doesn't exist
    os.makedirs(f'models\\{sub_folder}', exist_ok=True)
    
    model.save(f'models\\{sub_folder}\\{current_date_time}_rnn_{sub_folder}_{shift}.keras')
    
    return model

if __name__ == "__main__":
    import json
    
    # Load configuration from JSON file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Train the model using the configuration
    model = train_model(config['feature_prediction'])