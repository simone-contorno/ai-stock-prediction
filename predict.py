# Recursive Neural Network for Target Prediction in Market Stocks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from utils import clear_zero_values

def normalize(input_data):
    """
    Normalize the input data to the range [0, 1]
    """
    if isinstance(input_data, pd.DataFrame):
        input_min = input_data.min()
        input_max = input_data.max()
        return (input_data - input_min) / (input_max - input_min)
    else:  # For numpy arrays
        input_min = np.min(input_data)
        input_max = np.max(input_data)
        return (input_data - input_min) / (input_max - input_min)

def plotting(y_test, y_pred, y_label, n=0, x_legend="Value", y_legend="Predicted Value"):
    """
    Plot the actual vs predicted values
    """
    if n > len(y_test) or n == 0:
        n = len(y_test)

    print("y_test shape:", y_test.shape)
    print("y_pred shape:", y_pred.shape)

    plt.figure("Prediction")
    plt.plot(range(len(y_test)), y_test, color='blue', label=x_legend)
    plt.plot(range(len(y_pred)), y_pred, color='red', label=y_legend)
    plt.xlabel("Sample Number")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_future(config):
    """
    Load model and make predictions
    """
    # Configuration
    features = config['data_preparation']['features']
    target_feature = config['data_preparation']['target_feature']
    include_target = config['data_preparation']['include_target']
    days = config['data_preparation']['days']
    model_path = config['prediction']['model_path']
    
    # Load the model
    model_output = load_model(model_path)
    
    # Import the dataset
    data = pd.read_csv('.\\csv\\S&P500.csv')#[-days:] # Take the last N days
    data = data[[feat for feat in data.columns if feat != "Price" and feat != "Adj Close"]]
    
    print("Data shape:", data.shape)
    print((data==0).sum())
    
    # Clear zero values
    data = clear_zero_values(data, features)
    
    # Look to the dataset description
    print(data.describe())
    
    # Configuration
    if include_target:
        input_feature = [feat for feat in features]
    else:
        input_feature = [feat for feat in features if feat != target_feature] 
    target_feature = [feat for feat in features if feat == target_feature]
    print("Input features:", input_feature)
    print("Target feature:", target_feature)
    
    # Assign input and target data
    input_data = data[input_feature] 
    print("Input shape:", input_data.shape)
    print(input_data.head())
    
    target = data[target_feature] 
    print("Target shape:", target.shape)
    print(target.head())
    print(target.tail())
    
    # For additional analysis if needed
    target_input = data['Open']
    print("Target input shape:", target_input.shape)
    print(target_input.tail())
    
    # Normalize the data
    input_data = normalize(input_data)
    
    # Prepare data for prediction
    print("Input shape:", input_data.shape)
    
    X, y = [], []
    shift = days
    if shift > 0:
        for i in range(len(data) - shift*2):
            X.append(input_data[i:i+shift]) # Past N days
            y.append(target[i+shift:i+shift*2]) # Next N days
        X_train, y_train = np.array(X), np.array(y)
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))  # (days, val per day)
    else:
        X_train, y_train = input_data.values, target
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    
    print("X_train shape:", X_train.shape)
    
    # Make predictions
    y_pred = model_output.predict(X_train)
    print("y_pred shape:", y_pred.shape)
    
    # For visualization, we can take a subset of the data
    """
    sample_size = config['prediction']['sample_size']
    y_test = target.values[-sample_size:]
    y_pred = y_pred[-sample_size:]
    
    if len(y_test.shape) > 1 and y_test.shape[1] == 1:
        y_test = y_test.flatten()
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    """

    # Plot the results
    y_test = target.values
    plt.figure("Prediction")
    plt.plot(range(len(y_test)), y_test[:, 0], color='blue', label='Actual')
    plt.plot(range(len(y_pred)), y_pred[:, 0], color='red', label='Predicted')
    plt.xlabel("Sample Number")
    plt.ylabel(target_feature[0])
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot the results
    #plotting(y_test, y_pred, target_feature[0])
    
    # Adjust prediction with offset
    #y_pred_off = y_pred + (y_test[0] - y_pred[0])
    #plotting(y_test, y_pred_off, target_feature[0])
    
    # Evaluate the model
    if config['prediction']['evaluate']:
        print("y_test shape:", y_test_full.shape)
        print("Mean absolute error:", mean_absolute_error(y_test_full, y_pred))
    
    return y_pred

if __name__ == "__main__":
    import json
    
    # Load configuration from JSON file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Make predictions using the configuration
    predictions = predict_future(config['target_prediction'])