"""
Stock Price Prediction - Main Script

This is the entry point for the stock price prediction application.
The application supports three modes of operation:
- Training: Train a new LSTM model on historical stock data
- Testing: Evaluate an existing model on test data
- Prediction: Use an existing model to predict future stock prices
"""

import os

# Disable oneDNN optimizations for TensorFlow
# This is a workaround for a known issue with TensorFlow and oneDNN optimizations
# that can cause crashes or performance issues on some systems
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Suppress TensorFlow INFO messages, showing only warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
import json
from src.train import train_model
from src.test import test_model
from src.predict import predict_future

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction using LSTM')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'], required=True,
                        help='Mode: train a new model, test an existing model, or make future predictions')
    args = parser.parse_args()
    
    # Load configuration
    current_folder = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(current_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Execute based on mode and type
    if args.mode == 'train':
        print("Training model...")
        train_model(config)
        print("Training completed.")
    
    elif args.mode == 'test':
        print("Testing model...")
        test_model(config)
        print("Testing completed.")
    
    elif args.mode == 'predict':
        print("Predicting future values...")
        predict_future(config)
        print("Prediction completed.")
    
    else:
        print(f"Mode not recognized: {args.mode}. Please use 'train', 'test', or 'predict'.")

if __name__ == "__main__":
    main()