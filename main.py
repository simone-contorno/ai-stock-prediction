# Stock Price Prediction - Main Script

import os

# Disable oneDNN optimizations for TensorFlow
# This is a workaround for a known issue with TensorFlow and oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
import json
from src.train import train_model
from src.test import test_model
from src.predict import predict_future

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction using RNN')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'], required=True,
                        help='Mode: train a new model, test an existing model, or make future predictions')
    args = parser.parse_args()
    
    # Load configuration
    with open('config.json', 'r') as f:
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