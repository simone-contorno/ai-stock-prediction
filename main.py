# Stock Price Prediction - Main Script

import argparse
import json
import os
from train import train_model
from test import predict_future

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction using RNN')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                        help='Mode: train a new model or predict using existing model')
    parser.add_argument('--type', type=str, choices=['feature', 'target'], required=True,
                        help='Type: feature prediction or target prediction')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Execute based on mode and type
    if args.mode == 'train' and args.type == 'feature':
        print("Training feature prediction model...")
        model = train_model(config['feature_prediction'])
        print("Training completed.")
    
    elif args.mode == 'predict' and args.type == 'target':
        print("Making predictions using target prediction model...")
        predictions = predict_future(config['target_prediction'])
        print("Prediction completed.")
    
    else:
        print(f"The combination of mode={args.mode} and type={args.type} is not supported yet.")

if __name__ == "__main__":
    main()