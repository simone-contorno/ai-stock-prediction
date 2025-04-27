# Stock Price Prediction with LSTM

This project implements Recursive Neural Networks (LSTM) for stock price prediction using historical data. The system uses LSTM (Long Short-Term Memory) architecture to predict stock prices based on historical market data.

## Project Structure

- `src/`: Source code directory
  - `models/`: Neural network model implementations
    - `lstm_model.py`: LSTM model builder implementation
  - `utils/`: Utility functions for data processing and visualization
    - `data.py`: Data loading and manipulation
    - `preprocessing.py`: Data preprocessing functions
    - `plotting.py`: Visualization utilities
    - `evaluation.py`: Model evaluation metrics
    - `logging.py`: Logging utilities
  - `train.py`: Training script implementation
  - `test.py`: Testing script implementation
  - `predict.py`: Prediction script implementation
- `csv/`: Directory containing historical stock data (e.g., S&P500.csv)
- `results/`: Directory containing trained models and their results
  - Organized by target feature (e.g., `close/`)
    - Subdirectories organized by timestamp (e.g., `2025-04-23_08-53-39/`)
      - `logs/`: Execution logs with detailed training information
      - `models/`: Saved model files (.keras) and scalers (.joblib)
      - `plots/`: Visualization plots (training history, predictions)
      - `test/`: Test results organized by timestamp
        - `logs/`: Test execution logs
        - `plots/`: Test visualization plots
        - `metrics.txt`: Detailed evaluation metrics
        - `test.csv`: Test data and predictions
      - `predict/`: Prediction results organized by timestamp
        - `logs/`: Prediction execution logs
        - `plots/`: Prediction visualization plots
        - `predictions_real.csv`: Prediction data and actual values
- `config.json`: Configuration file for model parameters and training settings
- `main.py`: Main entry point for running the application
- `download_dataset.py`: Script for downloading stock data using yfinance
- `requirements.txt`: Project dependencies

## Model Architecture

The project implements an LSTM-based neural network for time series prediction:

### Input Layer
- Accepts sequences of historical stock data with configurable features (Open, Close, Volume, High, Low)
- Sequence length determined by the `days` parameter in configuration

### LSTM Layers
- Configurable number of LSTM layers (default: 1)
- Configurable units per layer (default: 256)
- Tanh activation function
- L2 regularization on both kernel and recurrent weights (default: 0.001)
- Optional dropout for regularization (default: 0.1)

### Dense Layers
- Hidden dense layer with configurable units (default: 128)
- Tanh activation function
- L2 regularization

### Output Layer
- Single unit with linear activation for regression

### Optimization
- Configurable optimizer (Adam or SGD)
- Gradient clipping to prevent exploding gradients (clipnorm: 1.0)
- Configurable learning rate with reduction on plateau
- Huber loss function for robustness to outliers

## Training Process

The training pipeline includes:

1. **Data Preprocessing**:
   - Removal of zero values
   - Feature normalization using MinMaxScaler
   - Sequence preparation for LSTM input
   - Train/test splitting with optional shuffling

2. **Model Training**:
   - Early stopping to prevent overfitting (patience: 50 epochs)
   - Learning rate reduction on plateau (factor: 0.5, patience: 25 epochs)
   - Restoration of best weights after training
   - Configurable batch size and epochs

3. **Evaluation**:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)

4. **Visualization**:
   - Training history plots (loss curves)
   - Prediction vs. actual value plots

5. **Model Persistence**:
   - Saving trained models in .keras format
   - Saving scalers for future predictions

## Configuration

The `config.json` file contains all necessary parameters:

### General Settings
- Random seed for reproducibility
- Input CSV path
- Results directory path
- Input features (Open, Close, Volume, High, Low)
- Target feature (default: Close)
- Prediction window (days)

### Training Parameters
- Test and validation split ratios
- Training epochs and batch size
- Learning rate and patience settings
- Model architecture configuration (LSTM units, dropout rate, dense layers)
- Regularization parameters (L2 regularization, gradient clipping)
- Optimizer settings (Adam, SGD with momentum)

## Requirements

- Python 3.x
- TensorFlow >= 2.13.0
- Pandas >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- yfinance >= 0.2.28
- Keras >= 2.13.0
- Additional dependencies listed in requirements.txt

## Usage

### Downloading Stock Data

```bash
# Download S&P500 (default) data
python download_dataset.py

# Download data for a specific ticker
python download_dataset.py --ticker AAPL
```

This will download historical stock data from Yahoo Finance. You can specify a different ticker symbol using the `--ticker` argument.

### Training a New Model

```bash
python main.py --mode train
```

This will train a new model using the parameters specified in `config.json`. The trained model and related artifacts will be saved in the `results` directory.

### Testing an Existing Model

```bash
python main.py --mode test
```

This will evaluate the model specified in the `prediction.model_path` parameter of `config.json` on test data.

### Making Predictions

```bash
python main.py --mode predict
```

This will use the model specified in the `prediction.model_path` parameter to make predictions on new data.

## Customization

You can customize the model by modifying the `config.json` file:

- Change the target feature (e.g., from Close to Open)
- Adjust the prediction window (days parameter)
- Modify model architecture (LSTM layers, units, dropout)
- Tune training parameters (learning rate, batch size, epochs)
- Select different optimization strategies

## Results Organization

All results are organized in a structured directory hierarchy:

```
results/
└── [stock_ticker]/
  └── [target_feature]/
      └── [timestamp]/
          ├── logs/
          ├── models/
          ├── plots/
          ├── test/
          └── predict/
```

This organization makes it easy to compare different model configurations and track experiments over time.

## License

This project is licensed under the terms of the MIT LICENSE file included in the repository.
