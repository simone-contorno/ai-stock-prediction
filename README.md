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
  - Subdirectories organized by timestamp containing:
    - `logs/`: Execution logs
    - `models/`: Saved model files
    - `plots/`: Visualization plots
- `config.json`: Configuration file for model parameters and training settings
- `main.py`: Main entry point for running the application
- `download_dataset.py`: Script for downloading stock data using yfinance
- `requirements.txt`: Project dependencies

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
- Additional dependencies listed in requirements.txt

## Usage

### Downloading Stock Data

```bash
python download_dataset.py
```

This will download S&P500 historical data by default. Edit the script to download data for other stocks.

### Training a New Model

```bash
python main.py --mode train
```

### Testing an Existing Model

```bash
python main.py --mode test
```

### Making Predictions

```bash
python main.py --mode predict
```

### Custom Configuration

You can modify the `config.json` file to customize model parameters, input features, and training settings.

## Model Architecture

The model uses an LSTM-based architecture with:
- Configurable number of LSTM layers (default: 1)
- LSTM layers with 256 units by default
- Dropout regularization (10% by default)
- Dense layers with 128 units
- L2 regularization and gradient clipping
- Configurable activation functions and optimizers
- All hyperparameters configurable via config.json
