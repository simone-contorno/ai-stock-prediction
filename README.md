# Stock Price Prediction with RNN

This project implements Recursive Neural Networks (RNN) for stock price prediction using historical data. The system uses LSTM (Long Short-Term Memory) architecture to predict stock prices based on historical market data.

## Project Structure

- `src/`: Source code directory
  - `models/`: Neural network model implementations
  - `utils/`: Utility functions for data processing and visualization
- `csv/`: Directory containing historical stock data (Apple.csv, Gold.csv, S&P500.csv)
- `results/`: Directory containing trained models and their results
  - `close/`: Results for close price predictions
    - Subdirectories organized by timestamp containing:
      - `logs/`: Execution logs
      - `models/`: Saved model files
      - `plots/`: Visualization plots
- `config.json`: Configuration file for model parameters and training settings
- `main.py`: Main entry point for running the application
- `download_dataset.py`: Script for downloading stock data
- `requirements.txt`: Project dependencies

## Configuration

The `config.json` file contains all necessary parameters:

### General Settings
- Random seed for reproducibility
- Input CSV path
- Results directory path
- Input features (Open, Close, Volume)
- Target feature
- Prediction window (days)

### Training Parameters
- Test and validation split ratios
- Training epochs and batch size
- Learning rate and patience settings
- Model architecture configuration (LSTM units, dropout rate, dense layers)

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

### Training a New Model

```bash
python main.py --mode train --type feature
```

### Making Predictions

```bash
python main.py --mode predict --type target
```

### Custom Configuration

```bash
python main.py --mode train --type feature --config custom_config.json
```

## Model Architecture

The model uses an LSTM-based architecture with:
- LSTM layers with 256 units
- Dropout regularization (10%)
- Dense layers with 128 units
- Configurable hyperparameters via config.json
