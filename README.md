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

The default model uses an LSTM-based architecture with the following components:

### Input Layer
- Input shape: (time_steps, features) where time_steps is the sequence length and features is the number of input features
- The input layer explicitly defines the expected shape for the model

### LSTM Layers
- Configurable number of LSTM layers (default: 1)
- Default configuration: 256 units in the first layer
- Funnel architecture that gradually reduces dimensionality in deeper layers
- Tanh activation function by default
- Weight initialization using GlorotUniform (Xavier initialization)
- L2 regularization applied to both kernel and recurrent weights to prevent overfitting

### Regularization
- Dropout layer after LSTM (default rate: 10%)
- L2 regularization on weights (configurable strength)
- Gradient clipping to prevent exploding gradients

### Dense Layers
- Hidden dense layer with 128 units and ReLU activation
- L2 regularization applied to weights

### Output Layer
- Dense layer with linear activation for regression task
- Single output unit for predicting the target feature

### Compilation
- Configurable optimizer: Adam (default) or SGD with momentum
- Default loss function: Huber
- Default Learning rate: 0.001 

## Training Process

The training process includes several optimization techniques:

### Early Stopping
- Monitors validation loss to prevent overfitting
- Stops training when validation loss stops improving
- Configurable patience (number of epochs with no improvement before stopping)
- Option to restore best weights from the epoch with the best validation loss

### Learning Rate Reduction
- Reduces learning rate when training plateaus
- Monitors training loss
- Configurable patience for learning rate reduction
- Configurable reduction factor
- Minimum learning rate to prevent too small steps

### Data Preprocessing
- Normalization of input and target features
- Sequence data preparation for LSTM
- Train/test splitting with optional shuffling

### Model Evaluation
- Comprehensive metrics calculation
- Visualization of training history
- Prediction plots comparing actual vs. predicted values

All hyperparameters are configurable via the `config.json` file, allowing for easy experimentation with different model architectures and training strategies.
