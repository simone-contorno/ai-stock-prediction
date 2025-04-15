# Stock Price Prediction with RNN

This project implements Recursive Neural Networks (RNN) for stock price prediction using historical data. It provides two main functionalities:

1. **Feature Prediction**: Predicts future values of a specific feature (e.g., Open price) based on historical data
2. **Target Prediction**: Uses pre-trained models to predict future values of a target feature (e.g., Close price)

## Project Structure

- `rnn_feature_prediction.py`: Script for training RNN models to predict stock features
- `rnn_target_prediction.py`: Script for making predictions using trained models
- `config.json`: Configuration file containing parameters for both prediction types
- `main.py`: Command-line interface for running the prediction models
- `csv/`: Directory containing historical stock data
- `models/`: Directory where trained models are saved

## Configuration

The `config.json` file contains parameters for both feature prediction and target prediction:

- **Data Preparation**: Features to use, target feature, number of days to predict, etc.
- **Training**: Test size, validation split, epochs, batch size, learning rate, etc.
- **Model**: LSTM units, dropout rate, dense units, etc.
- **Prediction**: Model path, sample size, evaluation flag, etc.

## Usage

### Training a Feature Prediction Model

```bash
python main.py --mode train --type feature
```

### Making Predictions with a Trained Model

```bash
python main.py --mode predict --type target
```

### Using a Custom Configuration

```bash
python main.py --mode train --type feature --config custom_config.json
```

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Model Architecture

The RNN model uses LSTM layers for sequence prediction, with dropout for regularization and dense layers for output. The architecture can be configured through the `config.json` file.
