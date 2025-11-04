# file: app/utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(actual, predicted)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE value as percentage
    """
    # Handle potential division by zero
    mask = actual != 0
    if mask.any():
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        return 0.0


def save_metrics_json(metrics: dict, path: str) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        path: Path where the JSON file will be saved
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Save to JSON
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)




def read_csv_path(csv_path: str) -> pd.DataFrame:
    """
    Helper function to read a CSV file with proper error handling.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with the CSV data
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file {csv_path}: {str(e)}")
