# file: app/predict.py
import pandas as pd
import joblib
from prophet import Prophet
from typing import Optional


def predict_prophet(model_path: str, horizon_days: int) -> pd.DataFrame:
    """
    Load a trained Prophet model and generate forecasts for the specified horizon.

    Args:
        model_path: Path to the saved Prophet model (created with joblib.dump)
        horizon_days: Number of days to forecast into the future

    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    # Validate inputs
    if horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")
    
    # Load the trained model
    model: Prophet = joblib.load(model_path)
    
    # Create future dataframe for the specified horizon
    future = model.make_future_dataframe(periods=horizon_days, freq='D')
    
    # Generate predictions
    forecast = model.predict(future)
    
    # Select only the required columns
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    return result


def save_forecast_csv(df_forecast: pd.DataFrame, out_path: str) -> None:
    """
    Save forecast DataFrame to CSV file.

    Args:
        df_forecast: Forecast DataFrame with columns ds, yhat, yhat_lower, yhat_upper
        out_path: Path where the CSV file will be saved
    """
    # Validate input DataFrame
    required_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    for col in required_columns:
        if col not in df_forecast.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Save to CSV
    df_forecast.to_csv(out_path, index=False)