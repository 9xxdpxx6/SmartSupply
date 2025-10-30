# file: app/train.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
from typing import Dict, Any, Optional


def train_prophet(
    shop_csv_path: str, 
    model_out_path: str, 
    include_regressors: bool = False
) -> Dict[str, Any]:
    """
    Train a Prophet model on shop-level sales data.

    Args:
        shop_csv_path: Path to the shop-level CSV file (ds, y columns)
        model_out_path: Path to save the trained model using joblib
        include_regressors: Whether to include additional regressors (explained below)

    Returns:
        Dictionary containing model path and backtest metrics
    """
    # Read the shop CSV file
    df = pd.read_csv(shop_csv_path)
    
    # Verify required columns exist
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV must contain 'ds' and 'y' columns")
    
    # Prepare the data for Prophet
    df_prophet = df[['ds', 'y']].copy()
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet['y'] = pd.to_numeric(df_prophet['y'])
    
    # Initialize Prophet model with weekly and yearly seasonality
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95  # Confidence interval
    )
    
    # If include_regressors is True, additional regressors can be added
    # These would need to be passed as additional parameters or loaded from another file
    # e.g., df_regressors = pd.read_csv(regressors_path)
    # df_prophet = df_prophet.merge(df_regressors, on='ds', how='left')
    # for col in df_regressors.columns:
    #     if col != 'ds':
    #         model.add_regressor(col)
    
    # Fit the model
    model.fit(df_prophet)
    
    # Define backtest period (last 20% of data by time)
    min_date = df_prophet['ds'].min()
    max_date = df_prophet['ds'].max()
    total_days = (max_date - min_date).days
    backtest_days = max(1, int(total_days * 0.2))  # Use at least 1 day
    backtest_start = max_date - pd.Timedelta(days=backtest_days - 1)  # Include the end date
    
    # Create future dataframe for backtesting
    future = model.make_future_dataframe(periods=0)  # Only historical dates
    mask = future['ds'] >= backtest_start
    future_backtest = future[mask].copy()
    
    # Make predictions for backtest period
    forecast = model.predict(future_backtest)
    
    # Get actual values for the backtest period
    actual_backtest = df_prophet[df_prophet['ds'] >= backtest_start]
    
    # Align actual and predicted values
    actual_values = actual_backtest['y'].values
    predicted_values = forecast['yhat'].values
    
    # Calculate metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Handle potential division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        if np.isnan(mape):
            # If division by zero caused NaN, calculate MAPE manually avoiding zero values
            mask = actual_values != 0
            if mask.any():
                mape = np.mean(np.abs((actual_values[mask] - predicted_values[mask]) / actual_values[mask])) * 100
            else:
                mape = 0.0
    
    # Save the trained model
    joblib.dump(model, model_out_path)
    
    # Return results
    results = {
        'model_path': model_out_path,
        'backtest_metrics': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'backtest_days': backtest_days
        },
        'data_info': {
            'total_records': len(df_prophet),
            'date_range': {
                'start': df_prophet['ds'].min().isoformat(),
                'end': df_prophet['ds'].max().isoformat()
            },
            'backtest_date_range': {
                'start': actual_backtest['ds'].min().isoformat(),
                'end': actual_backtest['ds'].max().isoformat()
            }
        }
    }
    
    return results


# TODO: Category model extension
# For category models, you would implement a similar function:
# def train_prophet_category(category_csv_path: str, category_name: str, model_out_path: str) -> Dict[str, Any]:
#     df = pd.read_csv(category_csv_path)
#     # Filter for specific category
#     df_category = df[df['category'] == category_name][['ds', 'y']].copy()
#     # Then similar training process as above
#     # The main difference is filtering the data for a specific category before training


if __name__ == "__main__":
    # Example usage
    # This would be run with actual file paths like:
    # results = train_prophet(
    #     shop_csv_path="data/shop_sales.csv",
    #     model_out_path="models/shop_prophet_model.pkl",
    #     include_regressors=False
    # )
    
    print("Example usage:")
    print("results = train_prophet(")
    print('    shop_csv_path="data/shop_sales.csv",')
    print('    model_out_path="models/shop_prophet_model.pkl",')
    print('    include_regressors=False')
    print(")")
    print()
    print("This will train a Prophet model with weekly and yearly seasonality,")
    print("perform a backtest on the last 20% of the data, and return metrics.")