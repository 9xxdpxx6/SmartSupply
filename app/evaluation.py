# file: app/evaluation.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import Dict, Any, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE value as percentage
    """
    mask = actual != 0
    if mask.any():
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        return 0.0


def rolling_cross_validation_prophet(
    shop_df: pd.DataFrame,
    initial_days: int = 180,
    horizon_days: int = 30,
    period_days: int = 30,
    include_regressors: bool = False,
    log_transform: bool = False
) -> Dict[str, Any]:
    """
    Perform rolling cross-validation for Prophet model.
    
    This function implements a time-series cross-validation scheme:
    1. Train on initial_days of data
    2. Predict next horizon_days
    3. Slide window by period_days
    4. Repeat until end of data
    
    Args:
        shop_df: DataFrame with columns 'ds' (datetime), 'y' (target), 
                 and optionally 'avg_price', 'avg_discount' if include_regressors=True
        initial_days: Number of days for initial training period
        horizon_days: Number of days to forecast ahead
        period_days: Number of days to slide the window forward
        include_regressors: Whether to include avg_price and avg_discount as regressors
        log_transform: Whether to apply log1p transformation to target variable
        
    Returns:
        Dictionary containing:
        - metrics: dict with mean and std for MAE, RMSE, MAPE
        - predictions_df: DataFrame with all predictions and actuals
        - cv_steps: list of dictionaries with metrics for each CV step
    """
    logger.info(f"Starting rolling cross-validation")
    logger.info(f"Parameters: initial={initial_days} days, horizon={horizon_days} days, period={period_days} days")
    
    # Prepare data
    df = shop_df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    if 'y' not in df.columns or 'ds' not in df.columns:
        raise ValueError("DataFrame must contain 'ds' and 'y' columns")
    
    # Apply log transformation if requested
    original_y = df['y'].copy()
    if log_transform:
        logger.info("Applying log1p transformation to target variable")
        df['y'] = np.log1p(df['y'])
    
    # Check for regressors
    has_regressors = include_regressors and 'avg_price' in df.columns and 'avg_discount' in df.columns
    if include_regressors and not has_regressors:
        logger.warning("Regressors requested but not found in DataFrame. Proceeding without regressors.")
        include_regressors = False
    
    # Prepare Prophet columns
    prophet_cols = ['ds', 'y']
    if has_regressors:
        prophet_cols.extend(['avg_price', 'avg_discount'])
    
    # Get date range
    min_date = df['ds'].min()
    max_date = df['ds'].max()
    total_days = (max_date - min_date).days + 1
    
    logger.info(f"Data range: {min_date.date()} to {max_date.date()} ({total_days} total days)")
    
    if total_days < initial_days + horizon_days:
        raise ValueError(f"Insufficient data: need at least {initial_days + horizon_days} days, got {total_days}")
    
    # Perform rolling CV
    all_predictions = []
    cv_steps = []
    step = 0
    
    current_start = min_date
    
    while True:
        # Calculate train end and test start/end
        train_end = current_start + pd.Timedelta(days=initial_days - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=horizon_days - 1)
        
        # Check if we have enough data for test period
        if test_end > max_date:
            logger.info(f"CV step {step}: Not enough data for test period, stopping")
            break
        
        logger.info(f"CV step {step}: Training on {current_start.date()} to {train_end.date()}, "
                   f"Testing on {test_start.date()} to {test_end.date()}")
        
        # Extract train and test data
        train_mask = (df['ds'] >= current_start) & (df['ds'] <= train_end)
        test_mask = (df['ds'] >= test_start) & (df['ds'] <= test_end)
        
        df_train = df[train_mask][prophet_cols].copy()
        df_test = df[test_mask].copy()
        
        if len(df_train) < initial_days * 0.9:  # Allow some tolerance for missing dates
            logger.warning(f"CV step {step}: Insufficient training data ({len(df_train)} rows), skipping")
            current_start += pd.Timedelta(days=period_days)
            step += 1
            continue
        
        if len(df_test) == 0:
            logger.warning(f"CV step {step}: No test data, stopping")
            break
        
        # Train Prophet model
        try:
            model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                seasonality_mode='additive'
            )
            
            if has_regressors:
                model.add_regressor('avg_price')
                model.add_regressor('avg_discount')
            
            model.fit(df_train)
            
            # Create future dataframe for test period
            periods = len(df_test)
            future = model.make_future_dataframe(periods=periods, freq='D')
            
            # Add regressors if needed
            if has_regressors:
                # Merge regressors from test data
                all_regressors = pd.concat([
                    df_train[['ds', 'avg_price', 'avg_discount']],
                    df_test[['ds', 'avg_price', 'avg_discount']]
                ], ignore_index=True)
                
                future = future.merge(
                    all_regressors,
                    on='ds',
                    how='left'
                )
                
                # Forward-fill regressors
                for col in ['avg_price', 'avg_discount']:
                    future[col] = future[col].ffill()
                    if future[col].isna().any():
                        last_value = df_train[col].iloc[-1] if not df_train.empty else 0
                        future[col] = future[col].fillna(last_value)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract predictions for test period
            test_mask_forecast = forecast['ds'] >= test_start
            forecast_test = forecast[test_mask_forecast].copy()
            
            # Align predictions with actual test data
            merged = forecast_test[['ds', 'yhat']].merge(
                df_test[['ds', 'y']],
                on='ds',
                how='inner'
            )
            
            if merged.empty:
                logger.warning(f"CV step {step}: No overlapping dates between predictions and test data, skipping")
                current_start += pd.Timedelta(days=period_days)
                step += 1
                continue
            
            # Get actual and predicted values
            actual_values = merged['y'].values
            predicted_values = merged['yhat'].values
            
            # Apply inverse transform if log_transform was used
            if log_transform:
                actual_values = np.expm1(actual_values)
                predicted_values = np.expm1(predicted_values)
            
            # Ensure non-negative values for predictions (realistic constraint)
            # Sales cannot be negative, so clip negative predictions to 0
            n_negative_pred = (predicted_values < 0).sum()
            if n_negative_pred > 0:
                logger.info(f"CV step {step}: Clamping {n_negative_pred} negative predictions to 0")
                predicted_values = np.clip(predicted_values, 0, None)
            
            # Also clip actual values if they're negative (data quality issue)
            n_negative_actual = (actual_values < 0).sum()
            if n_negative_actual > 0:
                logger.warning(f"CV step {step}: Found {n_negative_actual} negative actual values (data quality issue)")
                actual_values = np.clip(actual_values, 0, None)
            
            # Calculate metrics
            mae_val = mean_absolute_error(actual_values, predicted_values)
            rmse_val = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mape_val = mape(actual_values, predicted_values)
            
            # Store predictions
            predictions_df_step = pd.DataFrame({
                'ds': merged['ds'].values,
                'actual': actual_values,
                'predicted': predicted_values,
                'cv_step': step
            })
            all_predictions.append(predictions_df_step)
            
            # Store step metrics
            cv_steps.append({
                'step': step,
                'train_start': current_start.isoformat(),
                'train_end': train_end.isoformat(),
                'test_start': test_start.isoformat(),
                'test_end': test_end.isoformat(),
                'n_train': len(df_train),
                'n_test': len(merged),
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'mape': float(mape_val)
            })
            
            logger.info(f"CV step {step}: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%")
            
        except Exception as e:
            logger.error(f"CV step {step}: Error during training/prediction: {str(e)}")
            logger.error(f"Skipping step {step}")
        
        # Move to next window
        current_start += pd.Timedelta(days=period_days)
        step += 1
    
    # Aggregate results
    if not all_predictions:
        raise ValueError("No valid CV steps completed. Check data and parameters.")
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Calculate aggregate metrics
    all_mae = [s['mae'] for s in cv_steps]
    all_rmse = [s['rmse'] for s in cv_steps]
    all_mape = [s['mape'] for s in cv_steps]
    
    metrics = {
        'mae': {
            'mean': float(np.mean(all_mae)),
            'std': float(np.std(all_mae)),
            'min': float(np.min(all_mae)),
            'max': float(np.max(all_mae))
        },
        'rmse': {
            'mean': float(np.mean(all_rmse)),
            'std': float(np.std(all_rmse)),
            'min': float(np.min(all_rmse)),
            'max': float(np.max(all_rmse))
        },
        'mape': {
            'mean': float(np.mean(all_mape)),
            'std': float(np.std(all_mape)),
            'min': float(np.min(all_mape)),
            'max': float(np.max(all_mape))
        },
        'n_cv_steps': len(cv_steps),
        'log_transform': log_transform,
        'include_regressors': include_regressors
    }
    
    logger.info(f"Cross-validation completed: {len(cv_steps)} steps")
    logger.info(f"Aggregate MAE: {metrics['mae']['mean']:.2f} ± {metrics['mae']['std']:.2f}")
    logger.info(f"Aggregate RMSE: {metrics['rmse']['mean']:.2f} ± {metrics['rmse']['std']:.2f}")
    logger.info(f"Aggregate MAPE: {metrics['mape']['mean']:.2f}% ± {metrics['mape']['std']:.2f}%")
    
    return {
        'metrics': metrics,
        'predictions_df': predictions_df,
        'cv_steps': cv_steps
    }


def plot_cv_results(df_preds: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """
    Plot cross-validation results: actual vs predicted values.
    
    Args:
        df_preds: DataFrame with columns 'ds', 'actual', 'predicted', 'cv_step'
        out_path: Optional path to save the plot. If None, displays the plot.
    """
    if df_preds.empty:
        raise ValueError("Empty predictions DataFrame")
    
    # Prepare data
    df = df_preds.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot actual values
    ax.plot(df['ds'], df['actual'], 
           label='Actual Sales', color='blue', linewidth=2, alpha=0.7, marker='o', markersize=3)
    
    # Plot predictions (grouped by CV step if needed)
    if 'cv_step' in df.columns:
        for step in sorted(df['cv_step'].unique()):
            step_data = df[df['cv_step'] == step]
            if len(step_data) > 0:
                ax.plot(step_data['ds'], step_data['predicted'], 
                       label=f'Predictions (Step {step})', 
                       linewidth=1.5, alpha=0.6, linestyle='--', marker='x', markersize=2)
    else:
        ax.plot(df['ds'], df['predicted'], 
               label='Predictions', color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    
    # Add scatter for better visibility
    ax.scatter(df['ds'], df['predicted'], color='red', alpha=0.4, s=20, zorder=5)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Results: Actual vs Predicted Sales', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {out_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rolling cross-validation for Prophet model")
    parser.add_argument("shop_csv", help="Path to shop-level CSV file")
    parser.add_argument("--initial-days", type=int, default=180,
                       help="Initial training period in days (default: 180)")
    parser.add_argument("--horizon-days", type=int, default=30,
                       help="Forecast horizon in days (default: 30)")
    parser.add_argument("--period-days", type=int, default=30,
                       help="Window slide period in days (default: 30)")
    parser.add_argument("--include-regressors", action="store_true",
                       help="Include avg_price and avg_discount as regressors")
    parser.add_argument("--log-transform", action="store_true",
                       help="Apply log1p transformation to target variable")
    parser.add_argument("--output-predictions", type=str, default="data/processed/cv_predictions.csv",
                       help="Output path for predictions CSV (default: data/processed/cv_predictions.csv)")
    parser.add_argument("--output-plot", type=str, default="data/processed/cv_plot.png",
                       help="Output path for visualization plot (default: data/processed/cv_plot.png)")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from: {args.shop_csv}")
    df = pd.read_csv(args.shop_csv)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Perform cross-validation
    results = rolling_cross_validation_prophet(
        shop_df=df,
        initial_days=args.initial_days,
        horizon_days=args.horizon_days,
        period_days=args.period_days,
        include_regressors=args.include_regressors,
        log_transform=args.log_transform
    )
    
    # Save predictions
    os.makedirs(os.path.dirname(args.output_predictions) if os.path.dirname(args.output_predictions) else '.', exist_ok=True)
    results['predictions_df'].to_csv(args.output_predictions, index=False)
    logger.info(f"Predictions saved to: {args.output_predictions}")
    
    # Create and save plot
    plot_cv_results(results['predictions_df'], args.output_plot)
    
    # Print summary
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    print(f"\nNumber of CV steps: {results['metrics']['n_cv_steps']}")
    print(f"\nAggregate Metrics:")
    print(f"  MAE:  {results['metrics']['mae']['mean']:.2f} ± {results['metrics']['mae']['std']:.2f} "
          f"(range: {results['metrics']['mae']['min']:.2f} - {results['metrics']['mae']['max']:.2f})")
    print(f"  RMSE: {results['metrics']['rmse']['mean']:.2f} ± {results['metrics']['rmse']['std']:.2f} "
          f"(range: {results['metrics']['rmse']['min']:.2f} - {results['metrics']['rmse']['max']:.2f})")
    print(f"  MAPE: {results['metrics']['mape']['mean']:.2f}% ± {results['metrics']['mape']['std']:.2f}% "
          f"(range: {results['metrics']['mape']['min']:.2f}% - {results['metrics']['mape']['max']:.2f}%)")
    print(f"\nConfiguration:")
    print(f"  Log transform: {results['metrics']['log_transform']}")
    print(f"  Include regressors: {results['metrics']['include_regressors']}")
    print(f"\nResults saved:")
    print(f"  Predictions: {args.output_predictions}")
    print(f"  Plot: {args.output_plot}")

