# file: app/train.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
import json
import os
import logging
from typing import Dict, Any

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


def train_prophet(
    shop_csv_path: str, 
    model_out_path: str, 
    include_regressors: bool = False,
    log_transform: bool = False,
    interval_width: float = 0.95,
    holdout_frac: float = 0.2,
    changepoint_prior_scale: float = 0.01,  # Консервативное значение для стабильности
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: str = 'additive',
    auto_tune: bool = False,
    skip_holdout: bool = False  # Если True, использует ВСЕ данные для обучения (без теста)
) -> Dict[str, Any]:
    """
    Train a Prophet model on shop-level sales data.
    
    Args:
        shop_csv_path: Path to the shop-level CSV file (ds, y, avg_price, avg_discount, day_of_week, is_weekend)
        model_out_path: Path to save the trained model using joblib
        include_regressors: Whether to include avg_price and avg_discount as regressors
        log_transform: If True, apply log1p transformation to y before training
        interval_width: Confidence interval width for Prophet (default 0.95)
        holdout_frac: Fraction of data to use for testing (default 0.2)
        changepoint_prior_scale: Flexibility of automatic changepoint detection (default 0.05, higher = more flexible)
        seasonality_prior_scale: Strength of seasonality components (default 10.0, higher = stronger seasonality)
        seasonality_mode: 'additive' or 'multiplicative' (default 'additive')
        auto_tune: If True, perform automatic grid search to find best configuration
        
    Returns:
        Dictionary containing model path, metrics, and data ranges
    """
    logger.info(f"Starting model training from: {shop_csv_path}")
    
    # Auto-tuning: perform grid search
    if auto_tune:
        logger.info("Auto-tuning enabled: performing grid search...")
        try:
            from app.tuning import grid_search_models
            import os
            
            analysis_dir = os.path.join(os.path.dirname(model_out_path) or 'models', '..', 'analysis')
            analysis_dir = os.path.normpath(analysis_dir)
            
            tuning_results = grid_search_models(
                shop_csv_path=shop_csv_path,
                holdout_frac=holdout_frac,
                output_dir=analysis_dir
            )
            
            if tuning_results.get('success', False):
                best_model = tuning_results['best_model']
                best_config = best_model.get('config', {})
                
                logger.info(f"Best model from grid search: {best_model['name']}")
                logger.info(f"Best metrics: MAPE={best_model['metrics']['mape']:.2f}%, "
                           f"Coverage={best_model['metrics']['coverage']*100:.1f}%")
                
                # Обновляем параметры из лучшей конфигурации
                if 'seasonality_prior_scale' in best_config:
                    seasonality_prior_scale = best_config['seasonality_prior_scale']
                if 'changepoint_prior_scale' in best_config:
                    changepoint_prior_scale = best_config['changepoint_prior_scale']
                if 'interval_width' in best_config:
                    interval_width = best_config['interval_width']
                if 'seasonality_mode' in best_config:
                    seasonality_mode = best_config['seasonality_mode']
                if 'include_regressors' in best_config:
                    include_regressors = best_config['include_regressors']
                
                logger.info(f"Using optimized parameters: seasonality_prior_scale={seasonality_prior_scale}, "
                           f"changepoint_prior_scale={changepoint_prior_scale}, interval_width={interval_width}, "
                           f"seasonality_mode={seasonality_mode}, include_regressors={include_regressors}")
            else:
                logger.warning("Grid search failed or returned no results, using default parameters")
        except Exception as e:
            logger.error(f"Auto-tuning failed: {str(e)}, falling back to default parameters")
            logger.error(f"Error details: {str(e)}", exc_info=True)
    
    # Read the shop CSV file
    if not os.path.exists(shop_csv_path):
        raise FileNotFoundError(f"Shop CSV file not found: {shop_csv_path}")
    
    df = pd.read_csv(shop_csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")
    
    # Verify required columns exist
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV must contain 'ds' and 'y' columns")
    
    # Prepare the data
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Save original y values before any transformation
    original_y = df['y'].copy()
    
    # Split data by time: train on first (1-holdout_frac), test on last holdout_frac
    # Разделение данных на train/test или использование всех данных
    if skip_holdout:
        # Используем ВСЕ данные для обучения (без тестового набора)
        logger.info("skip_holdout=True: Using ALL data for training (no test set split)")
        df_train = df.copy()
        df_test = pd.DataFrame(columns=df.columns)  # Пустой test set
        n_train = len(df_train)
        n_test = 0
        if n_train < 30:
            raise ValueError(f"Insufficient data for training: {n_train} rows (minimum 30 required)")
        logger.info(f"Training on ALL {n_train} days ({df_train['ds'].min().date()} to {df_train['ds'].max().date()})")
        logger.info("⚠️ No test set - metrics will not be calculated. Use for production forecasts.")
    else:
        n_total = len(df)
        n_train = int(n_total * (1 - holdout_frac))
        n_test = n_total - n_train
        
        if n_train < 30:
            raise ValueError(f"Insufficient data for training: {n_train} rows (minimum 30 required)")
        
        df_train = df.iloc[:n_train].copy()
        df_test = df.iloc[n_train:].copy()
    
    # Apply log transformation if requested (after split to preserve original test values)
    # Always save original test y values for metrics calculation
    df_test_original_y = df_test['y'].copy()
    if log_transform:
        logger.info("Applying log1p transformation to target variable")
        df_train['y'] = np.log1p(df_train['y'])
        df_test['y'] = np.log1p(df_test['y'])
    
    train_range = {
        'start': df_train['ds'].min().isoformat(),
        'end': df_train['ds'].max().isoformat()
    }
    
    # Обработка test_range: если skip_holdout=True, df_test пустой
    if skip_holdout or len(df_test) == 0:
        test_range = {
            'start': None,
            'end': None
        }
        logger.info(f"Train period: {train_range['start']} to {train_range['end']} ({n_train} rows)")
        logger.info(f"Test period: N/A (skip_holdout=True, using all data for training)")
    else:
        test_range = {
            'start': df_test['ds'].min().isoformat(),
            'end': df_test['ds'].max().isoformat()
        }
        logger.info(f"Train period: {train_range['start']} to {train_range['end']} ({n_train} rows)")
        logger.info(f"Test period: {test_range['start']} to {test_range['end']} ({n_test} rows)")
    
    # Prepare data for Prophet
    prophet_cols = ['ds', 'y']
    if include_regressors:
        if 'avg_price' not in df.columns or 'avg_discount' not in df.columns:
            logger.warning("Regressors requested but avg_price/avg_discount not found in CSV. Proceeding without regressors.")
            include_regressors = False
        else:
            prophet_cols.extend(['avg_price', 'avg_discount'])
    
    df_prophet_train = df_train[prophet_cols].copy()
    
    # Validate seasonality_mode
    if seasonality_mode not in ['additive', 'multiplicative']:
        raise ValueError(f"seasonality_mode must be 'additive' or 'multiplicative', got '{seasonality_mode}'")
    
    # Определяем, достаточно ли данных для yearly seasonality
    # Prophet рекомендует минимум 730 дней (2 года) для надежной yearly seasonality
    days_span = (df['ds'].max() - df['ds'].min()).days
    use_yearly = days_span >= 730  # Только если данных >= 2 лет
    
    if not use_yearly and days_span < 730:
        logger.warning(f"Data span ({days_span} days) < 730 days. Disabling yearly_seasonality for stability.")
        logger.info("Using only weekly_seasonality. This is recommended for datasets < 2 years.")
        logger.warning("⚠️ LONG-TERM FORECAST WARNING: For forecasts > 90 days with data < 2 years, "
                      "the model may show flat/cyclical patterns due to missing yearly_seasonality.")
    
    # Initialize Prophet model with configurable hyperparameters
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=use_yearly,  # Автоматически отключаем для коротких данных
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode
    )
    
    # Для данных >= 365 дней (но < 730): добавляем месячную сезонность как компромисс
    if days_span >= 365 and days_span < 730:
        logger.info(f"Data span ({days_span} days) >= 365 but < 730. Adding monthly seasonality as a compromise.")
        # Добавляем месячную сезонность для сохранения сезонных паттернов на длинных горизонтах
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        logger.info("Added monthly seasonality (period=30.5 days, fourier_order=5) to preserve seasonal patterns on long horizons")
    
    logger.info(f"Prophet model config: changepoint_prior_scale={changepoint_prior_scale}, "
                f"seasonality_prior_scale={seasonality_prior_scale}, seasonality_mode={seasonality_mode}")
    
    # Add regressors if requested
    if include_regressors:
        logger.info("Adding regressors: avg_price, avg_discount")
        model.add_regressor('avg_price')
        model.add_regressor('avg_discount')
    
    # Fit the model
    logger.info("Fitting Prophet model...")
    model.fit(df_prophet_train)
    logger.info("Model fitted successfully")
    
    # Вычисление метрик (пропускается если skip_holdout)
    if skip_holdout:
        logger.info("Skipping metrics calculation (skip_holdout=True)")
        use_cv = False
        metrics_dict = {
            'mae': None,
            'rmse': None,
            'mape': None,
            'coverage': None,
            'log_transform': log_transform,
            'interval_width': interval_width,
            'holdout_frac': holdout_frac,
            'used_cross_validation': False,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'seasonality_mode': seasonality_mode,
            'auto_tune': auto_tune,
            'skip_holdout': skip_holdout,
            'note': 'Model trained on ALL data. No test set metrics available. Ready for production forecasts.'
        }
    elif n_test < 7:
        # Check if test set is too small for proper evaluation
        use_cv = True
        logger.warning(f"Test set too small ({n_test} < 7). Using time-series cross-validation instead.")
    else:
        use_cv = False
    
    # Calculate metrics (только если не skip_holdout)
    if not skip_holdout:
        if use_cv:
            # Time-series cross-validation
            logger.info("Performing time-series cross-validation...")
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Perform cross-validation with minimum training period of 30 days
            min_periods = min(30, n_train)
            cv_results = cross_validation(
                model, 
                initial=f'{min_periods} days',
                period='7 days',
                horizon='7 days'
            )
            
            cv_metrics = performance_metrics(cv_results)
            
            # Get actual and predicted values from CV
            actual_values = cv_results['y'].values
            predicted_values = cv_results['yhat'].values
            
            # Apply inverse transform if log_transform was used
            if log_transform:
                logger.info("Applying inverse log1p (expm1) transformation to predictions")
                actual_values = np.expm1(actual_values)
                predicted_values = np.expm1(predicted_values)
            
            # Ensure non-negative values for predictions (realistic constraint)
            # Sales cannot be negative, so clip negative predictions to 0
            n_negative_pred = (predicted_values < 0).sum()
            if n_negative_pred > 0:
                logger.info(f"Clamping {n_negative_pred} negative CV predictions to 0 (sales cannot be negative)")
                predicted_values = np.clip(predicted_values, 0, None)
            
            # Also clip actual values if they're negative (data quality issue)
            n_negative_actual = (actual_values < 0).sum()
            if n_negative_actual > 0:
                logger.warning(f"Found {n_negative_actual} negative actual values in CV (data quality issue)")
                actual_values = np.clip(actual_values, 0, None)
            
            # Calculate metrics
            mae_val = mean_absolute_error(actual_values, predicted_values)
            rmse_val = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mape_val = mape(actual_values, predicted_values)
            
            logger.info(f"Cross-validation metrics: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%")
        
        else:
            # Standard test set evaluation
            # Create future dataframe for test period
            periods = len(df_test)
            future = model.make_future_dataframe(periods=periods, freq='D')
            
            # Add regressors for future dates if needed
            if include_regressors:
                # Combine train and test regressors for complete coverage
                all_regressors = pd.concat([
                    df_train[['ds', 'avg_price', 'avg_discount']],
                    df_test[['ds', 'avg_price', 'avg_discount']]
                ], ignore_index=True)
                
                # Merge regressors
                future = future.merge(
                    all_regressors,
                    on='ds',
                    how='left'
                )
                
                # Forward-fill regressors for any missing dates
                for col in ['avg_price', 'avg_discount']:
                    if future[col].isna().any():
                        # Forward fill, then use last known value if still NaN
                        future[col] = future[col].ffill()
                        if future[col].isna().any():
                            last_known_value = all_regressors[col].iloc[-1] if not all_regressors.empty else 0
                            future[col] = future[col].fillna(last_known_value)
            
            # Make predictions
            logger.info(f"Generating predictions for {periods} periods...")
            forecast = model.predict(future)
            
            # Extract predictions for test period only
            test_mask = forecast['ds'] >= df_test['ds'].min()
            forecast_test = forecast[test_mask].copy()
            
            # Align actual and predicted values
            forecast_test = forecast_test.sort_values('ds')
            df_test_aligned = df_test.sort_values('ds')
            
            # Merge on ds to ensure alignment
            merged = forecast_test[['ds', 'yhat']].merge(
                df_test_aligned[['ds']],
                on='ds',
                how='inner'
            )
            
            # Get actual y values for metrics (original values if log_transform, transformed otherwise)
            if log_transform:
                # Use original y values before log transform
                merged = merged.merge(
                    pd.DataFrame({'ds': df_test['ds'].values, 'y_original': df_test_original_y.values}),
                    on='ds',
                    how='inner'
                )
                actual_values = merged['y_original'].values
                logger.info("Applying inverse log1p (expm1) transformation to predictions")
                predicted_values = np.expm1(merged['yhat'].values)
            else:
                # Use transformed y values
                merged = merged.merge(
                    df_test_aligned[['ds', 'y']],
                    on='ds',
                    how='inner'
                )
                actual_values = merged['y'].values
                predicted_values = merged['yhat'].values
            
            # Ensure non-negative values for predictions (realistic constraint)
            # Sales cannot be negative, so clip negative predictions to 0
            n_negative_pred = (predicted_values < 0).sum()
            if n_negative_pred > 0:
                logger.info(f"Clamping {n_negative_pred} negative predictions to 0 (sales cannot be negative)")
                predicted_values = np.clip(predicted_values, 0, None)
            
            # Also clip actual values if they're negative (data quality issue)
            n_negative_actual = (actual_values < 0).sum()
            if n_negative_actual > 0:
                logger.warning(f"Found {n_negative_actual} negative actual values (data quality issue)")
                actual_values = np.clip(actual_values, 0, None)
            
            # Calculate metrics
            mae_val = mean_absolute_error(actual_values, predicted_values)
            rmse_val = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mape_val = mape(actual_values, predicted_values)
            
            logger.info(f"Test metrics: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%")
            
            # Prepare metrics dictionary for test case
            metrics_dict = {
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'mape': float(mape_val),
                'log_transform': log_transform,
                'interval_width': interval_width,
                'holdout_frac': holdout_frac,
                'used_cross_validation': use_cv,
                'changepoint_prior_scale': changepoint_prior_scale,
                'seasonality_prior_scale': seasonality_prior_scale,
                'seasonality_mode': seasonality_mode,
                'auto_tune': auto_tune,
                'skip_holdout': skip_holdout
            }
    
    # Save the trained model
    model_dir = os.path.dirname(model_out_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Saving model to: {model_out_path}")
    joblib.dump(model, model_out_path)
    
    # Save metrics to JSON file
    metrics_path = model_out_path.replace('.pkl', '_metrics.json')
    logger.info(f"Saving metrics to: {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Prepare return dictionary
    results = {
        'model_path': model_out_path,
        'metrics': metrics_dict,
        'train_range': train_range,
        'test_range': test_range,
        'n_train': n_train,
        'n_test': n_test if not use_cv else len(cv_results) if use_cv else n_test
    }
    
    logger.info("Training completed successfully")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Prophet model on shop-level sales data")
    parser.add_argument("shop_csv", help="Path to shop-level CSV file")
    parser.add_argument("model_out", help="Output path for trained model")
    parser.add_argument("--include-regressors", action="store_true", 
                       help="Include avg_price and avg_discount as regressors")
    parser.add_argument("--log-transform", action="store_true",
                       help="Apply log1p transformation to target variable")
    parser.add_argument("--interval-width", type=float, default=0.95,
                       help="Confidence interval width (default: 0.95)")
    parser.add_argument("--holdout-frac", type=float, default=0.2,
                       help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--auto-tune", action="store_true",
                       help="Perform automatic grid search to find best configuration")
    
    args = parser.parse_args()
    
    result = train_prophet(
        shop_csv_path=args.shop_csv,
        model_out_path=args.model_out,
        include_regressors=args.include_regressors,
        log_transform=args.log_transform,
        interval_width=args.interval_width,
        holdout_frac=args.holdout_frac,
        auto_tune=args.auto_tune
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nModel saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['model_path'].replace('.pkl', '_metrics.json')}")
    print(f"\nTraining period: {result['train_range']['start']} to {result['train_range']['end']}")
    print(f"Training samples: {result['n_train']}")
    
    # Проверка test_range на None (когда skip_holdout=True)
    test_start = result['test_range'].get('start') if result.get('test_range') else None
    test_end = result['test_range'].get('end') if result.get('test_range') else None
    if test_start and test_end:
        print(f"Test period: {test_start} to {test_end}")
        print(f"Test samples: {result['n_test']}")
    else:
        print(f"Test period: N/A (skip_holdout=True)")
        print(f"Test samples: 0")
    print(f"\nMetrics:")
    mae_val = result['metrics'].get('mae')
    rmse_val = result['metrics'].get('rmse')
    mape_val = result['metrics'].get('mape')
    
    if mae_val is not None:
        print(f"  MAE:  {mae_val:.4f}")
    else:
        print(f"  MAE:  N/A (skip_holdout=True)")
    
    if rmse_val is not None:
        print(f"  RMSE: {rmse_val:.4f}")
    else:
        print(f"  RMSE: N/A (skip_holdout=True)")
    
    if mape_val is not None:
        print(f"  MAPE: {mape_val:.2f}%")
    else:
        print(f"  MAPE: N/A (skip_holdout=True)")
    
    print(f"  Log transform: {result['metrics'].get('log_transform', False)}")
    print(f"  Used cross-validation: {result['metrics']['used_cross_validation']}")
