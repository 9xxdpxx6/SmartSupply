# file: app/predict.py
import pandas as pd
import numpy as np
import joblib
import os
import logging
from prophet import Prophet
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def smooth_forecast_transition(
    forecast: pd.DataFrame,
    model: Prophet,
    smooth_days: int = 14,
    smooth_alpha: float = 0.6,
    max_change_pct: float = 0.015
) -> pd.DataFrame:
    """
    Smooth the transition from historical data to forecast to reduce initial overestimation
    and eliminate sharp jumps between days.
    
    Uses aggressive smoothing:
    - Strong initial mixing with historical average
    - Hard limit on maximum day-to-day changes
    
    Args:
        forecast: Forecast DataFrame with columns ds, yhat, yhat_lower, yhat_upper
        model: Trained Prophet model (to access historical data)
        smooth_days: Number of days to apply smoothing (default 14)
        smooth_alpha: Weight of historical average for first day (0-1, default 0.6 = 60% history)
        max_change_pct: Maximum allowed day-to-day change as percentage (default 0.015 = 1.5%)
        
    Returns:
        Smoothed forecast DataFrame
    """
    if smooth_days <= 0:
        return forecast
    
    # Get last actual value from historical data (более точный якорь)
    if hasattr(model, 'history') and model.history is not None:
        # Используем последнее фактическое значение, а не среднее
        last_actual = model.history['y'].iloc[-1] if len(model.history) > 0 else None
        
        # Если доступно, используем последние 7 дней для более точной оценки тренда
        last_days = model.history.tail(7)['y'].values if len(model.history) >= 7 else model.history['y'].values
        last_avg = last_days.mean() if len(last_days) > 0 else None
        
        # Используем последнее значение, но если оно сильно отличается от среднего - используем среднее
        if last_actual is not None and last_avg is not None:
            if abs(last_actual - last_avg) / (last_avg + 1e-8) < 0.3:  # Если разница < 30%
                historical_avg = last_actual  # Используем последнее значение
            else:
                historical_avg = last_avg  # Используем среднее для стабильности
        elif last_actual is not None:
            historical_avg = last_actual
        elif last_avg is not None:
            historical_avg = last_avg
        else:
            historical_avg = forecast['yhat'].iloc[0] * 0.75  # Fallback: 75% от прогноза
    else:
        # Fallback: use first forecast value adjusted more aggressively
        historical_avg = forecast['yhat'].iloc[0] * 0.75
    
    forecast_smooth = forecast.copy()
    # Ограничиваем сглаживание максимум 14 днями, чтобы не портить долгосрочные прогнозы
    n_smooth = min(smooth_days, len(forecast_smooth), 14)
    
    # First day: ОЧЕНЬ агрессивное смешивание - используем 95% веса истории
    first_day_alpha = 0.95  # Фиксируем на 95% - практически только история
    
    # Определяем тренд последних дней для корректировки
    trend_adjustment = 1.0
    if hasattr(model, 'history') and model.history is not None and len(model.history) >= 7:
        recent_trend = model.history.tail(7)['y'].values
        if len(recent_trend) >= 2:
            trend_slope = (recent_trend[-1] - recent_trend[0]) / len(recent_trend)
            # Если тренд падающий, используем более низкое значение
            if trend_slope < 0:
                trend_adjustment = 0.92  # Дополнительное снижение на 8% для падающего тренда
                logger.info(f"  - Detected DOWNWARD trend, applying {trend_adjustment:.0%} adjustment")
            elif trend_slope > 0:
                trend_adjustment = 1.02  # Небольшое увеличение для растущего тренда
                logger.info(f"  - Detected UPWARD trend, applying {trend_adjustment:.0%} adjustment")
    
    historical_avg = historical_avg * trend_adjustment
    
    # Для первого дня: практически только история (95%)
    first_value = float(forecast_smooth.iloc[0]['yhat'] * 0.05 + historical_avg * 0.95)
    
    # АБСОЛЮТНОЕ ограничение: первый день НЕ МОЖЕТ быть выше последнего значения
    # Используем последнее значение как максимум (без процента)
    if hasattr(model, 'history') and model.history is not None and len(model.history) > 0:
        last_actual = float(model.history['y'].iloc[-1])
        # Максимум = последнее значение (без увеличения)
        max_first_day = last_actual
        if first_value > max_first_day:
            logger.info(f"  - First day capped at {max_first_day:.2f} (100% of last actual {last_actual:.2f})")
            first_value = max_first_day
        else:
            logger.info(f"  - First day: {first_value:.2f} (vs last actual: {last_actual:.2f}, ratio: {first_value/last_actual:.2%})")
    
    forecast_smooth.loc[forecast_smooth.index[0], 'yhat'] = first_value
    
    # Для CI: балансируем между покрытием и практичностью
    # Используем более разумный подход: сохраняем пропорции от Prophet, но немного расширяем
    if hasattr(model, 'history') and model.history is not None and len(model.history) > 0:
        last_actual = float(model.history['y'].iloc[-1])
        
        # Используем оригинальные границы от Prophet как базу
        original_lower = float(forecast_smooth.iloc[0]['yhat_lower'])
        original_upper = float(forecast_smooth.iloc[0]['yhat_upper'])
        original_range = original_upper - original_lower
        
        # Если CI слишком узкий (< 15% от значения), расширяем умеренно
        min_reasonable_width = first_value * 0.15  # Минимум 15% (было 20%)
        if original_range < min_reasonable_width:
            # Используем историческую волатильность для разумного расширения
            last_30 = model.history.tail(30)['y'].values if len(model.history) >= 30 else model.history['y'].values
            last_std = float(np.std(last_30)) if len(last_30) > 1 else first_value * 0.10
            
            # Используем ±1.96*std (примерно 95% для нормального распределения)
            # Но ограничиваем максимальное расширение
            expansion = min(last_std * 1.96, first_value * 0.25)  # Максимум 25% расширение
            
            forecast_smooth.loc[forecast_smooth.index[0], 'yhat_lower'] = float(
                max(0, first_value - expansion)
            )
            forecast_smooth.loc[forecast_smooth.index[0], 'yhat_upper'] = float(
                first_value + expansion
            )
        else:
            # CI уже достаточно широкий, просто центрируем вокруг сглаженного значения
            # Сохраняем пропорцию, но смещаем центр к first_value
            current_center = (original_upper + original_lower) / 2
            adjustment = first_value - current_center
            
            forecast_smooth.loc[forecast_smooth.index[0], 'yhat_lower'] = float(
                max(0, original_lower + adjustment * 0.7)  # Частичное смещение
            )
            forecast_smooth.loc[forecast_smooth.index[0], 'yhat_upper'] = float(
                original_upper + adjustment * 0.7
            )
    else:
        # Fallback: умеренное расширение CI на основе текущих значений
        # Используем более консервативные границы (85-115% вместо 75-135%)
        forecast_smooth.loc[forecast_smooth.index[0], 'yhat_lower'] = float(
            max(0, first_value * 0.85)
        )
        forecast_smooth.loc[forecast_smooth.index[0], 'yhat_upper'] = float(
            first_value * 1.15
        )
    
    # Subsequent days: жесткое ограничение на максимальные изменения
    for i in range(1, n_smooth):
        prev_value = forecast_smooth.iloc[i-1]['yhat']
        curr_forecast = forecast_smooth.iloc[i]['yhat']
        
        # Максимальное изменение - уменьшаем до 1% для первых дней (вместо 1.5%)
        # Для первых 3 дней - еще более строго (0.5%)
        if i <= 3:
            effective_max_change = prev_value * 0.005  # 0.5% для первых 3 дней
        elif i <= 7:
            effective_max_change = prev_value * 0.01   # 1% для дней 4-7
        else:
            effective_max_change = prev_value * max_change_pct  # Обычное ограничение
        
        # Вычисляем фактическое изменение
        change = curr_forecast - prev_value
        
        # Если изменение слишком большое, ограничиваем с сохранением направления
        if abs(change) > effective_max_change:
            direction = 1 if change > 0 else -1
            smoothed_value = float(prev_value + direction * effective_max_change)
        else:
            smoothed_value = float(curr_forecast)
        
        # Дополнительно: если прогноз растет слишком быстро в первые дни - ограничиваем рост
        if i <= 5 and smoothed_value > prev_value * 1.01:  # Рост > 1% в первые 5 дней
            smoothed_value = float(prev_value)  # Полностью ограничиваем рост - остаемся на прежнем уровне
        
        forecast_smooth.loc[forecast_smooth.index[i], 'yhat'] = smoothed_value
        
        # Adjust confidence intervals proportionally (без излишнего расширения)
        original_range = forecast_smooth.iloc[i]['yhat_upper'] - forecast_smooth.iloc[i]['yhat_lower']
        adjustment = (smoothed_value - curr_forecast) / curr_forecast if curr_forecast != 0 else 0
        
        # Просто смещаем CI пропорционально изменению прогноза
        forecast_smooth.loc[forecast_smooth.index[i], 'yhat_lower'] = float(
            max(0, forecast_smooth.iloc[i]['yhat_lower'] + adjustment * forecast_smooth.iloc[i]['yhat_lower'])
        )
        forecast_smooth.loc[forecast_smooth.index[i], 'yhat_upper'] = float(
            forecast_smooth.iloc[i]['yhat_upper'] + adjustment * forecast_smooth.iloc[i]['yhat_upper']
        )
        
        # Убеждаемся, что upper >= lower
        if forecast_smooth.loc[forecast_smooth.index[i], 'yhat_upper'] < forecast_smooth.loc[forecast_smooth.index[i], 'yhat_lower']:
            center = smoothed_value
            width = original_range
            forecast_smooth.loc[forecast_smooth.index[i], 'yhat_lower'] = float(max(0, center - width/2))
            forecast_smooth.loc[forecast_smooth.index[i], 'yhat_upper'] = float(center + width/2)
    
    return forecast_smooth


def predict_prophet(
    model_path: str, 
    horizon_days: int, 
    last_known_regressors_csv: Optional[str] = None,
    log_transform: bool = False,
    regressor_fill_method: str = 'forward',
    smooth_transition: bool = False,
    smooth_days: int = 14,
    smooth_alpha: float = 0.6,
    max_change_pct: float = 0.015
) -> pd.DataFrame:
    """
    Load a trained Prophet model and generate forecasts for the specified horizon.
    
    Args:
        model_path: Path to the saved Prophet model (created with joblib.dump)
        horizon_days: Number of days to forecast into the future (1-365)
        last_known_regressors_csv: Optional path to shop CSV with regressors (avg_price, avg_discount)
        log_transform: If True, apply expm1 inverse transformation to predictions
        regressor_fill_method: Method to fill regressors for future dates ('forward' or 'median')
        smooth_transition: If True, apply smoothing to reduce initial overestimation (default False)
        smooth_days: Number of days to apply smoothing (default 14, only used if smooth_transition=True)
        smooth_alpha: Weight of historical average for first day (0-1, default 0.6 = 60% history, only used if smooth_transition=True)
        max_change_pct: Maximum allowed day-to-day change as percentage (default 0.015 = 1.5%, only used if smooth_transition=True)
        
    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    # Validate inputs
    if not (1 <= horizon_days <= 365):
        raise ValueError(f"horizon_days must be between 1 and 365, got {horizon_days}")
    
    # Предупреждение для длинных горизонтов
    if horizon_days > 180:
        logger.warning(f"⚠️ LONG HORIZON ({horizon_days} days): Forecast quality degrades significantly beyond 90-180 days.")
        logger.warning("   For best results, use horizon <= 90 days, especially with data < 2 years.")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load the trained model
    model: Prophet = joblib.load(model_path)
    
    # Проверяем объем данных в модели (после загрузки!)
    if hasattr(model, 'history') and model.history is not None:
        model_days = (model.history['ds'].max() - model.history['ds'].min()).days
        if model_days < 730 and horizon_days > 90:
            logger.warning(f"⚠️ Data span ({model_days} days) < 730 days with horizon ({horizon_days} days) > 90 days.")
            logger.warning("   The forecast may become flat/cyclical due to missing yearly_seasonality.")
            logger.warning("   Recommendation: use horizon <= 90 days OR collect 2+ years of data.")
    
    # Check if model requires regressors
    requires_regressors = len(model.extra_regressors) > 0
    logger.info(f"Model requires regressors: {requires_regressors}")
    
    if requires_regressors:
        if last_known_regressors_csv is None:
            raise ValueError("Model requires regressors but last_known_regressors_csv not provided")
        
        if not os.path.exists(last_known_regressors_csv):
            raise FileNotFoundError(f"Regressors CSV file not found: {last_known_regressors_csv}")
        
        logger.info(f"Loading regressors from: {last_known_regressors_csv}")
        df_regressors = pd.read_csv(last_known_regressors_csv)
        df_regressors['ds'] = pd.to_datetime(df_regressors['ds'])
        df_regressors = df_regressors.sort_values('ds')
        
        # Check required regressor columns
        required_regressor_cols = ['avg_price', 'avg_discount']
        missing_cols = [col for col in required_regressor_cols if col not in df_regressors.columns]
        if missing_cols:
            raise ValueError(f"Missing required regressor columns: {', '.join(missing_cols)}")
    
    # Create future dataframe for the specified horizon
    logger.info(f"Creating future dataframe for {horizon_days} days")
    future = model.make_future_dataframe(periods=horizon_days, freq='D')
    
    # Add regressors for future dates if needed
    if requires_regressors:
        # Get the last known regressor values
        last_regressor_values = df_regressors[['ds', 'avg_price', 'avg_discount']].copy()
        
        # Merge with future dataframe
        future = future.merge(
            last_regressor_values,
            on='ds',
            how='left'
        )
        
        # Fill regressors for future dates
        for col in ['avg_price', 'avg_discount']:
            if future[col].isna().any():
                if regressor_fill_method == 'median':
                    # Use median of last known values
                    median_value = last_regressor_values[col].median()
                    logger.info(f"Filling {col} with median: {median_value:.2f}")
                    future[col] = future[col].fillna(median_value)
                else:  # default: forward-fill
                    # Forward-fill using last known values
                    last_value = last_regressor_values[col].iloc[-1]
                    logger.info(f"Forward-filling {col} with last known value: {last_value:.2f}")
                    future[col] = future[col].ffill().fillna(last_value)
    
    # Generate predictions
    logger.info("Generating predictions...")
    forecast = model.predict(future)
    
    # Get the last date from training history to separate historical and future predictions
    last_training_date = model.history['ds'].max()
    logger.info(f"Last training date: {last_training_date.date()}")
    
    # Select only future predictions (dates after last training date)
    forecast_future = forecast[forecast['ds'] > last_training_date].copy()
    
    # Ensure we have exactly horizon_days or less (in case model didn't generate enough)
    if len(forecast_future) > horizon_days:
        forecast_future = forecast_future.head(horizon_days).copy()
    
    logger.info(f"Selected {len(forecast_future)} future predictions (requested: {horizon_days})")
    logger.info(f"Future forecast date range: {forecast_future['ds'].min().date()} to {forecast_future['ds'].max().date()}")
    
    # Extract required columns
    result = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Apply smoothing to reduce initial overestimation if requested
    if smooth_transition:
        logger.info(f"Applying AGGRESSIVE smoothing to first {smooth_days} days")
        logger.info(f"  - First day: 95% weight of last actual value + 5% forecast")
        logger.info(f"  - Max change: 0.5% for days 1-3, 1% for days 4-7, {max_change_pct*100:.1f}% for days 8+")
        
        # Логируем значения ДО сглаживания
        first_before = result['yhat'].iloc[0] if len(result) > 0 else 0
        
        result = smooth_forecast_transition(result, model, smooth_days, smooth_alpha, max_change_pct)
        
        # Логируем значения ПОСЛЕ сглаживания
        if len(result) > 0:
            first_after = result['yhat'].iloc[0]
            logger.info(f"  - First day BEFORE smoothing: {first_before:.2f}")
            logger.info(f"  - First day AFTER smoothing: {first_after:.2f}")
            logger.info(f"  - Reduction: {((first_before - first_after) / first_before * 100):.1f}%")
            
            # ОТКЛЮЧЕНО: Агрессивная коррекция тренда ухудшает качество прогноза
            # Вместо этого полагаемся на правильно настроенную модель Prophet
            pass
    else:
        logger.warning("⚠️ Smooth transition is DISABLED - forecast may be overestimated at the start!")
    
    # Apply inverse log transformation if needed
    if log_transform:
        logger.info("Applying inverse log1p (expm1) transformation to predictions")
        result['yhat'] = np.expm1(result['yhat'])
        result['yhat_lower'] = np.expm1(result['yhat_lower'])
        result['yhat_upper'] = np.expm1(result['yhat_upper'])
    
    # Ensure non-negative values for sales (realistic constraint)
    # Sales cannot be negative, so clip negative values to 0
    # Use explicit comparison with small epsilon to handle floating point precision
    epsilon = 1e-10
    
    n_negative_yhat = (result['yhat'] < -epsilon).sum()
    n_negative_lower = (result['yhat_lower'] < -epsilon).sum()
    n_negative_upper = (result['yhat_upper'] < -epsilon).sum()
    
    if n_negative_yhat > 0:
        logger.info(f"Clamping {n_negative_yhat} negative forecast values to 0 (sales cannot be negative)")
        result['yhat'] = result['yhat'].clip(lower=0.0)
    
    if n_negative_lower > 0:
        logger.info(f"Clamping {n_negative_lower} negative lower bound values to 0")
        result['yhat_lower'] = result['yhat_lower'].clip(lower=0.0)
    
    if n_negative_upper > 0:
        logger.info(f"Clamping {n_negative_upper} negative upper bound values to 0")
        result['yhat_upper'] = result['yhat_upper'].clip(lower=0.0)
    
    # Ensure yhat_upper >= yhat_lower after all clipping
    result['yhat_upper'] = result[['yhat_upper', 'yhat_lower']].max(axis=1)
    
    # Final safety check - ensure all values are truly non-negative
    result['yhat'] = result['yhat'].clip(lower=0.0)
    result['yhat_lower'] = result['yhat_lower'].clip(lower=0.0)
    result['yhat_upper'] = result['yhat_upper'].clip(lower=0.0)
    
    # Reset index for cleaner output
    result = result.reset_index(drop=True)
    
    logger.info(f"Forecast generated successfully: {len(result)} predictions (all values >= 0)")
    
    # Automatically save to default location
    default_output_path = "data/processed/forecast_shop.csv"
    save_forecast_csv(result, default_output_path)
    
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
    
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Save to CSV
    df_forecast.to_csv(out_path, index=False)
    logger.info(f"Forecast saved to: {out_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate forecast using trained Prophet model")
    parser.add_argument("model_path", help="Path to trained Prophet model (.pkl file)")
    parser.add_argument("horizon_days", type=int, help="Number of days to forecast (1-365)")
    parser.add_argument("--regressors-csv", type=str, default=None,
                       help="Path to shop CSV with regressors (required if model uses regressors)")
    parser.add_argument("--log-transform", action="store_true",
                       help="Apply inverse log1p (expm1) transformation to predictions")
    parser.add_argument("--regressor-fill", type=str, choices=['forward', 'median'], default='forward',
                       help="Method to fill regressors for future dates: 'forward' (default) or 'median'")
    parser.add_argument("--output", type=str, default="data/processed/forecast_shop.csv",
                       help="Output CSV path (default: data/processed/forecast_shop.csv)")
    
    args = parser.parse_args()
    
    # Generate forecast
    df_forecast = predict_prophet(
        model_path=args.model_path,
        horizon_days=args.horizon_days,
        last_known_regressors_csv=args.regressors_csv,
        log_transform=args.log_transform,
        regressor_fill_method=args.regressor_fill
    )
    
    # Save forecast
    save_forecast_csv(df_forecast, args.output)
    
    print("\n" + "="*60)
    print("Forecast generated successfully!")
    print("="*60)
    print(f"\nModel: {args.model_path}")
    print(f"Horizon: {args.horizon_days} days")
    print(f"Output: {args.output}")
    print(f"\nForecast preview:")
    print(df_forecast.head(10).to_string(index=False))
    print(f"\n... ({len(df_forecast) - 10} more rows)")
    print(f"\nSummary statistics:")
    print(f"  Mean forecast: {df_forecast['yhat'].mean():.2f}")
    print(f"  Min forecast: {df_forecast['yhat'].min():.2f}")
    print(f"  Max forecast: {df_forecast['yhat'].max():.2f}")
    print(f"  Forecast range: {df_forecast['ds'].min().date()} to {df_forecast['ds'].max().date()}")
